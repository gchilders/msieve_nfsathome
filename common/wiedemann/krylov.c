/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may
benefit from your work.

$Id$
--------------------------------------------------------------------*/

/* Wiedemann stage 1: the Krylov sequence.

   Computes a_i = x^T A^i y for i < L and writes the a_i to disk. The
   only matrix operation is the forward product, so this never needs
   A^T and never needs a second copy of the matrix on the card.

   A is nrows x ncols with more columns than rows, which is not square
   and so cannot be iterated. We use instead the square operator that
   pads A out to ncols x ncols with zero rows: it has the same nullspace
   and the same nonzeros, so it costs nothing. In practice that just
   means every vector is ncols long and the rows past nrows stay
   zero. */

#include "wiedemann.h"

/* savefile names can be long, so give the derived paths room */
#define BW_PATH_LEN 512

/*-----------------------------------------------------------------------*/
static void bw_seq_path(msieve_obj *obj, uint32 seq, char *buf) {

	snprintf(buf, BW_PATH_LEN, "%s.bw.a.%u", obj->savefile.name, seq);
}

static void bw_chk_path(msieve_obj *obj, uint32 seq, char *buf) {

	snprintf(buf, BW_PATH_LEN, "%s.bw.chk.%u", obj->savefile.name, seq);
}

/*-----------------------------------------------------------------------*/
static void dump_krylov_state(msieve_obj *obj, bw_params_t *params,
				void *z, uint32 n, uint32 iter,
				v_t *tmp) {

	/* Save the current vector and the iteration count. The sequence
	   file is the other half of the state; it is flushed and its
	   header rewritten before this is called, so a restart that
	   truncates the sequence to iter terms is consistent.

	   Written to .chk0 and renamed into place only once every write
	   has succeeded, so an interrupted dump cannot destroy the
	   previous one. Same discipline as dump_lanczos_state(). */

	char buf[BW_PATH_LEN], buf_old[BW_PATH_LEN], buf_bak[BW_PATH_LEN];
	FILE *fp;
	uint32 status = 1;
	uint32 vbits = VBITS;

	bw_chk_path(obj, params->seq, buf_old);
	snprintf(buf, sizeof(buf), "%s.bw.chk0.%u",
			obj->savefile.name, params->seq);
	snprintf(buf_bak, sizeof(buf_bak), "%s.bw.bak.chk.%u",
			obj->savefile.name, params->seq);

	fp = fopen(buf, "wb");
	if (fp == NULL) {
		printf("error: cannot open Wiedemann checkpoint file\n");
		exit(-1);
	}

	status &= (fwrite(&vbits, sizeof(uint32), 1, fp) == 1);
	status &= (fwrite(&n, sizeof(uint32), 1, fp) == 1);
	status &= (fwrite(&iter, sizeof(uint32), 1, fp) == 1);
	status &= (fwrite(&params->seed1, sizeof(uint32), 1, fp) == 1);
	status &= (fwrite(&params->seed2, sizeof(uint32), 1, fp) == 1);

	vv_copyout(tmp, z, n);
	status &= (fwrite(tmp, sizeof(v_t), n, fp) == n);

	fclose(fp);

	if (status == 0) {
		printf("error: cannot write Wiedemann checkpoint\n");
		exit(-1);
	}

	remove(buf_bak);
	if (rename(buf_old, buf_bak) != 0)
		remove(buf_old);
	rename(buf, buf_old);
}

/*-----------------------------------------------------------------------*/
static uint32 read_krylov_state(msieve_obj *obj, bw_params_t *params,
				void *z, uint32 n, v_t *tmp) {

	/* Returns the iteration to resume at, or 0 if there is no usable
	   checkpoint. A checkpoint from a different VBITS or a different
	   matrix is refused rather than trusted. */

	char buf[BW_PATH_LEN];
	FILE *fp;
	uint32 read_vbits = 0, read_n = 0, iter = 0;
	uint32 seed1 = 0, seed2 = 0;
	uint32 status = 1;

	bw_chk_path(obj, params->seq, buf);
	fp = fopen(buf, "rb");
	if (fp == NULL)
		return 0;

	status &= (fread(&read_vbits, sizeof(uint32), 1, fp) == 1);
	status &= (fread(&read_n, sizeof(uint32), 1, fp) == 1);
	status &= (fread(&iter, sizeof(uint32), 1, fp) == 1);
	status &= (fread(&seed1, sizeof(uint32), 1, fp) == 1);
	status &= (fread(&seed2, sizeof(uint32), 1, fp) == 1);

	if (status == 0 || read_vbits != VBITS || read_n != n) {
		fclose(fp);
		logprintf(obj, "Wiedemann checkpoint does not match this "
				"matrix, ignoring it\n");
		return 0;
	}

	status &= (fread(tmp, sizeof(v_t), n, fp) == n);
	fclose(fp);

	if (status == 0) {
		logprintf(obj, "Wiedemann checkpoint is truncated, "
				"ignoring it\n");
		return 0;
	}

	/* x and y must be regenerated exactly as they were, or the
	   sequence already on disk means nothing */

	params->seed1 = seed1;
	params->seed2 = seed2;
	vv_copyin(z, tmp, n);
	return iter;
}

/*-----------------------------------------------------------------------*/
static FILE *open_sequence(msieve_obj *obj, bw_params_t *params,
				uint32 ncols, uint32 num_terms,
				uint32 resume) {

	/* On a fresh start this writes a new header; on a restart it
	   reopens the file and positions after the first num_terms
	   records, discarding anything past them */

	char buf[BW_PATH_LEN];
	FILE *fp;
	bw_seq_header_t hdr;
	size_t term_size = (size_t)params->m_mult * VBITS * sizeof(v_t);

	bw_seq_path(obj, params->seq, buf);

	hdr.magic = BW_SEQ_MAGIC;
	hdr.vbits = VBITS;
	hdr.m = params->m_mult * VBITS;
	hdr.n = params->n_mult * VBITS;
	hdr.seq = params->seq;
	hdr.seq_width = VBITS;
	hdr.ncols = ncols;
	hdr.num_terms = num_terms;
	hdr.seed1 = params->seed1;
	hdr.seed2 = params->seed2;

	if (resume) {
		bw_seq_header_t old;

		fp = fopen(buf, "r+b");
		if (fp == NULL)
			return NULL;

		/* the seeds have to match as well: they decide x and y, so
		   terms written under different ones do not belong to the
		   same sequence even though everything else agrees */

		if (fread(&old, sizeof(old), 1, fp) != 1 ||
		    old.magic != BW_SEQ_MAGIC ||
		    old.vbits != VBITS ||
		    old.m != hdr.m || old.n != hdr.n ||
		    old.ncols != ncols || old.num_terms < num_terms ||
		    old.seed1 != hdr.seed1 || old.seed2 != hdr.seed2) {
			fclose(fp);
			return NULL;
		}
		rewind(fp);
		if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
			fclose(fp);
			return NULL;
		}
		if (fseeko(fp, (off_t)sizeof(hdr) +
				(off_t)num_terms * term_size, SEEK_SET) != 0) {
			fclose(fp);
			return NULL;
		}
		return fp;
	}

	fp = fopen(buf, "wb");
	if (fp == NULL)
		return NULL;
	if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
		fclose(fp);
		return NULL;
	}
	return fp;
}

/*-----------------------------------------------------------------------*/
static int32 update_sequence_count(FILE *fp, uint32 num_terms) {

	/* Rewrite the term count in the header, through the handle that
	   is already open on the file rather than a second one: two
	   handles on one file is the kind of thing that works until it
	   does not, and a failure here is not cosmetic. The header count
	   is half the restart state, so if it silently stops being
	   updated the run becomes unresumable -- open_sequence() refuses
	   a header holding fewer terms than the checkpoint claims, and
	   every Krylov iteration computed so far is lost.

	   Returns 0 on success. The caller stops on failure, which leaves
	   a consistent file rather than one that cannot be restarted. */

	off_t pos = ftello(fp);

	if (pos < 0)
		return -1;
	if (fseeko(fp, (off_t)offsetof(bw_seq_header_t, num_terms),
				SEEK_SET) != 0)
		return -1;
	if (fwrite(&num_terms, sizeof(uint32), 1, fp) != 1)
		return -1;
	if (fflush(fp) != 0)
		return -1;
	if (fseeko(fp, pos, SEEK_SET) != 0)
		return -1;
	return 0;
}

/*-----------------------------------------------------------------------*/
int32 bw_krylov(msieve_obj *obj, packed_matrix_t *matrix,
			bw_params_t *params, uint32 max_ncols) {

	uint32 i, j;
	uint32 n = matrix->ncols;
	uint32 m = params->m_mult * VBITS;
	uint32 num_seq_cols = params->n_mult * VBITS;
	uint32 num_terms;
	uint32 iter;
	uint32 report_interval;
	uint32 dump_interval;
	uint32 next_report, next_dump;
	void *cur, *next, *tmp_vec;
	void **xblk;
	v_t *host_tmp;
	v_t *a;
	FILE *seq_fp;
	int32 status = 0;
	time_t start_time;

	/* L = N/m + N/n is what the generator needs; the margin covers
	   the O(1) in that bound and costs almost nothing */

	num_terms = max_ncols / m + max_ncols / num_seq_cols + 32;

	logprintf(obj, "commencing Wiedemann Krylov, sequence %u of %u\n",
			params->seq, params->n_mult);
	logprintf(obj, "m = %u, n = %u, %u terms of %u x %u\n",
			m, num_seq_cols, num_terms, m, VBITS);

	/* the projection blocks, and the two vectors the recurrence
	   bounces between */

	xblk = (void **)xmalloc(params->m_mult * sizeof(void *));
	for (i = 0; i < params->m_mult; i++)
		xblk[i] = vv_alloc(n, matrix->extra);
	cur = vv_alloc(n, matrix->extra);
	next = vv_alloc(n, matrix->extra);
	host_tmp = (v_t *)aligned_malloc(n * sizeof(v_t), 64);
	a = (v_t *)aligned_malloc((size_t)m * sizeof(v_t), 64);

	/* resume if we can, otherwise start from scratch. Either way x
	   and y come from the same two seeds, so the terms already on
	   disk and the ones we are about to compute agree */

	iter = read_krylov_state(obj, params, cur, n, host_tmp);

	for (i = 0; i < params->m_mult; i++) {
		uint32 seed1 = params->seed1 + 0x9e3779b9u * (i + 1);
		uint32 seed2 = params->seed2 + 0x7f4a7c15u * (i + 1);

		for (j = 0; j < n; j++)
			host_tmp[j] = v_random(&seed1, &seed2);
		vv_copyin(xblk[i], host_tmp, n);
	}

	if (iter == 0) {
		uint32 seed1 = params->seed1;
		uint32 seed2 = params->seed2;

		/* y is specific to this sequence: the n columns of the
		   whole method are split into n_mult blocks of VBITS,
		   and this process owns block params->seq */

		for (i = 0; i < 1 + params->seq; i++) {
			for (j = 0; j < n; j++)
				host_tmp[j] = v_random(&seed1, &seed2);
		}
		vv_copyin(cur, host_tmp, n);
	}
	else {
		logprintf(obj, "restarting Krylov at term %u\n", iter);
	}

	seq_fp = open_sequence(obj, params, max_ncols, iter, iter != 0);
	if (seq_fp == NULL) {
		logprintf(obj, "error: cannot open Wiedemann sequence file\n");
		status = -1;
		goto cleanup;
	}

	report_interval = MAX(1, num_terms / 100);
	dump_interval = MAX(1000, num_terms / 50);
	next_report = iter + report_interval;
	next_dump = iter + dump_interval;
	start_time = time(NULL);

	for (; iter < num_terms; iter++) {

		/* a_i = x^T z, one VBITS x VBITS block per projection
		   vector. This is the same outer product the Lanczos
		   iteration uses */

		for (i = 0; i < params->m_mult; i++)
			vv_mul_BxN_NxB(matrix, xblk[i], cur,
					a + i * VBITS, n);

		if (fwrite(a, sizeof(v_t), m, seq_fp) != m) {
			logprintf(obj, "error: cannot write Wiedemann "
					"sequence term %u\n", iter);
			status = -1;
			goto cleanup;
		}

		/* z <- A z. mul_core only writes rows [0, nrows), so the
		   zero rows that pad A out to a square operator have to
		   be cleared here; one extra pass over the vector is
		   about 1% of the product and keeps the invariant
		   obvious. (It could be hoisted: only the buffer that
		   held the random y is ever dirty past nrows.) */

		vv_clear(next, n);
		mul_MxN_NxB(matrix, cur, next, NULL);

		tmp_vec = cur;
		cur = next;
		next = tmp_vec;

		if (iter + 1 >= next_report) {
			double pct = 100.0 * (iter + 1) / num_terms;
			fprintf(stderr, "Krylov %u of %u, %.1f%%\r",
					iter + 1, num_terms, pct);
			fflush(stderr);
			next_report = iter + 1 + report_interval;
		}

		/* an interrupt is honored at the next checkpoint, so that
		   what is on disk is always consistent; ask for one right
		   away rather than waiting out the rest of the interval,
		   which is thousands of products on a large matrix */

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
			next_dump = iter + 1;

		if (iter + 1 >= next_dump) {
			fflush(seq_fp);
			if (update_sequence_count(seq_fp, iter + 1) != 0) {
				logprintf(obj, "error: cannot update Wiedemann "
						"sequence header\n");
				status = -1;
				goto cleanup;
			}
			dump_krylov_state(obj, params, cur, n,
					iter + 1, host_tmp);
			next_dump = iter + 1 + dump_interval;

			if (obj->flags & MSIEVE_FLAG_STOP_SIEVING) {
				logprintf(obj, "Krylov stopped at term %u\n",
						iter + 1);
				status = -1;
				goto cleanup;
			}
		}
	}

	fflush(seq_fp);
	if (update_sequence_count(seq_fp, num_terms) != 0) {
		logprintf(obj, "error: cannot update Wiedemann sequence "
				"header\n");
		status = -1;
		goto cleanup;
	}
	fclose(seq_fp);
	seq_fp = NULL;

	logprintf(obj, "Krylov sequence %u complete, %u terms, %.1f sec\n",
			params->seq, num_terms,
			difftime(time(NULL), start_time));

cleanup:
	if (seq_fp != NULL)
		fclose(seq_fp);
	aligned_free(a);
	aligned_free(host_tmp);
	vv_free(next);
	vv_free(cur);
	for (i = 0; i < params->m_mult; i++)
		vv_free(xblk[i]);
	free(xblk);
	return status;
}
