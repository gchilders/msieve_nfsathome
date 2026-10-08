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

/* Wiedemann stage 3: turn the generator into dependencies.

   The generator annihilates the sequence, so W = sum_k A^k y F_k has
   x^T A^i W = 0 for a long stretch of i, which forces A W = 0 -- or
   A^j W = 0 for some small j, when a column lands on a vector the
   operator kills a step or two later than expected.

   Three things happen after that. Each column of W is walked forward
   until it dies, keeping the last nonzero value. The rows that
   form_post_lanczos_matrix() stripped out are reinstated, which means
   finding the combinations of those columns that satisfy them too.
   What survives is packed one word per matrix column, which is the
   form the square root already reads. */

#include "wiedemann.h"

#define BW_PATH_LEN 512

/* how many times A may be applied before a column is given up on.
   Needing more than one or two would mean something is wrong */
#define MKSOL_MAX_STEPS 8

/*-----------------------------------------------------------------------*/
static v_t *read_generator(msieve_obj *obj, bw_params_t *params,
				uint32 ncols, uint32 *degree_out) {

	char buf[BW_PATH_LEN];
	FILE *fp;
	bw_gen_header_t hdr;
	v_t *f;
	size_t num;

	snprintf(buf, sizeof(buf), "%s.bw.f", obj->savefile.name);
	fp = fopen(buf, "rb");
	if (fp == NULL) {
		logprintf(obj, "error: cannot open Wiedemann generator %s\n",
				buf);
		return NULL;
	}

	if (fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
	    hdr.magic != BW_GEN_MAGIC) {
		logprintf(obj, "error: Wiedemann generator is corrupt\n");
		fclose(fp);
		return NULL;
	}
	if (hdr.vbits != VBITS || hdr.ncols != ncols ||
	    hdr.m != params->m_mult * VBITS ||
	    hdr.n != params->n_mult * VBITS) {
		logprintf(obj, "error: Wiedemann generator does not match "
				"this matrix\n");
		fclose(fp);
		return NULL;
	}

	/* one VBITS x VBITS block per coefficient, k = 0 .. degree */

	num = (size_t)(hdr.degree + 1) * VBITS;
	f = (v_t *)aligned_malloc(num * sizeof(v_t), 64);
	if (fread(f, sizeof(v_t), num, fp) != num) {
		logprintf(obj, "error: Wiedemann generator is truncated\n");
		aligned_free(f);
		fclose(fp);
		return NULL;
	}
	fclose(fp);

	*degree_out = hdr.degree;
	return f;
}

/*-----------------------------------------------------------------------*/
int32 bw_mksol(msieve_obj *obj, packed_matrix_t *matrix,
			bw_params_t *params, uint32 max_ncols,
			v_t *post_lanczos_matrix,
			v_t **solution_out, uint32 *num_deps_found) {

	uint32 i, j, k;
	uint32 n = matrix->ncols;
	uint32 degree = 0;
	v_t *f = NULL;
	v_t *host_w = NULL, *host_next = NULL, *host_res = NULL;
	v_t *combos = NULL;
	void *w = NULL, *z = NULL, *prod = NULL, *swap;
	v_t alive, acc;
	uint32 num_combos = 0;
	int32 status = -1;

	*solution_out = NULL;
	*num_deps_found = 0;

	if (params->n_mult != 1) {
		logprintf(obj, "error: mksol handles a single sequence so "
				"far\n");
		return -1;
	}

	f = read_generator(obj, params, max_ncols, &degree);
	if (f == NULL)
		return -1;

	logprintf(obj, "commencing Wiedemann mksol, generator degree %u\n",
			degree);

	w = vv_alloc(n, matrix->extra);
	z = vv_alloc(n, matrix->extra);
	prod = vv_alloc(n, matrix->extra);
	host_w = (v_t *)aligned_malloc((size_t)max_ncols * sizeof(v_t), 64);
	host_next = (v_t *)aligned_malloc((size_t)max_ncols * sizeof(v_t), 64);
	host_res = (v_t *)aligned_malloc((size_t)max_ncols * sizeof(v_t), 64);
	combos = (v_t *)xmalloc(VBITS * sizeof(v_t));

	/* W = sum_k A^k y F_k. y has to be regenerated exactly as the
	   Krylov stage had it, which is why the seeds are fixed and
	   travel in the parameters */

	{
		uint32 seed1 = params->seed1;
		uint32 seed2 = params->seed2;

		for (i = 0; i < 1 + params->seq; i++) {
			for (j = 0; j < n; j++)
				host_w[j] = v_random(&seed1, &seed2);
		}
		vv_copyin(z, host_w, n);
	}

	vv_clear(w, n);
	for (k = 0; ; k++) {

		vv_mul_NxB_BxB_acc(matrix, z, f + (size_t)k * VBITS, w, n);

		if (k == degree)
			break;

		vv_clear(prod, n);
		mul_MxN_NxB(matrix, z, prod, NULL);
		swap = z; z = prod; prod = swap;
	}

	/* Walk the columns forward until each dies, keeping the last
	   nonzero value. A column already in the nullspace dies on the
	   first step, so applying A to the whole block uniformly would
	   throw it away; that is what the liveness mask prevents. */

	vv_copyout(host_w, w, n);
	for (i = 0; i < max_ncols; i++)
		host_res[i] = v_zero;
	alive = v_zero;
	for (i = 0; i < VBITS; i++)
		bw_v_set_bit(&alive, i);

	for (k = 0; k < MKSOL_MAX_STEPS; k++) {
		v_t died;

		vv_clear(prod, n);
		mul_MxN_NxB(matrix, w, prod, NULL);
		vv_copyout(host_next, prod, n);

		/* a column is dead once every entry of it is zero */

		acc = v_zero;
		for (i = 0; i < n; i++)
			acc = v_or(acc, host_next[i]);

		died = v_zero;
		for (i = 0; i < VWORDS; i++)
			died.w[i] = alive.w[i] & ~acc.w[i];

		if (!v_is_all_zeros(died)) {
			for (i = 0; i < n; i++)
				host_res[i] = v_xor(host_res[i],
						v_and(host_w[i], died));
			for (i = 0; i < VWORDS; i++)
				alive.w[i] &= ~died.w[i];

			logprintf(obj, "mksol: %u columns reached the "
					"nullspace after %u steps\n",
					bw_v_popcount(died), k + 1);
		}

		if (v_is_all_zeros(alive))
			break;

		swap = w; w = prod; prod = swap;
		{
			v_t *h = host_w; host_w = host_next; host_next = h;
		}
	}

	if (!v_is_all_zeros(alive)) {
		logprintf(obj, "warning: %u Wiedemann columns were still "
				"alive after %u steps and are discarded\n",
				bw_v_popcount(alive), (uint32)MKSOL_MAX_STEPS);
	}

	/* Gate: whatever we kept must actually be killed by the matrix.
	   One product, and it catches any mistake upstream of here */

	vv_copyin(w, host_res, n);
	vv_clear(prod, n);
	mul_MxN_NxB(matrix, w, prod, NULL);
	vv_copyout(host_next, prod, n);
	acc = v_zero;
	for (i = 0; i < n; i++)
		acc = v_or(acc, host_next[i]);
	if (!v_is_all_zeros(acc)) {
		logprintf(obj, "error: Wiedemann solution is not in the "
				"nullspace (%u columns bad)\n",
				bw_v_popcount(acc));
		goto cleanup;
	}

	/* Reinstate the stripped rows: find the combinations of these
	   columns that satisfy them as well. Same accumulation block
	   Lanczos does when it folds its post-Lanczos matrix back in. */

	if (post_lanczos_matrix != NULL) {
		v_t *pl = (v_t *)xmalloc(POST_LANCZOS_ROWS * sizeof(v_t));

		for (i = 0; i < POST_LANCZOS_ROWS; i++)
			pl[i] = v_zero;
		for (j = 0; j < n; j++) {
			for (i = 0; i < POST_LANCZOS_ROWS; i++) {
				if (v_bitset(post_lanczos_matrix[j], i))
					pl[i] = v_xor(pl[i], host_res[j]);
			}
		}

		num_combos = bw_gf2_nullspace(pl, POST_LANCZOS_ROWS, combos);
		free(pl);
	}
	else {
		for (i = 0; i < VBITS; i++) {
			combos[i] = v_zero;
			bw_v_set_bit(combos + i, i);
		}
		num_combos = VBITS;
	}

	/* Apply them: bit d of the answer for matrix column i is the
	   parity of that column's vector against combination d */

	for (i = 0; i < max_ncols; i++) {
		v_t out = v_zero;

		for (j = 0; j < num_combos; j++) {
			if (bw_v_parity(host_res[i], combos[j]))
				bw_v_set_bit(&out, j);
		}
		host_next[i] = out;
	}

	logprintf(obj, "Wiedemann found %u dependencies\n", num_combos);

	*solution_out = host_next;
	*num_deps_found = num_combos;
	host_next = NULL;
	status = 0;

cleanup:
	free(combos);
	aligned_free(host_res);
	aligned_free(host_next);
	aligned_free(host_w);
	vv_free(prod);
	vv_free(z);
	vv_free(w);
	aligned_free(f);
	return status;
}
