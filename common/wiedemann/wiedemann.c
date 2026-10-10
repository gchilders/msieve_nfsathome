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

/* Block Wiedemann driver: sets the matrix up the same way block
   Lanczos does, then runs whichever stages were asked for. */

#include "wiedemann.h"

/* x and y are generated from these unless bw_seed= says otherwise.
   They are fixed, not taken from obj->seed1, because the three stages
   normally run as separate processes and every one of them has to
   regenerate exactly the same vectors. If a run fails to produce
   enough dependencies, changing bw_seed is the retry. */

#define BW_DEFAULT_SEED1 0x4f1bbcddu
#define BW_DEFAULT_SEED2 0x9c2a5f73u

#ifdef HAVE_MPI
	#define BW_RANK(obj)	((obj)->mpi_rank)
#else
	#define BW_RANK(obj)	0
#endif

/*-----------------------------------------------------------------------*/
static int32 bw_stage_sync(msieve_obj *obj, int32 status) {

	/* The stages hand work to each other through files, so a stage
	   cannot start until every rank has finished writing the one
	   before it. The wait has to carry the status rather than being
	   a bare barrier: a rank that failed and left early would
	   strand the rest here for the life of the job, which on a
	   scheduler costs the whole allocation and looks like a hang
	   rather than the error it is. Every rank reaches this whether
	   it succeeded or not, learns whether any of them failed, and
	   they stop together. */

#ifdef HAVE_MPI
	int32 worst = status;

	if (obj->mpi_size > 1) {
		MPI_TRY(MPI_Allreduce(&status, &worst, 1, MPI_INT,
					MPI_MIN, MPI_COMM_WORLD))
		if (worst != 0 && status == 0) {
			logprintf(obj, "another rank failed; stopping "
					"this one too\n");
		}
	}
	return worst;
#else
	(void)obj;
	return status;
#endif
}

/*-----------------------------------------------------------------------*/
static uint32 parse_uint(const char *args, const char *key, uint32 def) {

	const char *tmp;

	if (args == NULL)
		return def;
	tmp = strstr(args, key);
	if (tmp == NULL)
		return def;
	return (uint32)strtoul(tmp + strlen(key), NULL, 10);
}

/*-----------------------------------------------------------------------*/
static int32 parse_params(msieve_obj *obj, bw_params_t *params) {

	const char *args = obj->nfs_args;
	const char *tmp;

	uint32 dflt = 1;
	uint32 rank_is_seq = 0;

#ifdef HAVE_MPI
	/* One rank per sequence is the whole MPI story for this solver,
	   so the rank count is the natural m and n: that is the shape
	   the Krylov length wants, since L = N/m + N/n only shrinks when
	   both grow. The rank then says which sequence to compute, so
	   bw_seq is not something to pass. */

	if (obj->mpi_size > 1) {
		dflt = obj->mpi_size;
		rank_is_seq = 1;
	}
#endif
	params->m_mult = parse_uint(args, "bw_m=", dflt);
	params->n_mult = parse_uint(args, "bw_n=", dflt);
	params->seq = parse_uint(args, "bw_seq=", 0);
	params->seed1 = parse_uint(args, "bw_seed=", BW_DEFAULT_SEED1);
	params->seed2 = BW_DEFAULT_SEED2;
	params->stage = BW_STAGE_ALL;

	if (args != NULL && (tmp = strstr(args, "bw_stage=")) != NULL) {
		tmp += 9;
		if (!strncmp(tmp, "krylov", 6))
			params->stage = BW_STAGE_KRYLOV;
		else if (!strncmp(tmp, "lingen", 6))
			params->stage = BW_STAGE_LINGEN;
		else if (!strncmp(tmp, "mksol", 5))
			params->stage = BW_STAGE_MKSOL;
		else if (!strncmp(tmp, "combine", 7))
			params->stage = BW_STAGE_COMBINE;
		else if (!strncmp(tmp, "all", 3))
			params->stage = BW_STAGE_ALL;
		else {
			logprintf(obj, "error: unknown bw_stage\n");
			return -1;
		}
	}

	if (params->m_mult == 0 || params->n_mult == 0) {
		logprintf(obj, "error: bw_m and bw_n must be nonzero\n");
		return -1;
	}

	/* The rank is the sequence, but only for the stages that own
	   one. lingen owns none of them -- it reads every sequence and
	   splits a product over the ranks instead -- so it neither
	   takes a sequence index nor has to be run on as many ranks as
	   there are sequences, and handing it one would reject every
	   rank past bw_n for a number it never looks at. */

	if (rank_is_seq) {
		if (args != NULL && strstr(args, "bw_seq=") != NULL) {
			logprintf(obj, "error: bw_seq is the MPI rank when "
					"running under MPI; drop it\n");
			return -1;
		}
		if (params->stage != BW_STAGE_LINGEN)
			params->seq = BW_RANK(obj);
	}

	if (params->stage != BW_STAGE_LINGEN &&
			params->seq >= params->n_mult) {
		logprintf(obj, "error: bw_seq must be less than bw_n\n");
		return -1;
	}

	/* The stages that touch the matrix own one sequence each, so
	   under MPI there has to be exactly one rank per sequence.
	   lingen is the exception: it spreads a product over whatever
	   ranks it is given and does not care how many sequences there
	   are, so a lingen-only run on a different rank count is fine
	   and just has to say bw_n= for itself. */

#ifdef HAVE_MPI
	if (rank_is_seq && params->stage != BW_STAGE_LINGEN &&
			params->n_mult != obj->mpi_size) {
		logprintf(obj, "error: bw_n is %u but there are %u MPI "
				"ranks; this stage runs one sequence per "
				"rank\n", params->n_mult, obj->mpi_size);
		return -1;
	}
#endif

	/* Several sequences cannot be driven from one process: each
	   stage owns one of them, so running them end to end would
	   compute sequence bw_seq and then ask lingen for all of them.
	   Under MPI they are driven from one process each, which is
	   exactly what makes the whole solve a single command. */

	if (params->stage == BW_STAGE_ALL && params->n_mult > 1 &&
			!rank_is_seq) {
		logprintf(obj, "error: bw_stage=all needs bw_n=1 without "
				"MPI; with more sequences either run under "
				"MPI with one rank per sequence, or run "
				"bw_stage=krylov once per bw_seq, then "
				"lingen, then bw_stage=mksol once per "
				"bw_seq, then bw_stage=combine\n");
		return -1;
	}
	return 0;
}

/*-----------------------------------------------------------------------*/
uint64 * block_wiedemann(msieve_obj *obj,
			uint32 nrows, uint32 max_nrows, uint32 start_row,
			uint32 num_dense_rows,
			uint32 ncols, uint32 max_ncols, uint32 start_col,
			la_col_t *B, uint32 *num_deps_found) {

	v_t *post_lanczos_matrix = NULL;
	packed_matrix_t packed_matrix;
	bw_params_t params;
	uint32 have_post_lanczos;
	uint64 *deps = NULL;
	int32 status = 0;

	*num_deps_found = 0;

	if (max_ncols <= max_nrows) {
		logprintf(obj, "matrix needs more columns than rows; "
				"try adding 2-3%% more relations\n");
		exit(-1);
	}

	if (parse_params(obj, &params) != 0)
		exit(-1);

	/* lingen touches no matrix at all -- it reads the sequences and
	   writes the generator -- so it skips the build entirely. That
	   saves loading a multi-gigabyte matrix for nothing, and it is
	   what lets the stage run on machines with no GPU, and under MPI
	   when the stages that do touch the matrix cannot. */

	if (params.stage == BW_STAGE_LINGEN) {
		bmp_mul_set_mpi(obj);
		bw_lingen(obj, &params, max_ncols);
		return NULL;
	}

	/* The matmuls require the packed dense rows to be a multiple of
	   VBITS: the dense part of the product is done in whole VBITS
	   batches, and a partial last batch would write over the sparse
	   rows that follow it. form_post_lanczos_matrix() is what
	   establishes that, so Wiedemann uses it exactly as Lanczos
	   does. The rows it strips out are reinstated when the solution
	   is assembled. */

	have_post_lanczos = form_post_lanczos_matrix(obj, &nrows,
					&num_dense_rows, ncols, B,
					&post_lanczos_matrix);
	if (num_dense_rows) {
		logprintf(obj, "matrix includes %u packed rows\n",
					num_dense_rows);
	}

	memset(&packed_matrix, 0, sizeof(packed_matrix_t));

	/* Tell the matmul layer how many vectors to expect before it
	   plans the matrix, because on a GPU whatever it does not
	   reserve is handed to the sparse blocks instead. Krylov holds
	   the m_mult projection blocks and the two its recurrence
	   alternates between; mksol holds three, and so does combining,
	   which allocates them before it discovers it has no vector
	   work to do. At eight ranks krylov's are ten full-length
	   vectors, several GB, and reserving two was enough to push a
	   matrix that fits into streaming. */

	packed_matrix.num_vectors =
		(params.stage == BW_STAGE_MKSOL ||
		 params.stage == BW_STAGE_COMBINE) ? 3 : params.m_mult + 2;

	if (have_post_lanczos)
		max_nrows -= POST_LANCZOS_ROWS;

	if (have_post_lanczos)
		count_matrix_nonzero(obj, nrows, num_dense_rows, ncols, B);

	packed_matrix_init(obj, &packed_matrix, B,
			   nrows, max_nrows, start_row,
			   ncols, max_ncols, start_col,
			   num_dense_rows, NUM_MEDIUM_ROWS);

	if (params.stage == BW_STAGE_ALL ||
	    params.stage == BW_STAGE_KRYLOV) {

		status = bw_krylov(obj, &packed_matrix, &params, max_ncols);
		if (params.stage != BW_STAGE_ALL && status != 0)
			goto done;
	}

	if (params.stage == BW_STAGE_ALL) {

		/* every sequence has to be on disk before lingen reads
		   them, and the generator has to be on disk before mksol
		   reads it back. Note that a rank whose Krylov failed
		   comes through here too, so that the ranks that did not
		   fail are told rather than left waiting. */

		status = bw_stage_sync(obj, status);
		if (status != 0)
			goto done;

		bmp_mul_set_mpi(obj);
		status = bw_lingen(obj, &params, max_ncols);
		status = bw_stage_sync(obj, status);
		if (status != 0)
			goto done;
	}

	if (params.stage == BW_STAGE_MKSOL ||
	    params.stage == BW_STAGE_COMBINE ||
	    params.stage == BW_STAGE_ALL) {

		v_t *solution = NULL;
		uint32 num_found = 0;
		uint32 i;

		/* Combining is one process's job: it reads every partial
		   and writes the dependencies. Letting each rank do it
		   would have all of them write the same .dep at once, so
		   the others stop here and leave it to rank 0. */

		if (params.stage == BW_STAGE_COMBINE && BW_RANK(obj) != 0) {
			logprintf(obj, "combining runs on rank 0 only\n");
			goto done;
		}

		/* with several sequences this returns nothing on the mksol
		   pass, having written a partial solution for the combine
		   stage to pick up */

		status = bw_mksol(obj, &packed_matrix, &params, max_ncols,
				post_lanczos_matrix, &solution,
				&num_found);

		/* with one rank per sequence that pass wrote a partial
		   each; rank 0 XORs them once they are all there. The
		   combining path does none of the vector work, so this
		   second call is only the read and the XOR. */

		if (params.stage == BW_STAGE_ALL && params.n_mult > 1) {
			status = bw_stage_sync(obj, status);
			aligned_free(solution);
			solution = NULL;
			num_found = 0;
			if (status != 0 || BW_RANK(obj) != 0)
				goto done;
			params.stage = BW_STAGE_COMBINE;
			if (bw_mksol(obj, &packed_matrix, &params, max_ncols,
					post_lanczos_matrix, &solution,
					&num_found) != 0)
				goto done;
		}
		else if (status != 0) {
			aligned_free(solution);
			goto done;
		}

		if (num_found > 0) {
			if (num_found > 64) {
				logprintf(obj, "saving only 64 "
						"dependencies\n");
				num_found = 64;
			}
			deps = (uint64 *)xmalloc(max_ncols * sizeof(uint64));
			for (i = 0; i < max_ncols; i++)
				deps[i] = solution[i].w[0];
			*num_deps_found = num_found;
		}
		aligned_free(solution);
	}

done:
	packed_matrix_free(&packed_matrix);
	free(post_lanczos_matrix);
	return deps;
}

/*-----------------------------------------------------------------------*/
uint32 bw_sequence_ncols(msieve_obj *obj) {

	/* The matrix dimension, taken from a sequence file rather than
	   from the matrix. lingen needs the number and nothing else of
	   the matrix, so reading it here means that stage can run on a
	   machine that has only the sequences -- no .mat, no GPU. */

	char buf[256];
	FILE *fp;
	bw_seq_header_t hdr;
	uint32 ncols = 0;

	snprintf(buf, sizeof(buf), "%s.bw.a.0", obj->savefile.name);
	fp = fopen(buf, "rb");
	if (fp == NULL)
		return 0;
	if (fread(&hdr, sizeof(hdr), 1, fp) == 1 &&
	    hdr.magic == BW_SEQ_MAGIC && hdr.vbits == VBITS)
		ncols = hdr.ncols;
	fclose(fp);
	return ncols;
}
