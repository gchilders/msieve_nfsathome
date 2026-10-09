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

	params->m_mult = parse_uint(args, "bw_m=", 1);
	params->n_mult = parse_uint(args, "bw_n=", 1);
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
	if (params->seq >= params->n_mult) {
		logprintf(obj, "error: bw_seq must be less than bw_n\n");
		return -1;
	}

	/* Several sequences cannot be driven from one process: each
	   stage owns one of them, so running them end to end would
	   compute sequence bw_seq and then ask lingen for all of them.
	   Say so here rather than failing later on a missing file. */

	if (params->stage == BW_STAGE_ALL && params->n_mult > 1) {
		logprintf(obj, "error: bw_stage=all needs bw_n=1; with more "
				"sequences run bw_stage=krylov once per "
				"bw_seq, then lingen, then bw_stage=mksol "
				"once per bw_seq, then bw_stage=combine\n");
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

	*num_deps_found = 0;

	if (max_ncols <= max_nrows) {
		logprintf(obj, "matrix needs more columns than rows; "
				"try adding 2-3%% more relations\n");
		exit(-1);
	}

	if (parse_params(obj, &params) != 0)
		exit(-1);

#ifdef HAVE_MPI
	if (obj->mpi_size > 1) {
		/* Sequence parallelism is the intended way to use several
		   devices and needs no MPI at all: run one process per
		   sequence with bw_seq= and -g. Splitting a single matrix
		   across ranks additionally needs the row and column
		   decompositions to line up, which they do only for the
		   default 1 x P grid; that is not wired up yet. */

		logprintf(obj, "error: block Wiedemann does not support an "
				"MPI grid yet; run one process per sequence\n");
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ASSERT);
	}
#endif

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

		if (bw_krylov(obj, &packed_matrix, &params, max_ncols) != 0)
			goto done;
	}

	if (params.stage == BW_STAGE_LINGEN ||
	    params.stage == BW_STAGE_ALL) {

		if (bw_lingen(obj, &params, max_ncols) != 0)
			goto done;
	}

	if (params.stage == BW_STAGE_MKSOL ||
	    params.stage == BW_STAGE_COMBINE ||
	    params.stage == BW_STAGE_ALL) {

		v_t *solution = NULL;
		uint32 num_found = 0;
		uint32 i;

		/* with several sequences this returns nothing on the mksol
		   pass, having written a partial solution for the combine
		   stage to pick up */

		if (bw_mksol(obj, &packed_matrix, &params, max_ncols,
				post_lanczos_matrix, &solution,
				&num_found) != 0)
			goto done;

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
