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

	num = (size_t)(hdr.degree + 1) * hdr.n;
	f = (v_t *)aligned_malloc(num * sizeof(v_t), 64);
	if (fread(f, sizeof(v_t), num, fp) != num) {
		logprintf(obj, "error: Wiedemann generator is truncated\n");
		aligned_free(f);
		fclose(fp);
		return NULL;
	}
	fclose(fp);

	/* Take the seeds from the generator rather than from the command
	   line. They decide y, and the generator only annihilates the
	   sequence built from the y the Krylov stage actually used; a
	   mksol launched without the bw_seed= that the Krylov run was
	   given would otherwise evaluate the generator against a
	   different vector and quietly produce nothing. The seeds
	   travelled here through the sequence files, so this is what was
	   used, not what was asked for. */

	if (hdr.seed1 != params->seed1 || hdr.seed2 != params->seed2) {
		logprintf(obj, "using Wiedemann seed %u from the generator, "
				"not %u\n", hdr.seed1, params->seed1);
		params->seed1 = hdr.seed1;
		params->seed2 = hdr.seed2;
	}

	*degree_out = hdr.degree;
	return f;
}

/*-----------------------------------------------------------------------*/
static void bw_sol_path(msieve_obj *obj, uint32 seq, char *buf) {

	snprintf(buf, BW_PATH_LEN, "%s.bw.w.%u", obj->savefile.name, seq);
}

static int32 write_partial(msieve_obj *obj, bw_params_t *params,
				v_t *w, uint32 ncols) {

	char buf[BW_PATH_LEN];
	FILE *fp;
	bw_sol_header_t hdr;
	int32 status = 0;

	bw_sol_path(obj, params->seq, buf);
	fp = fopen(buf, "wb");
	if (fp == NULL) {
		logprintf(obj, "error: cannot write Wiedemann partial "
				"solution %s\n", buf);
		return -1;
	}

	hdr.magic = BW_SOL_MAGIC;
	hdr.vbits = VBITS;
	hdr.m = params->m_mult * VBITS;
	hdr.n = params->n_mult * VBITS;
	hdr.ncols = ncols;
	hdr.seq = params->seq;
	hdr.seed1 = params->seed1;
	hdr.seed2 = params->seed2;

	if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1 ||
	    fwrite(w, sizeof(v_t), ncols, fp) != ncols) {
		logprintf(obj, "error: cannot write Wiedemann partial "
				"solution\n");
		status = -1;
	}
	if (fclose(fp) != 0)
		status = -1;
	return status;
}

static int32 read_partials(msieve_obj *obj, bw_params_t *params,
				v_t *w, uint32 ncols, v_t *tmp) {

	/* XOR every sequence's partial together. They must agree about
	   the matrix and about the seeds, or they are answers to
	   different questions and summing them is meaningless. */

	char buf[BW_PATH_LEN];
	uint32 jb, i;
	uint32 seed1 = 0, seed2 = 0;

	for (i = 0; i < ncols; i++)
		w[i] = v_zero;

	for (jb = 0; jb < params->n_mult; jb++) {
		FILE *fp;
		bw_sol_header_t hdr;

		bw_sol_path(obj, jb, buf);
		fp = fopen(buf, "rb");
		if (fp == NULL) {
			logprintf(obj, "error: cannot open Wiedemann partial "
					"solution %s\n", buf);
			return -1;
		}
		if (fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
		    hdr.magic != BW_SOL_MAGIC) {
			logprintf(obj, "error: Wiedemann partial solution %u "
					"is corrupt\n", jb);
			fclose(fp);
			return -1;
		}
		if (hdr.vbits != VBITS || hdr.ncols != ncols ||
		    hdr.m != params->m_mult * VBITS ||
		    hdr.n != params->n_mult * VBITS || hdr.seq != jb) {
			logprintf(obj, "error: Wiedemann partial solution %u "
					"does not match this matrix\n", jb);
			fclose(fp);
			return -1;
		}
		if (jb == 0) {
			seed1 = hdr.seed1;
			seed2 = hdr.seed2;
		}
		else if (hdr.seed1 != seed1 || hdr.seed2 != seed2) {
			logprintf(obj, "error: Wiedemann partial solution %u "
					"used seed %u, partial 0 used %u\n",
					jb, hdr.seed1, seed1);
			fclose(fp);
			return -1;
		}
		if (fread(tmp, sizeof(v_t), ncols, fp) != ncols) {
			logprintf(obj, "error: Wiedemann partial solution %u "
					"is truncated\n", jb);
			fclose(fp);
			return -1;
		}
		fclose(fp);

		for (i = 0; i < ncols; i++)
			w[i] = v_xor(w[i], tmp[i]);
	}

	params->seed1 = seed1;
	params->seed2 = seed2;
	logprintf(obj, "combined %u Wiedemann partial solutions\n",
			params->n_mult);
	return 0;
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

	/* One sequence does the whole of stage 3 in one process, as
	   before. Several split it: each process sums only its own
	   sequence and writes the partial, and a combine run XORs them
	   and carries on. The sum is the only part that is per-sequence;
	   everything after it is shared, so both paths meet below. */

	uint32 combining = (params->stage == BW_STAGE_COMBINE);
	uint32 partial = (params->n_mult > 1 && !combining);

	*solution_out = NULL;
	*num_deps_found = 0;

	if (!combining) {
		f = read_generator(obj, params, max_ncols, &degree);
		if (f == NULL)
			return -1;

		logprintf(obj, "commencing Wiedemann mksol, generator "
				"degree %u\n", degree);
		if (partial) {
			logprintf(obj, "summing sequence %u of %u only\n",
					params->seq, params->n_mult);
		}
	}

	w = vv_alloc(n, matrix->extra);
	z = vv_alloc(n, matrix->extra);
	prod = vv_alloc(n, matrix->extra);
	host_w = (v_t *)aligned_malloc((size_t)max_ncols * sizeof(v_t), 64);
	host_next = (v_t *)aligned_malloc((size_t)max_ncols * sizeof(v_t), 64);
	host_res = (v_t *)aligned_malloc((size_t)max_ncols * sizeof(v_t), 64);
	combos = (v_t *)xmalloc(VBITS * sizeof(v_t));

	for (i = 0; i < max_ncols; i++)
		host_w[i] = v_zero;

	/* y is regenerated exactly as the Krylov stage had it, which is
	   why the seeds are fixed and travel in the parameters */

	/* The generator annihilates y itself, not just the sequence seen
	   through x, so sum_k A^k y F_k is zero. That is expected: it is
	   f(A)y for a generator that is a multiple of X, and the vector
	   we want comes from dividing that factor out first.

	   In the block case the valuation is per column, so each column
	   of the generator is shifted down by its own lowest nonzero
	   term. What is evaluated is then killed by A^(e_j) but not,
	   generically, by anything smaller, and the walk below finds
	   exactly where each column dies. */

	if (!combining) {
		uint32 *shift = (uint32 *)xmalloc(VBITS * sizeof(uint32));
		uint32 nrow = params->n_mult * VBITS;
		v_t *fs;
		uint32 max_shift = 0, min_shift = degree + 1;

		for (j = 0; j < VBITS; j++) {
			shift[j] = degree + 1;
			for (k = 0; k <= degree; k++) {
				for (i = 0; i < nrow; i++) {
					if (v_bitset(f[(size_t)k * nrow + i],
							j))
						break;
				}
				if (i < nrow) {
					shift[j] = k;
					break;
				}
			}
			if (shift[j] <= degree) {
				max_shift = MAX(max_shift, shift[j]);
				min_shift = MIN(min_shift, shift[j]);
			}
		}

		fs = (v_t *)aligned_malloc((size_t)(degree + 1) *
						nrow * sizeof(v_t), 64);
		for (i = 0; i < (uint32)(degree + 1) * nrow; i++)
			fs[i] = v_zero;

		for (j = 0; j < VBITS; j++) {
			if (shift[j] > degree)
				continue;
			for (k = 0; k + shift[j] <= degree; k++) {
				v_t *src = f + (size_t)(k + shift[j]) * nrow;
				v_t *dst = fs + (size_t)k * nrow;

				for (i = 0; i < nrow; i++) {
					if (v_bitset(src[i], j))
						bw_v_set_bit(dst + i, j);
				}
			}
		}

		/* A column with no nonzero coefficient at all contributes
		   nothing to W. That is not the only way to lose a
		   column -- a column can be nonzero here and still
		   evaluate to zero, see the warning after the sum -- but
		   the two have different causes, so they are counted
		   apart. */

		for (j = 0, i = 0; j < VBITS; j++) {
			if (shift[j] > degree)
				i++;
		}
		if (i > 0) {
			logprintf(obj, "warning: %u of %u generator columns "
					"are empty and yield no solution\n",
					i, (uint32)VBITS);
		}
		logprintf(obj, "mksol: generator valuations run %u to %u\n",
				min_shift, max_shift);

		aligned_free(f);
		f = fs;
		free(shift);
	}

	/* W = sum_k A^k y F_k, with y the whole n-column block. Split by
	   sequence that is W = sum_jb sum_k A^k y_jb F_k[jb], where
	   F_k[jb] is the VBITS rows of the generator belonging to
	   sequence jb -- so each term stays VBITS wide and uses the same
	   vector operations a single sequence would. The partial
	   solutions simply XOR together. */

	if (!combining) {
		uint32 nrow = params->n_mult * VBITS;
		uint32 first = partial ? params->seq : 0;
		uint32 last = partial ? params->seq + 1 : params->n_mult;
		uint32 jb;

		/* one product per generator coefficient per sequence this
		   process owns, and they all cost the same, so counting
		   them is the whole of the estimate */

		uint32 total = (last - first) * MAX(1, degree);
		uint32 done = 0;
		uint32 report_interval = MAX(1, MIN(total / 100,
					BW_REPORT_MAX_ITER));
		uint32 next_report = report_interval;
		uint32 log_eta_at = MAX(1, MIN(total / 50,
					BW_LOG_ETA_MAX_ITER));
		time_t start_time = time(NULL);

		logprintf(obj, "mksol: %u products of %u x %u\n",
				total, max_ncols, VBITS);

		vv_clear(w, n);

		for (jb = first; jb < last; jb++) {
			uint32 seed1 = params->seed1;
			uint32 seed2 = params->seed2;
			uint32 cleared = 0;

			for (i = 0; i < 1 + jb; i++) {
				for (j = 0; j < n; j++)
					host_w[j] = v_random(&seed1, &seed2);
			}
			vv_copyin(z, host_w, n);

			for (k = 0; ; k++) {

				vv_mul_NxB_BxB_acc(matrix, z,
						f + (size_t)k * nrow +
						(size_t)jb * VBITS, w, n);

				if (k == degree)
					break;

				/* the same two clears the Krylov loop
				   needs, for the same reason, and reset
				   per sequence because copying y back in
				   dirties one buffer again */

				if (cleared < 2) {
					vv_clear(prod, n);
					cleared++;
				}
				mul_MxN_NxB(matrix, z, prod, NULL);
				swap = z; z = prod; prod = swap;

				if (++done >= next_report) {
					double pct = 100.0 * done / total;
					double elapsed = difftime(
						time(NULL), start_time);
					uint32 eta = (uint32)(elapsed *
						(total - done) / done);

					if (BW_IS_NODE_0(obj)) {
						fprintf(stderr, "mksol %u of "
							"%u, %.1f%%, ETA "
							"%dh%2dm    \r",
							done, total, pct,
							eta / 3600,
							(eta % 3600) / 60);
						fflush(stderr);
					}
					if (log_eta_at && done >= log_eta_at) {
						logprintf(obj, "mksol at "
							"%.1f%%, ETA "
							"%dh%2dm\n", pct,
							eta / 3600,
							(eta % 3600) / 60);
						log_eta_at = 0;
					}
					next_report = done + report_interval;
				}
			}
		}
	}

	/* The two paths meet here, both with W in host_w and on the
	   card. A partial run stops instead: its sum is only one term of
	   W, so the walk below would be walking the wrong vector. */

	if (combining) {
		if (read_partials(obj, params, host_w, max_ncols,
					host_next) != 0)
			goto cleanup;
		vv_copyin(w, host_w, n);
	}
	else {
		vv_copyout(host_w, w, n);

		if (partial) {
			status = write_partial(obj, params, host_w, max_ncols);
			if (status == 0) {
				logprintf(obj, "Wiedemann partial solution %u "
						"of %u written\n", params->seq,
						params->n_mult);
			}
			goto cleanup;
		}
	}

	/* Walk the columns forward until each dies, keeping the last
	   nonzero value. A column already in the nullspace dies on the
	   first step, so applying A to the whole block uniformly would
	   throw it away; that is what the liveness mask prevents. */

	{
		/* If this is ever zero the valuations above were wrong and
		   everything downstream is vacuous, so it is worth saying
		   out loud rather than discovering it from an empty .dep */

		v_t wacc = v_zero;

		for (i = 0; i < n; i++)
			wacc = v_or(wacc, host_w[i]);
		if (v_is_all_zeros(wacc)) {
			logprintf(obj, "error: Wiedemann solution block is "
					"entirely zero\n");
			goto cleanup;
		}
		logprintf(obj, "mksol: solution block has %u nonzero "
				"columns\n", bw_v_popcount(wacc));

		/* Fewer than VBITS means some columns evaluated to zero
		   even after their own valuation was stripped: the
		   generator is a higher power of X against y than its
		   coefficients show, and one shift was not enough. Those
		   columns are lost, and with them most of the rank of the
		   answer -- at m = n = 256 on a 550K matrix, 31 of 64
		   columns went this way and the 33 dependencies that came
		   out spanned only 2. Stripping repeatedly, re-evaluating
		   each time, is what this wants; until then a run that
		   trips this is not worth continuing. */

		if (bw_v_popcount(wacc) < VBITS) {
			logprintf(obj, "warning: %u of %u solution columns "
					"evaluated to zero; the dependencies "
					"will be short of rank\n",
					(uint32)VBITS - bw_v_popcount(wacc),
					(uint32)VBITS);
		}
	}

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

	/* What survived has to be something. If no column ever died then
	   nothing was kept, and the gate below would be asking whether
	   the matrix kills the zero vector -- which it does, so every
	   check from here on would pass while the dependencies written
	   out were all zero, and the only sign of trouble would be
	   -nc3 reporting "GCD is 1" sixty-four times hours later.

	   The usual cause is y not matching the one the Krylov stage
	   used, so that the generator annihilates a different sequence
	   than the one being evaluated here. */

	acc = v_zero;
	for (i = 0; i < n; i++)
		acc = v_or(acc, host_res[i]);
	if (v_is_all_zeros(acc)) {
		logprintf(obj, "error: no Wiedemann column reached the "
				"nullspace within %u steps; the solution "
				"would be empty\n", (uint32)MKSOL_MAX_STEPS);
		goto cleanup;
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
