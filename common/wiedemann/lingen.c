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

/* Wiedemann stage 2: the linear generator.

   Given the sequence a_i = x^T A^i y, find F with sum_k a_{i+k} F_k = 0
   for every i in range. That is what lets mksol collapse the Krylov
   vectors into something the matrix kills.

   This is the quadratic sigma-basis algorithm, which is the base case
   of the recursive one. It is O(L^2) in the sequence length, so it is
   fine for a test matrix and hopeless for a production one: the
   recursion that makes it subquadratic is the next piece of work, and
   it will keep calling this for small blocks.

   Method. Set b = m + n and build the m x b series G = [E | I], where
   E is the sequence. Maintain a b x b polynomial matrix pi, starting at
   the identity, and the residual R = G pi. At step t, look at the
   coefficient of X^t in R: eliminate it column by column, taking pivots
   in order of increasing column degree so the degrees stay balanced,
   then multiply each pivot column by X, which clears that coefficient.
   After T steps G pi = 0 mod X^T.

   Writing pi = [pi_top ; pi_bot], that says E pi_top = pi_bot mod X^T,
   and pi_bot has low degree, so every coefficient of E pi_top above
   that degree vanishes. Those are the relations we want -- except that
   they come out as a convolution, sum_k a_{i-k} F_k, while mksol needs
   the correlation sum_k a_{i+k} F_k. Feeding the algorithm the reversed
   sequence converts one into the other. */

#include "wiedemann.h"

#define BW_PATH_LEN 512

/* a matrix of GF(2) polynomials, stored so that a whole column is
   contiguous: the inner loop of the algorithm is "add column j to
   column k", and that wants to be one flat XOR */

typedef struct {
	uint32 nrows;
	uint32 ncols;
	uint32 words;		/* uint64 per polynomial */
	uint64 *data;		/* [col][row][word] */
} polmat_t;

/*-----------------------------------------------------------------------*/
static void pm_init(polmat_t *p, uint32 nrows, uint32 ncols, uint32 bits) {

	p->nrows = nrows;
	p->ncols = ncols;
	p->words = (bits + 63) / 64;
	p->data = (uint64 *)xcalloc((size_t)ncols * nrows * p->words,
					sizeof(uint64));
}

static void pm_free(polmat_t *p) {

	free(p->data);
	p->data = NULL;
}

static uint64 *pm_col(polmat_t *p, uint32 col) {

	return p->data + (size_t)col * p->nrows * p->words;
}

static uint64 *pm_entry(polmat_t *p, uint32 col, uint32 row) {

	return pm_col(p, col) + (size_t)row * p->words;
}

static uint32 pm_coeff(polmat_t *p, uint32 col, uint32 row, uint32 k) {

	return (uint32)((pm_entry(p, col, row)[k >> 6] >> (k & 63)) & 1);
}

static void pm_set_coeff(polmat_t *p, uint32 col, uint32 row, uint32 k) {

	pm_entry(p, col, row)[k >> 6] |= (uint64)1 << (k & 63);
}

/* column k ^= column j */

static void pm_col_xor(polmat_t *p, uint32 k, uint32 j) {

	uint64 *dst = pm_col(p, k);
	uint64 *src = pm_col(p, j);
	size_t i, num = (size_t)p->nrows * p->words;

	for (i = 0; i < num; i++)
		dst[i] ^= src[i];
}

/* column j *= X, one polynomial at a time; bits shifted off the top
   are coefficients past the horizon and are not needed again */

static void pm_col_shift(polmat_t *p, uint32 j) {

	uint64 *col = pm_col(p, j);
	uint32 r, w;

	for (r = 0; r < p->nrows; r++) {
		uint64 *e = col + (size_t)r * p->words;
		uint64 carry = 0;

		for (w = 0; w < p->words; w++) {
			uint64 next = e[w] >> 63;
			e[w] = (e[w] << 1) | carry;
			carry = next;
		}
	}
}

/*-----------------------------------------------------------------------*/
static v_t *read_sequence(msieve_obj *obj, bw_params_t *params,
			uint32 ncols, uint32 *num_terms_out, uint32 *m_out) {

	char buf[BW_PATH_LEN];
	FILE *fp;
	bw_seq_header_t hdr;
	v_t *a;
	size_t num;

	snprintf(buf, sizeof(buf), "%s.bw.a.%u", obj->savefile.name,
			params->seq);
	fp = fopen(buf, "rb");
	if (fp == NULL) {
		logprintf(obj, "error: cannot open Wiedemann sequence %s\n",
				buf);
		return NULL;
	}
	if (fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
	    hdr.magic != BW_SEQ_MAGIC) {
		logprintf(obj, "error: Wiedemann sequence is corrupt\n");
		fclose(fp);
		return NULL;
	}
	if (hdr.vbits != VBITS || hdr.ncols != ncols ||
	    hdr.m != params->m_mult * VBITS ||
	    hdr.n != params->n_mult * VBITS) {
		logprintf(obj, "error: Wiedemann sequence does not match "
				"this matrix\n");
		fclose(fp);
		return NULL;
	}
	if (hdr.num_terms == 0) {
		logprintf(obj, "error: Wiedemann sequence is empty\n");
		fclose(fp);
		return NULL;
	}

	num = (size_t)hdr.num_terms * hdr.m;
	a = (v_t *)aligned_malloc(num * sizeof(v_t), 64);
	if (fread(a, sizeof(v_t), num, fp) != num) {
		logprintf(obj, "error: Wiedemann sequence is truncated\n");
		aligned_free(a);
		fclose(fp);
		return NULL;
	}
	fclose(fp);

	*num_terms_out = hdr.num_terms;
	*m_out = hdr.m;
	return a;
}

/*-----------------------------------------------------------------------*/
static uint32 check_generator(msieve_obj *obj, v_t *a, uint32 num_terms,
				uint32 m, uint32 degree, v_t *f,
				uint32 num_checks) {

	/* The relation mksol relies on, tested directly on the sequence:
	   sum_k a_{i+k} F_k must be zero. Sampled rather than exhaustive,
	   because a full check costs as much as the generator did, and a
	   wrong orientation or an off-by-one shows up immediately. */

	uint32 bad = 0;
	uint32 c, k, r;
	uint32 limit;
	v_t *acc;

	if (num_terms <= degree + 1)
		return 1;
	limit = num_terms - degree - 1;
	acc = (v_t *)xmalloc(m * sizeof(v_t));

	for (c = 0; c < num_checks; c++) {
		uint32 i = (limit <= num_checks) ? c :
				(uint32)((uint64)c * limit / num_checks);

		if (i >= limit)
			break;
		for (r = 0; r < m; r++)
			acc[r] = v_zero;

		/* a_{i+k} is m rows of VBITS bits; F_k is VBITS rows of
		   VBITS bits. Accumulate a_{i+k} * F_k */

		for (k = 0; k <= degree; k++) {
			v_t *ak = a + (size_t)(i + k) * m;
			v_t *fk = f + (size_t)k * VBITS;

			for (r = 0; r < m; r++) {
				uint32 j;

				for (j = 0; j < VBITS; j++) {
					if (v_bitset(ak[r], j))
						acc[r] = v_xor(acc[r], fk[j]);
				}
			}
		}
		for (r = 0; r < m; r++) {
			if (!v_is_all_zeros(acc[r])) {
				bad++;
				break;
			}
		}
	}
	free(acc);
	if (bad)
		logprintf(obj, "generator check failed at %u of %u sampled "
				"points\n", bad, num_checks);
	return bad;
}

/*-----------------------------------------------------------------------*/
int32 bw_lingen(msieve_obj *obj, bw_params_t *params, uint32 max_ncols) {

	uint32 m = 0, n = params->n_mult * VBITS;
	uint32 b, num_terms, T;
	uint32 t, i, j, r, c;
	v_t *a = NULL, *f = NULL;
	polmat_t pi, R;
	uint32 *delta = NULL, *order = NULL, *is_pivot = NULL;
	uint64 *dcol = NULL;
	uint32 mwords;
	uint32 degree = 0;
	int32 status = -1;
	time_t start_time;
	char buf[BW_PATH_LEN];
	FILE *fp;
	bw_gen_header_t hdr;

	a = read_sequence(obj, params, max_ncols, &num_terms, &m);
	if (a == NULL)
		return -1;

	if (params->n_mult != 1) {
		logprintf(obj, "error: lingen handles a single sequence so "
				"far\n");
		aligned_free(a);
		return -1;
	}

	b = m + n;
	T = num_terms;
	mwords = (m + 63) / 64;

	logprintf(obj, "commencing Wiedemann lingen, %u terms, m = %u, "
			"n = %u\n", num_terms, m, n);
	logprintf(obj, "this is the quadratic generator; expect it to be "
			"slow on a large matrix\n");
	start_time = time(NULL);

	/* R starts as G = [E | I], with E the reversed sequence so that
	   the relations come out in the order mksol wants. pi starts as
	   the identity. Both are capped at T coefficients: anything past
	   the horizon is never read. */

	pm_init(&R, m, b, T + 64);
	pm_init(&pi, b, b, T + 64);

	for (i = 0; i < T; i++) {
		v_t *ai = a + (size_t)(T - 1 - i) * m;

		for (r = 0; r < m; r++) {
			for (c = 0; c < n; c++) {
				if (v_bitset(ai[r], c))
					pm_set_coeff(&R, c, r, i);
			}
		}
	}
	for (r = 0; r < m; r++)
		pm_set_coeff(&R, n + r, r, 0);
	for (j = 0; j < b; j++)
		pm_set_coeff(&pi, j, j, 0);

	delta = (uint32 *)xcalloc(b, sizeof(uint32));
	order = (uint32 *)xmalloc(b * sizeof(uint32));
	is_pivot = (uint32 *)xmalloc(b * sizeof(uint32));
	dcol = (uint64 *)xmalloc((size_t)b * mwords * sizeof(uint64));

	for (t = 0; t < T; t++) {

		/* the coefficient of X^t in R, one m-bit column per
		   column of R */

		for (j = 0; j < b; j++) {
			uint64 *d = dcol + (size_t)j * mwords;

			for (i = 0; i < mwords; i++)
				d[i] = 0;
			for (r = 0; r < m; r++) {
				if (pm_coeff(&R, j, r, t))
					d[r >> 6] |= (uint64)1 << (r & 63);
			}
			is_pivot[j] = 0;
		}

		/* visit columns in order of increasing degree, which is
		   what keeps the basis minimal */

		for (j = 0; j < b; j++)
			order[j] = j;
		for (i = 1; i < b; i++) {
			uint32 key = order[i];
			j = i;
			while (j > 0 && delta[order[j - 1]] > delta[key]) {
				order[j] = order[j - 1];
				j--;
			}
			order[j] = key;
		}

		for (r = 0; r < m; r++) {
			uint32 piv = (uint32)-1;
			uint64 *dp;

			for (i = 0; i < b; i++) {
				uint32 col = order[i];

				if (is_pivot[col])
					continue;
				if (dcol[(size_t)col * mwords + (r >> 6)] &
						((uint64)1 << (r & 63))) {
					piv = col;
					break;
				}
			}
			if (piv == (uint32)-1)
				continue;

			is_pivot[piv] = 1;
			dp = dcol + (size_t)piv * mwords;

			/* clear this row from every column that is not
			   itself a pivot; a pivot column keeps whatever
			   it has, because multiplying it by X moves this
			   coefficient out of the way anyway */

			for (c = 0; c < b; c++) {
				uint64 *dc = dcol + (size_t)c * mwords;

				if (c == piv || is_pivot[c])
					continue;
				if (!(dc[r >> 6] & ((uint64)1 << (r & 63))))
					continue;
				for (i = 0; i < mwords; i++)
					dc[i] ^= dp[i];
				pm_col_xor(&pi, c, piv);
				pm_col_xor(&R, c, piv);
			}
		}

		for (j = 0; j < b; j++) {
			if (is_pivot[j]) {
				pm_col_shift(&pi, j);
				pm_col_shift(&R, j);
				delta[j]++;
			}
		}

		if ((t & 1023) == 0) {
			fprintf(stderr, "lingen %u of %u, %.1f%%\r", t, T,
					100.0 * t / T);
			fflush(stderr);
		}
	}

	/* every coefficient below T must now be gone from R; that is the
	   invariant the whole method rests on, and it is cheap to check */

	for (t = 0; t < T; t++) {
		for (j = 0; j < b; j++) {
			for (r = 0; r < m; r++) {
				if (pm_coeff(&R, j, r, t)) {
					logprintf(obj, "error: lingen "
						"residual is nonzero at "
						"term %u\n", t);
					goto cleanup;
				}
			}
		}
	}

	/* take the n columns of least degree and read the generator out
	   of the top n rows of pi */

	for (j = 0; j < b; j++)
		order[j] = j;
	for (i = 1; i < b; i++) {
		uint32 key = order[i];
		j = i;
		while (j > 0 && delta[order[j - 1]] > delta[key]) {
			order[j] = order[j - 1];
			j--;
		}
		order[j] = key;
	}

	degree = 0;
	for (i = 0; i < n; i++)
		degree = MAX(degree, delta[order[i]]);

	f = (v_t *)aligned_malloc((size_t)(degree + 1) * VBITS * sizeof(v_t),
					64);
	for (i = 0; i < (size_t)(degree + 1) * VBITS; i++)
		f[i] = v_zero;

	for (i = 0; i < n; i++) {
		uint32 col = order[i];

		for (r = 0; r < n; r++) {
			for (t = 0; t <= delta[col]; t++) {
				if (pm_coeff(&pi, col, r, t))
					bw_v_set_bit(f + (size_t)t * VBITS + r,
							i);
			}
		}
	}

	logprintf(obj, "generator has degree %u, %.1f sec\n", degree,
			difftime(time(NULL), start_time));

	if (check_generator(obj, a, num_terms, m, degree, f, 64) != 0)
		goto cleanup;

	logprintf(obj, "generator verified against the sequence\n");

	snprintf(buf, sizeof(buf), "%s.bw.f", obj->savefile.name);
	fp = fopen(buf, "wb");
	if (fp == NULL) {
		logprintf(obj, "error: cannot write Wiedemann generator\n");
		goto cleanup;
	}
	hdr.magic = BW_GEN_MAGIC;
	hdr.vbits = VBITS;
	hdr.m = m;
	hdr.n = n;
	hdr.ncols = max_ncols;
	hdr.degree = degree;
	if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1 ||
	    fwrite(f, sizeof(v_t), (size_t)(degree + 1) * VBITS, fp) !=
			(size_t)(degree + 1) * VBITS) {
		logprintf(obj, "error: cannot write Wiedemann generator\n");
		fclose(fp);
		goto cleanup;
	}
	fclose(fp);
	status = 0;

cleanup:
	aligned_free(f);
	free(dcol);
	free(is_pivot);
	free(order);
	free(delta);
	pm_free(&pi);
	pm_free(&R);
	aligned_free(a);
	return status;
}
