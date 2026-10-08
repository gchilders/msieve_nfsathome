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
static v_t *read_sequences(msieve_obj *obj, bw_params_t *params,
			uint32 ncols, uint32 *num_terms_out, uint32 *m_out) {

	/* The n columns of the method are split into n_mult sequences of
	   VBITS, each computed by its own process with nothing shared.
	   Here they are put back together: term i is m rows of n bits,
	   held as n_mult v_t per row, with block jb coming from file jb.
	   Every file has to agree about m, n, the matrix, and how many
	   terms it holds. */

	char buf[BW_PATH_LEN];
	FILE *fp;
	bw_seq_header_t hdr;
	uint32 k = params->n_mult;
	uint32 jb, t, r;
	uint32 m = 0, num_terms = 0;
	v_t *a = NULL, *one = NULL;

	for (jb = 0; jb < k; jb++) {
		size_t num;

		snprintf(buf, sizeof(buf), "%s.bw.a.%u",
				obj->savefile.name, jb);
		fp = fopen(buf, "rb");
		if (fp == NULL) {
			logprintf(obj, "error: cannot open Wiedemann "
					"sequence %s\n", buf);
			goto fail;
		}
		if (fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
		    hdr.magic != BW_SEQ_MAGIC) {
			logprintf(obj, "error: Wiedemann sequence %u is "
					"corrupt\n", jb);
			fclose(fp);
			goto fail;
		}
		if (hdr.vbits != VBITS || hdr.ncols != ncols ||
		    hdr.m != params->m_mult * VBITS ||
		    hdr.n != k * VBITS || hdr.seq != jb) {
			logprintf(obj, "error: Wiedemann sequence %u does "
					"not match this matrix\n", jb);
			fclose(fp);
			goto fail;
		}
		if (hdr.num_terms == 0) {
			logprintf(obj, "error: Wiedemann sequence %u is "
					"empty\n", jb);
			fclose(fp);
			goto fail;
		}

		if (jb == 0) {
			m = hdr.m;
			num_terms = hdr.num_terms;

			/* The sequences are the record of which x and y were
			   used; carry that forward so the generator can say
			   so too, and mksol need not be told again. */

			params->seed1 = hdr.seed1;
			params->seed2 = hdr.seed2;

			a = (v_t *)aligned_malloc((size_t)num_terms * m * k *
							sizeof(v_t), 64);
			one = (v_t *)aligned_malloc((size_t)num_terms * m *
							sizeof(v_t), 64);
		}
		else if (hdr.m != m || hdr.num_terms < num_terms) {
			logprintf(obj, "error: Wiedemann sequence %u holds "
					"%u terms, expected %u\n", jb,
					hdr.num_terms, num_terms);
			fclose(fp);
			goto fail;
		}
		else if (hdr.seed1 != params->seed1 ||
			 hdr.seed2 != params->seed2) {

			/* different seeds mean different x, so these terms
			   are not part of the same sequence as sequence 0
			   even though every other field agrees */

			logprintf(obj, "error: Wiedemann sequence %u was "
					"built with seed %u, sequence 0 with "
					"%u\n", jb, hdr.seed1, params->seed1);
			fclose(fp);
			goto fail;
		}

		num = (size_t)num_terms * m;
		if (fread(one, sizeof(v_t), num, fp) != num) {
			logprintf(obj, "error: Wiedemann sequence %u is "
					"truncated\n", jb);
			fclose(fp);
			goto fail;
		}
		fclose(fp);

		for (t = 0; t < num_terms; t++) {
			for (r = 0; r < m; r++) {
				a[((size_t)t * m + r) * k + jb] =
						one[(size_t)t * m + r];
			}
		}
	}

	aligned_free(one);
	*num_terms_out = num_terms;
	*m_out = m;
	return a;

fail:
	aligned_free(one);
	aligned_free(a);
	return NULL;
}

/*-----------------------------------------------------------------------*/
static uint32 check_generator(msieve_obj *obj, v_t *a, uint32 num_terms,
				uint32 m, uint32 nblk, uint32 degree,
				v_t *f, uint32 num_checks) {

	/* The relation mksol relies on, tested directly on the sequence:
	   sum_k a_{i+k} F_k must be zero. Sampled rather than exhaustive,
	   because a full check costs as much as the generator did, and a
	   wrong orientation or an off-by-one shows up immediately.

	   a_{i+k} is m rows of n bits, held as nblk v_t per row; F_k is
	   n rows of VBITS bits, one v_t each. */

	uint32 bad = 0;
	uint32 c, k, r, jb;
	uint32 limit;
	uint32 n = nblk * VBITS;
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

		for (k = 0; k <= degree; k++) {
			v_t *ak = a + (size_t)(i + k) * m * nblk;
			v_t *fk = f + (size_t)k * n;

			for (r = 0; r < m; r++) {
				for (jb = 0; jb < nblk; jb++) {
					v_t w = ak[(size_t)r * nblk + jb];
					uint32 j;

					for (j = 0; j < VBITS; j++) {
						if (v_bitset(w, j))
							acc[r] = v_xor(acc[r],
								fk[jb * VBITS
									+ j]);
					}
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
/* A view of coefficients [first, first+len) of p. Shares storage, so it
   is never freed */

static void bmp_view(bmp_t *out, const bmp_t *in, uint32 first,
			uint32 len) {

	out->nrows = in->nrows;
	out->ncols = in->ncols;
	out->rwords = in->rwords;
	out->len = len;
	out->data = in->data + (size_t)first * in->nrows * in->rwords;
}

/*-----------------------------------------------------------------------*/
void quadratic_basis(const bmp_t *G, uint32 T, uint32 *delta,
				bmp_t *pi_out) {

	/* The base of the recursion, and the whole algorithm when the
	   problem is small: eliminate the coefficient of X^t from the
	   residual one t at a time, taking pivots in order of increasing
	   column degree so the degrees stay balanced, and multiplying
	   each pivot column by X.

	   This works column at a time, so it wants the column-major
	   layout rather than the coefficient-major one the multiply
	   uses; the two conversions are linear in the size and happen
	   once per call. */

	uint32 m = G->nrows;
	uint32 b = G->ncols;
	uint32 mwords = (m + 63) / 64;
	polmat_t R, pi;
	uint32 *order, *is_pivot;
	uint64 *dcol;
	uint32 t, i, j, r, c, maxdelta;

	pm_init(&R, m, b, T + 64);
	pm_init(&pi, b, b, T + 64);

	for (t = 0; t < T; t++) {
		const uint64 *gc = G->data +
				(size_t)t * G->nrows * G->rwords;

		for (r = 0; r < m; r++) {
			const uint64 *row = gc + (size_t)r * G->rwords;

			for (c = 0; c < b; c++) {
				if ((row[c >> 6] >> (c & 63)) & 1)
					pm_set_coeff(&R, c, r, t);
			}
		}
	}
	for (j = 0; j < b; j++)
		pm_set_coeff(&pi, j, j, 0);

	order = (uint32 *)xmalloc(b * sizeof(uint32));
	is_pivot = (uint32 *)xmalloc(b * sizeof(uint32));
	dcol = (uint64 *)xmalloc((size_t)b * mwords * sizeof(uint64));

	for (t = 0; t < T; t++) {

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
	}

	/* How long the result actually is, measured rather than inferred.

	   delta is the running total over the whole recursion, so sizing
	   from it would allocate the accumulated degree at every leaf.
	   But the local shift count is not a bound either: pivots are
	   taken in order of global delta, so a column with few local
	   shifts can have one with many XORed into it, and its degree is
	   then larger than its own shift count. Truncating to that loses
	   real coefficients, and the basis silently stops satisfying
	   G pi = 0. */

	maxdelta = 0;
	for (c = 0; c < b; c++) {
		for (r = 0; r < b; r++) {
			uint64 *e = pm_entry(&pi, c, r);
			uint32 w = pi.words;

			while (w > 0 && e[w - 1] == 0)
				w--;
			if (w > 0) {
				uint32 top = 64 * (w - 1);

				for (i = 63; i > 0; i--) {
					if (e[w - 1] & ((uint64)1 << i))
						break;
				}
				maxdelta = MAX(maxdelta, top + i);
			}
		}
	}

	bmp_init(pi_out, b, b, maxdelta + 1);
	for (t = 0; t <= maxdelta; t++) {
		uint64 *pc = bmp_coeff(pi_out, t);

		for (r = 0; r < b; r++) {
			uint64 *row = pc + (size_t)r * pi_out->rwords;

			for (c = 0; c < b; c++) {
				if (pm_coeff(&pi, c, r, t))
					row[c >> 6] |= (uint64)1 << (c & 63);
			}
		}
	}

	free(dcol);
	free(is_pivot);
	free(order);
	pm_free(&pi);
	pm_free(&R);
}

/*-----------------------------------------------------------------------*/
#ifndef LINGEN_BASE_CASE
#define LINGEN_BASE_CASE 64
#endif

void recursive_basis(const bmp_t *G, uint32 T, uint32 *delta,
				bmp_t *pi_out) {

	/* Divide and conquer on the order. Solve the first half, push the
	   series through what that produced, and solve what is left of
	   it; the two bases compose by multiplication.

	   G * pi1 has its first T1 coefficients zero by construction,
	   which is exactly what makes the slice at T1 the right input for
	   the second half. */

	uint32 T1, T2;
	bmp_t Gv, pi1, pi2, E, Gsub;

	if (T <= LINGEN_BASE_CASE) {
		quadratic_basis(G, T, delta, pi_out);
		return;
	}

	T1 = T / 2;
	T2 = T - T1;

	bmp_view(&Gv, G, 0, T1);
	recursive_basis(&Gv, T1, delta, &pi1);

	bmp_init(&E, G->nrows, pi1.ncols, T + pi1.len - 1);
	{
		bmp_t Gfull;

		bmp_view(&Gfull, G, 0, T);
		bmp_mul(&E, &Gfull, &pi1);
	}

	bmp_view(&Gsub, &E, T1, MIN(T2, E.len - T1));
	recursive_basis(&Gsub, T2, delta, &pi2);

	bmp_init(pi_out, pi1.nrows, pi2.ncols, pi1.len + pi2.len - 1);
	bmp_mul(pi_out, &pi1, &pi2);

	bmp_free(&E);
	bmp_free(&pi2);
	bmp_free(&pi1);
}

/*-----------------------------------------------------------------------*/
int32 bw_lingen(msieve_obj *obj, bw_params_t *params, uint32 max_ncols) {

	uint32 m = 0, n = params->n_mult * VBITS;
	uint32 b, num_terms, T;
	uint32 t, i, j, r, c;
	v_t *a = NULL, *f = NULL;
	bmp_t G, pi;
	uint32 *delta = NULL, *order = NULL;
	uint32 degree = 0;
	int32 status = -1;
	time_t start_time;
	char buf[BW_PATH_LEN];
	FILE *fp;
	bw_gen_header_t hdr;

	a = read_sequences(obj, params, max_ncols, &num_terms, &m);
	if (a == NULL)
		return -1;

	b = m + n;
	T = num_terms;

	logprintf(obj, "commencing Wiedemann lingen, %u terms, m = %u, "
			"n = %u\n", num_terms, m, n);
	start_time = time(NULL);

	/* G = [E | I] with E the reversed sequence, so that the relations
	   come out as the correlation mksol needs rather than the
	   convolution the basis naturally produces */

	bmp_init(&G, m, b, T);
	for (t = 0; t < T; t++) {
		v_t *ai = a + (size_t)(T - 1 - t) * m * params->n_mult;
		uint64 *gc = bmp_coeff(&G, t);

		for (r = 0; r < m; r++) {
			uint64 *row = gc + (size_t)r * G.rwords;

			/* column c of the sequence lives in block
			   c / VBITS, which is the file it came from */

			for (c = 0; c < n; c++) {
				v_t w = ai[(size_t)r * params->n_mult +
						c / VBITS];

				if (v_bitset(w, c % VBITS))
					row[c >> 6] |= (uint64)1 << (c & 63);
			}
			if (t == 0)
				row[(n + r) >> 6] |=
					(uint64)1 << ((n + r) & 63);
		}
	}

	delta = (uint32 *)xcalloc(b, sizeof(uint32));
	recursive_basis(&G, T, delta, &pi);

	/* G pi must be zero below X^T; that is the whole invariant, and
	   the recursion has enough moving parts to be worth checking */

	{
		bmp_t chk;
		uint32 bad = 0;

		bmp_init(&chk, m, b, T + pi.len - 1);
		bmp_mul(&chk, &G, &pi);
		for (t = 0; t < T && !bad; t++) {
			uint64 *cc = bmp_coeff(&chk, t);
			size_t w, num = (size_t)m * chk.rwords;

			for (w = 0; w < num; w++) {
				if (cc[w]) {
					bad = 1;
					break;
				}
			}
		}
		bmp_free(&chk);
		if (bad) {
			logprintf(obj, "error: lingen residual is nonzero "
					"at term %u\n", t - 1);
			goto cleanup;
		}
	}

	/* read the generator out of the top n rows of the columns of
	   least degree */

	order = (uint32 *)xmalloc(b * sizeof(uint32));
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

	/* The solution block is VBITS wide, because that is what a v_t
	   holds and what the dependency file records, so only the VBITS
	   columns of least degree are kept. Each coefficient of the
	   generator is then n rows of VBITS bits -- one v_t per row of
	   y, across all the sequences. */

	degree = 0;
	for (i = 0; i < VBITS; i++)
		degree = MAX(degree, delta[order[i]]);
	if (degree >= pi.len)
		degree = pi.len - 1;

	f = (v_t *)aligned_malloc((size_t)(degree + 1) * n * sizeof(v_t),
					64);
	for (i = 0; i < (uint32)(degree + 1) * n; i++)
		f[i] = v_zero;

	for (i = 0; i < VBITS; i++) {
		uint32 col = order[i];

		for (t = 0; t <= degree; t++) {
			uint64 *pc = bmp_coeff(&pi, t);

			for (r = 0; r < n; r++) {
				uint64 *row = pc + (size_t)r * pi.rwords;

				if ((row[col >> 6] >> (col & 63)) & 1)
					bw_v_set_bit(f + (size_t)t * n + r,
							i);
			}
		}
	}

	logprintf(obj, "generator has degree %u, %.1f sec\n", degree,
			difftime(time(NULL), start_time));

	if (check_generator(obj, a, num_terms, m, params->n_mult, degree,
			f, 64) != 0)
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
	hdr.seed1 = params->seed1;
	hdr.seed2 = params->seed2;
	if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1 ||
	    fwrite(f, sizeof(v_t), (size_t)(degree + 1) * n, fp) !=
			(size_t)(degree + 1) * n) {
		logprintf(obj, "error: cannot write Wiedemann generator\n");
		fclose(fp);
		goto cleanup;
	}
	fclose(fp);
	status = 0;

cleanup:
	aligned_free(f);
	free(order);
	free(delta);
	bmp_free(&pi);
	bmp_free(&G);
	aligned_free(a);
	return status;
}
