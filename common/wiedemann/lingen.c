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

/* column j *= X, one polynomial at a time; bits shifted off the top
   are coefficients past the horizon and are not needed again */

static void pm_col_shift(polmat_t *p, uint32 j, uint32 lo, uint32 hi) {

	uint64 *col = pm_col(p, j);
	uint32 r, w;

	for (r = lo; r < hi; r++) {
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
	uint32 k;
	uint32 jb, t, r;
	uint32 m = 0, num_terms = 0;
	v_t *a = NULL, *one = NULL;

	/* m and n come from the sequences themselves, the way the seeds
	   below already do. Krylov wrote them there, and this stage owns
	   no sequence of its own, so it has nothing better to go on --
	   under MPI bw_m and bw_n default to the rank count, which is
	   the right answer only when lingen happens to be run on as many
	   ranks as there are sequences. It is free to run on more or
	   fewer, since it splits a product rather than owning a
	   sequence, and then the defaults would be wrong and the shape
	   check below would reject the files for a number nobody
	   chose. */

	snprintf(buf, sizeof(buf), "%s.bw.a.0", obj->savefile.name);
	fp = fopen(buf, "rb");
	if (fp == NULL) {
		logprintf(obj, "error: cannot open Wiedemann sequence %s\n",
				buf);
		return NULL;
	}
	if (fread(&hdr, sizeof(hdr), 1, fp) != 1 ||
	    hdr.magic != BW_SEQ_MAGIC || hdr.vbits != VBITS ||
	    hdr.m == 0 || hdr.n == 0) {
		logprintf(obj, "error: Wiedemann sequence 0 is corrupt\n");
		fclose(fp);
		return NULL;
	}
	fclose(fp);

	if (hdr.m != params->m_mult * VBITS ||
	    hdr.n != params->n_mult * VBITS) {
		logprintf(obj, "lingen: taking m = %u, n = %u from the "
				"sequences\n", hdr.m, hdr.n);
	}
	params->m_mult = hdr.m / VBITS;
	params->n_mult = hdr.n / VBITS;
	k = params->n_mult;

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

	int32 bad = 0;
	int32 c;
	uint32 limit;
	uint32 n = nblk * VBITS;

	if (num_terms <= degree + 1)
		return 1;
	limit = num_terms - degree - 1;

	/* The points are independent and there is one scan of the whole
	   sequence in each, so this is the one place in lingen where the
	   work per unit of scheduling is seconds rather than
	   microseconds. It was 190 of the 448 sec the stage took on a
	   4.8M matrix, all of it on one core of 46. */

#ifdef _OPENMP
	#pragma omp parallel for schedule(dynamic, 1) reduction(+:bad)
#endif
	for (c = 0; c < (int32)num_checks; c++) {

		uint32 k, r, jb;
		uint32 i = (limit <= num_checks) ? (uint32)c :
				(uint32)((uint64)c * limit / num_checks);
		v_t *acc;

		/* a point past the end of the sequence is not a failure,
		   there is simply nothing there to test */

		if (i >= limit)
			continue;

		acc = (v_t *)xmalloc(m * sizeof(v_t));
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
		free(acc);
	}
	if (bad)
		logprintf(obj, "generator check failed at %u of %u sampled "
				"points\n", (uint32)bad, num_checks);
	return (uint32)bad;
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
/* The base case eliminates one pivot row at a time, and each pivot
   touches only a few dozen columns -- a few microseconds of work. That
   is far too little to synchronise around: an `omp for' per pivot row
   measured 1.8x on 16 cores and then got worse, because the barrier
   costs about as much as the work between two of them.

   So the rows are taken a block at a time. Within a block the
   elimination is replayed on dcol alone, which is mwords per column
   rather than a whole column of pi and R, and all it records is, for
   each column, a QB_RBLK-bit mask of which of the block's pivot
   columns were XORed into it. Nothing in pi or R moves until the end
   of the block, and then every column moves at once.

   Applying a mask of k bits naively costs k/2 column XORs, which would
   be ~5x the work the row-at-a-time version does. The method of four
   Russians brings that back: the pivot columns are taken QB_GBITS at a
   time and every XOR of that group is tabulated once, so a column
   spends one XOR per group however many bits it has set. At
   QB_GBITS = 4 that is 16 XORs for a 64-bit mask instead of 32, and
   ~1.8x the total work for 22x fewer barriers. */

#define QB_RBLK 64		/* pivot rows per block; cmask is a uint64 */
#define QB_GBITS 4
#define QB_GSIZE (1 << QB_GBITS)
#define QB_NGRP (QB_RBLK / QB_GBITS)

/* Rows per band, below which a thread is not worth adding: the
   inner XOR is then a handful of words and the loop costs more
   than the XOR, while the per-thread copies of the pivot search
   go on growing. Measured on a 48-core EPYC at b = 256, where
   this caps the team at 16: the base case runs 7.7 sec at 16
   threads against 11.1 at 48. It is a band size and not a thread
   count on purpose, so a wider b uses more of the machine --
   b = 1024 would allow 64. */

#define QB_MIN_BAND 16

static INLINE uint32 qb_ctz(uint64 x) {

#if defined(__GNUC__)
	return (uint32)__builtin_ctzll(x);
#else
	uint32 n = 0;

	while (!(x & 1)) {
		x >>= 1;
		n++;
	}
	return n;
#endif
}

static INLINE void qb_xor(uint64 *dst, const uint64 *src, size_t n) {

	size_t i;

	for (i = 0; i < n; i++)
		dst[i] ^= src[i];
}

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
	uint64 *dcol_shared;
	uint32 nteam, nrank, myrank;
	uint32 pb0, pb1, rb0, rb1;
	uint32 t, i, j, r, c, maxdelta;
	int32 tp;

	/* pi needs T + 1 bits and no more. A column is shifted at most
	   once per step, so after T steps no column has been shifted more
	   than T times, and a XOR of columns is bounded by the largest of
	   them -- so every entry has degree at most T. That is a global
	   bound and holds whatever the pivot order does, which is what
	   makes it safe where the per-column shift count below is not.

	   It is worth being exact about: at the leaf sizes the recursion
	   actually produces, T + 64 bits rounds up to two words and
	   T + 1 to one, which halves every column XOR and shift pi takes
	   in the elimination. Locally that was 3.9 -> 3.1 sec of base
	   case, 2.1 -> 1.5 of it in the elimination. R keeps its slack:
	   it starts with T coefficients and is shifted too, so it needs
	   nearly 2T and the rounding lands on two words either way. */

	pm_init(&R, m, b, T + 64);
	pm_init(&pi, b, b, T + 1);

	/* The ranks take a band of rows each, exactly as the threads do
	   inside one. This was the last part of lingen still done in
	   full by every rank, and the only part that got slower as
	   ranks were added -- each redid all of it while competing for
	   the same cache: 13.5, 18.1, 22.2 sec at 1, 2 and 4 ranks. */

	nrank = bmp_mpi_size();
	myrank = bmp_mpi_rank();
	pb0 = (uint32)((uint64)b * myrank / nrank);
	pb1 = (uint32)((uint64)b * (myrank + 1) / nrank);
	rb0 = (uint32)((uint64)m * myrank / nrank);
	rb1 = (uint32)((uint64)m * (myrank + 1) / nrank);

	for (t = 0; t < T; t++) {
		const uint64 *gc = G->data +
				(size_t)t * G->nrows * G->rwords;

		for (r = rb0; r < rb1; r++) {
			const uint64 *row = gc + (size_t)r * G->rwords;

			for (c = 0; c < b; c++) {
				if ((row[c >> 6] >> (c & 63)) & 1)
					pm_set_coeff(&R, c, r, t);
			}
		}
	}
	for (j = 0; j < b; j++)
		pm_set_coeff(&pi, j, j, 0);

	dcol_shared = (uint64 *)xmalloc((size_t)b * mwords * sizeof(uint64));

	/* and the thread bands then divide this rank's band, not the
	   whole matrix */

	nteam = 1;
#ifdef _OPENMP
	nteam = (uint32)omp_get_max_threads();
	if (nteam > (pb1 - pb0) / QB_MIN_BAND)
		nteam = (pb1 - pb0) / QB_MIN_BAND;
	if (nteam < 1)
		nteam = 1;
#endif

	/* One region for the whole elimination, and almost nothing
	   shared inside it. */

#ifdef _OPENMP
	#pragma omp parallel num_threads(nteam)
#endif
	{
	uint32 t, i, j, r, r0, k;
	int32 jp;	/* the shared-out loop, which OpenMP wants signed */
	uint32 nth = 1, mytid = 0;
	uint32 plo, phi, rlo, rhi;
	size_t pioff, Roff, piw, Rw;
	uint64 *tab_pi, *tab_R;
	uint32 *order = (uint32 *)xmalloc(b * sizeof(uint32));
	uint32 *is_pivot = (uint32 *)xmalloc(b * sizeof(uint32));
	uint32 *pivcols = (uint32 *)xmalloc(QB_RBLK * sizeof(uint32));
	uint32 *mydelta = (uint32 *)xmalloc(b * sizeof(uint32));
	uint64 *dcol = (uint64 *)xmalloc((size_t)b * mwords * sizeof(uint64));
	uint64 *cmask = (uint64 *)xmalloc(b * sizeof(uint64));
	uint64 *fmask = (uint64 *)xmalloc(b * sizeof(uint64));
	uint64 *pmask = (uint64 *)xmalloc(QB_RBLK * sizeof(uint64));
	uint64 *qmask = (uint64 *)xmalloc(QB_RBLK * sizeof(uint64));

	/* Each thread owns a band of rows of pi and of R, and does every
	   column over its own band. Splitting by column instead looked
	   natural and did not work: the tables are then built by a few
	   threads and read by all of them, so on a multi-die part every
	   lookup crosses the fabric. Measured on a 48-core EPYC at
	   b = 256, the apply would not move off 5.2 sec however many
	   threads it was given. By rows nothing is shared -- each thread
	   builds the slice of the tables it is about to read -- so the
	   only barrier left is the one for dcol, once per step.

	   It is the same split the distributed lingen uses, for the same
	   reason: a band of the output needs only that band of the
	   inputs. */

#ifdef _OPENMP
	nth = (uint32)omp_get_num_threads();
	mytid = (uint32)omp_get_thread_num();
#endif
	plo = pb0 + (uint32)((uint64)(pb1 - pb0) * mytid / nth);
	phi = pb0 + (uint32)((uint64)(pb1 - pb0) * (mytid + 1) / nth);
	rlo = rb0 + (uint32)((uint64)(rb1 - rb0) * mytid / nth);
	rhi = rb0 + (uint32)((uint64)(rb1 - rb0) * (mytid + 1) / nth);
	pioff = (size_t)plo * pi.words;
	Roff = (size_t)rlo * R.words;
	piw = (size_t)(phi - plo) * pi.words;
	Rw = (size_t)(rhi - rlo) * R.words;
	tab_pi = (uint64 *)xmalloc(QB_NGRP * QB_GSIZE *
				(piw ? piw : 1) * sizeof(uint64));
	tab_R = (uint64 *)xmalloc(QB_NGRP * QB_GSIZE *
				(Rw ? Rw : 1) * sizeof(uint64));

	/* delta is the one thing every thread both reads and advances,
	   and sharing it is a race that is easy to miss: the pivot order
	   is sorted on delta at the top of a step, while a thread that
	   has run ahead is advancing it at the bottom of the same step.
	   Nothing orders those two, and the threads then disagree about
	   the pivot order and silently compute different things -- found
	   by hashing each thread's pivot choice and comparing. Every
	   thread advances its own copy by the same rule instead, which
	   costs b words and removes the question. */

	memcpy(mydelta, delta, b * sizeof(uint32));

#ifdef LINGEN_PROFILE
	double qt = 0;
	uint32 tid = mytid;

	/* every thread runs the same sequence of steps, so thread 0's
	   elapsed time between barriers is the wall time of the phase;
	   letting all of them report would turn these into totals over
	   threads, which is not what the other base-case lines mean */

#define QB_TICK(slot) if (tid == 0) {					\
		lingen_prof_add(slot, lingen_wtime() - qt);		\
		qt = lingen_wtime();					\
	}
#else
#define QB_TICK(slot)
#endif

	for (t = 0; t < T; t++) {

#ifdef LINGEN_PROFILE
		if (tid == 0)
			qt = lingen_wtime();
#endif

		/* the only barrier in the step: dcol is taken from every
		   row of R, so every band has to be written first */

#ifdef _OPENMP
		#pragma omp barrier
		#pragma omp for
#endif
		for (jp = 0; jp < (int32)b; jp++) {
			uint64 *d = dcol_shared + (size_t)jp * mwords;
			uint32 iw, rw;

			for (iw = 0; iw < mwords; iw++)
				d[iw] = 0;
			for (rw = rb0; rw < rb1; rw++) {
				if (pm_coeff(&R, (uint32)jp, rw, t))
					d[rw >> 6] |= (uint64)1 << (rw & 63);
			}
		}

		/* only this rank's rows of R were there to read, and the
		   bands are disjoint, so the XOR is the whole of it. This
		   is the one exchange per step -- everything after it is
		   local again. */

		if (nrank > 1) {
#ifdef _OPENMP
			#pragma omp single
#endif
			bmp_combine_words(dcol_shared,
					(size_t)b * mwords);
		}

		/* from here to the end of the step every thread works on
		   its own copy, so there is nothing to order */

		memcpy(dcol, dcol_shared,
				(size_t)b * mwords * sizeof(uint64));
		for (j = 0; j < b; j++)
			is_pivot[j] = 0;

		QB_TICK(LP_QB_BUILD)

		for (j = 0; j < b; j++)
			order[j] = j;
		for (i = 1; i < b; i++) {
			uint32 key = order[i];

			j = i;
			while (j > 0 && mydelta[order[j - 1]] > mydelta[key]) {
				order[j] = order[j - 1];
				j--;
			}
			order[j] = key;
		}

		QB_TICK(LP_QB_SORT)

		for (r0 = 0; r0 < m; r0 += QB_RBLK) {
			uint32 nr = MIN(QB_RBLK, m - r0);
			uint32 rr, ngrp, g;
#ifdef LINGEN_PROFILE
			double bt = lingen_wtime();
#define QB_BTICK(slot) if (tid == 0) {					\
		lingen_prof_add(slot, lingen_wtime() - bt);		\
		bt = lingen_wtime();					\
	}
#else
#define QB_BTICK(slot)
#endif

			for (j = 0; j < b; j++)
				cmask[j] = 0;
			k = 0;

			/* the elimination, on dcol only */

			for (rr = 0; rr < nr; rr++) {
				uint32 piv = (uint32)-1;
				uint64 *dp;

				r = r0 + rr;
				for (i = 0; i < b; i++) {
					uint32 col = order[i];

					if (is_pivot[col])
						continue;
					if (dcol[(size_t)col * mwords +
							(r >> 6)] &
						((uint64)1 << (r & 63))) {
						piv = col;
						break;
					}
				}
				if (piv == (uint32)-1)
					continue;

				is_pivot[piv] = 1;
				pivcols[k] = piv;
				pmask[k] = cmask[piv];
				dp = dcol + (size_t)piv * mwords;

				for (j = 0; j < b; j++) {
					uint64 *dc = dcol +
						(size_t)j * mwords;
					uint32 iw;

					if (j == piv || is_pivot[j])
						continue;
					if (!(dc[r >> 6] &
						((uint64)1 << (r & 63))))
						continue;
					for (iw = 0; iw < mwords; iw++)
						dc[iw] ^= dp[iw];
					cmask[j] ^= (uint64)1 << k;
				}
				k++;
			}

			QB_BTICK(LP_QB_SYM)
			if (k == 0)
				continue;

			/* A pivot column is itself a XOR of columns taken
			   earlier in this block, so the masks have to be
			   resolved back to what the columns held when the
			   block started. qmask[j] is pivot j written that
			   way; pmask[j] only ever names pivots before j,
			   so one forward pass does it. */

			for (j = 0; j < k; j++) {
				uint64 mm = pmask[j];
				uint64 q = (uint64)1 << j;

				while (mm) {
					q ^= qmask[qb_ctz(mm)];
					mm &= mm - 1;
				}
				qmask[j] = q;
			}

			/* and then every column, pivots included: a pivot
			   column's own mask resolves to qmask[j] without
			   its own bit, which is exactly right for a XOR
			   applied in place */

			for (j = 0; j < b; j++) {
				uint64 mm = cmask[j];
				uint64 f = 0;

				while (mm) {
					f ^= qmask[qb_ctz(mm)];
					mm &= mm - 1;
				}
				fmask[j] = f;
			}

			QB_BTICK(LP_QB_MASK)

			ngrp = (k + QB_GBITS - 1) / QB_GBITS;

			/* every XOR of each group of QB_GBITS pivot
			   columns, tabulated once. Entry v is entry
			   v-with-its-lowest-bit-cleared plus one more
			   column, so each costs a single pass. */

			for (g = 0; g < ngrp; g++) {
				uint64 *tp = tab_pi +
					(size_t)g * QB_GSIZE * piw;
				uint64 *tr = tab_R +
					(size_t)g * QB_GSIZE * Rw;
				uint32 v;
				size_t w;

				for (w = 0; w < piw; w++)
					tp[w] = 0;
				for (w = 0; w < Rw; w++)
					tr[w] = 0;

				for (v = 1; v < QB_GSIZE; v++) {
					uint32 low = v & (v - 1);
					uint32 s = g * QB_GBITS + qb_ctz(v);
					const uint64 *sp, *sr;

					if (s >= k) {
						memcpy(tp + (size_t)v * piw,
							tp + (size_t)low * piw,
							piw * sizeof(uint64));
						memcpy(tr + (size_t)v * Rw,
							tr + (size_t)low * Rw,
							Rw * sizeof(uint64));
						continue;
					}
					sp = pm_col(&pi, pivcols[s]) + pioff;
					sr = pm_col(&R, pivcols[s]) + Roff;
					for (w = 0; w < piw; w++) {
						tp[(size_t)v * piw + w] =
							tp[(size_t)low * piw
								+ w] ^ sp[w];
					}
					for (w = 0; w < Rw; w++) {
						tr[(size_t)v * Rw + w] =
							tr[(size_t)low * Rw
								+ w] ^ sr[w];
					}
				}
			}

			QB_BTICK(LP_QB_TAB)

			/* one pass over the columns, one XOR per group.
			   In place is safe: the tables are copies, and a
			   column reads nothing but itself. */

			for (j = 0; j < b; j++) {
				uint64 f = fmask[j];
				uint64 *cp, *cr;
				uint32 g2;

				if (f == 0)
					continue;
				cp = pm_col(&pi, j) + pioff;
				cr = pm_col(&R, j) + Roff;

				for (g2 = 0; g2 < ngrp; g2++) {
					uint32 v = (uint32)((f >>
						(g2 * QB_GBITS)) &
						(QB_GSIZE - 1));

					if (v == 0)
						continue;
					qb_xor(cp, tab_pi + ((size_t)g2 *
						QB_GSIZE + v) * piw, piw);
					qb_xor(cr, tab_R + ((size_t)g2 *
						QB_GSIZE + v) * Rw, Rw);
				}
			}
			QB_BTICK(LP_QB_APP)
		}
#undef QB_BTICK

		QB_TICK(LP_QB_ELIM)

		for (j = 0; j < b; j++) {
			if (is_pivot[j]) {
				pm_col_shift(&pi, j, plo, phi);
				pm_col_shift(&R, j, rlo, rhi);
				mydelta[j]++;
			}
		}

		QB_TICK(LP_QB_SHIFT)
	}

	/* every copy advanced by the same rule, so any of them is the
	   answer; the region's closing barrier publishes it */

	if (mytid == 0)
		memcpy(delta, mydelta, b * sizeof(uint32));

	free(tab_R);
	free(tab_pi);
	free(qmask);
	free(pmask);
	free(fmask);
	free(cmask);
	free(dcol);
	free(mydelta);
	free(pivcols);
	free(is_pivot);
	free(order);
	}
#undef QB_TICK

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
		for (r = pb0; r < pb1; r++) {
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

	/* every rank saw only its own rows, so the length of the answer
	   is the longest any of them found -- and they all have to
	   allocate the same pi_out for the combine below to line up */

	if (nrank > 1) {
		uint64 md = maxdelta;

		bmp_combine_max(&md);
		maxdelta = (uint32)md;
	}

	/* back to the coefficient-major layout the multiply wants. Each
	   coefficient is a separate destination block, so this shares
	   out with nothing to coordinate. */

	bmp_init(pi_out, b, b, maxdelta + 1);
#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for (tp = 0; tp <= (int32)maxdelta; tp++) {
		uint64 *pc = bmp_coeff(pi_out, (uint32)tp);
		uint32 rw, cw;

		for (rw = pb0; rw < pb1; rw++) {
			uint64 *row = pc + (size_t)rw * pi_out->rwords;

			for (cw = 0; cw < b; cw++) {
				if (pm_coeff(&pi, cw, rw, (uint32)tp))
					row[cw >> 6] |= (uint64)1 << (cw & 63);
			}
		}
	}

	/* each rank filled its own rows and left the rest zero, so the
	   XOR puts the whole basis back on every one of them */

	if (nrank > 1)
		bmp_combine(pi_out);

	free(dcol_shared);
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
#ifdef LINGEN_PROFILE
	double t0;
#endif

	if (T <= LINGEN_BASE_CASE) {
#ifdef LINGEN_PROFILE
		t0 = lingen_wtime();
#endif
		quadratic_basis(G, T, delta, pi_out);
#ifdef LINGEN_PROFILE
		lingen_prof_add(LP_BASE, lingen_wtime() - t0);
#endif
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
#ifdef LINGEN_PROFILE
		t0 = lingen_wtime();
#endif
		bmp_mul(&E, &Gfull, &pi1);
#ifdef LINGEN_PROFILE
		lingen_prof_add(LP_MUL_E, lingen_wtime() - t0);
#endif
	}

	bmp_view(&Gsub, &E, T1, MIN(T2, E.len - T1));
	recursive_basis(&Gsub, T2, delta, &pi2);

	bmp_init(pi_out, pi1.nrows, pi2.ncols, pi1.len + pi2.len - 1);
#ifdef LINGEN_PROFILE
	t0 = lingen_wtime();
#endif
	bmp_mul(pi_out, &pi1, &pi2);
#ifdef LINGEN_PROFILE
	lingen_prof_add(LP_MUL_PI, lingen_wtime() - t0);
#endif

	bmp_free(&E);
	bmp_free(&pi2);
	bmp_free(&pi1);
}

/*-----------------------------------------------------------------------*/
int32 bw_lingen(msieve_obj *obj, bw_params_t *params, uint32 max_ncols) {

	uint32 m = 0, n = 0;
	uint32 b, num_terms, T;
	uint32 t, i, j, r, c;
	v_t *a = NULL, *f = NULL;
	bmp_t G, pi;
	uint32 *delta = NULL, *order = NULL;
	uint32 degree = 0;
	int32 status = -1;
	double recursion_secs = 0;
	time_t start_time, phase_time;
#ifdef LINGEN_PROFILE
	double leaf_in_recursion = 0;
	uint64 ops_in_recursion = 0;
#endif
	char buf[BW_PATH_LEN];
	FILE *fp;
	bw_gen_header_t hdr;

	bmp_mul_fft_set_budget(obj);

	phase_time = time(NULL);
	a = read_sequences(obj, params, max_ncols, &num_terms, &m);
	if (a == NULL)
		return -1;

	/* after the read, not before: read_sequences takes m and n from
	   the sequence headers, which is the only place that knows them
	   for certain */

	n = params->n_mult * VBITS;
	b = m + n;
	T = num_terms;

	logprintf(obj, "commencing Wiedemann lingen, %u terms, m = %u, "
			"n = %u\n", num_terms, m, n);
	logprintf(obj, "lingen: read the sequences in %.1f sec\n",
			difftime(time(NULL), phase_time));
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

	phase_time = time(NULL);
	recursive_basis(&G, T, delta, &pi);
	recursion_secs = difftime(time(NULL), phase_time);
#ifdef LINGEN_PROFILE
	/* snapshot before the residual check, which is itself a large
	   product and would otherwise be counted as recursion work */
	leaf_in_recursion = lingen_prof_time(LP_SCHOOL);
	ops_in_recursion = lingen_prof_count(LP_OPS);
#endif
	logprintf(obj, "lingen: recursion %.1f sec, pi has %u coefficients\n",
			recursion_secs, pi.len);

	/* G pi must be zero below X^T; that is the whole invariant, and
	   the recursion has enough moving parts to be worth checking */

	phase_time = time(NULL);
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
	logprintf(obj, "lingen: residual check %.1f sec\n",
			difftime(time(NULL), phase_time));

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
	   holds and what the dependency file records, so only VBITS
	   columns are kept, of least degree. Each coefficient of the
	   generator is then n rows of VBITS bits -- one v_t per row of
	   y, across all the sequences.

	   Least degree is not the only condition. A column whose top
	   block is already nonzero at X^0 evaluates in mksol to
	   sum_k A^k y F_k, and that is exactly what the generator
	   annihilates, so the column yields the zero vector and one
	   fewer dependency. Nor can it be rescued by shifting: with
	   V(s) = sum_k A^k y F_{k+s} the recurrence is
	   V(s) = y F_s + A V(s+1), so a column with valuation v gives
	   A V(1) = 0 and every shift up to v lands in the nullspace
	   together, while shifting past v gives A V(v+1) = y F_v, which
	   is not in the nullspace at all. The only remedy is to pick a
	   different column, and there are b of them to choose from.

	   This is not a rare case. At m = n = 256 on a 4.8M matrix, 31
	   of the 64 columns of least degree started at X^0; the run
	   reported 33 dependencies and wrote a .dep in which two bits
	   were set. At m = n = 64 only one column did. */

	{
		uint32 nsel = 0;

		for (i = 0; i < b && nsel < VBITS; i++) {
			uint32 col = order[i];

			/* valuation of this column of the top n rows */

			for (t = 0; t < pi.len; t++) {
				uint64 *pc = bmp_coeff(&pi, t);

				for (r = 0; r < n; r++) {
					uint64 *row = pc +
							(size_t)r * pi.rwords;

					if ((row[col >> 6] >> (col & 63)) & 1)
						break;
				}
				if (r < n)
					break;
			}

			/* t == 0 is annihilated, t == pi.len is an empty
			   column; both give nothing */

			if (t > 0 && t < pi.len)
				order[nsel++] = col;
		}

		if (nsel < VBITS) {
			logprintf(obj, "error: only %u of %u generator "
					"columns are usable\n", nsel,
					(uint32)VBITS);
			goto cleanup;
		}
	}

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

#ifdef LINGEN_PROFILE
	{
		/* BASE, MUL_E and MUL_PI are wall time from the sequential
		   spine of the recursion, so they add up to it. SCHOOL is
		   summed over threads, so SCHOOL divided by the recursion
		   wall time is the number of threads that were really
		   working -- compare it against the number available. */

		double rec = lingen_prof_time(LP_BASE) +
				lingen_prof_time(LP_MUL_E) +
				lingen_prof_time(LP_MUL_PI);
		double leaf = leaf_in_recursion;
		int nthreads = 1;

#ifdef HAVE_OMP
		nthreads = omp_get_max_threads();
#endif
		logprintf(obj, "lingen profile: base case %.1f sec (%" PRIu64
				" calls)\n", lingen_prof_time(LP_BASE),
				lingen_prof_count(LP_BASE));
		{
			uint32 kk;

			logprintf(obj, "lingen profile: fft transforms %.1f, "
					"pointwise %.1f, inverse %.1f sec\n",
					fft_prof_trans, fft_prof_point,
					fft_prof_inv);
			for (kk = 0; kk <= 32; kk++) {
				if (fft_prof_calls_by_k[kk] == 0)
					continue;
				logprintf(obj, "lingen profile:   n=%-7u "
						"%8.1f sec over %" PRIu64
						" products\n",
						(uint32)1 << kk,
						fft_prof_point_by_k[kk],
						fft_prof_calls_by_k[kk]);
			}
		}
		if (bmp_words_calls) {
			logprintf(obj, "lingen profile: base case sent %.1f "
					"MB over %" PRIu64 " exchanges, one "
					"per step\n",
					(double)bmp_words_sent * 8 / 1048576.0,
					bmp_words_calls);
		}
		logprintf(obj, "lingen profile: base case symbolic %.1f, masks "
				"%.1f, tables %.1f, apply %.1f sec\n",
				lingen_prof_time(LP_QB_SYM),
				lingen_prof_time(LP_QB_MASK),
				lingen_prof_time(LP_QB_TAB),
				lingen_prof_time(LP_QB_APP));
		logprintf(obj, "lingen profile: base case build %.1f, sort "
				"%.1f, eliminate %.1f, shift %.1f sec\n",
				lingen_prof_time(LP_QB_BUILD),
				lingen_prof_time(LP_QB_SORT),
				lingen_prof_time(LP_QB_ELIM),
				lingen_prof_time(LP_QB_SHIFT));
		logprintf(obj, "lingen profile: residual products %.1f sec, "
				"composition products %.1f sec\n",
				lingen_prof_time(LP_MUL_E),
				lingen_prof_time(LP_MUL_PI));
		logprintf(obj, "lingen profile: leaf products in the "
				"recursion %.1f sec over all threads, %.1f "
				"sec more in the residual check, %" PRIu64
				" unbalanced splits\n", leaf,
				lingen_prof_time(LP_SCHOOL) - leaf,
				lingen_prof_count(LP_SPLIT));
		if (leaf > 0) {
			/* each word XOR reads two 8-byte words and writes
			   one, so the rate below is a lower bound on the
			   traffic the leaf asks the memory system for */

			double ops = (double)ops_in_recursion;

			logprintf(obj, "lingen profile: %.3e word XORs, "
					"%.2f G/sec, about %.1f GB/sec\n",
					ops, ops / leaf / 1e9,
					ops * 24 / leaf / 1e9);
		}
		/* whatever the recursion spent outside those three is
		   allocation: every level xcallocs its operands, and at
		   the top those are hundreds of megabytes to zero */

		logprintf(obj, "lingen profile: %.1f sec of the recursion "
				"was neither, i.e. allocation\n",
				recursion_secs - rec);
		if (rec > 0) {
			logprintf(obj, "lingen profile: %.2f threads busy on "
					"average of %d available (%.0f%% of "
					"the recursion is leaf work)\n",
					leaf / rec, nthreads,
					100.0 * leaf / (rec * nthreads));
		}
	}
#endif

	phase_time = time(NULL);
	if (check_generator(obj, a, num_terms, m, params->n_mult, degree,
			f, 64) != 0)
		goto cleanup;

	logprintf(obj, "generator verified against the sequence, %.1f sec\n",
			difftime(time(NULL), phase_time));

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
