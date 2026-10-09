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

/* Matrices of GF(2) polynomials, stored coefficient-major: coefficient
   k is a whole dense bit matrix. Multiplication is then a polynomial
   product whose coefficients happen to be matrices, so Karatsuba runs
   in the degree dimension and the only primitive needed underneath is
   a dense GF(2) matrix product.

   That is the reason for this layout. The obvious alternative is to
   keep a polynomial per matrix entry and multiply entries with a
   transform, but over GF(2) a transform has to move into GF(2^w),
   whose multiplication is carry-less and which neither a CPU without
   PCLMUL nor any GPU does natively. Karatsuba over matrix coefficients
   stays in GF(2) throughout and costs d^1.585 matrix products, which
   is enough to turn the quadratic generator into something usable. */

#include "wiedemann.h"

/* below this many coefficients, schoolbook beats the recursion */
#ifndef BMP_KARATSUBA_CUTOFF
#define BMP_KARATSUBA_CUTOFF 16
#endif

/* and below this many, spawning tasks for the sub-products costs more
   than running them in turn */
#ifndef BMP_TASK_CUTOFF
#define BMP_TASK_CUTOFF 128
#endif

/*-----------------------------------------------------------------------*/
#ifdef LINGEN_PROFILE

/* Per-thread slots rather than atomics: the leaf counter is hit once
   per schoolbook product, which is often, and false sharing on a
   single accumulator would show up as the very thing being measured. */

#define LP_MAX_THREADS 256

static double lp_time[LP_NUM][LP_MAX_THREADS];
static uint64 lp_count[LP_NUM][LP_MAX_THREADS];

double lingen_wtime(void) {

#ifdef HAVE_OMP
	return omp_get_wtime();
#else
	return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static uint32 lp_thread(void) {

#ifdef HAVE_OMP
	return (uint32)omp_get_thread_num() % LP_MAX_THREADS;
#else
	return 0;
#endif
}

void lingen_prof_add(uint32 slot, double secs) {

	uint32 t = lp_thread();

	lp_time[slot][t] += secs;
	lp_count[slot][t]++;
}

void lingen_prof_bump(uint32 slot, uint64 amount) {

	lp_count[slot][lp_thread()] += amount;
}

double lingen_prof_time(uint32 slot) {

	double sum = 0;
	uint32 i;

	for (i = 0; i < LP_MAX_THREADS; i++)
		sum += lp_time[slot][i];
	return sum;
}

uint64 lingen_prof_count(uint32 slot) {

	uint64 sum = 0;
	uint32 i;

	for (i = 0; i < LP_MAX_THREADS; i++)
		sum += lp_count[slot][i];
	return sum;
}
#endif

/* index of the lowest set bit; the argument is never zero */

static INLINE uint32 bw_ctz64(uint64 x) {

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

/*-----------------------------------------------------------------------*/
void bmp_init(bmp_t *p, uint32 nrows, uint32 ncols, uint32 len) {

	p->nrows = nrows;
	p->ncols = ncols;
	p->rwords = (ncols + 63) / 64;
	p->len = len;
	p->data = (uint64 *)xcalloc((size_t)len * nrows * p->rwords,
					sizeof(uint64));
}

void bmp_free(bmp_t *p) {

	free(p->data);
	p->data = NULL;
}

/* coefficient k, as a contiguous nrows x ncols bit matrix */

uint64 *bmp_coeff(bmp_t *p, uint32 k) {

	return p->data + (size_t)k * p->nrows * p->rwords;
}

static size_t bmp_coeff_words(const bmp_t *p) {

	return (size_t)p->nrows * p->rwords;
}

/*-----------------------------------------------------------------------*/
static void bm_mul_acc(uint64 *c, const uint64 *a, const uint64 *b,
			uint32 nrows, uint32 kdim,
			uint32 rwords_a, uint32 rwords_c) {

	/* c (nrows x ncols) ^= a (nrows x kdim) * b (kdim x ncols).
	   Row r of the product is the sum of the rows of b picked out by
	   the set bits of row r of a, which is the whole of GF(2) matrix
	   multiplication. */

	uint32 r, w;
	uint32 awords = (kdim + 63) / 64;

	for (r = 0; r < nrows; r++) {
		const uint64 *arow = a + (size_t)r * rwords_a;
		uint64 *crow = c + (size_t)r * rwords_c;
		uint32 aw;

		/* walk the set bits rather than testing all kdim of them:
		   a whole zero word is skipped at once, and inside a word
		   the loop runs once per set bit instead of 64 times with
		   an unpredictable branch. This primitive is most of the
		   time the recursion spends */

		for (aw = 0; aw < awords; aw++) {
			uint64 bits = arow[aw];
			uint32 base = 64 * aw;

			while (bits) {
				uint32 s = base + bw_ctz64(bits);
				const uint64 *brow = b +
						(size_t)s * rwords_c;

				bits &= bits - 1;
				for (w = 0; w < rwords_c; w++)
					crow[w] ^= brow[w];
			}
		}
	}
}

/*-----------------------------------------------------------------------*/
void bmp_mul_school(bmp_t *c, const bmp_t *a, const bmp_t *b) {

	/* the reference product, and the base of the recursion */

	uint32 i, j;
#ifdef LINGEN_PROFILE
	double t0 = lingen_wtime();
#endif

	for (i = 0; i < a->len; i++) {
		const uint64 *ai = a->data +
				(size_t)i * a->nrows * a->rwords;

		for (j = 0; j < b->len; j++) {
			const uint64 *bj = b->data +
					(size_t)j * b->nrows * b->rwords;
			uint64 *cij = c->data +
					(size_t)(i + j) * c->nrows * c->rwords;

			bm_mul_acc(cij, ai, bj, a->nrows, a->ncols,
					a->rwords, c->rwords);
		}
	}
#ifdef LINGEN_PROFILE
	lingen_prof_add(LP_SCHOOL, lingen_wtime() - t0);

	/* How many word XORs that asked for. Counted from the shape
	   rather than inside the bit walk, which would perturb the loop
	   being measured: the operands are dense random over GF(2), so
	   half the kdim bits are set and the estimate is exact to within
	   the sampling. Against the wall time it says whether the leaf
	   is running at memory speed or nowhere near it. */

	lingen_prof_bump(LP_OPS, (uint64)a->len * b->len * a->nrows *
				(a->ncols / 2) * c->rwords);
#endif
}

/*-----------------------------------------------------------------------*/
static void bmp_slice(bmp_t *out, const bmp_t *in, uint32 first,
			uint32 len) {

	/* a view of coefficients [first, first+len) -- no copy, so it
	   must not be freed */

	out->nrows = in->nrows;
	out->ncols = in->ncols;
	out->rwords = in->rwords;
	out->len = len;
	out->data = in->data + (size_t)first * in->nrows * in->rwords;
}

static void bmp_add_into(bmp_t *dst, const bmp_t *src, uint32 offset) {

	size_t words = bmp_coeff_words(src);
	size_t i, num = (size_t)src->len * words;
	uint64 *d = dst->data + (size_t)offset * words;

	for (i = 0; i < num; i++)
		d[i] ^= src->data[i];
}

/*-----------------------------------------------------------------------*/
static void bmp_mul_kara(bmp_t *c, const bmp_t *a, const bmp_t *b) {

	/* Karatsuba in the degree dimension. Over GF(2) addition and
	   subtraction are the same, so the middle term is just
	   (a0+a1)(b0+b1) with the two outer products folded back in.

	   Operands of very different lengths are split into balanced
	   pieces first: the residual product in the recursion is a long
	   series against a much shorter basis, and letting that fall all
	   the way back to schoolbook would undo the point of the
	   exercise. */

	uint32 h, i;
	bmp_t a0, a1, b0, b1;
	bmp_t as, bs, z0, z2, z1;
	size_t awords, bwords;

	if (a->len <= BMP_KARATSUBA_CUTOFF ||
	    b->len <= BMP_KARATSUBA_CUTOFF) {
		bmp_mul_school(c, a, b);
		return;
	}

	/* chop the longer operand into pieces the size of the shorter */

	if (a->len > 2 * b->len || b->len > 2 * a->len) {
		const bmp_t *lng = (a->len > b->len) ? a : b;
		const bmp_t *shrt = (a->len > b->len) ? b : a;
		uint32 step = shrt->len;
		uint32 off;

#ifdef LINGEN_PROFILE
		lingen_prof_bump(LP_SPLIT, 1);
#endif
		for (off = 0; off < lng->len; off += step) {
			uint32 piece = MIN(step, lng->len - off);
			bmp_t sl, prod;

			bmp_slice(&sl, lng, off, piece);
			bmp_init(&prod, c->nrows, c->ncols,
					piece + shrt->len - 1);
			if (lng == a)
				bmp_mul_kara(&prod, &sl, shrt);
			else
				bmp_mul_kara(&prod, shrt, &sl);
			bmp_add_into(c, &prod, off);
			bmp_free(&prod);
		}
		return;
	}

	h = (MAX(a->len, b->len) + 1) / 2;
	if (h >= a->len || h >= b->len) {
		bmp_mul_school(c, a, b);
		return;
	}

	bmp_slice(&a0, a, 0, h);
	bmp_slice(&a1, a, h, a->len - h);
	bmp_slice(&b0, b, 0, h);
	bmp_slice(&b1, b, h, b->len - h);

	awords = bmp_coeff_words(a);
	bwords = bmp_coeff_words(b);

	/* as = a0 + a1 and bs = b0 + b1, each held at the full half
	   length so the halves may be short */

	bmp_init(&as, a->nrows, a->ncols, h);
	bmp_init(&bs, b->nrows, b->ncols, h);
	for (i = 0; i < a0.len * awords; i++)
		as.data[i] ^= a0.data[i];
	for (i = 0; i < a1.len * awords; i++)
		as.data[i] ^= a1.data[i];
	for (i = 0; i < b0.len * bwords; i++)
		bs.data[i] ^= b0.data[i];
	for (i = 0; i < b1.len * bwords; i++)
		bs.data[i] ^= b1.data[i];

	bmp_init(&z0, c->nrows, c->ncols, a0.len + b0.len - 1);
	bmp_init(&z2, c->nrows, c->ncols, a1.len + b1.len - 1);
	bmp_init(&z1, c->nrows, c->ncols, as.len + bs.len - 1);

	/* The three sub-products share no state, and each one splits
	   again, so tasks here give a deep and well balanced tree. Only
	   worth the overhead while the pieces are still large; below
	   that the sequential path is faster than scheduling it. */

#ifdef HAVE_OMP
	if (a->len >= BMP_TASK_CUTOFF && b->len >= BMP_TASK_CUTOFF) {
#pragma omp task shared(z0, a0, b0)
		bmp_mul_kara(&z0, &a0, &b0);
#pragma omp task shared(z2, a1, b1)
		bmp_mul_kara(&z2, &a1, &b1);
#pragma omp task shared(z1, as, bs)
		bmp_mul_kara(&z1, &as, &bs);
#pragma omp taskwait
	}
	else
#endif
	{
		bmp_mul_kara(&z0, &a0, &b0);
		bmp_mul_kara(&z2, &a1, &b1);
		bmp_mul_kara(&z1, &as, &bs);
	}

	/* z1 <- z1 - z0 - z2, which in characteristic 2 is an XOR */

	bmp_add_into(&z1, &z0, 0);
	bmp_add_into(&z1, &z2, 0);

	bmp_add_into(c, &z0, 0);
	bmp_add_into(c, &z1, h);
	bmp_add_into(c, &z2, 2 * h);

	bmp_free(&z1);
	bmp_free(&z2);
	bmp_free(&z0);
	bmp_free(&bs);
	bmp_free(&as);
}

/*-----------------------------------------------------------------------*/
void bmp_mul(bmp_t *c, const bmp_t *a, const bmp_t *b) {

	/* c = a * b. c must already be the right length, a->len +
	   b->len - 1, and zeroed.

	   One team is created here and one thread starts the recursion;
	   the sub-products below spawn into it. The team is built per
	   call rather than per level, so the nesting costs nothing. */

	if (a->len == 0 || b->len == 0)
		return;

	/* The transform wins once the operands are long enough to pay
	   for it, which is most of the time at the sizes that matter;
	   Karatsuba stays for the short products the recursion ends in,
	   and for anything whose transforms would not fit in memory. */

	if (bmp_mul_fft_ok(c, a, b)) {
		bmp_mul_fft(c, a, b);
		return;
	}

#ifdef HAVE_OMP
#pragma omp parallel
#pragma omp single
#endif
	bmp_mul_kara(c, a, b);
}
