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

/* Cantor's additive FFT, for the matrix polynomial product lingen
   spends its time in.

   Karatsuba costs d^1.585 matrix products; this costs one transform
   per matrix ENTRY plus one dense GF(2^64) matrix product per
   evaluation point. The transform is b^2 work and the pointwise stage
   is b^3, so the wider the matrix the better this looks -- which is
   the opposite of Karatsuba, whose b^3 is paid d^1.585 times.

   Why additive. In characteristic 2 the multiplicative group of
   GF(2^64) has odd order, so there is no root of unity of order 2^k
   and no ordinary FFT. Cantor's transform instead evaluates on a
   GF(2)-linear subspace, using the basis

        beta_0 = 1,    beta_i^2 + beta_i = beta_{i-1}

   whose point is self-similarity: psi(X) = X^2 + X sends beta_i to
   beta_{i-1}, so psi maps the span of the first k basis elements onto
   the span of the first k-1, two to one. Taylor expanding at psi,

        f(X) = f0(psi(X)) + X * f1(psi(X))

   halves the problem, and since psi(u) = psi(u+1) each pair of points
   shares one sub-evaluation:

        f(u)     = f0(v) + u * f1(v)
        f(u + 1) = f(u) + f1(v)

   Index the points by the bit pattern of their basis coefficients with
   beta_0 least significant and v's index is u >> 1, so the butterfly
   is a plain stride-2 pass with one field multiply.

   Why it is exact. A GF(2)[x] polynomial is cut into 32-bit chunks,
   one per GF(2^64) coefficient. Two chunks have degree at most 31, so
   their product has degree at most 62 and never reaches the degree-64
   field polynomial: the field multiply IS the polynomial multiply, and
   sums over GF(2) cannot raise the degree however long the convolution
   runs. Nothing is approximated and nothing can overflow. */

#include "wiedemann.h"

#if defined(__PCLMUL__)
#include <wmmintrin.h>
#endif

/* GF(2)[x] / (x^64 + x^4 + x^3 + x + 1) */
#define GF64_POLY ((uint64)0x1b)

#define CHUNK_BITS 32
#define CANTOR_MAX 32

/* Below this many coefficients the transform cannot win: it rounds the
   length up to a power of two and pays b^2 transforms to save on a
   product that Karatsuba already does cheaply. */
#ifndef LINGEN_FFT_MIN_LEN
#define LINGEN_FFT_MIN_LEN 128
#endif

/* The transforms are held in full, three of them, at 8 bytes per
   evaluation point per matrix entry. Past this the fallback is
   Karatsuba, which streams instead. */
#ifndef LINGEN_FFT_MAX_MB
#define LINGEN_FFT_MAX_MB 24576
#endif

/*-----------------------------------------------------------------------*/
#if !defined(__PCLMUL__)

/* the shift-and-XOR carry-less multiply, for anything without the
   instruction. Only the fallback path needs it. */

static void clmul64(uint64 a, uint64 b, uint64 *lo, uint64 *hi) {

	uint64 l = 0, h = 0;
	uint32 i;

	for (i = 0; i < 64; i++) {
		uint64 m = (uint64)0 - ((b >> i) & 1);

		l ^= (a << i) & m;
		h ^= (i == 0 ? (uint64)0 : (a >> (64 - i))) & m;
	}
	*lo = l;
	*hi = h;
}
#endif

static INLINE uint64 gf64_mul(uint64 a, uint64 b) {

	/* x^64 == GF64_POLY, and GF64_POLY has degree 4, so folding the
	   high half down twice lands entirely inside 64 bits: the second
	   fold multiplies something of degree at most 3 by it.

	   Three carry-less multiplies and two XORs, and on the SIMD path
	   none of it touches memory. Going through a uint64 pair instead
	   costs a store and two loads per multiply, which at one multiply
	   per inner-loop step was enough to wipe out the transform's
	   whole advantage over Karatsuba. */

#if defined(__PCLMUL__)
	__m128i m = _mm_cvtsi64_si128((long long)GF64_POLY);
	__m128i p = _mm_clmulepi64_si128(_mm_cvtsi64_si128((long long)a),
					_mm_cvtsi64_si128((long long)b), 0x00);
	__m128i t = _mm_clmulepi64_si128(p, m, 0x01);
	__m128i u = _mm_clmulepi64_si128(t, m, 0x01);

	return (uint64)_mm_cvtsi128_si64(
			_mm_xor_si128(_mm_xor_si128(p, t), u));
#else
	uint64 lo, hi, t0, t1;

	clmul64(a, b, &lo, &hi);
	clmul64(hi, GF64_POLY, &t0, &t1);
	lo ^= t0;
	clmul64(t1, GF64_POLY, &t0, &t1);
	lo ^= t0;
	return lo;
#endif
}

static uint64 gf64_sqr(uint64 a) {

	return gf64_mul(a, a);
}

static uint64 gf64_trace(uint64 a) {

	uint64 t = a;
	uint32 i;

	for (i = 1; i < 64; i++) {
		a = gf64_sqr(a);
		t ^= a;
	}
	return t;
}

/*-----------------------------------------------------------------------*/
/* Solve y^2 + y = c. The map is GF(2)-linear with kernel {0,1}, so
   this is a 64x64 system over GF(2) rather than anything to do with
   the field, and carrying the identity alongside hands back the
   preimage directly. Returns 0 when c is in the image. */

static int32 gf64_solve_quadratic(uint64 c, uint64 *out) {

	uint64 col[64], id[64];
	uint64 rhs = c, acc = 0;
	int32 piv_row[64];
	int32 i, j, rank = 0;

	for (i = 0; i < 64; i++) {
		uint64 e = (uint64)1 << i;

		col[i] = gf64_sqr(e) ^ e;
		id[i] = e;
		piv_row[i] = -1;
	}

	for (i = 0; i < 64; i++) {
		uint64 v = col[i], w = id[i];

		for (j = 63; j >= 0; j--) {
			if (!((v >> j) & 1))
				continue;
			if (piv_row[j] < 0) {
				col[rank] = v;
				id[rank] = w;
				piv_row[j] = rank;
				rank++;
				v = 0;
				break;
			}
			v ^= col[piv_row[j]];
			w ^= id[piv_row[j]];
		}
	}

	for (j = 63; j >= 0; j--) {
		if (!((rhs >> j) & 1))
			continue;
		if (piv_row[j] < 0)
			return -1;
		rhs ^= col[piv_row[j]];
		acc ^= id[piv_row[j]];
	}
	if (rhs != 0)
		return -1;

	*out = acc;
	return 0;
}

/*-----------------------------------------------------------------------*/
static uint64 cantor_beta[CANTOR_MAX];
static uint32 cantor_count;
static uint64 *cantor_point;
static uint32 cantor_point_k;

/* The basis alone, with no point table: cheap, and needed by the test
   below as well as by the transform. Doing it there too matters --
   leaving it to the transform means the test runs with a basis size of
   zero and silently never says yes. */

static void cantor_basis_init(void) {

	uint32 i;

	if (cantor_count != 0)
		return;

	cantor_beta[0] = 1;
	cantor_count = 1;
	for (i = 1; i < CANTOR_MAX; i++) {
		uint64 b;

		/* a root exists only where the trace vanishes */

		if (gf64_trace(cantor_beta[i - 1]) != 0)
			break;
		if (gf64_solve_quadratic(cantor_beta[i - 1], &b) != 0)
			break;
		cantor_beta[i] = b;
		cantor_count = i + 1;
	}
}

static void cantor_init(uint32 kmax) {

	uint32 i, j, n;

	cantor_basis_init();

	if (cantor_point != NULL && cantor_point_k >= kmax)
		return;

	free(cantor_point);
	n = (uint32)1 << kmax;
	cantor_point = (uint64 *)xmalloc((size_t)n * sizeof(uint64));
	cantor_point_k = kmax;
	for (i = 0; i < n; i++) {
		uint64 v = 0;

		for (j = 0; j < kmax; j++) {
			if ((i >> j) & 1)
				v ^= cantor_beta[j];
		}
		cantor_point[i] = v;
	}
}

/*-----------------------------------------------------------------------*/
/* Taylor expansion at X^2+X, in place: a[2i], a[2i+1] become the pair
   (a_i, b_i) of f = sum_i (a_i + b_i X)(X^2+X)^i. One level is linear
   because (X^2+X)^h = X^2h + X^h when h is a power of two. */

static void taylor_fwd(uint64 *a, uint32 n) {

	uint32 h, i;

	if (n <= 2)
		return;

	h = n >> 2;
	for (i = 0; i < h; i++) {
		uint64 f1 = a[h + i], f2 = a[2 * h + i], f3 = a[3 * h + i];

		a[h + i] = f1 ^ f2 ^ f3;
		a[2 * h + i] = f2 ^ f3;
		a[3 * h + i] = f3;
	}
	taylor_fwd(a, n >> 1);
	taylor_fwd(a + (n >> 1), n >> 1);
}

static void taylor_inv(uint64 *a, uint32 n) {

	uint32 h, i;

	if (n <= 2)
		return;

	taylor_inv(a, n >> 1);
	taylor_inv(a + (n >> 1), n >> 1);

	h = n >> 2;
	for (i = 0; i < h; i++) {
		uint64 a1 = a[h + i], b0 = a[2 * h + i], b1 = a[3 * h + i];

		a[h + i] = a1 ^ b0;
		a[2 * h + i] = b0 ^ b1;
		a[3 * h + i] = b1;
	}
}

/*-----------------------------------------------------------------------*/
static void fft_fwd(uint64 *a, uint32 k, uint64 *scratch) {

	uint32 n = (uint32)1 << k, half = n >> 1, j;
	uint64 *f0 = scratch, *f1 = scratch + half;

	if (k == 0)
		return;

	taylor_fwd(a, n);

	for (j = 0; j < half; j++) {
		f0[j] = a[2 * j];
		f1[j] = a[2 * j + 1];
	}

	fft_fwd(f0, k - 1, a);
	fft_fwd(f1, k - 1, a + half);

	for (j = 0; j < half; j++) {
		uint64 t = f0[j] ^ gf64_mul(cantor_point[2 * j], f1[j]);

		a[2 * j] = t;
		a[2 * j + 1] = t ^ f1[j];
	}
}

static void fft_inv(uint64 *a, uint32 k, uint64 *scratch) {

	uint32 n = (uint32)1 << k, half = n >> 1, j;
	uint64 *f0 = scratch, *f1 = scratch + half;

	if (k == 0)
		return;

	for (j = 0; j < half; j++) {
		uint64 g1 = a[2 * j] ^ a[2 * j + 1];

		f1[j] = g1;
		f0[j] = a[2 * j] ^ gf64_mul(cantor_point[2 * j], g1);
	}

	fft_inv(f0, k - 1, a);
	fft_inv(f1, k - 1, a + half);

	for (j = 0; j < half; j++) {
		a[2 * j] = f0[j];
		a[2 * j + 1] = f1[j];
	}

	taylor_inv(a, n);
}

/*-----------------------------------------------------------------------*/
static void transpose64(uint64 *a) {

	/* b[j] bit i = a[i] bit j. The word-at-a-time version of this is
	   a known shuffle and about ten times quicker, but it is easy to
	   get subtly wrong; this is not where the time goes. */

	uint64 b[64];
	uint32 i, j;

	for (j = 0; j < 64; j++) {
		uint64 v = 0;

		for (i = 0; i < 64; i++) {
			if ((a[i] >> j) & 1)
				v |= (uint64)1 << i;
		}
		b[j] = v;
	}
	memcpy(a, b, sizeof(b));
}

/* bmp_t is coefficient-major, so one matrix entry's polynomial is a
   bit-strided walk. Lift 64 coefficients of 64 entries at once and
   transpose instead. */

static void gather_entries(const bmp_t *p, uint32 r, uint32 sbase,
				uint64 **out, uint32 outwords) {

	uint32 kb, t;
	uint64 blk[64];

	for (kb = 0; kb < p->len; kb += 64) {
		uint32 nk = MIN(p->len - kb, 64);

		for (t = 0; t < 64; t++) {
			blk[t] = (t < nk) ?
				p->data[(size_t)(kb + t) * p->nrows *
						p->rwords +
					(size_t)r * p->rwords + (sbase >> 6)]
				: 0;
		}
		transpose64(blk);
		if ((kb >> 6) < outwords) {
			for (t = 0; t < 64; t++)
				out[t][kb >> 6] = blk[t];
		}
	}
}

static void scatter_entries(bmp_t *p, uint32 r, uint32 sbase,
				uint64 * const *in, uint32 inwords) {

	uint32 kb, t;
	uint64 blk[64];

	for (kb = 0; kb < p->len; kb += 64) {
		uint32 nk = MIN(p->len - kb, 64);

		for (t = 0; t < 64; t++)
			blk[t] = ((kb >> 6) < inwords) ? in[t][kb >> 6] : 0;
		transpose64(blk);
		for (t = 0; t < nk; t++) {
			p->data[(size_t)(kb + t) * p->nrows * p->rwords +
				(size_t)r * p->rwords + (sbase >> 6)] ^= blk[t];
		}
	}
}

/*-----------------------------------------------------------------------*/
static void pack_poly(uint64 *dst, uint32 nchunk, const uint64 *src,
			uint32 nbits) {

	uint32 i;

	for (i = 0; i < nchunk; i++) {
		uint32 bit = i * CHUNK_BITS;
		uint32 w = bit >> 6, off = bit & 63;
		uint64 v = 0;

		if (bit < nbits) {
			v = src[w] >> off;
			v &= ((uint64)1 << CHUNK_BITS) - 1;
		}
		dst[i] = v;
	}
}

static void unpack_poly(uint64 *dst, uint32 nwords, const uint64 *src,
			uint32 nchunk) {

	uint32 i;

	memset(dst, 0, (size_t)nwords * sizeof(uint64));
	for (i = 0; i < nchunk; i++) {
		uint32 bit = i * CHUNK_BITS;
		uint32 w = bit >> 6, off = bit & 63;

		if (w >= nwords)
			break;
		dst[w] ^= src[i] << off;
		if (off && w + 1 < nwords)
			dst[w + 1] ^= src[i] >> (64 - off);
	}
}

/*-----------------------------------------------------------------------*/
static void transform_all(const bmp_t *p, uint64 *ft, uint32 k, uint32 n) {

	/* One transform per matrix entry, and they are independent, so
	   this threads over rows. It has to: Karatsuba spreads itself
	   over the cores through its task tree, and a serial transform
	   stage hands that advantage straight back. */

	uint32 nch = (p->len + CHUNK_BITS - 1) / CHUNK_BITS;
	uint32 pw = (p->len + 63) / 64;
	int32 r;

#ifdef HAVE_OMP
#pragma omp parallel
#endif
	{
		uint32 sb, t, i;
		uint64 *buf = (uint64 *)xmalloc((size_t)n * sizeof(uint64));
		uint64 *sc = (uint64 *)xmalloc((size_t)n * sizeof(uint64));
		uint64 **ent = (uint64 **)xmalloc(64 * sizeof(uint64 *));

		for (t = 0; t < 64; t++) {
			ent[t] = (uint64 *)xcalloc((size_t)pw + 2,
						sizeof(uint64));
		}

#ifdef HAVE_OMP
#pragma omp for schedule(static)
#endif
		for (r = 0; r < (int32)p->nrows; r++) {
			for (sb = 0; sb < p->ncols; sb += 64) {
				gather_entries(p, (uint32)r, sb, ent, pw);
				for (t = 0; t < 64 && sb + t < p->ncols; t++) {
					memset(buf, 0,
						(size_t)n * sizeof(uint64));
					pack_poly(buf, nch, ent[t], p->len);
					fft_fwd(buf, k, sc);
					for (i = 0; i < n; i++) {
						ft[(size_t)i * p->nrows *
							p->ncols +
							(size_t)r * p->ncols +
							(sb + t)] = buf[i];
					}
				}
			}
		}

		for (t = 0; t < 64; t++)
			free(ent[t]);
		free(ent);
		free(sc);
		free(buf);
	}
}

/*-----------------------------------------------------------------------*/
uint32 bmp_mul_fft_ok(const bmp_t *c, const bmp_t *a, const bmp_t *b) {

	uint32 nca, ncb, ncc, k = 0;
	double mb;

	if (a->len < LINGEN_FFT_MIN_LEN || b->len < LINGEN_FFT_MIN_LEN)
		return 0;

	cantor_basis_init();

	nca = (a->len + CHUNK_BITS - 1) / CHUNK_BITS;
	ncb = (b->len + CHUNK_BITS - 1) / CHUNK_BITS;
	ncc = nca + ncb - 1;
	while (((uint32)1 << k) < ncc)
		k++;
	if (k >= cantor_count)
		return 0;

	mb = (double)((uint32)1 << k) * sizeof(uint64) / 1048576.0 *
		((double)a->nrows * a->ncols + (double)b->nrows * b->ncols +
		 (double)c->nrows * c->ncols);
	if (mb > (double)LINGEN_FFT_MAX_MB)
		return 0;

	return 1;
}

void bmp_mul_fft(bmp_t *c, const bmp_t *a, const bmp_t *b) {

	uint32 nca = (a->len + CHUNK_BITS - 1) / CHUNK_BITS;
	uint32 ncb = (b->len + CHUNK_BITS - 1) / CHUNK_BITS;
	uint32 ncc = nca + ncb - 1;
	uint32 cw = (c->len + 63) / 64;
	uint32 k = 0, n, i, r, s, t;
	uint64 *fa, *fb, *fc;

	while (((uint32)1 << k) < ncc)
		k++;
	n = (uint32)1 << k;

	cantor_init(k);

	fa = (uint64 *)xmalloc((size_t)n * a->nrows * a->ncols *
				sizeof(uint64));
	fb = (uint64 *)xmalloc((size_t)n * b->nrows * b->ncols *
				sizeof(uint64));
	fc = (uint64 *)xcalloc((size_t)n * c->nrows * c->ncols,
				sizeof(uint64));

	transform_all(a, fa, k, n);
	transform_all(b, fb, k, n);

	/* One dense GF(2^64) matrix product per evaluation point. This is
	   the b^3 term, and the only part that grows with the width. */

#ifdef HAVE_OMP
#pragma omp parallel for schedule(static) private(r, s, t)
#endif
	for (i = 0; i < n; i++) {
		const uint64 *A = fa + (size_t)i * a->nrows * a->ncols;
		const uint64 *B = fb + (size_t)i * b->nrows * b->ncols;
		uint64 *C = fc + (size_t)i * c->nrows * c->ncols;

		for (r = 0; r < a->nrows; r++) {
			for (t = 0; t < a->ncols; t++) {
				uint64 av = A[(size_t)r * a->ncols + t];

				if (av == 0)
					continue;
				for (s = 0; s < b->ncols; s++) {
					C[(size_t)r * c->ncols + s] ^=
						gf64_mul(av, B[(size_t)t *
							b->ncols + s]);
				}
			}
		}
	}

	/* and back, threaded the same way; rows of c are disjoint so the
	   scatter needs no serialising */

#ifdef HAVE_OMP
#pragma omp parallel
#endif
	{
		uint32 rr, ss, tt, ii;
		uint64 *buf = (uint64 *)xmalloc((size_t)n * sizeof(uint64));
		uint64 *sc2 = (uint64 *)xmalloc((size_t)n * sizeof(uint64));
		uint64 **ent2 = (uint64 **)xmalloc(64 * sizeof(uint64 *));
		int32 ri;

		for (tt = 0; tt < 64; tt++) {
			ent2[tt] = (uint64 *)xcalloc((size_t)cw + 2,
						sizeof(uint64));
		}

#ifdef HAVE_OMP
#pragma omp for schedule(static)
#endif
		for (ri = 0; ri < (int32)c->nrows; ri++) {
			rr = (uint32)ri;
			for (ss = 0; ss < c->ncols; ss += 64) {
				for (tt = 0; tt < 64; tt++) {
					if (ss + tt >= c->ncols) {
						memset(ent2[tt], 0,
							((size_t)cw + 2) *
							sizeof(uint64));
						continue;
					}
					for (ii = 0; ii < n; ii++) {
						buf[ii] = fc[(size_t)ii *
							c->nrows * c->ncols +
							(size_t)rr * c->ncols +
							ss + tt];
					}
					fft_inv(buf, k, sc2);
					unpack_poly(ent2[tt], cw, buf, ncc);
				}
				scatter_entries(c, rr, ss, ent2, cw);
			}
		}

		for (tt = 0; tt < 64; tt++)
			free(ent2[tt]);
		free(ent2);
		free(sc2);
		free(buf);
	}

	free(fc);
	free(fb);
	free(fa);
}
