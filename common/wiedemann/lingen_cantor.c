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

/* Below this many coefficients schoolbook takes the product instead.
   The cutoff is deliberately low: an operand of at most CHUNK_BITS
   coefficients is a single chunk, so k comes out 0 and there is no
   transform left to pay for -- the whole product becomes one
   carry-less multiply per matrix entry, against len^2 word XORs for
   schoolbook. The first guess of 128 was wrong for exactly that
   reason, and it hurt twice over, because Karatsuba in [16,128) is
   where the recursion spent its time and none of it ran in parallel.

   Measured on the 4.8M matrix at b=128: the recursion went 20.0 ->
   13.0 sec for a byte-identical generator, with 32, 16, 8 and 4 all
   the same within the timer. It sits at BMP_KARATSUBA_CUTOFF so the
   division of labour is simply schoolbook below, transform above, and
   Karatsuba only where the panel budget turns the transform down. */
#ifndef LINGEN_FFT_MIN_LEN
#define LINGEN_FFT_MIN_LEN 16
#endif

/* How much the transforms may hold. Taken from the machine rather than
   guessed: msieve already measures memory for the filtering strategy,
   so the same get_ram_size() answers this, halved because lingen is
   carrying G, pi and the recursion's own operands alongside. bw_mem_mb=
   overrides it, the way filter_mem_mb= does for filtering. */

#ifndef LINGEN_FFT_MEM_FRACTION
#define LINGEN_FFT_MEM_FRACTION 2
#endif

static double fft_budget_mb;

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

	/* a[j] bit i becomes a[i] bit j, by recursive block swap: at each
	   step exchange the off-diagonal halves of every 2j x 2j block,
	   which is six passes instead of the 4096 bit tests the obvious
	   loop needs. This runs once per 64 coefficients of 64 entries
	   on the way in and on the way out, and again for every panel
	   the product is cut into, so it is worth the care.

	   Which way round the shift goes is the whole of it, and getting
	   it backwards transposes something else entirely; the version
	   below was checked against the obvious loop rather than
	   remembered. */

	uint64 m = 0x00000000ffffffffULL;
	uint32 j, k;

	for (j = 32; j != 0; j >>= 1, m ^= m << j) {
		for (k = 0; k < 64; k = ((k | j) + 1) & ~j) {
			uint64 t = ((a[k] >> j) ^ a[k | j]) & m;

			a[k | j] ^= t;
			a[k] ^= t << j;
		}
	}
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
/* Transform the entries of p in rows [r0, r0+nr) and columns
   [c0, c0+nc), into ft[point][r - r0][s - c0]. c0 is a multiple of 64
   because the gather lifts whole 64-column groups.

   One transform per matrix entry, and they are independent, so this
   threads over rows. It has to: Karatsuba spreads itself over the
   cores through its task tree, and a serial transform stage hands
   that advantage straight back. */

static void transform_panel(const bmp_t *p, uint64 *ft, uint32 k, uint32 n,
				uint32 r0, uint32 nr, uint32 c0, uint32 nc) {

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
		for (r = (int32)r0; r < (int32)(r0 + nr); r++) {
			for (sb = c0; sb < c0 + nc; sb += 64) {
				gather_entries(p, (uint32)r, sb, ent, pw);
				for (t = 0; t < 64 && sb + t < c0 + nc; t++) {
					memset(buf, 0,
						(size_t)n * sizeof(uint64));
					pack_poly(buf, nch, ent[t], p->len);
					fft_fwd(buf, k, sc);
					for (i = 0; i < n; i++) {
						ft[(size_t)i * nr * nc +
							(size_t)((uint32)r -
								r0) * nc +
							(sb + t - c0)] = buf[i];
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

/* The reverse, for one panel of c: inverse transform each entry and
   XOR it in. Panels are disjoint in (row, column) and rows inside a
   panel are disjoint, so nothing needs serialising. */

static void inverse_panel(bmp_t *c, const uint64 *fc, uint32 k, uint32 n,
				uint32 r0, uint32 nr, uint32 c0, uint32 nc,
				uint32 ncc) {

	uint32 cw = (c->len + 63) / 64;
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
			ent[t] = (uint64 *)xcalloc((size_t)cw + 2,
						sizeof(uint64));
		}

#ifdef HAVE_OMP
#pragma omp for schedule(static)
#endif
		for (r = (int32)r0; r < (int32)(r0 + nr); r++) {
			for (sb = c0; sb < c0 + nc; sb += 64) {
				for (t = 0; t < 64; t++) {
					if (sb + t >= c0 + nc) {
						memset(ent[t], 0,
							((size_t)cw + 2) *
							sizeof(uint64));
						continue;
					}
					for (i = 0; i < n; i++) {
						buf[i] = fc[(size_t)i * nr *
							nc + (size_t)
							((uint32)r - r0) * nc +
							(sb + t - c0)];
					}
					fft_inv(buf, k, sc);
					unpack_poly(ent[t], cw, buf, ncc);
				}
				scatter_entries(c, (uint32)r, sb, ent, cw);
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
/* Peak words held for a panel of g output rows and g output columns
   against a contraction dimension of kdim: one panel of a, one of b,
   and the piece of c they produce. */

static size_t panel_words(uint32 g, uint32 kdim, uint32 n) {

	return (size_t)n * ((size_t)2 * g * kdim + (size_t)g * g);
}

/* Set once by bw_lingen, which has the arguments to read. Anything
   calling the transform without doing so -- a test harness, say --
   falls back to the same measurement on first use. */

void bmp_mul_fft_set_budget(msieve_obj *obj) {

	uint64 ram = 0;
	const char *tmp;

	if (obj != NULL && obj->nfs_args != NULL &&
	    (tmp = strstr(obj->nfs_args, "bw_mem_mb=")) != NULL) {
		fft_budget_mb = (double)strtoul(tmp + 10, NULL, 10);
		if (obj != NULL) {
			logprintf(obj, "lingen: transforms limited to "
					"%.0f MB\n", fft_budget_mb);
		}
		return;
	}

	ram = get_ram_size();
	fft_budget_mb = (double)ram / 1048576.0 / LINGEN_FFT_MEM_FRACTION;
	if (obj != NULL) {
		logprintf(obj, "lingen: %.0f MB RAM, transforms may use "
				"%.0f MB\n", (double)ram / 1048576.0,
				fft_budget_mb);
	}
}

static double fft_budget(void) {

	if (fft_budget_mb == 0)
		bmp_mul_fft_set_budget(NULL);
	return fft_budget_mb;
}

/* The largest panel that fits the budget. Blocking only ever adds
   transform work -- the pointwise total is untouched -- and the
   pointwise outweighs one block of transforms by about b / log2(n), so
   on a wide matrix the extra is small. On a narrow one it is not,
   which is also where the memory fits whole and no blocking happens. */

static uint32 choose_panel(uint32 rows, uint32 cols, uint32 kdim, uint32 n) {

	uint32 g = MAX(rows, cols);

	g = (g + 63) & ~(uint32)63;	/* whole 64-column groups */

	while (g > 64) {
		double mb = (double)panel_words(g, kdim, n) *
				sizeof(uint64) / 1048576.0;

		if (mb <= fft_budget())
			break;
		g >>= 1;
		if (g < 64)
			g = 64;
	}
	return g;
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

	/* The product is cut into panels to fit the budget, so what has
	   to fit is the smallest panel, not the whole thing. Only if
	   even that is too big does Karatsuba take over, and it streams
	   where this holds transforms. */

	mb = (double)panel_words(64, a->ncols, (uint32)1 << k) *
			sizeof(uint64) / 1048576.0;
	if (mb > fft_budget())
		return 0;

	return 1;
}


/* Rows [r0, r0+nr) of c only. With one process that is all of them;
   with several it is this rank's band, and the bands are XORed
   together afterwards. Splitting the output rows is the one cut that
   needs no extra communication during the product itself: a band of c
   wants that band of a and the whole of b, and every rank already
   holds both, because the recursion around this is replicated. */

void bmp_mul_fft_rows(bmp_t *c, const bmp_t *a, const bmp_t *b,
			uint32 r0, uint32 nr) {

	uint32 nca = (a->len + CHUNK_BITS - 1) / CHUNK_BITS;
	uint32 ncb = (b->len + CHUNK_BITS - 1) / CHUNK_BITS;
	uint32 ncc = nca + ncb - 1;
	uint32 kdim = a->ncols;		/* the contraction dimension */
	uint32 k = 0, n, g, pi, pj;
	uint64 *fa, *fb, *fc;

	while (((uint32)1 << k) < ncc)
		k++;
	n = (uint32)1 << k;

	cantor_init(k);
	g = choose_panel(nr, c->ncols, kdim, n);

	/* One panel of a's rows, one of b's columns, and the piece of c
	   they make. With a single panel this is the whole product and
	   nothing is transformed twice.

	   Columns of c are the outer loop so that b's panel is
	   transformed once each; a's panel is then redone for every
	   column panel, which is the entire cost of blocking. Only the
	   transform repeats -- the pointwise work is the same however
	   the output is cut up. */

	fa = (uint64 *)xmalloc(panel_words(g, kdim, n) * sizeof(uint64));
	fb = fa + (size_t)n * g * kdim;
	fc = fb + (size_t)n * kdim * g;

	for (pj = 0; pj < c->ncols; pj += g) {
		uint32 nj = MIN(g, c->ncols - pj);

		transform_panel(b, fb, k, n, 0, kdim, pj, nj);

		for (pi = r0; pi < r0 + nr; pi += g) {
			uint32 ni = MIN(g, r0 + nr - pi);
			int32 i;

			transform_panel(a, fa, k, n, pi, ni, 0, kdim);
			memset(fc, 0, (size_t)n * ni * nj * sizeof(uint64));

			/* a dense GF(2^64) matrix product at each
			   evaluation point: the b^3 term, and the only
			   part that grows with the width */

#ifdef HAVE_OMP
#pragma omp parallel for schedule(static)
#endif
			for (i = 0; i < (int32)n; i++) {
				const uint64 *A = fa + (size_t)i * ni * kdim;
				const uint64 *B = fb + (size_t)i * kdim * nj;
				uint64 *C = fc + (size_t)i * ni * nj;
				uint32 r, s, t;

				for (r = 0; r < ni; r++) {
					for (t = 0; t < kdim; t++) {
						uint64 av = A[(size_t)r *
								kdim + t];

						if (av == 0)
							continue;
						for (s = 0; s < nj; s++) {
							C[(size_t)r * nj + s]
								^= gf64_mul(av,
								B[(size_t)t *
									nj + s]);
						}
					}
				}
			}

			inverse_panel(c, fc, k, n, pi, ni, pj, nj, ncc);
		}
	}

	free(fa);
}

void bmp_mul_fft(bmp_t *c, const bmp_t *a, const bmp_t *b) {

	bmp_mul_fft_rows(c, a, b, 0, c->nrows);
}
