/*--------------------------------------------------------------------
The algebraic square root's lift, on a GPU.

What the lift does, per Newton step, is three mpz_poly_mul and four
mpz_poly_mod_q over numbers that reach fifteen gigabits. GMP does the
multiplies on eighteen cores and the reductions on seven, because that
is how many coefficients there are to spread them over; the rest of a
46-core machine waits. Moving both onto a card measured about ten times
faster end to end on an H200.

Three things make that worth the trouble rather than a wash:

  - The operands stay on the card. One poly_mul is forty-nine products
    of the same seven numbers, and none of the intermediates is ever
    wanted on the host, so p1 and the Horner accumulators are uploaded
    once and the products never cross PCIe.

  - The reduction is Barrett, which is two multiplies of the size the
    transform is already doing, rather than a division.

  - Barrett's parameter comes from the previous step's by Newton. The
    lift squares q every step, so mu(q^2) is mu(q)^2 to half precision
    and one Newton iteration finishes it. A division at this size costs
    more than every reduction in the step it would serve.

The transform is a Goldilocks NTT -- p = 2^64 - 2^32 + 1, sixteen bits
to a coefficient -- and not a floating point FFT, because a double FFT
stores eight bits of product in twenty-four bytes and runs out of both
memory and rounding margin well before these sizes.

Everything here declines rather than fails. No CUDA build, no card,
too little memory on it, a modulus too large for one prime: each
returns nonzero and the caller runs the code that was already there.
--------------------------------------------------------------------*/

#include "sqrt_gpu.h"

#ifdef HAVE_CUDA
/* gpu_pick(): which card -g chose, or the first one. Its header is
   already extern "C" guarded, so nvcc gets the unmangled name */
#include "cuda_xface.h"
#endif

#ifdef HAVE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_OK(f) do { cudaError_t _e = (f); if (_e != cudaSuccess) \
	return -1; } while (0)

/* which precondition gave out. "GPU multiply failed" on its own
   costs a whole run to place; the line costs nothing to carry */

#define GFAIL(c) do { if (!(c)->failat) (c)->failat = __LINE__; \
	return -1; } while (0)
#define CUDA_OKN(f) do { cudaError_t _e = (f); if (_e != cudaSuccess) \
	return NULL; } while (0)

#define PRIME      0xFFFFFFFF00000001ULL
#define EPS        0xFFFFFFFFULL
#define GENERATOR  7ULL
#define DIGIT_BITS 16
#define DIGIT_MASK 0xFFFFULL
#define SPLIT_BITS 14
#define SPLIT_SIZE (1 << SPLIT_BITS)
#define SPLIT_MASK (SPLIT_SIZE - 1)
#define CARRY_CHUNK 64
#define THREADS 256

/* a modulus beyond this needs more than one prime; the lift's q is
   nowhere near it on any job that finishes, and the caller falls back */
#define MAX_TRANSFORM_LOG 32

/*------------------- the field -------------------*/

__host__ __device__ static inline uint64 addm(uint64 a, uint64 b)
{
	uint64 s = a + b;

	if (s < a) s += EPS;
	if (s >= PRIME) s -= PRIME;
	return s;
}

__host__ __device__ static inline uint64 subm(uint64 a, uint64 b)
{
	return addm(a, PRIME - b);
}

__host__ __device__ static inline uint64 fold(uint64 lo, uint64 hi)
{
	uint32 h1 = (uint32)(hi >> 32);
	uint32 h0 = (uint32)hi;
	uint64 t0, t1, t2;

	t0 = lo - (uint64)h1;
	if (lo < (uint64)h1) t0 -= EPS;
	t1 = (uint64)h0 * EPS;
	t2 = t0 + t1;
	if (t2 < t0) t2 += EPS;
	if (t2 >= PRIME) t2 -= PRIME;
	return t2;
}

__device__ static inline uint64 mulm(uint64 a, uint64 b)
{
	return fold(a * b, __umul64hi(a, b));
}

static inline uint64 mulm_host(uint64 a, uint64 b)
{
	unsigned __int128 x = (unsigned __int128)a * (unsigned __int128)b;

	return fold((uint64)x, (uint64)(x >> 64));
}

static uint64 powm_host(uint64 a, uint64 e)
{
	uint64 r = 1;

	while (e) {
		if (e & 1) r = mulm_host(r, a);
		a = mulm_host(a, a);
		e >>= 1;
	}
	return r;
}

__device__ static inline uint64 twiddle(const uint64 *t1, const uint64 *t0,
					uint64 t)
{
	return mulm(t1[t >> SPLIT_BITS], t0[t & SPLIT_MASK]);
}

/*------------------- transform -------------------*/

__global__ void k_expand(const uint16 *src, uint64 nsrc, uint64 *dst,
			uint64 n)
{
	uint64 i = (uint64)blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) dst[i] = (i < nsrc) ? (uint64)src[i] : 0;
}

/* radix 4: two levels fused so the intermediate stays in registers.
   Derived as two radix-2 stages rather than from the radix-4 butterfly,
   so it is correct for the same reason the pair below is */

__global__ void k_dif4(uint64 *a, uint64 n, uint32 ll, uint32 ls,
			const uint64 *t1, const uint64 *t0)
{
	uint64 tid = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint32 lq = ll - 2;
	uint64 q = 1ULL << lq, h = 1ULL << (ll - 1);
	uint64 j, i0, u0, u1, u2, u3, x0, x1, x2, x3, wa, wb, wc;

	if (tid >= (n >> 2)) return;
	j = tid & (q - 1);
	i0 = ((tid >> lq) << ll) | j;
	wa = twiddle(t1, t0, j << ls);
	wb = twiddle(t1, t0, (j << ls) + (n >> 2));
	wc = twiddle(t1, t0, j << (ls + 1));
	u0 = a[i0]; u1 = a[i0 + q]; u2 = a[i0 + h]; u3 = a[i0 + h + q];
	x0 = addm(u0, u2);
	x2 = mulm(subm(u0, u2), wa);
	x1 = addm(u1, u3);
	x3 = mulm(subm(u1, u3), wb);
	a[i0]         = addm(x0, x1);
	a[i0 + q]     = mulm(subm(x0, x1), wc);
	a[i0 + h]     = addm(x2, x3);
	a[i0 + h + q] = mulm(subm(x2, x3), wc);
}

__global__ void k_dit4(uint64 *a, uint64 n, uint32 ll, uint32 ls,
			const uint64 *t1, const uint64 *t0)
{
	uint64 tid = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint32 lq = ll - 1;
	uint64 q = 1ULL << lq, L = 1ULL << ll;
	uint64 j, i0, u0, v0, u1, v1, x0, x1, x2, x3, p, r, wa, wb, wc;

	if (tid >= (n >> 2)) return;
	j = tid & (q - 1);
	i0 = ((tid >> lq) << (ll + 1)) | j;
	wa = twiddle(t1, t0, j << ls);
	wb = twiddle(t1, t0, j << (ls - 1));
	wc = twiddle(t1, t0, (j << (ls - 1)) + (n >> 2));
	u0 = a[i0];
	v0 = mulm(a[i0 + q], wa);
	u1 = a[i0 + L];
	v1 = mulm(a[i0 + L + q], wa);
	x0 = addm(u0, v0); x1 = subm(u0, v0);
	x2 = addm(u1, v1); x3 = subm(u1, v1);
	p = mulm(x2, wb);
	r = mulm(x3, wc);
	a[i0]         = addm(x0, p);
	a[i0 + L]     = subm(x0, p);
	a[i0 + q]     = addm(x1, r);
	a[i0 + L + q] = subm(x1, r);
}

__global__ void k_dif2(uint64 *a, uint64 n, uint32 ll, uint32 ls,
			const uint64 *t1, const uint64 *t0)
{
	uint64 tid = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint32 lh = ll - 1;
	uint64 half = 1ULL << lh, j, idx, u, v;

	if (tid >= (n >> 1)) return;
	j = tid & (half - 1);
	idx = ((tid >> lh) << ll) | j;
	u = a[idx]; v = a[idx + half];
	a[idx] = addm(u, v);
	a[idx + half] = mulm(subm(u, v), twiddle(t1, t0, j << ls));
}

__global__ void k_dit2(uint64 *a, uint64 n, uint32 ll, uint32 ls,
			const uint64 *t1, const uint64 *t0)
{
	uint64 tid = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint32 lh = ll - 1;
	uint64 half = 1ULL << lh, j, idx, u, v;

	if (tid >= (n >> 1)) return;
	j = tid & (half - 1);
	idx = ((tid >> lh) << ll) | j;
	u = a[idx];
	v = mulm(a[idx + half], twiddle(t1, t0, j << ls));
	a[idx] = addm(u, v);
	a[idx + half] = subm(u, v);
}

__global__ void k_pointwise(uint64 *a, const uint64 *b, uint64 n, uint64 ninv)
{
	uint64 i = (uint64)blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) a[i] = mulm(mulm(a[i], b[i]), ninv);
}

/*------------------- carry, borrow, lengths -------------------*/

/* every one of these is chunked the same way: each chunk works from a
   standing start and reports what it owes the one above, then the debts
   move rightwards until none are left. Two passes settles it at every
   size tried */

/* n is the width to settle over, which for an accumulate is the wider
   of the product and the accumulator already sitting there: the digits
   above the product are still the accumulator's, and a carry has to be
   able to reach them. vlen and acclen bound the two reads, and ovf says
   the top chunk carried out of n entirely, which is never right */

__global__ void k_carry(const uint64 *v, uint16 *out, const uint16 *acc,
			uint64 *cout, uint64 n, uint64 nchunk,
			uint64 vlen, uint64 acclen, int *ovf)
{
	uint64 c = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint64 lo, hi, i, carry = 0;

	if (c >= nchunk) return;
	lo = c * CARRY_CHUNK;
	hi = lo + CARRY_CHUNK;
	if (hi > n) hi = n;
	for (i = lo; i < hi; i++) {
		uint64 x = (i < vlen ? v[i] : 0) + carry +
			((acc && i < acclen) ? (uint64)acc[i] : 0);

		out[i] = (uint16)(x & DIGIT_MASK);
		carry = x >> DIGIT_BITS;
	}
	cout[c] = carry;
	if (carry != 0 && c + 1 == nchunk) *ovf = 1;
}

__global__ void k_carry_fix(uint16 *out, const uint64 *cin, uint64 *cout,
			uint64 nc, uint64 nchunk, int *any, int *ovf)
{
	uint64 c = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint64 lo, hi, i, carry;

	if (c >= nchunk) return;
	cout[c] = 0;
	if (c == 0) return;
	carry = cin[c - 1];
	if (carry == 0) return;
	lo = c * CARRY_CHUNK;
	hi = lo + CARRY_CHUNK;
	if (hi > nc) hi = nc;
	for (i = lo; i < hi && carry != 0; i++) {
		uint64 x = (uint64)out[i] + carry;

		out[i] = (uint16)(x & DIGIT_MASK);
		carry = x >> DIGIT_BITS;
	}
	cout[c] = carry;
	if (carry != 0) {
		*any = 1;
		if (c + 1 == nchunk) *ovf = 1;
	}
}

/* alen and blen because an operand is routinely shorter than the width
   it is used at, and reading past it is reading past an allocation */

__global__ void k_add(const uint16 *a, const uint16 *b, uint16 *r,
			uint64 *cout, uint64 n, uint64 nchunk,
			uint64 alen, uint64 blen, int *ovf)
{
	uint64 c = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint64 lo, hi, i, carry = 0;

	if (c >= nchunk) return;
	lo = c * CARRY_CHUNK;
	hi = lo + CARRY_CHUNK;
	if (hi > n) hi = n;
	for (i = lo; i < hi; i++) {
		uint64 x = (i < alen ? (uint64)a[i] : 0) +
			   (i < blen ? (uint64)b[i] : 0) + carry;

		r[i] = (uint16)(x & DIGIT_MASK);
		carry = x >> DIGIT_BITS;
	}
	cout[c] = carry;
	if (carry != 0 && c + 1 == nchunk) *ovf = 1;
}

__global__ void k_sub(const uint16 *a, const uint16 *b, uint16 *r,
			uint64 *bout, uint64 n, uint64 nchunk,
			uint64 alen, uint64 blen)
{
	uint64 c = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint64 lo, hi, i;
	int64 borrow = 0;

	if (c >= nchunk) return;
	lo = c * CARRY_CHUNK;
	hi = lo + CARRY_CHUNK;
	if (hi > n) hi = n;
	for (i = lo; i < hi; i++) {
		int64 x = (i < alen ? (int64)a[i] : 0) -
			  (i < blen ? (int64)b[i] : 0) - borrow;

		if (x < 0) { x += 65536; borrow = 1; }
		else borrow = 0;
		r[i] = (uint16)x;
	}
	bout[c] = (uint64)borrow;
}

__global__ void k_sub_fix(uint16 *r, const uint64 *bin, uint64 *bout,
			uint64 n, uint64 nchunk, int *any)
{
	uint64 c = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint64 lo, hi, i;
	int64 borrow;

	if (c >= nchunk) return;
	bout[c] = 0;
	if (c == 0) return;
	borrow = (int64)bin[c - 1];
	if (borrow == 0) return;
	lo = c * CARRY_CHUNK;
	hi = lo + CARRY_CHUNK;
	if (hi > n) hi = n;
	for (i = lo; i < hi && borrow != 0; i++) {
		int64 x = (int64)r[i] - borrow;

		if (x < 0) { x += 65536; borrow = 1; }
		else borrow = 0;
		r[i] = (uint16)x;
	}
	bout[c] = (uint64)borrow;
	if (borrow != 0) *any = 1;
}

__global__ void k_top(const uint16 *a, uint64 n, unsigned long long *idx)
{
	uint64 i = (uint64)blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n && a[i] != 0)
		atomicMax(idx, (unsigned long long)(i + 1));
}

__global__ void k_diff(const uint16 *a, const uint16 *b, uint64 n,
			unsigned long long *idx, uint64 alen, uint64 blen)
{
	uint64 i = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint16 av, bv;

	if (i >= n) return;
	av = (i < alen) ? a[i] : 0;
	bv = (i < blen) ? b[i] : 0;
	if (av != bv)
		atomicMax(idx, (unsigned long long)(i + 1));
}

__global__ void k_zero(uint16 *p, uint64 n)
{
	uint64 i = (uint64)blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) p[i] = 0;
}

/* big by small, for Horner's reduction: mod(x)'s coefficients are the
   algebraic poly's scaled by powers of its leading coefficient, so a
   few hundred bits at the very most against a q of fifteen gigabits.
   Each output digit sums at most SMALL_MAX products of two sixteen bit
   digits, which stays inside a uint64 and leaves the chunked carry to
   finish the job -- two passes over the number rather than the sixteen
   a schoolbook multiply would take */

#define SMALL_MAX 32

__global__ void k_small_mul(const uint16 *big, uint64 nbig,
			const uint16 *small, uint32 nsmall, uint64 *out,
			uint64 n)
{
	uint64 i = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
	uint64 acc = 0;
	uint32 t;

	if (i >= n) return;
	for (t = 0; t < nsmall; t++) {
		uint64 j = i - t;

		if (i >= t && j < nbig)
			acc += (uint64)big[j] * (uint64)small[t];
	}
	out[i] = acc;
}

/*------------------- numbers on the card -------------------*/

typedef struct {
	uint16 *d;		/* magnitude, base 2^16 */
	uint64 ndig;
	int neg;		/* the lift's accumulators do go negative */
} gnum;

typedef struct {
	uint64 n;		/* transform length */
	uint32 ln;
	uint64 max_digits;	/* digits an operand may have */
	uint64 nc;		/* digits a product may have */
	uint64 nchunk;
	uint64 *A, *B;
	uint64 *twf1, *twf0, *twi1, *twi0;
	uint64 *dc0, *dc1;
	int *dany;
	int *dovf;		/* a settle carried out of its width */
	unsigned long long *dword;	/* len and compare report here */
	uint64 ninv;
	void *hstage;		/* pinned, for results coming back */

	uint32 degree;
	gnum p1[MAX_POLY_DEGREE + 1];	/* the multiplicand, resident */
	gnum p2cur;			/* one coefficient of p2 at a time */
	gnum tmp[MAX_POLY_DEGREE + 2];	/* Horner's accumulators */
	gnum modc[MAX_POLY_DEGREE + 1];	/* mod(x), small */
	uint32 modlen[MAX_POLY_DEGREE + 1];
	int modneg[MAX_POLY_DEGREE + 1];
	int have_mod;

	gnum q, mu, t1, t2, t3;
	gnum hs1;		/* a product waiting to be added with signs */
	uint64 k;			/* digits Barrett works in */
	int have_mu;
	uint64 k_prev;
	mpz_t qprev;

	msieve_obj *obj;
	int ok;				/* cleared if a self check fails */
	int failat;			/* line that declined, for the log */
	int trace;			/* column by column, for a disagreement */
	uint64 fa, fb;		/* and the two numbers behind it */
	int checked;			/* steps verified against the CPU */
} gctx;

#endif /* HAVE_CUDA */

#ifdef HAVE_CUDA

/*------------------- numbers: host side -------------------*/

static int gnum_alloc(gctx *c, gnum *x)
{
	x->ndig = 0;
	x->neg = 0;
	CUDA_OK(cudaMalloc((void **)&x->d, (c->nc + 8) * sizeof(uint16)));
	CUDA_OK(cudaMemset(x->d, 0, (c->nc + 8) * sizeof(uint16)));
	return 0;
}

static void gnum_free(gnum *x)
{
	if (x->d) cudaFree(x->d);
	x->d = NULL;
}

/* GMP's limbs are already the base 2^16 digits on a little endian
   machine, so this is a copy and not a conversion */

static int gnum_put(gctx *c, gnum *x, const mpz_t v)
{
	uint64 nd = (mpz_sizeinbase(v, 2) + DIGIT_BITS - 1) / DIGIT_BITS;

	if (mpz_sgn(v) == 0) {
		x->ndig = 0;
		x->neg = 0;
		return 0;
	}
	if (nd > c->nc)
		return -1;
	CUDA_OK(cudaMemcpy(x->d, (const void *)mpz_limbs_read(v),
			nd * sizeof(uint16), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemset(x->d + nd, 0, 8 * sizeof(uint16)));
	x->ndig = nd;
	x->neg = (mpz_sgn(v) < 0);
	return 0;
}

static int gnum_get(gctx *c, mpz_t v, gnum *x)
{
	uint64 nd = x->ndig ? x->ndig : 1;
	uint64 nolimb = (nd * 2 + sizeof(mp_limb_t) - 1) / sizeof(mp_limb_t);

	CUDA_OK(cudaMemset((char *)x->d + nd * 2, 0,
			nolimb * sizeof(mp_limb_t) - nd * 2));
	CUDA_OK(cudaMemcpy(c->hstage, x->d, nolimb * sizeof(mp_limb_t),
			cudaMemcpyDeviceToHost));
	memcpy((void *)mpz_limbs_write(v, nolimb), c->hstage,
			nolimb * sizeof(mp_limb_t));
	mpz_limbs_finish(v, nolimb);
	if (x->neg) mpz_neg(v, v);
	return 0;
}

/* These two used to allocate and free a device word per call and fold a
   failed allocation into their result -- a cmp that could not run said
   "equal", which the caller acts on by zeroing an accumulator. Both now
   report through the context's scratch word and return an error */

static int gnum_len(gctx *c, gnum *x, uint64 width, uint64 *out)
{
	uint64 blocks = (width + THREADS - 1) / THREADS;
	unsigned long long hidx = 0;

	*out = 0;
	if (width == 0)
		return 0;
	CUDA_OK(cudaMemcpy(c->dword, &hidx, sizeof(hidx),
			cudaMemcpyHostToDevice));
	k_top<<<(unsigned)blocks, THREADS>>>(x->d, width, c->dword);
	CUDA_OK(cudaMemcpy(&hidx, c->dword, sizeof(hidx),
			cudaMemcpyDeviceToHost));
	*out = (uint64)hidx;
	return 0;
}

/* -1, 0, 1 on magnitudes alone */

static int gnum_cmp_mag(gctx *c, gnum *a, gnum *b, uint64 width, int *out)
{
	uint64 blocks = (width + THREADS - 1) / THREADS;
	unsigned long long hidx = 0;
	uint16 av = 0, bv = 0;

	*out = 0;
	if (width == 0)
		return 0;
	CUDA_OK(cudaMemcpy(c->dword, &hidx, sizeof(hidx),
			cudaMemcpyHostToDevice));
	k_diff<<<(unsigned)blocks, THREADS>>>(a->d, b->d, width, c->dword,
					a->ndig, b->ndig);
	CUDA_OK(cudaMemcpy(&hidx, c->dword, sizeof(hidx),
			cudaMemcpyDeviceToHost));
	if (hidx == 0)
		return 0;
	if (hidx - 1 < a->ndig)
		CUDA_OK(cudaMemcpy(&av, a->d + (hidx - 1), sizeof(uint16),
				cudaMemcpyDeviceToHost));
	if (hidx - 1 < b->ndig)
		CUDA_OK(cudaMemcpy(&bv, b->d + (hidx - 1), sizeof(uint16),
				cudaMemcpyDeviceToHost));
	*out = av > bv ? 1 : -1;
	return 0;
}

static int gnum_add_mag(gctx *c, gnum *r, gnum *a, gnum *b, uint64 width)
{
	uint64 nchunk = (width + CARRY_CHUNK - 1) / CARRY_CHUNK;
	uint64 bc = (nchunk + THREADS - 1) / THREADS;
	int passes = 0, ovf = 0;

	CUDA_OK(cudaMemcpy(c->dovf, &ovf, sizeof(int),
			cudaMemcpyHostToDevice));
	k_add<<<(unsigned)bc, THREADS>>>(a->d, b->d, r->d, c->dc0, width,
					nchunk, a->ndig, b->ndig, c->dovf);
	for (;;) {
		int any = 0;
		uint64 *t;

		CUDA_OK(cudaMemcpy(c->dany, &any, sizeof(int),
					cudaMemcpyHostToDevice));
		k_carry_fix<<<(unsigned)bc, THREADS>>>(r->d, c->dc0, c->dc1,
					width, nchunk, c->dany, c->dovf);
		CUDA_OK(cudaMemcpy(&any, c->dany, sizeof(int),
					cudaMemcpyDeviceToHost));
		t = c->dc0; c->dc0 = c->dc1; c->dc1 = t;
		if (!any) break;
		if (++passes > 16) return -1;
	}

	/* a carry out of the last chunk has nowhere above it to go, and
	   silently wraps the sum. The deliberate one is on the subtract
	   side, in Newton, where the dropped borrow is the implicit B */

	CUDA_OK(cudaMemcpy(&ovf, c->dovf, sizeof(int),
			cudaMemcpyDeviceToHost));
	if (ovf)
		return -1;
	r->ndig = width;
	return 0;
}

static int gnum_sub_mag(gctx *c, gnum *r, gnum *a, gnum *b, uint64 width)
{
	uint64 nchunk = (width + CARRY_CHUNK - 1) / CARRY_CHUNK;
	uint64 bc = (nchunk + THREADS - 1) / THREADS;
	int passes = 0;

	k_sub<<<(unsigned)bc, THREADS>>>(a->d, b->d, r->d, c->dc0, width,
					nchunk, a->ndig, b->ndig);
	for (;;) {
		int any = 0;
		uint64 *t;

		CUDA_OK(cudaMemcpy(c->dany, &any, sizeof(int),
					cudaMemcpyHostToDevice));
		k_sub_fix<<<(unsigned)bc, THREADS>>>(r->d, c->dc0, c->dc1,
					width, nchunk, c->dany);
		CUDA_OK(cudaMemcpy(&any, c->dany, sizeof(int),
					cudaMemcpyDeviceToHost));
		t = c->dc0; c->dc0 = c->dc1; c->dc1 = t;
		if (!any) break;
		if (++passes > 16) return -1;
	}
	r->ndig = width;
	return 0;
}

/* r = a + b with signs. Same sign adds magnitudes and keeps it; opposite
   signs subtract the smaller from the larger and take the larger's. This
   is the piece the lift needs that a plain bignum multiply does not:
   Horner's reduction subtracts mod(x)'s coefficients, which carry the
   algebraic polynomial's signs, and the accumulators go negative */

/* the add and subtract paths write r across the whole width, so r is
   clean above its new length. These two copies did not, and left
   whatever r held before sitting above it -- the stale tail that the
   carry and compare kernels then had to be taught to step around */

static int gnum_copy(gctx *c, gnum *r, gnum *x, uint64 width)
{
	uint64 nd = x->ndig;

	if (r != x) {
		if (nd)
			CUDA_OK(cudaMemcpy(r->d, x->d, nd * sizeof(uint16),
					cudaMemcpyDeviceToDevice));
		r->ndig = nd;
	}
	if (width > r->ndig) {
		uint64 blocks = ((width - r->ndig) + THREADS - 1) / THREADS;

		k_zero<<<(unsigned)blocks, THREADS>>>(r->d + r->ndig,
					width - r->ndig);
	}
	r->neg = x->neg;
	return 0;
}

static int gnum_add_signed(gctx *c, gnum *r, gnum *a, gnum *b, uint64 width)
{
	int cmp;

	if (a->ndig == 0)
		return gnum_copy(c, r, b, width);
	if (b->ndig == 0)
		return gnum_copy(c, r, a, width);
	if (a->neg == b->neg) {
		if (gnum_add_mag(c, r, a, b, width)) return -1;
		r->neg = a->neg;
		if (gnum_len(c, r, width, &r->ndig)) return -1;
		return 0;
	}
	if (gnum_cmp_mag(c, a, b, width, &cmp)) return -1;
	if (cmp == 0) {
		uint64 blocks = (width + THREADS - 1) / THREADS;

		k_zero<<<(unsigned)blocks, THREADS>>>(r->d, width);
		r->ndig = 0;
		r->neg = 0;
		return 0;
	}
	if (cmp > 0) {
		if (gnum_sub_mag(c, r, a, b, width)) return -1;
		r->neg = a->neg;
	}
	else {
		if (gnum_sub_mag(c, r, b, a, width)) return -1;
		r->neg = b->neg;
	}
	if (gnum_len(c, r, width, &r->ndig)) return -1;
	return 0;
}

/*------------------- the multiply -------------------*/

static void transform(gctx *c, uint64 *buf, int forward)
{
	uint64 bq = ((c->n >> 2) + THREADS - 1) / THREADS;
	uint64 bh = ((c->n >> 1) + THREADS - 1) / THREADS;
	uint64 *t1 = forward ? c->twf1 : c->twi1;
	uint64 *t0 = forward ? c->twf0 : c->twi0;
	uint32 ll, ln = c->ln;

	if (forward) {
		for (ll = ln; ll >= 2; ll -= 2)
			k_dif4<<<(unsigned)bq, THREADS>>>(buf, c->n, ll,
					ln - ll, t1, t0);
		if (ln & 1)
			k_dif2<<<(unsigned)bh, THREADS>>>(buf, c->n, 1,
					ln - 1, t1, t0);
	}
	else {
		if (ln & 1)
			k_dit2<<<(unsigned)bh, THREADS>>>(buf, c->n, 1,
					ln - 1, t1, t0);
		for (ll = (ln & 1) ? 2 : 1; ll <= ln - 1; ll += 2)
			k_dit4<<<(unsigned)bq, THREADS>>>(buf, c->n, ll,
					ln - ll, t1, t0);
	}
}

/* r = a*b on magnitudes, or r += a*b when acc is set -- which is only
   valid while both are non-negative, so the caller checks. Digit
   pointers rather than gnum, because Barrett's shifts are by whole
   digits and want to be an offset rather than a copy */

static int gmul_raw(gctx *c, gnum *r, const uint16 *ad, uint64 an,
			const uint16 *bd, uint64 bn, int acc)
{
	uint64 bn_blk = (c->n + THREADS - 1) / THREADS;
	uint64 nc = an + bn + 1;
	uint64 width, nch, bc;
	int passes = 1, ovf = 0;

	if (an == 0 || bn == 0) {
		if (!acc) {
			uint64 blocks = (c->nc + THREADS - 1) / THREADS;

			k_zero<<<(unsigned)blocks, THREADS>>>(r->d, c->nc);
			r->ndig = 0;
			r->neg = 0;
		}
		return 0;
	}

	/* the context is sized so this cannot happen; clamping instead
	   would truncate the product and say nothing about it */

	/* against the transform actually in use, not the largest buffer:
	   a convolution longer than n wraps round and is silently wrong */

	if (nc > c->nc || nc > c->n)
		GFAIL(c);

	/* an accumulate settles across the wider of the product and what is
	   already in r: the digits above the product are still r's own, and
	   a carry out of the product has to be able to reach them */

	width = (acc && r->ndig > nc) ? r->ndig : nc;
	nch = (width + CARRY_CHUNK - 1) / CARRY_CHUNK;
	bc = (nch + THREADS - 1) / THREADS;

	k_expand<<<(unsigned)bn_blk, THREADS>>>(ad, an, c->A, c->n);
	transform(c, c->A, 1);
	k_expand<<<(unsigned)bn_blk, THREADS>>>(bd, bn, c->B, c->n);
	transform(c, c->B, 1);
	k_pointwise<<<(unsigned)bn_blk, THREADS>>>(c->A, c->B, c->n, c->ninv);
	transform(c, c->A, 0);

	CUDA_OK(cudaMemcpy(c->dovf, &ovf, sizeof(int),
			cudaMemcpyHostToDevice));
	k_carry<<<(unsigned)bc, THREADS>>>(c->A, r->d, acc ? r->d : NULL,
					c->dc0, width, nch, nc,
					acc ? r->ndig : 0, c->dovf);
	for (;;) {
		int any = 0;
		uint64 *t;

		CUDA_OK(cudaMemcpy(c->dany, &any, sizeof(int),
					cudaMemcpyHostToDevice));
		k_carry_fix<<<(unsigned)bc, THREADS>>>(r->d, c->dc0, c->dc1,
					width, nch, c->dany, c->dovf);
		CUDA_OK(cudaMemcpy(&any, c->dany, sizeof(int),
					cudaMemcpyDeviceToHost));
		t = c->dc0; c->dc0 = c->dc1; c->dc1 = t;
		if (!any) break;
		if (++passes > 16) GFAIL(c);
	}
	CUDA_OK(cudaMemcpy(&ovf, c->dovf, sizeof(int),
			cudaMemcpyDeviceToHost));
	if (ovf)
		GFAIL(c);
	r->ndig = width;
	return 0;
}

/*------------------- Barrett, and mu by Newton -------------------*/

/* mu(q^2) from mu(q): the square lands about half the digits right and
   one Newton step finishes it. The division it replaces costs more than
   every reduction in the step it would serve */

static int mu_newton(gctx *c, gnum *out, gnum *mu_old, uint64 k_old)
{
	uint64 k = c->k, w2 = 2 * k, shift, elen;
	int fix;

	if (4 * k_old < 2 * k)
		GFAIL(c);
	shift = 4 * k_old - 2 * k;

	if (gmul_raw(c, &c->t1, mu_old->d, mu_old->ndig,
			mu_old->d, mu_old->ndig, 0)) GFAIL(c);
	if (c->t1.ndig <= shift) GFAIL(c);
	CUDA_OK(cudaMemcpy(out->d, c->t1.d + shift,
			(c->t1.ndig - shift) * sizeof(uint16),
			cudaMemcpyDeviceToDevice));
	out->ndig = c->t1.ndig - shift;
	out->neg = 0;

	for (fix = 0; fix < 6; fix++) {
		uint64 h, off, sh, plen, tlen;

		if (gmul_raw(c, &c->t1, c->q.d, c->q.ndig,
				out->d, out->ndig, 0)) GFAIL(c);
		if (gnum_len(c, &c->t1, c->t1.ndig, &tlen)) GFAIL(c);
		if (tlen > w2)
			break;			/* reached B/q */

		{	/* e = B - q*y, a fixed width subtract from zero */
			uint64 blocks = (w2 + THREADS - 1) / THREADS;

			k_zero<<<(unsigned)blocks, THREADS>>>(c->t2.d, w2);
			c->t2.ndig = 0;
		}
		if (gnum_sub_mag(c, &c->t2, &c->t2, &c->t1, w2)) GFAIL(c);
		if (gnum_len(c, &c->t2, w2, &elen)) GFAIL(c);
		if (elen == 0) break;

		/* e is about 1.5k digits while y is only half accurate, so
		   y*e would want 2.5k and the context holds 2k. The
		   correction is only about k/2 digits, so everything below
		   the top k+1 digits of e lands under it */

		h = elen > k + 1 ? k + 1 : elen;
		off = elen - h;
		sh = w2 - off;
		if (gmul_raw(c, &c->t3, out->d, out->ndig,
				c->t2.d + off, h, 0)) GFAIL(c);

		/* the true length, not gmul_raw's an+bn+1: at convergence
		   the product is exactly sh digits and the correction is
		   zero, but the claim overshoots and the loop would never
		   finish */

		if (gnum_len(c, &c->t3, c->t3.ndig, &plen)) GFAIL(c);
		if (plen <= sh) break;
		{
			gnum shifted = c->t3;

			shifted.d = c->t3.d + sh;
			shifted.ndig = plen - sh;
			if (gnum_add_mag(c, out, out, &shifted, k + 2))
				GFAIL(c);
			if (gnum_len(c, out, k + 2, &out->ndig))
				GFAIL(c);
		}
	}
	return 0;
}

/* r = x mod q, for x below b^(2k). Two multiplies the size of the ones
   around them, then a subtract and at most two corrections. A negative
   x comes back as q - |x| mod q, which is what mpz_fdiv_r would give */

static int barrett(gctx *c, gnum *r, gnum *x)
{
	uint64 k = c->k, w = k + 1, xlen;
	int guard, neg = x->neg;

	/* ndig is gmul_raw's claim of an + bn + 1, which overshoots;
	   the bound below is tight enough to care about the difference */

	if (gnum_len(c, x, x->ndig, &xlen))
		GFAIL(c);
	x->ndig = xlen;
	if (xlen > 2 * k) {
		c->fa = xlen;
		c->fb = k;
		GFAIL(c);
	}

	if (x->ndig + 1 < k) {
		if (r != x) {
			CUDA_OK(cudaMemcpy(r->d, x->d,
				(x->ndig ? x->ndig : 1) * sizeof(uint16),
				cudaMemcpyDeviceToDevice));
		}
		CUDA_OK(cudaMemset(r->d + x->ndig, 0,
				(w - x->ndig) * sizeof(uint16)));
		if (gnum_len(c, r, w, &r->ndig)) GFAIL(c);
	}
	else {
		if (gmul_raw(c, &c->t1, x->d + (k - 1), x->ndig - (k - 1),
				c->mu.d, c->mu.ndig, 0)) GFAIL(c);
		if (c->t1.ndig > w) {
			if (gmul_raw(c, &c->t2, c->t1.d + w,
					c->t1.ndig - w,
					c->q.d, c->q.ndig, 0)) GFAIL(c);
		}
		else {
			uint64 blocks = (c->nc + THREADS - 1) / THREADS;

			k_zero<<<(unsigned)blocks, THREADS>>>(c->t2.d, c->nc);
			c->t2.ndig = 0;
		}
		if (gnum_sub_mag(c, r, x, &c->t2, w)) GFAIL(c);
		if (gnum_len(c, r, w, &r->ndig)) GFAIL(c);
	}

	for (guard = 0; guard < 4; guard++) {
		int cmp;

		if (gnum_cmp_mag(c, r, &c->q, w, &cmp)) GFAIL(c);
		if (cmp < 0)
			break;
		if (gnum_sub_mag(c, r, r, &c->q, w)) GFAIL(c);
		if (gnum_len(c, r, w, &r->ndig)) GFAIL(c);
	}
	if (guard >= 4) {
		c->fa = (uint64)guard;
		c->fb = k;
		GFAIL(c);
	}

	/* fdiv_r semantics: a negative value lands in [0, q) */

	if (neg && r->ndig != 0) {
		if (gnum_sub_mag(c, r, &c->q, r, w)) GFAIL(c);
		if (gnum_len(c, r, w, &r->ndig)) GFAIL(c);
	}
	r->neg = 0;
	return 0;
}

#endif /* HAVE_CUDA */

#ifdef HAVE_CUDA

/* a column of the accumulator, reduced and squeezed into one line.
   Bit length and the bottom thirty-two bits is enough to see which
   column two runs of the same arithmetic first disagree on */

static void trace_val(gctx *c, const char *who, uint32 col,
			const char *stage, uint32 idx, mpz_t v, mpz_t q)
{
	mpz_t t;
	uint64 bits;
	unsigned long lo;

	mpz_init(t);
	mpz_fdiv_r(t, v, q);
	bits = (uint64)mpz_sizeinbase(t, 2);
	if (mpz_sgn(t) == 0) bits = 0;
	mpz_fdiv_r_2exp(t, t, 32);
	lo = mpz_get_ui(t);
	logprintf(c->obj, (char *)"square root: trace %s col %u %s tmp%u %" PRIu64 " bits, low %08lx\n",
			(char *)who, col, (char *)stage, idx, bits, lo);
	mpz_clear(t);
}

/*------------------- Horner -------------------*/

/* mpz_poly_mul's loop, kept line for line, because it is the version
   that ships and the only thing being changed is where the arithmetic
   happens. Two things are free here that are not on the host: shifting
   the accumulator up by one is a swap of descriptors rather than of
   several gigabytes.

   What is NOT free, whatever is said elsewhere in this tree, is the
   top term of the reduction. mod(x) is not monic by the time it gets
   here -- rsa120 reports a leading coefficient that is not 1 -- so it
   is multiplied like every other one. Treating it as a plain subtract
   drops that factor and the lift then disagrees with GMP in exactly
   one accumulator, tmp[d], which the bubble spreads over all of them
   by the next column. */

static int horner(gctx *c, mpz_poly_t *p1, mpz_poly_t *p2, mpz_poly_t *alg)
{
	uint32 d = alg->degree, d1 = p1->degree, d2 = p2->degree;
	uint32 pd, i, j;
	uint64 w, wch;

	/* the transform already follows this step's q rather than the
	   largest the lift reaches; so should everything beside it. Two
	   operands below q multiply to under b^(2k-1) and the row stays
	   there, mod(x)'s coefficients being applied to a reduced one */

	w = 2 * c->k + 4;
	if (w > c->nc)
		w = c->nc;
	wch = (w + CARRY_CHUNK - 1) / CARRY_CHUNK;

	/* p1 stays on the card for every one of the products below;
	   p2 arrives one coefficient at a time, which is all Horner
	   ever looks at */

	for (j = 0; j <= d1; j++)
		if (gnum_put(c, &c->p1[j], p1->coeff[j])) GFAIL(c);

	if (gnum_put(c, &c->p2cur, p2->coeff[d2])) GFAIL(c);
	for (j = 0; j <= d1; j++) {
		if (gmul_raw(c, &c->tmp[j], c->p1[j].d, c->p1[j].ndig,
				c->p2cur.d, c->p2cur.ndig, 0)) GFAIL(c);
		c->tmp[j].neg = c->p1[j].neg ^ c->p2cur.neg;
	}
	/* ndig alone would leave the previous call's digits sitting in
	   these, and they are rotated into live slots on the first pass */

	for (j = d1 + 1; j <= d + 1; j++) {
		uint64 blocks = (w + THREADS - 1) / THREADS;

		k_zero<<<(unsigned)blocks, THREADS>>>(c->tmp[j].d, w);
		c->tmp[j].ndig = 0;
		c->tmp[j].neg = 0;
	}
	pd = d1;

	for (i = d2; i-- > 0; ) {

		/* shift up by one, bubbling the top to the bottom, which
		   tmp[0] then overwrites */

		for (j = pd + 1; j-- > 0; ) {
			gnum t = c->tmp[j + 1];

			c->tmp[j + 1] = c->tmp[j];
			c->tmp[j] = t;
		}

		if (gnum_put(c, &c->p2cur, p2->coeff[i])) GFAIL(c);

		for (j = 0; j <= d1; j++) {
			int sgn = c->p1[j].neg ^ c->p2cur.neg;

			if (j == 0) {
				if (gmul_raw(c, &c->tmp[0], c->p1[0].d,
						c->p1[0].ndig, c->p2cur.d,
						c->p2cur.ndig, 0)) GFAIL(c);
				c->tmp[0].neg = sgn;
				continue;
			}

			/* the carry kernel can fold the product straight
			   into the accumulator, but only while both are
			   positive; once an accumulator has gone negative
			   the product has to be formed and added with
			   signs */

			if (!c->tmp[j].neg && !sgn) {
				if (gmul_raw(c, &c->tmp[j], c->p1[j].d,
						c->p1[j].ndig, c->p2cur.d,
						c->p2cur.ndig, 1)) GFAIL(c);
			}
			else {
				if (gmul_raw(c, &c->hs1, c->p1[j].d,
						c->p1[j].ndig, c->p2cur.d,
						c->p2cur.ndig, 0)) GFAIL(c);
				c->hs1.neg = sgn;
				if (gnum_add_signed(c, &c->tmp[j], &c->tmp[j],
						&c->hs1, w)) GFAIL(c);
			}
		}

		if (c->trace) {
			mpz_t v;
			uint32 t;

			mpz_init(v);
			for (t = 0; t <= d + 1; t++) {
				gnum_get(c, v, &c->tmp[t]);
				trace_val(c, "gpu", i, "product", t, v,
						c->qprev);
			}
			mpz_clear(v);
		}

		pd = d + 1;
		while (pd && c->tmp[pd].ndig == 0)
			pd--;
		if (pd <= d)
			continue;

		/* mod(x)'s coefficients are the algebraic poly's and are
		   not reduced -- a few hundred bits each. Multiplying an
		   unreduced accumulator by them puts about thirty digits
		   on every coefficient per column, and after a handful of
		   columns they sit just past the b^(2k) that is the only
		   thing Barrett accepts -- k being pinned to q's own digit
		   count, so it cannot simply be raised to cover them.
		   Reducing the single coefficient they all multiply holds
		   the whole row under b^(2k-1) for one reduction a column,
		   and changes nothing: the result is taken mod q anyway */

		if (c->trace) {
			mpz_t v;

			mpz_init(v);
			gnum_get(c, v, &c->tmp[pd]);
			trace_val(c, "gpu", i, "prebar", pd, v, c->qprev);
			mpz_clear(v);
		}
		if (barrett(c, &c->tmp[pd], &c->tmp[pd]))
			GFAIL(c);
		if (c->tmp[pd].ndig == 0) {
			pd--;
			continue;
		}

		for (j = 0; j <= d; j++) {

			/* every coefficient of mod(x), the top one included:
			   it is not 1, so it gets no path of its own */

			if (c->modlen[j] == 0)
				continue;		/* coefficient is 0 */

			{
				uint64 blocks = (w + THREADS - 1) / THREADS;

				k_small_mul<<<(unsigned)blocks, THREADS>>>(
					c->tmp[pd].d, c->tmp[pd].ndig,
					c->modc[j].d, c->modlen[j],
					c->A, w);
			}
			{	/* the products land unreduced; the usual
				   chunked carry turns them into digits */
				uint64 bc = (wch + THREADS - 1) / THREADS;
				int passes = 1;

				k_carry<<<(unsigned)bc, THREADS>>>(c->A,
					c->hs1.d, NULL, c->dc0, w, wch,
					w, 0, c->dovf);
				for (;;) {
					int any = 0;
					uint64 *t;

					CUDA_OK(cudaMemcpy(c->dany, &any,
						sizeof(int),
						cudaMemcpyHostToDevice));
					k_carry_fix<<<(unsigned)bc, THREADS>>>(
						c->hs1.d, c->dc0, c->dc1, w,
						wch, c->dany, c->dovf);
					CUDA_OK(cudaMemcpy(&any, c->dany,
						sizeof(int),
						cudaMemcpyDeviceToHost));
					t = c->dc0; c->dc0 = c->dc1;
					c->dc1 = t;
					if (!any) break;
					if (++passes > 16) GFAIL(c);
				}
				if (gnum_len(c, &c->hs1, w, &c->hs1.ndig))
					GFAIL(c);
			}
			/* subtract, so the product's sign is flipped */
			c->hs1.neg = !(c->tmp[pd].neg ^ c->modneg[j]);
			if (gnum_add_signed(c, &c->tmp[j], &c->tmp[j],
					&c->hs1, w)) GFAIL(c);
		}
		if (c->trace) {
			mpz_t v;
			uint32 t;

			mpz_init(v);
			for (t = 0; t <= d + 1; t++) {
				gnum_get(c, v, &c->tmp[t]);
				trace_val(c, "gpu", i, "reduced", t, v,
						c->qprev);
			}
			mpz_clear(v);
		}
		pd--;
	}

	p1->degree = pd;
	return 0;
}

/*------------------- q, and its Barrett parameter -------------------*/

static int ensure_mu(gctx *c, mpz_t q)
{
	uint64 k_old = c->k;
	int reuse;

	/* exactly q's digit count: Barrett's estimate is only accurate
	   while b^(k-1) <= q < b^k, and a k one too large leaves the
	   error scaled by b^(k-1)/q, which runs to a whole digit */

	c->k = (mpz_sizeinbase(q, 2) + DIGIT_BITS - 1) / DIGIT_BITS;
	if (2 * c->k > c->nc)
		return -1;
	if (gnum_put(c, &c->q, q))
		return -1;

	/* the lift squares q every step, so the previous mu is half of
	   this one and Newton finishes it. Anything else -- the first
	   step, or a caller that jumped -- falls back to dividing, which
	   is correct but is the cost this exists to avoid */

	reuse = c->have_mu && mpz_sgn(c->qprev) != 0;
	if (reuse) {
		mpz_t sq;

		mpz_init(sq);
		mpz_mul(sq, c->qprev, c->qprev);
		reuse = (mpz_cmp(sq, q) == 0);
		mpz_clear(sq);
	}

	if (reuse && mu_newton(c, &c->mu, &c->mu, k_old) == 0) {
		mpz_set(c->qprev, q);
		return 0;
	}

	{
		mpz_t mu, b2k;

		mpz_init(mu);
		mpz_init(b2k);
		mpz_ui_pow_ui(b2k, 2,
			(unsigned long)(2 * c->k * DIGIT_BITS));
		mpz_tdiv_q(mu, b2k, q);
		if (gnum_put(c, &c->mu, mu)) {
			mpz_clear(mu); mpz_clear(b2k);
			return -1;
		}
		mpz_clear(mu);
		mpz_clear(b2k);
	}
	c->have_mu = 1;
	mpz_set(c->qprev, q);
	return 0;
}

/*------------------- the reference, for the first steps -------------------*/

/* mpz_poly_mul's loop again, in GMP, so the card can be checked against
   it the first couple of times it is used. The lift's early steps are
   small enough that this costs a moment, and a wrong square root costs
   the hours between here and a dependency that yields nothing */

static int reference_mul_mod(mpz_poly_t *out, mpz_poly_t *p1,
			mpz_poly_t *p2, mpz_poly_t *alg, mpz_t q,
			mpz_poly_t *raw, gctx *c)
{
	uint32 d = alg->degree, d1 = p1->degree, d2 = p2->degree;
	uint32 pd, i, j;
	mpz_t tmp[MAX_POLY_DEGREE + 2], t;

	for (i = 0; i < MAX_POLY_DEGREE + 2; i++)
		mpz_init(tmp[i]);
	mpz_init(t);

	for (j = 0; j <= d1; j++)
		mpz_mul(tmp[j], p1->coeff[j], p2->coeff[d2]);
	pd = d1;

	for (i = d2; i-- > 0; ) {
		for (j = pd + 1; j-- > 0; )
			mpz_swap(tmp[j + 1], tmp[j]);
		for (j = 0; j <= d1; j++) {
			if (j == 0)
				mpz_mul(tmp[0], p1->coeff[0], p2->coeff[i]);
			else
				mpz_addmul(tmp[j], p1->coeff[j],
						p2->coeff[i]);
		}
		if (c != NULL && c->trace) {
			uint32 t;

			for (t = 0; t <= d + 1; t++)
				trace_val(c, "cpu", i, "product", t, tmp[t], q);
		}
		pd = d + 1;
		while (pd && mpz_sgn(tmp[pd]) == 0)
			pd--;
		if (pd <= d)
			continue;
		for (j = 0; j <= d; j++)
			mpz_submul(tmp[j], alg->coeff[j], tmp[pd]);
		if (c != NULL && c->trace)
			trace_val(c, "cpu", i, "prebar", pd, tmp[pd], q);
		if (c != NULL && c->trace) {
			uint32 t;

			for (t = 0; t <= d + 1; t++)
				trace_val(c, "cpu", i, "reduced", t, tmp[t], q);
		}
		pd--;
	}

	for (j = 0; j <= pd; j++)
		mpz_fdiv_r(out->coeff[j], tmp[j], q);
	out->degree = pd;
	if (raw != NULL) {
		for (j = 0; j <= pd; j++)
			mpz_set(raw->coeff[j], tmp[j]);
		raw->degree = pd;
	}

	for (i = 0; i < MAX_POLY_DEGREE + 2; i++)
		mpz_clear(tmp[i]);
	mpz_clear(t);
	return 0;
}

#endif /* HAVE_CUDA */

#ifdef HAVE_CUDA

/*------------------- setting up -------------------*/

/* below this the card loses to the host: the lift's first steps have a
   q of a few hundred bits, where a transform is all launch overhead */
#define GPU_MIN_Q_BITS 1000000

static uint64 next_pow2(uint64 x)
{
	uint64 p = 1;

	while (p < x) p <<= 1;
	return p;
}

static int build_tables(uint64 w, uint64 n, uint64 **d1, uint64 **d0)
{
	uint64 n1 = ((n >> 1) >> SPLIT_BITS) + 1;
	uint64 *h1 = (uint64 *)malloc(n1 * sizeof(uint64));
	uint64 *h0 = (uint64 *)malloc(SPLIT_SIZE * sizeof(uint64));
	uint64 step = powm_host(w, SPLIT_SIZE), i;
	int err = 0;

	if (h1 == NULL || h0 == NULL) { free(h1); free(h0); return -1; }
	h0[0] = 1;
	for (i = 1; i < SPLIT_SIZE; i++) h0[i] = mulm_host(h0[i - 1], w);
	h1[0] = 1;
	for (i = 1; i < n1; i++) h1[i] = mulm_host(h1[i - 1], step);

	if (*d1) { cudaFree(*d1); *d1 = NULL; }
	if (*d0) { cudaFree(*d0); *d0 = NULL; }
	if (cudaMalloc((void **)d1, n1 * sizeof(uint64)) != cudaSuccess ||
	    cudaMalloc((void **)d0, SPLIT_SIZE * sizeof(uint64))
							!= cudaSuccess ||
	    cudaMemcpy(*d1, h1, n1 * sizeof(uint64),
			cudaMemcpyHostToDevice) != cudaSuccess ||
	    cudaMemcpy(*d0, h0, SPLIT_SIZE * sizeof(uint64),
			cudaMemcpyHostToDevice) != cudaSuccess)
		err = -1;
	free(h1);
	free(h0);
	return err;
}

/* the transform follows q rather than the context: the buffers are
   sized for the largest modulus the lift will reach, but a step whose
   q is a sixteenth of that wants a sixteenth of the transform, not the
   whole thing. Only the twiddle tables change, and they are small */

static int set_transform(gctx *c, uint64 need_digits)
{
	uint64 n = next_pow2(need_digits);
	uint32 ln = 0;
	uint64 w, winv;

	while ((1ULL << ln) < n) ln++;
	if (n == c->n)
		return 0;
	if (ln > MAX_TRANSFORM_LOG)
		return -1;

	w = powm_host(GENERATOR, (PRIME - 1) / n);
	if (powm_host(w, n) != 1 || powm_host(w, n / 2) != PRIME - 1)
		return -1;
	winv = powm_host(w, n - 1);

	/* build_tables frees before it allocates, so from here the old
	   tables are gone. Say so first: a half built set left behind a
	   c->n that the next call would match and reuse */

	c->n = 0;
	c->ln = 0;
	if (build_tables(w, n, &c->twf1, &c->twf0)) return -1;
	if (build_tables(winv, n, &c->twi1, &c->twi0)) return -1;
	c->n = n;
	c->ln = ln;
	c->ninv = powm_host(n % PRIME, PRIME - 2);
	return 0;
}

void *sqrt_gpu_init(msieve_obj *obj, uint64 max_q_bits, uint32 degree)
{
	gctx *c;
	uint64 nums, bytes, gfree = 0, gtotal = 0, maxn;
	uint32 i;

	if (obj->nfs_args != NULL &&
	    strstr(obj->nfs_args, "gpu_sqrt=0") != NULL)
		return NULL;

	if (cudaSetDevice((int)gpu_pick(obj)) != cudaSuccess)
		return NULL;

	c = (gctx *)calloc(1, sizeof(gctx));
	if (c == NULL)
		return NULL;

	c->obj = obj;
	c->degree = degree;
	c->max_digits = (max_q_bits + DIGIT_BITS - 1) / DIGIT_BITS;

	/* a product runs to 2*max_digits+1; Barrett's first one runs to
	   2k+2 with k = digits(q)+1, which is six digits more */

	c->nc = 2 * c->max_digits + 8;
	c->nchunk = (c->nc + CARRY_CHUNK - 1) / CARRY_CHUNK;
	maxn = next_pow2(c->nc);

	/* exactness, which is a hard integer bound and not a margin: no
	   coefficient of a product may reach the modulus */

	if ((double)c->max_digits * 65535.0 * 65535.0 >= (double)PRIME ||
	    maxn > (1ULL << MAX_TRANSFORM_LOG)) {
		logprintf(obj, (char *)"square root: modulus too large for one "
				"NTT prime, using CPU\n");
		free(c);
		return NULL;
	}

	/* p1, one coefficient of p2, the Horner accumulators, Barrett's
	   q and mu and three scratch, and one more for the signed adds */

	nums = (degree + 1) + 1 + (degree + 2) + 5 + 1;
	bytes = nums * (c->nc + 8) * sizeof(uint16) +
		2 * maxn * sizeof(uint64);

	if (cudaMemGetInfo(&gfree, &gtotal) != cudaSuccess ||
	    (double)bytes > 0.90 * (double)gfree) {
		logprintf(obj, (char *)"square root: needs %.1f GB on the GPU, "
				"%.1f GB free, using CPU\n",
				(double)bytes / 1073741824.0,
				(double)gfree / 1073741824.0);
		free(c);
		return NULL;
	}

	if (cudaMalloc((void **)&c->A, maxn * sizeof(uint64)) != cudaSuccess ||
	    cudaMalloc((void **)&c->B, maxn * sizeof(uint64)) != cudaSuccess ||
	    cudaMalloc((void **)&c->dc0, c->nchunk * sizeof(uint64))
							!= cudaSuccess ||
	    cudaMalloc((void **)&c->dc1, c->nchunk * sizeof(uint64))
							!= cudaSuccess ||
	    cudaMalloc((void **)&c->dany, sizeof(int)) != cudaSuccess ||
	    cudaMalloc((void **)&c->dovf, sizeof(int)) != cudaSuccess ||
	    cudaMalloc((void **)&c->dword,
			sizeof(unsigned long long)) != cudaSuccess ||
	    cudaHostAlloc(&c->hstage, (c->nc + 8) * sizeof(uint16),
			cudaHostAllocDefault) != cudaSuccess)
		goto fail;

	for (i = 0; i <= degree; i++)
		if (gnum_alloc(c, &c->p1[i])) goto fail;
	for (i = 0; i <= degree + 1; i++)
		if (gnum_alloc(c, &c->tmp[i])) goto fail;
	if (gnum_alloc(c, &c->p2cur)) goto fail;
	if (gnum_alloc(c, &c->q)) goto fail;
	if (gnum_alloc(c, &c->mu)) goto fail;
	if (gnum_alloc(c, &c->t1)) goto fail;
	if (gnum_alloc(c, &c->t2)) goto fail;
	if (gnum_alloc(c, &c->t3)) goto fail;
	if (gnum_alloc(c, &c->hs1)) goto fail;

	/* mod(x)'s coefficients are the algebraic poly's scaled by powers
	   of its leading coefficient: a few hundred bits at the most, so
	   they get their own small buffers rather than full-width ones */

	for (i = 0; i <= degree; i++) {
		c->modc[i].ndig = 0;
		c->modc[i].neg = 0;
		if (cudaMalloc((void **)&c->modc[i].d,
				SMALL_MAX * sizeof(uint16)) != cudaSuccess)
			goto fail;
		cudaMemset(c->modc[i].d, 0, SMALL_MAX * sizeof(uint16));
	}

	mpz_init(c->qprev);
	c->n = 0;
	c->ok = 1;
	c->checked = 0;
	c->have_mu = 0;
	c->have_mod = 0;

	logprintf(obj, (char *)"square root: using GPU for the lift, %.1f GB of "
			"buffers for a modulus up to %" PRIu64 " bits\n",
			(double)bytes / 1073741824.0, max_q_bits);
	return c;

fail:
	logprintf(obj, (char *)"square root: GPU allocation failed, using CPU\n");
	sqrt_gpu_free(c);
	return NULL;
}

void sqrt_gpu_free(void *ctx)
{
	gctx *c = (gctx *)ctx;
	uint32 i;

	if (c == NULL)
		return;
	for (i = 0; i < MAX_POLY_DEGREE + 1; i++) {
		gnum_free(&c->p1[i]);
		gnum_free(&c->modc[i]);
	}
	for (i = 0; i < MAX_POLY_DEGREE + 2; i++)
		gnum_free(&c->tmp[i]);
	gnum_free(&c->p2cur);
	gnum_free(&c->q); gnum_free(&c->mu);
	gnum_free(&c->t1); gnum_free(&c->t2); gnum_free(&c->t3);
	gnum_free(&c->hs1);
	if (c->A) cudaFree(c->A);
	if (c->B) cudaFree(c->B);
	if (c->dc0) cudaFree(c->dc0);
	if (c->dc1) cudaFree(c->dc1);
	if (c->dany) cudaFree(c->dany);
	if (c->dovf) cudaFree(c->dovf);
	if (c->dword) cudaFree(c->dword);
	if (c->twf1) cudaFree(c->twf1);
	if (c->twf0) cudaFree(c->twf0);
	if (c->twi1) cudaFree(c->twi1);
	if (c->twi0) cudaFree(c->twi0);
	if (c->hstage) cudaFreeHost(c->hstage);
	if (c->have_mu || mpz_sgn(c->qprev) != 0) mpz_clear(c->qprev);
	free(c);
}

int sqrt_gpu_ok(void *ctx)
{
	gctx *c = (gctx *)ctx;

	return c != NULL && c->ok;
}

/*------------------- what the lift calls -------------------*/

static int usable(gctx *c, mpz_t q)
{
	return c != NULL && c->ok &&
		mpz_sizeinbase(q, 2) >= GPU_MIN_Q_BITS;
}

int sqrt_gpu_mod_q(void *ctx, mpz_poly_t *p, mpz_t q, mpz_poly_t *res)
{
	gctx *c = (gctx *)ctx;
	uint32 i;

	if (!usable(c, q))
		return -1;
	c->failat = 0;
	c->fa = c->fb = 0;
	if (set_transform(c, 2 * ((mpz_sizeinbase(q, 2) + DIGIT_BITS - 1) /
				DIGIT_BITS) + 8) || ensure_mu(c, q)) {
		c->ok = 0;
		logprintf(c->obj, (char *)"square root: GPU cannot hold this "
				"modulus, using CPU\n");
		return -1;
	}

	/* Barrett takes an x below b^(2k) and nothing larger. prod was
	   cut down once against the modulus the lift ends at, while q
	   restarts at the seed and squares, so until the last step or
	   two its coefficients are far longer than that. A size, not a
	   malfunction: decline and leave the context usable, because
	   the multiplies that follow are the ones worth having */

	for (i = 0; i <= p->degree; i++) {
		uint64 nd = (mpz_sizeinbase(p->coeff[i], 2) +
				DIGIT_BITS - 1) / DIGIT_BITS;

		if (nd > 2 * c->k)
			return -1;
	}

	for (i = 0; i <= p->degree; i++) {
		if (gnum_put(c, &c->tmp[0], p->coeff[i]) ||
		    barrett(c, &c->tmp[1], &c->tmp[0]) ||
		    gnum_get(c, res->coeff[i], &c->tmp[1])) {
			c->ok = 0;
			logprintf(c->obj, (char *)"square root: GPU reduction "
					"of prod failed at line %u (%" PRIu64 ", "
					"%" PRIu64 "), reverting to CPU\n",
					c->failat, c->fa, c->fb);
			return -1;
		}
	}
	i = p->degree;
	while (i && mpz_sgn(res->coeff[i]) == 0)
		i--;
	res->degree = i;
	return 0;
}

int sqrt_gpu_mul_mod_q(void *ctx, mpz_poly_t *p1, mpz_poly_t *p2,
			mpz_poly_t *alg, mpz_t q)
{
	gctx *c = (gctx *)ctx;
	uint32 i, d = alg->degree;
	mpz_poly_t ref, save, raw, dbg;
	int check, bad = 0;

	if (!usable(c, q))
		return -1;
	/* sized from this step's q, not the largest the lift reaches:
	   a step whose modulus is a sixteenth of the final one wants a
	   sixteenth of the transform, and Horner's products run to twice
	   the modulus plus the slack Barrett needs */

	c->failat = 0;
	c->fa = c->fb = 0;
	if (set_transform(c, 2 * ((mpz_sizeinbase(q, 2) + DIGIT_BITS - 1) /
			DIGIT_BITS) + 8) || ensure_mu(c, q)) {
		c->ok = 0;
		logprintf(c->obj, (char *)"square root: GPU cannot hold this "
				"modulus, using CPU\n");
		return -1;
	}

	if (!c->have_mod) {
		for (i = 0; i <= d; i++) {
			uint64 nd = (mpz_sizeinbase(alg->coeff[i], 2) +
					DIGIT_BITS - 1) / DIGIT_BITS;

			if (mpz_sgn(alg->coeff[i]) == 0) {
				c->modlen[i] = 0;
				c->modneg[i] = 0;
				continue;
			}
			if (nd > SMALL_MAX)
				{
					c->ok = 0;	/* cannot change: the
									   algebraic poly is fixed */
					logprintf(c->obj, (char *)"square root: mod(x) coefficient "
							"too large for the GPU, using CPU\n");
					return -1;
				}
			if (cudaMemcpy(c->modc[i].d,
					(const void *)mpz_limbs_read(
							alg->coeff[i]),
					nd * sizeof(uint16),
					cudaMemcpyHostToDevice) != cudaSuccess)
				return -1;
			c->modlen[i] = (uint32)nd;
			c->modneg[i] = (mpz_sgn(alg->coeff[i]) < 0);
		}
		c->have_mod = 1;
		logprintf(c->obj, (char *)"square root: mod(x) has degree %u and is %s\n",
				d, mpz_cmp_ui(alg->coeff[d], 1) == 0 ?
					(char *)"monic" : (char *)"not monic");
	}

	/* the first couple of times, keep what went in so the answer can
	   be checked against GMP doing the same loop */

	check = (c->checked < 2);
	/* the column by column trace is what found the one bug this
	   check has caught so far; it stays, behind a flag, because it
	   costs a readback per accumulator per column when it is on */

	c->trace = (c->checked == 0 && c->obj->nfs_args != NULL &&
			strstr(c->obj->nfs_args, "gpu_sqrt_trace") != NULL);
	if (check) {
		mpz_poly_init(&save);
		mpz_poly_init(&ref);
		mpz_poly_init(&raw);
		mpz_poly_init(&dbg);
		for (i = 0; i <= p1->degree; i++)
			mpz_set(save.coeff[i], p1->coeff[i]);
		save.degree = p1->degree;
	}

	if (horner(c, p1, p2, alg)) {
		c->ok = 0;
		logprintf(c->obj, (char *)"square root: GPU multiply failed "
				"at line %u (%" PRIu64 ", %" PRIu64 "), "
				"reverting to CPU\n",
				c->failat, c->fa, c->fb);
		if (check) {
			mpz_poly_free(&save); mpz_poly_free(&ref);
			mpz_poly_free(&raw); mpz_poly_free(&dbg);
		}
		return -1;
	}

	if (check) {
		for (i = 0; i <= p1->degree; i++)
			gnum_get(c, dbg.coeff[i], &c->tmp[i]);
		dbg.degree = p1->degree;
	}

	for (i = 0; i <= p1->degree; i++) {
		if (barrett(c, &c->tmp[i], &c->tmp[i]) ||
		    gnum_get(c, p1->coeff[i], &c->tmp[i])) {
			c->ok = 0;
			logprintf(c->obj, (char *)"square root: GPU reduction "
					"after multiply failed at line %u "
					"(%" PRIu64 ", %" PRIu64 "), reverting "
					"to CPU\n", c->failat, c->fa, c->fb);
			if (check) {
				mpz_poly_free(&save); mpz_poly_free(&ref);
				mpz_poly_free(&raw); mpz_poly_free(&dbg);
			}
			return -1;
		}
	}
	i = p1->degree;
	while (i && mpz_sgn(p1->coeff[i]) == 0)
		i--;
	p1->degree = i;

	if (check) {
		reference_mul_mod(&ref, &save, p2, alg, q, &raw, c);

		/* the product and the reduction fail differently, and
		   the log should not make us guess which one did */

		{
			uint32 at;
			int rawbad = (raw.degree != dbg.degree);
			mpz_t ra, da;

			/* congruence, not equality: horner reduces tmp[pd] part
			   way through and the reference does not, so the two
			   only ever agree mod q */

			mpz_init(ra);
			mpz_init(da);
			for (at = 0; !rawbad && at <= raw.degree; at++) {
				mpz_fdiv_r(ra, raw.coeff[at], q);
				mpz_fdiv_r(da, dbg.coeff[at], q);
				if (mpz_cmp(ra, da) != 0)
					rawbad = 1;
			}
			mpz_clear(ra);
			mpz_clear(da);
			logprintf(c->obj, (char *)"square root: before reducing, mod q "
					"GPU and CPU %s (gpu degree %u, cpu %u)\n",
					rawbad ? (char *)"differ" : (char *)"agree",
					dbg.degree, raw.degree);
		}
		if (ref.degree != p1->degree)
			bad = 1;
		for (i = 0; !bad && i <= ref.degree; i++)
			if (mpz_cmp(ref.coeff[i], p1->coeff[i]) != 0)
				bad = 1;
		if (bad) {

			/* which coefficient, and in what way. A difference that is
			   a multiple of q means the reduction went wrong and the
			   product did not; anything else is the other way round */

			{
				mpz_t diff;
				uint32 at = 0;
				int modq = 0;

				mpz_init(diff);
				for (at = 0; at <= ref.degree &&
						at <= p1->degree; at++)
					if (mpz_cmp(ref.coeff[at], p1->coeff[at]) != 0)
						break;
				if (at <= ref.degree && at <= p1->degree) {
					mpz_sub(diff, ref.coeff[at], p1->coeff[at]);
					modq = (mpz_divisible_p(diff, q) != 0);
					logprintf(c->obj, (char *)"square root: GPU and CPU "
						"differ at coefficient %u of %u, ref %" PRIu64
						" bits, gpu %" PRIu64 " bits, diff %" PRIu64
						" bits, %s a multiple of q\n",
						at, ref.degree,
						(uint64)mpz_sizeinbase(ref.coeff[at], 2),
						(uint64)mpz_sizeinbase(p1->coeff[at], 2),
						(uint64)mpz_sizeinbase(diff, 2),
						modq ? (char *)"is" : (char *)"is not");
				}
				else {
					logprintf(c->obj, (char *)"square root: GPU and CPU "
						"differ in degree, ref %u, gpu %u\n",
						ref.degree, p1->degree);
				}
				mpz_clear(diff);
			}
			logprintf(c->obj, (char *)"square root: GPU and CPU "
					"disagree, using CPU for the rest "
					"of the lift\n");
			for (i = 0; i <= ref.degree; i++)
				mpz_set(p1->coeff[i], ref.coeff[i]);
			p1->degree = ref.degree;
			c->ok = 0;
		}
		else {
			c->checked++;
			if (c->checked == 2)
				logprintf(c->obj, (char *)"square root: GPU lift "
					"verified against CPU, continuing "
					"on the GPU\n");
		}
		mpz_poly_free(&save);
		mpz_poly_free(&ref);
		mpz_poly_free(&raw);
		mpz_poly_free(&dbg);
	}
	return 0;
}

#endif /* HAVE_CUDA */
