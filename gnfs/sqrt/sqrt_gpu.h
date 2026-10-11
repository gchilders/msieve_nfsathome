/*--------------------------------------------------------------------
The algebraic square root's lift, on a GPU.

The Newton lift is multiplies and reductions of numbers that reach
fifteen gigabits on a large job, and GMP does both on one core apiece:
mpz_poly_mul spreads its products over the coefficients and mpz_poly_mod_q
over its own, so a 46-thread machine runs the lift on seven or eighteen
of them. A Goldilocks NTT with Barrett reduction measures about ten times
faster end to end on an H200.

Nothing here is required. Every entry point returns failure rather than
doing anything when there is no CUDA build, no usable card, or not enough
memory on it, and the caller falls back to the code that was already
there. The CPU path is unchanged in every case, which is the point: this
runs for days at a time and a wrong square root is found hours later or
not at all.
--------------------------------------------------------------------*/

#ifndef _GNFS_SQRT_SQRT_GPU_H_
#define _GNFS_SQRT_SQRT_GPU_H_

#include "sqrt.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HAVE_CUDA

/* No card to ask, so everything declines and the compiler folds the
   branch away: a build without CUDA compiles to the code that was
   always there, with no call and no #ifdef at any of the call sites.
   The runtime fallback below is needed in either build anyway -- a
   CUDA binary still runs on machines with no GPU, or too small a one
   -- so the call sites would look like this regardless. */

static INLINE void *sqrt_gpu_init(msieve_obj *obj, uint64 max_q_bits,
				uint32 degree) {
	(void)obj; (void)max_q_bits; (void)degree;
	return NULL;
}
static INLINE void sqrt_gpu_free(void *ctx) { (void)ctx; }

static INLINE int sqrt_gpu_mul_mod_q(void *ctx, mpz_poly_t *p1,
				mpz_poly_t *p2, mpz_poly_t *alg, mpz_t q) {
	(void)ctx; (void)p1; (void)p2; (void)alg; (void)q;
	return -1;
}
static INLINE int sqrt_gpu_poly_mul(void *ctx, mpz_poly_t *p1,
			mpz_poly_t *p2, mpz_poly_t *alg) {
	(void)ctx; (void)p1; (void)p2; (void)alg;
	return -1;
}

static INLINE int sqrt_gpu_mod_q(void *ctx, mpz_poly_t *p, mpz_t q,
				mpz_poly_t *res) {
	(void)ctx; (void)p; (void)q; (void)res;
	return -1;
}
static INLINE int sqrt_gpu_ok(void *ctx) { (void)ctx; return 0; }

#else

/* Build a context for a lift whose final modulus reaches max_q_bits,
   over polynomials of the given degree. Returns NULL -- having said why
   in the log -- when the GPU cannot or should not be used, which the
   caller must treat as "use the CPU path", not as an error.

   The context holds the transform buffers and every operand the lift's
   inner loop needs, so that the thirty-six to forty-nine products of one
   poly_mul and the reduction after them never cross PCIe. That is most
   of what makes this worth doing; a card that cannot hold them is a card
   this declines. */

void *sqrt_gpu_init(msieve_obj *obj, uint64 max_q_bits, uint32 degree);

void sqrt_gpu_free(void *ctx);

/* p1 <- (p1 * p2 mod alg) mod q, which is mpz_poly_mul followed by
   mpz_poly_mod_q, fused: the accumulators stay on the card between
   them. Returns 0 if it did the work, nonzero if the caller should.

   q changes once per lift step, and the Barrett parameter it needs is
   derived from the previous step's by Newton rather than divided for
   again -- a division at this size costs more than every reduction it
   would serve. Passing a q that is not the square of the last one is
   allowed and simply falls back to computing it the slow way. */

int sqrt_gpu_mul_mod_q(void *ctx, mpz_poly_t *p1, mpz_poly_t *p2,
			mpz_poly_t *alg, mpz_t q);

/* res <- p mod q, coefficient by coefficient. Same contract. */

/* p1 *= p2 mod alg(x), with no integer modulus: what the relation
   product tree does at every node. Declines when the operands are too
   small to be worth a transform or too large for the context, and the
   caller then runs mpz_poly_mul as before. NOT thread safe */

int sqrt_gpu_poly_mul(void *ctx, mpz_poly_t *p1, mpz_poly_t *p2,
			mpz_poly_t *alg);

int sqrt_gpu_mod_q(void *ctx, mpz_poly_t *p, mpz_t q, mpz_poly_t *res);

/* Whether the context is still trustworthy. The first few lift steps
   are small enough to run on both paths and compare, so that a wrong
   answer is caught in the second it takes rather than in the hours
   before the dependency fails; if they ever disagree the context turns
   itself off and everything after it runs on the CPU. */

int sqrt_gpu_ok(void *ctx);

#endif /* HAVE_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* _GNFS_SQRT_SQRT_GPU_H_ */
