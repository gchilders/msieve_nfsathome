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

/* Block Wiedemann, offered alongside block Lanczos.

   Where Lanczos applies A and then A^T every iteration, Wiedemann only
   ever applies A. That buys two things: the matrix needs only one copy
   on the card, and the Krylov stage splits into sequences that do not
   talk to each other at all. It costs about 1.5x as many matrix-vector
   products, and it needs a linear generator (lingen), which Lanczos
   does not.

   Three stages, each able to run on its own so that the middle one can
   sit on a different machine from the other two:

     krylov   a_i = x^T A^i y, for i < L.  Only forward products.
     lingen   find F with A(X) F(X) = 0 mod X^L.  No matrix at all.
     mksol    w = sum_k F_k A^k y, the solution.  Only forward products.

   Everything here is written against the matrix-multiply interface in
   ../lanczos/lanczos.h, which has one CPU and one GPU implementation
   chosen at link time, so this code compiles unchanged into both. */

#ifndef _COMMON_WIEDEMANN_WIEDEMANN_H_
#define _COMMON_WIEDEMANN_WIEDEMANN_H_

#include "../lanczos/lanczos.h"

#ifdef __cplusplus
extern "C" {
#endif

/* which stages to run. Running them separately leaves files behind for
   the next stage to pick up; running them together is only sensible on
   a matrix small enough that lingen is cheap */

enum {
	BW_STAGE_ALL = 0,
	BW_STAGE_KRYLOV,
	BW_STAGE_LINGEN,
	BW_STAGE_MKSOL
};

/* m and n are the two block widths of the method. Both are counted in
   units of VBITS, because a VBITS-wide block is what the matrix-vector
   product and the outer product natively handle.

   The Krylov sequence has length L = N/m + N/n, so raising either
   shortens it; but lingen costs about (m+n)^2 * N, so raising either
   makes the middle stage much more expensive. n_mult is also the number
   of independent sequences, which is where cross-GPU parallelism comes
   from. Both default to 1 and the balance has to be measured, not
   guessed. */

typedef struct {
	uint32 m_mult;		/* m = m_mult * VBITS */
	uint32 n_mult;		/* n = n_mult * VBITS, also the sequence count */
	uint32 seq;		/* which sequence this process computes */
	uint32 stage;
	uint32 seed1, seed2;	/* x and y come from here, so pinning these
				   makes a run reproducible */
} bw_params_t;

/* the on-disk Krylov sequence. One header, then num_terms records of
   m_mult * VBITS v_t each: a_i is m rows of seq_width bits, and a row
   of VBITS bits is exactly one v_t */

#define BW_SEQ_MAGIC 0x31535742		/* "BWS1" */

typedef struct {
	uint32 magic;
	uint32 vbits;
	uint32 m;		/* total, not in units of VBITS */
	uint32 n;
	uint32 seq;
	uint32 seq_width;	/* columns this sequence carries, = VBITS */
	uint32 ncols;		/* N, the padded square dimension */
	uint32 num_terms;	/* terms actually present */
} bw_seq_header_t;

#define BW_CHK_MAGIC 0x314b5742		/* "BWK1" */

/* the generator lingen produces and mksol consumes. One header, then
   degree+1 coefficient blocks of VBITS v_t each */

#define BW_GEN_MAGIC 0x31465742		/* "BWF1" */

typedef struct {
	uint32 magic;
	uint32 vbits;
	uint32 m;
	uint32 n;
	uint32 ncols;
	uint32 degree;		/* coefficients are k = 0 .. degree */
} bw_gen_header_t;

/* stage entry points. Each returns 0 on success and -1 if it stopped
   early (interrupt, or a missing input from an earlier stage) */

int32 bw_krylov(msieve_obj *obj, packed_matrix_t *matrix,
			bw_params_t *params, uint32 max_ncols);

/* Reads the Krylov sequence, writes <savefile>.bw.f. The generator is
   checked against the sequence before it is written, so a success here
   means the relation mksol depends on actually holds */

int32 bw_lingen(msieve_obj *obj, bw_params_t *params, uint32 max_ncols);

/* On success *solution_out is an aligned_malloc'd array of max_ncols
   v_t, one per matrix column, with dependency d in bit d. The caller
   owns it. post_lanczos_matrix may be NULL */

int32 bw_mksol(msieve_obj *obj, packed_matrix_t *matrix,
			bw_params_t *params, uint32 max_ncols,
			v_t *post_lanczos_matrix,
			v_t **solution_out, uint32 *num_deps_found);

/* A matrix of GF(2) polynomials, held coefficient-major: coefficient k
   is a whole dense nrows x ncols bit matrix, rows padded to a uint64
   boundary. Multiplying two of these is a polynomial product whose
   coefficients are matrices, so Karatsuba runs in the degree dimension
   over a plain GF(2) matrix product and never needs the GF(2^w)
   arithmetic a transform would. In lingen_matpoly.c */

typedef struct {
	uint32 nrows;
	uint32 ncols;
	uint32 rwords;		/* uint64 per row */
	uint32 len;		/* coefficients */
	uint64 *data;		/* [k][row][word] */
} bmp_t;

void bmp_init(bmp_t *p, uint32 nrows, uint32 ncols, uint32 len);
void bmp_free(bmp_t *p);
uint64 *bmp_coeff(bmp_t *p, uint32 k);

/* c = a * b; c must already be len a->len + b->len - 1 and zeroed.
   bmp_mul_school is the same product done the obvious way, kept as the
   base of the recursion and as something to test against */

void bmp_mul(bmp_t *c, const bmp_t *a, const bmp_t *b);
void bmp_mul_school(bmp_t *c, const bmp_t *a, const bmp_t *b);

/* The two halves of lingen, exposed so the recursion can be checked
   against the base case on small random inputs rather than only on a
   real sequence. Both take G (m x b, known to T coefficients) and the
   running column degrees, and produce a basis pi with G pi = 0 mod
   X^T. delta is updated in place */

void quadratic_basis(const bmp_t *G, uint32 T, uint32 *delta,
			bmp_t *pi_out);
void recursive_basis(const bmp_t *G, uint32 T, uint32 *delta,
			bmp_t *pi_out);

/* Small dense GF(2) helpers on v_t, in bw_gf2.c. That file depends on
   nothing but this header, so it builds and tests on its own.

   bw_gf2_nullspace finds a basis for the u with <rows[i], u> = 0 for
   all i < num_rows, which must be at most VBITS. out needs room for
   VBITS entries; returns how many were found, always at least
   VBITS - num_rows */

uint32 bw_v_parity(v_t a, v_t b);
uint32 bw_v_popcount(v_t a);
uint32 bw_v_lowest_set(v_t a, uint32 *bit);
void bw_v_set_bit(v_t *a, uint32 bit);
uint32 bw_gf2_nullspace(v_t *rows, uint32 num_rows, v_t *out);

/* external interface, deliberately identical to block_lanczos() so the
   two are interchangeable at the call site in gnfs/gf2.c */

uint64 *block_wiedemann(msieve_obj *obj,
			uint32 nrows, uint32 max_nrows, uint32 start_row,
			uint32 num_dense_rows,
			uint32 ncols, uint32 max_ncols, uint32 start_col,
			la_col_t *B, uint32 *num_deps_found);

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_WIEDEMANN_WIEDEMANN_H_ */
