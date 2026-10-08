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

/* stage entry points. Each returns 0 on success and -1 if it stopped
   early (interrupt, or a missing input from an earlier stage) */

int32 bw_krylov(msieve_obj *obj, packed_matrix_t *matrix,
			bw_params_t *params, uint32 max_ncols);

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
