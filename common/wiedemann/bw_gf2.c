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

/* Small dense GF(2) linear algebra on v_t, shared by the Wiedemann
   stages. Deliberately free of any dependency on the matrix layer or
   on logging, so it can be built and tested on its own. */

#include "wiedemann.h"

/*-----------------------------------------------------------------------*/
uint32 bw_v_parity(v_t a, v_t b) {

	/* the GF(2) dot product: parity of the bitwise AND */

	uint64 acc = 0;
	uint32 i;

	for (i = 0; i < VWORDS; i++)
		acc ^= a.w[i] & b.w[i];

	acc ^= acc >> 32;
	acc ^= acc >> 16;
	acc ^= acc >> 8;
	acc ^= acc >> 4;
	acc ^= acc >> 2;
	acc ^= acc >> 1;
	return (uint32)(acc & 1);
}

/*-----------------------------------------------------------------------*/
uint32 bw_v_popcount(v_t a) {

	uint32 i, j, count = 0;

	for (i = 0; i < VWORDS; i++) {
		uint64 w = a.w[i];

		for (j = 0; j < 64; j++)
			count += (uint32)((w >> j) & 1);
	}
	return count;
}

/*-----------------------------------------------------------------------*/
uint32 bw_v_lowest_set(v_t a, uint32 *bit) {

	uint32 i, j;

	for (i = 0; i < VWORDS; i++) {
		if (a.w[i] == 0)
			continue;
		for (j = 0; j < 64; j++) {
			if (a.w[i] & ((uint64)1 << j)) {
				*bit = 64 * i + j;
				return 1;
			}
		}
	}
	return 0;
}

/*-----------------------------------------------------------------------*/
void bw_v_set_bit(v_t *a, uint32 bit) {

	a->w[bit >> 6] |= (uint64)1 << (bit & 63);
}

/*-----------------------------------------------------------------------*/
uint32 bw_gf2_nullspace(v_t *rows, uint32 num_rows, v_t *out) {

	/* Basis for the u with <rows[i], u> = 0 for every i, where a row
	   and u are both VBITS bits wide. num_rows must be at most VBITS;
	   in mksol it is always POST_LANCZOS_ROWS = VBITS - 16, which is
	   what lets a column of the input sit in a v_t.

	   Column echelon form, carrying along a record of the
	   combinations used. Whenever a column reduces to zero, whatever
	   combination produced it is a nullspace vector. At least
	   VBITS - num_rows always come back. */

	v_t *col = (v_t *)xmalloc(VBITS * sizeof(v_t));
	v_t *comb = (v_t *)xmalloc(VBITS * sizeof(v_t));
	uint32 *pivot_row = (uint32 *)xmalloc(VBITS * sizeof(uint32));
	uint32 *pivot_col = (uint32 *)xmalloc(VBITS * sizeof(uint32));
	uint32 num_pivots = 0;
	uint32 num_found = 0;
	uint32 c, i, j;

	for (c = 0; c < VBITS; c++) {
		col[c] = v_zero;
		comb[c] = v_zero;
		bw_v_set_bit(comb + c, c);
		for (i = 0; i < num_rows; i++) {
			if (v_bitset(rows[i], c))
				bw_v_set_bit(col + c, i);
		}
	}

	for (c = 0; c < VBITS; c++) {
		uint32 bit;

		for (j = 0; j < num_pivots; j++) {
			if (v_bitset(col[c], pivot_row[j])) {
				col[c] = v_xor(col[c], col[pivot_col[j]]);
				comb[c] = v_xor(comb[c], comb[pivot_col[j]]);
			}
		}

		if (bw_v_lowest_set(col[c], &bit)) {
			pivot_row[num_pivots] = bit;
			pivot_col[num_pivots] = c;
			num_pivots++;
		}
		else {
			out[num_found++] = comb[c];
		}
	}

	free(col);
	free(comb);
	free(pivot_row);
	free(pivot_col);
	return num_found;
}
