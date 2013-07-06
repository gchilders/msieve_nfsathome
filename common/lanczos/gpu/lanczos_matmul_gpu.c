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

#include "lanczos_gpu.h"

/*-------------------------------------------------------------------*/
static void mul_unpacked(packed_matrix_t *matrix,
			  uint64 *x, uint64 *b) 
{
	uint32 ncols = matrix->ncols;
	uint32 num_dense_rows = matrix->num_dense_rows;
	la_col_t *A = matrix->unpacked_cols;
	uint32 i, j;

	memset(b, 0, ncols * sizeof(uint64));
	
	for (i = 0; i < ncols; i++) {
		la_col_t *col = A + i;
		uint32 *row_entries = col->data;
		uint64 tmp = x[i];

		for (j = 0; j < col->weight; j++) {
			b[row_entries[j]] ^= tmp;
		}
	}

	if (num_dense_rows) {
		for (i = 0; i < ncols; i++) {
			la_col_t *col = A + i;
			uint32 *row_entries = col->data + col->weight;
			uint64 tmp = x[i];
	
			for (j = 0; j < num_dense_rows; j++) {
				if (row_entries[j / 32] & 
						((uint32)1 << (j % 32))) {
					b[j] ^= tmp;
				}
			}
		}
	}
}

/*-------------------------------------------------------------------*/
static void mul_trans_unpacked(packed_matrix_t *matrix,
				uint64 *x, uint64 *b) 
{
	uint32 ncols = matrix->ncols;
	uint32 num_dense_rows = matrix->num_dense_rows;
	la_col_t *A = matrix->unpacked_cols;
	uint32 i, j;

	for (i = 0; i < ncols; i++) {
		la_col_t *col = A + i;
		uint32 *row_entries = col->data;
		uint64 accum = 0;

		for (j = 0; j < col->weight; j++) {
			accum ^= x[row_entries[j]];
		}
		b[i] = accum;
	}

	if (num_dense_rows) {
		for (i = 0; i < ncols; i++) {
			la_col_t *col = A + i;
			uint32 *row_entries = col->data + col->weight;
			uint64 accum = b[i];
	
			for (j = 0; j < num_dense_rows; j++) {
				if (row_entries[j / 32] &
						((uint32)1 << (j % 32))) {
					accum ^= x[j];
				}
			}
			b[i] = accum;
		}
	}
}

/*-------------------------------------------------------------------*/
void matrix_extra_init(msieve_obj *obj, packed_matrix_t *p) {

}

/*-------------------------------------------------------------------*/
void matrix_extra_free(packed_matrix_t *p) {

}

/*-------------------------------------------------------------------*/
void mul_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	uint64 *x = (uint64 *)x_in;
	uint64 *b = (uint64 *)b_in;

	mul_unpacked(A, x, b);
}

/*-------------------------------------------------------------------*/
void mul_trans_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	uint64 *x = (uint64 *)x_in;
	uint64 *b = (uint64 *)b_in;

	mul_trans_unpacked(A, x, b);
}
