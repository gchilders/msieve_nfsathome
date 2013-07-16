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

#ifndef _COMMON_LANCZOS_LANCZOS_H_
#define _COMMON_LANCZOS_LANCZOS_H_

#include <common.h>

#ifdef __cplusplus
extern "C" {
#endif

/* for matrices of dimension exceeding MIN_POST_LANCZOS_DIM,
   the first POST_LANCZOS_ROWS rows are handled in a separate
   Gauss elimination phase after the Lanczos iteration
   completes. This means the lanczos code will produce about
   64 - POST_LANCZOS_ROWS dependencies on average. 
   
   The code will still work if POST_LANCZOS_ROWS is 0, but I 
   don't know why you would want to do that. The first rows are 
   essentially completely dense, and removing them from the main 
   Lanczos iteration greatly reduces the amount of arithmetic 
   in a matrix multiply, as well as the memory footprint of 
   the matrix */

#define POST_LANCZOS_ROWS 48
#define MIN_POST_LANCZOS_DIM 10000

/* routines for cache-efficient multiplication of
   sparse matrices */

/* the smallest matrix size that will be converted 
   to packed format */

#define MIN_NROWS_TO_PACK 30000

/* the number of moderately dense rows that are
   packed less tightly */

#define NUM_MEDIUM_ROWS 3000

/* structure representing a nonzero element of
   the matrix after packing into block format. 
   The two fields are the row and column offsets
   from the top left corner of the block */

typedef struct {
	uint16 row_off;
	uint16 col_off;
} entry_idx_t;

/* struct representing one block */

typedef struct {
	uint32 num_entries;       /* number of nonzero matrix entries */
	union {
		entry_idx_t *entries;     /* nonzero entries */
		uint16 *med_entries;	  /* nonzero entries for medium dense rows */
	} d;
} packed_block_t;

#define MAX_THREADS 32
#define MIN_NROWS_TO_THREAD 200000

/* struct representing a packed matrix */

typedef struct packed_matrix_t {
	uint32 nrows;
	uint32 max_nrows;
	uint32 start_row;

	uint32 ncols;
	uint32 max_ncols;
	uint32 start_col;

	uint32 num_dense_rows;
	uint32 num_threads;
	uint32 vsize;

	la_col_t *unpacked_cols;  /* used if no packing takes place */

	/* used for block matrix multiplies */

	uint32 block_size;
	uint32 num_block_rows;
	uint32 num_block_cols;

	uint32 superblock_size;  /* in units of blocks */
	uint32 num_superblock_rows;
	uint32 num_superblock_cols;

	uint32 first_block_size;/* block size for the smallest row numbers */

	uint64 **dense_blocks;  /* for holding dense matrix rows; 
				   dense_blocks[i] holds the i_th batch of
				   64 matrix rows */
	packed_block_t *blocks; /* sparse part of matrix, in block format */

	void * extra; /* implementation-specific stuff */

#ifdef HAVE_MPI
	uint32 mpi_size;
	uint32 mpi_nrows;
	uint32 mpi_ncols;
	uint32 mpi_la_row_rank;
	uint32 mpi_la_col_rank;
	MPI_Comm mpi_la_row_grid;
	MPI_Comm mpi_la_col_grid;

	/* needed on root node only */
	int32 col_counts[MAX_MPI_GRID_DIM];
	int32 col_offsets[MAX_MPI_GRID_DIM]; 
	int32 row_counts[MAX_MPI_GRID_DIM];
	int32 row_offsets[MAX_MPI_GRID_DIM];
	int32 subcol_counts[MAX_MPI_GRID_DIM];
	int32 subcol_offsets[MAX_MPI_GRID_DIM];    

	uint32 nsubcols;
#endif

} packed_matrix_t;

void packed_matrix_init(msieve_obj *obj, 
			packed_matrix_t *packed_matrix,
			la_col_t *A, 
			uint32 nrows, uint32 max_nrows, uint32 start_row, 
			uint32 ncols, uint32 max_ncols, uint32 start_col, 
			uint32 num_dense_rows, uint32 first_block_size);

void packed_matrix_free(packed_matrix_t *packed_matrix);

void matrix_extra_init(msieve_obj *obj, 
			packed_matrix_t *packed_matrix);

void matrix_extra_free(packed_matrix_t *packed_matrix);

size_t packed_matrix_sizeof(packed_matrix_t *packed_matrix);

/* top-level calls for matrix multiplies */

void mul_MxN_Nx64(packed_matrix_t *A, 
			void *x, void *scratch);

void mul_sym_NxN_Nx64(packed_matrix_t *A, void *x, 
			void *b, void *scratch);

/* easy base cases for small problems */

void mul_unpacked(packed_matrix_t *matrix,
			  uint64 *x, uint64 *b); 

void mul_trans_unpacked(packed_matrix_t *matrix,
				uint64 *x, uint64 *b);

/* implementation-specifc matrix-vector product */

void mul_core(packed_matrix_t *A, void *x, void *b);
void mul_trans_core(packed_matrix_t *A, void *x, void *b);

#ifdef HAVE_MPI
void global_xor(void *send_buf, void *recv_buf, 
		uint32 bufsize, uint32 mpi_nodes, 
		uint32 mpi_rank, MPI_Comm comm);

void global_chunk_info(uint32 total_size, uint32 num_nodes, 
		uint32 my_id, uint32 *chunk_size, uint32 *chunk_start);

void global_allgather(void *send_buf, void *recv_buf, 
                        uint32 bufsize, uint32 mpi_nodes, 
                        uint32 mpi_rank, MPI_Comm comm);

void global_xor_scatter(void *send_buf, void *recv_buf, 
			void *scratch, uint32 bufsize, 
			uint32 mpi_nodes, uint32 mpi_rank, 
			MPI_Comm comm);
#endif

/* top-level calls for vector-vector operations */

void *v_alloc(uint32 n, void *extra);
void v_free(void *v);
void v_copyin(void *dest, uint64 *src, uint32 n);
void v_copy(void *dest, void *src, uint32 n);
void v_copyout(uint64 *dest, void *src, uint32 n);
void v_clear(void *v, uint32 n);
void v_xor(void *dest, void *src, uint32 n);
void v_mask(void *v, uint64 mask, uint32 n);

void v_mul_Nx64_64x64_acc(packed_matrix_t *A, void *v, uint64 *x, 
			void *y, uint32 n);

void v_mul_64xN_Nx64(packed_matrix_t *A, void *x, void *y, 
			uint64 *xy, uint32 n);

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_LANCZOS_LANCZOS_H_ */
