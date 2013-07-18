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

#include "lanczos.h"

/*-------------------------------------------------------------------*/
void mul_unpacked(packed_matrix_t *matrix,
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
void mul_trans_unpacked(packed_matrix_t *matrix,
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
static int compare_row_off(const void *x, const void *y) {
	entry_idx_t *xx = (entry_idx_t *)x;
	entry_idx_t *yy = (entry_idx_t *)y;

	if (xx->row_off > yy->row_off)
		return 1;
	if (xx->row_off < yy->row_off)
		return -1;

	return (int)xx->col_off - (int)yy->col_off;
}

/*--------------------------------------------------------------------*/
static void pack_med_block(packed_block_t *b)
{
	uint32 j, k, m;
	uint16 *med_entries;
	entry_idx_t *e;

	/* convert the first block in the stripe to a somewhat-
	   compressed format. Entries in this first block are stored 
	   by row, and all rows are concatenated into a single 
	   16-bit array */

	e = b->d.entries;
	qsort(e, (size_t)b->num_entries, 
			sizeof(entry_idx_t), compare_row_off);
	for (j = k = 1; j < b->num_entries; j++) {
		if (e[j].row_off != e[j-1].row_off)
			k++;
	}

	/* we need a 16-bit word for each element and two more
	   16-bit words at the start of each of the k packed
	   arrays making up med_entries. The first extra word
	   gives the row number and the second gives the number
	   of entries in that row. We also need a few extra words 
	   at the array end because the multiply code uses a 
	   software pipeline and would fetch off the end of 
	   med_entries otherwise */

	med_entries = (uint16 *)xmalloc((b->num_entries + 
					2 * k + 8) * sizeof(uint16));
	j = k = 0;
	while (j < b->num_entries) {
		for (m = 0; j + m < b->num_entries; m++) {
			if (m > 0 && e[j+m].row_off != e[j+m-1].row_off)
				break;
			med_entries[k+m+2] = e[j+m].col_off;
		}
		med_entries[k] = e[j].row_off;
		med_entries[k+1] = m;
		j += m;
		k += m + 2;
	}
	med_entries[k] = med_entries[k+1] = 0;
	free(b->d.entries);
	b->d.med_entries = med_entries;
}

/*--------------------------------------------------------------------*/
static void pack_matrix_core(packed_matrix_t *p, la_col_t *A)
{
	uint32 i, j, k;
	uint32 dense_row_blocks;
	packed_block_t *curr_stripe;

	uint32 ncols = p->ncols;
	uint32 block_size = p->block_size;
	uint32 num_block_rows = p->num_block_rows;
	uint32 num_block_cols = p->num_block_cols;
	uint32 num_dense_rows = p->num_dense_rows;
	uint32 first_block_size = p->first_block_size;

	/* pack the dense rows 64 at a time */

	dense_row_blocks = (num_dense_rows + 63) / 64;
	if (dense_row_blocks) {
		p->dense_blocks = (uint64 **)xmalloc(dense_row_blocks *
						sizeof(uint64 *));
		for (i = 0; i < dense_row_blocks; i++) {
			p->dense_blocks[i] = (uint64 *)xmalloc(ncols *
							sizeof(uint64));
		}

		for (i = 0; i < ncols; i++) {
			la_col_t *c = A + i;
			uint32 *src = c->data + c->weight;
			for (j = 0; j < dense_row_blocks; j++) {
				p->dense_blocks[j][i] = 
						(uint64)src[2 * j + 1] << 32 |
						(uint64)src[2 * j];
			}
		}
	}

	/* allocate blocks in row-major order; a 'stripe' is
	   a vertical column of blocks. The first block in each
	   column has first_block_size rows instead of block_size */

	p->blocks = curr_stripe = (packed_block_t *)xcalloc(
						(size_t)num_block_rows *
						        num_block_cols,
						sizeof(packed_block_t));

	/* we convert the sparse part of the matrix to packed
	   format one stripe at a time. This limits the worst-
	   case memory use of the packing process */

	for (i = 0; i < num_block_cols; i++, curr_stripe++) {

		uint32 curr_cols = MIN(block_size, ncols - i * block_size);
		packed_block_t *b;

		/* count the number of nonzero entries in each block */

		for (j = 0; j < curr_cols; j++) {
			la_col_t *c = A + i * block_size + j;
			uint32 limit = first_block_size;

			for (k = 0, b = curr_stripe; k < c->weight; k++) {
				uint32 index = c->data[k];

				while (index >= limit) {
					b += num_block_cols;
					limit += block_size;
				}
				b->num_entries++;
			}
		}

		/* concatenate the nonzero elements of the matrix
		   columns corresponding to this stripe.
		   
		   We technically can combine the previous pass through
		   the columns with this pass, but on some versions of
		   libc the number of reallocations causes an incredible
		   slowdown */

		for (j = 0, b = curr_stripe; j < num_block_rows; 
						j++, b += num_block_cols) {
			b->d.entries = (entry_idx_t *)xmalloc(
						b->num_entries *
						sizeof(entry_idx_t));
			b->num_entries = 0;
		}

		for (j = 0; j < curr_cols; j++) {
			la_col_t *c = A + i * block_size + j;
			uint32 limit = first_block_size;
			uint32 start_row = 0;

			for (k = 0, b = curr_stripe; k < c->weight; k++) {
				entry_idx_t *e;
				uint32 index = c->data[k];

				while (index >= limit) {
					b += num_block_cols;
					start_row = limit;
					limit += block_size;
				}

				e = b->d.entries + b->num_entries++;
				e->row_off = index - start_row;
				e->col_off = j;
			}

			free(c->data);
			c->data = NULL;
		}

		pack_med_block(curr_stripe);
	}
}

/*-------------------------------------------------------------------*/
void packed_matrix_init(msieve_obj *obj,
			packed_matrix_t *p, la_col_t *A,
			uint32 nrows, uint32 max_nrows, uint32 start_row, 
			uint32 ncols, uint32 max_ncols, uint32 start_col, 
			uint32 num_dense_rows, uint32 first_block_size) {

	uint32 i;
	uint32 block_size;
	uint32 superblock_size;
	uint32 num_threads;

	/* initialize */

	p->unpacked_cols = A;
	p->nrows = nrows;
	p->max_nrows = max_nrows;
	p->start_row = start_row;
	p->ncols = ncols;
	p->max_ncols = max_ncols;
	p->start_col = start_col;
	p->num_dense_rows = num_dense_rows;
	p->num_threads = 1;
#ifdef HAVE_MPI
	p->mpi_size = obj->mpi_size;
	p->mpi_nrows = obj->mpi_nrows;
	p->mpi_ncols = obj->mpi_ncols;
	p->mpi_la_row_rank = obj->mpi_la_row_rank;
	p->mpi_la_col_rank = obj->mpi_la_col_rank;
	p->mpi_la_row_grid = obj->mpi_la_row_grid;
	p->mpi_la_col_grid = obj->mpi_la_col_grid;
#endif

	/* determine the number of threads to use */

	num_threads = obj->num_threads;
	if (num_threads < 2 || max_nrows < MIN_NROWS_TO_THREAD)
		num_threads = 1;

	p->num_threads = num_threads = MIN(num_threads, MAX_THREADS);

/*XXX 	if (max_nrows <= MIN_NROWS_TO_PACK) */{
		matrix_extra_init(obj, p);
		return;
	}

	/* determine the block sizes. We assume that the largest
	   cache in the system is unified and shared across all
	   threads. When performing matrix multiplies 'A*x=b', 
	   we choose the block size small enough so that one block
	   of b fits in L1 cache, and choose the superblock size
	   to be small enough so that a superblock's worth of x
	   or b takes up 3/4 of the largest cache in the system.
	   
	   Making the block size too small increases memory use
	   and puts more pressure on the larger caches, while
	   making the superblock too small reduces the effectiveness
	   of L1 cache and increases the synchronization overhead
	   in multithreaded runs */

	block_size = 8192;
	superblock_size = 3 * obj->cache_size2 / (4 * sizeof(uint64));

	/* possibly override from the command line */

	if (obj->nfs_args != NULL) {

		const char *tmp;

		tmp = strstr(obj->nfs_args, "la_block=");
		if (tmp != NULL)
			block_size = atoi(tmp + 9);

		tmp = strstr(obj->nfs_args, "la_superblock=");
		if (tmp != NULL)
			superblock_size = atoi(tmp + 14);
	}

	logprintf(obj, "using block size %u and superblock size %u for "
			"processor cache size %u kB\n", 
				block_size, superblock_size,
				obj->cache_size2 / 1024);

	p->unpacked_cols = NULL;
	p->first_block_size = first_block_size;

	p->block_size = block_size;
	p->num_block_cols = (ncols + block_size - 1) / block_size;
	p->num_block_rows = 1 + (nrows - first_block_size + 
				block_size - 1) / block_size;

	p->superblock_size = (superblock_size + block_size - 1) / block_size;
	p->num_superblock_cols = (p->num_block_cols + p->superblock_size - 1) / 
					p->superblock_size;
	p->num_superblock_rows = (p->num_block_rows - 1 + 
				p->superblock_size - 1) / p->superblock_size;

	/* do the core work of packing the matrix */

	pack_matrix_core(p, A);

	matrix_extra_init(obj, p);
}

/*-------------------------------------------------------------------*/
void packed_matrix_free(packed_matrix_t *p) {

	uint32 i;

	matrix_extra_free(p);

	if (p->unpacked_cols) {
		la_col_t *A = p->unpacked_cols;
		for (i = 0; i < p->ncols; i++) {
			free(A[i].data);
			A[i].data = NULL;
		}
	}
	else {
		for (i = 0; i < (p->num_dense_rows + 63) / 64; i++)
			free(p->dense_blocks[i]);

		for (i = 0; i < p->num_block_rows * p->num_block_cols; i++) 
			free(p->blocks[i].d.entries);

		free(p->dense_blocks);
		free(p->blocks);
	}
}

/*-------------------------------------------------------------------*/
size_t packed_matrix_sizeof(packed_matrix_t *p) {

	uint32 i, j;
	size_t mem_use;

	/* account for the vectors used in the lanczos iteration */

#ifdef HAVE_MPI
	mem_use = (6 * p->nsubcols + 2 * 
			MAX(p->nrows, p->ncols)) * sizeof(uint64);
#else
	mem_use = 7 * p->max_ncols * sizeof(uint64);
#endif

	/* and for the matrix */

	if (p->unpacked_cols) {
		la_col_t *A = p->unpacked_cols;
		mem_use += p->ncols * (sizeof(la_col_t) +
				(p->num_dense_rows + 31) / 32);
		for (i = 0; i < p->ncols; i++) {
			mem_use += A[i].weight * sizeof(uint32);
		}
	}
	else {
		uint32 num_blocks = p->num_block_rows * 
					p->num_block_cols;

		mem_use += sizeof(uint64) * p->num_threads * 
				p->first_block_size;

		mem_use += sizeof(packed_block_t) * num_blocks;

		mem_use += p->ncols * sizeof(uint64) *
				((p->num_dense_rows + 63) / 64);

		for (j = 0; j < num_blocks; j++) {
			packed_block_t *b = p->blocks + j;

			if (j < p->num_block_cols) {
				mem_use += (b->num_entries + 
					    2 * p->first_block_size) * 
						sizeof(uint16);
			}
			else {
				mem_use += b->num_entries *
						sizeof(entry_idx_t);
			}
		}
	}
	return mem_use;
}

/*-------------------------------------------------------------------*/
void mul_MxN_Nx64(packed_matrix_t *A, void *x, 
			void *scratch) {
    
	/* Multiply the vector x[] by the matrix A and put the 
	   result in scratch[]. The MPI version needs an extra
	   scratch array because MPI reduction operations really
	   want to be out-of-place */

#ifdef HAVE_MPI
	uint64 *scratch2 = (uint64 *)scratch + MAX(A->ncols, A->nrows);

	if (A->mpi_size <= 1) {
#endif
		mul_core(A, x, scratch);
#ifdef HAVE_MPI
		return;
	}
    
	/* make each MPI column gather its own part of x */
	
	global_allgather(x, scratch, A->ncols, A->mpi_nrows, 
			A->mpi_la_row_rank, A->mpi_la_col_grid);
		
	mul_core(A, scratch, scratch2);
	
	/* make each MPI row combine all of its vectors. The
	   matrix-vector product is redundantly stored in each
	   MPI column, but this routine is called very rarely
	   so it's not worth removing the rdundancy */
	
	global_xor(scratch2, scratch, A->nrows, A->mpi_ncols,
			   A->mpi_la_col_rank, A->mpi_la_row_grid);

#endif
}

/*-------------------------------------------------------------------*/
void mul_sym_NxN_Nx64(packed_matrix_t *A, void *x, 
			void *b, void *scratch) {

	/* Multiply x by A and write to scratch, then
	   multiply scratch by the transpose of A and
	   write to b. x may alias b, but the two must
	   be distinct from scratch */

#ifdef HAVE_MPI
	uint64 *scratch2 = (uint64 *)scratch + MAX(A->ncols, A->nrows);
        
	if (A->mpi_size <= 1) {
#endif
		mul_core(A, x, scratch);
		mul_trans_core(A, scratch, b);
#ifdef HAVE_MPI
		return;
	}
    
	/* make each MPI column gather its own part of x */
	 
	global_allgather(x, scratch, A->ncols, A->mpi_nrows, 
			A->mpi_la_row_rank, A->mpi_la_col_grid);
	
	mul_core(A, scratch, scratch2);
		
	/* make each MPI row combine its own part of A*x */
	
	global_xor(scratch2, scratch, A->nrows, A->mpi_ncols,
			   A->mpi_la_col_rank, A->mpi_la_row_grid);
		
	mul_trans_core(A, scratch, scratch2);
		
	/* make each MPI row combine and scatter its own part of A^T * A*x */
		
	global_xor_scatter(scratch2, b, scratch,  A->ncols, A->mpi_nrows, 
			A->mpi_la_row_rank, A->mpi_la_col_grid);
#endif
}
