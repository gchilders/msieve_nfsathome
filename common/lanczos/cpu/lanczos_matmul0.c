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

#include "lanczos_cpu.h"

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
static void mul_packed(packed_matrix_t *matrix, 
			uint64 *x, uint64 *b) 
{
	uint32 i, j;
	task_control_t task = {NULL, NULL, NULL, NULL};
	cpudata_t *cpudata = (cpudata_t *)matrix->extra;

	cpudata->x = x;
	cpudata->b = b;

	/* start accumulating the dense matrix multiply results;
	   each thread has scratch space for these, so we don't have
	   to wait for the tasks to finish */

	task.run = mul_packed_small_core;

	for (i = 0; i < matrix->num_threads - 1; i++) {
		task.data = cpudata->tasks + i;
		threadpool_add_task(cpudata->threadpool, &task, 1);
	}
	mul_packed_small_core(cpudata->tasks + i, i);

	/* switch to the sparse blocks */

	task.run = mul_packed_core;

	for (i = 0; i < matrix->num_superblock_cols; i++) {

		la_task_t *t = cpudata->tasks;
				
		for (j = 0; j < matrix->num_threads; j++)
			t[j].block_num = i;

		for (j = 0; j < matrix->num_threads - 1; j++) {
			task.data = t + j;
			threadpool_add_task(cpudata->threadpool, &task, 1);
		}

		mul_packed_core(t + j, j);
		if (j) {
			threadpool_drain(cpudata->threadpool, 1);
		}
	}

	/* xor the small vectors from each thread */

	memcpy(b, cpudata->thread_data[0].tmp_b, 
			matrix->first_block_size *
			sizeof(uint64));

	for (i = 1; i < matrix->num_threads; i++) {
		v_xor(b, cpudata->thread_data[i].tmp_b, 
				matrix->first_block_size);
	}

#if defined(GCC_ASM32A) && defined(HAS_MMX)
	ASM_G volatile ("emms");
#elif defined(MSC_ASM32A) && defined(HAS_MMX)
	ASM_M emms
#endif
}

/*-------------------------------------------------------------------*/
static void mul_trans_packed(packed_matrix_t *matrix, 
			uint64 *x, uint64 *b) 
{
	uint32 i, j;
	task_control_t task = {NULL, NULL, NULL, NULL};
	cpudata_t *cpudata = (cpudata_t *)matrix->extra;

	cpudata->x = x;
	cpudata->b = b;

	task.run = mul_trans_packed_core;

	for (i = 0; i < matrix->num_superblock_rows; i++) {

		la_task_t *t = cpudata->tasks;
				
		for (j = 0; j < matrix->num_threads; j++)
			t[j].block_num = i;

		for (j = 0; j < matrix->num_threads - 1; j++) {
			task.data = t + j;
			threadpool_add_task(cpudata->threadpool, &task, 1);
		}

		mul_trans_packed_core(t + j, j);
		if (j) {
			threadpool_drain(cpudata->threadpool, 1);
		}
	}

	if (matrix->num_dense_rows) {
		/* add in the dense matrix multiply blocks; these don't 
		   use scratch space, but need all of b to accumulate 
		   results so we have to wait until all tasks finish */

		task.run = mul_trans_packed_small_core;

		for (i = 0; i < matrix->num_threads - 1; i++) {
			task.data = cpudata->tasks + i;
			threadpool_add_task(cpudata->threadpool, &task, 1);
		}

		mul_trans_packed_small_core(cpudata->tasks + i, i);
		if (i) {
			threadpool_drain(cpudata->threadpool, 1);
		}
	}

#if defined(GCC_ASM32A) && defined(HAS_MMX)
	ASM_G volatile ("emms");
#elif defined(MSC_ASM32A) && defined(HAS_MMX)
	ASM_M emms
#endif
}

/*--------------------------------------------------------------------*/
static void matrix_thread_init(void *data, int thread_num) {

	packed_matrix_t *p = (packed_matrix_t *)data;
	cpudata_t *cpudata = (cpudata_t *)p->extra;
	thread_data_t *t = cpudata->thread_data + thread_num;

	/* we use this scratch vector for both matrix multiplies
	   and vector-vector operations; it has to be large enough
	   to support both */

	t->tmp_b = (uint64 *)xmalloc(MAX(64, p->first_block_size) *
					sizeof(uint64));
}

/*-------------------------------------------------------------------*/
static void matrix_thread_free(void *data, int thread_num) {

	packed_matrix_t *p = (packed_matrix_t *)data;
	cpudata_t *cpudata = (cpudata_t *)p->extra;
	thread_data_t *t = cpudata->thread_data + thread_num;

	free(t->tmp_b);
}

/*-------------------------------------------------------------------*/
void matrix_extra_init(msieve_obj *obj, packed_matrix_t *p) {

	uint32 i;
	thread_control_t control;
	cpudata_t *cpudata;

	p->extra = cpudata = (cpudata_t *)xcalloc(1, sizeof(cpudata_t));

	/* start the thread pool */

	control.init = matrix_thread_init;
	control.shutdown = matrix_thread_free;
	control.data = p;

	if (p->num_threads > 1) {
		cpudata->threadpool = threadpool_init(p->num_threads - 1, 
						200, &control);
	}
	matrix_thread_init(p, p->num_threads - 1);

	/* pre-generate the structures to drive the thread pool */

	cpudata->tasks = (la_task_t *)xmalloc(sizeof(la_task_t) * 
					p->num_threads);

	for (i = 0; i < p->num_threads; i++) {
		cpudata->tasks[i].matrix = p;
		cpudata->tasks[i].task_num = i;
	}
}

/*-------------------------------------------------------------------*/
void matrix_extra_free(packed_matrix_t *p) {

	if (p->unpacked_cols == NULL) {

		cpudata_t *cpudata = (cpudata_t *)p->extra;

		if (p->num_threads > 1) {
			threadpool_drain(cpudata->threadpool, 1);
			threadpool_free(cpudata->threadpool);
		}
		matrix_thread_free(p, p->num_threads - 1);

		free(cpudata->tasks);
		free(cpudata);
	}
}

/*-------------------------------------------------------------------*/
void mul_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	/* Multiply the vector x[] by the matrix A and put the 
	   result in b[]. x must not alias b */

	uint64 *x = (uint64 *)x_in;
	uint64 *b = (uint64 *)b_in;

	if (A->unpacked_cols)
		mul_unpacked(A, x, b);
	else
		mul_packed(A, x, b);
}

/*-------------------------------------------------------------------*/
void mul_trans_core(packed_matrix_t *A, void *x_in, void *b_in) {
    
	/* Multiply the vector x[] by the transpose of matrix A 
	   and put the result in b[]. x must not alias b */

	uint64 *x = (uint64 *)x_in;
	uint64 *b = (uint64 *)b_in;

	if (A->unpacked_cols)
		mul_trans_unpacked(A, x, b);
	else
		mul_trans_packed(A, x, b);
}
