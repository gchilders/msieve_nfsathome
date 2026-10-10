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

#ifndef _COMMON_LANCZOS_GPU_LANCZOS_GPU_H_
#define _COMMON_LANCZOS_GPU_LANCZOS_GPU_H_

#include <cuda_xface.h>
#include <spmv_engine.h>
#include "../lanczos.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	uint32 num_rows;
	uint32 num_cols;
	uint32 num_col_entries;
	uint32 blocksize;
	uint32 start;                   /* first col (row for the transpose) */
	CUdeviceptr col_entries;        /* uint32 */
	CUdeviceptr row_entries;        /* uint32 */

	/* column indices followed by row pointers (at row_offset), in
	   host memory while the matrix is set up; for streamed blocks
	   it stays there, pinned */
	uint32 *host_data;
	size_t row_offset;
	size_t bytes;

	/* nonzero if the block lives in host memory and is copied to
	   the card for every product. sched_idx holds its positions in
	   the stream schedule, for the forward and the single-copy
	   transpose product */
	uint32 streamed;
	uint32 pinned;          /* host_data came from cuMemHostAlloc */
	uint32 sched_idx[2];
} block_row_t;

/* a device buffer that streamed blocks are copied into */

typedef struct {
	CUdeviceptr buf;
	block_row_t *blk;       /* block it holds, or being copied in */
	CUevent ready;          /* the copy of blk is done */
	CUevent freed;          /* the last kernel reading buf is done */
} stream_slot_t;

/* implementation-specific structure */

typedef struct {

	gpu_info_t *gpu_info;

	CUcontext gpu_context;
	CUmodule gpu_module;

	gpu_launch_t *launch;

	/* gpu product data */

	CUdeviceptr gpu_scratch;

	/* matrix data */

	CUdeviceptr *dense_blocks;

	uint32 num_block_rows;
	block_row_t *block_rows;

	uint32 num_trans_block_rows;
	block_row_t *trans_block_rows;

	/* scan engine data */

	libhandle_t spmv_engine_handle;
	spmv_engine_init_func spmv_engine_init;
	spmv_engine_free_func spmv_engine_free;
	spmv_engine_run_func spmv_engine_run;
	spmv_engine_run_func spmv_engine_run_trans;
	spmv_engine_set_kernel_func spmv_engine_set_kernel;
	void * spmv_engine;

	/* use managed memory to store the matrix data */
	uint32 use_cudamanaged;

	/* store only A on the card; A^T * x scatters through A's blocks */
	uint32 single_copy;

	/* the solver never applies A^T, so there is no transpose product
	   to serve at all -- no second copy and no scatter either */
	uint32 forward_only;

	/* matrix blocks that don't fit on the card are streamed from
	   pinned host memory. sched lists them in the order one Lanczos
	   iteration uses them, so the copies can run ahead of the SpMV */
	CUstream copy_stream;
	uint32 num_slots;
	stream_slot_t *slots;
	uint32 sched_len;
	block_row_t **sched;
	size_t streamed_bytes;
	size_t staging_bytes;

} gpudata_t;


typedef struct {
	gpudata_t *gpudata;
	v_t *host_vec;
	CUdeviceptr gpu_vec;
} gpuvec_t;

/* #define LANCZOS_GPU_DEBUG */

/* ordinal list of GPU kernels */
enum {
	GPU_K_MASK = 0,
	GPU_K_XOR,
	GPU_K_INNER_PROD,
	GPU_K_OUTER_PROD,
	GPU_K_OUTER_PROD_BIG,
	NUM_GPU_FUNCTIONS /* must be last */
};

void vv_xor_gpu(void *dest, void *src, uint32 n, gpudata_t *d);

void mul_BxN_NxB_gpu(packed_matrix_t *matrix,
		   CUdeviceptr x, CUdeviceptr y,
		   CUdeviceptr xy, uint32 n);

void mul_NxB_BxB_acc_gpu(packed_matrix_t *matrix, 
			CUdeviceptr v, CUdeviceptr x,
			CUdeviceptr y, uint32 n);

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_LANCZOS_GPU_LANCZOS_GPU_H_ */
