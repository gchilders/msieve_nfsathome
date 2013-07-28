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

#include "lanczos_gpu_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*------------------------------------------------------------------------*/
__global__ void
lanczos_kernel_mask(uint64 *x, uint64 mask, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = grid_id; i < n; i += num_threads)
		x[i] &= mask;
}

/*------------------------------------------------------------------------*/
__global__ void
lanczos_kernel_xor(uint64 *dest, uint64 *src, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = grid_id; i < n; i += num_threads)
		dest[i] ^= src[i];
}

/*------------------------------------------------------------------------*/
__global__ void
lanczos_kernel_inner_prod(uint64 *y, uint64 *v, 
			uint64 *lookup, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = grid_id; i < n; i += num_threads) {

		uint64 vi = load_bypassL1(v + i);
		uint64 yi = load_bypassL1(y + i);
		yi ^=    lookup[ 0*64 + bfe(vi,  0, 6)]
		       ^ lookup[ 1*64 + bfe(vi,  6, 6)]
		       ^ lookup[ 2*64 + bfe(vi, 12, 6)]
		       ^ lookup[ 3*64 + bfe(vi, 18, 6)]
		       ^ lookup[ 4*64 + bfe(vi, 24, 6)]
		       ^ lookup[ 5*64 + bfe(vi, 30, 6)]
		       ^ lookup[ 6*64 + bfe(vi, 36, 6)]
		       ^ lookup[ 7*64 + bfe(vi, 42, 6)]
		       ^ lookup[ 8*64 + bfe(vi, 48, 6)]
		       ^ lookup[ 9*64 + bfe(vi, 54, 6)]
		       ^ lookup[10*64 + bfe(vi, 60, 6)];
		store_bypassL1(yi, y + i);
	}
}

/*------------------------------------------------------------------------*/
/* thanks to Patrick Stach for ideas on this */

#define MAX_OUTER_THREADS 256

__global__ void
lanczos_kernel_outer_prod(uint64 *x, uint64 *y,
			uint32 *xy, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 block_id = threadIdx.x;
	__shared__ uint64 scratch[3 * MAX_OUTER_THREADS];
	uint64 *s = scratch + (block_id & ~0x1f);

	scratch[block_id + 0*MAX_OUTER_THREADS] = 0;
	scratch[block_id + 1*MAX_OUTER_THREADS] = 0;
	scratch[block_id + 2*MAX_OUTER_THREADS] = 0;

	for (i = grid_id; i < n; i += num_threads) {

		uint32 j; 
		uint32 k = block_id & 0x1f;
		uint64 xi = x[i];
		uint64 yi = y[i];

		if (k != 0)
			xi = (xi >> (2 * k)) | (xi << (64 - (2 * k)));

#pragma unroll
		for (j = 0; j < 32; j++) {

			uint32 off = bfe(xi, 2 * j, 2);
			uint64 tmp = yi;

			if (off == 0) {
				tmp = 0;
				off = 1;
			}

			s[((k + j) & 0x1f) + 
				MAX_OUTER_THREADS * (off - 1)] ^= tmp;
		}
	}

	s = scratch + block_id;
	__syncthreads();
	s[0*MAX_OUTER_THREADS] ^= s[2*MAX_OUTER_THREADS];
	s[1*MAX_OUTER_THREADS] ^= s[2*MAX_OUTER_THREADS];
	__syncthreads();

	for (i = MAX_OUTER_THREADS / 2; i >= 32; i >>= 1) {
		if (block_id < i) {
			s[0*MAX_OUTER_THREADS] ^= s[0*MAX_OUTER_THREADS + i];
			s[1*MAX_OUTER_THREADS] ^= s[1*MAX_OUTER_THREADS + i];
		}
		__syncthreads();
	}


	if (block_id < 64) {
		uint32 *t = (uint32 *)scratch;

		i = 4 * (block_id / 2);

		if (block_id % 2 == 0)
			atomicXor(&xy[i], t[block_id]);
		else
			atomicXor(&xy[i + 1], t[block_id]);
	}
	else if (block_id < 128) {
		uint32 *t = (uint32 *)(scratch + MAX_OUTER_THREADS);

		i = 4 * ((block_id - 64) / 2) + 2;

		if (block_id % 2 == 0)
			atomicXor(&xy[i], t[block_id - 64]);
		else
			atomicXor(&xy[i + 1], t[block_id - 64]);
	}
}

/*------------------------------------------------------------------------*/
texture<uint2, cudaTextureType1D, cudaReadModeElementType> matmul_tex;

__global__ void
lanczos_kernel_matmul(uint32 num_block_cols,
		      uint32 curr_col,
		      uint32 block_size,
		      uint32 first_block_size,
		      uint32 *counts,
		      uint32 *offsets,
		      gpu_entry_idx_t *mat_entries,
		      uint64 *x,
		      uint64 *b)
{
	__shared__ uint64 shared_sums[MATMUL_THREADS];

	uint32 curr_row = blockIdx.x;
	uint64 *curr_x = x + curr_col * block_size;
	uint64 *curr_b = (curr_row == 0) ? b : b + first_block_size +
				(curr_row - 1) * block_size;
	uint32 which_block = curr_row * num_block_cols + curr_col;
	uint32 num_entries = counts[which_block];
	gpu_entry_idx_t *entries = mat_entries + offsets[which_block];
	uint32 i = MATMUL_THREADS * (1 + num_entries / 
			MATMUL_THREADS) + threadIdx.x;
	uint64 sum = 0;
	gpu_entry_idx_t e;

	while (i >= MATMUL_THREADS) {

		if (i < num_entries + MATMUL_THREADS) {

			e = entries[i];

			sum ^= curr_x[e.col_off];

			if (e.row_off_head) {
				curr_b[e.row_off] ^= sum;
				sum = 0;
			}
		}

		i -= MATMUL_THREADS;
	}

	shared_sums[i] = sum;
	__syncthreads();

	/* start parallel reverse exclusive segmented prefix scan */

	if (threadIdx.x == 0) {
		uint32 j = MATMUL_THREADS - 1;

		sum = shared_sums[j];
		shared_sums[j] = 0;

		for (j--; (int32)j >= 0; j--) {

			uint64 sum0 = shared_sums[j];

			e = entries[j];
			if (e.row_off_head) {
				shared_sums[j] = sum;
				sum = sum0;
			}
			else {
				sum ^= sum0;
			}
		}
	}
	/* end parallel reverse exclusive segmented prefix scan */

	__syncthreads();

	e = entries[i];
	if (e.row_off_head)
		curr_b[e.row_off] ^= shared_sums[i];
}

#ifdef __cplusplus
}
#endif
