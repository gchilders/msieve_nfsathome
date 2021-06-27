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
lanczos_kernel_mask(v_t *x, v_t mask, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = grid_id; i < n; i += num_threads)
		x[i] = v_and(x[i], mask);
}

/*------------------------------------------------------------------------*/
__global__ void
lanczos_kernel_xor(v_t *dest, v_t *src, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = grid_id; i < n; i += num_threads)
		dest[i] = v_xor(dest[i], src[i]);
}

/*------------------------------------------------------------------------*/
__global__ void
lanczos_kernel_inner_prod(v_t *y, v_t *v, 
			v_t *lookup, uint32 n)
{
	uint32 i, j;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = grid_id; i < n; i += num_threads) {

		v_t vi = v[i];
		v_t accum;
		for (j = 0; j < VWORDS; j++) accum.w[j] = 0;
		
		for (j = 0; j < 8 * VWORDS; j++) {
			uint32 k = j*256 + ((vi.w[(j >> 3)] >> (8*(j % 8))) & 255);
			// uint32 k = j*256 + bfe(vi.w[(j >> 3)], 8*(j % 8), 8); // faster?			
			accum = v_xor(accum, lookup[k]);
		}
		y[i] = v_xor(y[i], accum);
	}
}	

/*------------------------------------------------------------------------*/

#if VWORDS == 1
/* thanks to Patrick Stach for ideas on this */

#define MAX_OUTER_THREADS 256

__global__ void
lanczos_kernel_outer_prod(v_t *x, v_t *y,
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
		uint64 xi = *((uint64 *) x + i); /* fixme */
		uint64 yi = *((uint64 *) y + i); 

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

#else

__global__ void
lanczos_kernel_outer_prod(v_t *x, v_t *y,
			v_t *xy, uint32 n)
{
	uint32 i, j, k;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 block_id = threadIdx.x;
	v_t xi, yi;
	__shared__ v_t c[3][32 * VWORDS];

	// Zero c[][] in this block
	for (i = block_id; i < 32 * VWORDS; i += blockDim.x) {
		for (j = 0; j < VWORDS; j++) {
			c[0][i].w[j] = 0;
			c[1][i].w[j] = 0;
			c[2][i].w[j] = 0;
		}
	}
	__syncthreads();

	for (i = grid_id; i < n; i += num_threads) {

		xi = x[i];
		yi = y[i]; 

#pragma unroll
		for (j = 0; j < 32 * VWORDS; j++) {
			// offset accesses by thread to reduce conflicts
			uint32 my_j = (j + block_id) & (32 * VWORDS - 1);
			v_t tmp = yi;
			// slightly faster than using bfe assembly instruction 
			k = (xi.w[my_j >> 5] >> (2*(my_j & 31))) & 3;

			// Each array element is hit by 
			// blockDim.x/(32 * VWORDS)/4 threads on average
			if (k == 0) {
				int m;
				for (m = 0; m < VWORDS; m++) tmp.w[m] = 0;
				k = 1;
			}
			v_atomicxor(&(c[k-1][my_j]), tmp);
			__syncthreads();
		}
	}

	// The heavy lifting is done. Just combine the table entries
	// in this block

	for (j = block_id; j < 32 * VWORDS; j += blockDim.x) {
		// Repurpose xi and yi
		xi = v_xor(c[0][j], c[2][j]);
		v_atomicxor(&xy[2 * j], xi);

		yi = v_xor(c[1][j], c[2][j]);
		v_atomicxor(&xy[2 * j + 1], yi);
	}	
}
#endif

#ifdef __cplusplus
}
#endif
