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

/* thanks to Patrick Stach for ideas on this */

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
	// Don't actually need third table. 
	// Can v_atomicxor c[2] into both c[0] and c[1]
	// Saves shared memory at the expense of additional atomicXor's

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
			// k = (xi.w[my_j >> 5] >> (2*(my_j & 31))) & 3;
			uint64 x = xi.w[my_j >> 5];
			uint32 hi = (uint32)(x >> 32);
			uint32 lo = (uint32)x;
			uint32 pos = 2 * (my_j & 31);
			// We never stride the hi and lo uint32 words
			if (pos < 32) {
				asm("bfe.u32 %0, %1, %2, %3; \n\t"
					: "=r"(k) : "r"(lo), "r"(pos), "r"(2));
			} else {
				asm("bfe.u32 %0, %1, %2, %3; \n\t"
					: "=r"(k) : "r"(hi), "r"(pos - 32), "r"(2));
			}

			// Each array element is hit by 
			// blockDim.x/(32 * VWORDS)/4 threads on average
			if (k != 0) {
				v_atomicxor(&(c[k-1][my_j]), yi);
			}
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

#ifdef __cplusplus
}
#endif
