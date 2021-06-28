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

#elif VWORDS == 2 || VWORDS == 4

__global__ void
lanczos_kernel_outer_prod(v_t *x, v_t *y,
			v_t *xy, uint32 n)
{
	// Use 16 * VWORDS threads
	uint32 i, j, k;
	uint32 block_id = threadIdx.x;
	v_t xi, yi, a;
	__shared__ v_t c[16][16 * VWORDS];

	for (j = block_id; j < 16 * VWORDS; j += blockDim.x) { 
		for (i = 1; i < 16; i++) {
			for (k = 0; k < VWORDS; k++) c[i][j].w[k] = 0;
		}
		for (i = blockIdx.x; i < n; i += gridDim.x) {
			xi = x[i];
			yi = y[i];

			// slightly faster than using bfe assembly instruction 
			k = (xi.w[j >> 4] >> (4*(j & 15))) & 15;

			c[k][j] = v_xor(c[k][j], yi);
		}
		a = v_xor(c[1][j], c[3][j]);
		a = v_xor(a, c[5][j]);
		a = v_xor(a, c[7][j]);
		a = v_xor(a, c[9][j]);
		a = v_xor(a, c[11][j]);
		a = v_xor(a, c[13][j]);
		a = v_xor(a, c[15][j]);
		v_atomicxor(&xy[4*j], a);

		a = v_xor(c[2][j], c[3][j]);
		a = v_xor(a, c[6][j]);
		a = v_xor(a, c[7][j]);
		a = v_xor(a, c[10][j]);
		a = v_xor(a, c[11][j]);
		a = v_xor(a, c[14][j]);
		a = v_xor(a, c[15][j]);
		v_atomicxor(&xy[4*j + 1], a);

		a = v_xor(c[4][j], c[5][j]);
		a = v_xor(a, c[6][j]);
		a = v_xor(a, c[7][j]);
		a = v_xor(a, c[12][j]);
		a = v_xor(a, c[13][j]);
		a = v_xor(a, c[14][j]);
		a = v_xor(a, c[15][j]);
		v_atomicxor(&xy[4*j + 2], a);

		a = v_xor(c[8][j], c[9][j]);
		a = v_xor(a, c[10][j]);
		a = v_xor(a, c[11][j]);
		a = v_xor(a, c[12][j]);
		a = v_xor(a, c[13][j]);
		a = v_xor(a, c[14][j]);
		a = v_xor(a, c[15][j]);
		v_atomicxor(&xy[4*j + 3], a);
	}
}

#else

__global__ void
lanczos_kernel_outer_prod(v_t *x, v_t *y,
			v_t *xy, uint32 n)
{
	// Use 16 * VWORDS threads
	uint32 i, j, k;
	uint32 block_id = threadIdx.x;
	v_t xi, yi;
	// Table is too large for shared memory. Use registers.
	v_t c[16]; // 128 registers for VWORDS=8

	for (j = block_id; j < 16 * VWORDS; j += blockDim.x) {
		for (i = 1; i < 16; i++) {
			for (k = 0; k < VWORDS; k++) c[i].w[k] = 0;
		}
		for (i = blockIdx.x; i < n; i += gridDim.x) {
			xi = x[i];
			yi = y[i];

			// slightly faster than using bfe assembly instruction 
			k = (xi.w[j >> 4] >> (4*(j & 15))) & 15;

			// c[k] = v_xor(c[k], yi);
			// Can't reference an array element in registers at runtime.
			// Doing so forces it into slow local memory.
			if (k == 1) c[1] = v_xor(c[1], yi);
			else if (k == 2) c[2] = v_xor(c[2], yi);
			else if (k == 3) c[3] = v_xor(c[3], yi);
			else if (k == 4) c[4] = v_xor(c[4], yi);
			else if (k == 5) c[5] = v_xor(c[5], yi);
			else if (k == 6) c[6] = v_xor(c[6], yi);
			else if (k == 7) c[7] = v_xor(c[7], yi);
			else if (k == 8) c[8] = v_xor(c[8], yi);
			else if (k == 9) c[9] = v_xor(c[9], yi);
			else if (k == 10) c[10] = v_xor(c[10], yi);
			else if (k == 11) c[11] = v_xor(c[11], yi);
			else if (k == 12) c[12] = v_xor(c[12], yi);
			else if (k == 13) c[13] = v_xor(c[13], yi);
			else if (k == 14) c[14] = v_xor(c[14], yi);
			else if (k == 15) c[15] = v_xor(c[15], yi);
		}
		c[0] = v_xor(c[1], c[3]);
		c[0] = v_xor(c[0], c[5]);
		c[0] = v_xor(c[0], c[7]);
		c[0] = v_xor(c[0], c[9]);
		c[0] = v_xor(c[0], c[11]);
		c[0] = v_xor(c[0], c[13]);
		c[0] = v_xor(c[0], c[15]);
		v_atomicxor(&xy[4*j], c[0]);

		c[0] = v_xor(c[2], c[3]);
		c[0] = v_xor(c[0], c[6]);
		c[0] = v_xor(c[0], c[7]);
		c[0] = v_xor(c[0], c[10]);
		c[0] = v_xor(c[0], c[11]);
		c[0] = v_xor(c[0], c[14]);
		c[0] = v_xor(c[0], c[15]);
		v_atomicxor(&xy[4*j + 1], c[0]);

		c[0] = v_xor(c[4], c[5]);
		c[0] = v_xor(c[0], c[6]);
		c[0] = v_xor(c[0], c[7]);
		c[0] = v_xor(c[0], c[12]);
		c[0] = v_xor(c[0], c[13]);
		c[0] = v_xor(c[0], c[14]);
		c[0] = v_xor(c[0], c[15]);
		v_atomicxor(&xy[4*j + 2], c[0]);

		c[0] = v_xor(c[8], c[9]);
		c[0] = v_xor(c[0], c[10]);
		c[0] = v_xor(c[0], c[11]);
		c[0] = v_xor(c[0], c[12]);
		c[0] = v_xor(c[0], c[13]);
		c[0] = v_xor(c[0], c[14]);
		c[0] = v_xor(c[0], c[15]);
		v_atomicxor(&xy[4*j + 3], c[0]);
	}
}

#endif

#ifdef __cplusplus
}
#endif
