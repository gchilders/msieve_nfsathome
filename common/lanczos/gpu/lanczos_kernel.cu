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
#if VBITS <= 256

/* y ^= v * x, where x is a VBITS x VBITS matrix. Each 4-bit piece of
   v[i] selects one of 16 precomputed XOR combinations of 4 rows of x,
   so there are VBITS/4 table lookups per vector element and no
   branches. The table is stored word-major, so the 16 entries a warp
   can hit for one word sit in distinct shared memory banks. Needs
   VBITS * VWORDS * 32 bytes of shared memory (32kB at VBITS=256) */

__global__ void
lanczos_kernel_inner_prod(v_t *y, v_t *v,
			v_t *x, uint32 n)
{
	uint32 i, j, w;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ uint64 t[VBITS / 4][VWORDS][16];

	for (i = threadIdx.x; i < (VBITS / 4) * 16; i += blockDim.x) {
		uint32 g = i / 16;
		uint32 m = i % 16;

		for (w = 0; w < VWORDS; w++) {
			uint64 val = 0;
			for (j = 0; j < 4; j++) {
				if (m & (1 << j))
					val ^= x[4 * g + j].w[w];
			}
			t[g][w][m] = val;
		}
	}

	__syncthreads();

	for (i = grid_id; i < n; i += num_threads) {
		v_t vi = v[i];
		v_t acc;

		for (w = 0; w < VWORDS; w++)
			acc.w[w] = 0;

#pragma unroll
		for (j = 0; j < VBITS / 4; j++) {
			uint32 m = (uint32)(vi.w[j / 16] >> (4 * (j % 16))) & 15;

#pragma unroll
			for (w = 0; w < VWORDS; w++)
				acc.w[w] ^= t[j][w][m];
		}
		y[i] = v_xor(y[i], acc);
	}
}

#else

__global__ void
lanczos_kernel_inner_prod(v_t *y, v_t *v,
			v_t *x, uint32 n)
{
	uint32 i, j;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;
	v_t acc;
	__shared__ v_t c[32*VWORDS][3];

	for (i = threadIdx.x; i < 32 * VWORDS; i += blockDim.x) {
		acc = x[2 * i];
		c[i][0] = acc;

		acc = v_xor(acc, x[2 * i + 1]);
		c[i][2] = acc;

		acc = v_xor(acc, x[2 * i]);
		c[i][1] = acc;
	}

	__syncthreads();

	for (i = grid_id; i < n; i += num_threads) {
		v_t vi = v[i];
		for (j = 0; j < VWORDS; j++) acc.w[j] = 0;

		for (j = 0; j < 32 * VWORDS; j++) {
			uint32 k = (vi.w[j >> 5] >> (2*(j & 31))) & 3;
			if (k != 0) acc = v_xor(acc, c[j][k-1]);
		}
		y[i] = v_xor(y[i], acc);
	}
}

#endif

/*------------------------------------------------------------------------*/

/* thanks to Patrick Stach for ideas on this */

/* xy ^= transpose(x) * y, a VBITS x VBITS result. For each pair of
   words (w_x, w_y), every element XORs y[i].w[w_y] into one of 15
   tables selected by each 4-bit piece of x[i].w[w_x]; afterwards row
   4c+b of the result is the XOR of the tables for piece c whose index
   has bit b set. Each half-warp owns a copy of the tables (16 pieces,
   one slot each), and lane k rotates its x word by k pieces so that
   the 16 lanes of a half-warp always update 16 different slots. That
   is 16 shared memory updates per element and word pair, half as many
   as with 2-bit pieces */

#define MAX_OUTER_THREADS 256

__global__ void
lanczos_kernel_outer_prod(v_t *x, v_t *y,
			v_t *xy, uint32 n) 
{
	uint32 i, j, w_x, w_y;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 tid = threadIdx.x;
	uint32 k = tid % 16;
	uint32 num_halves = blockDim.x / 16;
	__shared__ uint64 scratch[MAX_OUTER_THREADS / 16][15][16];
	uint64 *s = &scratch[tid / 16][0][0];
	uint64 *flat = &scratch[0][0][0];

	for (w_x = 0; w_x < VWORDS; w_x++) {
		for (w_y = 0; w_y < VWORDS; w_y++) {

			for (i = tid; i < num_halves * 15 * 16; i += blockDim.x)
				flat[i] = 0;
			__syncthreads();

			for (i = grid_id; i < n; i += num_threads) {
				uint64 xi = x[i].w[w_x];
				uint64 yi = y[i].w[w_y];

				if (k != 0)
					xi = (xi >> (4 * k)) | (xi << (64 - 4 * k));

#pragma unroll
				for (j = 0; j < 16; j++) {
					uint32 m = bfe(xi, 4 * j, 4);
					uint64 tmp = yi;

					if (m == 0) {
						tmp = 0;
						m = 1;
					}

					s[16 * (m - 1) + ((k + j) & 15)] ^= tmp;
				}
			}
			__syncthreads();

			/* fold the half-warp copies together */

			for (i = tid; i < 15 * 16; i += blockDim.x) {
				uint64 acc = flat[i];
				for (j = 1; j < num_halves; j++)
					acc ^= flat[j * 15 * 16 + i];
				flat[i] = acc;
			}
			__syncthreads();

			if (tid < 64) {
				uint32 c = tid / 4;
				uint32 b = tid % 4;
				uint32 m;
				uint64 res = 0;

				for (m = 1; m < 16; m++) {
					if (m & (1 << b))
						res ^= scratch[0][m - 1][c];
				}
				atomicXor(&xy[64 * w_x + tid].w[w_y], res);
			}
			__syncthreads();
		}
	}
}

#ifdef __cplusplus
}
#endif
