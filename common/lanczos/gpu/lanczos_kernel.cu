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
   as with 2-bit pieces.

   Sweeping w_x and w_y independently would make VWORDS * VWORDS passes
   over x and y, and each pass reads one 8-byte word of every 32-byte
   v_t (at VBITS=256). Consecutive elements are a whole v_t apart, so
   each of those reads pulls its own sector and three quarters of the
   DRAM traffic is thrown away. Carrying OUTER_GY y words through the
   element loop at once lets a single read of y[i] serve all of them,
   which cuts the passes to VWORDS * (VWORDS / OUTER_GY) at the cost of
   OUTER_GY copies of the tables. The XORs are the same multiset in a
   different order, so the result is bit-identical either way.

   OUTER_GY_BIG = 2 is the measured optimum, not the largest value that
   works. On a 2080 Ti at n=8M, VBITS=256, it is 1.5x the ungrouped
   kernel; grouping 4 halves the passes again but needs four copies of
   the tables, and the occupancy that costs drops it back to 1.13x. Two
   copies at 128 threads need the same 30kB the ungrouped kernel uses at
   256, so this stays inside the 48kB every architecture gives a block,
   with no opt-in and no compute-capability floor.

   This only pays on a large matrix, which is why both kernels are here
   and mul_BxN_NxB_gpu() chooses between them on n; see the note on
   OUTER_PROD_BIG_MIN_N. Below that the ungrouped kernel is faster, and
   it stays the one small jobs run. */

__global__ void
lanczos_kernel_outer_prod_big(v_t *x, v_t *y,
			v_t *xy, uint32 n)
{
	uint32 i, j, g, w_x, w_y0;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 tid = threadIdx.x;
	uint32 k = tid % 16;
	uint32 num_halves = blockDim.x / 16;
	uint32 half_stride = 15 * 16;
	uint32 copy_stride = num_halves * half_stride;
	__shared__ uint64 scratch[OUTER_GY_BIG][MAX_OUTER_THREADS_BIG / 16][15][16];
	uint64 *s = &scratch[0][tid / 16][0][0];
	uint64 *flat = &scratch[0][0][0][0];

	for (w_x = 0; w_x < VWORDS; w_x++) {
		for (w_y0 = 0; w_y0 < VWORDS; w_y0 += OUTER_GY_BIG) {

			for (i = tid; i < OUTER_GY_BIG * copy_stride; i += blockDim.x)
				flat[i] = 0;
			__syncthreads();

			for (i = grid_id; i < n; i += num_threads) {
				uint64 xi = x[i].w[w_x];
				uint64 yi[OUTER_GY_BIG];

#pragma unroll
				for (g = 0; g < OUTER_GY_BIG; g++)
					yi[g] = y[i].w[w_y0 + g];

				if (k != 0)
					xi = (xi >> (4 * k)) | (xi << (64 - 4 * k));

#pragma unroll
				for (j = 0; j < 16; j++) {
					uint32 m = bfe(xi, 4 * j, 4);
					uint32 nonzero = (m != 0);
					uint32 off;

					if (m == 0)
						m = 1;
					off = 16 * (m - 1) + ((k + j) & 15);

#pragma unroll
					for (g = 0; g < OUTER_GY_BIG; g++) {
						s[g * copy_stride + off] ^=
							nonzero ? yi[g] : 0;
					}
				}
			}
			__syncthreads();

			/* fold the half-warp copies together, for each y word */

			for (i = tid; i < OUTER_GY_BIG * half_stride; i += blockDim.x) {
				uint64 *base = flat + (i / half_stride) *
							copy_stride;
				uint32 r = i % half_stride;
				uint64 acc = base[r];

				for (j = 1; j < num_halves; j++)
					acc ^= base[j * half_stride + r];
				base[r] = acc;
			}
			__syncthreads();

			if (tid < 64) {
				uint32 c = tid / 4;
				uint32 b = tid % 4;
				uint32 m;

#pragma unroll
				for (g = 0; g < OUTER_GY_BIG; g++) {
					uint64 *base = flat + g * copy_stride;
					uint64 res = 0;

					for (m = 1; m < 16; m++) {
						if (m & (1 << b))
							res ^= base[(m - 1) * 16 + c];
					}
					atomicXor(&xy[64 * w_x + tid].w[w_y0 + g],
							res);
				}
			}
			__syncthreads();
		}
	}
}

/*------------- one y word at a time, as originally -----------------*/

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
