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

typedef int int32;
typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef long long int64;

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
texture<uint2, cudaTextureType1D, cudaReadModeElementType> inner_tex;

__device__ uint64
uint2_to_uint64(uint2 v)
{
	return (uint64)v.y << 32 | v.x;
}

__device__ uint32
bfe(uint64 x, uint32 pos, uint32 bits)
{
#if __CUDA_ARCH__ >= 200

	uint32 res;
	uint32 hi = (uint32)(x >> 32);
	uint32 lo = (uint32)x;

	if (pos < 32) {
	       if (pos + bits > 32) {
			res = ((lo >> pos) | (hi << (32 - pos))) &
				((1 << bits) - 1);
	       }
	       else {
			asm("bfe.u32 %0, %1, %2, %3; \n\t"
				: "=r"(res) : "r"(lo), "r"(pos), "r"(bits));
	       }
	}
	else {
		asm("bfe.u32 %0, %1, %2, %3; \n\t"
			: "=r"(res) : "r"(hi), "r"(pos - 32), "r"(bits));
	}

	return res;

#else
	return (x >> pos) & ((1 << bits) - 1);
#endif
}

__global__ void
lanczos_kernel_inner_prod(uint64 *y, uint64 *v, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = grid_id; i < n; i += num_threads) {

		uint64 vi = v[i];
		uint64 yi = y[i];
		yi ^=  uint2_to_uint64(tex1Dfetch(inner_tex, 
					0*64 + (int)bfe(vi,  0, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					1*64 + (int)bfe(vi,  6, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					2*64 + (int)bfe(vi, 12, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					3*64 + (int)bfe(vi, 18, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					4*64 + (int)bfe(vi, 24, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					5*64 + (int)bfe(vi, 30, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					6*64 + (int)bfe(vi, 36, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					7*64 + (int)bfe(vi, 42, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					8*64 + (int)bfe(vi, 48, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					9*64 + (int)bfe(vi, 54, 6)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					10*64 + (int)bfe(vi, 60, 6)));
		y[i] = yi;
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

#ifdef __cplusplus
}
#endif
