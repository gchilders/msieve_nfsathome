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

__global__ void
lanczos_kernel_inner_prod(uint64 *y, uint64 *v,
			uint64 *lookup, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = grid_id; i < n; i += num_threads) {
		uint64 word = v[i];
		y[i] ^=  uint2_to_uint64(tex1Dfetch(inner_tex, 
					0*256 + (int)((word >>  0) & 0xff)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					1*256 + (int)((word >>  8) & 0xff)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					2*256 + (int)((word >> 16) & 0xff)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					3*256 + (int)((word >> 24) & 0xff)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					4*256 + (int)((word >> 32) & 0xff)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					5*256 + (int)((word >> 40) & 0xff)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					6*256 + (int)((word >> 48) & 0xff)))
		       ^ uint2_to_uint64(tex1Dfetch(inner_tex, 
					7*256 + (int)((word >> 56) & 0xff)));
	}
}

/*------------------------------------------------------------------------*/
/* thanks to Patrick Stach for ideas on this */

#define MAX_OUTER_THREADS 256
//#define MAX_OUTER_THREADS 192

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

			uint32 off = (xi >> (2 * j)) & 0x3;
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

#if MAX_OUTER_THREADS == 256
	for (i = MAX_OUTER_THREADS / 2; i >= 32; i >>= 1) {
		if (block_id < i) {
			s[0*MAX_OUTER_THREADS] ^= s[0*MAX_OUTER_THREADS + i];
			s[1*MAX_OUTER_THREADS] ^= s[1*MAX_OUTER_THREADS + i];
		}
		__syncthreads();
	}

#elif MAX_OUTER_THREADS == 192
	if (block_id < 96) {
		s[0*MAX_OUTER_THREADS] ^= s[0*MAX_OUTER_THREADS + 96];
		s[1*MAX_OUTER_THREADS] ^= s[1*MAX_OUTER_THREADS + 96];
	}
	__syncthreads();

	if (block_id < 32) {
		s[0*MAX_OUTER_THREADS] ^= s[0*MAX_OUTER_THREADS + 32] ^
		                          s[0*MAX_OUTER_THREADS + 64];
		s[1*MAX_OUTER_THREADS] ^= s[1*MAX_OUTER_THREADS + 32] ^
		                          s[1*MAX_OUTER_THREADS + 64];
	}
	__syncthreads();
#endif

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
