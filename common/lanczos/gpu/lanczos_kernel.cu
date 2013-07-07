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

__global__ void
lanczos_kernel_outer_prod(uint64 *x, uint64 *y,
			uint32 *xy, uint32 n)
{
	uint32 i;
	uint32 limit;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 grid_id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 block_id = threadIdx.x;
	__shared__ uint64 scratch[MAX_OUTER_THREADS];

	/* round n up to the next multiple of the block size;
	   we have to synchronize across the whole block, which
	   means the loop below must execute only for whole
	   blocks */

	scratch[block_id] = 0;

	limit = (n + MAX_OUTER_THREADS - 1) &
			~(MAX_OUTER_THREADS - 1);
	__syncthreads();

	for (i = grid_id; i < limit; i += num_threads) {

		uint32 j, k;
		uint64 xi = 0;
		uint64 yi = 0;

		if (i < n) {
			xi = x[i];
			yi = y[i];
		}

		for (j = 0, k = block_id & 0x3f; j < 64; 
					j++, k = (k + 1) & 0x3f) {

			uint64 tmp = 0;
			if ((xi >> k) & (uint64)1)
				tmp = yi;

			scratch[(block_id & ~0x3f) | k] ^= tmp;
			__syncthreads();
		}
	}

	for (i = MAX_OUTER_THREADS / 2; i >= 64; i >>= 1) {
		if (block_id < i)
			scratch[block_id] ^= scratch[block_id + i];
		__syncthreads();
	}

	if (block_id < 128) {
		uint32 *s = (uint32 *)scratch;
		atomicXor(&xy[block_id], s[block_id]);
	}
}

#ifdef __cplusplus
}
#endif
