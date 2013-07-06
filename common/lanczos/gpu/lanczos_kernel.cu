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

typedef unsigned char uint8;
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
	uint32 my_threadid = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = my_threadid; i < n; i += num_threads)
		x[i] &= mask;
}

/*------------------------------------------------------------------------*/
__global__ void
lanczos_kernel_xor(uint64 *dest, uint64 *src, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 my_threadid = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = my_threadid; i < n; i += num_threads)
		dest[i] ^= src[i];
}

/*------------------------------------------------------------------------*/
__global__ void
lanczos_kernel_inner_prod(uint64 *y, uint64 *v,
			uint64 *lookup, uint32 n)
{
	uint32 i;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 my_threadid = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = my_threadid; i < n; i += num_threads) {
		uint64 word = v[i];
		y[i] ^=  lookup[ 0*256 + ((uint8)(word >>  0)) ]
		       ^ lookup[ 1*256 + ((uint8)(word >>  8)) ]
		       ^ lookup[ 2*256 + ((uint8)(word >> 16)) ]
		       ^ lookup[ 3*256 + ((uint8)(word >> 24)) ]
		       ^ lookup[ 4*256 + ((uint8)(word >> 32)) ]
		       ^ lookup[ 5*256 + ((uint8)(word >> 40)) ]
		       ^ lookup[ 6*256 + ((uint8)(word >> 48)) ]
		       ^ lookup[ 7*256 + ((uint8)(word >> 56)) ];
	}
}

#ifdef __cplusplus
}
#endif
