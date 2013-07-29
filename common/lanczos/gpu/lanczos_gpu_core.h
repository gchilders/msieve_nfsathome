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

#ifndef _COMMON_LANCZOS_GPU_LANCZOS_GPU_CORE_H_
#define _COMMON_LANCZOS_GPU_LANCZOS_GPU_CORE_H_

#if defined(__CUDACC__) /*------------- device code -------------*/

typedef short int16;
typedef unsigned short uint16;
typedef int int32;
typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef long long int64;

#if defined(_WIN64) || defined(__LP64__)
	#define PTR_CONSTRAINT(x) "l"(x)
#else
	#define PTR_CONSTRAINT(x) "r"(x)
#endif

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

	return (uint32)(x >> pos) & ((1 << bits) - 1);
#endif
}

__device__ uint64
load_bypassL1(uint64 *addr)
{
#if __CUDA_ARCH__ >= 200

	uint64 res;

	asm("ld.global.cg.u64 %0, [%1]; \n\t"
		: "=l"(res) : PTR_CONSTRAINT(addr));

	return res;
#else
	return addr[0];
#endif
}

__device__ uint32
load_bypassL1(uint32 *addr)
{
#if __CUDA_ARCH__ >= 200

	uint32 res;

	asm("ld.global.cg.u32 %0, [%1]; \n\t"
		: "=r"(res) : PTR_CONSTRAINT(addr));

	return res;
#else
	return addr[0];
#endif
}

__device__ void
store_bypassL1(uint64 x, uint64 *addr)
{
#if __CUDA_ARCH__ >= 200

	asm("st.global.cg.u64 [%0], %1; \n\t"
		: : PTR_CONSTRAINT(addr), "l"(x));
#else
	addr[0] = x;
#endif
}

__device__ void
store_bypassL1(uint32 x, uint32 *addr)
{
#if __CUDA_ARCH__ >= 200

	asm("st.global.cg.u32 [%0], %1; \n\t"
		: : PTR_CONSTRAINT(addr), "r"(x));
#else
	addr[0] = x;
#endif
}

__device__ uint32
load_streaming(uint32 *addr)
{
#if __CUDA_ARCH__ >= 200

	uint32 res;

	asm("ld.global.cs.u32 %0, [%1]; \n\t"
		: "=r"(res) : PTR_CONSTRAINT(addr));

	return res;
#else
	return addr[0];
#endif
}

__device__ void
store_streaming(uint32 x, uint32 *addr)
{
#if __CUDA_ARCH__ >= 200

	asm("st.global.cs.u32 [%0], %1; \n\t"
		: : PTR_CONSTRAINT(addr), "r"(x));
#else
	addr[0] = x;
#endif
}

__device__ uint64
uint2_to_uint64(uint2 v)
{
	return (uint64)v.y << 32 | v.x;
}

#endif  /*--------------------------- device code -----------------*/

#ifdef __cplusplus
extern "C" {
#endif

#define MATMUL_THREADS 256

typedef union {

	uint32 w;

	struct {
		uint16 row_off : 15;
		uint16 row_off_head : 1;
		uint16 col_off : 15;
		uint16 col_off_head : 1;
	} d; 
} gpu_entry_idx_t;

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_LANCZOS_GPU_LANCZOS_GPU_CORE_H_ */
