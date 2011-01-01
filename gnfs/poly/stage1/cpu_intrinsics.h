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

#ifndef CPU_INTRINSICS_H
#define CPU_INTRINSICS_H

#include <mp.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef _MSC_VER

#include <intrin.h>
#pragma intrinsic(__emulu)

#define PROD32(hi, lo, a, b)		\
	{	uint64 __t = __emulu(a,b);	\
		hi = (uint32)(__t >> 32);	\
		lo = (uint32)(__t); }

#elif defined(GCC_ASM32X)

#define PROD32(hi, lo, a, b) \
	asm("mull %2  \n\t"      \
	    :"=d"(hi), "=a"(lo)  \
	    :"%rm"(a), "1"(b)    \
	    :"cc")

#else

#define PROD32(hi, lo, a, b) \
	{ uint64 t = (uint64)(a) * (b); \
	  hi = (uint32)(t >> 32);	\
	  lo = (uint32)(t); }

#endif

/*-------------------- Addition ---------------------------------*/
static INLINE uint128
add128(uint128 a, uint128 b)
{
	uint32 c;
	uint32 acc;
	uint128 res;

	acc = a.w[0] + b.w[0];
	res.w[0] = acc;
	c = (acc < a.w[0]);

	acc = a.w[1] + c;
	c = (acc < a.w[1]);
	res.w[1] = acc + b.w[1];
	c += (res.w[1] < acc);

	acc = a.w[2] + c;
	c = (acc < a.w[2]);
	res.w[2] = acc + b.w[2];
	c += (res.w[2] < acc);

	res.w[3] = a.w[3] + b.w[3] + c;
	return res;
}

/*----------------- Multiplication ----------------------------------*/

static INLINE uint128
mul64(uint64 a, uint64 b)
{
	uint32 a0 = (uint32)a;
	uint32 a1 = (uint32)(a >> 32);
	uint32 b0 = (uint32)b;
	uint32 b1 = (uint32)(b >> 32);
	uint64 acc;
	uint32 prod_lo, prod_hi;
	uint128 res;

	PROD32(prod_hi, prod_lo, a0, b0);
	res.w[0] = prod_lo;
	acc = (uint64)prod_hi;

	PROD32(prod_hi, prod_lo, a0, b1);
	acc += (uint64)prod_hi << 32 | prod_lo;

	PROD32(prod_hi, prod_lo, a1, b0);
	acc += prod_lo;
	res.w[1] = (uint32)acc;
	acc = (acc >> 32) + prod_hi;

	PROD32(prod_hi, prod_lo, a1, b1);
	acc += (uint64)prod_hi << 32 | prod_lo;
	res.w[2] = (uint32)acc;
	res.w[3] = (uint32)(acc >> 32);

	return res;
}

/*------------------------- Miscellaneous -------------------------------*/

#define HOST_BATCH_SIZE 8192

#define INVERT_BATCH_SIZE 512

#ifdef __cplusplus
}
#endif

#endif /* !CPU_INTRINSICS_H */

