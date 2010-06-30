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

#include "lanczos.h"

/*-------------------------------------------------------------------*/
void mul_Nx64_64x64_acc(uint64 *v, uint64 *x,
			uint64 *y, uint32 n) {

	/* let v[][] be a n x 64 matrix with elements in GF(2), 
	   represented as an array of n 64-bit words. Let c[][]
	   be an 8 x 256 scratch matrix of 64-bit words.
	   This code multiplies v[][] by the 64x64 matrix 
	   x[][], then XORs the n x 64 result into y[][] */

	uint32 i, j, k;
	uint64 c[8 * 256];

	/* fill c[][] with a bunch of "partial matrix multiplies". 
	   For 0<=i<256, the j_th row of c[][] contains the matrix 
	   product

	   	( i << (8*j) ) * x[][]

	   where the quantity in parentheses is considered a 
	   1 x 64 vector of elements in GF(2). The resulting
	   table will dramatically speed up matrix multiplies
	   by x[][]. */

	for (i = 0; i < 8; i++) {
		uint64 *xtmp = x + 8 * i;
		uint64 *ctmp = c + 256 * i;

		for (j = 0; j < 256; j++) {
			uint64 accum = 0;
			uint32 index = j;

			for (k = 0; k < 8; k++) {
				if (index & ((uint32)1 << k))
					accum ^= xtmp[k];
			}
			ctmp[j] = accum;
		}
	}

#if defined(GCC_ASM32A) && defined(HAS_MMX) && defined(NDEBUG)
	i = 0;
	ASM_G volatile(
		     ALIGN_LOOP
		     "0:                                   \n\t"
		     "movq (%3,%0,8), %%mm0                \n\t"
		     "movl (%1,%0,8), %%eax                \n\t"
		     "incl %0                              \n\t"
		     "movzbl %%al, %%ecx                   \n\t"
		     "movq (%2,%%ecx,8), %%mm1             \n\t"
		     "movzbl %%ah, %%ecx                   \n\t"
		     "pxor 1*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "shrl $16, %%eax                      \n\t"
		     "movzbl %%al, %%ecx                   \n\t"
		     "pxor 2*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movzbl %%ah, %%ecx                   \n\t"
		     "pxor 3*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movl 4-8(%1,%0,8), %%eax             \n\t"
		     "movzbl %%al, %%ecx                   \n\t"
		     "pxor 4*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movzbl %%ah, %%ecx                   \n\t"
		     "shrl $16, %%eax                      \n\t"
		     "cmpl %4, %0                          \n\t"
		     "pxor 5*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movzbl %%al, %%ecx                   \n\t"
		     "pxor 6*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movzbl %%ah, %%ecx                   \n\t"
		     "pxor 7*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "pxor %%mm0, %%mm1                    \n\t"
		     "movq %%mm1, -8(%3,%0,8)              \n\t"
		     "jne 0b                               \n\t"
		     "emms                                 \n\t"
			:"+r"(i)
			:"r"(v), "r"(c), "r"(y), "g"(n)
			:"%eax", "%ecx", "%mm0", "%mm1", "memory");

#elif defined(MSC_ASM32A)
	i = 0;
	ASM_M
	{
		push	ebx
		mov	edi,y
		lea	ebx,c
		mov	esi,v
		mov	ecx,i
		align 16
	L0:	movq	mm0,[edi+ecx*8]
		mov	eax,[esi+ecx*8]
		inc	ecx
		movzx	edx, al
		movq	mm1,[ebx+edx*8]
		movzx	edx,ah
		pxor	mm1,[1*256*8+ebx+edx*8]
		shr	eax,16
		movzx	edx,al
		pxor	mm1,[2*256*8+ebx+edx*8]
		movzx	edx,ah
		pxor	mm1,[3*256*8+ebx+edx*8]
		mov	eax,[4-8+esi+ecx*8]
		movzx	edx,al
		pxor	mm1,[4*256*8+ebx+edx*8]
		movzx	edx,ah
		shr	eax,16
		cmp	ecx,n
		pxor	mm1,[5*256*8+ebx+edx*8]
		movzx	edx,al
		pxor	mm1,[6*256*8+ebx+edx*8]
		movzx	edx,ah
		pxor	mm1,[7*256*8+ebx+edx*8]
		pxor	mm1, mm0
		movq	[-8+edi+ecx*8],mm1
		jne	L0
		pop	ebx
		emms
	}
#else
	for (i = 0; i < n; i++) {
		uint64 word = v[i];
		y[i] ^=  c[ 0*256 + ((uint8)(word >>  0)) ]
		       ^ c[ 1*256 + ((uint8)(word >>  8)) ]
		       ^ c[ 2*256 + ((uint8)(word >> 16)) ]
		       ^ c[ 3*256 + ((uint8)(word >> 24)) ]
		       ^ c[ 4*256 + ((uint8)(word >> 32)) ]
		       ^ c[ 5*256 + ((uint8)(word >> 40)) ]
		       ^ c[ 6*256 + ((uint8)(word >> 48)) ]
		       ^ c[ 7*256 + ((uint8)(word >> 56)) ];
	}
#endif
}

/*-------------------------------------------------------------------*/
void mul_64xN_Nx64(uint64 *x, uint64 *y,
		   uint64 *xy, uint32 n) {

	/* Let x and y be n x 64 matrices. This routine computes
	   the 64 x 64 matrix xy[][] given by transpose(x) * y */

	uint32 i;
	uint64 c[8 * 256] = {0};

	memset(xy, 0, 64 * sizeof(uint64));

#if defined(GCC_ASM32A) && defined(HAS_MMX) && defined(NDEBUG)
	i = 0;
	ASM_G volatile(
		     ALIGN_LOOP
		     "0:                                   \n\t"
		     "movq (%3,%0,8), %%mm0                \n\t"
		     "movl (%1,%0,8), %%eax                \n\t"
		     "incl %0                              \n\t"
		     "movzbl %%al, %%ecx                   \n\t"
		     "movq %%mm0, %%mm1                    \n\t"
		     "pxor (%2,%%ecx,8), %%mm1             \n\t"
		     "movq %%mm1, (%2,%%ecx,8)             \n\t"
		     "movzbl %%ah, %%ecx                   \n\t"
		     "movq %%mm0, %%mm1                    \n\t"
		     "pxor 1*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movq %%mm1, 1*256*8(%2,%%ecx,8)      \n\t"
		     "shrl $16, %%eax                      \n\t"
		     "movzbl %%al, %%ecx                   \n\t"
		     "movq %%mm0, %%mm1                    \n\t"
		     "pxor 2*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movq %%mm1, 2*256*8(%2,%%ecx,8)      \n\t"
		     "movzbl %%ah, %%ecx                   \n\t"
		     "movq %%mm0, %%mm1                    \n\t"
		     "pxor 3*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movq %%mm1, 3*256*8(%2,%%ecx,8)      \n\t"
		     "movl 4-8(%1,%0,8), %%eax             \n\t"
		     "movzbl %%al, %%ecx                   \n\t"
		     "movq %%mm0, %%mm1                    \n\t"
		     "pxor 4*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movq %%mm1, 4*256*8(%2,%%ecx,8)      \n\t"
		     "movzbl %%ah, %%ecx                   \n\t"
		     "shrl $16, %%eax                      \n\t"
		     "cmpl %4, %0                          \n\t"
		     "movq %%mm0, %%mm1                    \n\t"
		     "pxor 5*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movq %%mm1, 5*256*8(%2,%%ecx,8)      \n\t"
		     "movzbl %%al, %%ecx                   \n\t"
		     "movq %%mm0, %%mm1                    \n\t"
		     "pxor 6*256*8(%2,%%ecx,8), %%mm1      \n\t"
		     "movq %%mm1, 6*256*8(%2,%%ecx,8)      \n\t"
		     "movzbl %%ah, %%ecx                   \n\t"
		     "pxor 7*256*8(%2,%%ecx,8), %%mm0      \n\t"
		     "movq %%mm0, 7*256*8(%2,%%ecx,8)      \n\t"
		     "jne 0b                               \n\t"
		     "emms                                 \n\t"
			:"+r"(i)
			:"r"(x), "r"(c), "r"(y), "g"(n)
			:"%eax", "%ecx", "%mm0", "%mm1", "memory");

#elif defined(MSC_ASM32A)
	i = 0;
	ASM_M
	{
		push	ebx
		mov	edi,y
		lea	ebx,c
		mov	esi,x
		mov	ecx,i
		align 16
	L0:	movq	mm0,[edi+ecx*8]
		mov	eax,[esi+ecx*8]
		inc	ecx
		movzx	edx,al
		movq	mm1,mm0
		pxor	mm1,[ebx+edx*8]
		movq	[ebx+edx*8],mm1
		movzx	edx,ah
		movq	mm1, mm0
		pxor	mm1,[1*256*8+ebx+edx*8]
		movq	[1*256*8+ebx+edx*8],mm1
		shr	eax,16
		movzx	edx,al
		movq	mm1,mm0
		pxor	mm1,[2*256*8+ebx+edx*8]
		movq	[2*256*8+ebx+edx*8],mm1
		movzx	edx,ah
		movq	mm1,mm0
		pxor	mm1,[3*256*8+ebx+edx*8]
		movq	[3*256*8+ebx+edx*8],mm1
		mov	eax,[4-8+esi+ecx*8]
		movzx	edx,al
		movq	mm1,mm0
		pxor	mm1,[4*256*8+ebx+edx*8]
		movq	[4*256*8+ebx+edx*8],mm1
		movzx	edx,ah
		shr	eax,16
		cmp	ecx,n
		movq	mm1,mm0
		pxor	mm1,[5*256*8+ebx+edx*8]
		movq	[5*256*8+ebx+edx*8],mm1
		movzx	edx,al
		movq	mm1,mm0
		pxor	mm1,[6*256*8+ebx+edx*8]
		movq	[6*256*8+ebx+edx*8],mm1
		movzx	edx,ah
		pxor	mm0,[7*256*8+ebx+edx*8]
		movq	[7*256*8+ebx+edx*8],mm0
		jne	L0
		emms
		pop	ebx
	}
#else

	for (i = 0; i < n; i++) {
		uint64 xi = x[i];
		uint64 yi = y[i];
		c[ 0*256 + ((uint8) xi       ) ] ^= yi;
		c[ 1*256 + ((uint8)(xi >>  8)) ] ^= yi;
		c[ 2*256 + ((uint8)(xi >> 16)) ] ^= yi;
		c[ 3*256 + ((uint8)(xi >> 24)) ] ^= yi;
		c[ 4*256 + ((uint8)(xi >> 32)) ] ^= yi;
		c[ 5*256 + ((uint8)(xi >> 40)) ] ^= yi;
		c[ 6*256 + ((uint8)(xi >> 48)) ] ^= yi;
		c[ 7*256 + ((uint8)(xi >> 56)) ] ^= yi;
	}
#endif

	for(i = 0; i < 8; i++) {

		uint32 j;
		uint64 a0, a1, a2, a3, a4, a5, a6, a7;

		a0 = a1 = a2 = a3 = 0;
		a4 = a5 = a6 = a7 = 0;

		for (j = 0; j < 256; j++) {
			if ((j >> i) & 1) {
				a0 ^= c[0*256 + j];
				a1 ^= c[1*256 + j];
				a2 ^= c[2*256 + j];
				a3 ^= c[3*256 + j];
				a4 ^= c[4*256 + j];
				a5 ^= c[5*256 + j];
				a6 ^= c[6*256 + j];
				a7 ^= c[7*256 + j];
			}
		}

		xy[ 0] = a0; xy[ 8] = a1; xy[16] = a2; xy[24] = a3;
		xy[32] = a4; xy[40] = a5; xy[48] = a6; xy[56] = a7;
		xy++;
	}
}
