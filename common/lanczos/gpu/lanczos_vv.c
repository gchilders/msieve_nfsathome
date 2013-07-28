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

#include "lanczos_gpu.h"

/*-------------------------------------------------------------------*/
void *v_alloc(uint32 n, void *extra) {

	gpuvec_t *v = (gpuvec_t *)xmalloc(sizeof(gpuvec_t));

	v->gpudata = (gpudata_t *)extra;

	v->host_vec = malloc(n * sizeof(uint64));

	CUDA_TRY(cuMemAlloc(&v->gpu_vec, n * sizeof(uint64)))

	return v;
}

void v_free(void *v_in) {

	gpuvec_t *v = (gpuvec_t *)v_in;

	CUDA_TRY(cuMemFree(v->gpu_vec))

	free(v->host_vec);
	free(v);
}

void v_copyin(void *dest_in, uint64 *src, uint32 n) {

	gpuvec_t *dest = (gpuvec_t *)dest_in;

	memcpy(dest->host_vec, src, n * sizeof(uint64));

	CUDA_TRY(cuMemcpyHtoD(dest->gpu_vec, src, n * sizeof(uint64)))
}

void v_copy(void *dest_in, void *src_in, uint32 n) {

	gpuvec_t *src = (gpuvec_t *)src_in;
	gpuvec_t *dest = (gpuvec_t *)dest_in;

	memcpy(dest->host_vec, src->host_vec, n * sizeof(uint64));

	CUDA_TRY(cuMemcpyDtoD(dest->gpu_vec, src->gpu_vec, n * sizeof(uint64)))
}

void v_copyout(uint64 *dest, void *src_in, uint32 n) {

	gpuvec_t *src = (gpuvec_t *)src_in;

	memcpy(dest, src->host_vec, n * sizeof(uint64));
}

void v_clear(void *v_in, uint32 n) {

	gpuvec_t * v = (gpuvec_t *)v_in;

	memset(v->host_vec, 0, n * sizeof(uint64));

	CUDA_TRY(cuMemsetD8(v->gpu_vec, 0, n * sizeof(uint64)));
}

void v_xor(void *dest_in, void *src_in, uint32 n) {

	gpuvec_t *src = (gpuvec_t *)src_in;
	gpuvec_t *dest = (gpuvec_t *)dest_in;
	gpudata_t *d = src->gpudata;
	gpu_launch_t *launch = d->launch + GPU_K_XOR;
	gpu_arg_t gpu_args[GPU_MAX_KERNEL_ARGS];

	uint32 num_blocks = (n + launch->threads_per_block - 1) / 
				launch->threads_per_block;

	gpu_args[0].ptr_arg = (void *)(size_t)dest->gpu_vec;
	gpu_args[1].ptr_arg = (void *)(size_t)src->gpu_vec;
	gpu_args[2].uint32_arg = n;
	gpu_launch_set(launch, gpu_args);

	CUDA_TRY(cuLaunchGrid(launch->kernel_func, MIN(1000, num_blocks), 1))

	CUDA_TRY(cuMemcpyDtoH(dest->host_vec, dest->gpu_vec, 
				n * sizeof(uint64)))
}

void v_mask(void *v_in, uint64 mask, uint32 n) {

	gpuvec_t *v = (gpuvec_t *)v_in;
	gpudata_t *d = v->gpudata;
	gpu_launch_t *launch = d->launch + GPU_K_MASK;
	gpu_arg_t gpu_args[GPU_MAX_KERNEL_ARGS];

	uint32 num_blocks = (n + launch->threads_per_block - 1) / 
				launch->threads_per_block;

	gpu_args[0].ptr_arg = (void *)(size_t)v->gpu_vec;
	gpu_args[1].uint64_arg = mask;
	gpu_args[2].uint32_arg = n;
	gpu_launch_set(launch, gpu_args);

	CUDA_TRY(cuLaunchGrid(launch->kernel_func, MIN(1000, num_blocks), 1))

	CUDA_TRY(cuMemcpyDtoH(v->host_vec, v->gpu_vec, n * sizeof(uint64)))
}

/*-------------------------------------------------------------------*/
static void core_Nx64_64x64_acc(uint64 *v, uint64 *c,
			uint64 *y, uint32 n) {

	uint32 i;

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
	ASM_M
	{
		push    ebx
		mov	    edi,c
		mov	    esi,v
		mov     ebx,y
		xor	    ecx,ecx
		align 16
	L0:	movq	mm0,[ebx+ecx*8]
		mov	eax,[esi+ecx*8]
		inc	ecx
		movzx	edx, al
		movq	mm1,[edi+edx*8]
		movzx	edx,ah
		pxor	mm1,[1*256*8+edi+edx*8]
		shr	eax,16
		movzx	edx,al
		pxor	mm1,[2*256*8+edi+edx*8]
		movzx	edx,ah
		pxor	mm1,[3*256*8+edi+edx*8]
		mov	eax,[4-8+esi+ecx*8]
		movzx	edx,al
		pxor	mm1,[4*256*8+edi+edx*8]
		movzx	edx,ah
		shr	eax,16
		cmp	ecx,n
		pxor	mm1,[5*256*8+edi+edx*8]
		movzx	edx,al
		pxor	mm1,[6*256*8+edi+edx*8]
		movzx	edx,ah
		pxor	mm1,[7*256*8+edi+edx*8]
		pxor	mm1, mm0
		movq	[-8+ebx+ecx*8],mm1
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
static const uint8 graycode[2 * 256] = {
   0, 0,    1, 0,    3, 1,    2, 0,    6, 2,    7, 0,    5, 1,    4, 0,   
  12, 3,   13, 0,   15, 1,   14, 0,   10, 2,   11, 0,    9, 1,    8, 0,   
  24, 4,   25, 0,   27, 1,   26, 0,   30, 2,   31, 0,   29, 1,   28, 0,   
  20, 3,   21, 0,   23, 1,   22, 0,   18, 2,   19, 0,   17, 1,   16, 0,   
  48, 5,   49, 0,   51, 1,   50, 0,   54, 2,   55, 0,   53, 1,   52, 0,  
  60, 3,   61, 0,   63, 1,   62, 0,   58, 2,   59, 0,   57, 1,   56, 0,   
  40, 4,   41, 0,   43, 1,   42, 0,   46, 2,   47, 0,   45, 1,   44, 0,  
  36, 3,   37, 0,   39, 1,   38, 0,   34, 2,   35, 0,   33, 1,   32, 0,   
  96, 6,   97, 0,   99, 1,   98, 0,  102, 2,  103, 0,  101, 1,  100, 0,  
 108, 3,  109, 0,  111, 1,  110, 0,  106, 2,  107, 0,  105, 1,  104, 0,  
 120, 4,  121, 0,  123, 1,  122, 0,  126, 2,  127, 0,  125, 1,  124, 0,  
 116, 3,  117, 0,  119, 1,  118, 0,  114, 2,  115, 0,  113, 1,  112, 0,  
  80, 5,   81, 0,   83, 1,   82, 0,   86, 2,   87, 0,   85, 1,   84, 0,   
  92, 3,   93, 0,   95, 1,   94, 0,   90, 2,   91, 0,   89, 1,   88, 0,   
  72, 4,   73, 0,   75, 1,   74, 0,   78, 2,   79, 0,   77, 1,   76, 0,   
  68, 3,   69, 0,   71, 1,   70, 0,   66, 2,   67, 0,   65, 1,   64, 0,  
 192, 7,  193, 0,  195, 1,  194, 0,  198, 2,  199, 0,  197, 1,  196, 0, 
 204, 3,  205, 0,  207, 1,  206, 0,  202, 2,  203, 0,  201, 1,  200, 0, 
 216, 4,  217, 0,  219, 1,  218, 0,  222, 2,  223, 0,  221, 1,  220, 0, 
 212, 3,  213, 0,  215, 1,  214, 0,  210, 2,  211, 0,  209, 1,  208, 0, 
 240, 5,  241, 0,  243, 1,  242, 0,  246, 2,  247, 0,  245, 1,  244, 0, 
 252, 3,  253, 0,  255, 1,  254, 0,  250, 2,  251, 0,  249, 1,  248, 0, 
 232, 4,  233, 0,  235, 1,  234, 0,  238, 2,  239, 0,  237, 1,  236, 0, 
 228, 3,  229, 0,  231, 1,  230, 0,  226, 2,  227, 0,  225, 1,  224, 0,  
 160, 6,  161, 0,  163, 1,  162, 0,  166, 2,  167, 0,  165, 1,  164, 0, 
 172, 3,  173, 0,  175, 1,  174, 0,  170, 2,  171, 0,  169, 1,  168, 0, 
 184, 4,  185, 0,  187, 1,  186, 0,  190, 2,  191, 0,  189, 1,  188, 0, 
 180, 3,  181, 0,  183, 1,  182, 0,  178, 2,  179, 0,  177, 1,  176, 0, 
 144, 5,  145, 0,  147, 1,  146, 0,  150, 2,  151, 0,  149, 1,  148, 0, 
 156, 3,  157, 0,  159, 1,  158, 0,  154, 2,  155, 0,  153, 1,  152, 0, 
 136, 4,  137, 0,  139, 1,  138, 0,  142, 2,  143, 0,  141, 1,  140, 0, 
 132, 3,  133, 0,  135, 1,  134, 0,  130, 2,  131, 0,  129, 1,  128, 0,  
};

static void mul_precomp_256x8(uint64 *c, uint64 *x) {

	/* Let c[][] be an 8 x 256 scratch matrix of 64-bit words;
	   let x[][] be a 64 x 64 matrix

	   Fill c[][] with a bunch of "partial matrix multiplies". 
	   For 0<=i<256, the j_th row of c[][] contains the matrix 
	   product

	   	( i << (8*j) ) * x[][]

	   where the quantity in parentheses is considered a 
	   1 x 64 vector of elements in GF(2). The resulting
	   table will dramatically speed up matrix multiplies
	   by x[][]. 
	 
	   We iterate through i in Gray code order and unroll
	   by 8 to minimize overhead */

	uint32 i;
	uint64 c0, c1, c2, c3, c4, c5, c6, c7;

	c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0;

	c[0*256] = c[1*256] = c[2*256] = c[3*256] = 
	c[4*256] = c[5*256] = c[6*256] = c[7*256] = 0;

	for (i = 1; i < 256; i++) {

		uint32 word = graycode[2 * i];
		uint32 bit = graycode[2 * i + 1];

		c0 ^= x[0*8 + bit]; c[0*256 + word] = c0;
		c1 ^= x[1*8 + bit]; c[1*256 + word] = c1;
		c2 ^= x[2*8 + bit]; c[2*256 + word] = c2;
		c3 ^= x[3*8 + bit]; c[3*256 + word] = c3;
		c4 ^= x[4*8 + bit]; c[4*256 + word] = c4;
		c5 ^= x[5*8 + bit]; c[5*256 + word] = c5;
		c6 ^= x[6*8 + bit]; c[6*256 + word] = c6;
		c7 ^= x[7*8 + bit]; c[7*256 + word] = c7;
	}
}

static void mul_precomp_64x11(uint64 *c, uint64 *x) {

	/* As above, except that the main loop breaks 64-bit
	   words into 11 groups of 6 bits */

	uint32 i;
	uint64 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10;

	c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = 0;

	c[0*64] = c[1*64] = c[2*64] = c[3*64] = 
	c[4*64] = c[5*64] = c[6*64] = c[7*64] = 
	c[8*64] = c[9*64] = c[10*64] = 0;

	for (i = 1; i < 64; i++) {

		uint32 word = graycode[2 * i];
		uint32 bit = graycode[2 * i + 1];

		c0 ^= x[0*6 + bit]; c[0*64 + word] = c0;
		c1 ^= x[1*6 + bit]; c[1*64 + word] = c1;
		c2 ^= x[2*6 + bit]; c[2*64 + word] = c2;
		c3 ^= x[3*6 + bit]; c[3*64 + word] = c3;
		c4 ^= x[4*6 + bit]; c[4*64 + word] = c4;
		c5 ^= x[5*6 + bit]; c[5*64 + word] = c5;
		c6 ^= x[6*6 + bit]; c[6*64 + word] = c6;
		c7 ^= x[7*6 + bit]; c[7*64 + word] = c7;
		c8 ^= x[8*6 + bit]; c[8*64 + word] = c8;
		c9 ^= x[9*6 + bit]; c[9*64 + word] = c9;
		if (i < 16)
			c10 ^= x[10*6 + bit]; c[10*64 + word] = c10;
	}
}

/*-------------------------------------------------------------------*/
void v_mul_Nx64_64x64_acc_cpu(uint64 *v, uint64 *x,
			uint64 *y, uint32 n) {

	uint64 c[8 * 256];

	mul_precomp_256x8(c, x);

	core_Nx64_64x64_acc(v, c, y, n);
}

/*-------------------------------------------------------------------*/
void v_mul_Nx64_64x64_acc(packed_matrix_t *matrix, 
			void *v_in, uint64 *x,
			void *y_in, uint32 n) {

	uint64 c[8 * 256];
	gpuvec_t *v = (gpuvec_t *)v_in;
	gpuvec_t *y = (gpuvec_t *)y_in;
	gpudata_t *d = (gpudata_t *)matrix->extra;
	gpu_launch_t *launch = d->launch + GPU_K_INNER_PROD;
	gpu_arg_t gpu_args[GPU_MAX_KERNEL_ARGS];
	uint32 num_blocks = (n + launch->threads_per_block - 1) / 
				launch->threads_per_block;

	mul_precomp_64x11(c, x);

	CUDA_TRY(cuMemcpyHtoD(d->inner_scratch, c, 
				256 * 8 * sizeof(uint64)))

	gpu_args[0].ptr_arg = (void *)(size_t)y->gpu_vec;
	gpu_args[1].ptr_arg = (void *)(size_t)v->gpu_vec;
	gpu_args[2].ptr_arg = (void *)(size_t)d->inner_scratch;
	gpu_args[3].uint32_arg = n;
	gpu_launch_set(launch, gpu_args);

	CUDA_TRY(cuLaunchGrid(launch->kernel_func, MIN(1000, num_blocks), 1))

#ifdef LANCZOS_GPU_DEBUG
	{
		uint64 *tmp = (uint64 *)xmalloc(n * sizeof(uint64));
		uint32 i;

		v_mul_Nx64_64x64_acc_cpu(v->host_vec, x, y->host_vec, n);

		CUDA_TRY(cuMemcpyDtoH(tmp, y->gpu_vec, n * sizeof(uint64)))

		for (i = 0; i < n; i++) {
			if (y->host_vec[i] != tmp[i]) {
				printf("error offset %u\n", i);
				exit(-1);
			}
		}
		free(tmp);
	}
#else
	CUDA_TRY(cuMemcpyDtoH(y->host_vec, y->gpu_vec, n * sizeof(uint64)))
#endif
}

/*-------------------------------------------------------------------*/
static void core_64xN_Nx64(uint64 *x, uint64 *c, 
			uint64 *y, uint32 n) {

	uint32 i;

	memset(c, 0, 8 * 256 * sizeof(uint64));

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
	ASM_M
	{
		push    ebx
		mov	    edi,c
		mov	    esi,x
		mov     ebx,y
		xor	    ecx,ecx
		align 16
    L0:	movq	mm0,[ebx+ecx*8]
		mov	    eax,[esi+ecx*8]
		inc	    ecx
		movzx	edx,al
		movq	mm1,mm0
		pxor	mm1,[edi+edx*8]
		movq	[edi+edx*8],mm1
		movzx	edx,ah
		movq	mm1, mm0
		pxor	mm1,[1*256*8+edi+edx*8]
		movq	[1*256*8+edi+edx*8],mm1
		shr	    eax,16
		movzx	edx,al
		movq	mm1,mm0
		pxor	mm1,[2*256*8+edi+edx*8]
		movq	[2*256*8+edi+edx*8],mm1
		movzx	edx,ah
		movq	mm1,mm0
		pxor	mm1,[3*256*8+edi+edx*8]
		movq	[3*256*8+edi+edx*8],mm1
		mov	    eax,[4-8+esi+ecx*8]
		movzx	edx,al
		movq	mm1,mm0
		pxor	mm1,[4*256*8+edi+edx*8]
		movq	[4*256*8+edi+edx*8],mm1
		movzx	edx,ah
		shr	    eax,16
		cmp	    ecx,n
		movq	mm1,mm0
		pxor	mm1,[5*256*8+edi+edx*8]
		movq	[5*256*8+edi+edx*8],mm1
		movzx	edx,al
		movq	mm1,mm0
		pxor	mm1,[6*256*8+edi+edx*8]
		movq	[6*256*8+edi+edx*8],mm1
		movzx	edx,ah
		pxor	mm0,[7*256*8+edi+edx*8]
		movq	[7*256*8+edi+edx*8],mm0
		jne	    L0
		emms 
		pop     ebx
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
}

/*-------------------------------------------------------------------*/
static void mul_64xN_Nx64_postproc(uint64 *c, uint64 *xy) {

	uint32 i, j;

	for (i = 0; i < 8; i++) {

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

/*-------------------------------------------------------------------*/
void v_mul_64xN_Nx64_cpu(uint64 *x, uint64 *y,
		   uint64 *xy, uint32 n) {

	uint64 c[8 * 256];

	core_64xN_Nx64(x, c, y, n);

	mul_64xN_Nx64_postproc(c, xy);
}

/*-------------------------------------------------------------------*/
void v_mul_64xN_Nx64_gpu(packed_matrix_t *matrix,
		   CUdeviceptr x, CUdeviceptr y,
		   CUdeviceptr xy, uint32 n) {


	gpudata_t *d = (gpudata_t *)matrix->extra;
	gpu_launch_t *launch = d->launch + GPU_K_OUTER_PROD;
	gpu_arg_t gpu_args[GPU_MAX_KERNEL_ARGS];
	uint32 num_blocks = (n + launch->threads_per_block - 1) / 
				launch->threads_per_block;

	num_blocks = MIN(num_blocks, 
			(uint32)(10 * d->gpu_info->num_compute_units));

	gpu_args[0].ptr_arg = (void *)(size_t)x;
	gpu_args[1].ptr_arg = (void *)(size_t)y;
	gpu_args[2].ptr_arg = (void *)(size_t)xy;
	gpu_args[3].uint32_arg = n;
	gpu_launch_set(launch, gpu_args);

	CUDA_TRY(cuLaunchGrid(launch->kernel_func, num_blocks, 1))
}

/*-------------------------------------------------------------------*/
void v_mul_64xN_Nx64(packed_matrix_t *matrix,
		   void *x_in, void *y_in,
		   uint64 *xy, uint32 n) {


	gpuvec_t *x = (gpuvec_t *)x_in;
	gpuvec_t *y = (gpuvec_t *)y_in;
	gpudata_t *d = (gpudata_t *)matrix->extra;

	CUDA_TRY(cuMemsetD8(d->outer_scratch, 0, 64 * sizeof(uint64)));

	v_mul_64xN_Nx64_gpu(matrix, x->gpu_vec, y->gpu_vec, 
			d->outer_scratch, n);

#ifdef LANCZOS_GPU_DEBUG
	{
		uint64 tmp[64];
		uint32 i;

		v_mul_64xN_Nx64_cpu(x->host_vec, y->host_vec, xy, n);

		CUDA_TRY(cuMemcpyDtoH(tmp, d->outer_scratch, 
					64 * sizeof(uint64)))

		for (i = 0; i < 64; i++) {
			if (xy[i] != tmp[i]) {
				printf("error offset %u\n", i);
				exit(-1);
			}
		}
	}
#else
	CUDA_TRY(cuMemcpyDtoH(xy, d->outer_scratch, 64 * sizeof(uint64)))
#endif
}
