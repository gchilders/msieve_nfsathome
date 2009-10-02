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

#include "stage1.h"

#define P_SMALL_BATCH_SIZE 1024

typedef struct {
	uint32 lattice_size[P_SMALL_BATCH_SIZE];
	uint32 p[P_SMALL_BATCH_SIZE];
	uint64 roots[P_SMALL_BATCH_SIZE];
} p_small_batch_t;

#define P_LARGE_BATCH_SIZE 16384

typedef struct {
	uint32 p[P_LARGE_BATCH_SIZE];
	uint64 roots[P_LARGE_BATCH_SIZE];
} p_large_batch_t;

typedef struct {
	uint32 p;
	uint32 q;
	uint64 offset;
	uint64 proot;
} found_t;

/*------------------------------------------------------------------------*/
static void 
store_small_p(uint64 p, uint32 num_roots,
		mpz_t *roots, void *extra)
{
	lattice_fb_t *L = (lattice_fb_t *)extra;
	p_small_batch_t *batch = (p_small_batch_t *)L->p_array;
	uint32 num;

	if (num_roots != 1) {
		printf("error: too many q roots\n");
		exit(-1);
	}
	num = L->num_p;
	batch->p[num] = (uint32)p;
	batch->lattice_size[num] = L->poly->sieve_size / ((double)p * p);
	batch->roots[num] = gmp2uint64(roots[0]);
	L->num_p++;
}

/*------------------------------------------------------------------------*/
static void 
store_large_p(uint64 p, uint32 num_roots,
		mpz_t *roots, void *extra)
{
	lattice_fb_t *L = (lattice_fb_t *)extra;
	p_large_batch_t *batch = (p_large_batch_t *)L->q_array;
	uint32 num;

	if (num_roots != 1) {
		printf("error: too many q roots\n");
		exit(-1);
	}
	num = L->num_q;
	batch->p[num] = (uint32)p;
	batch->roots[num] = gmp2uint64(roots[0]);
	L->num_q++;
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_gpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_small, sieve_fb_t *sieve_large, 
		uint32 small_p_min, uint32 small_p_max, 
		uint32 large_p_min, uint32 large_p_max,
		gpu_info_t *gpu_info, CUfunction gpu_kernel)
{
	uint32 i;
	uint32 min_small, min_large;
	uint32 quit = 0;
	p_small_batch_t * p_array;
	p_large_batch_t * q_array;
	found_t *found_array;
	uint32 found_array_size;

	int32 num_p_offset;
	int32 num_q_offset;
	CUdeviceptr gpu_p_array;
	CUdeviceptr gpu_q_array;
	CUdeviceptr gpu_found_array;
	void *gpu_ptr;
	uint32 threads_per_block;
	uint32 num_blocks;

	p_array = L->p_array = (p_small_batch_t *)xmalloc(
					sizeof(p_small_batch_t));
	q_array = L->q_array = (p_large_batch_t *)xmalloc(
					sizeof(p_large_batch_t));

	CUDA_TRY(cuMemAlloc(&gpu_p_array, 
			sizeof(p_small_batch_t)))
	CUDA_TRY(cuMemAlloc(&gpu_q_array, 
			sizeof(p_large_batch_t))) 

	num_blocks = gpu_info->num_compute_units;
	threads_per_block = 128;
	if (gpu_info->registers_per_block == 16384)
		threads_per_block = 256;

	CUDA_TRY(cuFuncSetBlockShape(gpu_kernel, 
				threads_per_block, 1, 1))

	found_array_size = num_blocks * threads_per_block;
	found_array = (found_t *)xmalloc(found_array_size *
					sizeof(found_t));
	CUDA_TRY(cuMemAlloc(&gpu_found_array, 
			found_array_size * sizeof(found_t)))

	i = 0;
	gpu_ptr = (void *)(size_t)gpu_p_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_p_offset = i;
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)gpu_q_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_q_offset = i;
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)gpu_found_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_TRY(cuParamSetSize(gpu_kernel, i))

	printf("------- %u-%u %u-%u\n",
			small_p_min, small_p_max,
			large_p_min, large_p_max);

	min_small = small_p_min;
	sieve_fb_reset(sieve_small, (uint64)small_p_min, 
			(uint64)small_p_max, 1, 1);

	while (min_small < small_p_max) {

		L->num_p = 0;
		for (i = 0; i < P_SMALL_BATCH_SIZE && 
				min_small < small_p_max; i++) {
			min_small = sieve_fb_next(sieve_small, L->poly,
						store_small_p, L);
		}
		if (L->num_p == 0)
			goto finished;

		printf("p batch %u %u\n", L->num_p, min_small);

		CUDA_TRY(cuMemcpyHtoD(gpu_p_array, p_array,
				sizeof(p_small_batch_t)))
		CUDA_TRY(cuParamSeti(gpu_kernel, num_p_offset, L->num_p))

		min_large = large_p_min;
		sieve_fb_reset(sieve_large, 
				(uint64)large_p_min, (uint64)large_p_max,
				1, 1);

		while (min_large <= large_p_max) {

			double curr_time;
			double elapsed;

			L->num_q = 0;
			for (i = 0; i < P_LARGE_BATCH_SIZE && 
					min_large < large_p_max; i++) {
				min_large = sieve_fb_next(sieve_large, L->poly,
							store_large_p, L);
			}
			if (L->num_q == 0)
				goto finished;

			CUDA_TRY(cuMemcpyHtoD(gpu_q_array, q_array,
					sizeof(p_large_batch_t)))

			CUDA_TRY(cuParamSeti(gpu_kernel, 
					num_q_offset, L->num_q))

			CUDA_TRY(cuMemsetD32(gpu_found_array, 0, 
					found_array_size * sizeof(found_t) / 
					sizeof(uint32)))

			num_blocks = gpu_info->num_compute_units;
			if (L->num_q < found_array_size) {
				num_blocks = (L->num_q + 
					threads_per_block - 1) /
					threads_per_block;
			}

			CUDA_TRY(cuLaunchGrid(gpu_kernel, num_blocks, 1))

			CUDA_TRY(cuMemcpyDtoH(found_array, gpu_found_array, 
				found_array_size * sizeof(found_t)))

			for (i = 0; i < found_array_size; i++) {
				found_t *f = found_array + i;

				if (f->p > 0) {
					handle_collision(L->poly, f->p,
						f->proot, f->offset, f->q);
				}
			}

			if (obj->flags & MSIEVE_FLAG_STOP_SIEVING) {
				quit = 1;
				goto finished;
			}
			curr_time = get_cpu_time();
			elapsed = curr_time - L->start_time;
			if (elapsed > L->deadline) {
				quit = 1;
				goto finished;
			}
		}
	}

finished:
	CUDA_TRY(cuMemFree(gpu_p_array))
	CUDA_TRY(cuMemFree(gpu_q_array))
	CUDA_TRY(cuMemFree(gpu_found_array))
	free(p_array);
	free(q_array);
	free(found_array);
	return quit;
}
