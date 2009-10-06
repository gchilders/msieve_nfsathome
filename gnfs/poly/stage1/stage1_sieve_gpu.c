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
#include "stage1_core.h"

#define HOST_BATCH_SIZE 50000

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_roots;
	uint32 num_p;
	uint32 num_p_alloc;

	uint32 *p;
	uint64 *roots[MAX_ROOTS];
} p_soa_var_t;

#define MAX_P_SOA_ARRAYS 5

typedef struct {
	uint32 num_arrays;
	p_soa_var_t soa[MAX_P_SOA_ARRAYS];
} p_soa_array_t;

static void
p_soa_array_init(p_soa_array_t *s, uint32 poly_degree)
{
	uint32 i, j;
	memset(s, 0, sizeof(p_soa_array_t));

	switch (poly_degree) {
	case 4:
		s->num_arrays = 5;
		s->soa[4].num_roots = 2;
		s->soa[3].num_roots = 4;
		s->soa[2].num_roots = 8;
		s->soa[1].num_roots = 16;
		s->soa[0].num_roots = 32;
		break;
	case 5:
		s->num_arrays = 3;
		s->soa[2].num_roots = 1;
		s->soa[1].num_roots = 5;
		s->soa[0].num_roots = 25;
		break;
	case 6:
		s->num_arrays = 3;
		s->soa[2].num_roots = 1;
		s->soa[1].num_roots = 6;
		s->soa[0].num_roots = 36;
		break;
	}

	for (i = 0; i < s->num_arrays; i++) {
		p_soa_var_t *soa = s->soa + i;

		soa->num_p_alloc = 1000;
		soa->p = (uint32 *)xmalloc(soa->num_p_alloc * 
					sizeof(uint32));
		for (j = 0; j < soa->num_roots; j++) {
			soa->roots[j] = (uint64 *)xmalloc(
						soa->num_p_alloc * 
						sizeof(uint64));
		}
	}
}

static void
p_soa_array_free(p_soa_array_t *s)
{
	uint32 i, j;

	for (i = 0; i < s->num_arrays; i++) {
		p_soa_var_t *soa = s->soa + i;

		free(soa->p);
		for (j = 0; j < soa->num_roots; j++)
			free(soa->roots[j]);
	}
}

static void
p_soa_array_reset(p_soa_array_t *s)
{
	uint32 i;

	for (i = 0; i < s->num_arrays; i++)
		s->soa[i].num_p = 0;
}

static void
p_soa_var_grow(p_soa_var_t *soa)
{
	uint32 i;

	soa->num_p_alloc *= 2;
	soa->p = (uint32 *)xrealloc(soa->p, 
				soa->num_p_alloc * 
				sizeof(uint32));
	for (i = 0; i < soa->num_roots; i++) {
		soa->roots[i] = (uint64 *)xrealloc(soa->roots[i], 
					soa->num_p_alloc * 
					sizeof(uint64));
	}
}

static void 
store_p_soa(uint64 p, uint32 num_roots,
		mpz_t *roots, void *extra)
{
	uint32 i, j;
	lattice_fb_t *L = (lattice_fb_t *)extra;
	p_soa_array_t *s = (p_soa_array_t *)L->q_array;

	for (i = 0; i < s->num_arrays; i++) {
		uint32 num;
		p_soa_var_t *soa = s->soa + i;

		if (soa->num_roots != num_roots)
			continue;

		num = soa->num_p;
		if (soa->num_p_alloc == num)
			p_soa_var_grow(soa);

		soa->p[num] = (uint32)p;
		for (j = 0; j < num_roots; j++)
			soa->roots[j][num] = gmp2uint64(roots[j]);

		soa->num_p++;
	}
	L->num_q++;
}

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 p_size;
	uint32 p_size_alloc;
	p_packed_t *curr;
	p_packed_t *packed_array;
} p_packed_var_t;

static void 
p_packed_init(p_packed_var_t *s)
{
	memset(s, 0, sizeof(p_packed_var_t));

	s->p_size_alloc = 5000;
	s->packed_array = s->curr = (p_packed_t *)xmalloc(s->p_size_alloc *
						sizeof(p_packed_t));
}

static void 
p_packed_free(p_packed_var_t *s)
{
	free(s->packed_array);
}

static void 
p_packed_reset(p_packed_var_t *s)
{
	s->num_p = s->p_size = 0;
	s->curr = s->packed_array;
}

static p_packed_t * 
p_packed_next(p_packed_t *curr)
{
	return (p_packed_t *)((uint64 *)curr + 
			P_PACKED_HEADER_WORDS + curr->num_roots);
}

static void 
store_p_packed(uint64 p, uint32 num_roots,
		mpz_t *roots, void *extra)
{
	uint32 i;
	lattice_fb_t *L = (lattice_fb_t *)extra;
	p_packed_var_t *s = (p_packed_var_t *)L->p_array;
	p_packed_t *curr;

	if ((p_packed_t *)((uint64 *)s->curr + s->p_size) + 1 >=
			s->packed_array + s->p_size_alloc ) {

		s->p_size_alloc *= 2;
		s->packed_array = (p_packed_t *)xrealloc(
						s->packed_array,
						s->p_size_alloc *
						sizeof(p_packed_t));
		s->curr = (p_packed_t *)((uint64 *)s->packed_array + s->p_size);
	}

	curr = s->curr;
	curr->p = (uint32)p;
	curr->lattice_size = L->poly->sieve_size / ((double)p * p);
	curr->num_roots = num_roots;
	curr->pad = 0;
	for (i = 0; i < num_roots; i++)
		curr->roots[i] = gmp2uint64(roots[i]);

	s->num_p++;
	s->curr = p_packed_next(s->curr);
	s->p_size = ((uint8 *)s->curr - 
			(uint8 *)s->packed_array) / sizeof(uint64);
	L->num_p++;
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_batch(msieve_obj *obj, lattice_fb_t *L,
			uint32 threads_per_block,
			gpu_info_t *gpu_info, CUfunction gpu_kernel)
{
	uint32 i, j, k;
	p_packed_var_t * p_array = (p_packed_var_t *)L->p_array;
	p_soa_array_t * q_array = (p_soa_array_t *)L->q_array;
	p_packed_t *packed_array = p_array->packed_array;
	uint32 num_blocks;
	void *gpu_ptr;

	for (i = 0; i < q_array->num_arrays; i++) {

		p_soa_var_t *soa = q_array->soa + i;
		p_soa_t *marshall = L->marshall;
		uint32 num_qroots = soa->num_roots;
		uint32 num_q_done = 0;

		if (soa->num_p == 0)
			continue;

		while (num_q_done < soa->num_p) {

			uint32 num_p_done = 0;
			uint32 packed_words = 0;
			uint32 curr_num_p = 0;
			p_packed_t *packed_start = packed_array;
			time_t curr_time;
			double elapsed;

			uint32 curr_num_q = MIN(P_SOA_BATCH_SIZE,
						soa->num_p - num_q_done);

			if (curr_num_q == P_SOA_BATCH_SIZE &&
					soa->num_p - curr_num_q < 
					P_SOA_BATCH_SIZE / 2) {
				curr_num_q /= 2;
			}

			for (j = 0; j < curr_num_q; j++) {
				marshall->p[j] = soa->p[num_q_done + j];

				for (k = 0; k < num_qroots; k++) {
					marshall->roots[k][j] =
						soa->roots[k][num_q_done + j];
				}
			}

			CUDA_TRY(cuMemcpyHtoD(L->gpu_q_array, marshall,
					P_SOA_BATCH_SIZE * (sizeof(uint32) +
						num_qroots * sizeof(uint64))))

			while (num_p_done < p_array->num_p) {
				p_packed_t *curr_packed = packed_start;

				do {
					uint32 next_words = packed_words +
							P_PACKED_HEADER_WORDS +
							curr_packed->num_roots;

					if (next_words >= L->p_array_max_words)
						break;

					curr_num_p++;
					packed_words = next_words;
					curr_packed = p_packed_next(
								curr_packed);
				} while (++num_p_done < p_array->num_p);

#if 0
				printf("qroots %u qnum %u pnum %u pwords %u\n",
						num_qroots, curr_num_q,
						curr_num_p, packed_words);
#endif
				CUDA_TRY(cuMemcpyHtoD(L->gpu_p_array, 
							packed_start,
							packed_words *
							sizeof(uint64)))

				j = 0;
				gpu_ptr = (void *)(size_t)L->gpu_q_array;
				CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
				CUDA_TRY(cuParamSetv(gpu_kernel, (int)j, 
						&gpu_ptr, sizeof(gpu_ptr)))
				j += sizeof(gpu_ptr);

				CUDA_ALIGN_PARAM(j, __alignof(uint32));
				CUDA_TRY(cuParamSeti(gpu_kernel, j, curr_num_q))
				j += sizeof(uint32);

				CUDA_ALIGN_PARAM(j, __alignof(uint32));
				CUDA_TRY(cuParamSeti(gpu_kernel, j, num_qroots))
				j += sizeof(uint32);

				CUDA_ALIGN_PARAM(j, __alignof(uint32));
				CUDA_TRY(cuParamSeti(gpu_kernel, j, curr_num_p))
				j += sizeof(uint32);

				gpu_ptr = (void *)(size_t)L->gpu_found_array;
				CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
				CUDA_TRY(cuParamSetv(gpu_kernel, (int)j, 
						&gpu_ptr, sizeof(gpu_ptr)))
				j += sizeof(gpu_ptr);

				CUDA_TRY(cuParamSetSize(gpu_kernel, j))

				num_blocks = gpu_info->num_compute_units;
				if (curr_num_q < L->found_array_size) {
					num_blocks = (curr_num_q + 
						threads_per_block - 1) /
						threads_per_block;
				}

				CUDA_TRY(cuLaunchGrid(gpu_kernel, 
							num_blocks, 1))

				CUDA_TRY(cuMemcpyDtoH(L->found_array, 
							L->gpu_found_array, 
							L->found_array_size * 
							sizeof(found_t)))

				for (j = 0; j < L->found_array_size; j++) {
					found_t *f = L->found_array + j;

					if (f->p > 0) {
						handle_collision(L->poly, 
							f->p, f->proot, 
							f->offset, f->q);
					}
				}

				packed_start = curr_packed;
				packed_words = 0;
				curr_num_p = 0;
			}

			if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
				return 1;

			curr_time = time(NULL);
			elapsed = curr_time - L->start_time;
			if (elapsed > L->deadline)
				return 1;

			num_q_done += curr_num_q;
		}
	}

	return 0;
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_gpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_small, sieve_fb_t *sieve_large, 
		uint32 small_p_min, uint32 small_p_max, 
		uint32 large_p_min, uint32 large_p_max,
		gpu_info_t *gpu_info, CUmodule gpu_module)
{
	uint32 i;
	uint32 min_small, min_large;
	uint32 quit = 0;
	p_packed_var_t * p_array;
	p_soa_array_t * q_array;
	uint32 degree = L->poly->degree;
	clock_t clock_start;

	CUfunction gpu_kernel;
	uint32 threads_per_block;

	L->marshall = (p_soa_t *)xmalloc(sizeof(p_soa_t));
	q_array = L->q_array = (p_soa_array_t *)xmalloc(
					sizeof(p_soa_array_t));
	p_array = L->p_array = (p_packed_var_t *)xmalloc(
					sizeof(p_packed_var_t));
	p_packed_init(p_array);
	p_soa_array_init(q_array, L->poly->degree);

	CUDA_TRY(cuModuleGetFunction(&gpu_kernel, gpu_module, 
				"sieve_kernel"))
	CUDA_TRY(cuMemAlloc(&L->gpu_q_array, 
			P_SOA_BATCH_SIZE * (sizeof(uint32) + 
				MAX_ROOTS * sizeof(uint64))))
	CUDA_TRY(cuModuleGetGlobal(&L->gpu_p_array, 
				   &L->p_array_max_words, 
				   gpu_module, "pbatch"))
	L->p_array_max_words /= sizeof(uint64);

	threads_per_block = 128;
	if (gpu_info->registers_per_block == 16384)
		threads_per_block = 256;

	CUDA_TRY(cuFuncSetBlockShape(gpu_kernel, 
				threads_per_block, 1, 1))

	L->found_array_size = threads_per_block *
				gpu_info->num_compute_units;
	L->found_array = (found_t *)xmalloc(L->found_array_size *
					sizeof(found_t));
	CUDA_TRY(cuMemAlloc(&L->gpu_found_array, 
			L->found_array_size * sizeof(found_t)))

	printf("------- %u-%u %u-%u\n",
			small_p_min, small_p_max,
			large_p_min, large_p_max);

	min_large = large_p_min;
	sieve_fb_reset(sieve_small, (uint64)large_p_min, 
			(uint64)large_p_max, degree, MAX_ROOTS);

	clock_start = clock();

	while (min_large < large_p_max) {

		L->num_q = 0;
		p_soa_array_reset(q_array);
		for (i = 0; i < HOST_BATCH_SIZE && 
				min_large < large_p_max; i++) {
			min_large = sieve_fb_next(sieve_small, L->poly,
						store_p_soa, L);
		}
		if (L->num_q == 0)
			goto finished;

		min_small = small_p_min;
		sieve_fb_reset(sieve_large, 
				(uint64)small_p_min, (uint64)small_p_max,
				1, MAX_ROOTS);

		while (min_small <= small_p_max) {

			L->num_p = 0;
			p_packed_reset(p_array);
			for (i = 0; i < HOST_BATCH_SIZE && 
					min_small < small_p_max; i++) {
				min_small = sieve_fb_next(sieve_large, L->poly,
							store_p_packed, L);
			}
			if (L->num_p == 0)
				goto finished;

			if (sieve_lattice_batch(obj, L, threads_per_block,
						gpu_info, gpu_kernel)) {
				quit = 1;
				goto finished;
			}
		}
	}

	printf("%lf\n", (double)(clock() - clock_start) / CLOCKS_PER_SEC);

finished:
	CUDA_TRY(cuMemFree(L->gpu_q_array))
	CUDA_TRY(cuMemFree(L->gpu_found_array))
	p_packed_free(p_array);
	p_soa_array_free(q_array);
	free(p_array);
	free(q_array);
	free(L->found_array);
	free(L->marshall);
	return quit;
}


