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

#include <stage1.h>
#include <stage1_core_gpu/stage1_core_2prog.h>

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_roots;
	uint32 num_p;
	uint32 num_p_alloc;

	uint32 *p;
	uint64 *roots[MAX_ROOTS];
} p_soa_var_t;

#define MAX_P_SOA_ARRAYS 10

typedef struct {
	uint32 num_arrays;
	uint32 num_p;
	p_soa_var_t soa[MAX_P_SOA_ARRAYS];
} p_soa_array_t;

static void
p_soa_array_init(p_soa_array_t *s, uint32 degree)
{
	uint32 i, j;
	memset(s, 0, sizeof(p_soa_array_t));

	switch (degree) {
	case 4:
		s->num_arrays = 3;
		s->soa[0].num_roots = 16;
		s->soa[1].num_roots = 8;
		s->soa[2].num_roots = 4;
		break;
	case 5:
		s->num_arrays = 2;
		s->soa[0].num_roots = 25;
		s->soa[1].num_roots = 5;
		break;
	case 6:
		s->num_arrays = 10;
		s->soa[9].num_roots = 6;
		s->soa[8].num_roots = 8;
		s->soa[7].num_roots = 12;
		s->soa[6].num_roots = 16;
		s->soa[5].num_roots = 24;
		s->soa[4].num_roots = 32;
		s->soa[3].num_roots = 36;
		s->soa[2].num_roots = 48;
		s->soa[1].num_roots = 64;
		s->soa[0].num_roots = 72;
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

	s->num_p = 0;
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
store_p_soa(uint32 p, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i, j;
	p_soa_array_t *s = (p_soa_array_t *)extra;

	for (i = 0; i < s->num_arrays; i++) {
		uint32 num;
		p_soa_var_t *soa = s->soa + i;

		if (soa->num_roots != num_roots)
			continue;

		num = soa->num_p;
		if (soa->num_p_alloc == num)
			p_soa_var_grow(soa);

		soa->p[num] = p;
		for (j = 0; j < num_roots; j++)
			soa->roots[j][num] = roots[j];

		soa->num_p++;
		s->num_p++;
		break;
	}
}

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 p_size_alloc;
	q_packed_t *curr;
	q_packed_t *packed_array;
} q_packed_var_t;

static void 
q_packed_init(q_packed_var_t *s)
{
	memset(s, 0, sizeof(q_packed_var_t));

	s->p_size_alloc = 1000;
	s->packed_array = s->curr = (q_packed_t *)xmalloc(s->p_size_alloc *
						sizeof(q_packed_t));
}

static void 
q_packed_free(q_packed_var_t *s)
{
	free(s->packed_array);
}

static void 
q_packed_reset(q_packed_var_t *s)
{
	s->num_p = 0;
	s->curr = s->packed_array;
}

static q_packed_t * 
q_packed_next(q_packed_t *curr)
{
	return (q_packed_t *)((uint64 *)curr + 
			Q_PACKED_HEADER_WORDS + curr->num_roots);
}

static void 
store_q_packed(uint32 p, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i;
	q_packed_var_t *s = (q_packed_var_t *)extra;
	q_packed_t *curr;

	if ((void *)s->curr >=
	    (void *)(s->packed_array + s->p_size_alloc - 1)) {
		uint32 *oldptr = (uint32 *)s->packed_array;

		/* we have to be careful here because reallocating
		   the array will cause memory to change out from
		   under s->curr, and we cannot index into the new
		   array because its entries are stored in
		   compressed format */
	   
		s->p_size_alloc *= 2;
		s->packed_array = (q_packed_t *)xrealloc(
						s->packed_array,
						s->p_size_alloc *
						sizeof(q_packed_t));
		s->curr = (q_packed_t *)((uint32 *)s->packed_array +
					((uint32 *)s->curr - oldptr));
	}

	curr = s->curr;
	curr->p = p;
	curr->num_roots = num_roots;
	for (i = 0; i < num_roots; i++)
		curr->roots[i] = roots[i];

	s->num_p++;
	s->curr = q_packed_next(s->curr);
}

#define MAX_SPECIAL_Q ((uint32)(-1))
#define MAX_OTHER ((uint32)(-1))

/*------------------------------------------------------------------------*/
static uint32
sieve_lattice_qbatch(msieve_obj *obj, lattice_fb_t *L,
		uint32 threads_per_block, p_soa_array_t *p_array,
		q_packed_var_t *q_array, float sieve_bound,
		gpu_info_t *gpu_info, CUfunction gpu_kernel)
{
	uint32 i, j;
	q_packed_t *packed_array = q_array->packed_array;
	p_soa_t *p_marshall = (p_soa_t *)L->p_marshall;
	uint32 num_blocks;
	uint32 num_p_offset;
	uint32 num_q_offset;
	uint32 num_proots_offset;
	uint32 found_array_size = L->found_array_size;
	found_t *found_array = (found_t *)L->found_array;
	void *gpu_ptr;

	if (sieve_bound < 0) {
		printf("error: sieve bound must be positive "
			"in sieve_lattice_qbatch()\n");
		exit(1);
	}

	i = 0;
	gpu_ptr = (void *)(size_t)L->gpu_p_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_p_offset = i;
	i += sizeof(uint32);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_proots_offset = i;
	i += sizeof(uint32);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_q_offset = i;
	i += sizeof(uint32);

	CUDA_ALIGN_PARAM(i, __alignof(float));
	CUDA_TRY(cuParamSetf(gpu_kernel, (int)i, sieve_bound))
	i += sizeof(float);

	gpu_ptr = (void *)(size_t)L->gpu_found_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_TRY(cuParamSetSize(gpu_kernel, i))

	for (i = 0; i < p_array->num_arrays; i++) {

		p_soa_var_t *soa = p_array->soa + i;
		uint32 num_proots = soa->num_roots;
		uint32 num_p_done = 0;
		float elapsed_ms;

		if (soa->num_p < threads_per_block)
			continue;

		CUDA_TRY(cuParamSeti(gpu_kernel, 
				num_proots_offset, num_proots))

		while (num_p_done < soa->num_p) {

			uint32 num_q_done = 0;
			uint32 packed_words;
			uint32 curr_num_q;
			q_packed_t *packed_start = packed_array;

			uint32 curr_num_p = MIN(3 * found_array_size,
						soa->num_p - num_p_done);

			curr_num_p = MIN(curr_num_p, P_SOA_BATCH_SIZE);

			memcpy(p_marshall->p, 
				soa->p + num_p_done,
				curr_num_p * sizeof(uint32));
			for (j = 0; j < num_proots; j++) {
				memcpy(p_marshall->roots[j],
					soa->roots[j] + num_p_done,
					curr_num_p * sizeof(uint64));
			}

			CUDA_TRY(cuMemcpyHtoD(L->gpu_p_array, p_marshall,
					P_SOA_BATCH_SIZE * (sizeof(uint32) +
						num_proots * sizeof(uint64))))

			CUDA_TRY(cuParamSeti(gpu_kernel, num_p_offset, 
						curr_num_p))

			while (num_q_done < q_array->num_p) {
				q_packed_t *curr_packed = packed_start;

				curr_num_q = 0;
				packed_words = 0;
				do {
					uint32 next_words = packed_words +
							Q_PACKED_HEADER_WORDS +
							curr_packed->num_roots;

					if (next_words >= Q_ARRAY_WORDS)
						break;

					curr_num_q++;
					packed_words = next_words;
					curr_packed = q_packed_next(
								curr_packed);
				} while (++num_q_done < q_array->num_p);

#if 0
				printf("proots %u pnum %u qnum %u qwords %u\n",
						num_proots, curr_num_p,
						curr_num_q, packed_words);
#endif
				CUDA_TRY(cuMemcpyHtoD(L->gpu_q_array, 
							packed_start,
							packed_words *
							sizeof(uint64)))

				CUDA_TRY(cuParamSeti(gpu_kernel, num_q_offset, 
							curr_num_q))

				num_blocks = gpu_info->num_compute_units;
				if (curr_num_p < found_array_size) {
					num_blocks = (curr_num_p + 
						threads_per_block - 1) /
						threads_per_block;
				}

				CUDA_TRY(cuEventRecord(L->start, 0))
				CUDA_TRY(cuLaunchGrid(gpu_kernel, 
							num_blocks, 1))
				CUDA_TRY(cuEventRecord(L->end, 0))
				CUDA_TRY(cuEventSynchronize(L->end))

				CUDA_TRY(cuMemcpyDtoH(found_array, 
							L->gpu_found_array, 
							threads_per_block * 
							num_blocks *
							sizeof(found_t)))

				for (j = 0; j < threads_per_block *
						num_blocks; j++) {
					found_t *f = found_array + j;

					if (f->p > 0) {

						handle_collision(L->poly,
							f->p, f->q, f->qroot,
							f->offset);
					}
				}

				CUDA_TRY(cuEventElapsedTime(&elapsed_ms,
							L->start, L->end))
				L->deadline -= elapsed_ms / 1000;

				if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
					return 1;

				if (L->deadline < 0)
					return 1;

				packed_start = curr_packed;
			}

			num_p_done += curr_num_p;
		}
	}

	return 0;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_specialq_64(msieve_obj *obj, lattice_fb_t *L, float sieve_bound,
		sieve_fb_t *sieve_special_q, sieve_fb_t *sieve_p,
		uint32 special_q_min, uint32 special_q_max,
		uint32 p_min, uint32 p_max)
{
	uint32 quit = 0;
	poly_search_t *poly = L->poly;
	uint32 degree = poly->degree;
	double cpu_start_time = get_cpu_time();
	p_soa_array_t *p_array;
	q_packed_var_t *q_array;
	uint32 p_min_roots, p_max_roots;
	uint32 q_min_roots, q_max_roots;
	uint32 threads_per_block;
	gpu_info_t *gpu_info = poly->gpu_info;
	CUmodule gpu_module = poly->gpu_module_2prog;
       	CUfunction gpu_kernel;
	uint32 host_p_batch_size;
	uint32 host_q_batch_size;

	if (p_max < ((uint32)1 << 24))
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel, 
				gpu_module, "sieve_kernel_48"))
	else
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel, 
				gpu_module, "sieve_kernel_64"))


	L->p_marshall = (p_soa_t *)xmalloc(sizeof(p_soa_t));
	p_array = L->p_array = (p_soa_array_t *)xmalloc(
					sizeof(p_soa_array_t));
	q_array = L->q_array = (q_packed_var_t *)xmalloc(
					sizeof(q_packed_var_t));
	p_soa_array_init(p_array, degree);
	q_packed_init(q_array);

	CUDA_TRY(cuMemAlloc(&L->gpu_p_array, sizeof(p_soa_t)))
	CUDA_TRY(cuModuleGetGlobal(&L->gpu_q_array, 
				NULL, gpu_module, "qbatch"))

	CUDA_TRY(cuFuncGetAttribute((int *)&threads_per_block, 
			CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
			gpu_kernel))

	CUDA_TRY(cuFuncSetBlockShape(gpu_kernel, 
				threads_per_block, 1, 1))

	L->found_array_size = threads_per_block *
				gpu_info->num_compute_units;
	L->found_array = (found_t *)xmalloc(L->found_array_size *
					sizeof(found_t));
	CUDA_TRY(cuMemAlloc(&L->gpu_found_array, 
			L->found_array_size * sizeof(found_t)))

	CUDA_TRY(cuEventCreate(&L->start, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&L->end, CU_EVENT_BLOCKING_SYNC))

	host_p_batch_size = MAX(50000, 12 * L->found_array_size);
	host_q_batch_size = MAX(10000, L->found_array_size / 3);

	p_min_roots = degree;
	p_max_roots = degree * degree;

	q_min_roots = 1;
	q_max_roots = degree * degree;

	if (degree == 6)
		p_max_roots *= 2;

	sieve_fb_reset(sieve_p, p_min, p_max, p_min_roots, p_max_roots);
	while (!quit) {

		double curr_time;

		p_soa_array_reset(p_array);
		while (sieve_fb_next(sieve_p, poly, store_p_soa,
					p_array) != P_SEARCH_DONE) {
			if (p_array->num_p == host_p_batch_size)
				break;
		}

		if (p_array->num_p == 0)
			break;

		sieve_fb_reset(sieve_special_q, special_q_min, special_q_max,
				q_min_roots, q_max_roots);
		while (!quit) {

			q_packed_reset(q_array);
			while (sieve_fb_next(sieve_special_q, poly, 
						store_q_packed,
						q_array) != P_SEARCH_DONE) {
				if (q_array->num_p == host_q_batch_size)
					break;
			}

			if (q_array->num_p == 0)
				break;

			quit = sieve_lattice_qbatch(obj, L, threads_per_block,
					p_array, q_array, sieve_bound,
					gpu_info, gpu_kernel);
		}

		curr_time = get_cpu_time();
		L->deadline -= curr_time - cpu_start_time;
		cpu_start_time = curr_time;
	}

	CUDA_TRY(cuMemFree(L->gpu_p_array))
	CUDA_TRY(cuMemFree(L->gpu_found_array))
	CUDA_TRY(cuEventDestroy(L->start))
	CUDA_TRY(cuEventDestroy(L->end))
	p_soa_array_free(p_array);
	q_packed_free(q_array);
	free(p_array);
	free(q_array);
	free(L->found_array);
	free(L->p_marshall);
	return quit;
}

/*------------------------------------------------------------------------*/
void
sieve_lattice_gpu_2prog(msieve_obj *obj, lattice_fb_t *L)
{
	uint32 i;
	poly_search_t *poly = L->poly;
	uint32 degree = poly->degree;
	uint32 p_min, p_max;
	uint32 special_q_min, special_q_max;
	double m0 = poly->m0;
	double coeff_max = poly->coeff_max;
	double p_size_max = poly->p_size_max;
	sieve_fb_t sieve_p, sieve_special_q;
	double sieve_bound = coeff_max / m0 / degree;

	p_min = MIN(MAX_OTHER / P_SCALE, sqrt(p_size_max / P_SCALE));
	p_max = p_min * P_SCALE;
	special_q_min = 1;
	special_q_max = p_min - 1;

	/* set up the special q factory */

	sieve_fb_init(&sieve_special_q, poly,
			0, 0, /* prime special q */
			1, degree,
			0);

	sieve_fb_init(&sieve_p, poly, 
			5, 5000,
			1, degree,
		       	0);

	for (i = 0; i < 2; i++) {
		uint32 quit;

		gmp_printf("coeff %Zd specialq %u - %u other %u - %u\n",
				poly->high_coeff,
				special_q_min, special_q_max,
				p_min, p_max);

		quit = sieve_specialq_64(obj, L, sieve_bound,
				&sieve_special_q, &sieve_p,
				special_q_min, special_q_max, p_min, p_max);

		if (quit || special_q_max < 250 ||
		    p_max >= MAX_OTHER / P_SCALE)
			break;

		p_min = p_max;
		p_max *= P_SCALE;
		special_q_max /= P_SCALE;
	}

	sieve_fb_free(&sieve_special_q);
	sieve_fb_free(&sieve_p);
}
