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
#include <stage1_core_gpu_sort/stage1_core_gpu_sort.h>

/* GPU collision search; this code looks for self-collisions
   among arithmetic progressions, by finding k1 and k2 such that
   for two arithmetic progressions r1+k*p1^2 and r2+k*p2^2 we
   have

      r1 + k1*p1^2 = r2 + k2*p2^2

   such that
      - p1 and p2 are coprime and < 2^32
      - the value where they coincide is of size smaller
        than a fixed bound

   This code uses a hashtable to find collisions across all the
   p1 and p2 in the set simultaneously. The size of the range is
   comparatively very large, and the algorithm as described above
   is only efficient for pretty small problems. We cannot
   practically fill up a sigle hashtable with all the possibilities
   because for e.g. 512-bit GNFS the range on k1 and k2 is typically 
   around 10^6 and the set has ~10^6 progressions. We can reduce the
   memory use by breaking the hashtable into blocks of size > p_min
   and filling each block individually, but that only reduces the 
   memory use and not the actual work required.
   
   To scale up the idea, we further use a 'special-q' formulation 
   where all the inputs to the hashtable are constrained to fall 
   on a third arithmetic progression r3 + k*q^2 for some k. We choose 
   a given q and for each of its roots run the complete hashtable search.
   This is analogous to lattice sieving across the interval.
   
   This allows us to choose q so that the hashtable problem is of 
   reasonable size but the collisions found are still over the 
   original, impractically large range. 

   A side effect of this is that the complete range must be < 2^96
   in size, though this appears to be sufficient for very large
   problems, e.g. 200-digit GNFS */

/*------------------------------------------------------------------------*/

typedef struct {
	uint32 num_roots;
	uint32 num_p;
	uint32 num_p_alloc;

	CUdeviceptr dev_p;
	CUdeviceptr dev_start_roots;
	CUdeviceptr dev_roots;
	CUdeviceptr dev_p_entry;
	CUdeviceptr dev_root_entry;
	CUstream stream;

	uint32 *p;
	uint64 *roots[MAX_ROOTS];

} p_soa_var_t;

#define MAX_P_SOA_ARRAYS 16

typedef struct {
	uint32 num_arrays;
	uint32 num_p;

	uint32 max_p_roots;
	p_soa_var_t *soa;

} p_soa_array_t;

static void
p_soa_array_init(p_soa_array_t *s, uint32 degree)
{
	uint32 i, j;
	memset(s, 0, sizeof(p_soa_array_t));

	s->soa = (p_soa_var_t *)xmalloc(MAX_P_SOA_ARRAYS *
					sizeof(p_soa_var_t));

	switch (degree) {
	case 4:
		s->num_arrays = 3;
		s->soa[0].num_roots = 2;
		s->soa[1].num_roots = 4;
		s->soa[2].num_roots = 8;
		s->max_p_roots = 8;
		break;

	case 5:
		s->num_arrays = 3;
		s->soa[0].num_roots = 1;
		s->soa[1].num_roots = 5;
		s->soa[2].num_roots = 25;
		s->max_p_roots = 25;
		break;

	case 6:
		s->num_arrays = 5;
		s->soa[0].num_roots = 2;
		s->soa[1].num_roots = 4;
		s->soa[2].num_roots = 6;
		s->soa[3].num_roots = 12;
		s->soa[4].num_roots = 36;
		s->max_p_roots = 36;
		break;

	case 7: /* ;) */
		s->num_arrays = 3;
		s->soa[0].num_roots = 1;
		s->soa[1].num_roots = 7;
		s->soa[2].num_roots = 49;
		s->max_p_roots = 49;
		break;
	}

	for (i = 0; i < s->num_arrays; i++) {
		p_soa_var_t *soa = s->soa + i;

		soa->num_p = 0;
		soa->num_p_alloc = 256;
		soa->p = (uint32 *)xmalloc(soa->num_p_alloc * sizeof(uint32));
		for (j = 0; j < soa->num_roots; j++) {
			soa->roots[j] = (uint64 *)xmalloc(soa->num_p_alloc *
								sizeof(uint64));
		}
	}
}

static void
p_soa_var_free(p_soa_var_t *soa)
{
	uint32 i;

	free(soa->p);
	for (i = 0; i < soa->num_roots; i++)
		free(soa->roots[i]);
}

static void
p_soa_array_free(p_soa_array_t *s)
{
	uint32 i;

	for (i = 0; i < s->num_arrays; i++)
		p_soa_var_free(s->soa + i);
}

static void
p_soa_array_compact(p_soa_array_t *s)
{
	uint32 i, j;

	i = 0;
	while (i < s->num_arrays) {
		if (s->soa[i].num_p == 0) {

			s->num_arrays--;
			p_soa_var_free(s->soa + i);
			for (j = i; j < s->num_arrays; j++)
				s->soa[j] = s->soa[j + 1];
		}
		else {
			i++;
		}
	}
}

static void
store_p_soa(uint32 p, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i, j;
	p_soa_array_t *s = (p_soa_array_t *)extra;

	j = (uint32)(-1);
	for (i = 0; i < s->num_arrays; i++) {

		if (s->soa[i].num_roots > num_roots)
			continue;

		if (j > i || s->soa[i].num_roots > s->soa[j].num_roots)
			j = i;
	}

	if (j < s->num_arrays) {
		p_soa_var_t *soa = s->soa + j;

		if (soa->num_p_alloc == soa->num_p) {
			soa->num_p_alloc *= 2;
			soa->p = (uint32 *)xrealloc(soa->p, soa->num_p_alloc *
							sizeof(uint32));
			for (j = 0; j < soa->num_roots; j++) {
			soa->roots[j] = (uint64 *)xrealloc(soa->roots[j],
					soa->num_p_alloc * sizeof(uint64));
			}
		}

		soa->p[soa->num_p] = p;
		for (j = 0; j < soa->num_roots; j++)
			soa->roots[j][soa->num_p] = roots[j];

		soa->num_p++;
		s->num_p++;
	}
}

/*------------------------------------------------------------------------*/
static void
check_found_array(lattice_fb_t *L)
{
	uint32 i;
	found_t *found_array = (found_t *)L->found_array;

	CUDA_TRY(cuMemcpyDtoH(found_array, L->gpu_found_array,
			L->found_array_size * sizeof(found_t)))
	CUDA_TRY(cuMemsetD8(L->gpu_found_array, 0,
		L->found_array_size * sizeof(found_t)))

	for (i = 0; i < L->found_array_size; i++) {
		found_t *found = found_array + i;

		if (found->p1 != 0) {
			uint128 res;

			res.w[0] = (uint32)found->proot;
			res.w[1] = (uint32)(found->proot >> 32);
			res.w[2] = 0;
			res.w[3] = 0;

			handle_collision(L->poly, found->p1,
				found->p2, found->q,
				found->qroot,
				res);
		}
	}
}

/*------------------------------------------------------------------------*/
static uint32
handle_special_q(msieve_obj *obj, lattice_fb_t *L,
		uint32 special_q, uint32 num_q_roots, uint64 *q_roots)
{
	uint32 i, j, k;
	uint32 j_offset, k_offset;
	uint32 quit = 0;
	p_soa_array_t *p_array = L->p_array;
	uint32 num_blocks;
	uint64 special_q2 = (uint64)special_q * special_q;
	uint64 sieve_size;
	uint64 sieve_start = 0;
	uint64 sieve_end;
	float elapsed_ms;
	void *gpu_ptr;

	CUDA_TRY(cuEventRecord(L->start, 0))

	if (2 * L->poly->sieve_size / special_q2 > (uint64)(-1)) {
		printf("error: sieve size too large "
			"in handle_special_q\n");
		return 0;
	}

	sieve_size = 2 * L->poly->sieve_size / special_q2;

	CUDA_TRY(cuMemcpyHtoD(L->gpu_q_array, q_roots,
			sizeof(uint64) * num_q_roots))

	for (i = 0; i < p_array->num_arrays; i++) {
		p_soa_var_t *soa = p_array->soa + i;
		uint32 num = soa->num_p;

		if (special_q == 1) {
			CUDA_TRY(cuMemcpyDtoD(soa->dev_roots,
					soa->dev_start_roots,
					soa->num_roots * num * sizeof(uint64)))
		}
		else {
			j = 0;
			gpu_ptr = (void *)(size_t)soa->dev_p;
			CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_TRANS], (int)j,
					&gpu_ptr, sizeof(gpu_ptr)))
			j += sizeof(gpu_ptr);

			CUDA_ALIGN_PARAM(j, __alignof(uint32));
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_TRANS], (int)j,
					(int)num))
			j += sizeof(uint32);

			gpu_ptr = (void *)(size_t)soa->dev_start_roots;
			CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_TRANS], (int)j,
					&gpu_ptr, sizeof(gpu_ptr)))
			j += sizeof(gpu_ptr);

			gpu_ptr = (void *)(size_t)soa->dev_roots;
			CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_TRANS], (int)j,
					&gpu_ptr, sizeof(gpu_ptr)))
			j += sizeof(gpu_ptr);

			CUDA_ALIGN_PARAM(j, __alignof(uint32));
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_TRANS], (int)j,
					(int)soa->num_roots))
			j += sizeof(uint32);

			CUDA_ALIGN_PARAM(j, __alignof(uint32));
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_TRANS], (int)j,
					(int)special_q))
			j += sizeof(uint32);

			CUDA_ALIGN_PARAM(j, __alignof(uint32));
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_TRANS], (int)j,
					(int)num_q_roots))
			j += sizeof(uint32);

			gpu_ptr = (void *)(size_t)L->gpu_q_array;
			CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_TRANS], (int)j,
					&gpu_ptr, sizeof(gpu_ptr)))
			j += sizeof(gpu_ptr);

			CUDA_TRY(cuParamSetSize(L->gpu_kernel[GPU_TRANS], j))

			num_blocks = (num - 1) /
					L->threads_per_block[GPU_TRANS] + 1;
			CUDA_TRY(cuLaunchGridAsync(L->gpu_kernel[GPU_TRANS],
					num_blocks, 1, soa->stream))
		}
	}

	while (1) {

		if (sieve_size - sieve_start >= L->sieve_step)
			sieve_end = sieve_start + L->sieve_step;
		else if (sieve_start == 0)
			sieve_end = sieve_size;
		else
			break;

		for (i = 0; i < p_array->num_arrays; i++) {
			p_soa_var_t *soa = p_array->soa + i;
			uint32 num = soa->num_p;

			j = 0;
			gpu_ptr = (void *)(size_t)soa->dev_p;
			CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_STEP], (int)j,
					&gpu_ptr, sizeof(gpu_ptr)))
			j += sizeof(gpu_ptr);

			CUDA_ALIGN_PARAM(j, __alignof(uint32));
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_STEP], (int)j,
					(int)num))
			j += sizeof(uint32);

			gpu_ptr = (void *)(size_t)soa->dev_roots;
			CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_STEP], (int)j,
					&gpu_ptr, sizeof(gpu_ptr)))
			j += sizeof(gpu_ptr);

			CUDA_ALIGN_PARAM(j, __alignof(uint32));
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_STEP], (int)j,
					(int)soa->num_roots))
			j += sizeof(uint32);

			gpu_ptr = (void *)(size_t)soa->dev_p_entry;
			CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_STEP], (int)j,
					&gpu_ptr, sizeof(gpu_ptr)))
			j += sizeof(gpu_ptr);

			gpu_ptr = (void *)(size_t)soa->dev_root_entry;
			CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_STEP], (int)j,
					&gpu_ptr, sizeof(gpu_ptr)))
			j += sizeof(gpu_ptr);

			CUDA_ALIGN_PARAM(j, __alignof(uint64));
			CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_STEP], (int)j,
					&sieve_end, sizeof(uint64)))
			j += sizeof(uint64);

			CUDA_ALIGN_PARAM(j, __alignof(uint32));
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_STEP], (int)j,
					(int)num_q_roots))
			j += sizeof(uint32);

			CUDA_ALIGN_PARAM(j, __alignof(uint32));
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_STEP], (int)j,
					(int)L->num_entries))
			j += sizeof(uint32);

			CUDA_TRY(cuParamSetSize(L->gpu_kernel[GPU_STEP], j))

			num_blocks = (num - 1) /
					L->threads_per_block[GPU_STEP] + 1;
			CUDA_TRY(cuLaunchGridAsync(L->gpu_kernel[GPU_STEP],
					num_blocks, 1, soa->stream))
		}

		j = 0;
		gpu_ptr = (void *)(size_t)L->gpu_p_array;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_SORT], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_MERGE], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_MERGE1], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		gpu_ptr = (void *)(size_t)L->gpu_root_array;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_SORT], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_MERGE], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_MERGE1], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		CUDA_TRY(cuParamSetSize(L->gpu_kernel[GPU_SORT], j))

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		j_offset = j;
		j += sizeof(uint32);

		CUDA_TRY(cuParamSetSize(L->gpu_kernel[GPU_MERGE], j))

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		k_offset = j;
		j += sizeof(uint32);

		CUDA_TRY(cuParamSetSize(L->gpu_kernel[GPU_MERGE1], j))

		num_blocks = (L->num_entries / 2 - 1) /
				L->threads_per_block[GPU_SORT] + 1;

		CUDA_TRY(cuLaunchGrid(L->gpu_kernel[GPU_SORT],
				num_blocks, num_q_roots))

		j = 2 * L->threads_per_block[GPU_SORT];
		for (; j < L->num_entries; j *= 2) {

			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_MERGE],
					(int)j_offset, j))
			CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_MERGE1],
					(int)j_offset, j))

			num_blocks = (L->num_entries / 2 - 1) /
					L->threads_per_block[GPU_MERGE1] + 1;

			for (k = j; k > L->threads_per_block[GPU_MERGE];
								k /= 2) {

				CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_MERGE1],
						(int)k_offset, k))

				CUDA_TRY(cuLaunchGrid(L->gpu_kernel[GPU_MERGE1],
						num_blocks, num_q_roots))
			}

			num_blocks = (L->num_entries / 2 - 1) /
					L->threads_per_block[GPU_MERGE] + 1;

			CUDA_TRY(cuLaunchGrid(L->gpu_kernel[GPU_MERGE],
					num_blocks, num_q_roots))
		}

		j = 0;
		gpu_ptr = (void *)(size_t)L->gpu_p_array;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_FINAL], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		gpu_ptr = (void *)(size_t)L->gpu_root_array;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_FINAL], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_FINAL], (int)j,
				L->num_entries))
		j += sizeof(uint32);

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_FINAL], (int)j,
				special_q))
		j += sizeof(uint32);

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		CUDA_TRY(cuParamSeti(L->gpu_kernel[GPU_FINAL], (int)j,
				num_q_roots))
		j += sizeof(uint32);

		gpu_ptr = (void *)(size_t)L->gpu_q_array;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_FINAL], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		gpu_ptr = (void *)(size_t)L->gpu_found_array;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(L->gpu_kernel[GPU_FINAL], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		CUDA_TRY(cuParamSetSize(L->gpu_kernel[GPU_FINAL], j))

		num_blocks = (L->num_entries * num_q_roots - 1) /
				L->threads_per_block[GPU_FINAL] + 1;
		if (num_blocks > L->poly->gpu_info->num_compute_units)
			num_blocks = L->poly->gpu_info->num_compute_units;

		CUDA_TRY(cuLaunchGrid(L->gpu_kernel[GPU_FINAL], num_blocks, 1))

		sieve_start = sieve_end;

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING) {
			quit = 1;
			break;
		}
	}

	CUDA_TRY(cuEventRecord(L->end, 0))
	CUDA_TRY(cuEventSynchronize(L->end))
	CUDA_TRY(cuEventElapsedTime(&elapsed_ms, L->start, L->end))
	L->deadline -= elapsed_ms / 1000;
	return quit;
}

/*------------------------------------------------------------------------*/

typedef struct {
	uint32 q;
	uint32 num_roots;
	uint64 roots[MAX_ROOTS];
} specialq_t;

static void
store_specialq(uint32 q, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i;
	specialq_t *specialq = (specialq_t *)extra;

	specialq->q = q;
	specialq->num_roots = num_roots;
	for (i = 0; i < num_roots; i++)
		specialq->roots[i] = roots[i];
}

/*------------------------------------------------------------------------*/
static uint32
sieve_specialq_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q, sieve_fb_t *sieve_p, 
		uint32 special_q_min, uint32 special_q_max, 
		uint32 p_min, uint32 p_max)
{
	uint32 i, j;
	uint32 quit = 0;
	specialq_t specialq;
	p_soa_array_t p_array;
	uint32 degree = L->poly->degree;
	uint32 num_p, num_roots;
	uint32 pass_cnt = 0;
	double cpu_start_time = get_cpu_time();
	CUmodule gpu_module = L->poly->gpu_module_sort;

	L->p_array = &p_array;
	p_soa_array_init(&p_array, degree);
	L->sieve_step = (uint64)p_min * p_min;

	/* build all the arithmetic progressions */

	sieve_fb_reset(sieve_p, p_min, p_max, 1, p_array.max_p_roots);
	while (sieve_fb_next(sieve_p, L->poly, store_p_soa,
			&p_array) != P_SEARCH_DONE) {
		;
	}
	p_soa_array_compact(&p_array);

	num_p = p_array.num_p;
	num_roots = 0;
	for (i = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;
		uint32 num = soa->num_p;

		num_roots += num * soa->num_roots;

		CUDA_TRY(cuStreamCreate(&soa->stream, 0))

		CUDA_TRY(cuMemAlloc(&soa->dev_p, num * sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&soa->dev_start_roots,
				soa->num_roots * num * sizeof(uint64)))
		CUDA_TRY(cuMemAlloc(&soa->dev_roots,
				soa->num_roots * num * sizeof(uint64) *
				BATCH_SPECIALQ_MAX))

		CUDA_TRY(cuMemcpyHtoD(soa->dev_p, soa->p,
				num * sizeof(uint32)))

		for (j = 0; j < soa->num_roots; j++) {
			CUDA_TRY(cuMemcpyHtoD(soa->dev_start_roots +
					j * num * sizeof(uint64),
					soa->roots[j], num * sizeof(uint64)))
		}
	}

#if 1
	printf("aprogs: %u entries, %u roots\n", num_p, num_roots);
#endif

	CUDA_TRY(cuEventCreate(&L->start, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&L->end, CU_EVENT_BLOCKING_SYNC))

	L->gpu_kernel = (CUfunction *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(CUfunction));
	L->threads_per_block = (uint32 *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(uint32));

	CUDA_TRY(cuModuleGetFunction(&L->gpu_kernel[GPU_TRANS],
				gpu_module, "sieve_kernel_trans"))
	CUDA_TRY(cuFuncGetAttribute((int *)&L->threads_per_block[GPU_TRANS],
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				L->gpu_kernel[GPU_TRANS]))
	CUDA_TRY(cuFuncSetBlockShape(L->gpu_kernel[GPU_TRANS],
				L->threads_per_block[GPU_TRANS], 1, 1))

	CUDA_TRY(cuModuleGetFunction(&L->gpu_kernel[GPU_STEP],
				gpu_module, "sieve_kernel_step"))
	CUDA_TRY(cuFuncGetAttribute((int *)&L->threads_per_block[GPU_STEP],
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				L->gpu_kernel[GPU_STEP]))
	CUDA_TRY(cuFuncSetBlockShape(L->gpu_kernel[GPU_STEP],
				L->threads_per_block[GPU_STEP], 1, 1))

	CUDA_TRY(cuModuleGetFunction(&L->gpu_kernel[GPU_FINAL],
				gpu_module, "sieve_kernel_final"))
	CUDA_TRY(cuFuncGetAttribute((int *)&L->threads_per_block[GPU_FINAL],
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				L->gpu_kernel[GPU_FINAL]))
	CUDA_TRY(cuFuncSetBlockShape(L->gpu_kernel[GPU_FINAL],
				L->threads_per_block[GPU_FINAL], 1, 1))

	CUDA_TRY(cuModuleGetFunction(&L->gpu_kernel[GPU_MERGE1],
				gpu_module, "sieve_kernel_merge1"))
	CUDA_TRY(cuFuncGetAttribute((int *)&i,
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				L->gpu_kernel[GPU_MERGE1]))
	L->threads_per_block[GPU_MERGE1] = 1 << (int)(log(i) / M_LN2);
	CUDA_TRY(cuFuncSetBlockShape(L->gpu_kernel[GPU_MERGE1],
				L->threads_per_block[GPU_MERGE1], 1, 1))

	CUDA_TRY(cuModuleGetFunction(&L->gpu_kernel[GPU_SORT],
				gpu_module, "sieve_kernel_sort"))
	CUDA_TRY(cuFuncGetAttribute((int *)&i,
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				L->gpu_kernel[GPU_SORT]))
	i = MIN(i, (L->poly->gpu_info->shared_mem_size - 4096) /
			SHARED_ELEM_SIZE / 2);
	L->threads_per_block[GPU_SORT] = 1 << (int)(log(i) / M_LN2);
	CUDA_TRY(cuFuncSetBlockShape(L->gpu_kernel[GPU_SORT],
				L->threads_per_block[GPU_SORT], 1, 1))
	CUDA_TRY(cuFuncSetSharedSize(L->gpu_kernel[GPU_SORT],
				2 * L->threads_per_block[GPU_SORT] *
				SHARED_ELEM_SIZE))

	CUDA_TRY(cuModuleGetFunction(&L->gpu_kernel[GPU_MERGE],
				gpu_module, "sieve_kernel_merge"))
	CUDA_TRY(cuFuncGetAttribute((int *)&i,
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				L->gpu_kernel[GPU_MERGE]))
	i = MIN(i, 2 * L->threads_per_block[GPU_SORT]);
	i = MIN(i, (L->poly->gpu_info->shared_mem_size - 4096) /
			SHARED_ELEM_SIZE / 2);
	L->threads_per_block[GPU_MERGE] = 1 << (int)(log(i) / M_LN2);
	CUDA_TRY(cuFuncSetBlockShape(L->gpu_kernel[GPU_MERGE],
				L->threads_per_block[GPU_MERGE], 1, 1))
	CUDA_TRY(cuFuncSetSharedSize(L->gpu_kernel[GPU_MERGE],
				2 * L->threads_per_block[GPU_MERGE] *
				SHARED_ELEM_SIZE))

	for (i = 2 * L->threads_per_block[GPU_SORT]; ; i *= 2) {

		if (i >= num_roots) {
			L->num_entries = i;
			break;
		}
	}

	CUDA_TRY(cuMemAlloc(&L->gpu_p_array,
			BATCH_SPECIALQ_MAX *
			L->num_entries * sizeof(uint32)))
	CUDA_TRY(cuMemsetD32(L->gpu_p_array, (uint32)(-1),
			BATCH_SPECIALQ_MAX * L->num_entries))
	CUDA_TRY(cuMemAlloc(&L->gpu_root_array,
			BATCH_SPECIALQ_MAX *
			L->num_entries * sizeof(uint64)))
	CUDA_TRY(cuMemsetD32(L->gpu_root_array, (uint32)(-1),
			2 * BATCH_SPECIALQ_MAX * L->num_entries))

	CUDA_TRY(cuMemAlloc(&L->gpu_q_array,
			sizeof(uint64) * BATCH_SPECIALQ_MAX))

	L->found_array_size = L->threads_per_block[GPU_FINAL] *
				L->poly->gpu_info->num_compute_units;
	CUDA_TRY(cuMemAlloc(&L->gpu_found_array, sizeof(found_t) *
			L->found_array_size))
	L->found_array = (found_t *)xmalloc(sizeof(found_t) *
			L->found_array_size);

	CUDA_TRY(cuMemsetD8(L->gpu_found_array, 0,
		L->found_array_size * sizeof(found_t)))

	num_roots = 0;
	for (i = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;
		uint32 num = soa->num_p;

		soa->dev_p_entry = L->gpu_p_array +
				num_roots * sizeof(uint32);
		soa->dev_root_entry = L->gpu_root_array +
				num_roots * sizeof(uint64);

		num_roots += num * soa->num_roots;
	}

	if (special_q_min == 1) {
		uint64 tmp_qroot = 0;

		quit = handle_special_q(obj, L, 1, 1, &tmp_qroot);
		if (quit || special_q_max == 1)
			goto finished;
	}

	sieve_fb_reset(sieve_special_q, special_q_min, special_q_max, 
			degree, MAX_ROOTS);
	while (sieve_fb_next(sieve_special_q, L->poly, store_specialq,
					&specialq) != P_SEARCH_DONE) {

		i = 0;
		while (i < specialq.num_roots) {
			uint32 curr_num_roots = MIN(specialq.num_roots - i,
							BATCH_SPECIALQ_MAX);

			quit = handle_special_q(obj, L,
				specialq.q, curr_num_roots,
				specialq.roots + i);

			if (++pass_cnt % 16 == 0)
				check_found_array(L);

			if (quit)
				goto finished;

			i += curr_num_roots;
		}

		if (get_cpu_time() - cpu_start_time > L->deadline) {
			quit = 1;
			goto finished;
		}
	}

finished:
	check_found_array(L);
	CUDA_TRY(cuMemFree(L->gpu_p_array))
	CUDA_TRY(cuMemFree(L->gpu_q_array))
	CUDA_TRY(cuMemFree(L->gpu_root_array))
	CUDA_TRY(cuMemFree(L->gpu_found_array))
	for (i = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;

		CUDA_TRY(cuMemFree(soa->dev_p));
		CUDA_TRY(cuMemFree(soa->dev_start_roots));
		CUDA_TRY(cuMemFree(soa->dev_roots));
		CUDA_TRY(cuStreamDestroy(soa->stream));
	}
	free(L->found_array);
	free(L->gpu_kernel);
	free(L->threads_per_block);
	p_soa_array_free(&p_array);
	CUDA_TRY(cuEventDestroy(L->start))
	CUDA_TRY(cuEventDestroy(L->end))
	L->deadline -= get_cpu_time() - cpu_start_time;
	return quit;
}

/*------------------------------------------------------------------------*/
uint32 
sieve_lattice_gpu_sort_core(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max)
{
	/* main driver for collision search */

	uint32 quit;
	sieve_fb_t sieve_p;
	uint32 p_min, p_max;
	uint32 degree = L->poly->degree;
	double p_size_max = L->poly->p_size_max;

	p_size_max /= special_q_max;
	if (sqrt(p_size_max * P_SCALE) > MAX_OTHER) {
		printf("error: invalid parameters for rational coefficient "
			"in sieve_lattice_gpu()\n");
		return 0;
	}

	/* size the problem; because special-q can have any factors,
	   we require that the progressions we generate use p that
	   have somewhat large factors. This minimizes the chance
	   that a given special-q has factors in common with many
	   progressions in the set */

	p_max = sqrt(p_size_max * P_SCALE);
	p_min = p_max / P_SCALE;

	gmp_printf("coeff %Zd specialq %u - %u other %u - %u\n",
			L->poly->high_coeff,
			special_q_min, special_q_max,
			p_min, p_max);

	sieve_fb_init(&sieve_p, L->poly, 
			100, 5000,
			1, degree,
		       	0);

	quit = sieve_specialq_64(obj, L,
			sieve_special_q, &sieve_p,
			special_q_min, special_q_max,
			p_min, p_max);

	sieve_fb_free(&sieve_p);
	return quit;
}
