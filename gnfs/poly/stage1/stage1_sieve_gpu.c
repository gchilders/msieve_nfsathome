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
#include <stage1_core_gpu/stage1_core.h>

/* GPU collision search; this code looks for self-collisions
   among arithmetic progressions, by finding k1 and k2 such that
   for two arithmetic progressions r1+k*p1^2 and r2+k*p2^2 we
   have

      r1 + k1*p1^2 = r2 + k2*p2^2

   such that
      - p1 and p2 are coprime and < 2^32
      - the value where they coincide is of size smaller
        than a fixed bound

   This code uses a sort routine to find collisions across all the
   p1 and p2 in the set simultaneously. We further use a 'special-q'
   formulation where all the inputs to the sort routine are
   constrained to fall on a third arithmetic progression r3 + k*q^2
   for some k. We choose a given q and for each of its roots run the
   complete sort. This is analogous to lattice sieving across the
   interval.
   
   This allows us to choose q so that the sort problem is of
   reasonable size but the collisions found are still over the
   original, impractically large range. */

enum {
	GPU_TRANS_PP32_R32 = 0,
	GPU_TRANS_PP32_R64,
	GPU_TRANS_PP64_R64,
	GPU_FINAL_32,
	GPU_FINAL_64,
	NUM_GPU_FUNCTIONS /* must be last */
};

static const char * gpu_kernel_names[] = 
{
	"sieve_kernel_trans_pp32_r32",
	"sieve_kernel_trans_pp32_r64",
	"sieve_kernel_trans_pp64_r64",
	"sieve_kernel_final_32",
	"sieve_kernel_final_64",
};

static const gpu_arg_type_list_t gpu_kernel_args[] = 
{
	/* sieve_kernel_trans_pp{32|64}_r{32|64} */
	{ 12,
		{
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		  GPU_ARG_UINT32,
		  GPU_ARG_UINT32,
		  GPU_ARG_UINT32,
		  GPU_ARG_UINT32,
		}
	},
	/* sieve_kernel_final_{32|64} */
	{ 6,
		{
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		  GPU_ARG_PTR,
		  GPU_ARG_PTR,
		  GPU_ARG_UINT32,
		}
	},
};

/*------------------------------------------------------------------------*/

typedef struct {
	uint32 num_roots;
	uint32 num_p;
	uint32 num_p_alloc;

	CUdeviceptr dev_p;
	CUdeviceptr dev_start_roots;
	CUstream stream;

	uint32 *p;

	union { 
		uint64 *roots64[MAX_ROOTS];
		uint32 *roots32[MAX_ROOTS];
		void *roots[MAX_ROOTS];
	} r;

} p_soa_var_t;

#define MAX_P_SOA_ARRAYS 16

typedef struct {
	uint32 num_arrays;
	uint32 pp_is_64;

	uint32 max_p_roots;
	p_soa_var_t *soa;

} p_soa_array_t;

static void
p_soa_array_init(p_soa_array_t *s, uint32 degree, uint32 pp_is_64)
{
	uint32 i, j;
	memset(s, 0, sizeof(p_soa_array_t));

	s->pp_is_64 = pp_is_64;
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
			soa->r.roots[j] = xmalloc(soa->num_p_alloc *
					((pp_is_64) ? sizeof(uint64) : 
					sizeof(uint32)));
		}
	}
}

static void
p_soa_var_free(p_soa_var_t *soa)
{
	uint32 i;

	free(soa->p);
	for (i = 0; i < soa->num_roots; i++)
		free(soa->r.roots[i]);
}

static void
p_soa_array_free(p_soa_array_t *s)
{
	uint32 i;

	for (i = 0; i < s->num_arrays; i++)
		p_soa_var_free(s->soa + i);

	free(s->soa);
}

static void
p_soa_array_compact(p_soa_array_t *s)
{
	uint32 i, j;

	for (i = j = 0; i < s->num_arrays; i++) {
		if (s->soa[i].num_p < 50) {
			p_soa_var_free(s->soa + i);
		}
		else {
			s->soa[j++] = s->soa[i];
		}
	}
	s->num_arrays = j;
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
				soa->r.roots[j] = xrealloc(soa->r.roots[j],
					soa->num_p_alloc * 
					((s->pp_is_64) ? sizeof(uint64) :
					 sizeof(uint32)));
			}
		}

		soa->p[soa->num_p] = p;
		if (s->pp_is_64) {
			for (j = 0; j < soa->num_roots; j++)
				soa->r.roots64[j][soa->num_p] = roots[j];
		}
		else {
			for (j = 0; j < soa->num_roots; j++)
				soa->r.roots32[j][soa->num_p] = (uint32)roots[j];
		}

		soa->num_p++;
	}
}

/*------------------------------------------------------------------------*/

typedef struct {
	uint32 num_specialq;
	uint32 max_specialq;
	specialq_t *specialq;
} specialq_array_t;

static void
specialq_array_init(specialq_array_t *q_array, uint32 max_specialq)
{
	memset(q_array, 0, sizeof(specialq_array_t));

	q_array->max_specialq = max_specialq;
	q_array->specialq = (specialq_t *)xmalloc(max_specialq *
						sizeof(specialq_t));
}

static void
specialq_array_free(specialq_array_t *q_array)
{
	free(q_array->specialq);
}

static void
specialq_array_reset(specialq_array_t *q_array, 
			uint32 num_removed)
{
	uint32 new_size;

	if (num_removed >= q_array->num_specialq) {
		q_array->num_specialq = 0;
		return;
	}

	new_size = q_array->num_specialq - num_removed;
	memmove(q_array->specialq, 
		q_array->specialq + num_removed,
		new_size * sizeof(specialq_t));
	q_array->num_specialq = new_size;
}

static void
store_specialq(uint32 q, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i;
	specialq_array_t *q_array = (specialq_array_t *)extra;

	for (i = 0; i < num_roots &&
			q_array->num_specialq < q_array->max_specialq; i++) {

		q_array->specialq[q_array->num_specialq].p = q;
		q_array->specialq[q_array->num_specialq].pp = (uint64)q * q;
		q_array->specialq[q_array->num_specialq].root = roots[i];

		q_array->num_specialq++;
	}
}

typedef struct {

	p_soa_array_t *p_array;

	uint32 num_entries;

	gpu_launch_t *launch;
	CUdeviceptr gpu_q_array;

	CUdeviceptr gpu_p_array;
	CUdeviceptr gpu_p_array_scratch;

	CUdeviceptr gpu_root_array;
	CUdeviceptr gpu_root_array_scratch;

	CUdeviceptr gpu_found_array;
	found_t *found_array;
	uint32 found_array_size;

	CUevent start;
	CUevent end;

	double gpu_elapsed;

} device_data_t;

/*------------------------------------------------------------------------*/
static void
check_found_array(poly_search_t *poly, device_data_t *d)
{
	uint32 i;
	uint32 found_array_size;
	found_t *found_array = d->found_array;

	CUDA_TRY(cuMemcpyDtoH(found_array, d->gpu_found_array,
			d->found_array_size * sizeof(found_t)))

	found_array_size = MIN(FOUND_ARRAY_SIZE - 1,
				found_array[0].p1);

	if (found_array_size == 0)
		return;

	/* clear only the first element */
	CUDA_TRY(cuMemsetD8(d->gpu_found_array, 0, sizeof(found_t)))

	for (i = 1; i <= found_array_size; i++) {
		found_t *found = found_array + i;
		uint32 p1 = found->p1;
		uint32 p2 = found->p2;
		uint32 q = found->q;
		uint64 qroot = found->qroot;
		int64 offset = found->offset;

		double dp = (double)q * p1 * p2;
		double coeff = poly->m0 * fabs((double)qroot + 
					(double)offset * q * q) /
					(dp * dp);

		if (coeff <= poly->coeff_max)
			handle_collision(poly, (uint64)p1 * p2, q,
					qroot, offset);
	}
}

#define MAX_SPECIAL_Q ((uint32)(-1))
#define MAX_OTHER ((uint32)1 << 27)

/*------------------------------------------------------------------------*/
static uint32
handle_special_q_batch(msieve_obj *obj, poly_search_t *poly,
			device_data_t *d, specialq_t *specialq_batch,
			uint32 num_specialq, uint32 shift,
			uint32 key_bits, uint32 num_aprog_vals)
{
	uint32 i, j;
	uint32 quit = 0;
	p_soa_array_t *p_array = d->p_array;
	uint32 num_blocks;
	float elapsed_ms;
	gpu_arg_t gpu_args[GPU_MAX_KERNEL_ARGS];
	sort_data_t sort_data;
	gpu_launch_t *launch;
	uint32 num_q, curr_q;
	uint32 root_bytes = (key_bits > 32) ? sizeof(uint64) : sizeof(uint32);

	CUDA_TRY(cuEventRecord(d->start, 0))

	for (i = num_q = curr_q = 0; i < num_specialq; i++) {
		if (specialq_batch[i].p != curr_q) {
			num_q++;
			curr_q = specialq_batch[i].p;
		}
	}
	
	CUDA_TRY(cuMemcpyHtoD(d->gpu_q_array, specialq_batch,
			sizeof(specialq_t) * num_specialq))

	CUDA_TRY(cuMemsetD8(d->gpu_root_array, 0,
			num_specialq * d->num_entries * 
			num_aprog_vals * root_bytes))

	for (i = j = 0; i < p_array->num_arrays; i++) {
		p_soa_var_t *soa = p_array->soa + i;
		uint32 num_p = soa->num_p;
		uint32 blocks_x, blocks_y;
		uint32 size_x, size_y;
		uint32 total_blocks;

		if (p_array->pp_is_64)
			launch = d->launch + GPU_TRANS_PP64_R64;
		else if (root_bytes == sizeof(uint64))
			launch = d->launch + GPU_TRANS_PP32_R64;
		else
			launch = d->launch + GPU_TRANS_PP32_R32;

		/* perform a block decomposition so that all the
		   soa's generate blocks with about the same amount
		   of arithmetic. There is a modular multiply for
		   each root and a modular inverse for each (p,q) pair, 
		   which we count as 5 multiplies */

		total_blocks = (5 * num_p * num_q +
			        num_p * soa->num_roots * num_specialq) /
				20000;
		total_blocks = MIN(total_blocks, 1000);
		total_blocks = MAX(total_blocks, 1);

		/* choose the number of threads per block to be
		   - a multiple of the warp size between 128 and 256
		   - that can generate the desired number of blocks
		     so the whole dataset is covered, while maximizing
		     the size of the blocks on the borders */

		size_x = MIN(256, launch->threads_per_block);
		while (1) {

			blocks_x = (num_p + size_x - 1) / size_x;
			blocks_y = (total_blocks + blocks_x - 1) / blocks_x;
			size_y = (num_specialq + blocks_y - 1) / blocks_y;

			if (size_x == 128 ||
			    blocks_x * size_x - num_p <= size_x / 3)
				break;

			size_x -= poly->gpu_info->warp_size;
		}

		gpu_args[0].ptr_arg = (void *)(size_t)soa->dev_p;
		gpu_args[1].uint32_arg = num_p;
		gpu_args[2].ptr_arg = (void *)(size_t)soa->dev_start_roots;
		gpu_args[3].uint32_arg = soa->num_roots;
		gpu_args[4].ptr_arg = (void *)(size_t)(
				d->gpu_p_array + j * sizeof(uint32));
		gpu_args[5].ptr_arg = (void *)(size_t)(
				d->gpu_root_array + j * root_bytes);
		gpu_args[6].ptr_arg = (void *)(size_t)d->gpu_q_array;
		gpu_args[7].uint32_arg = num_specialq;
		gpu_args[8].uint32_arg = size_y;
		gpu_args[9].uint32_arg = d->num_entries;
		gpu_args[10].uint32_arg = shift;
		gpu_args[11].uint32_arg = num_aprog_vals;
		gpu_launch_set(launch, gpu_args);

		CUDA_TRY(cuFuncSetBlockShape(launch->kernel_func, 
				size_x, 1, 1))

		CUDA_TRY(cuLaunchGridAsync(launch->kernel_func,
				blocks_x, blocks_y, soa->stream))

		j += num_p * soa->num_roots;
	}

	sort_data.keys_in = d->gpu_root_array;
	sort_data.keys_in_scratch = d->gpu_root_array_scratch;
	sort_data.data_in = d->gpu_p_array;
	sort_data.data_in_scratch = d->gpu_p_array_scratch;
	sort_data.num_elements = num_specialq * d->num_entries * num_aprog_vals;
	sort_data.num_arrays = 1;
	sort_data.key_bits = key_bits;
	poly->sort_engine_run(poly->sort_engine, &sort_data);

	if (root_bytes == sizeof(uint64))
		launch = d->launch + GPU_FINAL_64;
	else
		launch = d->launch + GPU_FINAL_32;

	gpu_args[0].ptr_arg = (void *)(size_t)(d->gpu_p_array);
	gpu_args[1].ptr_arg = (void *)(size_t)(d->gpu_root_array);
	gpu_args[2].uint32_arg = num_specialq * d->num_entries * num_aprog_vals;
	gpu_args[3].ptr_arg = (void *)(size_t)(d->gpu_q_array);
	gpu_args[4].ptr_arg = (void *)(size_t)(d->gpu_found_array);
	gpu_args[5].uint32_arg = shift;
	gpu_launch_set(launch, gpu_args);

	num_blocks = 1 + (num_specialq * d->num_entries * 
			num_aprog_vals - 1) /
			launch->threads_per_block;
	num_blocks = MIN(num_blocks, 1000);

	CUDA_TRY(cuLaunchGrid(launch->kernel_func, num_blocks, 1))

	CUDA_TRY(cuEventRecord(d->end, 0))
	CUDA_TRY(cuEventSynchronize(d->end))
	CUDA_TRY(cuEventElapsedTime(&elapsed_ms, d->start, d->end))
	if (elapsed_ms < 60000 && elapsed_ms > 0) {
		/* this function should execute in under a second. If
		   it takes a very long time, assume that the system
		   was in hibernation and don't let it count. */
		d->gpu_elapsed += elapsed_ms / 1000;
	}

	if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
		quit = 1;

	return quit;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_specialq(msieve_obj *obj, poly_search_t *poly,
		sieve_fb_t *sieve_special_q, sieve_fb_t *sieve_p,
		uint32 special_q_min, uint32 special_q_max,
		uint32 p_min, uint32 p_max, double deadline, 
		double *elapsed)
{
	uint32 i, j;
	uint32 quit = 0;
	uint32 all_q_done = 0;
	uint32 degree = poly->degree;
	specialq_array_t q_array;
	p_soa_array_t p_array;
	device_data_t data;
	double cpu_start_time = get_cpu_time();
	CUmodule gpu_module = poly->gpu_module;
	uint32 unused_bits;
	uint32 pp_is_64 = (p_max >= 65536);
	uint32 start_root_bytes = pp_is_64 ? sizeof(uint64) : sizeof(uint32);
	uint32 max_batch_specialq32;
	uint32 max_batch_specialq64;

	*elapsed = 0;

	memset(&data, 0, sizeof(device_data_t));

	data.p_array = &p_array;
	p_soa_array_init(&p_array, degree, pp_is_64);

	/* build all the arithmetic progressions */

	sieve_fb_reset(sieve_p, p_min, p_max, 1, p_array.max_p_roots);
	while (sieve_fb_next(sieve_p, poly, store_p_soa,
			&p_array) != P_SEARCH_DONE) {
		;
	}

	p_soa_array_compact(&p_array);
	if (p_array.num_arrays == 0) {
		p_soa_array_free(&p_array);
		return 0;
	}

	for (i = j = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;

		j += soa->num_p * soa->num_roots;
	}

	data.num_entries = j;

#if 1
	printf("aprogs: %u roots\n", data.num_entries);
#endif

	/* a single transformed array will not have enough
	   elements for the GPU to sort efficiently; instead,
	   we batch multiple arrays together and sort the
	   entire batch. Array elements get their array number
	   added into unused bits above each p, so the number
	   of such unused bits is one limit on the batch size.
	   Another limit is the amount of RAM on the card;
	   we try to use less than 30% of it. Finally, we top
	   out at a few ten millions of elements, which is far 
	   into asymptotically optimal throughput territory 
	   for all current GPUs
	
	   The sizing below assumes we can batch as many special-Q
	   as we want. However, for small degree-5 problems the 
	   number of special-Q available is too small to efficiently 
	   use the card. In that case, we will compensate later by 
	   making each arithmetic progression contribute multiple 
	   offsets */

	unused_bits = 1;
	while (!(p_max & (1 << (31 - unused_bits))))
		unused_bits++;

	max_batch_specialq32 = 30000000 / data.num_entries;

	max_batch_specialq32 = MIN(max_batch_specialq32,
				(uint32)(0.3 * poly->gpu_info->global_mem_size /
				 (2 * (sizeof(uint32) + sizeof(uint32)) * 
				  data.num_entries)));

	max_batch_specialq64 = 20000000 / data.num_entries;

	max_batch_specialq64 = MIN(max_batch_specialq64,
				(uint32)(0.3 * poly->gpu_info->global_mem_size /
				 (2 * (sizeof(uint32) + sizeof(uint64)) * 
				  data.num_entries)));

	if (pp_is_64) {

		/* roots to be sorted are always 64-bit integers */

		i = max_batch_specialq64 * sizeof(uint32);
		j = max_batch_specialq64 * sizeof(uint64);
	}
	else {
		/* roots to be sorted are 32-bit integers except
		   for small degree-5 problems, where they may
		   be 64-bit integers if that's more efficient.
		   Allocate the worst-case memory */

		i = max_batch_specialq32 * sizeof(uint32);
		j = MAX(max_batch_specialq64 * sizeof(uint64),
			max_batch_specialq32 * sizeof(uint32));
	}

	CUDA_TRY(cuMemAlloc(&data.gpu_p_array,
			data.num_entries * i))
	CUDA_TRY(cuMemAlloc(&data.gpu_p_array_scratch,
			data.num_entries * i))
	CUDA_TRY(cuMemAlloc(&data.gpu_root_array,
			data.num_entries * j))
	CUDA_TRY(cuMemAlloc(&data.gpu_root_array_scratch,
			data.num_entries * j))

	/* allocate the q's one batch at a time */

	specialq_array_init(&q_array, MAX_ROOTS + (pp_is_64 ? 
					max_batch_specialq64 :
					max_batch_specialq32));

	CUDA_TRY(cuMemAlloc(&data.gpu_q_array, sizeof(specialq_t) *
				(pp_is_64 ? max_batch_specialq64 :
				 	max_batch_specialq32)))

	/* set up the root generator */

	for (i = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;
		uint32 num = soa->num_p;

		CUDA_TRY(cuStreamCreate(&soa->stream, 0))

		CUDA_TRY(cuMemAlloc(&soa->dev_p, num * sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&soa->dev_start_roots,
				soa->num_roots * num * start_root_bytes))

		CUDA_TRY(cuMemcpyHtoD(soa->dev_p, soa->p,
				num * sizeof(uint32)))

		for (j = 0; j < soa->num_roots; j++) {
			CUDA_TRY(cuMemcpyHtoD(soa->dev_start_roots +
					j * num * start_root_bytes,
					soa->r.roots[j], 
					num * start_root_bytes))
		}
	}

	/* load GPU kernels */

	data.launch = (gpu_launch_t *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(gpu_launch_t));

	for (i = 0; i < NUM_GPU_FUNCTIONS; i++) {
		gpu_launch_t *launch = data.launch + i;

		gpu_launch_init(gpu_module, gpu_kernel_names[i],
				gpu_kernel_args + (i / 3), launch);

		if (i == GPU_FINAL_32 || i == GPU_FINAL_64) {
			/* performance of the cleanup functions is not 
			   that sensitive to the block shape; set it 
			   once up front */

			launch->threads_per_block = 
					MIN(256, launch->threads_per_block);
			CUDA_TRY(cuFuncSetBlockShape(launch->kernel_func,
					launch->threads_per_block, 1, 1))
		}
	}

	CUDA_TRY(cuEventCreate(&data.start, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&data.end, CU_EVENT_BLOCKING_SYNC))

	/* set up found array */

	data.found_array_size = FOUND_ARRAY_SIZE;
	CUDA_TRY(cuMemAlloc(&data.gpu_found_array, sizeof(found_t) *
			data.found_array_size))
	data.found_array = (found_t *)xmalloc(sizeof(found_t) *
			data.found_array_size);

	CUDA_TRY(cuMemsetD8(data.gpu_found_array, 0, sizeof(found_t)))

	/* account for 'trivial' special-q */

	if (special_q_min == 1) {

		q_array.specialq[0].p = 1;
		q_array.specialq[0].root = 0;
		q_array.num_specialq++;
	}

	/* handle special-q in batches */

	sieve_fb_reset(sieve_special_q, special_q_min, 
			special_q_max, degree, MAX_ROOTS);

	while (!quit && !all_q_done) {

		uint32 batch_size;
		uint32 max_batch_size;
		uint32 key_bits;
		uint32 num_aprog_vals = 1;

		max_batch_size = pp_is_64 ? 
				max_batch_specialq64 :
				max_batch_specialq32;
		max_batch_size = MIN(max_batch_size, 
				(uint32)1 << unused_bits);

		while (q_array.num_specialq < max_batch_size) {
			if (sieve_fb_next(sieve_special_q, poly,
				store_specialq, &q_array) == P_SEARCH_DONE) {

				all_q_done = 1;
				break;
			}
		}

		batch_size = MIN(max_batch_size, q_array.num_specialq);
		if (batch_size == 0)
			continue;

		if (batch_size < max_batch_size / 3) {

			/* current batch of q is too small to utilize 
			   the card efficiently. Have each (p,q) pair
			   generate multiple offsets, centered about 0 */

			num_aprog_vals = max_batch_size / batch_size;

			/* if we were set up for 32-bit sort keys but
			   now require 64-bit sort keys, make sure to 
			   respect the 64-bit special-q limit */

			key_bits = 1 + ceil(log((double)p_max * p_max *
					((num_aprog_vals + 1) / 2)) / M_LN2);

			if (key_bits > 32)
				num_aprog_vals = MIN(max_batch_size,
				        max_batch_specialq64) / batch_size;
		}

		key_bits = ceil(log((double)p_max * p_max *
				((num_aprog_vals + 1) / 2)) / M_LN2);
		if (num_aprog_vals > 1)
			key_bits++;

		quit = handle_special_q_batch(obj, poly, &data,
				q_array.specialq, batch_size,
				32 - unused_bits, key_bits, 
				num_aprog_vals);

		check_found_array(poly, &data);

		specialq_array_reset(&q_array, batch_size);

		*elapsed = get_cpu_time() - cpu_start_time +
				data.gpu_elapsed;
		if (*elapsed > deadline)
			quit = 1;
	}

	CUDA_TRY(cuMemFree(data.gpu_p_array))
	CUDA_TRY(cuMemFree(data.gpu_p_array_scratch))
	CUDA_TRY(cuMemFree(data.gpu_root_array))
	CUDA_TRY(cuMemFree(data.gpu_root_array_scratch))
	CUDA_TRY(cuMemFree(data.gpu_found_array))
	CUDA_TRY(cuMemFree(data.gpu_q_array))
	for (i = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;

		CUDA_TRY(cuMemFree(soa->dev_p));
		CUDA_TRY(cuMemFree(soa->dev_start_roots));
		CUDA_TRY(cuStreamDestroy(soa->stream));
	}
	free(data.found_array);
	free(data.launch);
	p_soa_array_free(&p_array);
	specialq_array_free(&q_array);
	CUDA_TRY(cuEventDestroy(data.start))
	CUDA_TRY(cuEventDestroy(data.end))
	return quit;
}

/*------------------------------------------------------------------------*/
double
sieve_lattice_gpu(msieve_obj *obj, poly_search_t *poly, double deadline)
{
	uint32 degree = poly->degree;
	uint32 num_pieces;
	uint32 p_min, p_max;
	uint32 special_q_min, special_q_max;
	uint32 special_q_min2, special_q_max2;
	uint32 special_q_fb_max;
	double p_size_max = poly->p_size_max;
	double sieve_bound = poly->coeff_max / poly->m0;
	double elapsed = 0;
	sieve_fb_t sieve_p, sieve_special_q;

	/* size the problem; we choose p_min so that we can use
	   exactly one offset from each progression (the one
	   nearest to m0) in the search. Choosing larger p
	   implies that we could use more of their offsets, but
	   it appears not to be optimal to do so since the
	   biggest part of the search difficulty is the sorting
	   phase, and larger p implies that we need to sort more
	   of them to find each collision */

	p_min = MIN(MAX_OTHER / P_SCALE, sqrt(0.5 / sieve_bound));
	p_min = MIN(p_min, sqrt(p_size_max) / P_SCALE);

	p_max = p_min * P_SCALE;

	special_q_min = 1;
	special_q_max = MIN(MAX_SPECIAL_Q, p_size_max / p_max / p_max);
	special_q_max = MAX(special_q_max, 1);

	/* set up the special q factory; special-q may have 
	   arbitrary factors, but many small factors are 
	   preferred since that will allow for many more roots
	   per special q, so we choose the factors to be as 
	   small as possible */

	special_q_fb_max = MIN(200000, special_q_max);
	sieve_fb_init(&sieve_special_q, poly,
			2, special_q_fb_max,
			1, degree,
			1);

	/* because special-q can have any factors, we require that
	   the progressions we generate use p that have somewhat
	   large factors. This minimizes the chance that a given
	   special-q has factors in common with many progressions
	   in the set */

	sieve_fb_init(&sieve_p, poly, 
			100, 5000,
			1, degree,
		       	0);

	/* large search problems can be randomized so that
	   multiple runs over the same range of leading
	   a_d will likely generate different results */

	num_pieces = MIN(450, (double)special_q_max * p_max
				/ log(special_q_max) / log(p_max)
				/ 3e9);

	if (num_pieces > 1) { /* randomize the special_q range */
		uint32 piece_length = (special_q_max - special_q_min)
				/ num_pieces;
		uint32 piece = get_rand(&obj->seed1, &obj->seed2)
				% num_pieces;

		printf("randomizing rational coefficient: "
			"using piece #%u of %u\n",
			piece + 1, num_pieces);

		special_q_min2 = special_q_min + piece * piece_length;
		special_q_max2 = special_q_min2 + piece_length;
	}
	else {
		special_q_min2 = special_q_min;
		special_q_max2 = special_q_max;
	}

	gmp_printf("coeff %Zd specialq %u - %u other %u - %u\n",
			poly->high_coeff,
			special_q_min2, special_q_max2,
			p_min, p_max);

	sieve_specialq(obj, poly, &sieve_special_q, &sieve_p,
			special_q_min2, special_q_max2, p_min, p_max,
			deadline, &elapsed);

	sieve_fb_free(&sieve_special_q);
	sieve_fb_free(&sieve_p);
	return elapsed;
}
