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

#ifdef HAVE_CUDA
#include <sort_engine.h> /* interface to GPU sorting library */
#endif

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

	uint32 *p;
	uint64 *start_roots;

	CUdeviceptr dev_p;
	CUdeviceptr dev_start_roots;
	CUstream stream;

	union { 
		uint64 *roots64;
		uint32 *roots32;
		void *roots;
	} r;

} p_soa_var_t;

#define MAX_P_SOA_ARRAYS 5

typedef struct {
	uint32 num_arrays;
	uint32 max_arrays;
	uint32 max_p_roots;
	uint32 pp_is_64;

	p_soa_var_t start_soa[MAX_P_SOA_ARRAYS];
	p_soa_var_t *soa[MAX_P_SOA_ARRAYS];
} p_soa_array_t;

static p_soa_array_t *
p_soa_array_init(uint32 degree)
{
	uint32 i;
	p_soa_array_t *s = (p_soa_array_t *)xcalloc(1, 
					sizeof(p_soa_array_t));
	p_soa_var_t *start_soa = s->start_soa;

	switch (degree) {
	case 4:
		s->max_arrays = 3;
		start_soa[0].num_roots = 2;
		start_soa[1].num_roots = 4;
		start_soa[2].num_roots = 8;
		break;

	case 5:
		s->max_arrays = 3;
		start_soa[0].num_roots = 1;
		start_soa[1].num_roots = 5;
		start_soa[2].num_roots = 25;
		break;

	case 6:
		s->max_arrays = 5;
		start_soa[0].num_roots = 2;
		start_soa[1].num_roots = 4;
		start_soa[2].num_roots = 6;
		start_soa[3].num_roots = 12;
		start_soa[4].num_roots = 36;
		break;

	case 7: /* ;) */
		s->max_arrays = 3;
		start_soa[0].num_roots = 1;
		start_soa[1].num_roots = 7;
		start_soa[2].num_roots = 49;
		break;
	}
	s->max_p_roots = start_soa[s->max_arrays - 1].num_roots;

	for (i = 0; i < s->max_arrays; i++) {
		p_soa_var_t *soa = s->start_soa + i;

		soa->num_p = 0;
		soa->num_p_alloc = 256;
		soa->p = (uint32 *)xmalloc(soa->num_p_alloc * sizeof(uint32));
		soa->start_roots = (uint64 *)xmalloc(soa->num_roots *
					soa->num_p_alloc * sizeof(uint64));
		soa->r.roots = xmalloc(soa->num_roots *
					soa->num_p_alloc * sizeof(uint64));
		CUDA_TRY(cuMemAlloc(&soa->dev_p, 
					soa->num_p_alloc * 
					sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&soa->dev_start_roots, 
					soa->num_p_alloc *
					soa->num_roots * sizeof(uint64)))
		CUDA_TRY(cuStreamCreate(&soa->stream, 0))
	}

	return s;
}

static void
p_soa_array_free(p_soa_array_t *s)
{
	uint32 i;

	for (i = 0; i < s->max_arrays; i++) {
		p_soa_var_t *soa = s->start_soa + i;

		free(soa->p);
		free(soa->start_roots);
		free(soa->r.roots);
		CUDA_TRY(cuMemFree(soa->dev_p));
		CUDA_TRY(cuMemFree(soa->dev_start_roots));
		CUDA_TRY(cuStreamDestroy(soa->stream));
	}

	free(s);
}

static void
p_soa_array_reset(p_soa_array_t *s)
{
	uint32 i;

	s->num_arrays = 0;
	s->pp_is_64 = 0;
	for (i = 0; i < s->max_arrays; i++)
		s->start_soa[i].num_p = 0;
}

static void
p_soa_array_start(p_soa_array_t *s, uint32 pp_is_64)
{
	uint32 i, j, k, m;
	uint32 root_bytes = pp_is_64 ? sizeof(uint64) : sizeof(uint32);

	for (i = j = 0; i < s->max_arrays; i++) {
		p_soa_var_t *soa = s->start_soa + i;
		uint32 num_p = soa->num_p;
		uint32 num_roots = soa->num_roots;
		uint64 *rs = soa->start_roots;
		void *root_array = soa->r.roots;

		if (soa->num_p < 50)
			continue;

		if (pp_is_64) {

			if (num_roots == 1) {
				root_array = rs;
			}
			else {
				uint64 *rd = soa->r.roots64;

				for (k = 0; k < num_p; k++) {
					uint64 *rd2 = rd;

					for (m = 0; m < num_roots; m++) {
						*rd2 = rs[m];
						rd2 += num_p;
					}
					rs += num_roots;
					rd++;
				}
			}
		}
		else {
			uint32 *rd = soa->r.roots32;

			for (k = 0; k < num_p; k++) {
				uint32 *rd2 = rd;

				for (m = 0; m < num_roots; m++) {
					*rd2 = (uint32)rs[m];
					rd2 += num_p;
				}
				rs += num_roots;
				rd++;
			}
		}

		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_p, 
				soa->p,
				num_p * sizeof(uint32),
				soa->stream))
		CUDA_TRY(cuMemcpyHtoDAsync(soa->dev_start_roots, 
				root_array,
				num_p * num_roots * root_bytes,
				soa->stream))
		s->soa[j++] = soa;
	}
	s->num_arrays = j;
	s->pp_is_64 = pp_is_64;
}

static void
store_p_soa(uint32 p, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i, j;
	p_soa_array_t *s = (p_soa_array_t *)extra;

	for (i = 0; i < s->max_arrays; i++) {

		p_soa_var_t *soa = s->start_soa + i;
		uint32 num_p;

		if (soa->num_roots != num_roots)
			continue;

		num_p = soa->num_p;

		if (soa->num_p_alloc == num_p) {
			soa->num_p_alloc *= 2;
			soa->p = (uint32 *)xrealloc(soa->p, soa->num_p_alloc *
							sizeof(uint32));
			soa->start_roots = (uint64 *)xrealloc(soa->start_roots,
						soa->num_p_alloc *
						num_roots *
						sizeof(uint64));
			soa->r.roots = xrealloc(soa->r.roots,
						soa->num_p_alloc *
						num_roots *
						sizeof(uint64));

			CUDA_TRY(cuMemFree(soa->dev_p));
			CUDA_TRY(cuMemFree(soa->dev_start_roots));
			CUDA_TRY(cuMemAlloc(&soa->dev_p, 
					soa->num_p_alloc * 
					sizeof(uint32)))
			CUDA_TRY(cuMemAlloc(&soa->dev_start_roots, 
					soa->num_p_alloc *
					soa->num_roots * sizeof(uint64)))
		}

		soa->p[num_p] = p;
		for (j = 0; j < num_roots; j++)
			soa->start_roots[num_p * num_roots + j] = roots[j];
		soa->num_p++;
		return;
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

	CUcontext gpu_context;
	gpu_info_t *gpu_info;
	CUmodule gpu_module;

	libhandle_t sort_engine_handle;
	void * sort_engine;
	sort_engine_init_func sort_engine_init;
	sort_engine_free_func sort_engine_free;
	sort_engine_run_func sort_engine_run;

	p_soa_array_t *p_array;

	uint32 num_entries;

	gpu_launch_t *launch;
	CUdeviceptr gpu_q_array;

	size_t max_sort_entries32;
	size_t max_sort_entries64;

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
check_found_array(poly_search_t *poly, poly_coeff_t *c,
		device_data_t *d)
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
		double coeff = c->m0 * fabs((double)qroot + 
					(double)offset * q * q) /
					(dp * dp);

		if (coeff <= c->coeff_max &&
		    handle_collision(c, (uint64)p1 * p2, q,
					qroot, offset) != 0) {
			poly->callback(c->high_coeff, c->p, c->m,
					poly->callback_data);
		}
	}
}

#define MAX_SPECIAL_Q ((uint32)(-1))
#define MAX_OTHER ((uint32)1 << 27)

/*------------------------------------------------------------------------*/
static uint32
handle_special_q_batch(msieve_obj *obj, device_data_t *d, 
			specialq_t *specialq_batch, uint32 num_specialq,
		       	uint32 shift, uint32 key_bits, uint32 num_aprog_vals)
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
		p_soa_var_t *soa = p_array->soa[i];
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
		   which we count as 3 multiplies */

		total_blocks = (3 * num_p * num_q +
			        num_p * soa->num_roots * num_specialq) /
				50000;
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

			size_x -= d->gpu_info->warp_size;
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
	sort_data.stream = 0;
	d->sort_engine_run(d->sort_engine, &sort_data);

	/* the sort engine may have swapped the input arrays */
	d->gpu_p_array = sort_data.data_in;
	d->gpu_p_array_scratch = sort_data.data_in_scratch;
	d->gpu_root_array = sort_data.keys_in;
	d->gpu_root_array_scratch = sort_data.keys_in_scratch;

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
		poly_coeff_t *c, void *gpu_data,
		void *sieve_special_q, void *sieve_p,
		uint32 special_q_min, uint32 special_q_max,
		uint32 p_min, uint32 p_max, double deadline, 
		double *elapsed)
{
	uint32 i, j;
	uint32 quit = 0;
	uint32 all_q_done = 0;
	uint32 degree = poly->degree;
	specialq_array_t q_array;
	device_data_t *data = (device_data_t *)gpu_data;
	p_soa_array_t *p_array = data->p_array;
	double cpu_start_time = get_cpu_time();
	uint32 unused_bits;
	uint32 pp_is_64 = (p_max >= 65536);
	uint32 max_batch_specialq32;
	uint32 max_batch_specialq64;

	*elapsed = 0;
	data->gpu_elapsed = 0;

	/* build all the arithmetic progressions */

	p_soa_array_reset(p_array);
	sieve_fb_reset(sieve_p, p_min, p_max, 1, p_array->max_p_roots);
	while (sieve_fb_next(sieve_p, c, store_p_soa,
			p_array) != P_SEARCH_DONE) {
		;
	}

	p_soa_array_start(p_array, pp_is_64);
	if (p_array->num_arrays == 0)
		return 0;

	for (i = j = 0; i < p_array->num_arrays; i++) {
		p_soa_var_t *soa = p_array->soa[i];

		j += soa->num_p * soa->num_roots;
	}

	data->num_entries = j;

#if 1
	printf("aprogs: %u roots\n", data->num_entries);
#endif


	unused_bits = 1;
	while (!(p_max & (1 << (31 - unused_bits))))
		unused_bits++;

	max_batch_specialq32 = data->max_sort_entries32 / data->num_entries;
	max_batch_specialq64 = data->max_sort_entries64 / data->num_entries;

	/* allocate the q's one batch at a time */

	specialq_array_init(&q_array, MAX_ROOTS + (pp_is_64 ? 
					max_batch_specialq64 :
					max_batch_specialq32));

	CUDA_TRY(cuMemAlloc(&data->gpu_q_array, sizeof(specialq_t) *
				(pp_is_64 ? max_batch_specialq64 :
				 	max_batch_specialq32)))

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
			if (sieve_fb_next(sieve_special_q, c,
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

		quit = handle_special_q_batch(obj, data,
				q_array.specialq, batch_size,
				32 - unused_bits, key_bits, 
				num_aprog_vals);

		check_found_array(poly, c, data);

		specialq_array_reset(&q_array, batch_size);

		*elapsed = get_cpu_time() - cpu_start_time +
				data->gpu_elapsed;
		if (*elapsed > deadline)
			quit = 1;
	}

	CUDA_TRY(cuMemFree(data->gpu_q_array))
	specialq_array_free(&q_array);
	return quit;
}

/*------------------------------------------------------------------------*/
double
sieve_lattice_gpu(msieve_obj *obj, poly_search_t *poly, 
		poly_coeff_t *c, void *gpu_data, double deadline)
{
	uint32 degree = poly->degree;
	uint32 num_pieces;
	uint32 p_min, p_max;
	uint32 special_q_min, special_q_max;
	uint32 special_q_min2, special_q_max2;
	uint32 special_q_fb_max;
	double elapsed = 0;
	double target = c->coeff_max / c->m0;
	void * sieve_p = sieve_fb_alloc();
	void * sieve_special_q = sieve_fb_alloc();

	/* Kleinjung shows that the third-to-largest algebraic
	   polynomial coefficient is of size approximately

	             (correction to m0) * m0
		    --------------------------
		    (leading rational coeff)^2
	
	   We have a bound 'coeff_max' on what this number is 
	   supposed to be, and we know m0 and an upper bound on 
	   the size of the leading rational coefficient P. Let 
	   P = p1*p2*q, where p1 and p2 are drawn from a fixed
	   set of candidates, and q (the 'special-q') is arbitrary
	   except that gcd(q,p1,p2)=1. Then the correction 'C' to 
	   m0 is < max(p1,p2)^2 */
	   
	p_max = MIN(MAX_OTHER, sqrt(c->p_size_max));
	p_max = MIN(p_max, sqrt(0.5 / target));

	special_q_max = MIN(MAX_SPECIAL_Q, 
			    c->p_size_max / p_max / p_max);
	special_q_max = MAX(special_q_max, 1);

	p_min = MAX(1, p_max / P_SCALE);
	special_q_min = 1;

	/* set up the special q factory; special-q may have 
	   arbitrary factors, but many small factors are 
	   preferred since that will allow for many more roots
	   per special q, so we choose the factors to be as 
	   small as possible */

	special_q_fb_max = MIN(200000, special_q_max);
	sieve_fb_init(sieve_special_q, c,
			2, special_q_fb_max,
			1, degree,
			1);

	/* because special-q can have any factors, we require that
	   the progressions we generate use p that have somewhat
	   large factors. This minimizes the chance that a given
	   special-q has factors in common with many progressions
	   in the set */

	sieve_fb_init(sieve_p, c, 
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
			c->high_coeff,
			special_q_min2, special_q_max2,
			p_min, p_max);

	sieve_specialq(obj, poly, c, gpu_data, sieve_special_q, sieve_p,
			special_q_min2, special_q_max2, p_min, p_max,
			deadline, &elapsed);

	sieve_fb_free(sieve_special_q);
	sieve_fb_free(sieve_p);
	return elapsed;
}

/*------------------------------------------------------------------------*/
static void
load_sort_engine(msieve_obj *obj, device_data_t *d)
{
	char libname[256];
	const char *arch;
	#if defined(WIN32) || defined(_WIN64)
	const char *suffix = ".dll";
	#else
	const char *suffix = ".so";
	#endif

	if (d->gpu_info->compute_version_major >= 2)
		arch = "sm20";
	else if (d->gpu_info->compute_version_minor >= 3)
		arch = "sm13";
	else
		arch = "sm10";

	sprintf(libname, "sort_engine_%s%s", arch, suffix);

	/* override from input args */

	if (obj->nfs_args != NULL) {
		char *tmp = strstr(obj->nfs_args, "sortlib=");

		if (tmp != NULL) {
			uint32 i;
			for (i = 0, tmp += 8; i < sizeof(libname) - 1; i++) {
				if (*tmp == 0 || isspace(*tmp))
					break;

				libname[i] = *tmp++;
			}
			libname[i] = 0;
		}
	}

	d->sort_engine_handle = load_dynamic_lib(libname);
	if (d->sort_engine_handle == NULL) {
		printf("error: failed to load GPU sorting engine\n");
		exit(-1);
	}

	/* the sort engine uses the same CUDA context */

	d->sort_engine_init = get_lib_symbol(
					d->sort_engine_handle,
					"sort_engine_init");
	d->sort_engine_free = get_lib_symbol(
					d->sort_engine_handle,
					"sort_engine_free");
	d->sort_engine_run = get_lib_symbol(
					d->sort_engine_handle,
					"sort_engine_run");
	if (d->sort_engine_init == NULL ||
	    d->sort_engine_free == NULL ||
	    d->sort_engine_run == NULL) {
		printf("error: cannot find GPU sorting function\n");
		exit(-1);
	}
	d->sort_engine = d->sort_engine_init();
}

/*------------------------------------------------------------------------*/
void *
gpu_data_init(msieve_obj *obj, poly_search_t *poly)
{
	uint32 i, j;
	device_data_t *d;
	gpu_config_t gpu_config;
	gpu_info_t *gpu_info;
	size_t sort_entries32;
	size_t sort_entries64;

	gpu_init(&gpu_config);
	if (gpu_config.num_gpu == 0) {
		printf("error: no CUDA-enabled GPUs found\n");
		exit(-1);
	}
	if (obj->which_gpu >= (uint32)gpu_config.num_gpu) {
		printf("error: GPU %u does not exist "
			"or is not CUDA-enabled\n", obj->which_gpu);
		exit(-1);
	}

	d = (device_data_t *)xcalloc(1, sizeof(device_data_t));

	d->gpu_info = gpu_info = (gpu_info_t *)xmalloc(sizeof(gpu_info_t));
	memcpy(gpu_info, gpu_config.info + obj->which_gpu,
			sizeof(gpu_info_t)); 

	logprintf(obj, "using GPU %u (%s)\n", obj->which_gpu, gpu_info->name);
	logprintf(obj, "selected card has CUDA arch %d.%d\n",
			gpu_info->compute_version_major,
			gpu_info->compute_version_minor);

	CUDA_TRY(cuCtxCreate(&d->gpu_context, 
			CU_CTX_BLOCKING_SYNC,
			gpu_info->device_handle))

	load_sort_engine(obj, d);

	/* load GPU kernels */

	CUDA_TRY(cuModuleLoad(&d->gpu_module, "stage1_core.ptx"))

	d->launch = (gpu_launch_t *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(gpu_launch_t));

	for (i = 0; i < NUM_GPU_FUNCTIONS; i++) {
		gpu_launch_t *launch = d->launch + i;

		gpu_launch_init(d->gpu_module, gpu_kernel_names[i],
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

	/* set up timing */

	CUDA_TRY(cuEventCreate(&d->start, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&d->end, CU_EVENT_BLOCKING_SYNC))

	/* set up found array */

	d->found_array_size = FOUND_ARRAY_SIZE;
	CUDA_TRY(cuMemAlloc(&d->gpu_found_array, sizeof(found_t) *
			d->found_array_size))
	d->found_array = (found_t *)xmalloc(sizeof(found_t) *
			d->found_array_size);

	CUDA_TRY(cuMemsetD8(d->gpu_found_array, 0, sizeof(found_t)))

	/* set up root generation arrays */

	d->p_array = p_soa_array_init(poly->degree);

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

	d->max_sort_entries32 = sort_entries32 = MIN(30000000,
			(uint32)(0.3 * d->gpu_info->global_mem_size /
				 (2 * (sizeof(uint32) + 
				       sizeof(uint32)))));

	d->max_sort_entries64 = sort_entries64 = MIN(20000000,
			(uint32)(0.3 * d->gpu_info->global_mem_size /
				 (2 * (sizeof(uint32) + 
				       sizeof(uint64)))));

	i = sizeof(uint32) * MAX(sort_entries32, sort_entries64);
	j = MAX(sort_entries32 * (sizeof(uint32) + sizeof(uint32)),
	        sort_entries64 * (sizeof(uint32) + sizeof(uint64)));

	CUDA_TRY(cuMemAlloc(&d->gpu_p_array, i))
	CUDA_TRY(cuMemAlloc(&d->gpu_p_array_scratch, i))
	CUDA_TRY(cuMemAlloc(&d->gpu_root_array, j))
	CUDA_TRY(cuMemAlloc(&d->gpu_root_array_scratch, j))
	return d;
}

/*------------------------------------------------------------------------*/
void gpu_data_free(void *gpu_data)
{
	device_data_t *d = (device_data_t *)gpu_data;

	CUDA_TRY(cuMemFree(d->gpu_p_array))
	CUDA_TRY(cuMemFree(d->gpu_p_array_scratch))
	CUDA_TRY(cuMemFree(d->gpu_root_array))
	CUDA_TRY(cuMemFree(d->gpu_root_array_scratch))

	free(d->found_array);
	free(d->launch);

	p_soa_array_free(d->p_array);

	CUDA_TRY(cuMemFree(d->gpu_found_array))

	CUDA_TRY(cuEventDestroy(d->start))
	CUDA_TRY(cuEventDestroy(d->end))

	d->sort_engine_free(d->sort_engine);
	unload_dynamic_lib(d->sort_engine_handle);

	CUDA_TRY(cuCtxDestroy(d->gpu_context)) 
	
	free(d->gpu_info);
	free(d);
}

