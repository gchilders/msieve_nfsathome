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
#include <cpu_intrinsics.h>
#include <stage1_core_gpu/stage1_core.h>

#define MAX_P ((uint32)(-1))

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 num_p_alloc;
	uint32 curr;

	uint32 *p;
	uint32 *mont_w;
	uint64 *mont_r;
	uint64 *p2;
	uint32 *lattice_size;
	uint64 *roots[POLY_BATCH_SIZE];
	uint64 *start_roots[POLY_BATCH_SIZE];
} p_soa_var_t;

static void
p_soa_var_init(p_soa_var_t *soa, uint32 batch_size)
{
	uint32 i;

	memset(soa, 0, sizeof(soa));
	soa->num_p_alloc = batch_size;
	soa->p = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	soa->mont_w = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	soa->mont_r = (uint64 *)xmalloc(batch_size * sizeof(uint64));
	soa->p2 = (uint64 *)xmalloc(batch_size * sizeof(uint64));
	soa->lattice_size = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	for (i = 0; i < POLY_BATCH_SIZE; i++) {
		soa->roots[i] = (uint64 *)xmalloc(batch_size * sizeof(uint64));
		soa->start_roots[i] = (uint64 *)xmalloc(batch_size * sizeof(uint64));
	}
}

static void
p_soa_var_free(p_soa_var_t *soa)
{
	uint32 i;

	free(soa->p);
	free(soa->mont_w);
	free(soa->mont_r);
	free(soa->p2);
	free(soa->lattice_size);
	for (i = 0; i < POLY_BATCH_SIZE; i++) {
		free(soa->roots[i]);
		free(soa->start_roots[i]);
	}
}

static void
p_soa_var_reset(p_soa_var_t *soa)
{
	soa->num_p = 0;
	soa->curr = 0;
	soa->p[0] = 0;
}

static void 
store_p_soa(uint32 p, uint32 num_roots, uint32 which_poly, 
		mpz_t *roots, void *extra)
{
	p_soa_var_t *soa = (p_soa_var_t *)extra;
	uint32 i, j, curr;
	uint64 p2 = (uint64)p * p;
	uint32 mont_w = montmul32_w((uint32)p2);
	uint64 mont_r = montmul64_r(p2);

	if (p != soa->p[soa->curr]) {
		soa->curr = soa->num_p;
		soa->num_p = MIN(soa->num_p + num_roots, soa->num_p_alloc);
	}
	curr = soa->curr;
	j = soa->num_p - curr;
	for (i = 0; i < j; i++, curr++) {
		soa->p[curr] = p;
		soa->mont_w[curr] = mont_w;
		soa->mont_r[curr] = mont_r;
		soa->p2[curr] = p2;
		soa->start_roots[which_poly][curr] = gmp2uint64(roots[i]);
	}
}

/*------------------------------------------------------------------------*/
static void
create_special_q_lattice(p_soa_var_t *p_array, uint32 num_roots,
			 uint64 special_q2, uint64 *special_q_roots)
{
	uint32 i, j;

	for (i = 0; i < p_array->num_p; i++) {
		uint64 p2 = p_array->p2[i];
		uint32 p2_w = p_array->mont_w[i];
		uint64 p2_r = p_array->mont_r[i];
		uint64 inv = mp_modinv_2(special_q2, p2);

		inv = montmul64(inv, p2_r, p2, p2_w);
		for (j = 0; j < num_roots; j++) {
			uint64 proot = p_array->start_roots[j][i];

			uint64 res = montmul64(mp_modsub_2(proot,
						special_q_roots[j] % p2, p2),
						inv, p2, p2_w);
			p_array->roots[j][i] = res;
		}
	}
}

/*------------------------------------------------------------------------*/
static uint32
handle_special_q(msieve_obj *obj, lattice_fb_t *L, uint32 threads_per_block,
			gpu_info_t *gpu_info, CUfunction gpu_kernel,
			uint32 special_q, uint64 *special_q_roots)
{
	uint32 i, j;
	p_soa_var_t * p_array = (p_soa_var_t *)L->p_array;
	p_soa_var_t * q_array = (p_soa_var_t *)L->q_array;
	uint32 num_blocks;
	uint32 num_p_offset;
	uint32 num_q_offset;
	void *gpu_ptr;
	uint32 num_poly = L->poly->num_poly;
	curr_poly_t *middle_poly = L->poly->batch + num_poly / 2;

	p_soa_t *p_marshall = (p_soa_t *)L->p_marshall;
	q_soa_t *q_marshall = (q_soa_t *)L->q_marshall;
	found_t *found_array = (found_t *)L->found_array;
	uint32 found_array_size = L->found_array_size;
	uint32 num_q_done = 0;

	i = 0;
	gpu_ptr = (void *)(size_t)L->gpu_p_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_p_offset = i;
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)L->gpu_q_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_q_offset = i;
	i += sizeof(uint32);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	CUDA_TRY(cuParamSeti(gpu_kernel, i, num_poly))
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)L->gpu_found_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_TRY(cuParamSetSize(gpu_kernel, i))

	if (special_q == 1) {
		for (i = 0; i < p_array->num_p; i++) {
			uint64 p2 = p_array->p2[i];

			p_array->lattice_size[i] = MIN((uint32)(-1),
					(2 * middle_poly->sieve_size / p2));
		}
		for (j = 0; j < num_poly; j++) {
			memcpy(p_array->roots[j], p_array->start_roots[j],
			       p_array->num_p * sizeof(uint64));
			memcpy(q_array->roots[j], q_array->start_roots[j],
			       q_array->num_p * sizeof(uint64));
		}
	}
	else {
		uint64 special_q2 = (uint64)special_q * special_q;

		create_special_q_lattice(p_array, num_poly,
						special_q2, special_q_roots);
		create_special_q_lattice(q_array, num_poly,
						special_q2, special_q_roots);
		for (i = 0; i < p_array->num_p; i++) {
			uint64 p2 = p_array->p2[i];

			p_array->lattice_size[i] = MIN((uint32)(-1),
					(2 * middle_poly->sieve_size / 
						((double)p2 * special_q2)));
		}
	}

	while (num_q_done < q_array->num_p) {

		uint32 num_p_done = 0;
		time_t curr_time;
		double elapsed;
		uint32 curr_num_q = MIN(3 * found_array_size,
					q_array->num_p - num_q_done);

		curr_num_q = MIN(curr_num_q, Q_SOA_BATCH_SIZE);

		/* force to be a multiple of the block size */
		curr_num_q -= (curr_num_q % threads_per_block);
		if (curr_num_q == 0)
			break;

		memcpy(q_marshall->p, 
			q_array->p + num_q_done,
			curr_num_q * sizeof(uint32));

		for (i = 0; i < num_poly; i++) {
			memcpy(q_marshall->roots[i],
				q_array->roots[i] + num_q_done,
				curr_num_q * sizeof(uint64));
		}

		CUDA_TRY(cuMemcpyHtoD(L->gpu_q_array, q_marshall,
				Q_SOA_BATCH_SIZE * (sizeof(uint32) +
					num_poly * sizeof(uint64))))
		CUDA_TRY(cuParamSeti(gpu_kernel, num_q_offset, curr_num_q))

		while (num_p_done < p_array->num_p) {

			uint32 curr_num_p = MIN(found_array_size / 3,
						p_array->num_p - num_p_done);

			curr_num_p = MIN(curr_num_p, P_SOA_BATCH_SIZE);
			memcpy(p_marshall->p, 
				p_array->p + num_p_done,
				curr_num_p * sizeof(uint32));
			memcpy(p_marshall->lattice_size, 
				p_array->lattice_size + num_p_done,
				curr_num_p * sizeof(uint32));

			for (i = 0; i < num_poly; i++) {
				memcpy(p_marshall->roots[i],
					p_array->roots[i] + num_p_done,
					curr_num_p * sizeof(uint64));
			}

			CUDA_TRY(cuMemcpyHtoD(L->gpu_p_array, p_marshall,
				P_SOA_BATCH_SIZE * (2 * sizeof(uint32) +
					num_poly * sizeof(uint64))))

			CUDA_TRY(cuParamSeti(gpu_kernel, num_p_offset, 
						curr_num_p))
#if 0
			printf("qnum %u pnum %u\n", curr_num_q, curr_num_p);
#endif

			num_blocks = gpu_info->num_compute_units;
			if (curr_num_q < found_array_size)
				num_blocks = curr_num_q / threads_per_block;

			CUDA_TRY(cuLaunchGrid(gpu_kernel, 
						num_blocks, 1))

			CUDA_TRY(cuMemcpyDtoH(found_array, 
						L->gpu_found_array, 
						num_blocks * 
						threads_per_block *
							sizeof(found_t)))

			for (i = 0; i < threads_per_block *
					num_blocks; i++) {
				found_t *f = found_array + i;

				if (f->p > 0) {
					uint128 proot, res;
					uint64 p2 = (uint64)f->p * f->p;

					proot.w[0] = (uint32)f->proot;
					proot.w[1] = (uint32)(f->proot >> 32);
					proot.w[2] = 0;
					proot.w[3] = 0;

					res =
					    add128(proot, mul64(f->offset, p2));

					handle_collision(L->poly, 
						f->which_poly,
						f->p, f->q,
						special_q,
						special_q_roots[f->which_poly],
						res);
				}
			}

			num_p_done += curr_num_p;
		}

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
			return 1;

		curr_time = time(NULL);
		elapsed = curr_time - L->start_time;
		if (elapsed > L->deadline)
			return 1;

		num_q_done += curr_num_q;
	}

	return 0;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_specialq_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max,
		sieve_fb_t *sieve_small_p,
		uint32 small_p_min, uint32 small_p_max,
		sieve_fb_t *sieve_large_p,
		uint32 large_p_min, uint32 large_p_max) 
{
	uint32 quit = 0;
	p_soa_var_t * p_array;
	p_soa_var_t * q_array;
	p_soa_var_t * special_q_array;
	uint32 num_poly = L->poly->num_poly;
	uint32 degree = L->poly->degree;
	uint32 max_roots = (degree != 5) ? degree : 1;
	uint32 host_p_batch_size;
	uint32 host_q_batch_size;
	uint32 special_q_batch_size = 1000;

	uint32 threads_per_block;
	gpu_info_t *gpu_info = L->poly->gpu_info;
	CUmodule gpu_module;
       	CUfunction gpu_kernel;

	if (large_p_max < ((uint32)1 << 24))
		gpu_module = L->poly->gpu_module48;
	else
		gpu_module = L->poly->gpu_module64;

	CUDA_TRY(cuModuleGetFunction(&gpu_kernel, 
			gpu_module, "sieve_kernel"))

	L->p_marshall = (p_soa_t *)xmalloc(sizeof(p_soa_t));
	L->q_marshall = (q_soa_t *)xmalloc(sizeof(q_soa_t));
	p_array = L->p_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	q_array = L->q_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	special_q_array = L->special_q_array = (p_soa_var_t *)xmalloc(
							sizeof(p_soa_var_t));

	CUDA_TRY(cuMemAlloc(&L->gpu_p_array, sizeof(p_soa_t)))
	CUDA_TRY(cuMemAlloc(&L->gpu_q_array, sizeof(q_soa_t)))

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

	host_p_batch_size = MAX(10000, L->found_array_size / 3);
	host_q_batch_size = MAX(50000, 12 * L->found_array_size);
	p_soa_var_init(p_array, host_p_batch_size);
	p_soa_var_init(q_array, host_q_batch_size);
	p_soa_var_init(special_q_array, special_q_batch_size);

	if (2 * L->poly->batch[num_poly / 2].sieve_size /
				((double)small_p_min * small_p_min
				 * special_q_min * special_q_min)
							> (uint32)(-1))
		logprintf(obj, "warning: sieve_size too large\n");

	sieve_fb_reset(sieve_large_p, large_p_min, large_p_max, 1, max_roots);
	while (1) {
		p_soa_var_reset(q_array);
		while (sieve_fb_next(sieve_large_p, L->poly, store_p_soa,
						q_array) != P_SEARCH_DONE)
			if (q_array->num_p == host_q_batch_size)
				break;

		if (q_array->num_p == 0)
			break;

		sieve_fb_reset(sieve_small_p, small_p_min, small_p_max,
						1, max_roots);
		while (1) {
			uint64 sqroots[POLY_BATCH_SIZE];
			uint32 i, j;

			p_soa_var_reset(p_array);
			while (sieve_fb_next(sieve_small_p, L->poly, 
						store_p_soa,
						p_array) != P_SEARCH_DONE)
				if (p_array->num_p == host_p_batch_size)
					break;

			if (p_array->num_p == 0)
				break;

			if (special_q_min <= 1) { /* handle trivial lattice */
				memset(sqroots, 0, sizeof(sqroots));

				if (handle_special_q(obj, L,
							threads_per_block,
							gpu_info, gpu_kernel,
							1, sqroots)) {
					quit = 1;
					goto finished;
				}

				if (special_q_max <= 1)
					continue;
			}

			sieve_fb_reset(sieve_special_q, special_q_min,
						special_q_max, 1, max_roots);
			while (1) {
				p_soa_var_reset(special_q_array);
				while (sieve_fb_next(sieve_special_q,
							L->poly, store_p_soa,
							special_q_array) !=
								P_SEARCH_DONE)
					if (special_q_array->num_p ==
							special_q_batch_size)
						break;

				if (special_q_array->num_p == 0)
					break;

				for (i = 0; i < special_q_array->num_p; i++) {
					for (j = 0; j < num_poly; j++)
						sqroots[j] =
						    special_q_array->start_roots[j][i];

					if (handle_special_q(obj, L,
							threads_per_block,
							gpu_info, gpu_kernel,
							special_q_array->p[i],
							sqroots)) {
						quit = 1;
						goto finished;
					}
				}
			}
		}
	}

finished:
	CUDA_TRY(cuMemFree(L->gpu_p_array))
	CUDA_TRY(cuMemFree(L->gpu_q_array))
	CUDA_TRY(cuMemFree(L->gpu_found_array))
	p_soa_var_free(p_array);
	p_soa_var_free(q_array);
	p_soa_var_free(special_q_array);
	free(p_array);
	free(q_array);
	free(special_q_array);
	free(L->p_marshall);
	free(L->q_marshall);
	free(L->found_array);
	return quit;
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_gpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_param_t *params,
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max)
{
	uint32 quit;
	uint32 large_p_min, large_p_max;
	uint32 small_p_min, small_p_max;
	sieve_fb_t sieve_large_p, sieve_small_p;
	curr_poly_t *middle_poly = L->poly->batch + L->poly->num_poly / 2;
	curr_poly_t *last_poly = L->poly->batch + L->poly->num_poly - 1;
	uint32 degree = L->poly->degree;
	uint32 max_roots = (degree != 5) ? degree : 1;

	sieve_fb_init(&sieve_large_p, L->poly,
			0, 0, /* prime large_p */
			1, max_roots,
			0);

	sieve_fb_init(&sieve_small_p, L->poly,
			0, 0, /* prime small_p */
			1, max_roots,
			0);

	large_p_min = sqrt(middle_poly->p_size_max / special_q_max);
	large_p_max = MIN(MAX_P, large_p_min * params->p_scale);

	small_p_max = large_p_min - 1;
	small_p_min = small_p_max / params->p_scale;

	while (1) {
		gmp_printf("coeff %Zd-%Zd specialq %u - %u "
			   "p1 %u - %u p2 %u - %u\n",
				L->poly->batch[0].high_coeff,
				last_poly->high_coeff,
				special_q_min, special_q_max,
				small_p_min, small_p_max,
				large_p_min, large_p_max);

		quit = sieve_specialq_64(obj, L,
				sieve_special_q,
				special_q_min, special_q_max,
				&sieve_small_p,
				small_p_min, small_p_max,
				&sieve_large_p,
				large_p_min, large_p_max);

		if (quit || large_p_max == MAX_P ||
		    large_p_max / small_p_min > params->max_diverge)
			break;

		large_p_min = large_p_max + 1;
		large_p_max = MIN(MAX_P, large_p_min * params->p_scale);

		small_p_max = small_p_min - 1;
		small_p_min = small_p_max / params->p_scale;
	}

	sieve_fb_free(&sieve_large_p);
	sieve_fb_free(&sieve_small_p);
	return quit;
}
