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
#include <stage1_core/cpu_intrinsics.h>
#include <stage1_core_gpu/stage1_core_deg5_64.h>

typedef struct {
	double bits; /* in leading rational coeff */
	double p_scale;
	uint32 num_pieces; /* for randomization */
	uint32 specialq_max;
	uint32 large_fb_max;
} sieve_fb_param_t;

static const sieve_fb_param_t sieve_fb_params[] = {

	{ 40.0, 2.00, 1,   100,   2000},
	{ 48.0, 2.00, 1,   200,   2000},
	{ 56.0, 2.00, 1,  1000,   5000},
	{ 64.0, 1.50, 1,  3500,   5000},
	{ 72.0, 1.50, 1,  7500,   5000},
	{ 80.0, 1.25, 1, 15000,  10000},
	{ 88.0, 1.25, 1, 15000,  25000},
	{ 96.0, 1.25, 1, 25000,  50000},
	{116.0, 1.10, 1, 25000, 100000},
	{128.0, 1.10, 1, 50000, 100000},
};

#define NUM_SIEVE_FB_PARAMS (sizeof(sieve_fb_params) / \
				sizeof(sieve_fb_params[0]))

/*------------------------------------------------------------------------*/
static void
handle_specialq_collision(poly_search_t *poly, uint32 which_poly,
		uint32 p1, uint32 p2, uint32 special_q,
		uint64 special_q_root, uint64 res)
{
	curr_poly_t *c = poly->batch + which_poly;

	if (mp_gcd_1(p1, p2) != 1)
		return;

	mpz_set_ui(poly->p, (unsigned long)p1);
	mpz_mul_ui(poly->p, poly->p, (unsigned long)p2);
	mpz_mul_ui(poly->p, poly->p, (unsigned long)special_q);

	mpz_gcd(poly->tmp3, poly->p, c->high_coeff);
	if (mpz_cmp_ui(poly->tmp3, 1))
		return;

	uint64_2gmp(special_q_root, poly->tmp1);
	uint64_2gmp(res, poly->tmp2);
	mpz_set_ui(poly->tmp3, (unsigned long)special_q);

	mpz_mul(poly->tmp3, poly->tmp3, poly->tmp3);
	mpz_addmul(poly->tmp1, poly->tmp2, poly->tmp3);
	mpz_sub(poly->tmp1, poly->tmp1, c->mp_sieve_size);
	mpz_add(poly->m0, c->trans_m0, poly->tmp1);

	/* check */
	mpz_pow_ui(poly->tmp1, poly->m0, (mp_limb_t)poly->degree);
	mpz_mul(poly->tmp2, poly->p, poly->p);
	mpz_sub(poly->tmp1, c->trans_N, poly->tmp1);
	mpz_tdiv_r(poly->tmp3, poly->tmp1, poly->tmp2);
	if (mpz_cmp_ui(poly->tmp3, (mp_limb_t)0)) {
		gmp_printf("poly %u %u %u %u %Zd\n", 
				which_poly, special_q, p1, p2, poly->m0);
		printf("crap\n");
		return;
	}

	mpz_mul_ui(poly->tmp1, c->high_coeff, (mp_limb_t)poly->degree);
	mpz_tdiv_qr(poly->m0, poly->tmp2, poly->m0, poly->tmp1);
	mpz_invert(poly->tmp3, poly->tmp1, poly->p);

	mpz_sub(poly->tmp4, poly->tmp3, poly->p);
	if (mpz_cmpabs(poly->tmp3, poly->tmp4) < 0)
		mpz_set(poly->tmp4, poly->tmp3);

	mpz_sub(poly->tmp5, poly->tmp2, poly->tmp1);
	if (mpz_cmpabs(poly->tmp2, poly->tmp5) > 0)
		mpz_add_ui(poly->m0, poly->m0, (mp_limb_t)1);
	else
		mpz_set(poly->tmp5, poly->tmp2);

	mpz_addmul(poly->m0, poly->tmp4, poly->tmp5);

	gmp_printf("poly %2u %u %u %u %Zd\n", 
			which_poly, special_q, p1, p2, poly->m0);

	poly->callback(c->high_coeff, poly->p, poly->m0, 
			c->coeff_max, poly->callback_data);
}

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 last_p;

	uint32 *p;
	uint32 *lattice_size;
	uint64 *roots[POLY_BATCH_SIZE];
	uint64 *start_roots[POLY_BATCH_SIZE];
} p_soa_var_t;

static void
p_soa_var_init(p_soa_var_t *soa, uint32 batch_size)
{
	uint32 i;

	memset(soa, 0, sizeof(soa));
	soa->p = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	soa->lattice_size = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	for (i = 0; i < POLY_BATCH_SIZE; i++) {
		soa->roots[i] = (uint64 *)xmalloc(batch_size * 
						sizeof(uint64));
		soa->start_roots[i] = (uint64 *)xmalloc(batch_size * 
						sizeof(uint64));
	}
}

static void
p_soa_var_free(p_soa_var_t *soa)
{
	uint32 i;

	free(soa->p);
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
	soa->last_p = 0;
}

static void 
store_p_soa(uint64 p, uint32 num_roots, uint32 which_poly, 
		mpz_t *roots, void *extra)
{
	lattice_fb_t *L = (lattice_fb_t *)extra;
	p_soa_var_t *soa;
	uint32 num;

	if (num_roots != 1) {
		printf("error: num_roots > 1\n");
		exit(-1);
	}

	if (L->fill_p)
		soa = (p_soa_var_t *)L->p_array;
	else
		soa = (p_soa_var_t *)L->q_array;

	num = soa->num_p;
	if (p != soa->last_p) {
		soa->p[num] = (uint32)p;
		soa->num_p++;
		soa->last_p = (uint32)p;
		soa->start_roots[which_poly][num] = gmp2uint64(roots[0]);
	}
	else {
		soa->start_roots[which_poly][num - 1] = gmp2uint64(roots[0]);
	}
}

/*------------------------------------------------------------------------*/
static uint32
sieve_lattice_batch(msieve_obj *obj, lattice_fb_t *L,
			uint32 threads_per_block,
			gpu_info_t *gpu_info, CUfunction gpu_kernel,
			p_soa_var_t *specialq_array,
			uint32 which_specialq)
{
	uint32 i, j;
	p_soa_var_t * p_array = (p_soa_var_t *)L->p_array;
	p_soa_var_t * q_array = (p_soa_var_t *)L->q_array;
	uint32 num_blocks;
	uint32 num_p_offset;
	uint32 num_q_offset;
	void *gpu_ptr;
	uint32 num_poly = L->poly->num_poly;

	p_soa_t *p_marshall = (p_soa_t *)L->p_marshall;
	q_soa_t *q_marshall = (q_soa_t *)L->q_marshall;
	found_t *found_array = (found_t *)L->found_array;
	uint32 found_array_size = L->found_array_size;
	uint32 num_q_done = 0;

	uint32 specialq = specialq_array->p[which_specialq];
	uint64 specialq2 = (uint64)specialq * specialq;

	for (i = 0; i < p_array->num_p; i++) {
		uint32 p = p_array->p[i];
		uint64 p2 = (uint64)p * p;
		uint32 p2_w = montmul32_w((uint32)p2);
		uint64 p2_r = montmul64_r(p2);
		uint64 inv = mp_modinv_2(specialq2, p2);

		inv = montmul64(inv, p2_r, p2, p2_w);
		for (j = 0; j < num_poly; j++) {
			uint64 proot = p_array->start_roots[j][i];
			uint64 sqroot = specialq_array->start_roots[j][which_specialq];
			uint64 res = montmul64(modsub64(proot, 
						sqroot, p2),
						inv, p2, p2_w);
			p_array->roots[j][i] = res;
		}

		p_array->lattice_size[i] = 2 * L->poly->batch[
					num_poly/2].sieve_size /
					((double)specialq2 * p2);
	}

	for (i = 0; i < q_array->num_p; i++) {
		uint32 p = q_array->p[i];
		uint64 p2 = (uint64)p * p;
		uint32 p2_w = montmul32_w((uint32)p2);
		uint64 p2_r = montmul64_r(p2);
		uint64 inv = mp_modinv_2(specialq2, p2);

		inv = montmul64(inv, p2_r, p2, p2_w);
		for (j = 0; j < num_poly; j++) {
			uint64 proot = q_array->start_roots[j][i];
			uint64 sqroot = specialq_array->start_roots[j][which_specialq];
			uint64 res = montmul64(modsub64(proot, 
						sqroot, p2),
						inv, p2, p2_w);
			q_array->roots[j][i] = res;
		}
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

	while (num_q_done < q_array->num_p) {

		uint32 num_p_done = 0;
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
#if 1
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
					handle_specialq_collision(L->poly, 
							f->which_poly,
							f->p, f->q,
							specialq,
							specialq_array->start_roots[f->which_poly][which_specialq],
							f->proot + f->offset *
								f->p * f->p);
				}
			}

			num_p_done += curr_num_p;
		}

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
			return 1;

		num_q_done += curr_num_q;
	}

	return 0;
}

/*------------------------------------------------------------------------*/
static void
sieve_specialq_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_specialq, 
		sieve_fb_t *sieve_large_p1, 
		sieve_fb_t *sieve_large_p2, 
		uint32 specialq_min, uint32 specialq_max, 
		uint32 large_p1_min, uint32 large_p1_max,
		uint32 large_p2_min, uint32 large_p2_max)
{
	uint32 i;
	uint32 min_large_p1, min_large_p2;
	p_soa_var_t * p_array;
	p_soa_var_t * q_array;
	p_soa_var_t * specialq_array;
	uint32 host_p_batch_size;
	uint32 host_q_batch_size;
	uint32 num_specialq;

	uint32 threads_per_block;
	gpu_info_t *gpu_info = L->gpu_info;
       	CUfunction gpu_kernel = L->gpu_kernel;

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

	/*------------------------------*/
	q_array = L->q_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	num_specialq = 1000;
	p_soa_var_init(q_array, num_specialq);
	sieve_fb_reset(sieve_specialq, (uint64)specialq_min, 
			(uint64)specialq_max, 1, 1);
	L->fill_p = 0;
	p_soa_var_reset(q_array);
	for (i = 0; i < num_specialq; i++) {
		if (sieve_fb_next(sieve_specialq, L->poly,
				store_p_soa, L) == P_SEARCH_DONE) {
			break;
		}
	}
	specialq_array = q_array;
	num_specialq = q_array->num_p;
	/*------------------------------*/

	L->p_marshall = (p_soa_t *)xmalloc(sizeof(p_soa_t));
	L->q_marshall = (q_soa_t *)xmalloc(sizeof(q_soa_t));
	p_array = L->p_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	q_array = L->q_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));

	host_p_batch_size = MAX(10000, L->found_array_size / 3);
	host_q_batch_size = MAX(50000, 12 * L->found_array_size);
	p_soa_var_init(p_array, host_p_batch_size);
	p_soa_var_init(q_array, host_q_batch_size);

	min_large_p1 = large_p1_min;
	sieve_fb_reset(sieve_large_p1, (uint64)large_p1_min, 
			(uint64)large_p1_max, 1, 1);

	while (min_large_p1 < large_p1_max) {

		L->fill_p = 1;
		p_soa_var_reset(p_array);
		for (i = 0; i < host_p_batch_size &&
				min_large_p1 != (uint32)P_SEARCH_DONE; i++) {
			min_large_p1 = sieve_fb_next(sieve_large_p1, L->poly,
						store_p_soa, L);
		}
		if (p_array->num_p == 0)
			goto finished;

		min_large_p2 = large_p2_min;
		sieve_fb_reset(sieve_large_p2, (uint64)large_p2_min, 
				(uint64)large_p2_max, 1, 1);

		while (min_large_p2 < large_p2_max) {

			L->fill_p = 0;
			p_soa_var_reset(q_array);
			for (i = 0; i < host_q_batch_size &&
					min_large_p2 != (uint32)P_SEARCH_DONE; i++) {
				min_large_p2 = sieve_fb_next(sieve_large_p2, 
							L->poly, store_p_soa, L);
			}
			if (q_array->num_p == 0)
				goto finished;

			for (i = 0; i < num_specialq; i++) {
				if (sieve_lattice_batch(obj, L, threads_per_block,
						gpu_info, gpu_kernel,
						specialq_array, i)) {
					goto finished;
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
	p_soa_var_free(specialq_array);
	free(p_array);
	free(q_array);
	free(specialq_array);
	free(L->p_marshall);
	free(L->q_marshall);
	free(L->found_array);
}

/*--------------------------------------------------------------------*/
static void 
get_poly_params(double bits, sieve_fb_param_t *params)
{
	uint32 i;
	const sieve_fb_param_t *low, *high;
	double j, k, dist;
	double max_bits;

	if (bits < sieve_fb_params[0].bits) {
		*params = sieve_fb_params[0];
		return;
	}

	max_bits = sieve_fb_params[NUM_SIEVE_FB_PARAMS - 1].bits;
	if (bits >= max_bits) {
		if (bits > max_bits + 5) {
			printf("error: no parameters for "
				"%.0lf bit inputs\n", bits + 0.5);
			exit(-1);
		}
		*params = sieve_fb_params[NUM_SIEVE_FB_PARAMS - 1];
		return;
	}

	for (i = 0; i < NUM_SIEVE_FB_PARAMS - 1; i++) {
		if (bits < sieve_fb_params[i+1].bits)
			break;
	}

	low = &sieve_fb_params[i];
	high = &sieve_fb_params[i+1];
	dist = high->bits - low->bits;
	j = bits - low->bits;
	k = high->bits - bits;

	params->bits = bits;
	params->p_scale = (low->p_scale * k + 
			   high->p_scale * j) / dist;
	params->num_pieces = (low->num_pieces * k + 
			      high->num_pieces * j) / dist;
	params->specialq_max = exp((log(low->specialq_max) * k +
			           log(high->specialq_max) * j) / dist);
	params->large_fb_max = exp((log(low->large_fb_max) * k +
			           log(high->large_fb_max) * j) / dist);
}

/*------------------------------------------------------------------------*/
void
sieve_lattice_specialq(msieve_obj *obj, poly_search_t *poly, uint32 deadline)
{
	lattice_fb_t L;
	sieve_fb_t sieve_specialq; 
	sieve_fb_t sieve_large_p1; 
	sieve_fb_t sieve_large_p2; 
	uint32 large_fb_max;
	uint32 specialq_min, specialq_max;
	uint64 large_p1_min, large_p1_max;
	uint64 large_p2_min, large_p2_max;
	double p_scale;
	uint32 num_pieces;
	sieve_fb_param_t params;

	double p_size_max = poly->batch[0].p_size_max;
	double sieve_size = poly->batch[0].sieve_size;
	double bits = log(sieve_size) / M_LN2;

	printf("p = %.2lf sieve = %.2lf bits\n", 
			log(p_size_max) / M_LN2, bits);

	get_poly_params(bits, &params);

	p_scale = params.p_scale;
	specialq_max = params.specialq_max;
	large_fb_max = params.large_fb_max;
	num_pieces = params.num_pieces;

	specialq_min = MAX(7, specialq_max / p_scale);
	large_p2_min = sqrt(p_size_max / specialq_min);
	large_p2_max = p_scale * large_p2_min;
	large_p1_min = large_p2_min / p_scale;
	large_p1_max = large_p2_min - 1;

	gmp_printf("coeff %Zd-%Zd q %u-%u "
		       "p1 %" PRIu64 "-%" PRIu64 " "
		       "p2 %" PRIu64 "-%" PRIu64 "\n",
			poly->batch[0].high_coeff,
			poly->batch[poly->num_poly - 1].high_coeff,
			specialq_min, specialq_max,
			large_p1_min, large_p1_max,
			large_p2_min, large_p2_max);

	sieve_fb_init(&sieve_specialq, poly, 
			5, 100, 1, 1, 1);
	sieve_fb_init(&sieve_large_p1, poly, 
			101, large_fb_max,  
			1, 1, 0);
	sieve_fb_init(&sieve_large_p2, poly, 
			0, 0, 
			1, 1, 0);

	L.poly = poly;
	L.start_time = time(NULL);
	L.deadline = deadline;
	L.gpu_info = poly->gpu_info;

	if (large_p2_max < ((uint64)1 << 24))
		L.gpu_module = poly->gpu_module48;
	else if (large_p2_max < ((uint64)1 << 32))
		L.gpu_module = poly->gpu_module64;
	else {
		printf("error: p2 too large\n");
		exit (-1);
	}

	CUDA_TRY(cuModuleGetFunction(&L.gpu_kernel, 
			L.gpu_module, "sieve_kernel"))

	sieve_specialq_64(obj, &L,
			&sieve_specialq, 
			&sieve_large_p1,
			&sieve_large_p2,
			specialq_min, specialq_max,
			(uint32)large_p1_min, 
			(uint32)large_p1_max,
			(uint32)large_p2_min, 
			(uint32)large_p2_max);

	sieve_fb_free(&sieve_specialq);
	sieve_fb_free(&sieve_large_p1);
	sieve_fb_free(&sieve_large_p2);
}
