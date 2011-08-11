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

/* main driver for stage 1 */

/*------------------------------------------------------------------------*/
static void
poly_search_init(poly_search_t *poly, poly_stage1_t *data)
{
	mpz_init_set(poly->N, data->gmp_N);
	mpz_init(poly->high_coeff);
	mpz_init(poly->trans_N);
	mpz_init(poly->trans_m0);
	mpz_init(poly->mp_sieve_size);
	mpz_init(poly->m);
	mpz_init(poly->p);
	mpz_init(poly->tmp1);
	mpz_init(poly->tmp2);
	mpz_init(poly->tmp3);
	mpz_init(poly->tmp4);
	mpz_init(poly->tmp5);

	mpz_init_set(poly->gmp_high_coeff_begin, 
			data->gmp_high_coeff_begin);
	mpz_init_set(poly->gmp_high_coeff_end, 
			data->gmp_high_coeff_end);

	poly->degree = data->degree;
	poly->norm_max = data->norm_max;
	poly->callback = data->callback;
	poly->callback_data = data->callback_data;

#ifdef HAVE_CUDA
	CUDA_TRY(cuCtxCreate(&poly->gpu_context, 
			CU_CTX_BLOCKING_SYNC,
			poly->gpu_info->device_handle))

	/* load two GPU kernels, one for special-q
	   collision search and one for ordinary collision 
	   search */

	CUDA_TRY(cuModuleLoad(&poly->gpu_module_sq, 
				"stage1_core_gpu_sq.ptx"))
	CUDA_TRY(cuModuleLoad(&poly->gpu_module_nosq, 
				"stage1_core_gpu_nosq.ptx"))
	CUDA_TRY(cuModuleLoad(&poly->gpu_module_sort, 
				"stage1_core_gpu_sort.ptx"))
#endif
}

/*------------------------------------------------------------------------*/
static void
poly_search_free(poly_search_t *poly)
{
	mpz_clear(poly->N);
	mpz_clear(poly->high_coeff);
	mpz_clear(poly->trans_N);
	mpz_clear(poly->trans_m0);
	mpz_clear(poly->mp_sieve_size);
	mpz_clear(poly->m);
	mpz_clear(poly->p);
	mpz_clear(poly->tmp1);
	mpz_clear(poly->tmp2);
	mpz_clear(poly->tmp3);
	mpz_clear(poly->tmp4);
	mpz_clear(poly->tmp5);
	mpz_clear(poly->gmp_high_coeff_begin);
	mpz_clear(poly->gmp_high_coeff_end);
#ifdef HAVE_CUDA
	CUDA_TRY(cuCtxDestroy(poly->gpu_context)) 
#endif
}

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 p, r;
	uint8 log_val;
} sieve_prime_t;

#define SIEVE_ARRAY_SIZE 8192

typedef struct {
	uint8 *sieve_array;
	sieve_prime_t *primes;
	uint32 num_primes;
	uint32 num_primes_alloc;
	uint32 curr_offset;
} sieve_t;

/*------------------------------------------------------------------------*/
static void
sieve_ad_block(sieve_t *sieve, poly_search_t *poly)
{
	uint32 i;
	uint32 log_target;
	double target;

	target = mpz_get_d(poly->gmp_high_coeff_begin) /
				HIGH_COEFF_MULTIPLIER;

	if (HIGH_COEFF_SIEVE_LIMIT < target)
		target = HIGH_COEFF_SIEVE_LIMIT;

	log_target = floor(log(target) / M_LN2 + 0.5);
	memset(sieve->sieve_array, (int)(log_target - 4),
			SIEVE_ARRAY_SIZE);

	for (i = 0; i < sieve->num_primes; i++) {
		uint32 p = sieve->primes[i].p;
		uint32 r = sieve->primes[i].r;
		uint8 log_val = sieve->primes[i].log_val;

		while (r < SIEVE_ARRAY_SIZE) {
			sieve->sieve_array[r] -= log_val;
			r += p;
		}
		sieve->primes[i].r = r - SIEVE_ARRAY_SIZE;
	}
}

/*------------------------------------------------------------------------*/
static int
find_next_ad(sieve_t *sieve, poly_search_t *poly)
{
	uint32 i, j, p, k;
	uint8 *sieve_array = sieve->sieve_array;

	while (1) {

		for (i = sieve->curr_offset; i < SIEVE_ARRAY_SIZE; i++) {

			if (!(sieve_array[i] & 0x80))
				continue;

			mpz_divexact_ui(poly->tmp1, poly->gmp_high_coeff_begin,
					(mp_limb_t)HIGH_COEFF_MULTIPLIER);
			mpz_add_ui(poly->tmp1, poly->tmp1, (mp_limb_t)i);
			mpz_mul_ui(poly->high_coeff, poly->tmp1,
					(mp_limb_t)HIGH_COEFF_MULTIPLIER);

			if (mpz_cmp(poly->high_coeff,
						poly->gmp_high_coeff_end) > 0)
				break;

			/* trial divide the a_d and skip it if it
			   does not have enough small factors */

			mpz_cdiv_q_ui(poly->tmp2, poly->tmp1,
				(mp_limb_t)HIGH_COEFF_SIEVE_LIMIT);
			for (j = p = 0; j < PRECOMPUTED_NUM_PRIMES; j++) {
				p += prime_delta[j];

				if (p > HIGH_COEFF_PRIME_LIMIT)
					break;

				for (k = 0; k < HIGH_COEFF_POWER_LIMIT; k++) {
					if (mpz_divisible_ui_p(poly->tmp1, 
							(mp_limb_t)p))
						mpz_divexact_ui(poly->tmp1, 
							poly->tmp1,
							(mp_limb_t)p);
					else
						break;
				}
			}
			if (mpz_cmp(poly->tmp1, poly->tmp2) > 0)
				continue;

			/* a_d is okay, search it */

			sieve->curr_offset = i + 1;
			return 0;
		}

		/* update lower bound for next sieve block */

		mpz_set_ui(poly->tmp1, (mp_limb_t)SIEVE_ARRAY_SIZE);
		mpz_mul_ui(poly->tmp1, poly->tmp1,
				(mp_limb_t)HIGH_COEFF_MULTIPLIER);
		mpz_add(poly->gmp_high_coeff_begin,
				poly->gmp_high_coeff_begin, poly->tmp1);

		if (mpz_cmp(poly->gmp_high_coeff_begin,
					poly->gmp_high_coeff_end) > 0)
			break;

		sieve->curr_offset = 0;
		sieve_ad_block(sieve, poly);
	}

	return 1;
}

/*------------------------------------------------------------------------*/
static void
init_ad_sieve(sieve_t *sieve, poly_search_t *poly)
{
	uint32 i, j, p;

	sieve->num_primes = 0;
	sieve->num_primes_alloc = 100;
	sieve->primes = (sieve_prime_t *)xmalloc(sizeof(sieve_prime_t) *
						sieve->num_primes_alloc);
	sieve->sieve_array = (uint8 *)xmalloc(sizeof(uint8) *
						SIEVE_ARRAY_SIZE);

	mpz_divexact_ui(poly->tmp1, poly->gmp_high_coeff_begin,
			(mp_limb_t)HIGH_COEFF_MULTIPLIER);
	for (i = p = 0; i < PRECOMPUTED_NUM_PRIMES; i++) {
		uint32 power;
		uint8 log_val;

		p += prime_delta[i];
		if (p > HIGH_COEFF_PRIME_LIMIT)
			break;

		log_val = floor(log(p) / M_LN2 + 0.5);
		power = p;
		for (j = 0; j < HIGH_COEFF_POWER_LIMIT; j++) {
			uint32 r = mpz_cdiv_ui(poly->tmp1, (mp_limb_t)power);

			if (sieve->num_primes >= sieve->num_primes_alloc) {
				sieve->num_primes_alloc *= 2;
				sieve->primes = (sieve_prime_t *)xrealloc(
					sieve->primes,
					sieve->num_primes_alloc *
						sizeof(sieve_prime_t));
			}

			sieve->primes[sieve->num_primes].p = power;
			sieve->primes[sieve->num_primes].r = r;
			sieve->primes[sieve->num_primes].log_val = log_val;
			sieve->num_primes++;

			if ((uint32)(-1) / power < p)
				break;

			power *= p;
		}
	}

	sieve->curr_offset = 0;
	sieve_ad_block(sieve, poly);
}

/*------------------------------------------------------------------------*/
static void
search_coeffs(msieve_obj *obj, poly_search_t *poly, uint32 deadline)
{
	uint32 digits = mpz_sizeinbase(poly->N, 10);
	double deadline_per_coeff;
	double cumulative_time = 0;
	sieve_t ad_sieve;

	/* determine the CPU time limit; I have no idea if
	   the following is appropriate */

	if (digits <= 100)
		deadline_per_coeff = 5;
	else if (digits <= 105)
		deadline_per_coeff = 20;
	else if (digits <= 110)
		deadline_per_coeff = 30;
	else if (digits <= 120)
		deadline_per_coeff = 50;
	else if (digits <= 130)
		deadline_per_coeff = 100;
	else if (digits <= 140)
		deadline_per_coeff = 200;
	else if (digits <= 150)
		deadline_per_coeff = 400;
	else if (digits <= 175)
		deadline_per_coeff = 800;
	else if (digits <= 200)
		deadline_per_coeff = 1600;
	else
		deadline_per_coeff = 3200;

	printf("deadline: %.0lf CPU-seconds per coefficient\n",
					deadline_per_coeff);

	/* set up lower limit on a_d */

	mpz_sub_ui(poly->tmp1, poly->gmp_high_coeff_begin, (mp_limb_t)1);
	mpz_fdiv_q_ui(poly->tmp1, poly->tmp1, 
			(mp_limb_t)HIGH_COEFF_MULTIPLIER);
	mpz_add_ui(poly->tmp1, poly->tmp1, (mp_limb_t)1);
	mpz_mul_ui(poly->gmp_high_coeff_begin, poly->tmp1, 
			(mp_limb_t)HIGH_COEFF_MULTIPLIER);

	init_ad_sieve(&ad_sieve, poly);

	while (1) {
		double elapsed;

		/* we only use a_d which are composed of
		   many small prime factors, in order to
		   have lots of projective roots going
		   into stage 2 */

		if (find_next_ad(&ad_sieve, poly))
			break;

		/* sieve for polynomials using
		   Kleinjung's improved algorithm */

		elapsed = sieve_lattice(obj, poly, deadline_per_coeff);
		cumulative_time += elapsed;

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
			break;

		if (deadline && cumulative_time > deadline)
			break;
	}
}

/*------------------------------------------------------------------------*/
void
poly_stage1_init(poly_stage1_t *data,
		 stage1_callback_t callback, void *callback_data)
{
	memset(data, 0, sizeof(poly_stage1_t));
	mpz_init_set_ui(data->gmp_N, (mp_limb_t)0);
	mpz_init_set_ui(data->gmp_high_coeff_begin, (mp_limb_t)0);
	mpz_init_set_ui(data->gmp_high_coeff_end, (mp_limb_t)0);
	data->callback = callback;
	data->callback_data = callback_data;
}

/*------------------------------------------------------------------------*/
void
poly_stage1_free(poly_stage1_t *data)
{
	mpz_clear(data->gmp_N);
	mpz_clear(data->gmp_high_coeff_begin);
	mpz_clear(data->gmp_high_coeff_end);
}

/*------------------------------------------------------------------------*/
void
poly_stage1_run(msieve_obj *obj, poly_stage1_t *data)
{
	/* pass external configuration in and run the search */

	poly_search_t poly;
#ifdef HAVE_CUDA
	gpu_config_t gpu_config;

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
	logprintf(obj, "using GPU %u (%s)\n", obj->which_gpu,
			gpu_config.info[obj->which_gpu].name);

	poly.gpu_info = gpu_config.info + obj->which_gpu; 
#endif

	poly_search_init(&poly, data);

	search_coeffs(obj, &poly, data->deadline);

	poly_search_free(&poly);
}
