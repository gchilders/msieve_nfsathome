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

/* structures for storing arithmetic progressions. Rational
   leading coeffs of NFS polynomials are assumed to be the 
   product of three groups of factors p, each of size at most
   32 bits, and candidates must satisfy a condition modulo p^2 */

#define MAX_P_BITS 32
#define MAX_P ((uint64)1 << MAX_P_BITS - 1)

typedef struct {
	uint32 bits; /* in leading rational coeff */
	double p_scale;
	uint32 num_pieces; /* for randomization */
	uint32 special_q_min;
	uint32 special_q_max;
} sieve_fb_param_t;

static const sieve_fb_param_t sieve_fb_params[] = {

	{ 40, 1.3,  1,       7,     2500},
	{ 48, 1.2,  1,       7,     7500},
	{ 56, 1.1,  5,     350,    35000},
	{ 64, 1.1, 10,   75000,   250000},
	{ 72, 1.1, 25,  750000,  2500000},
	{ 80, 1.1, 50, 5000000, 25000000},
};

#define NUM_SIEVE_FB_PARAMS (sizeof(sieve_fb_params) / \
				sizeof(sieve_fb_params[0]))

/*------------------------------------------------------------------------*/
void
handle_collision(poly_search_t *poly, uint32 which_poly,
		uint32 p1, uint32 p2, uint32 special_q,
		uint64 special_q_root, uint64 res)
{
	curr_poly_t *c = poly->batch + which_poly;

	/* p1, p2, and special_q should always be pairwise coprime
         * when we get here, but let's be defensive and check anyway. */
	if (mp_gcd_1(special_q, p1) != 1)
		return;
	if (mp_gcd_1(special_q, p2) != 1)
		return;
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
static void
get_poly_params(double bits, sieve_fb_param_t *params)
{
	uint32 i;
	const sieve_fb_param_t *low, *high;
	double j, k, dist, max_bits;

	if (bits < sieve_fb_params[0].bits) {
		*params = sieve_fb_params[0];

		return;
	}

	max_bits = sieve_fb_params[NUM_SIEVE_FB_PARAMS - 1].bits;
	if (bits >= max_bits) {
		if (bits > max_bits + 5) {
			printf("error: no factor base parameters for "
				"%.0lf bit leading rational "
				"coefficient\n", bits + 0.5);
			exit (-1);
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

	params->special_q_min = exp((log(low->special_q_min) * k +
					log(high->special_q_min) * j) / dist);
	params->special_q_max = exp((log(low->special_q_max) * k +
					log(high->special_q_max) * j) / dist);
}

/*------------------------------------------------------------------------*/
void
sieve_lattice(msieve_obj *obj, poly_search_t *poly, uint32 deadline)
{
	lattice_fb_t L;
	sieve_fb_t sieve_special_q, sieve_large_p1, sieve_large_p2;
	uint32 special_q_min, special_q_max;
	uint32 large_fb_max;
	uint32 num_pieces;
	sieve_fb_param_t params;
	double bits;
	double p_scale;
	uint32 degree = poly->degree;
	uint32 max_roots = (degree != 5) ? degree : 1;
	curr_poly_t *middle_poly = poly->batch + poly->num_poly / 2;
	curr_poly_t *last_poly = poly->batch + poly->num_poly - 1;

	bits = log(middle_poly->p_size_max) / M_LN2;
	printf("p = %.2lf sieve = %.2lf bits\n",
			bits, log(middle_poly->sieve_size) / M_LN2);

	get_poly_params(bits, &params);

	if (degree != 5) {
		printf("error: degree %u not currently supported in special_q mode\n", degree);
		exit (-1);
	}

	p_scale = params.p_scale;
	num_pieces = params.num_pieces;
	special_q_min = params.special_q_min;
	special_q_max = params.special_q_max;
	large_fb_max = MIN(250000, special_q_min);

	sieve_fb_init(&sieve_special_q, poly,
			0, 0, /* prime special_q */
			1, max_roots,
			0, 0);

	sieve_fb_init(&sieve_large_p1, poly,
			5, large_fb_max,
			1, max_roots,
			0, 1 /* 1 (mod 4) */);

	sieve_fb_init(&sieve_large_p2, poly,
			5, large_fb_max,
			1, max_roots,
			0, 3 /* 3 (mod 4) */);

	L.poly = poly;
	L.start_time = time(NULL);
	L.deadline = deadline;
#ifdef HAVE_CUDA
	L.gpu_info = poly->gpu_info;
#endif

	gmp_printf("coeff %Zd-%Zd"
		   " %" PRIu32 " %" PRIu32 "\n",
		   poly->batch[0].high_coeff, last_poly->high_coeff,
		   special_q_min, special_q_max);

	while (1) {
		uint32 done;
		uint32 special_q_min2, special_q_max2;
		uint32 large_p_min, large_p_max;

		special_q_min2 = special_q_min;
		if (special_q_min2 <= special_q_max / p_scale)
			special_q_max2 = special_q_min2 * p_scale;
		else
			break;

		large_p_max = sqrt(middle_poly->p_size_max / special_q_min2);
		large_p_min = large_p_max / p_scale;
		if (large_p_min <= special_q_max) { /* shouldn't happen */
			printf("error: special_q is too large\n");
			exit (-1);
		}

		if (num_pieces > 1) { /* randomize special_q */
			uint32 piece_len = (special_q_max2 - special_q_min2)
							/ num_pieces;
			uint32 piece = get_rand(&obj->seed1,
						&obj->seed2) % num_pieces;

			special_q_min2 += piece * piece_len;
			special_q_max2 = special_q_min2 + piece_len;
		}

#ifdef HAVE_CUDA
		if (large_p_max < ((uint32)1 << 24))
			L.gpu_module = poly->gpu_module48;
		else
			L.gpu_module = poly->gpu_module64;

		CUDA_TRY(cuModuleGetFunction(&L.gpu_kernel, 
				L.gpu_module, "sieve_kernel"))
		if (degree != 5)
			CUDA_TRY(cuModuleGetGlobal(&L.gpu_p_array, 
				NULL, L.gpu_module, "pbatch"))
#endif

		if (degree != 5) {
			done = sieve_lattice_deg46_64(obj, &L,
					&sieve_special_q,
					&sieve_large_p1, &sieve_large_p2,
					special_q_min2, special_q_max2,
					large_p_min, large_p_max);
		}
		else { /* degree 5 */
			done = sieve_lattice_deg5_64(obj, &L,
					&sieve_special_q,
					&sieve_large_p1, &sieve_large_p2,
					special_q_min2, special_q_max2,
					large_p_min, large_p_max);
		}

		if (done)
			break;

		special_q_min *= p_scale + 1;
	}

	sieve_fb_free(&sieve_special_q);
	sieve_fb_free(&sieve_large_p1);
	sieve_fb_free(&sieve_large_p2);
}
