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

#define MAX_P_BITS 32
#define MAX_P (((uint64)1 << MAX_P_BITS) - 1)

#define MIN_SPECIAL_Q 17

static const sieve_fb_param_t sieve_fb_params[] = {

	{ 40, 1.3, 100,    1,       1,        1},
	{ 48, 1.3,  25,    1,       1,     1500},
	{ 56, 1.2,  10,   25,    1000,   250000},
	{ 64, 1.2,   5,  100,  150000,  1500000},
	{ 72, 1.1,   5,  500,  750000,  7500000},
	{ 80, 1.1,   5, 2500, 5000000, 50000000},
};

#define NUM_SIEVE_FB_PARAMS (sizeof(sieve_fb_params) / \
				sizeof(sieve_fb_params[0]))

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
	params->max_diverge = (low->max_diverge * k +
				high->max_diverge * j) / dist;

	params->num_pieces = exp((log(low->num_pieces) * k +
					log(high->num_pieces) * j) / dist);
	params->special_q_min = exp((log(low->special_q_min) * k +
					log(high->special_q_min) * j) / dist);
	params->special_q_max = exp((log(low->special_q_max) * k +
					log(high->special_q_max) * j) / dist);
}

/*------------------------------------------------------------------------*/
#ifdef HAVE_CUDA
void
sieve_lattice_gpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_param_t *params,
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max,
		uint32 large_fb_max)
{
	uint32 done;
	uint32 large_p1_min, large_p1_max;
	uint32 large_p2_min, large_p2_max;
	sieve_fb_t sieve_large_p1, sieve_large_p2;
	uint32 degree = L->poly->degree;
	uint32 max_roots = (degree != 5) ? degree : 1;
	curr_poly_t *middle_poly = L->poly->batch + L->poly->num_poly / 2;
	double p_scale = params->p_scale;

	sieve_fb_init(&sieve_large_p1, L->poly,
			0, 0,
			1, max_roots,
			0, 0);

	sieve_fb_init(&sieve_large_p2, L->poly,
			5, large_fb_max,
			1, max_roots,
			0, 0);

	large_p1_min = sqrt(middle_poly->p_size_max / special_q_max);
	if (large_p1_min < MAX_P / p_scale)
		large_p1_max = large_p1_min * p_scale;
	else
		large_p1_max = MAX_P;

	large_p2_max = large_p1_min - 1;
	if (params->special_q_max <= large_p2_max / p_scale)
		large_p2_min = large_p2_max / p_scale;
	else
		large_p2_min = params->special_q_max;

	while (1) {
		if (large_p1_max < ((uint32)1 << 24))
			L->gpu_module = L->poly->gpu_module48;
		else
			L->gpu_module = L->poly->gpu_module64;

		CUDA_TRY(cuModuleGetFunction(&L->gpu_kernel, 
				L->gpu_module, "sieve_kernel"))
		if (degree != 5)
			CUDA_TRY(cuModuleGetGlobal(&L->gpu_p_array, 
				NULL, L->gpu_module, "pbatch"))

		if (degree != 5) {
			done = sieve_lattice_deg46_64(obj, L,
				sieve_special_q,
				&sieve_large_p1, &sieve_large_p2,
				special_q_min, special_q_max,
				large_p1_min, large_p1_max,
				large_p2_min, large_p2_max);
		}
		else { /* degree 5 */
			done = sieve_lattice_deg5_64(obj, L,
				sieve_special_q,
				&sieve_large_p1, &sieve_large_p2,
				special_q_min, special_q_max,
				large_p1_min, large_p1_max,
				large_p2_min, large_p2_max);
		}

		if (done)
			goto finished;

		if (large_p1_max == MAX_P ||
		    large_p2_min == params->special_q_max ||
		    large_p1_max / large_p2_min > params->max_diverge)
			break;

		large_p1_min = large_p1_max + 1;
		if (large_p1_min < MAX_P / p_scale)
			large_p1_max = large_p1_min * p_scale;
		else
			large_p1_max = MAX_P;

		large_p2_max = large_p2_min - 1;
		if (params->special_q_max <= large_p2_max / p_scale)
			large_p2_min = large_p2_max / p_scale;
		else
			large_p2_min = params->special_q_max;
	}

finished:
	sieve_fb_free(&sieve_large_p1);
	sieve_fb_free(&sieve_large_p2);
}
#endif

/*------------------------------------------------------------------------*/
void
sieve_lattice(msieve_obj *obj, poly_search_t *poly, uint32 deadline)
{
	lattice_fb_t L;
	sieve_fb_t sieve_special_q;
	uint32 special_q_min, special_q_max;
	uint32 large_fb_max;
	uint32 num_pieces;
	double p_scale;
	sieve_fb_param_t params;
	double bits;
	uint32 degree = poly->degree;
	curr_poly_t *middle_poly = poly->batch + poly->num_poly / 2;
	curr_poly_t *last_poly = poly->batch + poly->num_poly - 1;
#ifdef HAVE_CUDA
	uint32 max_roots = (degree != 5) ? degree : 1;
#else
	uint32 max_roots = degree;
#endif

	bits = log(middle_poly->p_size_max) / M_LN2;
	printf("p = %.2lf sieve = %.2lf bits\n",
			bits, log(middle_poly->sieve_size) / M_LN2);

	get_poly_params(bits, &params);

	num_pieces = params.num_pieces;
	special_q_min = params.special_q_min;
	special_q_max = params.special_q_max;
	large_fb_max = MIN(500000, special_q_min);
	p_scale = params.p_scale;

	sieve_fb_init(&sieve_special_q, poly,
			0, 0, /* prime special_q */
			1, max_roots,
			0, 0);

	L.poly = poly;
	L.start_time = time(NULL);
	L.deadline = deadline;
#ifdef HAVE_CUDA
	L.gpu_info = poly->gpu_info;
#endif

	gmp_printf("coeff %Zd-%Zd specialq %u - %u\n",
		   poly->batch[0].high_coeff, last_poly->high_coeff,
		   special_q_min, special_q_max);

	if (num_pieces > 1) { /* randomize special_q */
		double piece_ratio = pow((double)special_q_max / special_q_min,
					 (double)1 / num_pieces);
		uint32 piece = get_rand(&obj->seed1,
					&obj->seed2) % num_pieces;

		special_q_min *= pow(piece_ratio, (double)piece);
		special_q_max = special_q_min * piece_ratio;
	}

	while (1) {
		uint32 special_q_min2, special_q_max2;

		if (special_q_min <= 1) {
			special_q_min2 = special_q_max2 = 1;
		}
		else {
			special_q_min2 = MAX(special_q_min, MIN_SPECIAL_Q);
			if (special_q_min2 > special_q_max)
				break;
			else if (special_q_min2 <= special_q_max / p_scale)
				special_q_max2 = special_q_min2 * p_scale;
			else
				special_q_max2 = special_q_max;
		}

#ifdef HAVE_CUDA
		sieve_lattice_gpu(obj, &L, &params, &sieve_special_q,
				special_q_min2, special_q_max2, large_fb_max);
#else
		sieve_lattice_cpu(obj, &L, &params, &sieve_special_q,
				special_q_min2, special_q_max2, large_fb_max);
#endif

		special_q_min = special_q_max2 + 1;
	}

	sieve_fb_free(&sieve_special_q);
}
