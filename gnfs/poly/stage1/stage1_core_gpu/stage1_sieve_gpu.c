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
#include <stage1_core_gpu/stage1_core_gpu_common.h>
#include <stage1_core_gpu/stage1_core_gpu_nosq.h>
#include <stage1_core_gpu/stage1_core_gpu_sq.h>

/*------------------------------------------------------------------------*/
static void
stage1_bounds_update(msieve_obj *obj, poly_search_t *poly)
{
	/* determine the parametrs for the collision search,
	   given one leading algebraic coefficient a_d */

	uint32 degree = poly->degree;
	double N = mpz_get_d(poly->N);
	double high_coeff = mpz_get_d(poly->high_coeff);
	double m0 = pow(N / high_coeff, 1./degree);
	double skewness_min, coeff_max, p_size_max, cutoff;
	double special_q_min, special_q_max;
	uint32 num_pieces;

	/* we don't know the optimal skewness for this polynomial
	   but at least can bound the skewness. The value of the
	   third-highest coefficient is from Kleinjung's 2006
	   poly selection algorithm as published in Math. Comp. */

	switch (degree) {
	case 4:
		skewness_min = sqrt(m0 / poly->norm_max);
		coeff_max = poly->norm_max;
		break;

	case 5:
		skewness_min = pow(m0 / poly->norm_max, 2./3.);
		coeff_max = poly->norm_max / sqrt(skewness_min);
		break;

	case 6:
		skewness_min = sqrt(m0 / poly->norm_max);
		coeff_max = poly->norm_max / skewness_min;
		break;

	default:
		printf("error: unexpected poly degree %d\n", degree);
		exit(-1);
	}

	/* we perform the collision search on a transformed version
	   of N and the low-order rational coefficient m. In the
	   transformed coordinates, a_d is 1 and a_{d-1} is 0. When
	   a hit is found, we undo the transformation to recover
	   the correction to m that makes the new polynomial 'work' */

	mpz_mul_ui(poly->trans_N, poly->high_coeff, (mp_limb_t)degree);
	mpz_pow_ui(poly->trans_N, poly->trans_N, (mp_limb_t)(degree - 1));
	mpz_mul_ui(poly->trans_N, poly->trans_N, (mp_limb_t)degree);
	mpz_mul(poly->trans_N, poly->trans_N, poly->N);
	mpz_root(poly->trans_m0, poly->trans_N, (mp_limb_t)degree);

	/* for leading rational coefficient l, the sieve size
	   will be l^2*cutoff. This is based on the norm limit
	   given above, and largely determines how difficult
	   it will be to find acceptable collisions in the
	   search */

	cutoff = coeff_max / m0 / degree;

	/* the GPU code for small inputs doesn't care how
	   large the sieve size is, so favor smaller
	   special-q and try to make the range of other
	   rational factors large */

	p_size_max = MIN(coeff_max / skewness_min,
			 (double)(MAX_SPECIAL_Q / SPECIAL_Q_SCALE) *
				 (MAX_OTHER / P_SCALE - 1) *
				 (MAX_OTHER / P_SCALE - 1));

	/* 2*sieve_size must fit in a float */
	p_size_max = MIN(p_size_max, sqrt(FLT_MAX / cutoff / 2.001));

	special_q_min = MIN(MAX_SPECIAL_Q / SPECIAL_Q_SCALE,
				p_size_max / 7200000 / 7200000);
	if (special_q_min > 1) {
		special_q_min = MAX(special_q_min, 251);
		special_q_max = special_q_min * SPECIAL_Q_SCALE;
	}
	else {
		special_q_min = special_q_max = 1;
	}

	/* very large problems are split into pieces, and we
	   randomly choose one to search. This allows multiple
	   machines to search the same a_d */

	num_pieces = (special_q_max - special_q_min)
			/ (log(special_q_max) - 1)
			/ 1000;
	num_pieces = MIN(num_pieces, 2000);

	if (num_pieces > 1) { /* randomize the special_q range */

		double piece_ratio = pow((double)special_q_max / special_q_min,
					 (double)1 / num_pieces);
		uint32 piece = get_rand(&obj->seed1,
					&obj->seed2) % num_pieces;

		printf("randomizing rational coefficient: "
			"using piece #%u of %u\n",
			piece + 1, num_pieces);

		special_q_min *= pow(piece_ratio, (double)piece);
		special_q_max = special_q_min * piece_ratio;
	}

	poly->special_q_min = (uint32)special_q_min;
	poly->special_q_max = (uint32)special_q_max;
	poly->special_q_fb_max = MIN((uint32)special_q_max, 100000);

	poly->coeff_max = coeff_max;
	poly->p_size_max = p_size_max;

	/* Kleinjung's improved algorithm computes a 'correction'
	   to m0, and the coefficient a_{d-2} will be small enough
	   if the correction is smaller than sieve_size */

	poly->sieve_size = p_size_max * p_size_max * cutoff;
	mpz_set_d(poly->mp_sieve_size, poly->sieve_size);
}

/*------------------------------------------------------------------------*/
void
sieve_lattice_gpu(msieve_obj *obj, lattice_fb_t *L)
{
	uint32 quit = 0;
	poly_search_t *poly = L->poly;
	uint32 special_q_min, special_q_max;
	sieve_fb_t sieve_special_q;

	stage1_bounds_update(obj, poly);
	special_q_min = poly->special_q_min;
	special_q_max = poly->special_q_max;

	printf("p = %.2lf bits, sieve = %.2lf bits\n",
			log(poly->p_size_max) / M_LN2,
			log(poly->sieve_size) / M_LN2);

	/* set up the special q factory; special-q may have 
	   arbitrary factors, but many small factors are 
	   preferred since that will allow for many more roots
	   per special q, so we choose the factors to be as 
	   small as possible */

	sieve_fb_init(&sieve_special_q, poly,
			5, poly->special_q_fb_max,
			1, poly->degree,
			0);

	if (special_q_min == 1) { /* handle trivial case */

		quit = sieve_lattice_gpu_nosq_core(obj, L);
	}

	if (quit || special_q_max == 1)
		goto finished;

	/* if special q max is more than P_SCALE times special q
	   min, then we split the range into P_SCALE-sized parts
	   and search them individually to keep the size of the
	   leading rational coefficient close to its target size.
	   The size of the other factors of the leading rational
	   coefficient are scaled appropriately */

	while (1) {

		uint32 special_q_min2 = special_q_min;
		uint32 special_q_max2 = MIN(special_q_max,
						special_q_min * P_SCALE);

		quit = sieve_lattice_gpu_sq_core(obj, L, &sieve_special_q,
						special_q_min2,
						special_q_max2);

		if (quit || special_q_max2 > special_q_max / P_SCALE)
			break;

		special_q_min = special_q_max2 + 1;
	}

finished:
	sieve_fb_free(&sieve_special_q);
}
