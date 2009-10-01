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

#include "stage1.h"

#if 1
#define CHECK
#endif

/* structures for storing arithmetic progressions. Rational
   leading coeffs of NFS polynomials are assumed to be the 
   product of two groups of factors p, each of size at most 32
   bits (32 bits is enough for 512-bit factorizations), and 
   candidates must satisfy a condition modulo p^2 */

#define MAX_P_BITS 32
#define MAX_P ((uint64)1 << MAX_P_BITS)

/*------------------------------------------------------------------------*/
static void 
lattice_fb_init(lattice_fb_t *L, poly_search_t *poly, 
		uint32 deadline)
{
	L->poly = poly;
	L->start_time = get_cpu_time();
	L->deadline = deadline;
	L->num_tests = 0;
}

/*------------------------------------------------------------------------*/
void
handle_collision(poly_search_t *poly,
		uint64 p, uint64 proot, uint64 res, uint64 q)
{
	uint64_2gmp(p, poly->tmp1);
	uint64_2gmp(q, poly->tmp2);
	uint64_2gmp(proot, poly->tmp3);
	uint64_2gmp(res, poly->tmp4);

	mpz_mul(poly->p, poly->tmp1, poly->tmp2);
	mpz_mul(poly->tmp1, poly->tmp1, poly->tmp1);
	if (mpz_cmp(poly->tmp4, poly->tmp2) > 0)
		mpz_submul(poly->tmp4, poly->tmp2, poly->tmp2);

	mpz_addmul(poly->tmp3, poly->tmp1, poly->tmp4);
	mpz_add(poly->m0, poly->trans_m0, poly->tmp3);

#ifdef CHECK
	gmp_printf("p %.0lf q %.0lf coeff %Zd\n", 
				(double)p, (double)q, poly->p);

	mpz_pow_ui(poly->tmp1, poly->m0, (mp_limb_t)poly->degree);
	mpz_mul(poly->tmp2, poly->p, poly->p);
	mpz_sub(poly->tmp1, poly->trans_N, poly->tmp1);
	mpz_tdiv_r(poly->tmp3, poly->tmp1, poly->tmp2);
	if (mpz_cmp_ui(poly->tmp3, (mp_limb_t)0)) {
		printf("crap\n");
		exit(-1);
	}
#endif

	mpz_mul_ui(poly->tmp1, poly->high_coeff, (mp_limb_t)poly->degree);
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

	poly->callback(poly->high_coeff, poly->p, poly->m0, 
			poly->coeff_max, poly->callback_data);
}

/*------------------------------------------------------------------------*/
void
sieve_lattice(msieve_obj *obj, poly_search_t *poly, 
		uint32 small_fb_max, uint32 large_fb_min, 
		uint32 large_fb_max, uint32 deadline)
{
	lattice_fb_t L;
	sieve_fb_t sieve_small, sieve_large;
	uint32 small_p_min, small_p_max;
	uint32 large_p_min, large_p_max;
	uint32 bits;
	double p_scale = 1.1;

	if (poly->p_size_max >= (double)MAX_P * MAX_P) {
		printf("error: rational leading coefficient is too large\n");
		exit(-1);
	}

	bits = mpz_sizeinbase(poly->N, 2);
	switch(poly->degree) {
	case 4:
		if (bits < 306)
			p_scale = 1.3;
		else if (bits < 323)
			p_scale = 1.2;
		break;

	case 5:
		if (bits < 363)
			p_scale = 1.3;
		else if (bits < 396)
			p_scale = 1.2;
		else if (bits < 420)
			p_scale = 1.1;
		else
			p_scale = 1.03;
		break;
	}

	large_p_min = sqrt(poly->p_size_max);
	if (large_p_min >= MAX_P / p_scale)
		large_p_max = MAX_P - 1;
	else
		large_p_max = p_scale * large_p_min;

	small_p_min = large_p_min / p_scale;
	small_p_max = large_p_min - 1;

	gmp_printf("coeff %Zd"
		   " %" PRIu64 " %" PRIu64 
		   " %" PRIu64 " %" PRIu64 "\n",
			poly->high_coeff, 
			(uint64)small_p_min, (uint64)small_p_max,
			(uint64)large_p_min, (uint64)large_p_max);

	sieve_fb_init(&sieve_small, poly, large_fb_min, large_fb_max);
	sieve_fb_init(&sieve_large, poly, 5, small_fb_max);
	lattice_fb_init(&L, poly, deadline);

	while (1) {
#if 1
		if (sieve_lattice_gpu(obj, &L, &sieve_small, &sieve_large,
					small_p_min, small_p_max,
					large_p_min, large_p_max)) {
			break;
		}
#else
		if (sieve_lattice_cpu(obj, &L, &sieve_small, &sieve_large,
					small_p_min, small_p_max,
					large_p_min, large_p_max)) {
			break;
		}
#endif
		small_p_max = small_p_min - 1;
		small_p_min = small_p_min / p_scale;

		if (poly->p_size_max / small_p_max >= MAX_P)
			break;
		large_p_min = poly->p_size_max / small_p_max;

		if (poly->p_size_max / small_p_min >= MAX_P)
			large_p_max = MAX_P - 1;
		else
			large_p_max = poly->p_size_max / small_p_min;
	}

	sieve_fb_free(&sieve_small);
	sieve_fb_free(&sieve_large);
}
