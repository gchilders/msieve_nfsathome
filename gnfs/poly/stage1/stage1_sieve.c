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

/* structure to store batches of p values that are small on
average, containing few factors on average */

#define P_SMALL_BATCH_SIZE 4096
#define MAX_SMALL_ROOTS 8

typedef struct {
	uint32 num_p;
	uint32 pad[15];

	uint32 num_roots[P_SMALL_BATCH_SIZE];
	uint32 p[P_SMALL_BATCH_SIZE];
	uint32 roots[2*MAX_SMALL_ROOTS][P_SMALL_BATCH_SIZE];
} p_small_batch_t;

/* structure to store batches of p values that are large on
average, containing many factors on average */

#define P_LARGE_BATCH_SIZE 512
#define MAX_LARGE_ROOTS MAX_ROOTS

typedef struct {
	uint32 num_p;
	uint32 pad[15];

	uint32 num_roots[P_LARGE_BATCH_SIZE];
	uint32 p[P_LARGE_BATCH_SIZE];
	uint32 roots[2*MAX_LARGE_ROOTS][P_LARGE_BATCH_SIZE];
} p_large_batch_t;


typedef struct {
	p_small_batch_t *p_array;
	p_large_batch_t *q_array;

	poly_search_t *poly;

	double start_time;
	uint32 deadline;
	uint32 num_tests;
} lattice_fb_t;

/*------------------------------------------------------------------------*/
static void 
lattice_fb_free(lattice_fb_t *L)
{
	aligned_free(L->p_array);
	aligned_free(L->q_array);
}

/*------------------------------------------------------------------------*/
static void 
lattice_fb_init(lattice_fb_t *L, poly_search_t *poly, 
		uint32 deadline)
{
	L->poly = poly;
	L->start_time = get_cpu_time();
	L->deadline = deadline;
	L->num_tests = 0;

	L->p_array = (p_small_batch_t *)aligned_malloc(
				sizeof(p_small_batch_t), 64);
	L->q_array = (p_large_batch_t *)aligned_malloc(
				sizeof(p_large_batch_t), 64);
}

/*------------------------------------------------------------------------*/
static void
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
static uint32 montmul_w(uint32 n) {

	uint32 res = 2 + n;
	res = res * (2 + n * res);
	res = res * (2 + n * res);
	res = res * (2 + n * res);
	return res * (2 + n * res);
}

static uint64 montmul(uint64 a, uint64 b,
			uint64 n, uint32 w) {

	uint32 a0 = (uint32)a;
	uint32 a1 = (uint32)(a >> 32);
	uint32 b0 = (uint32)b;
	uint32 b1 = (uint32)(b >> 32);
	uint32 n0 = (uint32)n;
	uint32 n1 = (uint32)(n >> 32);
	uint64 prod;
	uint32 acc0, acc1, acc2, nmult;

	prod = (uint64)a0 * b0;
	acc0 = (uint32)prod;
	prod = (prod >> 32) + (uint64)a1 * b0;
	acc1 = (uint32)prod;
	acc2 = (uint32)(prod >> 32);

	nmult = acc0 * w;
	prod = acc0 + (uint64)nmult * n0;
	prod = (prod >> 32) + (uint64)acc1 + (uint64)nmult * n1;
	acc0 = (uint32)prod;
	prod = (prod >> 32) + (uint64)acc2;
	acc1 = (uint32)prod;
	acc2 = (uint32)(prod >> 32);

	prod = (uint64)acc0 + (uint64)a0 * b1;
	acc0 = (uint32)prod;
	prod = (prod >> 32) + (uint64)acc1 + (uint64)a1 * b1;
	acc1 = (uint32)prod;
	acc2 = (uint32)(prod >> 32) + acc2;

	nmult = acc0 * w;
	prod = acc0 + (uint64)nmult * n0;
	prod = (prod >> 32) + (uint64)acc1 + (uint64)nmult * n1;
	acc0 = (uint32)prod;
	prod = (prod >> 32) + (uint64)acc2;
	acc1 = (uint32)prod;
	acc2 = (uint32)(prod >> 32);

	prod = (uint64)acc1 << 32 | acc0;
	if (acc2 || prod >= n)
		return prod - n;
	else
		return prod;
}

static uint64 montmul_r(uint64 n, uint32 w) {

	uint32 n1 = (uint32)(n >> 32);
	uint32 shift;
	uint32 i;
	uint64 shifted_n;
	uint64 res;

#if defined(GCC_ASM32X) || defined(GCC_ASM64X) 
	ASM_G("bsrl %1, %0": "=r"(shift) : "rm"(n1) : "cc");
#else
	uint32 mask = 0x80000000;
	shift = 31;
	if ((n1 >> 16) == 0) {
		mask = 0x8000;
		shift -= 16;
	}

	while ( !(n1 & mask) ) {
		shift--;
		mask >>= 1;
	}
#endif

	shift = 31 - shift;
	shifted_n = n << shift;
	res = -shifted_n;

	for (i = 64 - shift; i < 72; i++) {
		if (res >> 63)
			res = res + res - shifted_n;
		else
			res = res + res;

		if (res >= shifted_n)
			res -= shifted_n;
	}

	res = res >> shift;
	res = montmul(res, res, n, w);
	res = montmul(res, res, n, w);
	return montmul(res, res, n, w);
}

/*------------------------------------------------------------------------*/
static uint32
sieve_lattice_batch(lattice_fb_t *L)
{
	uint32 i, j, k, m;
	double sieve_size = L->poly->sieve_size;
	p_small_batch_t *pbatch = L->p_array;
	p_large_batch_t *qbatch = L->q_array;
	uint32 num_p = pbatch->num_p;
	uint32 num_q = qbatch->num_p;
	uint32 num_tests;

	uint32 lattice_size[P_SMALL_BATCH_SIZE];
	uint64 inv[P_SMALL_BATCH_SIZE];
	uint64 p2[P_SMALL_BATCH_SIZE];

	for (i = num_tests = 0; i < num_p; i++) {
		uint32 p = pbatch->p[i];

		num_tests += pbatch->num_roots[i];
		lattice_size[i] = sieve_size / ((double)p * p);
	}

	for (i = 0; i < num_q; i++) {
		uint32 q = qbatch->p[i];
		uint64 q2 = (uint64)q * q;
		uint32 num_qroots = qbatch->num_roots[i];
		uint64 qroots[MAX_ROOTS];
		uint64 invprod;
		uint32 q2_w = montmul_w((uint32)q2);
		uint64 q2_r = montmul_r(q2, q2_w);

		for (j = 0; j < num_qroots; j++) {
			uint32 qr0 = qbatch->roots[2*j][i];
			uint32 qr1 = qbatch->roots[2*j+1][i];
			qroots[j] = (uint64)qr1 << 32 | qr0;
		}

		/* Montgomery's batch inverse algorithm */

		for (j = 0; j < num_p; j++) {
			uint32 p = pbatch->p[j];
			p2[j] = montmul((uint64)p * p, q2_r, q2, q2_w);
		}

		inv[0] = invprod = p2[0];
		for (j = 1; j < num_p; j++) {
			inv[j] = invprod = montmul(invprod, p2[j], q2, q2_w);
		}

		invprod = mp_modinv_2(invprod, q2);
		invprod = montmul(invprod, q2_r, q2, q2_w);
		invprod = montmul(invprod, q2_r, q2, q2_w);
		for (j = num_p - 1; j; j--) {
			inv[j] = montmul(invprod, inv[j-1], q2, q2_w);
			invprod = montmul(invprod, p2[j], q2, q2_w);
		}
		inv[0] = invprod;

		/* do the tests */

		for (j = 0; j < num_p; j++) {
			uint32 num_proots = pbatch->num_roots[j];
			uint32 plattice = lattice_size[j];
			uint64 pinv = inv[j];

			for (k = 0; k < num_proots; k++) {

				uint32 pr0 = pbatch->roots[2*k][j];
				uint32 pr1 = pbatch->roots[2*k+1][j];
				uint64 proot = (uint64)pr1 << 32 | pr0;
						
				for (m = 0; m < num_qroots; m++) {
	
					uint64 res = montmul(pinv,
								mp_modsub_2(
								   qroots[m],
								   proot, q2),
								q2, q2_w);

					if (res < plattice ||
					    res >= q2 - plattice) {
						handle_collision(L->poly,
							(uint64)pbatch->p[j],
							proot, res, (uint64)q);
					}
				}
			}
		}

		L->num_tests += num_qroots * num_tests;
		if (L->num_tests >= 8000000) {

			double curr_time = get_cpu_time();
			double elapsed = curr_time - L->start_time;

			if (elapsed > L->deadline)
				return 1;
			L->num_tests = 0;
		}
	}

	return 0;
}

/*------------------------------------------------------------------------*/
static void 
store_small_p(uint64 p, uint32 num_roots,
		mpz_t *roots, void *extra)
{
	lattice_fb_t *L = (lattice_fb_t *)extra;
	p_small_batch_t *batch = L->p_array;
	uint32 num;
	uint32 i;

	num = batch->num_p;
	batch->p[num] = (uint32)p;
	batch->num_roots[num] = num_roots;

	for (i = 0; i < num_roots; i++) {
		uint64 root = gmp2uint64(roots[i]);

		batch->roots[2*i][num] = (uint32)root;
		batch->roots[2*i+1][num] = (uint32)(root >> 32);
	}
	batch->num_p++;
}

/*------------------------------------------------------------------------*/
static void 
store_large_p(uint64 p, uint32 num_roots,
		mpz_t *roots, void *extra)
{
	lattice_fb_t *L = (lattice_fb_t *)extra;
	p_large_batch_t *batch = L->q_array;
	uint32 num;
	uint32 i;

	num = batch->num_p;
	batch->p[num] = (uint32)p;
	batch->num_roots[num] = num_roots;

	for (i = 0; i < num_roots; i++) {
		uint64 root = gmp2uint64(roots[i]);

		batch->roots[2*i][num] = (uint32)root;
		batch->roots[2*i+1][num] = (uint32)(root >> 32);
	}
	batch->num_p++;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_lattice_core(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_small, sieve_fb_t *sieve_large, 
		uint32 small_p_min, uint32 small_p_max, 
		uint32 large_p_min, uint32 large_p_max)
{
	uint32 i;
	uint32 min_small, min_large;
	uint32 degree = L->poly->degree;

	printf("p %u %u %u %u\n",
			small_p_min, small_p_max,
			large_p_min, large_p_max);

	min_small = small_p_min;
	sieve_fb_reset(sieve_small, 
			(uint64)small_p_min, (uint64)small_p_max,
			1, degree);

	while (min_small < small_p_max) {

		L->p_array->num_p = 0;
		for (i = 0; i < P_SMALL_BATCH_SIZE && 
				min_small < small_p_max; i++) {
			min_small = sieve_fb_next(sieve_small, L->poly,
						store_small_p, L);
		}
		if (L->p_array->num_p == 0)
			return 0;

		printf("batch %u %u\n", L->p_array->num_p, min_small);

		min_large = large_p_min;
		sieve_fb_reset(sieve_large, 
				(uint64)large_p_min, (uint64)large_p_max,
				degree, MAX_ROOTS);

		while (min_large <= large_p_max) {

			L->q_array->num_p = 0;
			for (i = 0; i < P_LARGE_BATCH_SIZE && 
					min_large < large_p_max; i++) {
				min_large = sieve_fb_next(sieve_large, L->poly,
							store_large_p, L);
			}
			if (L->q_array->num_p == 0)
				break;

			if (sieve_lattice_batch(L) ||
			    obj->flags & MSIEVE_FLAG_STOP_SIEVING)
				return 1;
		}
	}

	return 0;
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
		if (sieve_lattice_core(obj, &L, &sieve_small, &sieve_large,
					small_p_min, small_p_max,
					large_p_min, large_p_max)) {
			break;
		}

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

	lattice_fb_free(&L);
	sieve_fb_free(&sieve_small);
	sieve_fb_free(&sieve_large);
}
