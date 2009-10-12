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
#include "stage1_core.h"

#define HOST_BATCH_SIZE 30000

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 num_p_alloc;
	uint32 last_p;

	uint32 *p;
	uint32 *lattice_size;
	uint64 *roots[POLY_BATCH_SIZE];
} p_soa_var_t;

static void
p_soa_var_init(p_soa_var_t *soa)
{
	uint32 i;
	memset(soa, 0, sizeof(p_soa_var_t));

	soa->num_p_alloc = 1000;
	soa->p = (uint32 *)xmalloc(soa->num_p_alloc * 
				sizeof(uint32));
	soa->lattice_size = (uint32 *)xmalloc(soa->num_p_alloc * 
				sizeof(uint32));
	for (i = 0; i < POLY_BATCH_SIZE; i++) {
		soa->roots[i] = (uint64 *)xmalloc(
					soa->num_p_alloc * 
					sizeof(uint64));
	}
}

static void
p_soa_var_free(p_soa_var_t *soa)
{
	uint32 i;

	free(soa->p);
	free(soa->lattice_size);
	for (i = 0; i < POLY_BATCH_SIZE; i++)
		free(soa->roots[i]);
}

static void
p_soa_var_reset(p_soa_var_t *soa)
{
	soa->num_p = 0;
	soa->last_p = 0;
}

static void
p_soa_var_grow(p_soa_var_t *soa)
{
	uint32 i;

	soa->num_p_alloc *= 2;
	soa->p = (uint32 *)xrealloc(soa->p, 
				soa->num_p_alloc * 
				sizeof(uint32));
	soa->lattice_size = (uint32 *)xrealloc(soa->lattice_size, 
				soa->num_p_alloc * 
				sizeof(uint32));
	for (i = 0; i < POLY_BATCH_SIZE; i++) {
		soa->roots[i] = (uint64 *)xrealloc(soa->roots[i], 
					soa->num_p_alloc * 
					sizeof(uint64));
	}
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
		if (soa->num_p_alloc == soa->num_p)
			p_soa_var_grow(soa);

		soa->p[num] = (uint32)p;
		soa->lattice_size[num] = (uint32)(L->poly->batch[
				which_poly].sieve_size / ((double)p * p));
		soa->num_p++;
		soa->last_p = (uint32)p;
		soa->roots[which_poly][num] = gmp2uint64(roots[0]);
	}
	else {
		soa->roots[which_poly][num - 1] = gmp2uint64(roots[0]);
	}
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

/*------------------------------------------------------------------------*/
static uint64 
montmul_r(uint64 n, uint32 w) {

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
static void
cpu_kernel(poly_search_t *poly, 
		p_soa_t *pbatch, uint32 num_p, 
		p_soa_t *qbatch, uint32 num_q)
{
	uint32 i, j, k;
	uint64 inv[P_SOA_BATCH_SIZE];
	uint64 p2[P_SOA_BATCH_SIZE];
	uint32 num_poly = poly->num_poly;

	for (i = 0; i < num_q; i++) {
		uint32 q = qbatch->p[i];
		uint64 q2 = (uint64)q * q;
		uint64 invprod;
		uint32 q2_w = montmul_w((uint32)q2);
		uint64 q2_r = montmul_r(q2, q2_w);

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
			uint32 plattice = pbatch->lattice_size[j];
			uint64 pinv = inv[j];

			for (k = 0; k < num_poly; k++) {

				uint64 proot = pbatch->roots[k][j];
				uint64 qroot = qbatch->roots[k][i];
				uint64 res = montmul(pinv,
							mp_modsub_2(
							   qroot,
							   proot, q2),
							q2, q2_w);

				if (res < plattice ||
				    res >= q2 - plattice) {
					handle_collision(poly, k,
						(uint64)pbatch->p[j],
						proot, res, (uint64)q);
				}
			}
		}
	}
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_batch(msieve_obj *obj, lattice_fb_t *L)
{
	uint32 j, k;
	p_soa_var_t * p_array = (p_soa_var_t *)L->p_array;
	p_soa_var_t * q_array = (p_soa_var_t *)L->q_array;
	uint32 num_poly = L->poly->num_poly;

	p_soa_t *p_marshall = L->p_marshall;
	p_soa_t *q_marshall = L->q_marshall;
	uint32 num_q_done = 0;

	while (num_q_done < q_array->num_p) {

		uint32 q_left;
		uint32 num_p_done = 0;
		time_t curr_time;
		double elapsed;

		uint32 curr_num_q = MIN(P_SOA_BATCH_SIZE,
					q_array->num_p - num_q_done);

		q_left = q_array->num_p - (num_q_done + curr_num_q);
		if (q_left > 0 && q_left < P_SOA_BATCH_SIZE / 4)
			curr_num_q /= 2;

		for (j = 0; j < curr_num_q; j++) {
			q_marshall->p[j] = q_array->p[num_q_done + j];
			q_marshall->lattice_size[j] = 
					q_array->lattice_size[num_q_done + j];

			for (k = 0; k < num_poly; k++) {
				q_marshall->roots[k][j] =
					q_array->roots[k][num_q_done + j];
			}
		}

		while (num_p_done < p_array->num_p) {

			uint32 p_left;
			uint32 curr_num_p = MIN(P_SOA_BATCH_SIZE,
						p_array->num_p - num_p_done);

			p_left = p_array->num_p - (num_p_done + curr_num_p);
			if (p_left > 0 && p_left < P_SOA_BATCH_SIZE / 4)
				curr_num_p /= 2;

			for (j = 0; j < curr_num_p; j++) {
				p_marshall->p[j] = p_array->p[num_p_done + j];
				p_marshall->lattice_size[j] = 
					p_array->lattice_size[num_p_done + j];

				for (k = 0; k < num_poly; k++) {
					p_marshall->roots[k][j] =
						p_array->roots[k][num_p_done+j];
				}
			}

			printf("qnum %u pnum %u\n", curr_num_q, curr_num_p);

			cpu_kernel(L->poly, p_marshall, curr_num_p,
					q_marshall, curr_num_q);

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
uint32
sieve_lattice_cpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_small, sieve_fb_t *sieve_large, 
		uint32 small_p_min, uint32 small_p_max, 
		uint32 large_p_min, uint32 large_p_max)
{
	uint32 i;
	uint32 min_small, min_large;
	uint32 quit = 0;
	p_soa_var_t * p_array;
	p_soa_var_t * q_array;
	clock_t clock_start;

	L->p_marshall = (p_soa_t *)xmalloc(sizeof(p_soa_t));
	L->q_marshall = (p_soa_t *)xmalloc(sizeof(p_soa_t));
	p_array = L->p_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	q_array = L->q_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	p_soa_var_init(p_array);
	p_soa_var_init(q_array);

	printf("------- %u-%u %u-%u\n",
			small_p_min, small_p_max,
			large_p_min, large_p_max);

	min_large = large_p_min;
	sieve_fb_reset(sieve_small, (uint64)large_p_min, 
			(uint64)large_p_max, 1, 1);

	clock_start = clock();

	while (min_large < large_p_max) {

		L->fill_p = 0;
		p_soa_var_reset(q_array);
		for (i = 0; i < HOST_BATCH_SIZE && 
				min_large < large_p_max; i++) {
			min_large = sieve_fb_next(sieve_small, L->poly,
						store_p_soa, L);
		}
		if (q_array->num_p == 0)
			goto finished;

		min_small = small_p_min;
		sieve_fb_reset(sieve_large, 
				(uint64)small_p_min, (uint64)small_p_max,
				1, 1);

		while (min_small <= small_p_max) {

			L->fill_p = 1;
			p_soa_var_reset(p_array);
			for (i = 0; i < HOST_BATCH_SIZE && 
					min_small < small_p_max; i++) {
				min_small = sieve_fb_next(sieve_large, L->poly,
							store_p_soa, L);
			}
			if (p_array->num_p == 0)
				goto finished;

			if (sieve_lattice_batch(obj, L)) {
				quit = 1;
				goto finished;
			}
		}
	}

	printf("%lf\n", (double)(clock() - clock_start) / CLOCKS_PER_SEC);

finished:
	p_soa_var_free(p_array);
	p_soa_var_free(q_array);
	free(p_array);
	free(q_array);
	free(L->p_marshall);
	free(L->q_marshall);
	return quit;
}


