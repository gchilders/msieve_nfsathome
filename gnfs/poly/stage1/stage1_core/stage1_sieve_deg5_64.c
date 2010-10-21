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
#include "cpu_intrinsics.h"

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 last_p;

	uint32 p[HOST_BATCH_SIZE];
	uint32 lattice_size[HOST_BATCH_SIZE];
	uint64 roots[POLY_BATCH_SIZE][HOST_BATCH_SIZE];
	uint64 start_roots[POLY_BATCH_SIZE][HOST_BATCH_SIZE];
} p_soa_var_t;

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

	soa = (p_soa_var_t *)L->fill_which_array;

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
static void
batch_invert(uint32 *plist, uint32 num_p, uint64 *invlist,
		uint64 q2, uint64 q2_r, uint32 q2_w)
{
	uint32 i;
	uint64 p2[INVERT_BATCH_SIZE];
	uint64 invprod;

	invlist[0] = invprod = wide_sqr32(plist[0]);
	for (i = 1; i < num_p; i++) {
		p2[i] = wide_sqr32(plist[i]);
		invlist[i] = invprod = montmul64(invprod, p2[i], q2, q2_w);
	}

	invprod = mp_modinv_2(invprod, q2);
	invprod = montmul64(invprod, q2_r, q2, q2_w);
	for (i = num_p - 1; i; i--) {
		invlist[i] = montmul64(invprod, invlist[i-1], q2, q2_w);
		invprod = montmul64(invprod, p2[i], q2, q2_w);
	}
	invlist[i] = invprod;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_lattice_batch(msieve_obj *obj, lattice_fb_t *L,
			uint32 which_special_q)
{
	uint32 i, j, k;
	p_soa_var_t * p_array = (p_soa_var_t *)L->p_array;
	p_soa_var_t * q_array = (p_soa_var_t *)L->q_array;
	p_soa_var_t * special_q_array = (p_soa_var_t *)L->special_q_array;
	uint32 num_poly = L->poly->num_poly;
	uint32 num_p = p_array->num_p;

	uint32 specialq = special_q_array->p[which_special_q];
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
			uint64 sqroot = special_q_array->start_roots[j][which_special_q];

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
			uint64 sqroot = special_q_array->start_roots[j][which_special_q];

			uint64 res = montmul64(modsub64(proot,
						sqroot, p2),
						inv, p2, p2_w);
			q_array->roots[j][i] = res;
		}
	}

	for (i = 0; i < q_array->num_p; i++) {

		uint32 q = q_array->p[i];
		uint64 q2 = wide_sqr32(q);
		uint32 q2_w = montmul32_w((uint32)q2);
		uint64 q2_r = montmul64_r(q2);

		uint32 num_p_done = 0;
		time_t curr_time;
		double elapsed;

		while (num_p_done < num_p) {

			uint32 *plist = p_array->p + num_p_done;

			uint64 pinvlist[INVERT_BATCH_SIZE];
			uint32 curr_num_p = MIN(INVERT_BATCH_SIZE,
						num_p - num_p_done);

			batch_invert(plist, curr_num_p, pinvlist,
					q2, q2_r, q2_w);

			for (j = 0; j < curr_num_p; j++) {

				uint32 p = plist[j];
				uint64 pinv = pinvlist[j];
				uint32 lattice_size = 
					   p_array->lattice_size[num_p_done+j];

				for (k = 0; k < num_poly; k++) {

					uint64 qroot, proot, res;

					qroot = q_array->roots[k][i];
					proot = p_array->roots[k][num_p_done+j];
					res = montmul64(pinv, 
							modsub64(qroot, proot,
							q2), q2, q2_w);

					if (res < lattice_size) {
						handle_collision(L->poly, k,
								p, q, specialq,
								special_q_array->start_roots[k][which_special_q],
								proot + res * p * p);
					}
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
	}

	return 0;
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_deg5_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		sieve_fb_t *sieve_large_p1, sieve_fb_t *sieve_large_p2,
		uint32 special_q_min, uint32 special_q_max,
		uint32 large_p1_min, uint32 large_p1_max,
		uint32 large_p2_min, uint32 large_p2_max)
{
	uint32 i;
	uint32 min_large_p2, min_large_p1, min_special_q;
	uint32 quit = 0;
	p_soa_var_t * p_array;
	p_soa_var_t * q_array;
	p_soa_var_t * special_q_array;

	p_array = L->p_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	q_array = L->q_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	special_q_array = L->special_q_array = (p_soa_var_t *)xmalloc(
							sizeof(p_soa_var_t));

	printf("------- %u-%u %u-%u %u-%u\n",
			special_q_min, special_q_max,
			large_p2_min, large_p2_max,
			large_p1_min, large_p1_max);

	min_large_p1 = large_p1_min;
	sieve_fb_reset(sieve_large_p1, (uint64)large_p1_min, 
			(uint64)large_p1_max, 1, 1);

	while (min_large_p1 < large_p1_max) {

		L->fill_which_array = q_array;
		p_soa_var_reset(q_array);
		for (i = 0; i < HOST_BATCH_SIZE && 
				min_large_p1 != (uint32)P_SEARCH_DONE; i++) {
			min_large_p1 = sieve_fb_next(sieve_large_p1, L->poly,
						store_p_soa, L);
		}
		if (q_array->num_p == 0)
			goto finished;

		min_large_p2 = large_p2_min;
		sieve_fb_reset(sieve_large_p2, 
				(uint64)large_p2_min, (uint64)large_p2_max,
				1, 1);

		while (min_large_p2 < large_p2_max) {

			L->fill_which_array = p_array;
			p_soa_var_reset(p_array);
			for (i = 0; i < HOST_BATCH_SIZE && 
				   min_large_p2 != (uint32)P_SEARCH_DONE; i++) {
				min_large_p2 = sieve_fb_next(sieve_large_p2, L->poly,
							store_p_soa, L);
			}
			if (p_array->num_p == 0)
				goto finished;





			min_special_q = special_q_min;
			sieve_fb_reset(sieve_special_q,
				(uint64)special_q_min, (uint64)special_q_max,
				1, 1);

			while (min_special_q < special_q_max) {

				L->fill_which_array = special_q_array;
				p_soa_var_reset(special_q_array);
				for (i = 0; i < HOST_BATCH_SIZE &&
					    min_special_q != (uint32)P_SEARCH_DONE; i++) {
					min_special_q = sieve_fb_next(sieve_special_q, L->poly, store_p_soa, L);
				}
				if (special_q_array->num_p == 0)
					goto finished;

				for (i = 0; i < special_q_array->num_p; i++) {
					if (sieve_lattice_batch(obj, L, i)) {
						quit = 1;
						goto finished;
					}
				}
			}
		}
	}

finished:
	free(p_array);
	free(q_array);
	return quit;
}


