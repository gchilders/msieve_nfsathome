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

typedef struct {
	double bits; /* in leading rational coeff */
	double p_scale;
	uint32 num_pieces; /* for randomization */
	uint32 specialq_max;
	uint32 large_fb_max;
} sieve_fb_param_t;

static const sieve_fb_param_t sieve_fb_params[] = {

	{ 40.0, 1.5,  1,   100,   2000},
	{ 48.0, 1.3,  1,   200,   2000},
	{ 56.0, 1.3,  1,  1000,   5000},
	{ 64.0, 1.3,  1,  1000,   5000},
	{ 72.0, 1.2,  1,  1000,   5000},
	{ 80.0, 1.1,  1, 35000,  10000},
	{ 88.0, 1.1,  1, 35000,  10000},
	{ 96.0, 1.1,  1, 15000,  10000},
	{116.0, 1.1,  1, 15000,  10000},
	{128.0, 1.1,  1, 15000,  10000},
};

#define NUM_SIEVE_FB_PARAMS (sizeof(sieve_fb_params) / \
				sizeof(sieve_fb_params[0]))

typedef struct {
	uint64 offset;
	uint32 p;
} hash_entry_t;

typedef struct {
	uint64 start_offset;
	uint64 offset;
} hash_list_t;

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 p;
	uint32 num_roots;
	uint32 pad;
	uint32 mont_w;
	uint64 mont_r;
	uint64 p2;
	hash_list_t roots[MAX_ROOTS];
} p_packed_t;

#define P_PACKED_HEADER_WORDS 4

typedef struct {
	uint32 num_p;
	uint32 num_roots;
	uint32 p_size;
	uint32 p_size_alloc;
	p_packed_t *curr;
	p_packed_t *packed_array;
} p_packed_var_t;

static void 
p_packed_init(p_packed_var_t *s)
{
	memset(s, 0, sizeof(p_packed_var_t));

	s->p_size_alloc = 100;
	s->packed_array = s->curr = (p_packed_t *)xmalloc(s->p_size_alloc *
						sizeof(p_packed_t));
}

static void 
p_packed_free(p_packed_var_t *s)
{
	free(s->packed_array);
}

static p_packed_t * 
p_packed_next(p_packed_t *curr)
{
	return (p_packed_t *)((uint64 *)curr + 
			P_PACKED_HEADER_WORDS + 2 * curr->num_roots);
}

static void 
store_p_packed(uint64 p, uint32 num_roots, uint32 which_poly,
		mpz_t *roots, void *extra)
{
	uint32 i;
	p_packed_var_t *s = (p_packed_var_t *)extra;
	p_packed_t *curr;

	if (which_poly != 0) {
		printf("error: polynomial batches not supported\n");
		exit(-1);
	}

	if ((p_packed_t *)((uint64 *)s->curr + s->p_size) + 1 >=
			s->packed_array + s->p_size_alloc ) {

		s->p_size_alloc *= 2;
		s->packed_array = (p_packed_t *)xrealloc(
						s->packed_array,
						s->p_size_alloc *
						sizeof(p_packed_t));
		s->curr = (p_packed_t *)((uint64 *)s->packed_array + s->p_size);
	}

	curr = s->curr;
	curr->p = (uint32)p;
	curr->pad = 0;
	curr->num_roots = num_roots;
	for (i = 0; i < num_roots; i++)
		curr->roots[i].start_offset = gmp2uint64(roots[i]);
	curr->p2 = p * p;
	curr->mont_w = montmul32_w((uint32)curr->p2);
	curr->mont_r = montmul64_r(curr->p2);

	s->num_p++;
	s->num_roots += num_roots;
	s->curr = p_packed_next(s->curr);
	s->p_size = ((uint8 *)s->curr - 
			(uint8 *)s->packed_array) / sizeof(uint64);
}

/*------------------------------------------------------------------------*/
static void
handle_specialq_collision(poly_search_t *poly,
		uint32 p1, uint32 p2, uint32 special_q,
		uint64 special_q_root, uint64 res)
{
	curr_poly_t *c = poly->batch + 0;

	if (mp_gcd_1(p1, p2) != 1)
		return;

	mpz_set_ui(poly->p, (unsigned long)p1);
	mpz_mul_ui(poly->p, poly->p, (unsigned long)p2);
	mpz_mul_ui(poly->p, poly->p, (unsigned long)special_q);

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
		gmp_printf("hit %u %u %u %Zd\n", special_q, p1, p2, poly->m0);
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

	gmp_printf("hit %u %u %u %Zd\n", special_q, p1, p2, poly->m0);

	poly->callback(c->high_coeff, poly->p, poly->m0, 
			c->coeff_max, poly->callback_data);
}

/*------------------------------------------------------------------------*/
static void
handle_special_q(hashtable_t *hashtable, 
		p_packed_var_t *hash_array, lattice_fb_t *L, 
		uint32 special_q, uint64 special_q_root,
		uint64 block_size, uint64 *inv_array)
{
	uint32 i, j;
	p_packed_t *tmp;
	uint32 num_entries = hash_array->num_p;
	uint64 special_q2 = (uint64)special_q * special_q;
	uint64 sieve_size = 2 * L->poly->batch[0].sieve_size / special_q2;
	uint64 sieve_start = 0;
	uint32 num_blocks = 0;

	tmp = hash_array->packed_array;

	for (i = 0; i < num_entries; i++) {
		uint64 p2 = tmp->p2;
		uint32 num_roots = tmp->num_roots;
		uint32 p2_w = tmp->mont_w;
		uint64 qinv = inv_array[i];

		if (qinv == 0) {
			for (j = 0; j < num_roots; j++)
				tmp->roots[j].offset = (uint64)(-1);
			tmp = p_packed_next(tmp);
			continue;
		}

		for (j = 0; j < num_roots; j++) {
			uint64 proot = tmp->roots[j].start_offset;
			tmp->roots[j].offset = montmul64(modsub64(proot, 
							special_q_root, p2),
						qinv, p2, p2_w);
		}

		tmp = p_packed_next(tmp);
	}

	while (sieve_start < sieve_size) {
		uint64 sieve_end = sieve_start + MIN(block_size,
						sieve_size - block_size);

		tmp = hash_array->packed_array;
		hashtable_reset(hashtable);

		for (i = 0; i < num_entries; i++) {

			uint32 num_roots = tmp->num_roots;

			for (j = 0; j < num_roots; j++) {
				uint64 offset = tmp->roots[j].offset;

				if (offset < sieve_end) {
					hash_entry_t *hit;
					hash_entry_t curr_entry;
					uint32 already_seen = 0;

					curr_entry.offset = offset;

					hit = (hash_entry_t *)hashtable_find(
							hashtable, &curr_entry,
							NULL, &already_seen);

					if (already_seen) {
						handle_specialq_collision(
								L->poly,
								tmp->p, hit->p, 
								special_q,
								special_q_root,
							       	offset);
					}
					else {
						hit->p = tmp->p;
					}
					tmp->roots[j].offset = offset + tmp->p2;
				}
			}

			tmp = p_packed_next(tmp);
		}

		sieve_start = sieve_end;
		num_blocks++;
	}

//	printf("%u\n", num_blocks); 
}

/*------------------------------------------------------------------------*/
#define SPECIALQ_BATCH_SIZE 10

static void
batch_invert(uint32 *qlist, uint32 num_q, uint64 *invlist,
		uint32 p, uint64 p2_r, uint32 p2_w)
{
	uint32 i;
	uint64 q2[SPECIALQ_BATCH_SIZE];
	uint64 invprod;
	uint64 p2 = wide_sqr32(p);

	invlist[0] = invprod = wide_sqr32(qlist[0]);
	for (i = 1; i < num_q; i++) {
		q2[i] = wide_sqr32(qlist[i]);
		invlist[i] = invprod = montmul64(invprod, q2[i], p2, p2_w);
	}

	invprod = mp_modinv_2(invprod, p2);
	invprod = montmul64(invprod, p2_r, p2, p2_w);
	for (i = num_q - 1; i; i--) {
		invlist[i] = montmul64(invprod, invlist[i-1], p2, p2_w);
		invprod = montmul64(invprod, q2[i], p2, p2_w);
	}
	invlist[i] = invprod;
}

/*------------------------------------------------------------------------*/
static void
sieve_specialq_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_small, sieve_fb_t *sieve_large, 
		uint32 small_p_min, uint32 small_p_max, 
		uint32 large_p_min, uint32 large_p_max)
{
	uint32 i, j;
	p_packed_t *qptr;
	p_packed_var_t specialq_array;
	p_packed_var_t hash_array;
	hashtable_t hashtable;
	uint64 *invtable;
	uint32 num_p, num_q;
	uint32 num_q_done;
	uint64 block_size;

	printf("------- %u-%u %u-%u\n",
			small_p_min, small_p_max,
			large_p_min, large_p_max);

	p_packed_init(&specialq_array);
	p_packed_init(&hash_array);

	sieve_fb_reset(sieve_small, (uint64)small_p_min,
			(uint64)small_p_max, 1, MAX_ROOTS);
	while (sieve_fb_next(sieve_small, L->poly, 
			store_p_packed, &specialq_array) != P_SEARCH_DONE) {
		;
	}

	sieve_fb_reset(sieve_large, (uint64)large_p_min,
			(uint64)large_p_max, 1, MAX_ROOTS);
	while (sieve_fb_next(sieve_large, L->poly, 
			store_p_packed, &hash_array) != P_SEARCH_DONE) {
		;
	}

	num_p = hash_array.num_p;
	num_q = specialq_array.num_p;
#if 1
	printf("special q: %u entries, %u roots\n", num_q, 
					specialq_array.num_roots);
	printf("aprogs: %u entries, %u roots\n", num_p, 
					hash_array.num_roots);
#endif

	hashtable_init(&hashtable, 3, 2);

	invtable = (uint64 *)xmalloc(num_p * SPECIALQ_BATCH_SIZE * 
					sizeof(uint64));
	num_q_done = 0;
	block_size = (uint64)large_p_min * large_p_min;
	qptr = specialq_array.packed_array;

	while (num_q_done < num_q) {

		p_packed_t *tmp;
		uint64 *invtmp;
		mp_t qprod;
		uint32 batch_q[SPECIALQ_BATCH_SIZE];
		uint64 batch_q_inv[SPECIALQ_BATCH_SIZE];
		uint32 qbatch_size = MIN(SPECIALQ_BATCH_SIZE,
					num_q - num_q_done);

		mp_clear(&qprod);
		qprod.nwords = qprod.val[0] = 1;

		for (i = 0, tmp = qptr; i < qbatch_size; i++) {
			batch_q[i] = tmp->p;
			mp_mul_1(&qprod, tmp->p, &qprod);
			tmp = p_packed_next(tmp);
		}

		for (i = 0, tmp = hash_array.packed_array; i < num_p; i++) {

			if (mp_gcd_1(mp_mod_1(&qprod, tmp->p), tmp->p) == 1)
				batch_invert(batch_q, qbatch_size, 
						batch_q_inv, tmp->p, 
						tmp->mont_r, tmp->mont_w);
			else
				memset(batch_q_inv, 0, sizeof(batch_q_inv));

			invtmp = invtable + i;
			for (j = 0; j < qbatch_size; j++) {
				*invtmp = batch_q_inv[j];
				invtmp += num_p;
			}
			tmp = p_packed_next(tmp);
		}

		invtmp = invtable;
		for (i = 0; i < qbatch_size; i++) {

			for (j = 0; j < qptr->num_roots; j++) {
				handle_special_q(&hashtable, &hash_array, L, 
						qptr->p, 
						qptr->roots[j].start_offset,
						block_size, invtmp);
			}

			qptr = p_packed_next(qptr);
			invtmp += num_p;
		}

		num_q_done += qbatch_size;
	}

#if 1
	printf("hashtable: %u entries, %5.2lf MB\n", 
			hashtable_get_num(&hashtable),
			(double)hashtable_sizeof(&hashtable) / 1048576);
#endif
	free(invtable);
	hashtable_free(&hashtable);
	p_packed_free(&specialq_array);
	p_packed_free(&hash_array);
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
sieve_lattice_hashtable(msieve_obj *obj, poly_search_t *poly, 
			uint32 deadline)
{
	lattice_fb_t L;
	sieve_fb_t sieve_small, sieve_large;
	uint32 small_p_min, small_p_max;
	uint64 large_p_min, large_p_max;
	uint64 large_p_min2, large_p_max2;
	uint32 specialq_max, large_fb_max;
	double p_scale;
	uint32 num_pieces;
	sieve_fb_param_t params;

	double p_size_max = poly->batch[0].p_size_max;
	double sieve_size = poly->batch[0].sieve_size;
	uint32 degree = poly->degree;
	double bits = log(sieve_size) / M_LN2;

	printf("p = %.2lf sieve = %.2lf bits\n", 
			log(p_size_max) / M_LN2, bits);

	get_poly_params(bits, &params);

	p_scale = params.p_scale;
	specialq_max = params.specialq_max;
	large_fb_max = params.large_fb_max;
	num_pieces = params.num_pieces;

	small_p_min = MAX(7, specialq_max / p_scale);
	small_p_max = specialq_max;
	large_p_min = sqrt(p_size_max / small_p_max);
	large_p_max = p_scale * large_p_min;

	gmp_printf("coeff %Zd %u %u %" PRIu64 " %" PRIu64 "\n",
			poly->batch[0].high_coeff,
			small_p_min, small_p_max,
			large_p_min, large_p_max);

	sieve_fb_init(&sieve_small, poly, 
			5, 100, 1, degree, 1, 0);
	sieve_fb_init(&sieve_large, poly, 
			101, large_fb_max, 1, degree, 1, 0);

	L.poly = poly;
	L.start_time = time(NULL);
	L.deadline = deadline;

	large_p_min2 = large_p_min;
	large_p_max2 = large_p_max;
	if (num_pieces > 1) {
		uint32 piece = get_rand(&obj->seed1, 
					&obj->seed2) % num_pieces;
		large_p_min2 = large_p_min + piece *
					((large_p_max - large_p_min) /
					num_pieces);
		large_p_max2 = large_p_min + (piece + 1) *
					((large_p_max - large_p_min) /
					num_pieces);
	}

	if (large_p_max2 < ((uint64)1 << 32)) {
		sieve_specialq_64(obj, &L,
			&sieve_small, &sieve_large,
			small_p_min, small_p_max,
			(uint32)large_p_min2, 
			(uint32)large_p_max2);
	}

	sieve_fb_free(&sieve_small);
	sieve_fb_free(&sieve_large);
}
