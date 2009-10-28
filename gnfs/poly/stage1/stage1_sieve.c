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

#define MAX_P ((uint64)(-1))

/*------------------------------------------------------------------------*/
static void 
lattice_fb_init(lattice_fb_t *L, poly_search_t *poly, 
		uint32 deadline)
{
	L->poly = poly;
	L->start_time = time(NULL);
	L->deadline = deadline;
	L->num_tests = 0;
}

/*------------------------------------------------------------------------*/
void
handle_collision(poly_search_t *poly, uint32 which_poly,
		uint64 p, uint128 proot, uint128 res, uint64 q)
{
	curr_poly_t *c = poly->batch + which_poly;

	uint64_2gmp(p, poly->tmp1);
	uint64_2gmp(q, poly->tmp2);
	mpz_import(poly->tmp3, 4, -1, sizeof(uint32), 0, 0, &proot);
	mpz_import(poly->tmp4, 4, -1, sizeof(uint32), 0, 0, &res);

	mpz_mul(poly->p, poly->tmp1, poly->tmp2);
	mpz_mul(poly->tmp1, poly->tmp1, poly->tmp1);
	mpz_addmul(poly->tmp3, poly->tmp1, poly->tmp4);
	mpz_sub(poly->tmp3, poly->tmp3, c->mp_sieve_size);
	mpz_add(poly->m0, c->trans_m0, poly->tmp3);

#ifdef CHECK
	gmp_printf("poly %2u p %.0lf q %.0lf coeff %Zd\n", 
			which_poly, (double)p, (double)q, poly->p);

	mpz_pow_ui(poly->tmp1, poly->m0, (mp_limb_t)poly->degree);
	mpz_mul(poly->tmp2, poly->p, poly->p);
	mpz_sub(poly->tmp1, c->trans_N, poly->tmp1);
	mpz_tdiv_r(poly->tmp3, poly->tmp1, poly->tmp2);
	if (mpz_cmp_ui(poly->tmp3, (mp_limb_t)0)) {
		printf("crap\n");
	return;//	exit(-1);
	}
#endif

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

	poly->callback(c->high_coeff, poly->p, poly->m0, 
			c->coeff_max, poly->callback_data);
}

/*------------------------------------------------------------------------*/
void
sieve_lattice(msieve_obj *obj, poly_search_t *poly, 
		uint32 small_fb_max, uint32 large_fb_min, 
		uint32 large_fb_max, gpu_info_t *gpu_info,
		uint32 deadline)
{
	lattice_fb_t L;
	sieve_fb_t sieve_small, sieve_large;
	uint64 small_p_min, small_p_max;
	uint64 large_p_min, large_p_max;
	uint32 bits;
	double p_scale = 1.1;
	uint32 degree = poly->degree;
	uint32 max_roots;
	curr_poly_t *middle_poly;
	curr_poly_t *last_poly;

	CUcontext gpu_context;
	CUmodule gpu_module64;
	CUmodule gpu_module96;
	CUmodule gpu_module128;
	CUfunction gpu_kernel64;
	CUfunction gpu_kernel96;
	CUfunction gpu_kernel128;

	middle_poly = poly->batch + poly->num_poly / 2;
	last_poly = poly->batch + poly->num_poly - 1;

	if (middle_poly->p_size_max >= (double)MAX_P * MAX_P) {
		printf("error: rational leading coefficient is "
			"too large at %le (%0.3f bits)\n",
	 		 middle_poly->p_size_max, 
			 log(middle_poly->p_size_max)/ M_LN2);
		exit(-1);
	}

	CUDA_TRY(cuCtxCreate(&gpu_context, 
			CU_CTX_BLOCKING_SYNC,
			gpu_info->device_handle))

	bits = mpz_sizeinbase(poly->N, 2);
	switch (degree) {
	case 4:
		p_scale = 1.3;
		CUDA_TRY(cuModuleLoad(&gpu_module64, 
				"stage1_core_deg46_64.ptx"))
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel64, 
				gpu_module64, 
				"sieve_kernel_64"))
		break;

	case 5:
		if (bits < 363)
			p_scale = 1.3;
		else if (bits < 396)
			p_scale = 1.2;

		CUDA_TRY(cuModuleLoad(&gpu_module64, 
				"stage1_core_deg5_64.ptx"))
		CUDA_TRY(cuModuleLoad(&gpu_module96, 
				"stage1_core_deg5_96.ptx"))
		CUDA_TRY(cuModuleLoad(&gpu_module128, 
				"stage1_core_deg5_128.ptx"))

		CUDA_TRY(cuModuleGetFunction(&gpu_kernel64, 
				gpu_module64, 
				"sieve_kernel_64"))
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel96, 
				gpu_module96, 
				"sieve_kernel_96"))
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel128, 
				gpu_module128, 
				"sieve_kernel_128"))
		break;

	case 6:
		if (bits < 512)
			p_scale = 1.2;
		else if (bits < 600)
			p_scale = 1.1;
		else
			p_scale = 1.03;

		CUDA_TRY(cuModuleLoad(&gpu_module64, 
				"stage1_core_deg46_64.ptx"))
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel64, 
				gpu_module64, 
				"sieve_kernel_64"))
		break;
	}


	large_p_min = sqrt(middle_poly->p_size_max);
	if (large_p_min >= MAX_P / p_scale)
		large_p_max = MAX_P - 1;
	else
		large_p_max = p_scale * large_p_min;

	small_p_min = large_p_min / p_scale;
	small_p_max = large_p_min - 1;

	gmp_printf("coeff %Zd-%Zd"
		   " %" PRIu64 " %" PRIu64 
		   " %" PRIu64 " %" PRIu64 "\n",
			poly->batch[0].high_coeff,
			last_poly->high_coeff,
			small_p_min, small_p_max,
			large_p_min, large_p_max);

	max_roots = degree;
	if (degree == 5)
		max_roots = 1;

	sieve_fb_init(&sieve_small, poly, 
			5, small_fb_max, 
			1, max_roots);
	sieve_fb_init(&sieve_large, poly, 
			large_fb_min, large_fb_max, 
			1, max_roots);
	lattice_fb_init(&L, poly, deadline);

	while (1) {
		uint32 done = 1;

		if (degree == 5) {
			if (large_p_max < ((uint64)1 << 32)) {
				done = sieve_lattice_gpu_deg5_64(obj, &L,
					&sieve_small, &sieve_large,
					(uint32)small_p_min, 
					(uint32)small_p_max,
					(uint32)large_p_min, 
					(uint32)large_p_max,
					gpu_info, gpu_kernel64);
			}
			else if (large_p_max < ((uint64)1 << 48)) {
				done = sieve_lattice_gpu_deg5_96(obj, &L,
					&sieve_small, &sieve_large,
					small_p_min, small_p_max,
					large_p_min, large_p_max,
					gpu_info, gpu_kernel96);
			}
			else {
				done = sieve_lattice_gpu_deg5_128(obj, &L,
					&sieve_small, &sieve_large,
					small_p_min, small_p_max,
					large_p_min, large_p_max,
					gpu_info, gpu_kernel128);
			}
		}
		else {	/* degree 4 or 6 */
			if (large_p_max < ((uint64)1 << 32)) {
				CUDA_TRY(cuModuleGetGlobal(&L.gpu_p_array, 
						NULL, gpu_module64, "pbatch"))
				done = sieve_lattice_gpu_deg46_64(obj, &L,
					&sieve_small, &sieve_large,
					(uint32)small_p_min, 
					(uint32)small_p_max,
					(uint32)large_p_min, 
					(uint32)large_p_max,
					gpu_info, gpu_kernel64);
			}
		}

		if (done)
			break;

		small_p_max = small_p_min - 1;
		small_p_min = small_p_min / p_scale;
		if (small_p_min < small_fb_max)
			break;

		if (middle_poly->p_size_max / small_p_max >= MAX_P)
			break;
		large_p_min = middle_poly->p_size_max / small_p_max;

		if (middle_poly->p_size_max / small_p_min >= MAX_P)
			large_p_max = MAX_P - 1;
		else
			large_p_max = middle_poly->p_size_max / small_p_min;
	}

	CUDA_TRY(cuCtxDestroy(gpu_context)) 
	sieve_fb_free(&sieve_small);
	sieve_fb_free(&sieve_large);
}
