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

typedef unsigned int uint32;
typedef unsigned long long uint64;

#define P_SMALL_BATCH_SIZE 1024

typedef struct {
	uint32 lattice_size[P_SMALL_BATCH_SIZE];
	uint32 p[P_SMALL_BATCH_SIZE];
	uint64 roots[P_SMALL_BATCH_SIZE];
} p_small_batch_t;

#define P_LARGE_BATCH_SIZE 16384

typedef struct {
	uint32 p[P_LARGE_BATCH_SIZE];
	uint64 roots[P_LARGE_BATCH_SIZE];
} p_large_batch_t;

typedef struct {
	uint32 p;
	uint32 q;
	uint64 offset;
	uint64 proot;
} found_t;

/*------------------------------------------------------------------------*/
__device__ uint64 
modsub(uint64 a, uint64 b, uint64 p) {

	uint64 t = 0, tr;
	tr = a - b;
	if (tr > a)
		t = p;
	return tr + t;
}

/*------------------------------------------------------------------------*/
__device__ uint64 
modinv(uint64 a, uint64 p) {

	uint64 ps1, ps2, dividend, divisor, rem, q, t;
	uint32 parity;

	q = 1; rem = a; dividend = p; divisor = a;
	ps1 = 1; ps2 = 0; parity = 0;

	while (divisor > 1) {
		rem = dividend - divisor;
		t = rem - divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t;
		if (rem >= divisor) {
			q = dividend / divisor;
			rem = dividend - q * divisor;
			q *= ps1;
		} } } } } } } } }

		q += ps2;
		parity = ~parity;
		dividend = divisor;
		divisor = rem;
		ps2 = ps1;
		ps1 = q;
	}
	
	if (parity == 0)
		return ps1;
	else
		return p - ps1;
}

/*------------------------------------------------------------------------*/
__device__ uint32 
montmul_w(uint32 n) {

	uint32 res = 2 + n;
	res = res * (2 + n * res);
	res = res * (2 + n * res);
	res = res * (2 + n * res);
	return res * (2 + n * res);
}

/*------------------------------------------------------------------------*/
__device__ uint64 
montmul(uint64 a, uint64 b,
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
__device__ uint64 
montmul_r(uint64 n, uint32 w) {

	uint32 shift;
	uint32 i;
	uint64 shifted_n;
	uint64 res;

	shift = __clzll(n);
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
#ifdef __cplusplus
extern "C" {
#endif

__global__ void
sieve_kernel_p1xq1(p_small_batch_t *pbatch, 
                 uint32 num_p,
		 p_large_batch_t *qbatch, 
		 uint32 num_q,
		 found_t *found_array)
{
	uint32 my_threadid;
	uint32 num_threads;
	uint32 i, j;
	found_t my_found;

	my_found.p = 0;
	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	num_threads = gridDim.x * blockDim.x;

	for (i = my_threadid; i < num_q; i += num_threads) {
		uint32 q = qbatch->p[i];
		uint64 q2 = (uint64)q * q;
		uint32 q2_w = montmul_w((uint32)q2);
		uint64 q2_r = montmul_r(q2, q2_w);
		uint64 qroot = qbatch->roots[i];

		for (j = 0; j < num_p; j++) {
			uint32 p = pbatch->p[j];
			uint64 p2 = (uint64)p * p;
			uint32 lattice_size = pbatch->lattice_size[j];
			uint64 pinv = modinv(p2, q2);
			uint64 proot = pbatch->roots[i];
			uint64 res;

			pinv = montmul(pinv, q2_r, q2, q2_w);

			res = montmul(pinv, modsub(qroot, proot, q2),
					q2, q2_w);

			if (res < lattice_size ||
			    res >= q2 - lattice_size) {
				my_found.p = p;
				my_found.q = q;
				my_found.offset = res;
				my_found.proot = proot;
			}
		}
	}

	if (my_found.p > 0)
		found_array[my_threadid] = my_found;
}

#ifdef __cplusplus
}
#endif
