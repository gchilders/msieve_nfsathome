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

#include "stage1_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_trans_32(uint32 *p_array, uint32 num_p, uint32 *start_roots,
			uint32 num_roots, uint32 *p_out, int32 *roots_out,
			specialq_t *q_batch, uint32 num_specialq, 
			uint32 num_entries, uint32 shift)
{
	uint32 offset, i, j, p, pp_w, q, end, gcd;
	uint32 pp, pp_r, inv, newroot;

	offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset >= num_p)
		return;

	p = p_array[offset];
	pp = p * p;
	pp_w = montmul32_w(pp);
	pp_r = montmul32_r(pp);
	end = num_p * num_roots;

	q = 0;
	for (i = 0; i < num_specialq; i++) {
		if (q != q_batch[i].p) {
			q = q_batch[i].p;
			gcd = gcd32(p, q);

			if (gcd == 1) {
				inv = modinv32(wide_sqr32(q) % pp, pp);
				inv = montmul32(inv, pp_r, pp, pp_w);
			}
		}

		for (j = offset; j < end; j += num_p) {

			if (gcd == 1) {
				newroot = modsub32(start_roots[j],
						q_batch[i].root % pp, pp);
				newroot = montmul32(newroot, inv, pp, pp_w);

				if (newroot > pp / 2)
					newroot -= pp;

				p_out[j + num_entries * i] = (i << shift) | p;
				roots_out[j + num_entries * i] = newroot;
			}
		}
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_trans_64(uint32 *p_array, uint32 num_p, uint64 *start_roots,
			uint32 num_roots, uint32 *p_out, int64 *roots_out,
			specialq_t *q_batch, uint32 num_specialq, 
			uint32 num_entries, uint32 shift)
{
	uint32 offset, i, j, p, pp_w, q, end, gcd;
	uint64 pp, pp_r, qq, tmp, inv, newroot;

	offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset >= num_p)
		return;

	p = p_array[offset];
	pp = wide_sqr32(p);
	pp_w = montmul32_w((uint32)pp);
	pp_r = montmul64_r(pp, pp_w);
	end = num_p * num_roots;

	q = 0;
	for (i = 0; i < num_specialq; i++) {
		if (q != q_batch[i].p) {
			q = q_batch[i].p;
			gcd = gcd32(p, q);

			if (gcd == 1) {
				qq = wide_sqr32(q) % pp;
				tmp = modinv32(q % p, p);
				tmp = wide_sqr32(tmp);
				tmp = montmul64(tmp, pp_r, pp, pp_w);
				inv = montmul64(qq, tmp, pp, pp_w);
				inv = modsub64((uint64)2, inv, pp);
				inv = montmul64(inv, tmp, pp, pp_w);
				inv = montmul64(inv, pp_r, pp, pp_w);
			}
		}

		for (j = offset; j < end; j += num_p) {

			if (gcd == 1) {
				newroot = modsub64(start_roots[j],
						q_batch[i].root % pp, pp);
				newroot = montmul64(newroot, inv, pp, pp_w);

				if (newroot > pp / 2)
					newroot -= pp;

				p_out[j + num_entries * i] = (i << shift) | p;
				roots_out[j + num_entries * i] = newroot;
			}
		}
	}
}

/*------------------------------------------------------------------------*/
__device__ void
store_hit(found_t *found_array, uint32 found_array_size,
		uint32 p1, uint32 p2,
		int64 root, specialq_t *q)
{
	/* don't use atomicInc because we don't want
	   wraparound to occur */

	uint32 index = atomicAdd(&found_array[0].p1, 1);

	if (index < found_array_size - 1) {

		found_t *f = found_array + index + 1;

		f->p1 = p1;
		f->p2 = p2;
		f->q = q->p;
		f->qroot = q->root;
		f->offset = root;
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_final_32(uint32 *p_array, int32 *roots, uint32 num_entries,
			specialq_t * q_batch, uint32 num_specialq, 
			found_t *found_array, uint32 shift)
{
	uint32 i, j;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 mask = (1 << shift) - 1;
	uint32 p_array_size = num_entries * num_specialq;

	for (i = my_threadid; i < p_array_size - 1; i += num_threads) {

		int32 root1 = roots[i];
		uint32 p1 = p_array[i];

		if (root1 == 0)
			continue;

		for (j = i + 1; j < p_array_size; j++) {
			int32 root2 = roots[j];
			uint32 p2 = p_array[j];

			if (root1 != root2)
				break;

			if (p1 >= p2 &&
			    (p1 >> shift) == (p2 >> shift) &&
			    gcd32( (p1 & mask), (p2 & mask) ) == 1) {

				store_hit(found_array, FOUND_ARRAY_SIZE,
						p1 & mask, p2 & mask, 
						(int64)root1,
						q_batch + (p1 >> shift));
			}
		}

		for (j = i - 1; (int32)j >= 0; j--) {
			int64 root2 = roots[j];
			uint32 p2 = p_array[j];

			if (root1 != root2)
				break;

			if (p1 >= p2 &&
			    (p1 >> shift) == (p2 >> shift) &&
			    gcd32( (p1 & mask), (p2 & mask) ) == 1) {

				store_hit(found_array, FOUND_ARRAY_SIZE,
						p1 & mask, p2 & mask, 
						(int64)root1,
						q_batch + (p1 >> shift));
			}
		}
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_final_64(uint32 *p_array, int64 *roots, uint32 num_entries,
			specialq_t * q_batch, uint32 num_specialq, 
			found_t *found_array, uint32 shift)
{
	uint32 i, j;
	uint32 num_threads = gridDim.x * blockDim.x;
	uint32 my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32 mask = (1 << shift) - 1;
	uint32 p_array_size = num_entries * num_specialq;

	for (i = my_threadid; i < p_array_size - 1; i += num_threads) {

		int64 root1 = roots[i];
		uint32 p1 = p_array[i];

		if (root1 == 0)
			continue;

		for (j = i + 1; j < p_array_size; j++) {
			int64 root2 = roots[j];
			uint32 p2 = p_array[j];

			if (root1 != root2)
				break;

			if (p1 >= p2 &&
			    (p1 >> shift) == (p2 >> shift) &&
			    gcd32( (p1 & mask), (p2 & mask) ) == 1) {

				store_hit(found_array, FOUND_ARRAY_SIZE,
						p1 & mask, p2 & mask, root1,
						q_batch + (p1 >> shift));
			}
		}

		for (j = i - 1; (int32)j >= 0; j--) {
			int64 root2 = roots[j];
			uint32 p2 = p_array[j];

			if (root1 != root2)
				break;

			if (p1 >= p2 &&
			    (p1 >> shift) == (p2 >> shift) &&
			    gcd32( (p1 & mask), (p2 & mask) ) == 1) {

				store_hit(found_array, FOUND_ARRAY_SIZE,
						p1 & mask, p2 & mask, root1,
						q_batch + (p1 >> shift));
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
