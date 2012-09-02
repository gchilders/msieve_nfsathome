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
			uint32 num_roots, uint32 *p_out, uint32 *roots_out,
			specialq_t *q_batch, uint32 num_specialq, 
			uint32 specialq_block, uint32 num_entries, uint32 shift)
{
	uint32 p, pp, pp_w, p_offset;
	uint32 specialq_start, specialq_end;
	uint32 q, qq_prod, qq_prod_offset, curr_offset, q_count;
	uint32 i, j, k, m, start_i, gcd, inv, curr_inv;
	uint32 qroot, newroot;
	specialq_t *curr_q;

	p_offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (p_offset >= num_p)
		return;

	p = p_array[p_offset];
	pp = p * p;
	pp_w = montmul32_w(pp);

	specialq_start = blockIdx.y * specialq_block;
	specialq_end = __min(specialq_start + specialq_block, num_specialq);

	qq_prod_offset = specialq_start * num_entries + p_offset;
	curr_q = q_batch + specialq_start;

	q = j = 0;
	for (i = specialq_start; j == 0 && i < specialq_end; i++) {

		if (q != curr_q->p) {
			q = curr_q->p;
			gcd = gcd32(p, q);
			if (gcd == 1)
				j = qq_prod = curr_q->pp % pp;
		}

		roots_out[qq_prod_offset] = j;
		qq_prod_offset += num_entries;
		curr_q++;
	}
	if (j == 0)
		return;

	for (start_i = i - 1; i < specialq_end; i++) {

		if (q != curr_q->p) {
			q = curr_q->p;
			gcd = gcd32(p, q);

			if (gcd == 1)
				j = qq_prod = montmul32(qq_prod, 
						curr_q->pp % pp, 
						pp, pp_w);
			else
				j = 0;
		}

		roots_out[qq_prod_offset] = j;
		qq_prod_offset += num_entries;
		curr_q++;
	}

	inv = modinv32(qq_prod, pp);
	inv = montmul32(inv, montmul32_r(pp), pp, pp_w);
	qq_prod_offset -= num_entries;

	for (i--; i > start_i; i--) {

		uint32 curr_qq_prod = roots_out[qq_prod_offset];

		if (curr_qq_prod > 0)
			break;

		qq_prod_offset -= num_entries;
	}

	curr_offset = qq_prod_offset - num_entries;
	q = i;
	q_count = 1;

	for (i--; (int32)i >= (int32)start_i; 
			i--, curr_offset -= num_entries) {

		uint32 curr_qq_prod = roots_out[curr_offset];

		if (curr_qq_prod == 0) {
			continue;
		}
		else if (curr_qq_prod == qq_prod) {
			q_count++;
			continue;
		}

		curr_inv = montmul32(curr_qq_prod, inv, pp, pp_w);
		inv = montmul32(inv, q_batch[q].pp % pp, pp, pp_w);

		do {
			qroot = q_batch[q].root % pp;

			for (j = qq_prod_offset, k = p_offset, m = 0; 
						m < num_roots; 
						j += num_p, k += num_p, m++) {

				newroot = modsub32(start_roots[k], 
							qroot, pp);
				newroot = montmul32(newroot, curr_inv, 
							pp, pp_w);
				if (newroot > pp / 2)
					newroot -= pp;

				p_out[j] = (q << shift) | p;
				roots_out[j] = newroot;
			}

			q--;
			qq_prod_offset -= num_entries;
		} while (--q_count);

		q = i;
		q_count = 1;
		qq_prod = curr_qq_prod;
		qq_prod_offset = curr_offset;
	}

	curr_inv = inv;
	while ((int32)q >= (int32)start_i) {

		qroot = q_batch[q].root % pp;

		for (j = qq_prod_offset, k = p_offset, m = 0; 
					m < num_roots; 
					j += num_p, k += num_p, m++) {

			newroot = modsub32(start_roots[k], 
						qroot, pp);
			newroot = montmul32(newroot, curr_inv, 
						pp, pp_w);
			if (newroot > pp / 2)
				newroot -= pp;

			p_out[j] = (q << shift) | p;
			roots_out[j] = newroot;
		}

		q--;
		qq_prod_offset -= num_entries;
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_trans_64(uint32 *p_array, uint32 num_p, uint64 *start_roots,
			uint32 num_roots, uint32 *p_out, int64 *roots_out,
			specialq_t *q_batch, uint32 num_specialq, 
			uint32 specialq_block, uint32 num_entries, uint32 shift)
{
	uint32 offset, i, j, p, pp_w, q, end, gcd;
	uint64 pp, pp_r, qq, tmp, inv, newroot;
	uint32 specialq_start, specialq_end;

	offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset >= num_p)
		return;

	p = p_array[offset];
	pp = wide_sqr32(p);
	pp_w = montmul32_w((uint32)pp);
	pp_r = montmul64_r(pp, pp_w);
	end = num_p * num_roots;

	specialq_start = blockIdx.y * specialq_block;
	specialq_end = __min(specialq_start + specialq_block, num_specialq);

	q = 0;
	for (i = specialq_start; i < specialq_end; i++) {
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
