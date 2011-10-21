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

#include "stage1_core_2prog.h"

#ifdef __cplusplus
extern "C" {
#endif

__constant__ uint64 qbatch[Q_ARRAY_WORDS];

/*------------------------------------------------------------------------*/
__device__ q_packed_t *
q_packed_next(q_packed_t *curr)
{
	return (q_packed_t *)((uint64 *)curr + 
			Q_PACKED_HEADER_WORDS + curr->num_roots);
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_48(p_soa_t *pbatch, uint32 num_p, uint32 num_roots,
	     uint32 num_q, float bound, found_t *found_array)
{
	uint32 i, j, k, m, p, pp_w, q, num_qroots;
	uint32 my_threadid, num_threads;
	uint64 pp, pp_r, qq, proot, tmp, inv;
	int64 offset;
	float sieve_size;
	q_packed_t *curr_q;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	num_threads = gridDim.x * blockDim.x;
	found_array[my_threadid].p = 0;

	for (i = my_threadid; i < num_p; i += num_threads) {
		p = pbatch->p[i];
		pp = wide_sqr32(p);
		pp_w = montmul24_w((uint32)pp);
		pp_r = montmul48_r(pp, pp_w);
		curr_q = (q_packed_t *)qbatch;
		sieve_size = bound * pp;

		for (j = 0; j < num_q; j++) {
			q = curr_q->p;
			qq = wide_sqr32(q);
			num_qroots = curr_q->num_roots;

			tmp = modinv32(q, p);
			tmp = wide_sqr32(tmp);
			tmp = montmul48(tmp, pp_r, pp, pp_w);
			inv = montmul48(qq, tmp, pp, pp_w);
			inv = modsub64((uint64)2, inv, pp);
			inv = montmul48(inv, tmp, pp, pp_w);
			inv = montmul48(inv, pp_r, pp, pp_w);

			for (k = 0; k < num_roots; k++) {

				proot = pbatch->roots[k][i];

				for (m = 0; m < num_qroots; m++) {

					offset = modsub64(proot,
							curr_q->roots[m], pp);
					offset = montmul48(offset,
							inv, pp, pp_w);

					if (offset < sieve_size ||
					    offset > pp - sieve_size) {

						found_t *f = found_array +
								my_threadid;

						if (offset > sieve_size)
							offset -= pp;

						f->p = p;
						f->q = q;
						f->qroot = curr_q->roots[m];
						f->offset = offset;
					}
				}
			}

			curr_q = q_packed_next(curr_q);
		}
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_64(p_soa_t *pbatch, uint32 num_p, uint32 num_roots,
	     uint32 num_q, float bound, found_t *found_array)
{
	uint32 i, j, k, m, p, pp_w, q, num_qroots;
	uint32 my_threadid, num_threads;
	uint64 pp, pp_r, qq, proot, tmp, inv;
	int64 offset;
	float sieve_size;
	q_packed_t *curr_q;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	num_threads = gridDim.x * blockDim.x;
	found_array[my_threadid].p = 0;

	for (i = my_threadid; i < num_p; i += num_threads) {
		p = pbatch->p[i];
		pp = wide_sqr32(p);
		pp_w = montmul32_w((uint32)pp);
		pp_r = montmul64_r(pp, pp_w);
		curr_q = (q_packed_t *)qbatch;
		sieve_size = bound * pp;

		for (j = 0; j < num_q; j++) {
			q = curr_q->p;
			qq = wide_sqr32(q);
			num_qroots = curr_q->num_roots;

			tmp = modinv32(q, p);
			tmp = wide_sqr32(tmp);
			tmp = montmul64(tmp, pp_r, pp, pp_w);
			inv = montmul64(qq, tmp, pp, pp_w);
			inv = modsub64((uint64)2, inv, pp);
			inv = montmul64(inv, tmp, pp, pp_w);
			inv = montmul64(inv, pp_r, pp, pp_w);

			for (k = 0; k < num_roots; k++) {

				proot = pbatch->roots[k][i];

				for (m = 0; m < num_qroots; m++) {

					offset = modsub64(proot,
							curr_q->roots[m], pp);
					offset = montmul64(offset,
							inv, pp, pp_w);

					if (offset < sieve_size ||
					    offset > pp - sieve_size) {

						found_t *f = found_array +
								my_threadid;

						if (offset > sieve_size)
							offset -= pp;

						f->p = p;
						f->q = q;
						f->qroot = curr_q->roots[m];
						f->offset = offset;
					}
				}
			}

			curr_q = q_packed_next(curr_q);
		}
	}
}

#ifdef __cplusplus
}
#endif
