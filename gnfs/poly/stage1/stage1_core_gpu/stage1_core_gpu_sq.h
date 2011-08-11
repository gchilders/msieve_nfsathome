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

#ifndef _STAGE1_CORE_GPU_SQ_H_
#define _STAGE1_CORE_GPU_SQ_H_

#ifdef __CUDACC__
#include "../cuda_intrinsics.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPECIALQ_BATCH_SIZE 40

/* the outer loop needs parallel access to different p,
   so we store in SOA format. */

#define P_SOA_BATCH_SIZE 2048

typedef struct {
	uint32 p[P_SOA_BATCH_SIZE];
	uint64 start_root[P_SOA_BATCH_SIZE];
	uint64 roots[SPECIALQ_BATCH_SIZE][P_SOA_BATCH_SIZE];
	float lsize[SPECIALQ_BATCH_SIZE][P_SOA_BATCH_SIZE];
} pbatch_soa_t;

#define Q_SOA_BATCH_SIZE (3*30*384)

typedef struct {
	uint32 p[Q_SOA_BATCH_SIZE];
	uint64 start_root[Q_SOA_BATCH_SIZE];
	uint64 roots[SPECIALQ_BATCH_SIZE+1][Q_SOA_BATCH_SIZE];
	float lsize[SPECIALQ_BATCH_SIZE][Q_SOA_BATCH_SIZE];
} qbatch_soa_t;

#ifndef __CUDACC__
uint32 sieve_lattice_gpu_sq_core(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max);
#endif

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_CORE_GPU_SQ_H_ */
