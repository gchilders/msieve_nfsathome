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

#ifndef _STAGE1_CORE_GPU_2PROG_H_
#define _STAGE1_CORE_GPU_2PROG_H_

#ifdef __CUDACC__
#include "cuda_intrinsics.h"
#define MAX_ROOTS 128
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* structure indicating a collision */

typedef struct {
	uint32 p;
	uint32 q;
	uint64 qroot;
	int64 offset;
} found_t;

#define Q_ARRAY_WORDS 1000

#define Q_PACKED_HEADER_WORDS 1

typedef struct {
	uint32 p;
	uint32 num_roots;
	uint64 roots[MAX_ROOTS];
} q_packed_t;

#define P_SOA_BATCH_SIZE (3*30*384)

typedef struct {
	uint32 p[P_SOA_BATCH_SIZE];
	uint64 roots[MAX_ROOTS][P_SOA_BATCH_SIZE];
} p_soa_t;

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_CORE_GPU_2PROG_H_ */
