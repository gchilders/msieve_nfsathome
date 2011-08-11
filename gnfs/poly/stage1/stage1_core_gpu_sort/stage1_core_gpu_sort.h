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

#ifndef _STAGE1_CORE_GPU_SORT_H_
#define _STAGE1_CORE_GPU_SORT_H_

#ifdef __CUDACC__
#include "../cuda_intrinsics.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_SPECIAL_Q ((uint32)(-1))
#define MAX_OTHER ((uint32)1 << 22)
#define SPECIAL_Q_SCALE 5

typedef struct {
	uint32 p1;
	uint32 p2;
	uint64 root;
} found_t;

#define NUM_GPU_FUNCTIONS 6
#define GPU_TRANS 0
#define GPU_STEP 1
#define GPU_SORT 2
#define GPU_MERGE 3
#define GPU_MERGE1 4
#define GPU_FINAL 5

#define SHARED_ELEM_SIZE (sizeof(uint32) + sizeof(uint64))

#ifndef __CUDACC__
uint32 sieve_lattice_gpu_sort_core(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max);
#endif

#ifdef __cplusplus
}

#endif

#endif /* !_STAGE1_CORE_GPU_SORT_H_ */
