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

#ifndef _STAGE1_CORE_GPU_COMMON_H_
#define _STAGE1_CORE_GPU_COMMON_H_

#ifdef __CUDACC__
#include "../cuda_intrinsics.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* structure indicating a collision */

typedef struct {
	uint32 p;
	uint32 q;
	uint32 k;
	uint64 offset;
	uint64 proot;
} found_t;

#define MAX_SPECIAL_Q ((uint32)(-1))
#define MAX_OTHER ((uint32)(-1))
#define SPECIAL_Q_SCALE 16

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_CORE_GPU_COMMON_H_ */
