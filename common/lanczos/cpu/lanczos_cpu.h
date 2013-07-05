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

#ifndef _COMMON_LANCZOS_CPU_LANCZOS_CPU_H_
#define _COMMON_LANCZOS_CPU_LANCZOS_CPU_H_

#include <thread.h>
#include "../lanczos.h"

#ifdef __cplusplus
extern "C" {
#endif

/* struct used by threads for computing partial
   matrix multiplies */

typedef struct {
	/* items for matrix-vector operations */

	uint64 *tmp_b;

	/* items for vector-vector operations */

	uint64 *x;
	uint64 *b;
	uint64 *y;
	uint32 vsize;

} thread_data_t;

typedef struct {
	struct packed_matrix_t *matrix;
	uint32 task_num;
	uint32 block_num;
} la_task_t;

/* implementation-specific structure */

typedef struct {

	uint64 *x;
	uint64 *b;

	struct threadpool *threadpool;
	thread_data_t thread_data[MAX_THREADS];
	la_task_t *tasks;
} cpudata_t;

/* for big jobs, we use a multithreaded framework that calls
   these routines for the heavy lifting */

void mul_packed_core(void *data, int thread_num);

void mul_packed_small_core(void *data, int thread_num);

void mul_trans_packed_core(void *data, int thread_num);

void mul_trans_packed_small_core(void *data, int thread_num);

/* internal stuff for vector-vector operations within the
   matrix multiply */

void mul_Nx64_64x64_acc(uint64 *v, uint64 *x, uint64 *y, uint32 n);

void mul_64xN_Nx64(uint64 *x, uint64 *y, uint64 *xy, uint32 n);

#ifdef __cplusplus
}
#endif

#endif /* !_COMMON_LANCZOS_CPU_LANCZOS_CPU_H_ */
