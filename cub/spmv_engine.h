/* simple DSO interface for sparse matrix multiply */

#ifndef _SPMV_ENGINE_H_
#define _SPMV_ENGINE_H_

#include <stdlib.h>
#include <cuda.h>
#include "../common/lanczos/gpu/lanczos_gpu_core.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	int num_rows;
	uint32_t num_col_entries;
	CUdeviceptr vector_in;    /* v_t */
	CUdeviceptr vector_out;    /* v_t */
	CUdeviceptr col_entries;    /* uint32 */
	CUdeviceptr row_entries;    /* uint32 */
} spmv_data_t;

typedef void * (*spmv_engine_init_func)(int * vbits);

typedef void (*spmv_engine_free_func)(void * e);

typedef void (*spmv_engine_run_func)(void * e,
				spmv_data_t * spmv_data);

/* optional: kernel for the gather products (spmv_engine_run) */
enum {
	SPMV_KERNEL_AUTO = 0,	/* per block, from nonzeros per row */
	SPMV_KERNEL_WARPMERGE = 1,
	SPMV_KERNEL_SEGSCAN = 2
};
typedef void (*spmv_engine_set_kernel_func)(void * e, int kernel);

#ifdef __cplusplus
}
#endif

#endif /* !_SPMV_ENGINE_H_ */
