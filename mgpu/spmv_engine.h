/* simple DSO interface for sparse matrix multiply */

#ifndef _SPMV_ENGINE_H_
#define _SPMV_ENGINE_H_

#include <stdlib.h>
#include <cuda.h>
#include "../common/lanczos/gpu/lanczos_gpu_core.h"

template<typename T>
struct v_bit_and : public std::binary_function<T, T, T> {
        MGPU_HOST_DEVICE T operator()(T a, T b) { return v_and(a, b); }
};

template<typename T>
struct v_bit_xor : public std::binary_function<T, T, T> {
        MGPU_HOST_DEVICE T operator()(T a, T b) { return v_xor(a, b); }
};

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	int num_rows;
	int num_col_entries;
	CUdeviceptr vector_in;    /* v_t */
	CUdeviceptr vector_out;    /* v_t */
	CUdeviceptr col_entries;    /* uint32 */
	CUdeviceptr row_entries;    /* uint32 */
} spmv_data_t;

typedef void (*spmv_engine_init_func)(int which_gpu);

typedef void (*spmv_engine_free_func)(void);

typedef int (*spmv_engine_preprocess_func)(spmv_data_t * spmv_data);

typedef void (*spmv_engine_run_func)(int spmv_preprocess,
				spmv_data_t * spmv_data);

#ifdef __cplusplus
}
#endif

#endif /* !_SPMV_ENGINE_H_ */
