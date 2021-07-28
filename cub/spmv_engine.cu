#include <stdio.h>
#include <vector>
#include <device_unaryspmv.cuh>

#include "spmv_engine.h"

#if defined(_WIN32) || defined (_WIN64)
	#define SPMV_ENGINE_DECL __declspec(dllexport)
#else
	#define SPMV_ENGINE_DECL __attribute__((visibility("default")))
#endif

using namespace cub;

typedef unsigned int uint32;

template<typename T>
struct v_bit_xor : public std::binary_function<T, T, T> {
        __host__ __device__ T operator()(T a, T b) { return v_xor(a, b); }
};

__device__ v_t operator+(const v_t& left, const v_t& right) {
	return v_xor(left, right);
	// return v_bit_xor<v_t>(left, right);
};

extern "C"
{

SPMV_ENGINE_DECL void 
spmv_engine_init(int which_gpu)
{
	
}

SPMV_ENGINE_DECL void 
spmv_engine_free(void)
{
	
}

SPMV_ENGINE_DECL void
spmv_engine_preprocess(spmv_data_t * data)
{
	
}

SPMV_ENGINE_DECL void 
spmv_engine_run(int preprocess_handle, spmv_data_t * data)
{
	void*    d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	cub::DeviceUnarySpmv::CsrMV(d_temp_storage, temp_storage_bytes, 
		(int *)data->row_entries, (int *)data->col_entries, (v_t *)data->vector_in, (v_t *)data->vector_out,
		data->num_rows, data->num_cols, data->num_col_entries, v_zero);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// Run SpMV
	cub::DeviceUnarySpmv::CsrMV(d_temp_storage, temp_storage_bytes,
		(int *)data->row_entries, (int *)data->col_entries, (v_t *)data->vector_in, (v_t *)data->vector_out,
		data->num_rows, data->num_cols, data->num_col_entries, v_zero);
		
	// Free temp storage
	cudaFree(d_temp_storage);
}

} // extern "C"
