/* simple DSO interface for prefix scan; values are
   32-bit unsigned integers */

#ifndef _SCAN_ENGINE_H_
#define _SCAN_ENGINE_H_

#include <stdlib.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	CUdeviceptr data_in;
	int num_elements;
} scan_data_t;

typedef void * (*scan_engine_init_func)(int which_gpu);

typedef void (*scan_engine_free_func)(void * engine);

typedef void (*scan_engine_run_func)(void * engine, 
				scan_data_t * scan_data);

#ifdef __cplusplus
}
#endif

#endif /* !_SCAN_ENGINE_H_ */
