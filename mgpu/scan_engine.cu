#include <stdio.h>
#include <moderngpu.cuh>

#include "scan_engine.h"

#if defined(_WIN32) || defined (_WIN64)
	#define SCAN_ENGINE_DECL __declspec(dllexport)
#else
	#define SCAN_ENGINE_DECL __attribute__((visibility("default")))
#endif

using namespace mgpu;

extern "C"
{

SCAN_ENGINE_DECL void * 
scan_engine_init(int which_gpu)
{
	ContextPtr ctxptr = CreateCudaDevice(which_gpu);

	return new ContextPtr(ctxptr);
}

SCAN_ENGINE_DECL void 
scan_engine_free(void * e)
{
	delete (ContextPtr *)e;
}

SCAN_ENGINE_DECL void 
scan_engine_run(void * e, scan_data_t * data)
{
	ContextPtr *ctxptr = (ContextPtr *)e;

	Scan<MgpuScanTypeExc, uint64 *, uint64, bit_xor<uint64>, uint64 * > (
				(uint64 *)data->data_in,
				data->num_elements,
				(uint64)0,
				bit_xor<uint64>(),
				(uint64 *)data->data_in + data->num_elements,
				0,
				(uint64 *)data->data_in,
				**ctxptr);
}

} // extern "C"
