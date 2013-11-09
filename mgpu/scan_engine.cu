#include <stdio.h>
#include <moderngpu.cuh>

#include "scan_engine.h"

#if defined(_WIN32) || defined (_WIN64)
	#define SCAN_ENGINE_DECL __declspec(dllexport)
#else
	#define SCAN_ENGINE_DECL __attribute__((visibility("default")))
#endif

using namespace mgpu;

typedef ScanOp<ScanOpTypeXor, uint64> scan_func;

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
	scan_func func;

	Scan<MgpuScanTypeExc, uint64 *, uint64 *, scan_func> (
				(uint64 *)data->data_in,
				data->num_elements,
				(uint64 *)data->data_in,
				func,
				0,
				true,
				**ctxptr);
}

} // extern "C"
