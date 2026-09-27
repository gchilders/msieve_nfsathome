/*--------------------------------------------------------------------
NVTX instrumentation macros for the GPU lanczos path.

NVTX 3 is header-only and ships with the CUDA toolkit. Calls are no-ops
when no profiler is attached, so leaving the macros enabled in release
builds is safe.

Define DISABLE_NVTX at compile time to compile the macros to no-ops
unconditionally (useful for toolchains where the nvtx3 header is absent).
--------------------------------------------------------------------*/

#ifndef _COMMON_LANCZOS_GPU_LANCZOS_NVTX_H_
#define _COMMON_LANCZOS_GPU_LANCZOS_NVTX_H_

#if defined(HAVE_CUDA) && !defined(DISABLE_NVTX)

#include "nvtx3/nvToolsExt.h"

/* Distinct ARGB colors for each phase so a glance at the nsys timeline
   tells normal SpMV from transpose SpMV, dense rows from sparse, etc. */
#define LANCZOS_NVTX_COLOR_MUL          0xFF1E88E5u  /* blue   */
#define LANCZOS_NVTX_COLOR_MUL_TRANS    0xFFE53935u  /* red    */
#define LANCZOS_NVTX_COLOR_DENSE        0xFFFFB300u  /* amber  */
#define LANCZOS_NVTX_COLOR_SPMV_RUN     0xFF43A047u  /* green  */
#define LANCZOS_NVTX_COLOR_VV           0xFF8E24AAu  /* purple */
#define LANCZOS_NVTX_COLOR_KERNEL       0xFF00ACC1u  /* cyan   */

static inline void lanczos_nvtx_push(const char *name, uint32_t argb)
{
    nvtxEventAttributes_t attr;
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.category = 0;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = argb;
    attr.payloadType = NVTX_PAYLOAD_UNKNOWN;
    attr.payload.llValue = 0;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

#define LANCZOS_NVTX_PUSH(name, color) lanczos_nvtx_push((name), (color))
#define LANCZOS_NVTX_POP()             nvtxRangePop()

#else /* !HAVE_CUDA || DISABLE_NVTX */

#define LANCZOS_NVTX_PUSH(name, color) ((void)0)
#define LANCZOS_NVTX_POP()             ((void)0)

#endif

#endif /* !_COMMON_LANCZOS_GPU_LANCZOS_NVTX_H_ */
