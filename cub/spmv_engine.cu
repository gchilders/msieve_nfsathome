// CSR SpMV over XOR (merge-path style partitioning)

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include "spmv_engine.h"

#ifndef DISABLE_NVTX
#include "nvtx3/nvToolsExt.h"
static inline void spmv_nvtx_push(const char *name) {
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = 0xFF43A047u;  /* green: matches host-side spmv_engine_run */
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}
#define SPMV_NVTX_PUSH(name) spmv_nvtx_push(name)
#define SPMV_NVTX_POP()      nvtxRangePop()
#else
#define SPMV_NVTX_PUSH(name) ((void)0)
#define SPMV_NVTX_POP()      ((void)0)
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__device__ __forceinline__ v_t vt_zero() {
    v_t z;
    #pragma unroll
    for (int i = 0; i < VWORDS; ++i) z.w[i] = 0ULL;
    return z;
}

__device__ __forceinline__ void vt_xor_inplace(v_t &a, const v_t &b) {
    #pragma unroll
    for (int i = 0; i < VWORDS; ++i) a.w[i] ^= b.w[i];
}

__device__ __forceinline__ void vt_store_xor(v_t* out, const v_t &val) {
    // XOR-accumulate "val" into *out using 64-bit atomics (row may be shared)
    unsigned long long* dst = out->w;
    #pragma unroll
    for (int i = 0; i < VWORDS; ++i) {
        atomicXor(&dst[i], val.w[i]);
    }
}

// Warp-wide XOR reduction of v_t
__device__ __forceinline__ v_t warp_xor_reduce(v_t v) {
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int i = 0; i < VWORDS; ++i) {
            unsigned long long p = __shfl_down_sync(0xFFFFFFFFu, v.w[i], offset);
            v.w[i] ^= p;
        }
    }
    return v;
}

// upper_bound: find smallest r such that rowptr[r] > idx. Returns r in [1..num_rows]
__device__ __forceinline__ int csr_upper_bound(const uint32_t* rowptr, int num_rows, uint32_t idx) {
    int lo = 0, hi = num_rows;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        uint32_t v = rowptr[mid];
        if (v <= idx) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// ------------------------------- Kernels ------------------------------- //

// Merge-path-style nz tiling: partition the nonzero stream evenly across threads.
// Each thread processes a contiguous slice [tbegin, tend) of the flattened CSR
// (i.e., the concatenation of all rows), emitting partial XORs per row using
// atomics when crossing row boundaries.
template<int TWarpItems>
__global__ void csr_spmv_xor_warpmerge_kernel(const uint32_t* __restrict__ rowptr,
                                          const uint32_t* __restrict__ colidx,
                                          const v_t* __restrict__ x,
                                          v_t* __restrict__ y,
                                          int num_rows,
                                          uint32_t total_nnz) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_cta = blockDim.x / WARP_SIZE;
    const int warp_global = blockIdx.x * warps_per_cta + warp_in_block;

    uint32_t seg_begin = (uint32_t)((uint64_t)warp_global * (uint64_t)TWarpItems);
    if (seg_begin >= total_nnz) return;
    uint32_t seg_end = min(seg_begin + (uint32_t)TWarpItems, total_nnz);

    // Locate starting row
    int row = csr_upper_bound(rowptr, num_rows, seg_begin) - 1;
    uint32_t row_end = rowptr[row + 1];

    uint32_t cur = seg_begin;
    while (cur < seg_end) {
        uint32_t segment_end = min(row_end, seg_end);
        if (segment_end > cur) {
            // XOR all nz in [cur, segment_end) using warp-strided loads
            v_t acc = vt_zero();
            for (uint32_t j = cur + lane; j < segment_end; j += WARP_SIZE) {
                int col = (int)colidx[j];
                vt_xor_inplace(acc, x[col]);
            }
            v_t sum = warp_xor_reduce(acc);
            if (lane == 0) {
                vt_store_xor(&y[row], sum);
            }
            cur = segment_end;
        }
        // Advance to next non-empty row if we've finished the current row
        if (cur == row_end) {
            do {
                ++row;
                if (row >= num_rows) break;
                row_end = rowptr[row + 1];
            } while (row_end == cur); // skip empty rows
        }
    }
}

// Same product as csr_spmv_xor_warpmerge_kernel, built for short rows.
// Lanes are grouped VWORDS at a time (as in the scatter kernel below) and
// each group loads x[col] for one nonzero per step, so a warp consumes
// WARP_SIZE / VWORDS nonzeros per step with full 32-byte gathers regardless
// of row length. Groups holding the same row are combined with a segmented
// XOR scan over the groups; the last run of each step carries into the next
// step, and every other completed run is XORed into y[row]. This replaces
// the full-warp reduction per row segment, which wastes most of the warp
// when column-slice blocks leave only a few nonzeros per row.
#ifndef SEGSCAN_UNROLL
#define SEGSCAN_UNROLL 4
#endif

template<int TWarpItems>
__global__ void csr_spmv_xor_segscan_kernel(const uint32_t* __restrict__ rowptr,
                                          const uint32_t* __restrict__ colidx,
                                          const v_t* __restrict__ x,
                                          v_t* __restrict__ y,
                                          int num_rows,
                                          uint32_t total_nnz) {
    const int GROUPS = WARP_SIZE / VWORDS;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_cta = blockDim.x / WARP_SIZE;
    const int warp_global = blockIdx.x * warps_per_cta + warp_in_block;
    const int word = lane % VWORDS;
    const int group = lane / VWORDS;

    uint32_t seg_begin = (uint32_t)((uint64_t)warp_global * (uint64_t)TWarpItems);
    if (seg_begin >= total_nnz) return;
    uint32_t seg_end = min(seg_begin + (uint32_t)TWarpItems, total_nnz);

    int row = csr_upper_bound(rowptr, num_rows, seg_begin) - 1;
    uint32_t row_end = rowptr[row + 1];

    int carry_row = -1;
    unsigned long long carry = 0;

    // each pass issues the gathers for SEGSCAN_UNROLL steps before
    // combining any of them, to keep enough loads in flight when x
    // does not fit in L2
    for (uint32_t pass = seg_begin; pass < seg_end; pass += GROUPS * SEGSCAN_UNROLL) {
        int rr[SEGSCAN_UNROLL];
        unsigned long long vv[SEGSCAN_UNROLL];

        #pragma unroll
        for (int u = 0; u < SEGSCAN_UNROLL; u++) {
            const uint32_t j = pass + u * GROUPS + group;
            rr[u] = -1;
            vv[u] = 0;
            if (group < GROUPS && j < seg_end) {
                while (row_end <= j)
                    row_end = rowptr[++row + 1];
                rr[u] = row;
                vv[u] = x[colidx[j]].w[word];
            }
        }

        #pragma unroll
        for (int u = 0; u < SEGSCAN_UNROLL; u++) {
            const uint32_t base = pass + u * GROUPS;
            if (base >= seg_end)
                break;
            const bool valid = rr[u] >= 0;
            const int last = (int)min((uint32_t)(GROUPS - 1), seg_end - 1 - base);
            const int r = rr[u];
            unsigned long long v = vv[u];

            // fold in the run carried from the previous step, or flush
            // it if this step starts a new row
            const int r0 = __shfl_sync(0xFFFFFFFFu, r, 0);
            if (carry_row >= 0) {
                if (carry_row == r0) {
                    if (group == 0) v ^= carry;
                } else if (group == 0) {
                    atomicXor(&y[carry_row].w[word], carry);
                }
            }

            // segmented inclusive XOR scan across groups; rows are
            // nondecreasing, so equal rows are contiguous
            #pragma unroll
            for (int d = 1; d < GROUPS; d <<= 1) {
                unsigned long long vs = __shfl_up_sync(0xFFFFFFFFu, v, d * VWORDS);
                int rs = __shfl_up_sync(0xFFFFFFFFu, r, d * VWORDS);
                if (group >= d && rs == r) v ^= vs;
            }

            // emit every run that ends inside this step except the last one
            const int rn = __shfl_down_sync(0xFFFFFFFFu, r, VWORDS);
            if (valid && group < last && rn != r)
                atomicXor(&y[r].w[word], v);

            // the last run may continue into the next step
            carry_row = __shfl_sync(0xFFFFFFFFu, r, last * VWORDS);
            carry = __shfl_sync(0xFFFFFFFFu, v, last * VWORDS + word);
        }
    }

    if (group == 0 && carry_row >= 0)
        atomicXor(&y[carry_row].w[word], carry);
}

// Transpose multiply y = A^T * x using the same (non-transposed) CSR block,
// so the transpose never has to be stored. Uses the same even nonzero
// partitioning as above, but each nonzero (r, c) scatters x[r] into y[c]
// with atomics. Lanes are grouped VWORDS at a time so one group handles one
// nonzero, one 64-bit word per lane; a warp-wide atomic then touches whole
// v_t entries instead of scattered words. Each group walks rowptr forward
// from the warp's starting row to track the row of its current nonzero.
template<int TWarpItems>
__global__ void csr_spmv_xor_scatter_kernel(const uint32_t* __restrict__ rowptr,
                                          const uint32_t* __restrict__ colidx,
                                          const v_t* __restrict__ x,
                                          v_t* __restrict__ y,
                                          int num_rows,
                                          uint32_t total_nnz) {
    const int GROUPS = WARP_SIZE / VWORDS;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_cta = blockDim.x / WARP_SIZE;
    const int warp_global = blockIdx.x * warps_per_cta + warp_in_block;
    const int word = lane % VWORDS;
    const int group = lane / VWORDS;
    if (group >= GROUPS) return; /* leftover lanes if VWORDS doesn't divide 32 */

    uint32_t seg_begin = (uint32_t)((uint64_t)warp_global * (uint64_t)TWarpItems);
    if (seg_begin >= total_nnz) return;
    uint32_t seg_end = min(seg_begin + (uint32_t)TWarpItems, total_nnz);

    int row = csr_upper_bound(rowptr, num_rows, seg_begin) - 1;
    uint32_t row_end = rowptr[row + 1];

    for (uint32_t j = seg_begin + group; j < seg_end; j += GROUPS) {
        while (row_end <= j)
            row_end = rowptr[++row + 1];

        unsigned long long xv = x[row].w[word];
        atomicXor(&y[colidx[j]].w[word], xv);
    }
}

// ------------------------------- Host side ------------------------------- //

// Engine state tuned on first run
struct SpmvEngine { int threads_per_block; int warp_items; bool tuned; int kernel_mode; };

enum { K_WARPMERGE, K_SEGSCAN, K_SCATTER };

// In SPMV_KERNEL_AUTO mode, blocks averaging fewer nonzeros per row than
// this use segscan, the rest warpmerge
#ifndef SEGSCAN_MAX_ROW_MEAN
#define SEGSCAN_MAX_ROW_MEAN 8
#endif

static void spmv_autotune(SpmvEngine* eng, const uint32_t* d_rowptr, int num_rows) {
    if (eng->tuned || num_rows <= 0) return;
    std::vector<uint32_t> h_rowptr(num_rows + 1);
    cudaMemcpy(h_rowptr.data(), d_rowptr, (num_rows + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    const uint64_t total_nnz = h_rowptr[num_rows];
    if (total_nnz == 0) {
        eng->threads_per_block = 256;
        eng->warp_items = 256;
        eng->tuned = true;
        return;
    }

    std::vector<uint32_t> lens(num_rows);
    uint32_t max_row = 0;
    uint64_t empty_rows = 0;
    for (int i = 0; i < num_rows; ++i) {
        uint32_t len = h_rowptr[i + 1] - h_rowptr[i];
        lens[i] = len;
        if (len == 0) ++empty_rows;
        if (len > max_row) max_row = len;
    }

    const double mean = double(total_nnz) / double(num_rows);
    // p90, p99 via nth_element
    auto lens_copy = lens;
    auto idx90 = (size_t)((num_rows - 1) * 0.90);
    auto idx99 = (size_t)((num_rows - 1) * 0.99);
    std::nth_element(lens_copy.begin(), lens_copy.begin() + idx90, lens_copy.end());
    uint32_t p90 = lens_copy[idx90];
    std::nth_element(lens_copy.begin(), lens_copy.begin() + idx99, lens_copy.end());
    uint32_t p99 = lens_copy[idx99];

    // Count heavy rows beyond tile thresholds
    size_t heavy2k = 0;
    for (int i = 0; i < num_rows; ++i) {
        uint32_t L = lens[i];
        if (L > 2048u) ++heavy2k;
    }
    double frac2k = (double)heavy2k / (double)num_rows;
    double empty_frac = (double)empty_rows / (double)num_rows;

    int TPB = 256;
    int warp_items = 512;

    // Heuristic driven by p99 and the fraction of heavy rows.
    if (p99 <= 512u) {
        warp_items = 512;  TPB = 512; // many light rows; more warps is good
    } else if (p99 <= 1024u) {
        warp_items = 1024; TPB = 512;
    } else if (p99 <= 2048u && frac2k < 0.005) {
        // Rare very long rows: avoid oversizing tiles; keep parallelism high
        warp_items = 1024; TPB = 512;
    } else {
        warp_items = 2048; TPB = 512;
    }

    // If the mean is low and most rows are short, prefer smaller tiles
    if (mean < 24.0 && p90 < 96u && warp_items > 512) {
        warp_items = 512; TPB = 512;
    }

    eng->threads_per_block = TPB;
    eng->warp_items = warp_items;
    eng->tuned = true;

#ifdef SPMV_DEBUG
    printf("[spmv] tuned: TPB=%d warp_items=%d (mean=%.1f p90=%u p99=%u max=%u empties=%.1f%%)\n",
           TPB, warp_items, mean, p90, p99, max_row, 100.0 * empty_frac);
#endif
}

#if defined(_WIN32) || defined (_WIN64)
  #define SPMV_API extern "C" __declspec(dllexport)
#else
  #define SPMV_API extern "C" __attribute__((visibility("default")))
#endif

SPMV_API void* spmv_engine_init(int* vbits) {
    if (vbits) *vbits = VBITS;
    SpmvEngine* e = new SpmvEngine();
    e->threads_per_block = 256; // default
    e->warp_items = 512;        // default
    e->tuned = false;
    e->kernel_mode = SPMV_KERNEL_AUTO;
    return (void*)e;
}

// kernel for the gather products (spmv_engine_run): SPMV_KERNEL_AUTO picks
// per block, warpmerge (one warp per row segment) for long rows and
// segscan for short ones; the others force one kernel
SPMV_API void spmv_engine_set_kernel(void* e, int kernel) {
    reinterpret_cast<SpmvEngine*>(e)->kernel_mode = kernel;
}

SPMV_API void spmv_engine_free(void* e) { delete reinterpret_cast<SpmvEngine*>(e); }

template<int TWarpItems>
static void spmv_launch(int kernel, int blocks, int tpb, const uint32_t* rowptr,
                        const uint32_t* colidx, const v_t* x, v_t* y,
                        int num_rows, uint32_t total_nnz) {
    // the cache preference never changes, so set it once per kernel
    static bool configured[3] = { false, false, false };
    if (!configured[kernel]) {
        if (kernel == K_SCATTER)
            cudaFuncSetCacheConfig(csr_spmv_xor_scatter_kernel<TWarpItems>, cudaFuncCachePreferL1);
        else if (kernel == K_SEGSCAN)
            cudaFuncSetCacheConfig(csr_spmv_xor_segscan_kernel<TWarpItems>, cudaFuncCachePreferL1);
        else
            cudaFuncSetCacheConfig(csr_spmv_xor_warpmerge_kernel<TWarpItems>, cudaFuncCachePreferL1);
        configured[kernel] = true;
    }

    if (kernel == K_SCATTER) {
        csr_spmv_xor_scatter_kernel<TWarpItems><<<blocks, tpb>>>(rowptr, colidx, x, y, num_rows, total_nnz);
    } else if (kernel == K_SEGSCAN) {
        csr_spmv_xor_segscan_kernel<TWarpItems><<<blocks, tpb>>>(rowptr, colidx, x, y, num_rows, total_nnz);
    } else {
        csr_spmv_xor_warpmerge_kernel<TWarpItems><<<blocks, tpb>>>(rowptr, colidx, x, y, num_rows, total_nnz);
    }
}

static void spmv_run_common(void* e, spmv_data_t* spmv_data, bool trans) {
    SpmvEngine* eng = reinterpret_cast<SpmvEngine*>(e);

    const uint32_t* rowptr = reinterpret_cast<const uint32_t*>(spmv_data->row_entries);
    const uint32_t* colidx = reinterpret_cast<const uint32_t*>(spmv_data->col_entries);
    const v_t* x           = reinterpret_cast<const v_t*>(spmv_data->vector_in);
    v_t* y                 = reinterpret_cast<v_t*>(spmv_data->vector_out);
    const int num_rows     = spmv_data->num_rows;
    const uint32_t total_nnz = spmv_data->num_col_entries;
    if (num_rows <= 0) return;

    // Tune once (reads rowptr to host, negligible vs SpMV execution)
    if (!eng->tuned) spmv_autotune(eng, rowptr, num_rows);

    int TPB = eng->threads_per_block;
    const int warps_per_block = max(1, TPB / WARP_SIZE);
    const uint64_t total_warps = ( (uint64_t)total_nnz + (uint64_t)eng->warp_items - 1 ) / (uint64_t)eng->warp_items;
    const int blocks = (int)((total_warps + warps_per_block - 1) / warps_per_block);
    const int tpb = warps_per_block * WARP_SIZE;
    int kernel;
    if (trans)
        kernel = K_SCATTER;
    else if (eng->kernel_mode == SPMV_KERNEL_SEGSCAN)
        kernel = K_SEGSCAN;
    else if (eng->kernel_mode == SPMV_KERNEL_WARPMERGE)
        kernel = K_WARPMERGE;
    else
        kernel = (uint64_t)total_nnz < (uint64_t)SEGSCAN_MAX_ROW_MEAN * num_rows ?
                 K_SEGSCAN : K_WARPMERGE;

    if (blocks > 0) {
        switch (eng->warp_items) {
            case 256:
                SPMV_NVTX_PUSH(trans ? "spmv_run_trans[wi=256]" : "spmv_run[wi=256]");
                spmv_launch<256>(kernel, blocks, tpb, rowptr, colidx, x, y, num_rows, total_nnz);
                SPMV_NVTX_POP();
                break;
            case 1024:
                SPMV_NVTX_PUSH(trans ? "spmv_run_trans[wi=1024]" : "spmv_run[wi=1024]");
                spmv_launch<1024>(kernel, blocks, tpb, rowptr, colidx, x, y, num_rows, total_nnz);
                SPMV_NVTX_POP();
                break;
            case 2048:
                SPMV_NVTX_PUSH(trans ? "spmv_run_trans[wi=2048]" : "spmv_run[wi=2048]");
                spmv_launch<2048>(kernel, blocks, tpb, rowptr, colidx, x, y, num_rows, total_nnz);
                SPMV_NVTX_POP();
                break;
            case 512:
            default:
                SPMV_NVTX_PUSH(trans ? "spmv_run_trans[wi=512]" : "spmv_run[wi=512]");
                spmv_launch<512>(kernel, blocks, tpb, rowptr, colidx, x, y, num_rows, total_nnz);
                SPMV_NVTX_POP();
                break;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("SpMV launch failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

SPMV_API void spmv_engine_run(void* e, spmv_data_t* spmv_data) {
    spmv_run_common(e, spmv_data, false);
}

// y ^= A^T * x where spmv_data describes a CSR block of A (not A^T):
// vector_in is indexed by row, vector_out by column.
SPMV_API void spmv_engine_run_trans(void* e, spmv_data_t* spmv_data) {
    spmv_run_common(e, spmv_data, true);
}
