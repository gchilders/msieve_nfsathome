// CSR SpMV over XOR (merge-path style partitioning)

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include "spmv_engine.h"

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

// ------------------------------- Host side ------------------------------- //

// Engine state tuned on first run
struct SpmvEngine { int threads_per_block; int warp_items; bool tuned; };

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
    return (void*)e;
}

SPMV_API void spmv_engine_free(void* e) { delete reinterpret_cast<SpmvEngine*>(e); }

SPMV_API void spmv_engine_run(void* e, spmv_data_t* spmv_data) {
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

    if (blocks > 0) {
        switch (eng->warp_items) {
            case 256:
                cudaFuncSetCacheConfig(csr_spmv_xor_warpmerge_kernel<256>, cudaFuncCachePreferL1);
                csr_spmv_xor_warpmerge_kernel<256><<<blocks, warps_per_block * WARP_SIZE>>>(rowptr, colidx, x, y, num_rows, total_nnz);
                break;
            case 512:
                cudaFuncSetCacheConfig(csr_spmv_xor_warpmerge_kernel<512>, cudaFuncCachePreferL1);
                csr_spmv_xor_warpmerge_kernel<512><<<blocks, warps_per_block * WARP_SIZE>>>(rowptr, colidx, x, y, num_rows, total_nnz);
                break;
            case 1024:
                cudaFuncSetCacheConfig(csr_spmv_xor_warpmerge_kernel<1024>, cudaFuncCachePreferL1);
                csr_spmv_xor_warpmerge_kernel<1024><<<blocks, warps_per_block * WARP_SIZE>>>(rowptr, colidx, x, y, num_rows, total_nnz);
                break;
            case 2048:
                cudaFuncSetCacheConfig(csr_spmv_xor_warpmerge_kernel<2048>, cudaFuncCachePreferL1);
                csr_spmv_xor_warpmerge_kernel<2048><<<blocks, warps_per_block * WARP_SIZE>>>(rowptr, colidx, x, y, num_rows, total_nnz);
                break;
            default:
                cudaFuncSetCacheConfig(csr_spmv_xor_warpmerge_kernel<512>, cudaFuncCachePreferL1);
                csr_spmv_xor_warpmerge_kernel<512><<<blocks, warps_per_block * WARP_SIZE>>>(rowptr, colidx, x, y, num_rows, total_nnz);
                break;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("SpMV launch failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
