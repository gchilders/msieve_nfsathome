# CUDA Lanczos Optimization Notes

Date: 2026-05-26

This is the merged working plan for optimizing the CUDA Lanczos path. It
combines the useful parts of the Claude and Codex reviews, with priorities
updated after both reviews agreed that the native Blackwell build issue is the
first blocker.

No profiling conclusions should be treated as final until the CUDA artifacts
are rebuilt for the actual GPU architecture and a fresh baseline is captured.

## Code Surface

| File | Role |
|---|---|
| `common/lanczos/lanczos.c` | Top-level Lanczos iteration |
| `common/lanczos/lanczos_matmul.c` | `mul_sym_NxN_NxB`, `mul_MxN_NxB` |
| `common/lanczos/gpu/lanczos_matmul_gpu.c` | GPU matrix setup and SpMV orchestration |
| `common/lanczos/gpu/lanczos_vv.c` | Host wrappers for vector-vector kernels |
| `common/lanczos/gpu/lanczos_kernel.cu` | `mask`, `xor`, `inner_prod`, `outer_prod` kernels |
| `cub/spmv_engine.cu` | CSR XOR SpMV kernel and autotune |
| `cub/Makefile` | Builds `spmv_engine.so` and `sort_engine.so` |

## Critical Finding: Native sm_120 Build Is Broken

The working GPU is an RTX 5070, compute capability 12.0. The checked CUDA
artifacts were not native Blackwell:

```bash
cuobjdump --list-elf cub/spmv_engine.so
# ELF file 1: spmv_engine.1.sm_90.cubin
# ELF file 2: spmv_engine.2.sm_90.cubin

cuobjdump --list-elf lanczos_kernel.fatbin
# ELF file 1: lanczos_kernel.1.sm_90.cubin
```

The top-level Makefile can ask nvcc for `sm_120`:

```make
lanczos_kernel.fatbin:
	$(NVCC) -arch sm_$(SM) -fatbin -DVBITS=$(VBITS) -o $@ $<
```

But the CUB submake is called as:

```make
cd cub && make WIN=$(WIN) WIN64=$(WIN64) VBITS=$(VBITS) sm=$(SM)0 && cd ..
```

For `CUDA=120`, that passes `sm=1200` into `cub/Makefile`.

`cub/Makefile` currently uses substring matches such as:

```make
ifeq (200, $(findstring 200, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_20,code=\"sm_20,compute_20\"
endif
```

`SM_ARCH=1200` therefore matches the old `200` case and tries to emit
`compute_20/sm_20`, which modern CUDA has removed. If stale artifacts remain,
Lanczos can silently keep using an older `spmv_engine.so`.

This must be fixed before interpreting any performance data.

### Build Fix

Required:

- Add explicit handling for `SM_ARCH=1200` or `SM_ARCH=120`, producing
  `compute_120,sm_120`.
- Replace all substring architecture matching in `cub/Makefile` with exact
  matching. Future architectures must not accidentally match older entries.
- Make the top-level `cub/built` target sensitive to at least:
  - `cub/Makefile`
  - `cub/spmv_engine.cu`
  - `cub/spmv_engine.h`
  - `cub/sort_engine.cu`
  - `cub/sort_engine.h`
  - `VBITS`
  - selected `SM`
- Consider passing `sm=$(SM)` rather than `sm=$(SM)0`, or rename the CUB-side
  variable to make the expected format explicit.

Verification:

```bash
make clean
make all CUDA=120 VBITS=64
cuobjdump --list-elf cub/spmv_engine.so
cuobjdump --list-ptx cub/spmv_engine.so
cuobjdump --list-elf lanczos_kernel.fatbin
cuobjdump --dump-resource-usage cub/spmv_engine.so
cuobjdump --dump-resource-usage lanczos_kernel.fatbin
```

Expected result: native `sm_120` cubins for both `cub/spmv_engine.so` and
`lanczos_kernel.fatbin`.

## Baseline Instrumentation Before Kernel Work

Add lightweight timing or NVTX ranges before tuning. Without per-direction
labels, normal SpMV and transpose SpMV are difficult to separate in nsys.

Useful labels:

- `mul_core`
- `mul_trans_core`
- each `spmv_engine_run`
- dense-row handling in `mul_packed_gpu`
- dense-row handling in `mul_packed_trans_gpu`
- `vv_mul_BxN_NxB`
- `vv_mul_NxB_BxB_acc`
- `lanczos_kernel_outer_prod`
- `lanczos_kernel_inner_prod`

Baseline capture:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --stats=true \
  ./msieve -nc2 "select_density=100" -g 0
```

For kernel details:

```bash
ncu --set full --kernel-name regex:csr_spmv_xor_warpmerge_kernel ./msieve ...
ncu --set full --kernel-name regex:lanczos_kernel_outer_prod ./msieve ...
ncu --set full --kernel-name regex:lanczos_kernel_inner_prod ./msieve ...
```

Track:

- Normal SpMV time vs transpose SpMV time.
- DRAM throughput.
- L2 hit rate.
- Atomic throughput and atomic stalls.
- Achieved occupancy and register/shared-memory limits.
- Stall reasons.
- Launch count per iteration.

## Per-Iteration Shape

At default `VBITS=64`, each Lanczos iteration roughly does:

| Call | Count | Notes |
|---|---:|---|
| `mul_sym_NxN_NxB` | 1 | One multiply by `A`, then one multiply by `A^T` |
| `vv_mul_BxN_NxB` | 2 common, more for checks | Outer products producing `VBITS x VBITS` data |
| `vv_mul_NxB_BxB_acc` | 3-4 common | Full-vector updates into `vnext` and `x` |
| `vv_mask`, `vv_xor`, `vv_copy`, clears | several | Streaming vector kernels |

SpMV is expected to dominate, but confirm this after native `sm_120` artifacts
are built.

## Priority Order

### 1. Fix Native sm_120 Build

This is prerequisite work, not an optimization experiment.

If the code is running `sm_90` cubins or PTX-JITed code on `sm_120` hardware,
then register counts, occupancy, scheduling, cache behavior, and any ncu
conclusions are not a clean baseline.

### 2. Add Timing Labels / NVTX

This is also prerequisite work.

The key split is normal SpMV vs transpose SpMV. The two directions may have
different row-length distributions and different bottlenecks. The optimization
choice depends on knowing which direction costs more.

### 3. Baseline With nsys / ncu

Use a repeatable matrix, ideally a checkpoint restart or a fixed `-nc2`
workload. Record:

- GPU model and compute capability.
- CUDA toolkit version.
- `CUDA` and `VBITS` build settings.
- `block_nnz`.
- matrix dimensions, nonzeros, dense rows.
- number of normal and transpose matrix blocks.
- total iteration time and kernel breakdown.

### 4. Sweep `block_nnz`

`block_nnz` is already exposed through the `-nc2` argument string:

```text
block_nnz=N
```

Suggested initial sweep:

```text
block_nnz=256000000
block_nnz=512000000
block_nnz=1000000000
block_nnz=1750000000
```

Why this matters:

- Smaller blocks reduce the active slice of `x[col]` for normal SpMV.
- That can improve L2 locality for random gathers.
- But smaller blocks also increase launch count, rowptr traffic, and repeated
  accumulation into the output vector.

Rough L2-fit target for RTX 5070:

- L2 size is about 48 MB.
- At `VBITS=64`, each `x` entry is 8 bytes.
- About 6 million `x` entries fit in L2.
- With a typical NFS matrix average around 200 nonzeros per column, this
  corresponds to about 1.2 billion nonzeros per block.

The current default, `block_nnz=1750000000`, sits roughly 50% above that
back-of-envelope L2-fit ceiling. The locality hypothesis therefore predicts
that the `256M` to `1B` end of the sweep should improve SpMV time. If it does
not, then launch overhead, rowptr traffic, atomics, or the actual row/column
distribution is dominating the simple L2 model.

Measure total iteration time, not only L2 hit rate. The best point is a balance
between locality and launch/accumulation overhead.

### 5. Lane-0 `csr_upper_bound` + Warp Broadcast

Current `csr_spmv_xor_warpmerge_kernel` computes:

```cuda
int row = csr_upper_bound(rowptr, num_rows, seg_begin) - 1;
uint32_t row_end = rowptr[row + 1];
```

All lanes in a warp compute the same starting row. That duplicates the binary
search and rowptr loads 32 times.

Low-risk experiment:

```cuda
int row;
uint32_t row_end;
if (lane == 0) {
    row = csr_upper_bound(rowptr, num_rows, seg_begin) - 1;
    row_end = rowptr[row + 1];
}
row = __shfl_sync(0xffffffffu, row, 0);
row_end = __shfl_sync(0xffffffffu, row_end, 0);
```

Do the same kind of lane-0/broadcast treatment for the row-advance path if it
stays warp-uniform.

Expected impact is small, but the patch is localized and correctness risk is
low.

### 6. SPMV Tile / Direction Tuning

Current `spmv_engine.cu` has one global autotune decision:

```c++
struct SpmvEngine { int threads_per_block; int warp_items; bool tuned; };
```

`spmv_autotune()` runs once and reuses the result for all normal and transpose
blocks.

Experiment:

- Add explicit override:

```text
spmv_warp_items=256
spmv_warp_items=512
spmv_warp_items=1024
spmv_warp_items=2048
```

- Then allow a direction split:

```text
spmv_warp_items=normal,transpose
spmv_warp_items=512,1024
```

If this matters, replace global one-shot tuning with per-direction or per-block
tuning. Per-direction tuning is likely the useful first split. Per-block tuning
only becomes interesting if `block_nnz` is reduced enough to create many more
matrix blocks.

## Conditional Next Steps

These depend on what the native baseline shows.

### If Atomics Are a Bottleneck: Reduce SpMV Atomics

Current SpMV uses `atomicXor` for every segment result. This is simple and
correct because:

- rows can straddle multiple warp tiles inside one kernel, and
- multiple matrix blocks accumulate into the same output vector.

Possible directions:

- Hybrid owned-row path:
  - For rows fully owned by a single warp tile, do non-atomic `y[row] ^= sum`.
  - For boundary or long rows, keep `atomicXor`.
- Row-per-warp or row-per-CTA kernel:
  - One warp/CTA owns a row and stores without atomics.
  - Use fallback for very long rows.

Important correction:

Do not assume the pre-SpMV `cuMemsetD8` can simply be removed. `mul_packed_gpu`
and `mul_packed_trans_gpu` clear the output once, then multiple matrix-block
SPMV launches XOR-accumulate into the same output vector. A plain store from
one block would overwrite contributions from earlier blocks.

Removing or reducing the memset requires a different accumulation scheme, not
just replacing atomics with stores.

One viable redesign:

- The first matrix-block SpMV initializes the output vector and uses plain
  stores where it owns a row.
- Later matrix-block SpMVs continue to XOR-accumulate into the initialized
  output vector.
- For rows split across warp tiles or handled by long-row fallback, keep
  atomics or use a second-stage reduction.

The first pass must initialize every output row, including rows with no
nonzeros in the first matrix block. That can mean a row-oriented first-block
kernel that stores zero for empty rows, or a separate compact initialization
for first-block-empty rows. Without that, later `atomicXor` calls would operate
on undefined output values.

For one-block matrices, this can remove the standalone memset and most SpMV
atomics. For two- or three-block matrices, it still removes the standalone
memset and reduces atomic traffic for the first block, but the later blocks
still need safe accumulation.

### If Full-Vector Update Traffic Is Material: Fuse `NxB * BxB` Updates

The Lanczos recurrence performs several full-vector update passes:

```text
vnext ^= v[0] * D
vnext ^= v[1] * E
vnext ^= v[2] * F   // conditional
x     ^= v[0] * X
```

Each pass launches `lanczos_kernel_inner_prod` and scans a large vector.

Experiment:

- Copy all small BxB matrices to one scratch region.
- Launch one fused kernel that updates `vnext` and optionally `x` in one scan.

Potential win:

- Fewer launches.
- Fewer vector reads/writes.

Risk:

- Register pressure. The stage-1 fusion experiments showed that correctness
  can be clean while performance regresses due to lost occupancy. Check
  `cuobjdump --dump-resource-usage` immediately.

### If Outer Product Is Material: Fuse or Rework Outer Products

The common iteration computes:

```text
v[0]^T   * vnext
vnext^T * vnext
```

These are two `vv_mul_BxN_NxB` calls, each scanning `N`.

At `VBITS=64`, the inner `w_x * w_y` loop inside one `outer_prod` call is only
`1 x 1`, so this is not a huge VBITS-loop fusion opportunity. The possible
benefit is saving one whole-vector scan and one launch by computing both outer
products together.

Treat this as a measured experiment, not a guaranteed win.

Also consider a two-stage reduction if `lanczos_kernel_outer_prod` shows global
atomic contention on the tiny `VBITS x VBITS` result:

- kernel 1 writes per-CTA partial results,
- kernel 2 reduces partials.

### If SpMV Is DRAM-Bound After `block_nnz`: Consider L2 Persistence

CUDA access-policy windows can bias L2 retention for the input vector slice.
This is a backstop if `block_nnz` cannot make the active `x` range cache well.

Do this only after measuring native `sm_120` behavior and the `block_nnz`
sweep.

### If Dense Rows Are Surprisingly Large: Consider Stream Overlap

The sparse SPMV loop and dense-row kernels in `mul_packed_gpu` are serialized
on stream 0. At `VBITS=64`, dense rows are likely small, so overlap is probably
not worth engineering first.

Only revisit if nsys shows dense-row kernels are material.

### If Memory Capacity Is the Limiter: Revisit A / A^T Storage

`gpu_matrix_init` builds independent CSR-like structures for normal and
transpose directions. This roughly doubles sparse matrix storage on the GPU.

Sharing storage or using a compact dual-view representation is more about
fitting larger jobs than reducing milliseconds per iteration. It is a larger
engineering project and should not be the first speed optimization.

### If Nothing Else Moves: Matrix Reordering

A matrix permutation or graph-partitioning pass could improve locality for
random gathers into `x`, but this is speculative and invasive:

- must preserve solver correctness,
- needs persisted permutations,
- may require external tooling or substantial preprocessing time.

Only consider after native profiling shows SpMV is still gather/locality bound
and simpler geometry changes fail.

## Things Not Recommended First

- Compressible memory: already tried in this tree and commented as not useful.
- L2 fetch granularity tweak: already tried and commented as not useful for
  `VBITS=256`.
- cuSPARSE: GF(2) XOR is not a supported cuSPARSE semiring.
- CUDA Graphs: potentially useful for launch overhead, but only after the
  kernel sequence is stable.
- Broad fusion before profiling: stage-1 experiments showed fusion can regress
  through register pressure even when it removes a launch.

## Immediate Checklist

1. Patch `cub/Makefile` architecture matching.
2. Patch top-level CUB build dependencies / stale stamp behavior.
3. Clean rebuild with `CUDA=120 VBITS=64`.
4. Verify `sm_120` artifacts with `cuobjdump`.
5. Add NVTX/timing labels around normal SpMV, transpose SpMV, dense rows, and
   vector-vector kernels.
6. Capture nsys baseline on a repeatable `-nc2` workload.
7. Capture ncu on the dominant SPMV kernel.
8. Sweep `block_nnz`.
9. Apply lane-0 `csr_upper_bound` broadcast and remeasure.
10. Choose the next structural kernel experiment from actual bottlenecks.
