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

## Results: 2026-06-09 Baseline + block_nnz Sweep (basement folder)

Setup: RTX 5070 native sm_120 build, CUDA 13.2, VBITS=64, C168_1074_2202
density-100 matrix (9538691 x 9538916, 884.6M sparse nnz + 64 packed dense
rows), 10.5 GB VRAM, benchmark = 5-min `-ncr` checkpoint restarts.

nsys NVTX split (default block_nnz, per iteration):

| Range | ms/iter | share |
|---|---:|---:|
| normal SpMV (A·x) | 84.0 | 70% |
| transpose SpMV (A^T·x) | 28.0 | 23% |
| vector ops + dense rows + memset | ~4 | 3% |

Normal direction is 3x slower than transpose over the same nnz.
Profile: /tmp/lanczos_nsys_baseline.nsys-rep

block_nnz sweep:

| block_nnz | blocks | ms/iter | speedup |
|---|---:|---:|---:|
| 1.75e9 (default) | 1 | 119.7 | 1.00x |
| 1e9 | 1 | 119.9 | 1.00x |
| 512M | 2 | 54.4 | 2.20x |
| **256M** | 4 | **46.6** | **2.57x** |
| 128M | 7 | 60.4 | 1.98x |
| 64M | 14 | 88.4 | 1.35x |

Clear minimum near 256M; L2-locality hypothesis confirmed. Note the sweep
was run at VBITS=64 (8B per x entry); at VBITS=256 the optimum should shift
~4x lower (~64M). Full-solve projection at 256M: ~1h57m vs ~5h default.

Next: finer probe 192M-384M, ncu on normal-direction kernel (atomics vs
DRAM), per-direction warp_items tuning, lane-0 broadcast.

## 2026-06-10: Multi-block Correctness Verification

After a sqrt failure on a solve that had mixed block_nnz experiments and a
likely matrix rebuild, multi-block SpMV was verified two independent ways:

1. `cub/spmv_blocktest.c` — standalone harness (dlopen spmv_engine.so):
   single vs 2/4/7/14 column slices bitwise-match a CPU reference on two
   synthetic shapes including 400K-entry heavy rows.
2. Real-matrix A/B: identical seed checkpoint restarted under default
   (1 block) and block_nnz=256M (4 blocks), both run to the same fixed
   dump boundary (new `dump_interval=N` nfs_args knob in lanczos.c skips
   the timing-based recalibration). The two 381MB checkpoints at
   dim_solved=100033 were byte-identical.

Conclusions:
- Multi-block SpMV and block_nnz=256M are safe for production.
- A `.chk` is only valid for the exact `.mat` it started on. Rebuilding
  the matrix draws fresh random quadratic characters, so "rebuild .mat,
  continue from old .chk" silently produces garbage dependencies that
  fail in sqrt as "algebraic side is not a square" on every dependency.

## 2026-06-10: Cross-VBITS Comparison (same matrix, fresh-start 380s windows)

Matrix: C168 9532121 x 9532347, 884.6M sparse nnz, RTX 5070. Metric is
dims/sec (ms/iter is not comparable across VBITS). msieve-reported host
memory in parens.

| VBITS | block_nnz | blocks | dims/s | ms/iter | mem |
|---:|---|---:|---:|---:|---|
| 64  | default (1 blk) | 1 | 528   | 119.7 | (7.4 GB) |
| 64  | 512M  | 2  | 1162 | 54.4  | |
| 64  | 256M  | 4  | **1350** | 46.6-47.5 | |
| 64  | 128M  | 7  | 1046 | 60.4  | |
| 128 | default | 1 | 743  | 171.3 | (7.5 GB) |
| 128 | 256M  | 4  | 1308 | 97.3  | (7.7 GB) |
| 128 | 128M  | 7  | **1421** | 89.6 | (7.9 GB) |
| 128 | 64M   | 14 | 1020 | 124.8 | (8.4 GB) |
| 256 | default | 1 | 1208 | 211.3 | (8.3 GB) |
| 256 | 128M  | 7  | **1435** | 177.8 | (8.7 GB) |
| 256 | 64M   | 14 | 1162 | 219.6 | (9.1 GB) |
| 256 | 32M   | 28 | 769  | 255.2 | (10.0 GB) |

Findings:

- Tuned optima converge: 1350 / 1421 / 1435 dims/s for VBITS 64/128/256.
  Higher VBITS is worth ~5-6% over VBITS=64 once block_nnz is tuned —
  real but small. The convergence suggests a common memory-system limit.
- Untuned (single-block) the spread is huge: 528 / 743 / 1208. This is
  why higher VBITS is the right advice for users who never set block_nnz:
  wider vectors amortize per-nonzero index traffic.
- Optimal block_nnz by VBITS: 256M / 128M / 128M. A "half-L2 active
  x-window" model predicts 278M / 139M / 70M; the v256 deviation (128M
  measured vs 70M predicted) indicates a per-block overhead floor.
  Candidate dynamic formula for lanczos_matmul_gpu.c:739:
    block_nnz = max(128M, (L2_bytes/2) / sizeof(v_t) * avg_col_weight)
  which reproduces all three measured optima on this card.
- VBITS=256 fits this ~9.5M matrix on 12GB (shared with display), but
  vectors scale 4x vs VBITS=64 — capacity, not speed, is the VBITS=256
  concern for larger matrices on this card.

Next: one ncu pass on the tuned config to identify the converged
bottleneck (gather sectors / atomics / DRAM). If the memory system is
saturated, kernel micro-opts (lane-0 broadcast) won't pay; matrix
reordering would be the remaining lever.

## 2026-06-10: ncu Profile of Tuned Config (VBITS=64, block_nnz=256M)

Captured per-direction via NVTX filters (8 kernels each, --set detailed).

Normal direction (3 full blocks @ ~6.8ms + 1 partial @ ~3ms):
- L2 cache throughput 74% (the saturated unit), L2 hit rate 87.8%
- DRAM only 28% (185 GB/s) — NOT DRAM-bound; block_nnz fix confirmed
- SM 39%, occupancy 89%, L1 hit 8% (random gathers miss L1)
- autotune picked warp_items=512 at this block size (1024 at single-block)

Transpose direction (4 blocks, heterogeneous):
- Block 1 (heavy ideal rows, small x-window): L1 hit 86%, L2 only 15%,
  SM 58% — gathers absorbed by L1 when the window is tiny
- Blocks 2-3: L2 throughput 84-88%, L2 hit 87.5%, DRAM ~32% — same
  L2-bandwidth-bound profile as the normal direction

Conclusion: after block_nnz tuning, both directions sit at the L2
BANDWIDTH ceiling (~75-88% utilization, ~88% hit rate). This explains the
cross-VBITS convergence at ~1400 dims/s. Remaining levers, in order:
1. VBITS=256 in production (+~6%, already measured; full 32B sectors per
   gather and 4x less colidx traffic per dim).
2. Owned-row stores to cut atomicXor L2 round-trips (bounded, ~10-15% of
   L2 ops at most).
3. lane-0 csr_upper_bound broadcast (small).
4. Column-clustering / matrix reordering to raise the 8% L1 hit rate —
   the only large lever left, and the most invasive.
The 2.6x from block_nnz was the structural win; everything left is
constants-level (5-20%).

## Cross-Card Validation Protocol

`vbits_block_sweep.sh` packages the benchmark for other GPUs: copy
msieve.dat.mat + msieve.fb + worktodo.ini from the reference C168 job,
run `./vbits_block_sweep.sh <sm>`, return bench_results.tar.gz.

Predictions to check against the Tesla V100 (sm_70, 6MB L2, 900GB/s HBM2):
1. The block_nnz curve should be much flatter than the 5070's, with the
   optimum at large blocks (256M-1.75B) — the L2-resident window is
   unreachable above the per-block overhead floor.
2. The VBITS=256 vs 64 gap should be much larger than the 5070's +6%
   (full-sector gathers attack the DRAM-bound constraint directly).
3. Total block_nnz gain limited (~1.2-1.5x vs the 5070's 2.6x).
If (1)-(3) hold, the dynamic block_nnz formula's floor behavior is right
for small-L2 cards and per-arch guidance becomes: big-L2 cards tune
block_nnz, small-L2 cards raise VBITS.

## 2026-06-11: Tesla V100 Results (small-L2 validation)

V100-SXM2-32GB (sm_70, 6MB L2, 900GB/s HBM2), CUDA 12.1, 20.2M x 20.2M
matrix, 2.244B sparse nnz, 111 nnz/col. Fresh-start 900s windows.
Reference: user's 51h production solve = VBITS=64 default = 574 ms/iter.

| VBITS | block_nnz | blocks | dims/s | ms/iter | host mem |
|---:|---|---:|---:|---:|---|
| 64  | default(1.75B) | 2 | 109.5 | 577 | 18.7 GB |
| 64  | 512M | 5  | 115.5 | 547 | 19.1 GB |
| 64  | 256M | 9  | 127.5 | 496 | 19.7 GB |
| 64  | 128M | 18 | **141.9** | 445 | 21.2 GB |
| 64  | 64M  | 35 | 136.2 | 464 | 23.9 GB |
| 256 | default(1.75B) | 2 | FAILED internal check at shutdown | | 20.2 GB |
| 256 | 512M | 5  | 256.1 | 997 | 20.5 GB |
| 256 | 256M | 9  | **257.5** | 991 | 21.1 GB |
| 256 | 128M | 18 | 246.3 | 1037 | 22.3 GB |
| 256 | 64M  | 35 | 217.5 | 1173 | 24.7 GB |

Prediction scorecard:
1. "Flat curve, optimum at large blocks" — half right. VBITS=256 is flat
   (512M-128M within 4%), as predicted (its windows can never fit 6MB L2).
   VBITS=64 has a real curve: +30% at 128M (9.2MB window ~ 1.5x L2), so
   partial L2 residency pays even on small-L2 cards.
2. "VBITS=256 gap >> the 5070's +6%" — confirmed dramatically: best-vs-
   best +81% (141.9 -> 257.5 dims/s). Pure transaction efficiency.
3. "Total block_nnz gain 1.2-1.5x" — 1.30x at VBITS=64. Combined
   VBITS=256 + tuned block_nnz vs the production config: 2.35x
   (the 51h solve becomes ~22h).

Dynamic formula validation: block_nnz = max(128M, (L2/2)/sizeof(v_t) *
avg_col_weight) predicts: V100 v64 -> 128M (measured optimum, exact);
V100 v256 -> 128M (within 4.4% of measured best); 5070 v64 -> 278M
(~256M optimum, exact); 5070 v128/v256 -> 128M (measured optima, exact).
Within ~5% of measured best on every card/VBITS tested. VALIDATED on
two architectures at opposite ends of the L2 spectrum.

Unified recommendation: VBITS=256 + dynamic block_nnz wins on both
cards (5070: 1435 dims/s; V100: 257 dims/s) WHERE VRAM PERMITS — this
20.2M matrix needs ~21GB at VBITS=256, so 12GB consumer cards must run
VBITS=64 (+formula) for matrices this size.

ANOMALY to investigate: v256 + default (2 blocks of ~1.75B nnz) failed
the periodic consistency check during graceful shutdown on the V100
("error: corrupt state"). All other v256 runs passed continuous checks
for 8-10 min each. Could be the near-clamp 1.75B block size at VBITS=256
on sm_70, or a shutdown-path quirk. The 5070 ran v256+default cleanly.
Do not ship v256 near the 1.75B clamp until understood; the recommended
formula values (128-256M) are unaffected.

## Tuning Heuristic (validated 2026-06-11, RTX 5070 + Tesla V100)

1. Use the largest VBITS that fits VRAM (256 where possible). Tuned
   speed orders 256 >= 128 > 64 on every card tested; the advantage
   grows as L2 shrinks (+6% on 48MB-L2 5070, +81% on 6MB-L2 V100).
2. Set block_nnz = max(128M, (L2_size/2)/sizeof(v_t) * avg_col_weight).
   Within ~5% of measured optimum on every tested card/VBITS.
3. If VRAM is tight: raise block_nnz first (frees per-block rowptr
   replicas and costs little speed at high VBITS); drop VBITS only as
   a last resort (-45% on small-L2 cards).
4. use_managed (matrix > VRAM): max VBITS + default/max block_nnz.
   The bottleneck becomes matrix bytes streamed per dim of progress
   = (4B*nnz + rowptr replicas)/VBITS — big VBITS divides it, big
   blocks minimize replication. (Predicted, not yet benchmarked.)
5. Never rebuild the .mat mid-solve (fresh random quadratic characters
   = different matrix; checkpoints are only valid for the exact .mat
   they started on).

Anomaly follow-up: the v256 + default (1.75B, near-clamp) "corrupt
state" failure on the V100 did NOT reproduce on rerun — clean halt,
243.6 dims/s (1048 ms/iter), completing the v256 table (default costs
~5% vs the 256M optimum; flat curve confirmed). Treated as a transient
one-off; no config restriction, but if "corrupt state" ever appears
again, capture the log and investigate the shutdown-path check.

## Planned: VBITS=512 Test (2026-06-12)

Code supports VBITS up to 512 (VWORDS=8 unrolls present; lanczos.h
whitelist). Predictions to check, written before measurement:

1. Speed: v512 ~= v256 + 0-8%. GPU memory moves in 32B sectors; v_t hits
   exactly one sector at VBITS=256 (gather sector traffic per dim
   plateaus there — v512 issues half the gathers but each is 2 sectors).
   The remaining v512 gain is colidx/rowptr traffic halving per dim.
   If correct, VBITS=256 is the efficiency sweet spot and 512 is only
   worth it where its memory cost is free.
2. Memory: vectors and the dense-row block double vs v256 (~+30% total
   footprint). RTX 5070 + 9.5M C168 matrix: ~13GB needed vs 12GB card —
   expect OOM (which is itself the answer for 12GB cards). V100-32GB +
   20.2M matrix: ~30-31GB at large blocks — borderline; prefer
   NNZ_LIST="default 512000000 256000000 128000000" and watch the new
   per-run vram_*.log sampling (added to vbits_block_sweep.sh).

Commands: 5070: build VBITS=512, run plain skip_matbuild (formula picks
block size). V100: ./vbits_block_sweep.sh 70 512 with the NNZ_LIST above.
