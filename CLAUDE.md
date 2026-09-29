# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Standard build (no GPU):
```bash
make all
```

With CUDA (replace `90` with your GPU's compute capability, e.g. `86`, `89`):
```bash
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH   # WSL2 only — needed at link time
make all CUDA=90
```

Other useful flags: `OMP=1` (default on), `MPI=1`, `ECM=1`, `VBITS=64` (default).

CUDA builds on Linux produce one self-contained binary: the CUB sort/SpMV engines are linked in and the stage-1 and Lanczos kernels are embedded (fatbin plus PTX fallback), so no `.ptx`, `.fatbin` or `cub/*.so` files are needed at runtime. `CUDA_ARCHS="86 120"` embeds several architectures (oldest first; `CUDA_PTX_ARCH` defaults to the first); `CUDA=cc` still builds one. Plain `CUDA=1` builds whichever of 80/86/89/90/120 the installed nvcc supports, plus PTX for compute_80 and for the oldest architecture nvcc supports from sm_60 up. `CUDA_SINGLE_BINARY=0` restores the old external-file layout (the `WIN=1` default). Changing `VBITS` or the architectures rebuilds the GPU objects (via the `cub/.build_config` stamp), but not the host C objects, so `make clean` when switching VBITS.

Clean: `make clean`

There are no automated tests. Validation is done by running actual NFS
factorizations; Greg gates filtering changes on a byte-identical msieve.dat.cyc
and msieve.dat.mat for the small and medium jobs under c:/dev/numbers on his
machine.

## Running

The binary is `./msieve`. Key NFS phase flags:
- `-nc1` — filtering (relation filtering → produces `.cyc`)
- `-nc2` — linear algebra (matrix build + Lanczos solver → produces `.dep`)
- `-nc3` — square root (uses `.dep` to find factors)
- `-nc` — all three phases in sequence
- `-ncr` — restart Lanczos from checkpoint (`.chk` file)
- `-t X` — use X threads for linear algebra
- `-g X` — select GPU index X

NFS options are passed as a quoted string:
```
./msieve -nc1 "target_density=80,100,120"
./msieve -nc2 "all_matbuild=1"
./msieve -nc2 "select_density=100"
```

### Setup helper script

`setup_job.sh` prepares a working directory from downloaded NFS@Home job files: looks for `*.gz`, `*.fb`, and `*.ini` files, renames them to `msieve.dat.gz`, `msieve.fb`, `worktodo.ini`, and decompresses the `.gz`.

### LA benchmark script

`bench_la.sh` times Lanczos on a prebuilt matrix: `./bench_la.sh -d 90 -a "single_copy=1"` runs `-nc2 "skip_matbuild=1 ..."` on `msieve.dat.mat.90` in its own `bench/<name>-<stamp>/` directory (symlinks to the matrix, so the real job files are never written), measures dims/sec after warmup, stops msieve with SIGTERM (which runs the Lanczos integrity check) and appends a row to `bench/results.tsv`. `-F` instead lets the solve finish and runs `-nc3` in the same directory.

## Architecture

This is a C library (`libmsieve.a`) plus a thin demo binary (`demo.c` → `msieve`). The library interface is in `include/msieve.h`.

### Object extensions

The Makefile uses different object file extensions to allow the same source filename in different subdirectories:
- `.o` — common code
- `.qo` — QS (quadratic sieve) code
- `.no` — NFS code

### NFS pipeline

For large numbers Msieve runs the Number Field Sieve in three phases:

**1. Filtering (`-nc1`)** — `gnfs/filter/filter.c` is the top-level entry point (`nfs_filter_relations`).
- Removes duplicate relations (`gnfs/filter/duplicate.c`)
- Writes a large-prime-only LP file (`gnfs/filter/singleton.c`: `nfs_write_lp_file`)
- Compacts the 64-bit LP file for common filtering, running disk-based singleton passes first if it is large (`gnfs/filter/singleton.c`: `nfs_compact_lp_file`)
- Reads LP file into memory and runs in-memory singleton removal (`filter_read_lp_file` → `filter_purge_singletons_core`)
- Runs clique removal, 2-way merge, full merge (`common/filter/`)
- Writes cycle file `msieve.dat.cyc` (or `msieve.dat.cyc.NNN` for multi-density)

There are three internal code paths based on data size relative to RAM:
- **Small** (`savefile_size < ram/2`): single LP pass
- **Medium** (large savefile, small LP after pruning): second LP write with tighter bounds
- **Large/partial**: iterates over `max_weight` thresholds

**2. Linear algebra (`-nc2`)** — `gnfs/gf2.c` (`nfs_solve_linear_system`)
- Builds matrix from cycles + relations → `msieve.dat.mat` (+ `msieve.dat.mat.idx` for MPI)
- Runs block Lanczos solver → `msieve.dat.dep`
- GPU-accelerated sparse matrix multiply when built with `CUDA=`

**3. Square root (`-nc3`)** — `gnfs/sqrt/`
- Processes each dependency from `.dep` to attempt factorization

### Custom additions (this fork)

Extensions added for NFS@Home distributed factoring workflows, all controlled via the `-nc1`/`-nc2` argument string:

**Multi-density filtering** (`gnfs/filter/filter.c`):
- `target_density=80,100,120` — comma-separated list; densities are sorted numerically, merged independently, each written to `msieve.dat.cyc.80` / `.cyc.100` / `.cyc.120`; stops at first failure
- The post-singleton in-memory `relation_array` is deep-copied after the first merge and restored from memory (not re-read from disk) for each subsequent density

**Multi-density matrix build** (`gnfs/gf2.c`):
- `all_matbuild=1` — scans for `.cyc.NNN` files, builds a `.mat.NNN` (and `.mat.idx.NNN`) for each; implies `only_matbuild=1`

**Density selection for LA** (`gnfs/gf2.c`):
- `select_density=100` — renames `msieve.dat.cyc.100` → `msieve.dat.cyc` and `msieve.dat.mat.100` → `msieve.dat.mat` (and `.mat.idx`), then skips the matrix build and runs Lanczos directly

**Single-copy GPU matrix** (`common/lanczos/gpu/lanczos_matmul_gpu.c`, `cub/spmv_engine.cu`):
- `single_copy=1` — stores only A on the GPU, not A^T; the transpose product scatters through A's column-slice blocks with atomic XOR (`spmv_engine_run_trans`). Its block_nnz default is a third of L2 worth of columns, floored at 2×nrows so the per-block row-pointer arrays stay under half the column indices; that saves ~40% of sparse-matrix VRAM on large-L2 cards (less on small-L2 ones). Speed depends on L2: on an RTX 5070 (48MB L2) it was 1.41x faster than two copies at VBITS=64, where both fit at their defaults (1163 vs 825 dims/s); on an RTX 3060 (3MB L2) it was ~19% slower than the two-copy default (251 vs 310 dims/s), and on an A100 MPI piece with 61M rows ~52% slower. The rule behind both losses: single copy only wins when its L2-sized blocks are bigger than its 2×nrows floor; when the floor binds, msieve logs a note that it will only save memory
- Two-copy block_nnz default: half of L2 worth of columns, floored at max(128M, 12 × max(nrows, ncols) × sizeof(v_t) / 32). The row-proportional part (every block pays for a full row-pointer array and a rewrite of the whole output vector) only matters for tall pieces, e.g. MPI: on the A100 piece above the fixed 128M floor was 13.5% slower than 1.75B
- `spmv_kernel=auto|segscan|warpmerge` — kernel for all gather SpMV launches (everything but the single-copy scatter). `auto` (default) picks per block: `segscan` (groups of VWORDS lanes per nonzero plus a segmented scan) below 8 nonzeros per row, else `warpmerge` (one warp per row segment)

**Streaming matrix blocks from host memory** (`common/lanczos/gpu/lanczos_matmul_gpu.c`: `plan_matrix`, `choose_streamed`, `setup_schedule`, `stream_load`):
- Automatic: if the sparse blocks don't fit in the GPU's free memory (less the Lanczos vectors and dense rows), an evenly spaced subset stays in pinned host memory and is copied into staging buffers on a separate stream every iteration, ahead of the SpMV that needs it. In single-copy mode the transpose product walks the blocks backwards so the last streamed blocks are reused. The copies are hidden while the streamed bytes per iteration copy faster than the iteration runs; on the C189 matrices they were the limit (~30-35 GB/s), so VBITS=128 (half-size vectors, more of the matrix resident) beat 256
- `max_gpu_mem=MB` — memory for matrix + vectors, used instead of the free memory and without the margin (to leave headroom, or to claim memory WDDM would give up); `stream_frac=F` — stream at least that fraction (testing); `stream_slots=N` — staging buffers (default 3; fewer are used if 3 don't fit)
- The whole block layout is planned before building (`plan_matrix`): forward blocks from the column weights, transpose blocks (two-copy) from prefix sums of per-row nonzero counts, using the same size search the old `extract_block_trans` did, so blocks are identical to before, but the transpose no longer rescans every nonzero at each search step. `gpu_matrix_init` builds exactly the planned blocks. Streamed blocks are packed straight into pinned memory; if pinning fails (WSL limits it), that block stays in ordinary memory with a warning and copies more slowly
- A margin only decides how much to stream once streaming is unavoidable: 256MB natively, 1.5GB under WDDM (native Windows or WSL, detected via `/dev/dxg`). A matrix that fits without it is loaded whole, as before. If no streaming layout fits, it warns and tries to load the whole matrix. Not used with `use_managed=1` (the streaming options are ignored with a note)
- On WSL with other GPU users (e.g. a browser), a matrix that fits still fills the card; with it nearly full, WDDM moved memory around and a C189 run got 3x slower, then hit NVIDIA driver faults. Use `max_gpu_mem` (e.g. 8500-10000 on a 12GB card) to keep headroom, or move other apps to an integrated GPU
- Several MPI ranks sharing one GPU each plan against the same free memory; give each rank its own `max_gpu_mem` share there
- Staging buffers are as large as the largest streamed block (up to 3 of them), so big blocks (e.g. VBITS=64 single-copy, or two-copy with the row floor) cost a lot of resident space when streaming; capping block size while streaming would trade that against per-block row-pointer overhead, untested

### Key data structures

- `filter_t` (`common/filter/filter.h`): holds `relation_array` (packed blob of `relation_ideal_t`), `relation_ptr` (pointer array into blob), `num_relations`, `num_ideals`, `target_excess`, `lp_file_size`
- `merge_t`: holds `relset_array`, `num_relsets`, `target_density`, `avg_cycle_weight`
- `relation_ideal_t`: variable-size struct; iterate with `next_relation_ptr(r)` macro — do not index directly

### Intermediate files

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `msieve.fb` | `-np` (poly select) | `-ns`, `-nc` |
| `msieve.dat` | sieving | `-nc1` |
| `msieve.dat.lp` | `-nc1` (internal) | `-nc1` |
| `msieve.dat.cyc[.NNN]` | `-nc1` | `-nc2` |
| `msieve.dat.mat[.NNN]` | `-nc2` | `-nc2` (Lanczos) |
| `msieve.dat.mat.idx[.NNN]` | `-nc2` (MPI mode) | `-nc2` (MPI) |
| `msieve.dat.dep` | `-nc2` | `-nc3` |
| `msieve.dat.chk` / `.bak.chk` | `-nc2` checkpoints | `-ncr` |
