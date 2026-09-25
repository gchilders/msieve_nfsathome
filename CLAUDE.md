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

Clean: `make clean`

There are no automated tests. Validation is done by running actual NFS factorizations.

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
- Runs disk-based singleton removal if the LP file is large (`common/filter/singleton.c`: `filter_purge_lp_singletons`)
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
- `single_copy=1` — stores only A on the GPU, not A^T; the transpose product scatters through A's column-slice blocks with atomic XOR (`spmv_engine_run_trans`). Its block_nnz default is a third of L2 worth of columns, floored at 2×nrows so the per-block row-pointer arrays stay under half the column indices; that saves ~40% of sparse-matrix VRAM on large-L2 cards (less on small-L2 ones). Speed depends on L2: on an RTX 5070 (48MB L2) it was 1.41x faster than two copies at VBITS=64, where both fit at their defaults (1163 vs 825 dims/s); on an RTX 3060 (3MB L2) it was ~19% slower than the two-copy default (251 vs 310 dims/s), so there it is only worth using when two copies don't fit
- `spmv_kernel=auto|segscan|warpmerge` — kernel for all gather SpMV launches (everything but the single-copy scatter). `auto` (default) picks per block: `segscan` (groups of VWORDS lanes per nonzero plus a segmented scan) below 8 nonzeros per row, else `warpmerge` (one warp per row segment)

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
