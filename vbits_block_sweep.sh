#!/bin/bash
# Cross-card block_nnz / VBITS benchmark sweep.
#
# Runs fresh-start Lanczos windows (380s each) over a grid of block_nnz
# values at VBITS=64 and VBITS=256, collecting everything needed to
# compare GPU architectures (see LANCZOS_OPTIMIZATION_NOTES.md).
#
# Usage:   ./vbits_block_sweep.sh <sm>  [vbits list]
# Example: ./vbits_block_sweep.sh 70            # Tesla V100
#          ./vbits_block_sweep.sh 86 "64 128"   # RTX 3080, custom VBITS
#
# Requirements in the working directory:
#   msieve.dat.mat   - prebuilt matrix (copy from the reference job)
#   msieve.fb        - polynomial file for the same job
#   worktodo.ini     - number being factored (same job)
#   (msieve.dat may be a 0-byte stub; LA with skip_matbuild never reads it)
# Build requirements: CUDA toolkit that supports the target arch
#   (NOTE: CUDA 13 dropped sm_70/Volta - use a 12.x toolkit for V100).
# GPU must be otherwise idle.
#
# Send back: bench_results.tar.gz

set -e
SM=${1:?usage: $0 <sm, e.g. 70> [vbits list, default "64 256"]}
VBITS_LIST=${2:-"64 256"}
OUT=bench_results
# Override via env for different matrices, e.g. large matrices should drop
# the smallest blocks (per-block rowptr arrays scale with nrows x nblocks
# and can exhaust VRAM). RUN_SECS is the per-point iteration budget.
NNZ_LIST=${NNZ_LIST:-"default 512000000 256000000 128000000 64000000 32000000"}
RUN_SECS=${RUN_SECS:-380}

mkdir -p "$OUT"
rm -f "$OUT"/*

{ nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
  nvcc --version | tail -2; } | tee "$OUT/environment.txt"

[ -f msieve.dat ] || touch msieve.dat
mv -f msieve.log "$OUT/msieve.log.pre" 2>/dev/null || true

for VB in $VBITS_LIST; do
    echo "=== building VBITS=$VB for sm_$SM ==="
    make clean > /dev/null 2>&1
    make all CUDA=$SM VBITS=$VB -j"$(nproc)" > "$OUT/build_v$VB.log" 2>&1
    cuobjdump --list-elf cub/spmv_engine.so | tee -a "$OUT/build_v$VB.log"

    for nnz in $NNZ_LIST; do
        rm -f msieve.dat.chk msieve.dat.bak.chk
        args="skip_matbuild=1"
        [ "$nnz" != "default" ] && args="$args block_nnz=$nnz"
        echo "=== VBITS=$VB nnz=$nnz start $(date +%H:%M:%S) ===" | tee -a "$OUT/sweep.log"
        timeout "$RUN_SECS" ./msieve -nc2 "$args" -g 0 -t 4 \
            > "$OUT/run_v${VB}_${nnz}.out" 2>&1 || true
        echo "=== VBITS=$VB nnz=$nnz done  $(date +%H:%M:%S) ===" | tee -a "$OUT/sweep.log"
        sleep 5
    done
done

cp msieve.log "$OUT/msieve.log"
tar czf bench_results.tar.gz "$OUT"
echo "done - send back bench_results.tar.gz"
