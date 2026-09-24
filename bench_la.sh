#!/bin/bash
# Time a short Lanczos run on a prebuilt matrix and report the ETA.
#
# Starts `msieve -nc2 "skip_matbuild=1 <args>"`, waits for the matrix to load
# and the first progress line to appear, lets it warm up, then measures
# dimensions/second over a fixed window and derives a full-run ETA from that.
# Then it stops msieve with SIGTERM, which makes Lanczos run its integrity
# check before writing a checkpoint, so a wrong matrix multiply shows up as
# FAIL (msieve also self-checks every 10000 dimensions while running).
#
# Each run gets its own directory under bench/ with symlinks to the matrix,
# so checkpoints, logs and .dep files never touch the real job files.
#
# Usage:   ./bench_la.sh [options] [-- extra msieve options]
#   -d N      use msieve.dat.mat.N / msieve.dat.cyc.N (default: msieve.dat.mat)
#   -a ARGS   extra -nc2 arguments, e.g. "single_copy=1 block_nnz=256000000"
#   -T SECS   measurement window (default 300)
#   -W SECS   warmup after the first progress line (default 30)
#   -g N      GPU index, passed to msieve and used for VRAM sampling (default 0)
#   -n NAME   label for the run (default: derived from -d and -a)
#   -F        full validation: after measuring, let Lanczos finish, then run
#             the square root (-nc3) in the same directory and report the
#             factors. All output (.dep etc.) stays in the bench directory
#
# Examples:
#   ./bench_la.sh -d 90
#   ./bench_la.sh -d 90 -a "single_copy=1"
#   ./bench_la.sh -a "use_managed=1" -T 600
#   ./bench_la.sh -d 90 -a "single_copy=1" -- -t 4
#   ./bench_la.sh -d 90 -a "single_copy=1" -F
#
# One summary line per run is appended to bench/results.tsv.

density=""
nc2_args=""
window=300
warmup=30
gpu=0
name=""
full=0

while getopts "d:a:T:W:g:n:Fh" opt; do
    case $opt in
        d) density=$OPTARG ;;
        a) nc2_args=$OPTARG ;;
        T) window=$OPTARG ;;
        W) warmup=$OPTARG ;;
        g) gpu=$OPTARG ;;
        n) name=$OPTARG ;;
        F) full=1 ;;
        *) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
[ "$1" = "--" ] && shift
extra_opts=("$@")

suffix=${density:+.$density}
mat="msieve.dat.mat$suffix"
cyc="msieve.dat.cyc$suffix"
for f in "$mat" msieve.fb worktodo.ini ./msieve; do
    [ -e "$f" ] || { echo "error: $f not found"; exit 1; }
done

if [ -z "$name" ]; then
    name="td${density:-default}"
    [ -n "$nc2_args" ] && name="$name-$(echo "$nc2_args" | tr ' =' '_-')"
fi
stamp=$(date +%Y%m%d-%H%M%S)
dir="bench/$name-$stamp"
mkdir -p "$dir"

ln -s "$PWD/$mat" "$dir/msieve.dat.mat"
[ -e "msieve.dat.mat.idx$suffix" ] && \
    ln -s "$PWD/msieve.dat.mat.idx$suffix" "$dir/msieve.dat.mat.idx"
[ -e "$cyc" ] && ln -s "$PWD/$cyc" "$dir/msieve.dat.cyc"
[ -e msieve.dat ] && ln -s "$PWD/msieve.dat" "$dir/msieve.dat"

# make msieve's -g index and nvidia-smi -i refer to the same card
export CUDA_DEVICE_ORDER=PCI_BUS_ID

vram() {
    nvidia-smi -i "$gpu" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

# prints "<dims solved> <total dims> <msieve ETA>" from the latest progress line
progress() {
    tr '\r' '\n' < "$dir/stderr.txt" | grep 'linear algebra completed' | \
        tail -1 | sed -E 's/.*completed ([0-9]+) of ([0-9]+) .*ETA *([^)]*)\).*/\1 \2 \3/'
}

vram_base=$(vram)
vram_peak=${vram_base:-0}
echo "run:     $dir"
echo "matrix:  $mat${cyc:+ (cycles: $cyc)}"
echo "args:    -nc2 \"skip_matbuild=1 $nc2_args\" -g $gpu ${extra_opts[*]}"
echo "VRAM in use before start: ${vram_base:-?} MiB"

./msieve -s "$dir/msieve.dat" -l "$dir/msieve.log" -g "$gpu" \
    "${extra_opts[@]}" -nc2 "skip_matbuild=1 $nc2_args" \
    > "$dir/stdout.txt" 2> "$dir/stderr.txt" &
pid=$!
trap 'kill -TERM $pid 2>/dev/null; wait $pid; exit 130' INT TERM

sample_vram() {
    local v
    v=$(vram)
    [ -n "$v" ] && [ "$v" -gt "$vram_peak" ] && vram_peak=$v
}

# phase 1: matrix load, until the first progress line
start=$(date +%s)
while [ -z "$(progress)" ]; do
    if ! kill -0 $pid 2>/dev/null; then
        break
    fi
    sample_vram
    sleep 2
done
load_secs=$(( $(date +%s) - start ))

status=OK
if ! kill -0 $pid 2>/dev/null; then
    status=FAIL
    echo "msieve exited before Lanczos started; see $dir/stdout.txt"
    tail -5 "$dir/stdout.txt"
else
    echo "matrix loaded after ${load_secs}s, warming up for ${warmup}s"

    # phase 2: warmup
    end=$(( $(date +%s) + warmup ))
    while [ "$(date +%s)" -lt "$end" ] && kill -0 $pid 2>/dev/null; do
        sample_vram
        sleep 1
    done

    # phase 3: measure; timestamps are taken when a new progress line
    # first appears, so poll every second
    read -r d0 total _ <<< "$(progress)"
    while [ "$(progress | cut -d' ' -f1)" = "$d0" ] && kill -0 $pid 2>/dev/null; do
        sleep 1
    done
    read -r d0 total _ <<< "$(progress)"
    t0=$(date +%s.%N)
    t1=$t0; d1=$d0
    end=$(( $(date +%s) + window ))
    while [ "$(date +%s)" -lt "$end" ] && kill -0 $pid 2>/dev/null; do
        read -r d total eta <<< "$(progress)"
        if [ "$d" != "$d1" ]; then
            d1=$d
            t1=$(date +%s.%N)
            msieve_eta=$eta
        fi
        sample_vram
        sleep 1
    done
    # in -F mode a solve that already finished is fine; the exit code
    # below still catches a crash
    [ $full = 1 ] || kill -0 $pid 2>/dev/null || status=FAIL

    if [ $full = 1 ]; then
        echo "measured; waiting for linear algebra to finish (msieve ETA ${msieve_eta:-?})"
    else
        # stop msieve; SIGTERM triggers an integrity check plus a checkpoint
        kill -TERM $pid 2>/dev/null
    fi
fi

wait $pid 2>/dev/null
rc=$?
# msieve exits 0 after a SIGTERM shutdown; errors (including a failed
# integrity check) exit nonzero
[ $rc -ne 0 ] && status=FAIL
grep -qE 'error:|corrupt state' "$dir/stdout.txt" "$dir/msieve.log" 2>/dev/null && status=FAIL
rm -f "$dir"/*.chk

factors=-
if [ $full = 1 ] && [ "$status" = OK ]; then
    if grep -q "recovered .* nontrivial dependencies" "$dir/msieve.log"; then
        echo "linear algebra done; running square root"
        ./msieve -s "$dir/msieve.dat" -l "$dir/msieve.log" -g "$gpu" \
            "${extra_opts[@]}" -nc3 >> "$dir/stdout.txt" 2>> "$dir/stderr.txt"
        factors=$(grep -oE "p(rp)?[0-9]+ factor: [0-9]+" "$dir/msieve.log" | \
                  awk '{print $NF}' | paste -sd' ')
        [ -n "$factors" ] || { factors=none; status=FAIL; }
    else
        echo "linear algebra did not report dependencies"
        status=FAIL
    fi
fi

if [ "$status" = OK ] && [ "${d1:-0}" -gt "${d0:-0}" ]; then
    read -r rate eta_h iter_ms <<< "$(awk -v d0="$d0" -v d1="$d1" \
        -v t0="$t0" -v t1="$t1" -v n="$total" -v vb="$(grep -o 'VBITS=[0-9]*' \
        "$dir/msieve.log" | head -1 | cut -d= -f2)" 'BEGIN {
            r = (d1 - d0) / (t1 - t0)
            printf "%.1f %.2f %.1f\n", r, n / r / 3600, 1000 * (vb - 0.76) / r
        }')"
    eta_fmt=$(awk -v h="$eta_h" 'BEGIN { printf "%dh%02dm", int(h), (h - int(h)) * 60 }')
else
    rate=-; eta_fmt=-; iter_ms=-
fi
[ "$status" = OK ] || { echo "run FAILED:"; tr '\r' '\n' < "$dir/stderr.txt" | tail -3;
                        tail -5 "$dir/stdout.txt"; }

sparse_mb=$(grep -o 'sparse matrix memory use: [0-9.]* MB' "$dir/stdout.txt" | awk '{print $5}')

echo
echo "status:               $status"
echo "dims/sec:             $rate"
echo "full-run ETA:         $eta_fmt (from dims/sec over the whole matrix)"
echo "msieve's ETA:         ${msieve_eta:--} (remaining, at last progress line)"
echo "ms per iteration:     $iter_ms (approx, VBITS-0.76 dims/iteration)"
echo "sparse matrix on GPU: ${sparse_mb:--} MB"
echo "peak VRAM:            $vram_peak MiB (${vram_base:-?} MiB in use before start)"
[ $full = 1 ] && echo "factors:              $factors"

results=bench/results.tsv
[ -f $results ] || printf "date\tname\tdensity\targs\tstatus\tdims_per_sec\teta_full\tms_per_iter\tsparse_mb\tpeak_vram_mib\tbase_vram_mib\tfactors\tdir\n" > $results
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$stamp" "$name" \
    "${density:-default}" "$nc2_args ${extra_opts[*]}" "$status" "$rate" \
    "$eta_fmt" "$iter_ms" "${sparse_mb:--}" "$vram_peak" "${vram_base:-?}" "$factors" "$dir" >> $results
