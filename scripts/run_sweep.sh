#!/usr/bin/env bash
# Sweep every dataset/{name}/{name}.mtx through cuSpFFT, capturing per-matrix
# raw stdout into logs/{name}.log, a global sweep transcript into
# logs/sweep_<timestamp>.log, and a structured row per (matrix, method)
# appended to logs/results.csv.
#
# Knobs (env overridable):
#   BIN          path to the cuSpFFT binary           (default: ./build/cuSpFFT)
#   LOG_DIR      output directory                      (default: logs)
#   FLAGS        benchmark flags                       (default: GPU-only suite)
#   TIMEOUT      per-matrix wall-clock cap, secs       (default: 600)
#   MATRIX_LIST  file listing one .mtx path per line   (default: dataset/*/*.mtx glob)
#                Lines starting with '#' or empty lines are ignored.
#
# Examples:
#   ./scripts/run_sweep.sh
#   FLAGS="--spfft --csc-mixed-radix --csc-cufft-smooth --csc-tile 128 --repeat 10" \
#       TIMEOUT=1800 ./scripts/run_sweep.sh
#   BIN=./build-release/cuSpFFT ./scripts/run_sweep.sh
#   # SpFFT-only sub-sweep on a curated host-RAM-safe list, separate log dir:
#   MATRIX_LIST=dataset_lists/spfft_safe.txt LOG_DIR=logs_spfft \
#       FLAGS="--spfft --repeat 10" ./scripts/run_sweep.sh

# NOTE: deliberately no `set -e` — one failed run must not abort the sweep.
set -uo pipefail

BIN="${BIN:-./build/cuSpFFT}"
LOG_DIR="${LOG_DIR:-logs}"
FLAGS="${FLAGS:---csc-mixed-radix --csc-cufft-smooth --csc-tile 128 --repeat 10}"
TIMEOUT="${TIMEOUT:-600}"

if [ ! -x "$BIN" ]; then
    echo "error: '$BIN' not found or not executable. Build first or set BIN=..." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
AWK_SCRIPT="$SCRIPT_DIR/parse_log.awk"
if [ ! -f "$AWK_SCRIPT" ]; then
    echo "error: $AWK_SCRIPT not found" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG="$LOG_DIR/sweep_${STAMP}.log"
CSV="$LOG_DIR/results.csv"

CSV_HEADER="timestamp,matrix,rows,cols,nnz,method,median_ms,min_ms,max_ms,n_iters,mem_mb,max_abs,max_rel,rms_abs,error"

# Migrate an older 14-column CSV out of the way so the new 15-column schema
# (with trailing 'error' field) doesn't end up appended to a header that
# doesn't declare it.
if [ -f "$CSV" ] && ! head -1 "$CSV" | grep -q ',error$'; then
    BACKUP="${CSV%.csv}_legacy_${STAMP}.csv"
    echo "Existing $CSV uses pre-error-column schema; rotating to $BACKUP" >&2
    mv "$CSV" "$BACKUP"
fi

if [ ! -f "$CSV" ]; then
    echo "$CSV_HEADER" > "$CSV"
fi

if [ -n "${MATRIX_LIST:-}" ]; then
    if [ ! -f "$MATRIX_LIST" ]; then
        echo "error: MATRIX_LIST=$MATRIX_LIST not found" >&2
        exit 1
    fi
    mapfile -t matrices < <(grep -vE '^[[:space:]]*(#|$)' "$MATRIX_LIST")
    SOURCE_DESC="MATRIX_LIST=$MATRIX_LIST"
else
    shopt -s nullglob
    matrices=(dataset/*/*.mtx)
    SOURCE_DESC="dataset/*/*.mtx"
fi
if [ "${#matrices[@]}" -eq 0 ]; then
    echo "error: no matrices to sweep ($SOURCE_DESC matched nothing)" >&2
    exit 1
fi

echo "Sweep $STAMP: ${#matrices[@]} matrices ($SOURCE_DESC) -> $SWEEP_LOG"
echo "FLAGS: $FLAGS"
echo "TIMEOUT: ${TIMEOUT}s per matrix"
echo

for mtx in "${matrices[@]}"; do
    name=$(basename "$(dirname "$mtx")")
    per_log="$LOG_DIR/${name}.log"
    {
        echo
        echo "=== $name ($mtx) ==="
        date
    } | tee -a "$SWEEP_LOG"

    timeout "$TIMEOUT" "$BIN" "$mtx" $FLAGS 2>&1 \
        | tee "$per_log" \
        | tee -a "$SWEEP_LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "(exit $rc — continuing)" | tee -a "$SWEEP_LOG"
    fi

    awk -v MATRIX="$name" -v TS="$STAMP" -f "$AWK_SCRIPT" "$per_log" >> "$CSV"
done

echo
echo "Sweep complete."
echo "  per-matrix logs : $LOG_DIR/<matrix>.log"
echo "  sweep transcript: $SWEEP_LOG"
echo "  CSV             : $CSV"
