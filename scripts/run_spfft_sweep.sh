#!/usr/bin/env bash
# Run the SpFFT-only sub-sweep on the host-RAM-safe matrix list.
#
# Output is segregated to logs_spfft/ so the main logs/ directory stays clean
# until you explicitly merge SpFFT rows into logs/results.csv.
#
# Hardcodes env vars to avoid the line-continuation / trailing-whitespace trap
# that breaks `VAR=value \` shell pipelines.
#
# Usage:
#     ./scripts/run_spfft_sweep.sh
#
# After it finishes:
#     awk -F, 'NR > 1 && $6 ~ /SpFFT/' logs_spfft/results.csv >> logs/results.csv
#     .venv/bin/python scripts/plot_results.py

set -uo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [ ! -f dataset_lists/spfft_safe.txt ]; then
    echo "error: dataset_lists/spfft_safe.txt missing." >&2
    echo "       Generate it first: .venv/bin/python scripts/list_spfft_safe.py" >&2
    exit 1
fi

export MATRIX_LIST=dataset_lists/spfft_safe.txt
export LOG_DIR=logs_spfft
export FLAGS="--spfft --sparse-only --repeat 10"
export TIMEOUT=900

echo "Running SpFFT sub-sweep:"
echo "  MATRIX_LIST = $MATRIX_LIST  ($(wc -l < "$MATRIX_LIST") entries)"
echo "  LOG_DIR     = $LOG_DIR"
echo "  FLAGS       = $FLAGS"
echo "  TIMEOUT     = ${TIMEOUT}s"
echo

exec ./scripts/run_sweep.sh
