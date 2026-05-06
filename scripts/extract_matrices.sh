#!/usr/bin/env bash
# Bulk-extract SuiteSparse archives in dataset/ into the
# `dataset/{name}/{name}.mtx` layout the rest of the project expects.
#
# Usage:
#   ./scripts/extract_matrices.sh           # acts on dataset/
#   ./scripts/extract_matrices.sh other_dir # acts on other_dir/

set -euo pipefail

DATA_DIR="${1:-dataset}"

if [ ! -d "$DATA_DIR" ]; then
    echo "error: '$DATA_DIR' is not a directory" >&2
    exit 1
fi

shopt -s nullglob
count=0
for f in "$DATA_DIR"/*.tar.gz "$DATA_DIR"/*.tgz "$DATA_DIR"/*.tar; do
    echo "Extracting $(basename "$f")"
    tar -xf "$f" -C "$DATA_DIR/"
    rm "$f"
    count=$((count + 1))
done

if [ "$count" -eq 0 ]; then
    echo "No .tar / .tar.gz / .tgz archives found in $DATA_DIR/"
else
    echo "Extracted $count archive(s)."
fi

mtx_count=$(ls -1 "$DATA_DIR"/*/*.mtx 2>/dev/null | wc -l)
echo "Matrices now available under $DATA_DIR/: $mtx_count"
