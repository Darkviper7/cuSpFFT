#!/usr/bin/env bash
# Profile cuSpFFT with Nsight Compute (ncu).
#
# Usage:
#   ./scripts/profile.sh [matrix.mtx] [cuSpFFT options...]
#
# Output, written to profiles/:
#
#   ncu_<name>_sparse.ncu-rep  — selected sparse kernels, matched by name
#
# Open with:
#   ncu-ui profiles/<file>.ncu-rep
#
# Requires sudo for GPU hardware counters (ERR_NVGPUCTRPERM).
# One-time fix to avoid sudo (requires reboot):
#   sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
#       > /etc/modprobe.d/nvidia.conf'

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SCRIPT_DIR}/.."

if [ -f "${ROOT}/build-release/cuSpFFT" ]; then
    BINARY="${ROOT}/build-release/cuSpFFT"
else
    BINARY="${ROOT}/build/cuSpFFT"
fi

MTX="${1:-${ROOT}/dataset/sstmodel/sstmodel.mtx}"
PROFILES="${ROOT}/profiles"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found at $BINARY — run cmake --build build first"
    exit 1
fi
if [ ! -f "$MTX" ]; then
    echo "ERROR: matrix not found: $MTX"
    exit 1
fi

NAME="$(basename "$MTX" .mtx)"
NCU_BIN="${NCU_BIN:-${CUDA_HOME:+${CUDA_HOME}/bin/}ncu}"
NCU=(sudo HOME="$HOME" "$NCU_BIN")
mkdir -p "$PROFILES"

echo "=========================================="
echo "cuSpFFT profile: $NAME"
echo "Binary: $BINARY"
echo "=========================================="

EXTRA_ARGS="${@:2}"   # all arguments after the matrix path forwarded to the binary

OUT_SPARSE="${PROFILES}/ncu_${NAME}_sparse"
echo ""
echo "[1/1] Profiling sparse CSC kernels -> $(basename ${OUT_SPARSE}).ncu-rep"

"${NCU[@]}" \
    --set full \
    --kernel-name 'regex:^(csc_build_bluestein_input.*|stockham_stage_kernel|stockham_radix3_stage_kernel|stockham_radix9_stage_kernel|pointwise_mul_batched_kernel|scale_complex_kernel|bluestein_finalize_kernel)$' \
    --export "${OUT_SPARSE}" \
    --force-overwrite \
    --print-summary per-kernel \
    "$BINARY" "$MTX" --sparse-only $EXTRA_ARGS 2>&1 | tee "${OUT_SPARSE}.txt"

echo "    ncu-ui ${OUT_SPARSE}.ncu-rep"

echo ""
echo "Profile files:"
ls -lh "$PROFILES/"*"${NAME}"*
