#!/usr/bin/env bash
# Profile cuSpFFT with Nsight Compute (ncu) and Nsight Systems (nsys).
#
# Usage:
#   ./scripts/profile.sh [matrix.mtx]
#
# Three profiler runs, all written to profiles/:
#
#   ncu_<name>_sparse.ncu-rep  — all custom sparse kernels, matched by name
#   ncu_<name>_dense.ncu-rep   — dense scatter + cuFFT kernels, via NVTX range
#   nsys_<name>.nsys-rep       — end-to-end CUDA/NVTX timeline
#
# Open with:
#   ncu-ui profiles/<file>.ncu-rep
#   nsys-ui profiles/<file>.nsys-rep
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
    echo "ERROR: binary not found at $BINARY — run cmake --build build-release first"
    exit 1
fi
if [ ! -f "$MTX" ]; then
    echo "ERROR: matrix not found: $MTX"
    exit 1
fi

NAME="$(basename "$MTX" .mtx)"
NCU=(sudo HOME="$HOME" /usr/local/cuda-12.5/bin/ncu)
NSYS=(/usr/local/cuda-12.5/bin/nsys)
mkdir -p "$PROFILES"

echo "=========================================="
echo "cuSpFFT profile: $NAME"
echo "Binary: $BINARY"
echo "=========================================="

EXTRA_ARGS="${@:2}"   # all arguments after the matrix path forwarded to the binary

OUT_SPARSE="${PROFILES}/ncu_${NAME}_sparse"
echo ""
echo "[1/3] Profiling Sparse CSC Stockham kernels -> $(basename ${OUT_SPARSE}).ncu-rep"

"${NCU[@]}" \
    --set full \
    --kernel-name 'regex:^(csc_build_bluestein_input_coalesced.*|stockham_smem_row.*|pointwise_mul_batched_kernel|scale_complex_kernel|bluestein_finalize_kernel)$' \
    --export "${OUT_SPARSE}" \
    --force-overwrite \
    --print-summary per-kernel \
    "$BINARY" "$MTX" --sparse-only $EXTRA_ARGS 2>&1 | tee "${OUT_SPARSE}.txt"

echo "    ncu-ui ${OUT_SPARSE}.ncu-rep"

# ---------------------------------------------------------------------------
# Run 1: Sparse CSC Stockham kernels — matched by kernel name
#
# Captures the current no-cuFFT optimized path:
#   csc_build_bluestein_input_coalesced_packed_kernel
#   stockham_two_stage_kernel
#   stockham_stage_mul_kernel
#   stockham_stage_kernel
#   scale_complex_kernel
#   bluestein_finalize_kernel
#
# We use kernel-name filtering instead of --nvtx-include here because main.cu
# uses push/pop NVTX ranges, while unqualified NCU NVTX filters match start/end
# ranges.  The kernel regex is less fragile for this benchmark.
# # ---------------------------------------------------------------------------
# OUT_SPARSE="${PROFILES}/ncu_${NAME}_sparse"
# echo ""
# echo "[1/3] Profiling Sparse CSC Stockham kernels -> $(basename ${OUT_SPARSE}).ncu-rep"

# "${NCU[@]}" \
#     --set full \
#     --kernel-name 'regex:^(csc_build_bluestein_input_coalesced.*|stockham_.*|scale_complex_kernel|bluestein_finalize_kernel)$' \
#     --export "${OUT_SPARSE}" \
#     --force-overwrite \
#     --print-summary per-kernel \
#     "$BINARY" "$MTX" --csc-bluestein --csc-stockham --csc-tile 128 2>&1 | tee "${OUT_SPARSE}.txt"

# echo "    ncu-ui ${OUT_SPARSE}.ncu-rep"

# # ---------------------------------------------------------------------------
# # Run 2: Dense baseline — captured via NVTX range filter
# #
# # cuFFT launches kernels with auto-generated internal names that cannot be
# # matched by regex. Instead, --nvtx-include captures every kernel launched
# # within the "Dense cuFFT (baseline)" NVTX range defined in main.cu.
# # This includes the COO-to-dense scatter plus cufftExecR2C transform kernels.
# # ---------------------------------------------------------------------------
# OUT_DENSE="${PROFILES}/ncu_${NAME}_dense"
# echo ""
# echo "[2/3] Profiling dense cuFFT kernels (via NVTX range) -> $(basename ${OUT_DENSE}).ncu-rep"

# "${NCU[@]}" \
#     --set basic \
#     --nvtx \
#     --nvtx-include "Dense cuFFT (baseline)" \
#     --export "${OUT_DENSE}" \
#     --force-overwrite \
#     --print-summary per-kernel \
#     "$BINARY" "$MTX" --dense-only 2>&1 | tee "${OUT_DENSE}.txt"

# echo "    ncu-ui ${OUT_DENSE}.ncu-rep"

# # ---------------------------------------------------------------------------
# # Run 3: Nsight Systems timeline
# #
# # Captures CUDA API calls, GPU kernels/memcpy activity, NVTX ranges, and
# # OS-runtime events for the full benchmark run. This complements ncu by showing
# # launch order, CPU/GPU overlap, allocation/copy overheads, and range timing.
# # ---------------------------------------------------------------------------
# OUT_NSYS="${PROFILES}/nsys_${NAME}"
# echo ""
# echo "[3/3] Profiling full timeline -> $(basename ${OUT_NSYS}).nsys-rep"

# "${NSYS[@]}" profile \
#     --trace=cuda,nvtx,osrt \
#     --sample=none \
#     --force-overwrite=true \
#     --output="${OUT_NSYS}" \
#     "$BINARY" "$MTX" 2>&1 | tee "${OUT_NSYS}.txt"

# echo "    nsys-ui ${OUT_NSYS}.nsys-rep"

echo ""
echo "Profile files:"
ls -lh "$PROFILES/"*"${NAME}"*
