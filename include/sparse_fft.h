#pragma once
#include "mtx_reader.h"
#include <cuComplex.h>
#include <vector>


struct SparseFFTResult {
    cuFloatComplex* d_output; // device pointer, size rows*output_cols, caller must free
    int             output_cols; // cols for full complex output, cols/2+1 for R2C-style output
    float           ms;       // total compute time in milliseconds
    size_t          mem_bytes; // peak device memory allocated (bytes)
    std::vector<cuFloatComplex> h_output; // optional host output for streamed methods
    // Added at end so existing positional brace-init returns stay valid.
    float           preprocess_ms = 0.f; // host setup: CSC build + H2D + precompute + plan (pre-kernel)
};

// FFT-style CSC variant:
//   Builds dense row chunks from binary CSC and uses a Bluestein chirp-z convolution
//   with power-of-two FFT kernels for the column transform.
// fft_backend: 0 = default two-stage Stockham loop,
//              1 = smem row kernel (auto-falls-back for fft_len > 4096),
//              3 = cuFFT C2C batched (fastest; d_work not allocated).
// NOTE: allocates full d_out = rows*output_cols on device. Use the streaming
//       variants below for matrices too large to hold the full output on GPU.
SparseFFTResult sparse_fft_csc_bluestein(const COOMatrix& coo, int u_tile = 256,
                                         bool use_stockham = false, int fft_backend = 0);
// Convenience wrappers:
SparseFFTResult sparse_fft_csc_bluestein_smem(const COOMatrix& coo, int u_tile = 256);
SparseFFTResult sparse_fft_csc_bluestein_cufft(const COOMatrix& coo, int u_tile = 256);

// Streaming Bluestein variants:
//   Keep only one tile of d_signal and two d_out_chunk slots on device; copy each
//   tile to host-pinned memory and return h_output.  Peak device memory is
//   O(tile * fft_len) — suitable for matrices where the full output won't fit on GPU.
// fft_backend: 0 = Stockham (ping-pong, d_work allocated),
//              3 = cuFFT C2C (in-place, d_work skipped — lowest device footprint).
SparseFFTResult sparse_fft_csc_stockham_streaming(const COOMatrix& coo, int u_tile = 128,
                                                   int fft_backend = 0);
SparseFFTResult sparse_fft_csc_bluestein_cufft_streaming(const COOMatrix& coo, int u_tile = 128);
SparseFFTResult sparse_fft_csc_stockham_graph(const COOMatrix& coo, int u_tile = 128);

// Experimental binary-CSC variant with byte-mask + LUT pass 1.
//   - Compresses each active column's row indices into (byte_pos, byte_mask) pairs
//     where each byte covers 8 consecutive row positions.
//   - Per u-tile, precomputes:
//       LUT_inner[u_local][mask] = Σ over set bits j in mask of exp(-2πi · j · u / rows)
//       byte_phase[u_local][byte_pos] = exp(-2πi · 8·byte_pos · u / rows)
//   - Pass 1 uses these tables instead of a sincosf per nonzero row:
//       contribution(byte_pos, mask) = byte_phase[u_local][byte_pos] · LUT_inner[u_local][mask]
//   - Bluestein convolution and finalize identical to the cuFFT streaming path,
//     so timing differences isolate the binary-specific front-end change.
SparseFFTResult sparse_fft_csc_bluestein_binary_lut(const COOMatrix& coo, int u_tile = 128);

// Experimental mixed-radix {2, 3} Stockham Bluestein variant (no cuFFT).
// Uses fft_len = next_3_smooth(2·cols−1) instead of next_pow2 — for benzene
// that's 18432 vs 32768 (~0.56× the work).  Combined with binary CSC pass-1
// (no values, packed indices) the goal is to beat dense cuFFT timing without
// any cuFFT call anywhere in the path.  Non-streaming; allocates full d_out.
SparseFFTResult sparse_fft_csc_bluestein_mixed_radix(const COOMatrix& coo, int u_tile = 128);

// Same algorithm as sparse_fft_csc_bluestein_mixed_radix but the entire tile
// loop is captured into a CUDA Graph and launched with a single API call,
// reducing per-tile kernel-launch latency on small matrices where launch
// overhead (~5 µs × hundreds of launches) dominates the runtime.
SparseFFTResult sparse_fft_csc_bluestein_mixed_radix_graph(const COOMatrix& coo, int u_tile = 128);

// "Best-possible cuFFT" baseline for the README's cuFFT comparison.
// Same Bluestein scaffolding as sparse_fft_csc_bluestein_cufft, but with
// fft_len = next_7_smooth(2·cols−1) instead of next_pow2.  cuFFT efficiently
// handles 7-smooth sizes via its own internal mixed-radix; for benzene this
// cuts fft_len from 32768 down to 16464, halving FFT work.  Non-streaming.
SparseFFTResult sparse_fft_csc_bluestein_cufft_smooth(const COOMatrix& coo, int u_tile = 128);

// Conjugate-symmetric variant of sparse_fft_csc_bluestein_cufft_smooth
// (Lever B): the input matrix is real, so F[rows-u][v] =
// conj(F[u][(cols-v) mod cols]).  The column transform runs only for
// u in [0, rows/2] (rows/2 + 1 rows) and a dual-write finalize kernel
// fills the mirror rows by conjugation, halving every per-tile kernel.
SparseFFTResult sparse_fft_csc_bluestein_cufft_smooth_sym(const COOMatrix& coo, int u_tile = 128);

// Experimental non-streaming binary-CSC variant using the Stockham smem FFT path.
// Same byte-mask + LUT pass 1 as sparse_fft_csc_bluestein_binary_lut, but the
// Bluestein convolution uses run_fft_power2_stockham_smem_stream (auto-falls
// back to the per-stage Stockham loop for fft_len > SMEM_MAX_FFT_N) and the
// fused stockham_smem_row_mul_inverse_kernel for the inverse half — no cuFFT.
// Allocates the full d_out = rows × output_stride on device (non-streaming).
//
// Set the environment variable CUSPFFT_BREAKDOWN=1 to enable per-phase timing
// breakdown (LUT precompute / pass-1 / FFT-mul-IFFT / finalize), printed to
// stdout after the main timing line.  Per-tile sync adds small overhead, so
// the reported total may be slightly higher than without breakdown.
SparseFFTResult sparse_fft_csc_stockham_binary_smem(const COOMatrix& coo, int u_tile = 128);
