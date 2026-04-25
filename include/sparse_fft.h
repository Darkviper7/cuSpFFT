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
};

// Direct sparse DFT with shared-memory tiling.
// Each output bin (u,v) accumulates contributions from all nnz entries.
// COO arrays are tiled through shared memory to reduce global-memory traffic.
// Complexity: O(nnz * rows * cols)
// Memory:     O(nnz + rows*cols)
SparseFFTResult sparse_fft(const COOMatrix& coo);

// Two-pass sparse FFT (COO):
//   Pass 1 — for each nonzero (r,c), scatter exp(-2πi*r*u/rows) into
//            compact G[sparse_col_id(c)][u]
//   Pass 2 — custom column DFT over only the columns present in the input,
//            producing an R2C-style half spectrum with cols/2+1 columns
// Complexity: O(nnz * rows  +  rows * (cols/2+1) * n_sparse_cols)
// Memory:     O(nnz + rows*n_sparse_cols + rows*(cols/2+1))
// For rows/cols <= 65536, COO indices are packed into one 32-bit word and
// sparse column lists are stored as 16-bit entries on device.
// Use this when nnz << rows*cols (typical sparse case).
// Will OOM for very large matrices (>50k nodes) due to G allocation.
SparseFFTResult sparse_fft_2pass(const COOMatrix& coo);

// Hybrid two-pass sparse FFT (COO + cuFFT):
//   Pass 1 — same COO row-frequency scatter as sparse_fft_2pass, but into a
//            dense column-major G[c][u] so missing columns are explicit zeros
//   Pass 2 — batched cuFFT C2C over each output row u
// Produces the full complex spectrum with output_cols == cols.
//
// This is mainly a comparison point against the custom sparse-column DFT:
// cuFFT makes pass 2 much faster, but it requires full rows*cols intermediate
// and output storage instead of compact rows*n_sparse_cols plus half spectrum.
SparseFFTResult sparse_fft_2pass_cufft(const COOMatrix& coo);

// GEMM two-pass sparse FFT (COO + cuBLAS):
//   Pass 1 — same compact COO scatter as sparse_fft_2pass
//   Pass 2 — materialize W[i,v] = exp(-2πi*sparse_cols[i]*v/cols), then call
//            cuBLAS Cgemm for F = G^T * W
// Produces an R2C-style half spectrum with output_cols == cols/2+1.
//
// This preserves compact G but spends extra memory on dense W so pass 2 can use
// optimized GEMM instead of the custom scalar DFT kernel.
SparseFFTResult sparse_fft_2pass_gemm(const COOMatrix& coo, bool use_tf32 = false);

// Two-pass sparse FFT (binary CSC):
//   Pass 1 — group the binary input by active column and compute compact
//            G[sparse_col_id][u] without atomics
//   Pass 2 — same custom sparse-column DFT as COO/CSR
// Produces an R2C-style half spectrum with output_cols == cols/2+1.
//
// This format stores no values and no empty columns. For rows/cols <= 65536,
// row indices and sparse column lists use 16-bit device storage.
SparseFFTResult sparse_fft_csc_2pass(const COOMatrix& coo, int u_tile = 1024,
                                      bool precompute_twiddle = false,
                                      bool tiled_pass2 = false,
                                      bool half_g = false);

// Streaming CSC variant:
//   Keeps only G_chunk and F_chunk on device, copies each output-row tile to
//   host, and returns h_output instead of d_output.  This minimizes device
//   memory for large matrices while preserving the same FP32 output format.
SparseFFTResult sparse_fft_csc_streaming(const COOMatrix& coo, int u_tile = 1024);

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

// Two-pass sparse FFT (CSR):
// Same two-pass decomposition as sparse_fft_2pass, but built on CSR format.
//
// Algorithmic advantages over COO 2-pass:
//   1. Fewer sincosf calls: COO calls sincosf once per (nonzero, output_row),
//      i.e. nnz*rows times.  CSR calls it once per (source_row, output_row),
//      i.e. rows² times.  Saving factor = nnz/rows = avg nonzeros per row.
//   2. Coalesced CSR column-index reads: col_idx entries for a given source row
//      are contiguous in memory, improving L1/L2 cache hit rate vs COO where
//      row membership must be inferred from the row_idx array.
//   3. For cols <= 65536, CSR column indices and sparse column lists are stored
//      as 16-bit entries on device.
//
// Complexity: O(rows² + nnz * rows  +  rows * (cols/2+1) * n_sparse_cols)
// Memory:     O(nnz + row_ptr + rows*n_sparse_cols + rows*(cols/2+1))
SparseFFTResult sparse_fft_csr_2pass(const CSRMatrix& csr);
