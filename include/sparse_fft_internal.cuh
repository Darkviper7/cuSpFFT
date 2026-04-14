#pragma once
// sparse_fft_internal.cuh -- shared includes, macros, constants, and inline helpers
// included by all sparse_fft_*.cu translation units.

#include "sparse_fft.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess)                                               \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err));                \
    } while (0)

#define CUFFT_CHECK(call)                                                     \
    do {                                                                      \
        cufftResult r = (call);                                               \
        if (r != CUFFT_SUCCESS)                                               \
            throw std::runtime_error("cuFFT error code: " +                   \
                                     std::to_string(r));                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t s = (call);                                            \
        if (s != CUBLAS_STATUS_SUCCESS)                                       \
            throw std::runtime_error("cuBLAS error code: " +                  \
                                     std::to_string((int)s));                 \
    } while (0)

// ---------------------------------------------------------------------------
// Shared __device__ helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ int packed_row_u16(uint32_t p) {
    return (int)(p >> 16);
}

__device__ __forceinline__ int packed_col_u16(uint32_t p) {
    return (int)(p & 0xffffu);
}

__device__ __forceinline__ int phase_idx(int a, int b, int period) {
    return (int)(((uint32_t)a * (uint32_t)b) % (uint32_t)period);
}

__device__ __forceinline__ void sincos_twiddle(
    int a, int b, int period, float inv_period, float* s, float* c)
{
    const float two_pi = 6.28318530717958647f;
    const int idx = phase_idx(a, b, period);
    __sincosf(-two_pi * (float)idx * inv_period, s, c);
}

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

// Direct DFT / COO
static constexpr int TILE     = 1024;
static constexpr int BLOCK_U  = 128;
static constexpr int NNZ_TILE = 128; // must equal BLOCK_U for 1:1 loading

// CSC build pass-1
static constexpr int CSC_BU          = 128;
static constexpr int CSC_BUILD_COLS  = 32;
static constexpr int CSC_BUILD_U     = 8;

// CSR build pass-1
static constexpr int CSR_BU           = 32;
static constexpr int CSR_BR           =  8;
static constexpr int SMEM_COLS        = 32;
static constexpr int SMEM_COLS_PADDED = SMEM_COLS + 1;

// Pass-2 column DFT
static constexpr int P2_BV     = 32;   // output-freq threads per block (x)
static constexpr int P2_BU     = 32;   // output-row  threads per block (y)
static constexpr int COL_TILE  = 128;
static constexpr int CSC_TILED_U   = 16;
static constexpr int CSC_TILED_V   = 16;
// Halving C from 64->32 cuts smem from 32 KB to 16 KB, allowing 6 blocks/SM
static constexpr int CSC_TILED_C   = 32;
// Register-blocking factors for col_dft_chunk_tiled_kernel[_packed].
static constexpr int CSC_TILED_RBU = 2;
// RBV=4: each block covers CSC_TILED_V*RBV=64 output columns.
static constexpr int CSC_TILED_RBV = 4;

// Stockham smem FFT (fft_len <= SMEM_MAX_FFT_N fits in smem ping-pong)
static constexpr int SMEM_MAX_FFT_N = 4096; // 2*4096*8 = 64 KB <= 96 KB V100 smem

// ---------------------------------------------------------------------------
// Host helper functions (inline for header use)
// ---------------------------------------------------------------------------
inline std::vector<int> make_sparse_cols(const int* col_idx, int nnz) {
    std::vector<int> cols(col_idx, col_idx + nnz);
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    return cols;
}

inline std::vector<int> make_sparse_col_ids(
    const int* col_idx, int nnz, const std::vector<int>& sparse_cols, int cols)
{
    std::vector<int> col_to_slot(cols, -1);
    for (int i = 0; i < (int)sparse_cols.size(); i++)
        col_to_slot[sparse_cols[i]] = i;

    std::vector<int> ids(nnz);
    for (int i = 0; i < nnz; i++)
        ids[i] = col_to_slot[col_idx[i]];
    return ids;
}

inline bool can_pack_u16(int rows, int cols) {
    return rows <= 65536 && cols <= 65536;
}

inline std::vector<uint32_t> pack_coo_u16(const COOMatrix& coo) {
    std::vector<uint32_t> packed(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) {
        packed[i] = ((uint32_t)coo.row_idx[i] << 16)
                  |  (uint32_t)coo.col_idx[i];
    }
    return packed;
}

inline std::vector<uint32_t> pack_coo_u16_with_col_ids(
    const COOMatrix& coo, const std::vector<int>& col_ids)
{
    std::vector<uint32_t> packed(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) {
        packed[i] = ((uint32_t)coo.row_idx[i] << 16)
                  |  (uint32_t)col_ids[i];
    }
    return packed;
}

inline std::vector<uint16_t> pack_cols_u16(const int* col_idx, int n) {
    std::vector<uint16_t> packed(n);
    for (int i = 0; i < n; i++)
        packed[i] = (uint16_t)col_idx[i];
    return packed;
}

inline int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

inline int log2_exact(int n) {
    int bits = 0;
    while ((1 << bits) < n) bits++;
    return bits;
}

// ---------------------------------------------------------------------------
// Bluestein chirp helpers and compact CSC structure
// ---------------------------------------------------------------------------
inline std::vector<cuFloatComplex> make_bluestein_chirp(int n) {
    std::vector<cuFloatComplex> chirp(n);
    const double pi = 3.1415926535897932384626433832795;
    for (int k = 0; k < n; k++) {
        const double kk = (double)k * (double)k;
        const double angle = -pi * kk / (double)n;
        chirp[k] = make_cuFloatComplex((float)std::cos(angle), (float)std::sin(angle));
    }
    return chirp;
}

inline std::vector<cuFloatComplex> make_bluestein_b(int n, int fft_len) {
    std::vector<cuFloatComplex> b(fft_len, make_cuFloatComplex(0.f, 0.f));
    const double pi = 3.1415926535897932384626433832795;
    b[0] = make_cuFloatComplex(1.f, 0.f);
    for (int k = 1; k < n; k++) {
        const double kk = (double)k * (double)k;
        const double angle = pi * kk / (double)n;
        cuFloatComplex z = make_cuFloatComplex((float)std::cos(angle), (float)std::sin(angle));
        b[k] = z;
        b[fft_len - k] = z;
    }
    return b;
}

struct CompactCSC {
    int n_sparse_cols = 0;
    std::vector<int> sparse_cols;
    std::vector<int> col_ptr;
    std::vector<int> row_idx;
};

inline CompactCSC make_compact_csc(const COOMatrix& coo) {
    CompactCSC csc;
    csc.sparse_cols = make_sparse_cols(coo.col_idx.data(), coo.nnz);
    csc.n_sparse_cols = (int)csc.sparse_cols.size();
    std::vector<int> col_ids =
        make_sparse_col_ids(coo.col_idx.data(), coo.nnz, csc.sparse_cols, coo.cols);

    csc.col_ptr.assign(csc.n_sparse_cols + 1, 0);
    for (int i = 0; i < coo.nnz; i++)
        csc.col_ptr[col_ids[i] + 1]++;

    for (int i = 0; i < csc.n_sparse_cols; i++)
        csc.col_ptr[i + 1] += csc.col_ptr[i];

    csc.row_idx.resize(coo.nnz);
    std::vector<int> next = csc.col_ptr;
    for (int i = 0; i < coo.nnz; i++) {
        const int dst = next[col_ids[i]]++;
        csc.row_idx[dst] = coo.row_idx[i];
    }

    return csc;
}
