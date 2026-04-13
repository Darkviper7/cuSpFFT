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
// Kernel 1: tiled direct DFT  (sparse_fft)
//
// Each thread owns one output bin (u, v) and sums contributions from all nnz
// entries. COO arrays are staged through shared memory in tiles so that the
// nnz broadcast pays global-memory bandwidth only once per tile per block.
// ---------------------------------------------------------------------------

static constexpr int TILE = 1024;

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

__global__ void sparse_fft_tiled_kernel(
    const int* __restrict__ row_idx,
    const int* __restrict__ col_idx,
    cuFloatComplex* __restrict__ output,
    int rows, int cols, int nnz,
    float inv_rows, float inv_cols)
{
    __shared__ int s_row[TILE];
    __shared__ int s_col[TILE];

    const int u          = blockIdx.y * blockDim.y + threadIdx.y;
    const int v          = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < nnz; base += TILE) {
        const int tile_size = min(TILE, nnz - base);
        for (int j = tid; j < tile_size; j += block_size) {
            s_row[j] = row_idx[base + j];
            s_col[j] = col_idx[base + j];
        }
        __syncthreads();

        if (u < rows && v < cols) {
            for (int k = 0; k < tile_size; k++) {
                float sr, cr, sc, cc;
                sincos_twiddle(u, s_row[k], rows, inv_rows, &sr, &cr);
                sincos_twiddle(v, s_col[k], cols, inv_cols, &sc, &cc);
                re += cr * cc - sr * sc;
                im += cr * sc + sr * cc;
            }
        }
        __syncthreads();
    }

    if (u < rows && v < cols)
        output[u * cols + v] = make_cuFloatComplex(re, im);
}

__global__ void sparse_fft_tiled_packed_kernel(
    const uint32_t* __restrict__ packed_idx,
    cuFloatComplex* __restrict__ output,
    int rows, int cols, int nnz,
    float inv_rows, float inv_cols)
{
    __shared__ uint32_t s_idx[TILE];

    const int u          = blockIdx.y * blockDim.y + threadIdx.y;
    const int v          = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < nnz; base += TILE) {
        const int tile_size = min(TILE, nnz - base);
        for (int j = tid; j < tile_size; j += block_size)
            s_idx[j] = packed_idx[base + j];
        __syncthreads();

        if (u < rows && v < cols) {
            for (int k = 0; k < tile_size; k++) {
                const uint32_t p = s_idx[k];
                float sr, cr, sc, cc;
                sincos_twiddle(u, packed_row_u16(p), rows, inv_rows, &sr, &cr);
                sincos_twiddle(v, packed_col_u16(p), cols, inv_cols, &sc, &cc);
                re += cr * cc - sr * sc;
                im += cr * sc + sr * cc;
            }
        }
        __syncthreads();
    }

    if (u < rows && v < cols)
        output[(size_t)u * cols + v] = make_cuFloatComplex(re, im);
}

// ---------------------------------------------------------------------------
// Kernel 2: COO pass 1 — build_intermediate  (sparse_fft_2pass)
//
// NNZ_TILE nonzeros are loaded cooperatively into shared memory (1 per thread).
// Each of the BLOCK_U threads handles one output row u and accumulates
// contributions from the tile into G[sparse_col_id][u] via atomicAdd. G stores
// only columns that appear in the input, in column-major sparse-column order.
//
// vs. original (1 nonzero per block, 614k blocks):
//   Grid shrinks to ceil(nnz/NNZ_TILE) * ceil(rows/BLOCK_U) ≈ 4.8k blocks.
//   COO pairs are explicitly staged in shared memory and reused by all threads.
// ---------------------------------------------------------------------------

static constexpr int BLOCK_U  = 128;
static constexpr int NNZ_TILE = 128; // must equal BLOCK_U for 1:1 loading
static constexpr int CSC_BU   = 128;
static constexpr int CSC_BUILD_COLS = 32;
static constexpr int CSC_BUILD_U    = 8;

__global__ void build_intermediate(
    const int* __restrict__ row_idx,
    const int* __restrict__ col_idx,
    cuFloatComplex* __restrict__ G,
    int rows, int cols, int nnz,
    float inv_rows)
{
    (void)cols;

    __shared__ int s_row[NNZ_TILE];
    __shared__ int s_col[NNZ_TILE];

    const int u    = blockIdx.y * BLOCK_U + threadIdx.x;
    const int base = blockIdx.x * NNZ_TILE;
    const int tid  = threadIdx.x;

    const int k = base + tid;
    if (k < nnz) {
        s_row[tid] = row_idx[k];
        s_col[tid] = col_idx[k];
    }
    __syncthreads();

    if (u >= rows) return;

    const int tile_size = min(NNZ_TILE, nnz - base);

    for (int i = 0; i < tile_size; i++) {
        float s, co;
        sincos_twiddle(u, s_row[i], rows, inv_rows, &s, &co);
        const size_t g_idx = (size_t)s_col[i] * rows + u;
        atomicAdd(&G[g_idx].x, co);
        atomicAdd(&G[g_idx].y,  s);
    }
}

__global__ void build_intermediate_packed(
    const uint32_t* __restrict__ packed_idx,
    cuFloatComplex* __restrict__ G,
    int rows, int cols, int nnz,
    float inv_rows)
{
    (void)cols;

    __shared__ uint32_t s_idx[NNZ_TILE];

    const int u    = blockIdx.y * BLOCK_U + threadIdx.x;
    const int base = blockIdx.x * NNZ_TILE;
    const int tid  = threadIdx.x;

    const int k = base + tid;
    if (k < nnz)
        s_idx[tid] = packed_idx[k];
    __syncthreads();

    if (u >= rows) return;

    const int tile_size = min(NNZ_TILE, nnz - base);

    for (int i = 0; i < tile_size; i++) {
        const uint32_t p = s_idx[i];
        float s, co;
        sincos_twiddle(u, packed_row_u16(p), rows, inv_rows, &s, &co);
        const size_t g_idx = (size_t)packed_col_u16(p) * rows + u;
        atomicAdd(&G[g_idx].x, co);
        atomicAdd(&G[g_idx].y,  s);
    }
}

__global__ void csc_build_intermediate(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    cuFloatComplex* __restrict__ G,
    int rows, int n_sparse_cols, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u >= rows) return;

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];

    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    G[(size_t)col_id * rows + u] = make_cuFloatComplex(re, im);
}

__global__ void csc_build_intermediate_packed(
    const int* __restrict__ col_ptr,
    const uint16_t* __restrict__ row_idx,
    cuFloatComplex* __restrict__ G,
    int rows, int n_sparse_cols, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u >= rows) return;

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];

    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, (int)row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    G[(size_t)col_id * rows + u] = make_cuFloatComplex(re, im);
}

__global__ void csc_build_intermediate_chunk(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    cuFloatComplex* __restrict__ G_chunk,
    int rows, int n_sparse_cols, int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];

    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    G_chunk[(size_t)col_id * tile_rows + u_local] = make_cuFloatComplex(re, im);
}

__global__ void csc_build_intermediate_chunk_packed(
    const int* __restrict__ col_ptr,
    const uint16_t* __restrict__ row_idx,
    cuFloatComplex* __restrict__ G_chunk,
    int rows, int n_sparse_cols, int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];

    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, (int)row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    G_chunk[(size_t)col_id * tile_rows + u_local] = make_cuFloatComplex(re, im);
}

__global__ void csc_build_intermediate_chunk_half(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    __half2* __restrict__ G_chunk,
    int rows, int n_sparse_cols, int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];

    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    G_chunk[(size_t)col_id * tile_rows + u_local] = __floats2half2_rn(re, im);
}

__global__ void csc_build_intermediate_chunk_half_packed(
    const int* __restrict__ col_ptr,
    const uint16_t* __restrict__ row_idx,
    __half2* __restrict__ G_chunk,
    int rows, int n_sparse_cols, int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];

    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, (int)row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    G_chunk[(size_t)col_id * tile_rows + u_local] = __floats2half2_rn(re, im);
}

// ---------------------------------------------------------------------------
// Kernel 3: CSR pass 1 — csr_build_intermediate  (sparse_fft_csr_2pass)
//
// 2-D block (CSR_BU x CSR_BR): threadIdx.y = source row, threadIdx.x = output row.
// Each warp owns one source row and 32 consecutive output rows.  For each
// source-column entry, the warp broadcasts the sparse-column id and writes
// consecutive G[sparse_col_id][u] locations. Rows with nnz > SMEM_COLS fall
// back to global memory after the cached prefix.
// ---------------------------------------------------------------------------

static constexpr int CSR_BU    = 32;
static constexpr int CSR_BR    =  8;
static constexpr int SMEM_COLS = 32;
static constexpr int SMEM_COLS_PADDED = SMEM_COLS + 1;

__global__ void csr_build_intermediate(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    cuFloatComplex* __restrict__ G,
    int rows, int cols, float inv_rows)
{
    (void)cols;

    __shared__ int s_col  [CSR_BR][SMEM_COLS_PADDED];
    __shared__ int s_start[CSR_BR];
    __shared__ int s_nnz  [CSR_BR];

    const int u_local  = threadIdx.x;
    const int r_local  = threadIdx.y;
    const int r_global = blockIdx.x * CSR_BR + r_local;
    const int u        = blockIdx.y * CSR_BU + u_local;

    const bool valid_r = (r_global < rows);
    if (u_local == 0) {
        if (valid_r) {
            const int start = row_ptr[r_global];
            const int nnz_r = row_ptr[r_global + 1] - start;
            s_start[r_local] = start;
            s_nnz  [r_local] = nnz_r;
        } else {
            s_start[r_local] = 0;
            s_nnz  [r_local] = 0;
        }
    }
    __syncthreads();

    const int start = s_start[r_local];
    const int nnz_r = s_nnz  [r_local];
    const int cached = min(nnz_r, SMEM_COLS);
    for (int j = u_local; j < cached; j += CSR_BU)
        s_col[r_local][j] = col_idx[start + j];
    __syncthreads();

    if (!valid_r || u >= rows || nnz_r == 0) return;

    float s, co;
    sincos_twiddle(u, r_global, rows, inv_rows, &s, &co);

    for (int i = 0; i < cached; i++) {
        const size_t g_idx = (size_t)s_col[r_local][i] * rows + u;
        atomicAdd(&G[g_idx].x, co);
        atomicAdd(&G[g_idx].y,  s);
    }
    for (int i = cached; i < nnz_r; i++) {
        int c = col_idx[start + i];
        const size_t g_idx = (size_t)c * rows + u;
        atomicAdd(&G[g_idx].x, co);
        atomicAdd(&G[g_idx].y,  s);
    }
}

__global__ void csr_build_intermediate_packed(
    const int* __restrict__ row_ptr,
    const uint16_t* __restrict__ col_idx,
    cuFloatComplex* __restrict__ G,
    int rows, int cols, float inv_rows)
{
    (void)cols;

    __shared__ uint16_t s_col[CSR_BR][SMEM_COLS_PADDED];
    __shared__ int s_start[CSR_BR];
    __shared__ int s_nnz  [CSR_BR];

    const int u_local  = threadIdx.x;
    const int r_local  = threadIdx.y;
    const int r_global = blockIdx.x * CSR_BR + r_local;
    const int u        = blockIdx.y * CSR_BU + u_local;

    const bool valid_r = (r_global < rows);
    if (u_local == 0) {
        if (valid_r) {
            const int start = row_ptr[r_global];
            const int nnz_r = row_ptr[r_global + 1] - start;
            s_start[r_local] = start;
            s_nnz  [r_local] = nnz_r;
        } else {
            s_start[r_local] = 0;
            s_nnz  [r_local] = 0;
        }
    }
    __syncthreads();

    const int start = s_start[r_local];
    const int nnz_r = s_nnz  [r_local];
    const int cached = min(nnz_r, SMEM_COLS);
    for (int j = u_local; j < cached; j += CSR_BU)
        s_col[r_local][j] = col_idx[start + j];
    __syncthreads();

    if (!valid_r || u >= rows || nnz_r == 0) return;

    float s, co;
    sincos_twiddle(u, r_global, rows, inv_rows, &s, &co);

    for (int i = 0; i < cached; i++) {
        const size_t g_idx = (size_t)s_col[r_local][i] * rows + u;
        atomicAdd(&G[g_idx].x, co);
        atomicAdd(&G[g_idx].y,  s);
    }
    for (int i = cached; i < nnz_r; i++) {
        int c = (int)col_idx[start + i];
        const size_t g_idx = (size_t)c * rows + u;
        atomicAdd(&G[g_idx].x, co);
        atomicAdd(&G[g_idx].y,  s);
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: custom pass 2 — col_dft_kernel  (both COO and CSR 2-pass)
//
// Replaces cuFFT entirely.  Computes the column-DFT of G using only the
// columns where G is actually nonzero (sparse_cols).
//
//   F[u, v] = sum_i G[i][u] * exp(-2πi * sparse_cols[i] * v / cols)
//
// G is compacted to n_sparse_cols * rows and read column-major here to match
// the coalesced pass-1 atomic layout; F remains row-major for callers.
//
// Shared memory: sparse_cols and column twiddles are tiled.  The twiddle
// exp(-2*pi*i*c*v/cols) is independent of u, so one block computes it once per
// sparse-column/output-v pair and reuses it across P2_BU output rows.
//
// Grid: (ceil(cols/P2_BV), ceil(rows/P2_BU))
// Block: (P2_BV, P2_BU) — each thread owns one output bin F[u][v].
// ---------------------------------------------------------------------------

static constexpr int P2_BV     = 32;  // output-freq threads per block (x)
static constexpr int P2_BU     = 32;  // output-row  threads per block (y)
static constexpr int COL_TILE  = 128;
static constexpr int CSC_TILED_U   = 16;
static constexpr int CSC_TILED_V   = 16;
// Halving C from 64→32 cuts smem from 32 KB to 16 KB, allowing 6 blocks/SM
// (V100: 96 KB / 16 KB) instead of 3 — triples active warps.  The total
// sincos + G-load work per kernel call is unchanged; we just sync twice as
// often, which is hidden by the extra occupancy.
static constexpr int CSC_TILED_C   = 32;
// Register-blocking factors for col_dft_chunk_tiled_kernel[_packed].
// Each thread owns a CSC_TILED_RBU × CSC_TILED_RBV tile of outputs,
// loading each G/W value once and reusing it across the tile.
// v is indexed as rv*CSC_TILED_V + tx (stride-coalesced) so both
// output writes and sW reads are coalesced for every (ru, rv) pair.
static constexpr int CSC_TILED_RBU = 2;
// RBV=4: each block covers CSC_TILED_V*RBV=64 output columns, halving the
// v-block count vs RBV=2.  Because every v-block reads the full G_chunk, this
// directly halves G_chunk DRAM traffic (the kernel's primary bottleneck).
// smem: sG[32][32]=8KB + sW[32][64]=16KB = 24KB → 4 blocks/SM (50% occupancy)
// vs RBV=2 at 16KB → 6 blocks/SM (75%).  The 2x bandwidth win dominates.
static constexpr int CSC_TILED_RBV = 4;

__global__ void col_dft_kernel(
    const cuFloatComplex* __restrict__ G,
    const int*            __restrict__ sparse_cols,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int cols, int output_cols, float inv_cols)
{
    __shared__ int s_col[COL_TILE];
    __shared__ cuFloatComplex s_twiddle[COL_TILE][P2_BV];

    const int v   = blockIdx.x * P2_BV + threadIdx.x;
    const int u   = blockIdx.y * P2_BU + threadIdx.y;
    const int tid = threadIdx.y * P2_BV + threadIdx.x;
    const int block_threads = P2_BV * P2_BU;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < n_sparse_cols; base += COL_TILE) {
        const int tile_size = min(COL_TILE, n_sparse_cols - base);

        // One entry per thread — fully coalesced load of the sparse col list
        if (tid < tile_size)
            s_col[tid] = sparse_cols[base + tid];
        __syncthreads();

        for (int t = tid; t < tile_size * P2_BV; t += block_threads) {
            const int i = t / P2_BV;
            const int vx = t - i * P2_BV;
            const int tile_v = blockIdx.x * P2_BV + vx;
            float s, co;
            if (tile_v < output_cols) {
                sincos_twiddle(tile_v, s_col[i], cols, inv_cols, &s, &co);
                s_twiddle[i][vx] = make_cuFloatComplex(co, s);
            } else {
                s_twiddle[i][vx] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        if (u < rows && v < output_cols) {
            for (int i = 0; i < tile_size; i++) {
                cuFloatComplex w = s_twiddle[i][threadIdx.x];
                cuFloatComplex g = G[(size_t)(base + i) * rows + u];
                re += g.x * w.x - g.y * w.y;
                im += g.x * w.y + g.y * w.x;
            }
        }
        __syncthreads();
    }

    if (u < rows && v < output_cols)
        F[(size_t)u * output_cols + v] = make_cuFloatComplex(re, im);
}

__global__ void col_dft_kernel_packed(
    const cuFloatComplex* __restrict__ G,
    const uint16_t*       __restrict__ sparse_cols,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int cols, int output_cols, float inv_cols)
{
    __shared__ uint16_t s_col[COL_TILE];
    __shared__ cuFloatComplex s_twiddle[COL_TILE][P2_BV];

    const int v   = blockIdx.x * P2_BV + threadIdx.x;
    const int u   = blockIdx.y * P2_BU + threadIdx.y;
    const int tid = threadIdx.y * P2_BV + threadIdx.x;
    const int block_threads = P2_BV * P2_BU;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < n_sparse_cols; base += COL_TILE) {
        const int tile_size = min(COL_TILE, n_sparse_cols - base);

        if (tid < tile_size)
            s_col[tid] = sparse_cols[base + tid];
        __syncthreads();

        for (int t = tid; t < tile_size * P2_BV; t += block_threads) {
            const int i = t / P2_BV;
            const int vx = t - i * P2_BV;
            const int tile_v = blockIdx.x * P2_BV + vx;
            float s, co;
            if (tile_v < output_cols) {
                sincos_twiddle(tile_v, (int)s_col[i], cols, inv_cols, &s, &co);
                s_twiddle[i][vx] = make_cuFloatComplex(co, s);
            } else {
                s_twiddle[i][vx] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        if (u < rows && v < output_cols) {
            for (int i = 0; i < tile_size; i++) {
                cuFloatComplex w = s_twiddle[i][threadIdx.x];
                cuFloatComplex g = G[(size_t)(base + i) * rows + u];
                re += g.x * w.x - g.y * w.y;
                im += g.x * w.y + g.y * w.x;
            }
        }
        __syncthreads();
    }

    if (u < rows && v < output_cols)
        F[(size_t)u * output_cols + v] = make_cuFloatComplex(re, im);
}

__global__ void col_dft_chunk_kernel(
    const cuFloatComplex* __restrict__ G_chunk,
    const int*            __restrict__ sparse_cols,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int cols, int output_cols, int u_base, int tile_rows, float inv_cols)
{
    __shared__ int s_col[COL_TILE];
    __shared__ cuFloatComplex s_twiddle[COL_TILE][P2_BV];

    const int v       = blockIdx.x * P2_BV + threadIdx.x;
    const int u_local = blockIdx.y * P2_BU + threadIdx.y;
    const int u       = u_base + u_local;
    const int tid     = threadIdx.y * P2_BV + threadIdx.x;
    const int block_threads = P2_BV * P2_BU;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < n_sparse_cols; base += COL_TILE) {
        const int tile_size = min(COL_TILE, n_sparse_cols - base);

        if (tid < tile_size)
            s_col[tid] = sparse_cols[base + tid];
        __syncthreads();

        for (int t = tid; t < tile_size * P2_BV; t += block_threads) {
            const int i = t / P2_BV;
            const int vx = t - i * P2_BV;
            const int tile_v = blockIdx.x * P2_BV + vx;
            float s, co;
            if (tile_v < output_cols) {
                sincos_twiddle(tile_v, s_col[i], cols, inv_cols, &s, &co);
                s_twiddle[i][vx] = make_cuFloatComplex(co, s);
            } else {
                s_twiddle[i][vx] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        if (u_local < tile_rows && u < rows && v < output_cols) {
            for (int i = 0; i < tile_size; i++) {
                cuFloatComplex w = s_twiddle[i][threadIdx.x];
                cuFloatComplex g = G_chunk[(size_t)(base + i) * tile_rows + u_local];
                re += g.x * w.x - g.y * w.y;
                im += g.x * w.y + g.y * w.x;
            }
        }
        __syncthreads();
    }

    if (u_local < tile_rows && u < rows && v < output_cols)
        F[(size_t)u * output_cols + v] = make_cuFloatComplex(re, im);
}

__global__ void col_dft_chunk_kernel_packed(
    const cuFloatComplex* __restrict__ G_chunk,
    const uint16_t*       __restrict__ sparse_cols,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int cols, int output_cols, int u_base, int tile_rows, float inv_cols)
{
    __shared__ uint16_t s_col[COL_TILE];
    __shared__ cuFloatComplex s_twiddle[COL_TILE][P2_BV];

    const int v       = blockIdx.x * P2_BV + threadIdx.x;
    const int u_local = blockIdx.y * P2_BU + threadIdx.y;
    const int u       = u_base + u_local;
    const int tid     = threadIdx.y * P2_BV + threadIdx.x;
    const int block_threads = P2_BV * P2_BU;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < n_sparse_cols; base += COL_TILE) {
        const int tile_size = min(COL_TILE, n_sparse_cols - base);

        if (tid < tile_size)
            s_col[tid] = sparse_cols[base + tid];
        __syncthreads();

        for (int t = tid; t < tile_size * P2_BV; t += block_threads) {
            const int i = t / P2_BV;
            const int vx = t - i * P2_BV;
            const int tile_v = blockIdx.x * P2_BV + vx;
            float s, co;
            if (tile_v < output_cols) {
                sincos_twiddle(tile_v, (int)s_col[i], cols, inv_cols, &s, &co);
                s_twiddle[i][vx] = make_cuFloatComplex(co, s);
            } else {
                s_twiddle[i][vx] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        if (u_local < tile_rows && u < rows && v < output_cols) {
            for (int i = 0; i < tile_size; i++) {
                cuFloatComplex w = s_twiddle[i][threadIdx.x];
                cuFloatComplex g = G_chunk[(size_t)(base + i) * tile_rows + u_local];
                re += g.x * w.x - g.y * w.y;
                im += g.x * w.y + g.y * w.x;
            }
        }
        __syncthreads();
    }

    if (u_local < tile_rows && u < rows && v < output_cols)
        F[(size_t)u * output_cols + v] = make_cuFloatComplex(re, im);
}

__global__ void col_dft_chunk_cached_w_kernel(
    const cuFloatComplex* __restrict__ G_chunk,
    const cuFloatComplex* __restrict__ W,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int output_cols, int u_base, int tile_rows)
{
    __shared__ cuFloatComplex s_twiddle[COL_TILE][P2_BV];

    const int v       = blockIdx.x * P2_BV + threadIdx.x;
    const int u_local = blockIdx.y * P2_BU + threadIdx.y;
    const int u       = u_base + u_local;
    const int tid     = threadIdx.y * P2_BV + threadIdx.x;
    const int block_threads = P2_BV * P2_BU;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < n_sparse_cols; base += COL_TILE) {
        const int tile_size = min(COL_TILE, n_sparse_cols - base);

        for (int t = tid; t < tile_size * P2_BV; t += block_threads) {
            const int i = t / P2_BV;
            const int vx = t - i * P2_BV;
            const int tile_v = blockIdx.x * P2_BV + vx;
            s_twiddle[i][vx] = (tile_v < output_cols)
                             ? W[(size_t)(base + i) * output_cols + tile_v]
                             : make_cuFloatComplex(0.f, 0.f);
        }
        __syncthreads();

        if (u_local < tile_rows && u < rows && v < output_cols) {
            for (int i = 0; i < tile_size; i++) {
                cuFloatComplex w = s_twiddle[i][threadIdx.x];
                cuFloatComplex g = G_chunk[(size_t)(base + i) * tile_rows + u_local];
                re += g.x * w.x - g.y * w.y;
                im += g.x * w.y + g.y * w.x;
            }
        }
        __syncthreads();
    }

    if (u_local < tile_rows && u < rows && v < output_cols)
        F[(size_t)u * output_cols + v] = make_cuFloatComplex(re, im);
}

__global__ void col_dft_chunk_tiled_kernel(
    const cuFloatComplex* __restrict__ G_chunk,
    const int*            __restrict__ sparse_cols,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int cols, int output_cols, int u_base, int tile_rows,
    int output_row_offset, float inv_cols)
{
    // Shared tiles enlarged for register blocking.
    // sG[c][u_local]:  col-tile × (CSC_TILED_U * RBU) u-rows
    // sW[c][vv]:       col-tile × (CSC_TILED_V * RBV) v-bins,
    //                  indexed as rv * CSC_TILED_V + tx (stride-coalesced)
    __shared__ cuFloatComplex sG[CSC_TILED_C][CSC_TILED_U * CSC_TILED_RBU];
    __shared__ cuFloatComplex sW[CSC_TILED_C][CSC_TILED_V * CSC_TILED_RBV];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * CSC_TILED_V + tx;
    const int block_threads = CSC_TILED_U * CSC_TILED_V;

    // RBU × RBV per-thread accumulators (zero-initialised by aggregate init)
    float re[CSC_TILED_RBU][CSC_TILED_RBV] = {};
    float im[CSC_TILED_RBU][CSC_TILED_RBV] = {};

    for (int base = 0; base < n_sparse_cols; base += CSC_TILED_C) {
        const int c_count = min(CSC_TILED_C, n_sparse_cols - base);

        // --- Load G tile (block-cooperative) ---
        for (int t = tid; t < c_count * (CSC_TILED_U * CSC_TILED_RBU); t += block_threads) {
            const int ci = t / (CSC_TILED_U * CSC_TILED_RBU);
            const int uu = t - ci * (CSC_TILED_U * CSC_TILED_RBU);
            const int global_u_local = blockIdx.y * (CSC_TILED_U * CSC_TILED_RBU) + uu;
            sG[ci][uu] = (global_u_local < tile_rows)
                       ? G_chunk[(size_t)(base + ci) * tile_rows + global_u_local]
                       : make_cuFloatComplex(0.f, 0.f);
        }

        // --- Load W tile (block-cooperative, stride-coalesced v layout) ---
        // vv linearly indexes the block's v-range; during compute thread tx
        // accesses sW[ci][rv * CSC_TILED_V + tx], i.e. a different 16-wide
        // coalesced stripe for each rv.
        for (int t = tid; t < c_count * (CSC_TILED_V * CSC_TILED_RBV); t += block_threads) {
            const int ci = t / (CSC_TILED_V * CSC_TILED_RBV);
            const int vv = t - ci * (CSC_TILED_V * CSC_TILED_RBV);
            const int global_v = blockIdx.x * (CSC_TILED_V * CSC_TILED_RBV) + vv;
            if (global_v < output_cols) {
                float s, co;
                sincos_twiddle(global_v, sparse_cols[base + ci], cols, inv_cols, &s, &co);
                sW[ci][vv] = make_cuFloatComplex(co, s);
            } else {
                sW[ci][vv] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        // --- Register-blocked accumulation ---
        // Each iteration: load RBU G values (warp-broadcast) + RBV W values
        // (coalesced per rv stripe), then compute the RBU×RBV outer product.
        // fmaf() forces FMA emission; -g[ru].y folds into a PTX negation flag.
        #pragma unroll 8
        for (int ci = 0; ci < c_count; ci++) {
            cuFloatComplex g[CSC_TILED_RBU], w[CSC_TILED_RBV];
            #pragma unroll
            for (int ru = 0; ru < CSC_TILED_RBU; ru++)
                g[ru] = sG[ci][ty * CSC_TILED_RBU + ru];
            #pragma unroll
            for (int rv = 0; rv < CSC_TILED_RBV; rv++)
                w[rv] = sW[ci][rv * CSC_TILED_V + tx];
            #pragma unroll
            for (int ru = 0; ru < CSC_TILED_RBU; ru++) {
                #pragma unroll
                for (int rv = 0; rv < CSC_TILED_RBV; rv++) {
                    re[ru][rv] = fmaf( g[ru].x,  w[rv].x, re[ru][rv]);
                    re[ru][rv] = fmaf(-g[ru].y,  w[rv].y, re[ru][rv]);
                    im[ru][rv] = fmaf( g[ru].x,  w[rv].y, im[ru][rv]);
                    im[ru][rv] = fmaf( g[ru].y,  w[rv].x, im[ru][rv]);
                }
            }
        }
        __syncthreads();
    }

    // --- Scatter RBU × RBV outputs (one row of threads writes coalesced v) ---
    #pragma unroll
    for (int ru = 0; ru < CSC_TILED_RBU; ru++) {
        const int u_local = blockIdx.y * (CSC_TILED_U * CSC_TILED_RBU) + ty * CSC_TILED_RBU + ru;
        const int u = u_base + u_local;
        if (u_local >= tile_rows || u >= rows) continue;
        #pragma unroll
        for (int rv = 0; rv < CSC_TILED_RBV; rv++) {
            const int v = blockIdx.x * (CSC_TILED_V * CSC_TILED_RBV) + rv * CSC_TILED_V + tx;
            if (v < output_cols)
                F[(size_t)(u - output_row_offset) * output_cols + v] =
                    make_cuFloatComplex(re[ru][rv], im[ru][rv]);
        }
    }
}

__global__ void col_dft_chunk_tiled_kernel_packed(
    const cuFloatComplex* __restrict__ G_chunk,
    const uint16_t*       __restrict__ sparse_cols,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int cols, int output_cols, int u_base, int tile_rows,
    int output_row_offset, float inv_cols)
{
    __shared__ cuFloatComplex sG[CSC_TILED_C][CSC_TILED_U * CSC_TILED_RBU];
    __shared__ cuFloatComplex sW[CSC_TILED_C][CSC_TILED_V * CSC_TILED_RBV];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * CSC_TILED_V + tx;
    const int block_threads = CSC_TILED_U * CSC_TILED_V;

    float re[CSC_TILED_RBU][CSC_TILED_RBV] = {};
    float im[CSC_TILED_RBU][CSC_TILED_RBV] = {};

    for (int base = 0; base < n_sparse_cols; base += CSC_TILED_C) {
        const int c_count = min(CSC_TILED_C, n_sparse_cols - base);

        for (int t = tid; t < c_count * (CSC_TILED_U * CSC_TILED_RBU); t += block_threads) {
            const int ci = t / (CSC_TILED_U * CSC_TILED_RBU);
            const int uu = t - ci * (CSC_TILED_U * CSC_TILED_RBU);
            const int global_u_local = blockIdx.y * (CSC_TILED_U * CSC_TILED_RBU) + uu;
            sG[ci][uu] = (global_u_local < tile_rows)
                       ? G_chunk[(size_t)(base + ci) * tile_rows + global_u_local]
                       : make_cuFloatComplex(0.f, 0.f);
        }

        for (int t = tid; t < c_count * (CSC_TILED_V * CSC_TILED_RBV); t += block_threads) {
            const int ci = t / (CSC_TILED_V * CSC_TILED_RBV);
            const int vv = t - ci * (CSC_TILED_V * CSC_TILED_RBV);
            const int global_v = blockIdx.x * (CSC_TILED_V * CSC_TILED_RBV) + vv;
            if (global_v < output_cols) {
                float s, co;
                sincos_twiddle(global_v, (int)sparse_cols[base + ci], cols, inv_cols, &s, &co);
                sW[ci][vv] = make_cuFloatComplex(co, s);
            } else {
                sW[ci][vv] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        #pragma unroll 8
        for (int ci = 0; ci < c_count; ci++) {
            cuFloatComplex g[CSC_TILED_RBU], w[CSC_TILED_RBV];
            #pragma unroll
            for (int ru = 0; ru < CSC_TILED_RBU; ru++)
                g[ru] = sG[ci][ty * CSC_TILED_RBU + ru];
            #pragma unroll
            for (int rv = 0; rv < CSC_TILED_RBV; rv++)
                w[rv] = sW[ci][rv * CSC_TILED_V + tx];
            #pragma unroll
            for (int ru = 0; ru < CSC_TILED_RBU; ru++) {
                #pragma unroll
                for (int rv = 0; rv < CSC_TILED_RBV; rv++) {
                    re[ru][rv] = fmaf( g[ru].x,  w[rv].x, re[ru][rv]);
                    re[ru][rv] = fmaf(-g[ru].y,  w[rv].y, re[ru][rv]);
                    im[ru][rv] = fmaf( g[ru].x,  w[rv].y, im[ru][rv]);
                    im[ru][rv] = fmaf( g[ru].y,  w[rv].x, im[ru][rv]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ru = 0; ru < CSC_TILED_RBU; ru++) {
        const int u_local = blockIdx.y * (CSC_TILED_U * CSC_TILED_RBU) + ty * CSC_TILED_RBU + ru;
        const int u = u_base + u_local;
        if (u_local >= tile_rows || u >= rows) continue;
        #pragma unroll
        for (int rv = 0; rv < CSC_TILED_RBV; rv++) {
            const int v = blockIdx.x * (CSC_TILED_V * CSC_TILED_RBV) + rv * CSC_TILED_V + tx;
            if (v < output_cols)
                F[(size_t)(u - output_row_offset) * output_cols + v] =
                    make_cuFloatComplex(re[ru][rv], im[ru][rv]);
        }
    }
}

__global__ void col_dft_chunk_tiled_half_kernel(
    const __half2* __restrict__ G_chunk,
    const int* __restrict__ sparse_cols,
    int n_sparse_cols,
    cuFloatComplex* __restrict__ F,
    int rows, int cols, int output_cols, int u_base, int tile_rows, float inv_cols)
{
    __shared__ cuFloatComplex sG[CSC_TILED_C][CSC_TILED_U * CSC_TILED_RBU];
    __shared__ cuFloatComplex sW[CSC_TILED_C][CSC_TILED_V * CSC_TILED_RBV];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * CSC_TILED_V + tx;
    const int block_threads = CSC_TILED_U * CSC_TILED_V;

    float re[CSC_TILED_RBU][CSC_TILED_RBV] = {};
    float im[CSC_TILED_RBU][CSC_TILED_RBV] = {};

    for (int base = 0; base < n_sparse_cols; base += CSC_TILED_C) {
        const int c_count = min(CSC_TILED_C, n_sparse_cols - base);

        for (int t = tid; t < c_count * (CSC_TILED_U * CSC_TILED_RBU); t += block_threads) {
            const int ci = t / (CSC_TILED_U * CSC_TILED_RBU);
            const int uu = t - ci * (CSC_TILED_U * CSC_TILED_RBU);
            const int global_u_local = blockIdx.y * (CSC_TILED_U * CSC_TILED_RBU) + uu;
            if (global_u_local < tile_rows) {
                float2 g = __half22float2(G_chunk[(size_t)(base + ci) * tile_rows + global_u_local]);
                sG[ci][uu] = make_cuFloatComplex(g.x, g.y);
            } else {
                sG[ci][uu] = make_cuFloatComplex(0.f, 0.f);
            }
        }

        for (int t = tid; t < c_count * (CSC_TILED_V * CSC_TILED_RBV); t += block_threads) {
            const int ci = t / (CSC_TILED_V * CSC_TILED_RBV);
            const int vv = t - ci * (CSC_TILED_V * CSC_TILED_RBV);
            const int global_v = blockIdx.x * (CSC_TILED_V * CSC_TILED_RBV) + vv;
            if (global_v < output_cols) {
                float s, co;
                sincos_twiddle(global_v, sparse_cols[base + ci], cols, inv_cols, &s, &co);
                sW[ci][vv] = make_cuFloatComplex(co, s);
            } else {
                sW[ci][vv] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        #pragma unroll 8
        for (int ci = 0; ci < c_count; ci++) {
            cuFloatComplex g[CSC_TILED_RBU], w[CSC_TILED_RBV];
            #pragma unroll
            for (int ru = 0; ru < CSC_TILED_RBU; ru++)
                g[ru] = sG[ci][ty * CSC_TILED_RBU + ru];
            #pragma unroll
            for (int rv = 0; rv < CSC_TILED_RBV; rv++)
                w[rv] = sW[ci][rv * CSC_TILED_V + tx];
            #pragma unroll
            for (int ru = 0; ru < CSC_TILED_RBU; ru++) {
                #pragma unroll
                for (int rv = 0; rv < CSC_TILED_RBV; rv++) {
                    re[ru][rv] = fmaf( g[ru].x,  w[rv].x, re[ru][rv]);
                    re[ru][rv] = fmaf(-g[ru].y,  w[rv].y, re[ru][rv]);
                    im[ru][rv] = fmaf( g[ru].x,  w[rv].y, im[ru][rv]);
                    im[ru][rv] = fmaf( g[ru].y,  w[rv].x, im[ru][rv]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ru = 0; ru < CSC_TILED_RBU; ru++) {
        const int u_local = blockIdx.y * (CSC_TILED_U * CSC_TILED_RBU) + ty * CSC_TILED_RBU + ru;
        const int u = u_base + u_local;
        if (u_local >= tile_rows || u >= rows) continue;
        #pragma unroll
        for (int rv = 0; rv < CSC_TILED_RBV; rv++) {
            const int v = blockIdx.x * (CSC_TILED_V * CSC_TILED_RBV) + rv * CSC_TILED_V + tx;
            if (v < output_cols)
                F[(size_t)u * output_cols + v] = make_cuFloatComplex(re[ru][rv], im[ru][rv]);
        }
    }
}

__global__ void col_dft_chunk_tiled_half_kernel_packed(
    const __half2* __restrict__ G_chunk,
    const uint16_t* __restrict__ sparse_cols,
    int n_sparse_cols,
    cuFloatComplex* __restrict__ F,
    int rows, int cols, int output_cols, int u_base, int tile_rows, float inv_cols)
{
    __shared__ cuFloatComplex sG[CSC_TILED_C][CSC_TILED_U * CSC_TILED_RBU];
    __shared__ cuFloatComplex sW[CSC_TILED_C][CSC_TILED_V * CSC_TILED_RBV];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * CSC_TILED_V + tx;
    const int block_threads = CSC_TILED_U * CSC_TILED_V;

    float re[CSC_TILED_RBU][CSC_TILED_RBV] = {};
    float im[CSC_TILED_RBU][CSC_TILED_RBV] = {};

    for (int base = 0; base < n_sparse_cols; base += CSC_TILED_C) {
        const int c_count = min(CSC_TILED_C, n_sparse_cols - base);

        for (int t = tid; t < c_count * (CSC_TILED_U * CSC_TILED_RBU); t += block_threads) {
            const int ci = t / (CSC_TILED_U * CSC_TILED_RBU);
            const int uu = t - ci * (CSC_TILED_U * CSC_TILED_RBU);
            const int global_u_local = blockIdx.y * (CSC_TILED_U * CSC_TILED_RBU) + uu;
            if (global_u_local < tile_rows) {
                float2 g = __half22float2(G_chunk[(size_t)(base + ci) * tile_rows + global_u_local]);
                sG[ci][uu] = make_cuFloatComplex(g.x, g.y);
            } else {
                sG[ci][uu] = make_cuFloatComplex(0.f, 0.f);
            }
        }

        for (int t = tid; t < c_count * (CSC_TILED_V * CSC_TILED_RBV); t += block_threads) {
            const int ci = t / (CSC_TILED_V * CSC_TILED_RBV);
            const int vv = t - ci * (CSC_TILED_V * CSC_TILED_RBV);
            const int global_v = blockIdx.x * (CSC_TILED_V * CSC_TILED_RBV) + vv;
            if (global_v < output_cols) {
                float s, co;
                sincos_twiddle(global_v, (int)sparse_cols[base + ci], cols, inv_cols, &s, &co);
                sW[ci][vv] = make_cuFloatComplex(co, s);
            } else {
                sW[ci][vv] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        #pragma unroll 8
        for (int ci = 0; ci < c_count; ci++) {
            cuFloatComplex g[CSC_TILED_RBU], w[CSC_TILED_RBV];
            #pragma unroll
            for (int ru = 0; ru < CSC_TILED_RBU; ru++)
                g[ru] = sG[ci][ty * CSC_TILED_RBU + ru];
            #pragma unroll
            for (int rv = 0; rv < CSC_TILED_RBV; rv++)
                w[rv] = sW[ci][rv * CSC_TILED_V + tx];
            #pragma unroll
            for (int ru = 0; ru < CSC_TILED_RBU; ru++) {
                #pragma unroll
                for (int rv = 0; rv < CSC_TILED_RBV; rv++) {
                    re[ru][rv] = fmaf( g[ru].x,  w[rv].x, re[ru][rv]);
                    re[ru][rv] = fmaf(-g[ru].y,  w[rv].y, re[ru][rv]);
                    im[ru][rv] = fmaf( g[ru].x,  w[rv].y, im[ru][rv]);
                    im[ru][rv] = fmaf( g[ru].y,  w[rv].x, im[ru][rv]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ru = 0; ru < CSC_TILED_RBU; ru++) {
        const int u_local = blockIdx.y * (CSC_TILED_U * CSC_TILED_RBU) + ty * CSC_TILED_RBU + ru;
        const int u = u_base + u_local;
        if (u_local >= tile_rows || u >= rows) continue;
        #pragma unroll
        for (int rv = 0; rv < CSC_TILED_RBV; rv++) {
            const int v = blockIdx.x * (CSC_TILED_V * CSC_TILED_RBV) + rv * CSC_TILED_V + tx;
            if (v < output_cols)
                F[(size_t)u * output_cols + v] = make_cuFloatComplex(re[ru][rv], im[ru][rv]);
        }
    }
}

__global__ void col_dft_chunk_broadcast_kernel(
    const cuFloatComplex* __restrict__ G_chunk,
    const int*            __restrict__ sparse_cols,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int cols, int output_cols, int u_base, int tile_rows, float inv_cols)
{
    __shared__ int s_col[COL_TILE];
    __shared__ cuFloatComplex s_twiddle[COL_TILE][P2_BV];

    const int lane    = threadIdx.x;
    const int v       = blockIdx.x * P2_BV + lane;
    const int u_local = blockIdx.y * P2_BU + threadIdx.y;
    const int u       = u_base + u_local;
    const int tid     = threadIdx.y * P2_BV + threadIdx.x;
    const int block_threads = P2_BV * P2_BU;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < n_sparse_cols; base += COL_TILE) {
        const int tile_size = min(COL_TILE, n_sparse_cols - base);

        if (tid < tile_size)
            s_col[tid] = sparse_cols[base + tid];
        __syncthreads();

        for (int t = tid; t < tile_size * P2_BV; t += block_threads) {
            const int i = t / P2_BV;
            const int vx = t - i * P2_BV;
            const int tile_v = blockIdx.x * P2_BV + vx;
            float s, co;
            if (tile_v < output_cols) {
                sincos_twiddle(tile_v, s_col[i], cols, inv_cols, &s, &co);
                s_twiddle[i][vx] = make_cuFloatComplex(co, s);
            } else {
                s_twiddle[i][vx] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        const bool valid_row = (u_local < tile_rows && u < rows);
        const bool valid_out = (v < output_cols);
        if (valid_row) {
            for (int i = 0; i < tile_size; i++) {
                float gx = 0.f, gy = 0.f;
                if (lane == 0) {
                    cuFloatComplex g = G_chunk[(size_t)(base + i) * tile_rows + u_local];
                    gx = g.x;
                    gy = g.y;
                }
                gx = __shfl_sync(0xffffffffu, gx, 0);
                gy = __shfl_sync(0xffffffffu, gy, 0);

                cuFloatComplex w = s_twiddle[i][lane];
                if (valid_out) {
                    re += gx * w.x - gy * w.y;
                    im += gx * w.y + gy * w.x;
                }
            }
        }
        __syncthreads();
    }

    if (u_local < tile_rows && u < rows && v < output_cols)
        F[(size_t)u * output_cols + v] = make_cuFloatComplex(re, im);
}

__global__ void col_dft_chunk_broadcast_kernel_packed(
    const cuFloatComplex* __restrict__ G_chunk,
    const uint16_t*       __restrict__ sparse_cols,
    int                                n_sparse_cols,
    cuFloatComplex*       __restrict__ F,
    int rows, int cols, int output_cols, int u_base, int tile_rows, float inv_cols)
{
    __shared__ uint16_t s_col[COL_TILE];
    __shared__ cuFloatComplex s_twiddle[COL_TILE][P2_BV];

    const int lane    = threadIdx.x;
    const int v       = blockIdx.x * P2_BV + lane;
    const int u_local = blockIdx.y * P2_BU + threadIdx.y;
    const int u       = u_base + u_local;
    const int tid     = threadIdx.y * P2_BV + threadIdx.x;
    const int block_threads = P2_BV * P2_BU;

    float re = 0.f, im = 0.f;

    for (int base = 0; base < n_sparse_cols; base += COL_TILE) {
        const int tile_size = min(COL_TILE, n_sparse_cols - base);

        if (tid < tile_size)
            s_col[tid] = sparse_cols[base + tid];
        __syncthreads();

        for (int t = tid; t < tile_size * P2_BV; t += block_threads) {
            const int i = t / P2_BV;
            const int vx = t - i * P2_BV;
            const int tile_v = blockIdx.x * P2_BV + vx;
            float s, co;
            if (tile_v < output_cols) {
                sincos_twiddle(tile_v, (int)s_col[i], cols, inv_cols, &s, &co);
                s_twiddle[i][vx] = make_cuFloatComplex(co, s);
            } else {
                s_twiddle[i][vx] = make_cuFloatComplex(0.f, 0.f);
            }
        }
        __syncthreads();

        const bool valid_row = (u_local < tile_rows && u < rows);
        const bool valid_out = (v < output_cols);
        if (valid_row) {
            for (int i = 0; i < tile_size; i++) {
                float gx = 0.f, gy = 0.f;
                if (lane == 0) {
                    cuFloatComplex g = G_chunk[(size_t)(base + i) * tile_rows + u_local];
                    gx = g.x;
                    gy = g.y;
                }
                gx = __shfl_sync(0xffffffffu, gx, 0);
                gy = __shfl_sync(0xffffffffu, gy, 0);

                cuFloatComplex w = s_twiddle[i][lane];
                if (valid_out) {
                    re += gx * w.x - gy * w.y;
                    im += gx * w.y + gy * w.x;
                }
            }
        }
        __syncthreads();
    }

    if (u_local < tile_rows && u < rows && v < output_cols)
        F[(size_t)u * output_cols + v] = make_cuFloatComplex(re, im);
}

__global__ void build_twiddle_matrix_kernel(
    const int* __restrict__ sparse_cols,
    cuFloatComplex* __restrict__ W,
    int n_sparse_cols, int cols, int output_cols, float inv_cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_sparse_cols || v >= output_cols) return;

    float s, co;
    sincos_twiddle(v, sparse_cols[i], cols, inv_cols, &s, &co);
    W[(size_t)v * n_sparse_cols + i] = make_cuFloatComplex(co, s);
}

__global__ void build_twiddle_matrix_packed_kernel(
    const uint16_t* __restrict__ sparse_cols,
    cuFloatComplex* __restrict__ W,
    int n_sparse_cols, int cols, int output_cols, float inv_cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_sparse_cols || v >= output_cols) return;

    float s, co;
    sincos_twiddle(v, (int)sparse_cols[i], cols, inv_cols, &s, &co);
    W[(size_t)v * n_sparse_cols + i] = make_cuFloatComplex(co, s);
}

__global__ void build_twiddle_matrix_rowmajor_kernel(
    const int* __restrict__ sparse_cols,
    cuFloatComplex* __restrict__ W,
    int n_sparse_cols, int cols, int output_cols, float inv_cols)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_sparse_cols || v >= output_cols) return;

    float s, co;
    sincos_twiddle(v, sparse_cols[i], cols, inv_cols, &s, &co);
    W[(size_t)i * output_cols + v] = make_cuFloatComplex(co, s);
}

__global__ void build_twiddle_matrix_rowmajor_packed_kernel(
    const uint16_t* __restrict__ sparse_cols,
    cuFloatComplex* __restrict__ W,
    int n_sparse_cols, int cols, int output_cols, float inv_cols)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_sparse_cols || v >= output_cols) return;

    float s, co;
    sincos_twiddle(v, (int)sparse_cols[i], cols, inv_cols, &s, &co);
    W[(size_t)i * output_cols + v] = make_cuFloatComplex(co, s);
}

__global__ void colmajor_to_rowmajor_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int rows, int output_cols)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    if (u < rows && v < output_cols)
        out[(size_t)u * output_cols + v] = in[(size_t)v * rows + u];
}

__device__ __forceinline__ unsigned bit_reverse_u32(unsigned x, int bits) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    x = (x << 16) | (x >> 16);
    return x >> (32 - bits);
}

__global__ void bit_reverse_copy_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int n, int log_n, int batch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * batch;
    if (idx >= total) return;
    const int b = idx / n;
    const int i = idx - b * n;
    const int r = (int)bit_reverse_u32((unsigned)i, log_n);
    out[(size_t)b * n + r] = in[(size_t)b * n + i];
}

__global__ void fft_stage_kernel(
    cuFloatComplex* __restrict__ data,
    int n, int half, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = (n / 2) * batch;
    if (idx >= butterflies) return;

    const int b = idx / (n / 2);
    const int k = idx - b * (n / 2);
    const int j = k & (half - 1);
    const int group = k / half;
    const int i0 = group * (half * 2) + j;
    const int i1 = i0 + half;

    const float two_pi = 6.28318530717958647f;
    const float angle = (inverse ? two_pi : -two_pi) * (float)j / (float)(half * 2);
    float s, c;
    __sincosf(angle, &s, &c);

    cuFloatComplex a = data[(size_t)b * n + i0];
    cuFloatComplex z = data[(size_t)b * n + i1];
    const float tx = z.x * c - z.y * s;
    const float ty = z.x * s + z.y * c;
    data[(size_t)b * n + i0] = make_cuFloatComplex(a.x + tx, a.y + ty);
    data[(size_t)b * n + i1] = make_cuFloatComplex(a.x - tx, a.y - ty);
}

__global__ void scale_complex_kernel(cuFloatComplex* data, int total, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

// ---------------------------------------------------------------------------
// Improvement 2: Shared-memory Stockham FFT (one block per row)
//
// When 2 * fft_len * 8 bytes fits in shared memory (≤ SMEM_MAX_FFT_N = 4096),
// load the entire row into smem ping-pong buffers and do ALL butterfly stages
// there.  This replaces log2(n) global-memory round-trips with a single
// global load + single global store — a log2(n)× reduction in DRAM traffic
// (11× for fft_len=2048, 12× for fft_len=4096).
//
// Applicable to sstmodel (fft_len = 2048).  Falls back to the per-stage stream
// variant for larger fft_len (benzene/pct20stif).
// ---------------------------------------------------------------------------
static constexpr int SMEM_MAX_FFT_N = 4096; // 2*4096*8 = 64 KB ≤ 96 KB V100 smem

// In-place: reads from inout into smem ping-pong, runs all stages, writes back.
// d_tmp is not touched; can be nullptr for smem path.
__global__ void stockham_smem_row_kernel(
    cuFloatComplex* __restrict__ inout,
    int n, int batch, int inverse)
{
    extern __shared__ cuFloatComplex smem[];   // layout: [0,n) = buf A, [n,2n) = buf B
    const int row = blockIdx.x;
    if (row >= batch) return;

    cuFloatComplex* sa = smem;
    cuFloatComplex* sb = smem + n;

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sa[i] = inout[(size_t)row * n + i];
    __syncthreads();

    cuFloatComplex* src = sa, *dst = sb;
    const float sign = inverse ? 6.28318530717958647f : -6.28318530717958647f;

    for (int l = 1; l < n; l <<= 1) {
        const int m = n / (2 * l);
        for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
            const int j = t / m, k = t - j * m;
            const int in0 = j*(2*m)+k, in1 = in0+m;
            const int out0 = j*m+k,    out1 = out0+n/2;
            float s, c;
            __sincosf(sign * (float)k / (float)(2*m), &s, &c);
            cuFloatComplex a = src[in0], z = src[in1];
            const float dx = a.x-z.x, dy = a.y-z.y;
            dst[out0] = make_cuFloatComplex(a.x+z.x, a.y+z.y);
            dst[out1] = make_cuFloatComplex(fmaf(dx,c,-dy*s), fmaf(dx,s,dy*c));
        }
        __syncthreads();
        cuFloatComplex* tmp = src; src = dst; dst = tmp;
    }

    if (inverse) {
        const float sc = 1.f / (float)n;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            src[i] = make_cuFloatComplex(src[i].x * sc, src[i].y * sc);
        __syncthreads();
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        inout[(size_t)row * n + i] = src[i];
}

// Mul-inverse variant: pre-multiplies each element by mul[i % n] (the B_fft
// pointwise factor) before the IFFT.  Mathematically equivalent to fusing the
// multiply into the first butterfly stage, but implemented as a single smem pass.
__global__ void stockham_smem_row_mul_inverse_kernel(
    cuFloatComplex* __restrict__ inout,
    const cuFloatComplex* __restrict__ mul,
    int n, int batch)
{
    extern __shared__ cuFloatComplex smem[];
    const int row = blockIdx.x;
    if (row >= batch) return;

    cuFloatComplex* sa = smem, *sb = smem + n;

    // Load and pre-multiply by B_fft element-wise (all n positions).
    // This is equivalent to what stockham_stage_mul_kernel does at l=1, m=n/2:
    // in0=k, in1=k+n/2 covers all [0,n), so pre-multiplying all elements first
    // then running the regular butterfly gives identical results.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        cuFloatComplex x = inout[(size_t)row * n + i];
        cuFloatComplex m = mul[i];
        sa[i] = make_cuFloatComplex(fmaf(x.x,m.x,-x.y*m.y), fmaf(x.x,m.y,x.y*m.x));
    }
    __syncthreads();

    cuFloatComplex* src = sa, *dst = sb;
    const float sign = 6.28318530717958647f; // inverse

    for (int l = 1; l < n; l <<= 1) {
        const int m = n / (2 * l);
        for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
            const int j = t / m, k = t - j * m;
            const int in0 = j*(2*m)+k, in1 = in0+m;
            const int out0 = j*m+k,    out1 = out0+n/2;
            float s, c;
            __sincosf(sign * (float)k / (float)(2*m), &s, &c);
            cuFloatComplex a = src[in0], z = src[in1];
            const float dx = a.x-z.x, dy = a.y-z.y;
            dst[out0] = make_cuFloatComplex(a.x+z.x, a.y+z.y);
            dst[out1] = make_cuFloatComplex(fmaf(dx,c,-dy*s), fmaf(dx,s,dy*c));
        }
        __syncthreads();
        cuFloatComplex* tmp = src; src = dst; dst = tmp;
    }

    const float sc = 1.f / (float)n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        src[i] = make_cuFloatComplex(src[i].x * sc, src[i].y * sc);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        inout[(size_t)row * n + i] = src[i];
}

__global__ void stockham_stage_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int n, int l, int m, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = (n / 2) * batch;
    if (idx >= butterflies) return;

    const int b = idx / (n / 2);
    const int t = idx - b * (n / 2);
    const int j = t / m;
    const int k = t - j * m;

    const int in0 = j * (2 * m) + k;
    const int in1 = in0 + m;
    const int out0 = j * m + k;
    const int out1 = out0 + n / 2;

    const float two_pi = 6.28318530717958647f;
    const float angle = (inverse ? two_pi : -two_pi) * (float)k / (float)(2 * m);
    float s, c;
    __sincosf(angle, &s, &c);

    cuFloatComplex a = in[(size_t)b * n + in0];
    cuFloatComplex z = in[(size_t)b * n + in1];
    const float dx = a.x - z.x;
    const float dy = a.y - z.y;
    out[(size_t)b * n + out0] = make_cuFloatComplex(a.x + z.x, a.y + z.y);
    out[(size_t)b * n + out1] = make_cuFloatComplex(fmaf(dx, c, -dy * s),
                                                    fmaf(dx, s,  dy * c));
}

__global__ void stockham_stage_mul_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    const cuFloatComplex* __restrict__ mul,
    int n, int l, int m, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = (n / 2) * batch;
    if (idx >= butterflies) return;

    const int b = idx / (n / 2);
    const int t = idx - b * (n / 2);
    const int j = t / m;
    const int k = t - j * m;

    const int in0 = j * (2 * m) + k;
    const int in1 = in0 + m;
    const int out0 = j * m + k;
    const int out1 = out0 + n / 2;

    const float two_pi = 6.28318530717958647f;
    const float angle = (inverse ? two_pi : -two_pi) * (float)k / (float)(2 * m);
    float s, c;
    __sincosf(angle, &s, &c);

    cuFloatComplex x0 = in[(size_t)b * n + in0];
    cuFloatComplex x1 = in[(size_t)b * n + in1];
    cuFloatComplex w0 = mul[in0];
    cuFloatComplex w1 = mul[in1];
    cuFloatComplex a = make_cuFloatComplex(fmaf(x0.x, w0.x, -x0.y * w0.y),
                                           fmaf(x0.x, w0.y,  x0.y * w0.x));
    cuFloatComplex z = make_cuFloatComplex(fmaf(x1.x, w1.x, -x1.y * w1.y),
                                           fmaf(x1.x, w1.y,  x1.y * w1.x));

    const float dx = a.x - z.x;
    const float dy = a.y - z.y;
    out[(size_t)b * n + out0] = make_cuFloatComplex(a.x + z.x, a.y + z.y);
    out[(size_t)b * n + out1] = make_cuFloatComplex(fmaf(dx, c, -dy * s),
                                                    fmaf(dx, s,  dy * c));
}

__global__ void stockham_two_stage_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int n, int l, int m, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = (n / 2) * batch;
    if (idx >= butterflies) return;

    const int b = idx / (n / 2);
    const int t = idx - b * (n / 2);
    const int m2 = m >> 1;
    const int j2 = t / m2;
    const int k = t - j2 * m2;
    const int j = j2 & (l - 1);
    const int lower = j2 >= l;

    const int base = j * (2 * m);
    const int i0 = base + k;
    const int i1 = i0 + m2;
    const int i2 = i0 + m;
    const int i3 = i2 + m2;

    const float two_pi = 6.28318530717958647f;
    const float sign = inverse ? two_pi : -two_pi;
    float s0, c0, s1, c1, sb, cb;
    __sincosf(sign * (float)k / (float)(2 * m), &s0, &c0);
    __sincosf(sign * (float)(k + m2) / (float)(2 * m), &s1, &c1);
    __sincosf(sign * (float)k / (float)m, &sb, &cb);

    const size_t row = (size_t)b * n;
    cuFloatComplex x0 = in[row + i0];
    cuFloatComplex x1 = in[row + i1];
    cuFloatComplex x2 = in[row + i2];
    cuFloatComplex x3 = in[row + i3];

    // Branchless: upper outputs (lower==false) use twiddle=(1,0) so d*c-d*s == d.
    // Both paths share the same instruction structure — compiler emits predicated
    // selects instead of a divergent BRA.
    const float d0x = x0.x - x2.x;
    const float d0y = x0.y - x2.y;
    const float d1x = x1.x - x3.x;
    const float d1y = x1.y - x3.y;
    // upper: a = sum; lower: a = twiddle(diff)
    const float a0x = lower ? fmaf(d0x, c0, -d0y * s0) : x0.x + x2.x;
    const float a0y = lower ? fmaf(d0x, s0,  d0y * c0) : x0.y + x2.y;
    const float a1x = lower ? fmaf(d1x, c1, -d1y * s1) : x1.x + x3.x;
    const float a1y = lower ? fmaf(d1x, s1,  d1y * c1) : x1.y + x3.y;

    const float dx = a0x - a1x;
    const float dy = a0y - a1y;
    const int out0 = j2 * m2 + k;
    const int out1 = out0 + n / 2;
    out[row + out0] = make_cuFloatComplex(a0x + a1x, a0y + a1y);
    out[row + out1] = make_cuFloatComplex(fmaf(dx, cb, -dy * sb),
                                          fmaf(dx, sb,  dy * cb));
}

__global__ void pointwise_mul_batched_kernel(
    cuFloatComplex* __restrict__ a,
    const cuFloatComplex* __restrict__ b,
    int n, int batch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * batch;
    if (idx >= total) return;
    const int k = idx % n;
    cuFloatComplex x = a[idx];
    cuFloatComplex y = b[k];
    a[idx] = make_cuFloatComplex(fmaf(x.x, y.x, -x.y * y.y),
                                 fmaf(x.x, y.y,  x.y * y.x));
}

__global__ void csc_build_bluestein_input_kernel(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    const int* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = sparse_cols[col_id];

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];
    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(re * q.x - im * q.y, re * q.y + im * q.x);
}

__global__ void csc_build_bluestein_input_packed_kernel(
    const int* __restrict__ col_ptr,
    const uint16_t* __restrict__ row_idx,
    const uint16_t* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = (int)sparse_cols[col_id];

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];
    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, (int)row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(fmaf(re, q.x, -im * q.y),
                            fmaf(re, q.y,  im * q.x));
}

__global__ void csc_build_bluestein_input_coalesced_kernel(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    const int* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x * CSC_BUILD_COLS + threadIdx.x;
    const int u_local = blockIdx.y * CSC_BUILD_U + threadIdx.y;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = sparse_cols[col_id];

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];
    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(re * q.x - im * q.y, re * q.y + im * q.x);
}

__global__ void csc_build_bluestein_input_coalesced_packed_kernel(
    const int* __restrict__ col_ptr,
    const uint16_t* __restrict__ row_idx,
    const uint16_t* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x * CSC_BUILD_COLS + threadIdx.x;
    const int u_local = blockIdx.y * CSC_BUILD_U + threadIdx.y;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = (int)sparse_cols[col_id];

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];
    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, (int)row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(re * q.x - im * q.y, re * q.y + im * q.x);
}

__global__ void bluestein_finalize_kernel(
    const cuFloatComplex* __restrict__ conv,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ out,
    int rows, int output_stride, int output_cols,
    int fft_len, int u_base, int tile_rows)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u_local = blockIdx.y * blockDim.y + threadIdx.y;
    const int u = u_base + u_local;
    if (v >= output_cols || u_local >= tile_rows || u >= rows) return;

    cuFloatComplex y = conv[(size_t)u_local * fft_len + v];
    cuFloatComplex q = chirp[v];
    out[(size_t)u * output_stride + v] =
        make_cuFloatComplex(fmaf(y.x, q.x, -y.y * q.y),
                            fmaf(y.x, q.y,  y.y * q.x));
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Return sorted, deduplicated list of column indices present in the sparse matrix.
// G[i][u] stores only these columns after pass 1, so pass 2 can skip all other
// columns entirely.
static std::vector<int> make_sparse_cols(const int* col_idx, int nnz) {
    std::vector<int> cols(col_idx, col_idx + nnz);
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    return cols;
}

static std::vector<int> make_sparse_col_ids(
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

static bool can_pack_u16(int rows, int cols) {
    return rows <= 65536 && cols <= 65536;
}

static std::vector<uint32_t> pack_coo_u16(const COOMatrix& coo) {
    std::vector<uint32_t> packed(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) {
        packed[i] = ((uint32_t)coo.row_idx[i] << 16)
                  |  (uint32_t)coo.col_idx[i];
    }
    return packed;
}

static std::vector<uint32_t> pack_coo_u16_with_col_ids(
    const COOMatrix& coo, const std::vector<int>& col_ids)
{
    std::vector<uint32_t> packed(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) {
        packed[i] = ((uint32_t)coo.row_idx[i] << 16)
                  |  (uint32_t)col_ids[i];
    }
    return packed;
}

static std::vector<uint16_t> pack_cols_u16(const int* col_idx, int n) {
    std::vector<uint16_t> packed(n);
    for (int i = 0; i < n; i++)
        packed[i] = (uint16_t)col_idx[i];
    return packed;
}

static int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static int log2_exact(int n) {
    int bits = 0;
    while ((1 << bits) < n) bits++;
    return bits;
}

static void run_fft_power2(cuFloatComplex* d_in,
                           cuFloatComplex* d_tmp,
                           int n, int batch, bool inverse)
{
    const int total = n * batch;
    const int threads = 256;
    const int log_n = log2_exact(n);
    bit_reverse_copy_kernel<<<(total + threads - 1) / threads, threads>>>(
        d_in, d_tmp, n, log_n, batch);
    CUDA_CHECK(cudaGetLastError());

    for (int half = 1; half < n; half <<= 1) {
        const int butterflies = (n / 2) * batch;
        fft_stage_kernel<<<(butterflies + threads - 1) / threads, threads>>>(
            d_tmp, n, half, batch, inverse ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
    }

    if (inverse) {
        scale_complex_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_tmp, total, 1.f / (float)n);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void run_fft_power2_stockham(cuFloatComplex* d_in,
                                    cuFloatComplex* d_tmp,
                                    int n, int batch, bool inverse)
{
    const int threads = 256;
    const int butterflies = (n / 2) * batch;
    cuFloatComplex* src = d_in;
    cuFloatComplex* dst = d_tmp;

    for (int l = 1; l < n; ) {
        const int m = n / (2 * l);
        if ((l << 1) < n) {
            stockham_two_stage_kernel<<<(butterflies + threads - 1) / threads, threads>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            l <<= 2;
        } else {
            stockham_stage_kernel<<<(butterflies + threads - 1) / threads, threads>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            l <<= 1;
        }
        cuFloatComplex* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (src != d_tmp) {
        CUDA_CHECK(cudaMemcpy(d_tmp, src,
                              (size_t)n * batch * sizeof(cuFloatComplex),
                              cudaMemcpyDeviceToDevice));
    }

    if (inverse) {
        const int total = n * batch;
        scale_complex_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_tmp, total, 1.f / (float)n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Returns pointer to whichever buffer (d_in or d_tmp) holds the FFT result,
// eliminating the D2D copy needed when the stage count is odd.
static cuFloatComplex* run_fft_power2_stockham_stream(cuFloatComplex* d_in,
                                                      cuFloatComplex* d_tmp,
                                                      int n, int batch, bool inverse,
                                                      cudaStream_t stream)
{
    const int threads = 256;
    const int butterflies = (n / 2) * batch;
    cuFloatComplex* src = d_in;
    cuFloatComplex* dst = d_tmp;

    for (int l = 1; l < n; ) {
        const int m = n / (2 * l);
        if ((l << 1) < n) {
            stockham_two_stage_kernel<<<(butterflies + threads - 1) / threads,
                                        threads, 0, stream>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            l <<= 2;
        } else {
            stockham_stage_kernel<<<(butterflies + threads - 1) / threads,
                                    threads, 0, stream>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            l <<= 1;
        }
        cuFloatComplex* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (inverse) {
        const int total = n * batch;
        scale_complex_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
            src, total, 1.f / (float)n);
        CUDA_CHECK(cudaGetLastError());
    }
    return src;
}

// Returns pointer to whichever buffer (d_in or d_tmp) holds the IFFT result.
static cuFloatComplex* run_fft_power2_stockham_mul_inverse_stream(
    cuFloatComplex* d_in,
    cuFloatComplex* d_tmp,
    const cuFloatComplex* d_mul,
    int n, int batch, cudaStream_t stream)
{
    const int threads = 256;
    const int butterflies = (n / 2) * batch;

    // First stage: multiply by chirp and do first butterfly d_in → d_tmp.
    int l = 1;
    int m = n / 2;
    stockham_stage_mul_kernel<<<(butterflies + threads - 1) / threads,
                                threads, 0, stream>>>(
        d_in, d_tmp, d_mul, n, l, m, batch, 1);
    CUDA_CHECK(cudaGetLastError());

    cuFloatComplex* src = d_tmp;
    cuFloatComplex* dst = d_in;
    for (l = 2; l < n; ) {
        m = n / (2 * l);
        if ((l << 1) < n) {
            stockham_two_stage_kernel<<<(butterflies + threads - 1) / threads,
                                        threads, 0, stream>>>(
                src, dst, n, l, m, batch, 1);
            CUDA_CHECK(cudaGetLastError());
            l <<= 2;
        } else {
            stockham_stage_kernel<<<(butterflies + threads - 1) / threads,
                                    threads, 0, stream>>>(
                src, dst, n, l, m, batch, 1);
            CUDA_CHECK(cudaGetLastError());
            l <<= 1;
        }
        cuFloatComplex* tmp = src;
        src = dst;
        dst = tmp;
    }

    const int total = n * batch;
    scale_complex_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        src, total, 1.f / (float)n);
    CUDA_CHECK(cudaGetLastError());
    return src;
}

// ---------------------------------------------------------------------------
// Improvement 2 host runners — smem path
//
// Returns d_in (in-place).  When fft_len > SMEM_MAX_FFT_N the call transparently
// falls back to the per-stage stream variant so callers need not branch.
// ---------------------------------------------------------------------------
static cuFloatComplex* run_fft_power2_stockham_smem_stream(
    cuFloatComplex* d_in, cuFloatComplex* d_tmp,
    int n, int batch, bool inverse, cudaStream_t stream)
{
    const size_t smem_needed = 2 * (size_t)n * sizeof(cuFloatComplex);
    if (n > SMEM_MAX_FFT_N || smem_needed > 96 * 1024UL) {
        return run_fft_power2_stockham_stream(d_in, d_tmp, n, batch, inverse, stream);
    }
    const int bdim = std::min(512, n / 2);
    stockham_smem_row_kernel<<<batch, bdim, smem_needed, stream>>>(
        d_in, n, batch, inverse ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    return d_in; // in-place: result always in d_in
}

static cuFloatComplex* run_fft_power2_stockham_smem_mul_inverse_stream(
    cuFloatComplex* d_in, cuFloatComplex* d_tmp,
    const cuFloatComplex* d_mul, int n, int batch, cudaStream_t stream)
{
    const size_t smem_needed = 2 * (size_t)n * sizeof(cuFloatComplex);
    if (n > SMEM_MAX_FFT_N || smem_needed > 96 * 1024UL) {
        return run_fft_power2_stockham_mul_inverse_stream(d_in, d_tmp, d_mul, n, batch, stream);
    }
    const int bdim = std::min(512, n / 2);
    stockham_smem_row_mul_inverse_kernel<<<batch, bdim, smem_needed, stream>>>(
        d_in, d_mul, n, batch);
    CUDA_CHECK(cudaGetLastError());
    return d_in; // in-place: result always in d_in
}

// ---------------------------------------------------------------------------
static std::vector<cuFloatComplex> make_bluestein_chirp(int n) {
    std::vector<cuFloatComplex> chirp(n);
    const double pi = 3.1415926535897932384626433832795;
    for (int k = 0; k < n; k++) {
        const double kk = (double)k * (double)k;
        const double angle = -pi * kk / (double)n;
        chirp[k] = make_cuFloatComplex((float)std::cos(angle), (float)std::sin(angle));
    }
    return chirp;
}

static std::vector<cuFloatComplex> make_bluestein_b(int n, int fft_len) {
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

static CompactCSC make_compact_csc(const COOMatrix& coo) {
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

SparseFFTResult sparse_fft(const COOMatrix& coo) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const bool use_packed = can_pack_u16(rows, cols);

    int *d_row = nullptr, *d_col = nullptr;
    uint32_t* d_packed = nullptr;
    if (use_packed) {
        std::vector<uint32_t> packed = pack_coo_u16(coo);
        CUDA_CHECK(cudaMalloc(&d_packed, coo.nnz * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), coo.nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row, coo.row_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col, coo.col_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    }

    cuFloatComplex* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)rows * cols * sizeof(cuFloatComplex)));

    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    if (use_packed) {
        sparse_fft_tiled_packed_kernel<<<grid, block>>>(
            d_packed, d_out, rows, cols, coo.nnz,
            1.f / (float)rows, 1.f / (float)cols);
    } else {
        sparse_fft_tiled_kernel<<<grid, block>>>(
            d_row, d_col, d_out, rows, cols, coo.nnz,
            1.f / (float)rows, 1.f / (float)cols);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_packed);

    const size_t input_bytes = use_packed
                             ? (size_t)coo.nnz * sizeof(uint32_t)
                             : (size_t)coo.nnz * 2 * sizeof(int);
    return {d_out, cols, ms, input_bytes
                     + (size_t)rows * cols * sizeof(cuFloatComplex)};
}

SparseFFTResult sparse_fft_2pass(const COOMatrix& coo) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const bool use_packed = can_pack_u16(rows, cols);

    // Distinct column indices — only these columns of G are nonzero after pass 1
    std::vector<int> sp_cols = make_sparse_cols(coo.col_idx.data(), coo.nnz);
    std::vector<int> col_ids = make_sparse_col_ids(coo.col_idx.data(), coo.nnz, sp_cols, cols);
    const int n_sp = (int)sp_cols.size();

    int *d_row = nullptr, *d_col = nullptr, *d_sp_cols = nullptr;
    uint32_t* d_packed = nullptr;
    uint16_t* d_sp_cols_packed = nullptr;
    if (use_packed) {
        std::vector<uint32_t> packed = pack_coo_u16_with_col_ids(coo, col_ids);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(sp_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_packed,        coo.nnz * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp   * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), coo.nnz * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(), n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row,     coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col,     coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp    * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row,     coo.row_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col,     col_ids.data(),     coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, sp_cols.data(),     n_sp    * sizeof(int), cudaMemcpyHostToDevice));
    }

    cuFloatComplex *d_G, *d_out;
    CUDA_CHECK(cudaMalloc(&d_G,   (size_t)rows * n_sp * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)rows * output_cols * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemset(d_G, 0, (size_t)rows * n_sp * sizeof(cuFloatComplex)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    // Pass 1: COO scatter into G
    dim3 p1_block(BLOCK_U);
    dim3 p1_grid((coo.nnz + NNZ_TILE - 1) / NNZ_TILE,
                 (rows    + BLOCK_U  - 1) / BLOCK_U);
    if (use_packed) {
        build_intermediate_packed<<<p1_grid, p1_block>>>(
            d_packed, d_G, rows, cols, coo.nnz, 1.f / (float)rows);
    } else {
        build_intermediate<<<p1_grid, p1_block>>>(
            d_row, d_col, d_G, rows, cols, coo.nnz, 1.f / (float)rows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: custom column DFT over sparse columns of G
    dim3 p2_block(P2_BV, P2_BU);
    dim3 p2_grid((output_cols + P2_BV - 1) / P2_BV,
                 (rows + P2_BU - 1) / P2_BU);
    if (use_packed) {
        col_dft_kernel_packed<<<p2_grid, p2_block>>>(
            d_G, d_sp_cols_packed, n_sp, d_out, rows, cols, output_cols, 1.f / (float)cols);
    } else {
        col_dft_kernel<<<p2_grid, p2_block>>>(
            d_G, d_sp_cols, n_sp, d_out, rows, cols, output_cols, 1.f / (float)cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_sp_cols);
    cudaFree(d_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_G);

    const size_t input_bytes = use_packed
                             ? (size_t)coo.nnz * sizeof(uint32_t) + (size_t)n_sp * sizeof(uint16_t)
                             : (size_t)coo.nnz * 2 * sizeof(int) + (size_t)n_sp * sizeof(int);
    return {d_out, output_cols, ms, input_bytes
                     + (size_t)rows * n_sp * sizeof(cuFloatComplex)
                     + (size_t)rows * output_cols * sizeof(cuFloatComplex)};
}

SparseFFTResult sparse_fft_2pass_cufft(const COOMatrix& coo) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const bool use_packed = can_pack_u16(rows, cols);

    int *d_row = nullptr, *d_col = nullptr;
    uint32_t* d_packed = nullptr;
    if (use_packed) {
        std::vector<uint32_t> packed = pack_coo_u16(coo);
        CUDA_CHECK(cudaMalloc(&d_packed, coo.nnz * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_packed, packed.data(),
                              coo.nnz * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row, coo.row_idx.data(),
                              coo.nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col, coo.col_idx.data(),
                              coo.nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    cuFloatComplex *d_G, *d_out;
    CUDA_CHECK(cudaMalloc(&d_G,   (size_t)rows * cols * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)rows * cols * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemset(d_G, 0, (size_t)rows * cols * sizeof(cuFloatComplex)));

    cufftHandle plan;
    int n[1] = {cols};
    int inembed[1] = {cols};
    int onembed[1] = {cols};
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                              inembed, rows, 1,
                              onembed, 1, cols,
                              CUFFT_C2C, rows));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    // Pass 1: COO scatter into dense column-major G[c][u].
    dim3 p1_block(BLOCK_U);
    dim3 p1_grid((coo.nnz + NNZ_TILE - 1) / NNZ_TILE,
                 (rows    + BLOCK_U  - 1) / BLOCK_U);
    if (use_packed) {
        build_intermediate_packed<<<p1_grid, p1_block>>>(
            d_packed, d_G, rows, cols, coo.nnz, 1.f / (float)rows);
    } else {
        build_intermediate<<<p1_grid, p1_block>>>(
            d_row, d_col, d_G, rows, cols, coo.nnz, 1.f / (float)rows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: each batch is one fixed output row u, strided through G columns.
    CUFFT_CHECK(cufftExecC2C(plan,
                             reinterpret_cast<cufftComplex*>(d_G),
                             reinterpret_cast<cufftComplex*>(d_out),
                             CUFFT_FORWARD));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cufftDestroy(plan);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_packed);
    cudaFree(d_G);

    const size_t input_bytes = use_packed
                             ? (size_t)coo.nnz * sizeof(uint32_t)
                             : (size_t)coo.nnz * 2 * sizeof(int);
    return {d_out, cols, ms, input_bytes
                     + 2 * (size_t)rows * cols * sizeof(cuFloatComplex)};
}

SparseFFTResult sparse_fft_2pass_gemm(const COOMatrix& coo, bool use_tf32) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const bool use_packed = can_pack_u16(rows, cols);

    std::vector<int> sp_cols = make_sparse_cols(coo.col_idx.data(), coo.nnz);
    std::vector<int> col_ids = make_sparse_col_ids(coo.col_idx.data(), coo.nnz, sp_cols, cols);
    const int n_sp = (int)sp_cols.size();

    int *d_row = nullptr, *d_col = nullptr, *d_sp_cols = nullptr;
    uint32_t* d_packed = nullptr;
    uint16_t* d_sp_cols_packed = nullptr;
    if (use_packed) {
        std::vector<uint32_t> packed = pack_coo_u16_with_col_ids(coo, col_ids);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(sp_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_packed, coo.nnz * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_packed, packed.data(),
                              coo.nnz * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row, coo.row_idx.data(),
                              coo.nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col, col_ids.data(),
                              coo.nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, sp_cols.data(),
                              n_sp * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    cuFloatComplex *d_G, *d_W, *d_F_colmajor, *d_out;
    CUDA_CHECK(cudaMalloc(&d_G,          (size_t)rows * n_sp * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_W,          (size_t)n_sp * output_cols * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_F_colmajor, (size_t)rows * output_cols * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out,        (size_t)rows * output_cols * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemset(d_G, 0, (size_t)rows * n_sp * sizeof(cuFloatComplex)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (use_tf32)
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    // Pass 1: COO scatter into compact column-major G[sparse_col_id][u].
    dim3 p1_block(BLOCK_U);
    dim3 p1_grid((coo.nnz + NNZ_TILE - 1) / NNZ_TILE,
                 (rows    + BLOCK_U  - 1) / BLOCK_U);
    if (use_packed) {
        build_intermediate_packed<<<p1_grid, p1_block>>>(
            d_packed, d_G, rows, cols, coo.nnz, 1.f / (float)rows);
    } else {
        build_intermediate<<<p1_grid, p1_block>>>(
            d_row, d_col, d_G, rows, cols, coo.nnz, 1.f / (float)rows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Pass 2a: materialize W[i, v] in column-major order for cuBLAS.
    dim3 w_block(16, 16);
    dim3 w_grid((n_sp + w_block.x - 1) / w_block.x,
                (output_cols + w_block.y - 1) / w_block.y);
    if (use_packed) {
        build_twiddle_matrix_packed_kernel<<<w_grid, w_block>>>(
            d_sp_cols_packed, d_W, n_sp, cols, output_cols, 1.f / (float)cols);
    } else {
        build_twiddle_matrix_kernel<<<w_grid, w_block>>>(
            d_sp_cols, d_W, n_sp, cols, output_cols, 1.f / (float)cols);
    }
    CUDA_CHECK(cudaGetLastError());

    // Pass 2b: F_colmajor[rows x output_cols] = G[rows x n_sp] * W[n_sp x output_cols].
    const cuFloatComplex alpha = make_cuFloatComplex(1.f, 0.f);
    const cuFloatComplex beta  = make_cuFloatComplex(0.f, 0.f);
    if (use_tf32) {
        CUBLAS_CHECK(cublasGemmEx(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  rows, output_cols, n_sp,
                                  reinterpret_cast<const cuComplex*>(&alpha),
                                  d_G, CUDA_C_32F, rows,
                                  d_W, CUDA_C_32F, n_sp,
                                  reinterpret_cast<const cuComplex*>(&beta),
                                  d_F_colmajor, CUDA_C_32F, rows,
                                  CUBLAS_COMPUTE_32F_FAST_TF32,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        CUBLAS_CHECK(cublasCgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 rows, output_cols, n_sp,
                                 reinterpret_cast<const cuComplex*>(&alpha),
                                 reinterpret_cast<const cuComplex*>(d_G), rows,
                                 reinterpret_cast<const cuComplex*>(d_W), n_sp,
                                 reinterpret_cast<const cuComplex*>(&beta),
                                 reinterpret_cast<cuComplex*>(d_F_colmajor), rows));
    }

    // Keep the public output layout row-major, same as the custom sparse kernels.
    dim3 copy_block(32, 8);
    dim3 copy_grid((output_cols + copy_block.x - 1) / copy_block.x,
                   (rows + copy_block.y - 1) / copy_block.y);
    colmajor_to_rowmajor_kernel<<<copy_grid, copy_block>>>(
        d_F_colmajor, d_out, rows, output_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cublasDestroy(handle);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_sp_cols);
    cudaFree(d_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_G);
    cudaFree(d_W);
    cudaFree(d_F_colmajor);

    const size_t input_bytes = use_packed
                             ? (size_t)coo.nnz * sizeof(uint32_t) + (size_t)n_sp * sizeof(uint16_t)
                             : (size_t)coo.nnz * 2 * sizeof(int) + (size_t)n_sp * sizeof(int);
    return {d_out, output_cols, ms, input_bytes
                     + (size_t)rows * n_sp * sizeof(cuFloatComplex)
                     + (size_t)n_sp * output_cols * sizeof(cuFloatComplex)
                     + 2 * (size_t)rows * output_cols * sizeof(cuFloatComplex)};
}

SparseFFTResult sparse_fft_csc_2pass(
    const COOMatrix& coo, int u_tile, bool precompute_twiddle, bool tiled_pass2, bool half_g)
{
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    // Pad to the next multiple of 4 elements (= 32 bytes) so every output row
    // starts on a 32-byte L2 sector boundary.  Without this, rows whose byte
    // offset is not sector-aligned (output_cols % 4 != 0) cause partial-sector
    // global stores (profiler: ~26.9/32 bytes used).  Only applied for the
    // tiled kernel; compare_sparse_to_dense already handles a padded pitch via
    // cudaMemcpy2D, using output_cols (actual bins) as the copy width.
    const int output_stride = tiled_pass2
        ? ((output_cols + 3) / 4 * 4)
        : output_cols;
    const bool use_packed = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");
    if (half_g && !tiled_pass2)
        throw std::runtime_error("Half G storage is only implemented for tiled CSC pass 2");

    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;

    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> row_idx_packed = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, row_idx_packed.data(),
                              coo.nnz * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              coo.nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              n_sp * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_G = nullptr, *d_out, *d_W = nullptr;
    __half2* d_G_half = nullptr;
    if (half_g)
        CUDA_CHECK(cudaMalloc(&d_G_half, (size_t)max_tile_rows * n_sp * sizeof(__half2)));
    else
        CUDA_CHECK(cudaMalloc(&d_G,      (size_t)max_tile_rows * n_sp * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)rows * output_stride * sizeof(cuFloatComplex)));
    if (precompute_twiddle)
        CUDA_CHECK(cudaMalloc(&d_W, (size_t)n_sp * output_cols * sizeof(cuFloatComplex)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    if (precompute_twiddle) {
        dim3 w_block(32, 8);
        dim3 w_grid((output_cols + w_block.x - 1) / w_block.x,
                    (n_sp + w_block.y - 1) / w_block.y);
        if (use_packed) {
            build_twiddle_matrix_rowmajor_packed_kernel<<<w_grid, w_block>>>(
                d_sp_cols_packed, d_W, n_sp, cols, output_cols, 1.f / (float)cols);
        } else {
            build_twiddle_matrix_rowmajor_kernel<<<w_grid, w_block>>>(
                d_sp_cols, d_W, n_sp, cols, output_cols, 1.f / (float)cols);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);

        // Pass 1: one thread computes one G_chunk[active_col][u_local], no atomics needed.
        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (half_g && use_packed) {
            csc_build_intermediate_chunk_half_packed<<<p1_grid, p1_block>>>(
                d_col_ptr, d_row_idx_packed, d_G_half, rows, n_sp, u_base, tile_rows, 1.f / (float)rows);
        } else if (half_g) {
            csc_build_intermediate_chunk_half<<<p1_grid, p1_block>>>(
                d_col_ptr, d_row_idx, d_G_half, rows, n_sp, u_base, tile_rows, 1.f / (float)rows);
        } else if (use_packed) {
            csc_build_intermediate_chunk_packed<<<p1_grid, p1_block>>>(
                d_col_ptr, d_row_idx_packed, d_G, rows, n_sp, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_intermediate_chunk<<<p1_grid, p1_block>>>(
                d_col_ptr, d_row_idx, d_G, rows, n_sp, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        // Pass 2: compute only this output-row tile, then reuse G_chunk for the next tile.
        if (tiled_pass2) {
            dim3 p2_block(CSC_TILED_V, CSC_TILED_U);
            // Grid uses output_stride (padded) so every block's v-base is
            // sector-aligned; the kernel bounds-checks v < output_stride.
            dim3 p2_grid(
                (output_stride + CSC_TILED_V * CSC_TILED_RBV - 1) / (CSC_TILED_V * CSC_TILED_RBV),
                (tile_rows     + CSC_TILED_U * CSC_TILED_RBU - 1) / (CSC_TILED_U * CSC_TILED_RBU));
            if (half_g && use_packed) {
                col_dft_chunk_tiled_half_kernel_packed<<<p2_grid, p2_block>>>(
                    d_G_half, d_sp_cols_packed, n_sp, d_out,
                    rows, cols, output_stride, u_base, tile_rows, 1.f / (float)cols);
            } else if (half_g) {
                col_dft_chunk_tiled_half_kernel<<<p2_grid, p2_block>>>(
                    d_G_half, d_sp_cols, n_sp, d_out,
                    rows, cols, output_stride, u_base, tile_rows, 1.f / (float)cols);
            } else if (use_packed) {
                col_dft_chunk_tiled_kernel_packed<<<p2_grid, p2_block>>>(
                    d_G, d_sp_cols_packed, n_sp, d_out,
                    rows, cols, output_stride, u_base, tile_rows, 0, 1.f / (float)cols);
            } else {
                col_dft_chunk_tiled_kernel<<<p2_grid, p2_block>>>(
                    d_G, d_sp_cols, n_sp, d_out,
                    rows, cols, output_stride, u_base, tile_rows, 0, 1.f / (float)cols);
            }
        } else if (precompute_twiddle) {
            dim3 p2_block(P2_BV, P2_BU);
            dim3 p2_grid((output_cols + P2_BV - 1) / P2_BV,
                         (tile_rows + P2_BU - 1) / P2_BU);
            col_dft_chunk_cached_w_kernel<<<p2_grid, p2_block>>>(
                d_G, d_W, n_sp, d_out, rows, output_cols, u_base, tile_rows);
        } else {
            dim3 p2_block(P2_BV, P2_BU);
            dim3 p2_grid((output_cols + P2_BV - 1) / P2_BV,
                         (tile_rows + P2_BU - 1) / P2_BU);
            if (use_packed) {
                col_dft_chunk_kernel_packed<<<p2_grid, p2_block>>>(
                    d_G, d_sp_cols_packed, n_sp, d_out,
                    rows, cols, output_cols, u_base, tile_rows, 1.f / (float)cols);
            } else {
                col_dft_chunk_kernel<<<p2_grid, p2_block>>>(
                    d_G, d_sp_cols, n_sp, d_out,
                    rows, cols, output_cols, u_base, tile_rows, 1.f / (float)cols);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_G);
    cudaFree(d_G_half);
    cudaFree(d_W);

    const size_t input_bytes = (size_t)(n_sp + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)coo.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)coo.nnz + (size_t)n_sp) * sizeof(int));
    const size_t twiddle_bytes = precompute_twiddle
                              ? (size_t)n_sp * output_cols * sizeof(cuFloatComplex)
                              : 0;
    return {d_out, output_stride, ms, input_bytes
                     + (size_t)max_tile_rows * n_sp * (half_g ? sizeof(__half2) : sizeof(cuFloatComplex))
                     + twiddle_bytes
                     + (size_t)rows * output_stride * sizeof(cuFloatComplex)};
}

SparseFFTResult sparse_fft_csc_streaming(const COOMatrix& coo, int u_tile) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const bool use_packed = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;

    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> row_idx_packed = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, row_idx_packed.data(),
                              coo.nnz * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              coo.nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              n_sp * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_G = nullptr, *d_out_chunk = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)max_tile_rows * n_sp * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk, (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));

    std::vector<cuFloatComplex> h_output((size_t)rows * output_stride);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);

        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_intermediate_chunk_packed<<<p1_grid, p1_block>>>(
                d_col_ptr, d_row_idx_packed, d_G, rows, n_sp, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_intermediate_chunk<<<p1_grid, p1_block>>>(
                d_col_ptr, d_row_idx, d_G, rows, n_sp, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        dim3 p2_block(CSC_TILED_V, CSC_TILED_U);
        dim3 p2_grid(
            (output_stride + CSC_TILED_V * CSC_TILED_RBV - 1) / (CSC_TILED_V * CSC_TILED_RBV),
            (tile_rows     + CSC_TILED_U * CSC_TILED_RBU - 1) / (CSC_TILED_U * CSC_TILED_RBU));
        if (use_packed) {
            col_dft_chunk_tiled_kernel_packed<<<p2_grid, p2_block>>>(
                d_G, d_sp_cols_packed, n_sp, d_out_chunk,
                rows, cols, output_stride, u_base, tile_rows, u_base, 1.f / (float)cols);
        } else {
            col_dft_chunk_tiled_kernel<<<p2_grid, p2_block>>>(
                d_G, d_sp_cols, n_sp, d_out_chunk,
                rows, cols, output_stride, u_base, tile_rows, u_base, 1.f / (float)cols);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_output.data() + (size_t)u_base * output_stride,
                              d_out_chunk,
                              (size_t)tile_rows * output_stride * sizeof(cuFloatComplex),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_G);
    cudaFree(d_out_chunk);

    const size_t input_bytes = (size_t)(n_sp + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)coo.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)coo.nnz + (size_t)n_sp) * sizeof(int));
    return {nullptr, output_stride, ms, input_bytes
                     + (size_t)max_tile_rows * n_sp * sizeof(cuFloatComplex)
                     + (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex),
            std::move(h_output)};
}

// fft_backend: 0 = default two-stage Stockham (original),
//              1 = smem row kernel (improvement 2; falls back for large fft_len),
//              2 = cooperative kernel (improvement 4; falls back if unsupported).
SparseFFTResult sparse_fft_csc_bluestein(const COOMatrix& coo, int u_tile,
                                         bool use_stockham, int fft_backend) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len = next_power_of_two(2 * cols - 1);
    const bool use_packed = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> row_idx_packed = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, row_idx_packed.data(),
                              coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp,  cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft,  fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(), cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // fft_backend==3 (cuFFT) operates in-place on d_signal — d_work is not needed.
    // Backends 0 and 1 use d_work as a ping-pong scratch buffer.
    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr, *d_work = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal, (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    if (fft_backend != 3)
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out,    (size_t)rows * output_stride * sizeof(cuFloatComplex)));

    // Stage h_b through d_work for the one-time d_b_fft precompute.
    // fft_backend==3 (cuFFT) uses an in-place plan directly on d_b_fft; all others
    // stage through d_work so the tile loop can overwrite it freely.
    if (fft_backend == 3) {
        // cuFFT backend: copy h_b directly to d_b_fft and FFT in-place.
        CUDA_CHECK(cudaMemcpy(d_b_fft, h_b.data(), fft_len * sizeof(cuFloatComplex),
                              cudaMemcpyHostToDevice));
        int fl = fft_len;
        cufftHandle plan_b;
        CUFFT_CHECK(cufftPlanMany(&plan_b, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, 1));
        CUFFT_CHECK(cufftExecC2C(plan_b,
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 CUFFT_FORWARD));
        CUFFT_CHECK(cufftDestroy(plan_b));
    } else {
        CUDA_CHECK(cudaMemcpy(d_work, h_b.data(), fft_len * sizeof(cuFloatComplex),
                              cudaMemcpyHostToDevice));
        if (use_stockham)
            run_fft_power2_stockham(d_work, d_b_fft, fft_len, 1, false);
        else
            run_fft_power2(d_work, d_b_fft, fft_len, 1, false);
    }

    // Build cuFFT plans for the tile loop (fft_backend==3 only).
    // Create one plan for full tiles and, if needed, a second for the last partial tile.
    // Query cufftGetSize so the internal workspace is included in mem_bytes.
    cufftHandle plan_cufft = 0, plan_cufft_last = 0;
    size_t cufft_workspace = 0;
    if (fft_backend == 3) {
        int fl = fft_len;
        const int full_tile = std::min(u_tile, rows);
        CUFFT_CHECK(cufftPlanMany(&plan_cufft, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, full_tile));
        size_t ws = 0;
        CUFFT_CHECK(cufftGetSize(plan_cufft, &ws));
        cufft_workspace = ws;
        const int last_tile = rows % u_tile;
        if (last_tile != 0 && last_tile != full_tile) {
            CUFFT_CHECK(cufftPlanMany(&plan_cufft_last, 1, &fl, nullptr, 1, fft_len,
                                      nullptr, 1, fft_len, CUFFT_C2C, last_tile));
            CUFFT_CHECK(cufftGetSize(plan_cufft_last, &ws));
            cufft_workspace = std::max(cufft_workspace, ws);
        }
    }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);
        CUDA_CHECK(cudaMemset(d_signal, 0, signal_bytes));

        if (use_stockham) {
            // 1D layout: blockIdx.x = col_id, threadIdx.x = u_local.
            // All u-threads share the same col_id → row_idx[p] is the same address
            // for every thread in the block → L1 broadcast, near-zero excess sectors.
            dim3 p1_block(CSC_BU);
            dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
            if (use_packed) {
                csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block>>>(
                    d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, d_signal,
                    rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
            } else {
                csc_build_bluestein_input_kernel<<<p1_grid, p1_block>>>(
                    d_col_ptr, d_row_idx, d_sp_cols, d_chirp, d_signal,
                    rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
            }
        } else {
            dim3 p1_block(CSC_BU);
            dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
            if (use_packed) {
                csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block>>>(
                    d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, d_signal,
                    rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
            } else {
                csc_build_bluestein_input_kernel<<<p1_grid, p1_block>>>(
                    d_col_ptr, d_row_idx, d_sp_cols, d_chirp, d_signal,
                    rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
            }
        }
        CUDA_CHECK(cudaGetLastError());

        cuFloatComplex* conv_result;
        if (use_stockham) {
            if (fft_backend == 1) {
                // smem row kernel — entire FFT in shared memory (improvement 2).
                // In-place on d_signal; d_work is ping-pong scratch (may be unused
                // if smem path activates, used as fallback otherwise).
                cuFloatComplex* fwd = run_fft_power2_stockham_smem_stream(
                    d_signal, d_work, fft_len, tile_rows, false, 0);
                cuFloatComplex* fwd_other = (fwd == d_signal) ? d_work : d_signal;
                conv_result = run_fft_power2_stockham_smem_mul_inverse_stream(
                    fwd, fwd_other, d_b_fft, fft_len, tile_rows, 0);
            } else if (fft_backend == 3) {
                // cuFFT C2C backend: uses cuFFT's internal smem-staged butterfly,
                // replacing O(N log N) global-memory passes with O(N) per direction.
                // Forward FFT in-place, pointwise ×B_fft, inverse FFT in-place, scale.
                cufftHandle tile_plan =
                    (plan_cufft_last && tile_rows != std::min(u_tile, rows))
                    ? plan_cufft_last : plan_cufft;
                CUFFT_CHECK(cufftExecC2C(tile_plan,
                    reinterpret_cast<cufftComplex*>(d_signal),
                    reinterpret_cast<cufftComplex*>(d_signal),
                    CUFFT_FORWARD));
                const int total_elems = tile_rows * fft_len;
                pointwise_mul_batched_kernel<<<(total_elems + 255) / 256, 256>>>(
                    d_signal, d_b_fft, fft_len, tile_rows);
                CUDA_CHECK(cudaGetLastError());
                CUFFT_CHECK(cufftExecC2C(tile_plan,
                    reinterpret_cast<cufftComplex*>(d_signal),
                    reinterpret_cast<cufftComplex*>(d_signal),
                    CUFFT_INVERSE));
                // cuFFT does not normalize the inverse; divide by fft_len.
                scale_complex_kernel<<<(total_elems + 255) / 256, 256>>>(
                    d_signal, total_elems, 1.f / (float)fft_len);
                CUDA_CHECK(cudaGetLastError());
                conv_result = d_signal;
            } else {
                // Default: per-stage two-stage Stockham loop.
                run_fft_power2_stockham(d_signal, d_work, fft_len, tile_rows, false);
                conv_result = run_fft_power2_stockham_mul_inverse_stream(
                    d_work, d_signal, d_b_fft, fft_len, tile_rows, 0);
            }
        } else {
            run_fft_power2(d_signal, d_work, fft_len, tile_rows, false);
            pointwise_mul_batched_kernel<<<((int)((size_t)tile_rows * fft_len) + 255) / 256, 256>>>(
                d_work, d_b_fft, fft_len, tile_rows);
            CUDA_CHECK(cudaGetLastError());
            run_fft_power2(d_work, d_signal, fft_len, tile_rows, true);
            conv_result = d_signal;
        }

        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block>>>(
            conv_result, d_chirp, d_out,
            rows, output_stride, output_cols, fft_len, u_base, tile_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (plan_cufft)      cufftDestroy(plan_cufft);
    if (plan_cufft_last) cufftDestroy(plan_cufft_last);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_work);  // nullptr-safe when fft_backend==3

    const size_t input_bytes = (size_t)(n_sp + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)coo.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)coo.nnz + (size_t)n_sp) * sizeof(int));
    const size_t work_bytes = (fft_backend != 3)
                            ? (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex) : 0;
    return {d_out, output_stride, ms, input_bytes
                     + (size_t)cols    * sizeof(cuFloatComplex)   // d_chirp
                     + (size_t)fft_len * sizeof(cuFloatComplex)   // d_b_fft
                     + (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)  // d_signal
                     + work_bytes                                                 // d_work (0 for backend 3)
                     + cufft_workspace                                            // cuFFT internal scratch
                     + (size_t)rows * output_stride * sizeof(cuFloatComplex)};   // d_out
}

SparseFFTResult sparse_fft_csc_bluestein_smem(const COOMatrix& coo, int u_tile) {
    return sparse_fft_csc_bluestein(coo, u_tile, true, 1);
}

SparseFFTResult sparse_fft_csc_bluestein_cufft(const COOMatrix& coo, int u_tile) {
    return sparse_fft_csc_bluestein(coo, u_tile, true, 3);
}

SparseFFTResult sparse_fft_csc_stockham_streaming(const COOMatrix& coo, int u_tile,
                                                   int fft_backend) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len = next_power_of_two(2 * cols - 1);
    const bool use_packed = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> row_idx_packed = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, row_idx_packed.data(),
                              coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp,  cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft,  fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(), cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // fft_backend==3 (cuFFT) operates in-place on d_signal — d_work not needed.
    // Backends 0/1 use d_work as ping-pong scratch across Stockham stages.
    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr, *d_work = nullptr;
    cuFloatComplex *d_out_chunk[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_signal, (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    if (fft_backend != 3)
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk[0], (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk[1], (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));

    // Precompute d_b_fft on the null stream before the tile loop.
    // Backend 3: copy h_b directly to d_b_fft and FFT in-place with a one-shot cuFFT plan.
    // Others: stage through d_signal (avoids a separate d_b allocation).
    if (fft_backend == 3) {
        CUDA_CHECK(cudaMemcpy(d_b_fft, h_b.data(), fft_len * sizeof(cuFloatComplex),
                              cudaMemcpyHostToDevice));
        int fl = fft_len;
        cufftHandle plan_b;
        CUFFT_CHECK(cufftPlanMany(&plan_b, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, 1));
        CUFFT_CHECK(cufftExecC2C(plan_b,
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 CUFFT_FORWARD));
        CUFFT_CHECK(cufftDestroy(plan_b));
    } else {
        CUDA_CHECK(cudaMemcpy(d_signal, h_b.data(), fft_len * sizeof(cuFloatComplex),
                              cudaMemcpyHostToDevice));
        run_fft_power2_stockham(d_signal, d_b_fft, fft_len, 1, false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cuFloatComplex* h_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pinned,
                             (size_t)rows * output_stride * sizeof(cuFloatComplex),
                             cudaHostAllocDefault));

    // compute_stream: build + FFTs + finalize
    // copy_stream:    async D2H transfers (overlaps with compute_stream)
    cudaStream_t compute_stream, copy_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&copy_stream));

    // Build cuFFT tile plans (fft_backend==3 only) now that compute_stream exists.
    cufftHandle plan_cufft = 0, plan_cufft_last = 0;
    size_t cufft_workspace = 0;
    if (fft_backend == 3) {
        int fl = fft_len;
        const int full_tile = std::min(u_tile, rows);
        CUFFT_CHECK(cufftPlanMany(&plan_cufft, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, full_tile));
        CUFFT_CHECK(cufftSetStream(plan_cufft, compute_stream));
        size_t ws = 0;
        CUFFT_CHECK(cufftGetSize(plan_cufft, &ws));
        cufft_workspace = ws;
        const int last_tile = rows % u_tile;
        if (last_tile != 0 && last_tile != full_tile) {
            CUFFT_CHECK(cufftPlanMany(&plan_cufft_last, 1, &fl, nullptr, 1, fft_len,
                                      nullptr, 1, fft_len, CUFFT_C2C, last_tile));
            CUFFT_CHECK(cufftSetStream(plan_cufft_last, compute_stream));
            CUFFT_CHECK(cufftGetSize(plan_cufft_last, &ws));
            cufft_workspace = std::max(cufft_workspace, ws);
        }
    }

    // compute_done[b]: signaled after finalize writes d_out_chunk[b]
    // d2h_done[b]:     signaled after D2H of d_out_chunk[b] completes
    // Initialise d2h_done as pre-signaled so the first two tiles don't stall.
    cudaEvent_t compute_done[2], d2h_done[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaEventCreate(&compute_done[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&d2h_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(d2h_done[i], compute_stream)); // immediately satisfied
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaEventSynchronize(t0));

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const int buf = (u_base / u_tile) & 1;
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);

        // Don't overwrite d_out_chunk[buf] until its in-flight D2H (if any) is done.
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, d2h_done[buf]));

        CUDA_CHECK(cudaMemsetAsync(d_signal, 0, signal_bytes, compute_stream));

        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block, 0, compute_stream>>>(
                d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_bluestein_input_kernel<<<p1_grid, p1_block, 0, compute_stream>>>(
                d_col_ptr, d_row_idx, d_sp_cols, d_chirp, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        cuFloatComplex* conv_result;
        if (fft_backend == 3) {
            cufftHandle tile_plan =
                (plan_cufft_last && tile_rows != std::min(u_tile, rows))
                ? plan_cufft_last : plan_cufft;
            CUFFT_CHECK(cufftExecC2C(tile_plan,
                reinterpret_cast<cufftComplex*>(d_signal),
                reinterpret_cast<cufftComplex*>(d_signal),
                CUFFT_FORWARD));
            const int total_elems = tile_rows * fft_len;
            pointwise_mul_batched_kernel<<<(total_elems + 255) / 256, 256, 0, compute_stream>>>(
                d_signal, d_b_fft, fft_len, tile_rows);
            CUDA_CHECK(cudaGetLastError());
            CUFFT_CHECK(cufftExecC2C(tile_plan,
                reinterpret_cast<cufftComplex*>(d_signal),
                reinterpret_cast<cufftComplex*>(d_signal),
                CUFFT_INVERSE));
            scale_complex_kernel<<<(total_elems + 255) / 256, 256, 0, compute_stream>>>(
                d_signal, total_elems, 1.f / (float)fft_len);
            CUDA_CHECK(cudaGetLastError());
            conv_result = d_signal;
        } else {
            cuFloatComplex* fwd_result = run_fft_power2_stockham_stream(
                d_signal, d_work, fft_len, tile_rows, false, compute_stream);
            cuFloatComplex* fwd_other = (fwd_result == d_signal) ? d_work : d_signal;
            conv_result = run_fft_power2_stockham_mul_inverse_stream(
                fwd_result, fwd_other, d_b_fft, fft_len, tile_rows, compute_stream);
        }

        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block, 0, compute_stream>>>(
            conv_result, d_chirp, d_out_chunk[buf],
            tile_rows, output_stride, output_cols, fft_len, 0, tile_rows);
        CUDA_CHECK(cudaGetLastError());

        // Signal that d_out_chunk[buf] is ready for D2H.
        CUDA_CHECK(cudaEventRecord(compute_done[buf], compute_stream));

        // Kick off async D2H on the copy stream — runs concurrently with next tile's compute.
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, compute_done[buf]));
        CUDA_CHECK(cudaMemcpyAsync(h_pinned + (size_t)u_base * output_stride,
                                   d_out_chunk[buf],
                                   (size_t)tile_rows * output_stride * sizeof(cuFloatComplex),
                                   cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaEventRecord(d2h_done[buf], copy_stream));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    std::vector<cuFloatComplex> h_output(h_pinned, h_pinned + (size_t)rows * output_stride);

    for (int i = 0; i < 2; i++) {
        cudaEventDestroy(compute_done[i]);
        cudaEventDestroy(d2h_done[i]);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    if (plan_cufft)      cufftDestroy(plan_cufft);
    if (plan_cufft_last) cufftDestroy(plan_cufft_last);
    cudaFreeHost(h_pinned);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_work);  // nullptr-safe when fft_backend==3
    cudaFree(d_out_chunk[0]);
    cudaFree(d_out_chunk[1]);

    const size_t input_bytes = (size_t)(n_sp + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)coo.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)coo.nnz + (size_t)n_sp) * sizeof(int));
    const size_t work_bytes = (fft_backend != 3)
                            ? (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex) : 0;
    return {nullptr, output_stride, ms, input_bytes
                     + (size_t)cols    * sizeof(cuFloatComplex)   // d_chirp
                     + (size_t)fft_len * sizeof(cuFloatComplex)   // d_b_fft
                     + (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)  // d_signal
                     + work_bytes                                                 // d_work (0 for backend 3)
                     + cufft_workspace                                            // cuFFT internal scratch
                     + 2 * (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex), // d_out_chunk[2]
            std::move(h_output)};
}

SparseFFTResult sparse_fft_csc_bluestein_cufft_streaming(const COOMatrix& coo, int u_tile) {
    return sparse_fft_csc_stockham_streaming(coo, u_tile, 3);
}

SparseFFTResult sparse_fft_csc_stockham_graph(const COOMatrix& coo, int u_tile) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len = next_power_of_two(2 * cols - 1);
    const bool use_packed = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> row_idx_packed = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, row_idx_packed.data(),
                              coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp,  cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft,  fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(), cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr, *d_work = nullptr, *d_out_chunk = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal,    (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_work,      (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk, (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));

    // Stage h_b through d_signal for the one-time d_b_fft precompute (eliminates d_b).
    CUDA_CHECK(cudaMemcpy(d_signal, h_b.data(), fft_len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    run_fft_power2_stockham(d_signal, d_b_fft, fft_len, 1, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    cuFloatComplex* h_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pinned,
                             (size_t)rows * output_stride * sizeof(cuFloatComplex),
                             cudaHostAllocDefault));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);

        CUDA_CHECK(cudaMemsetAsync(d_signal, 0, signal_bytes, stream));

        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_bluestein_input_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx, d_sp_cols, d_chirp, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        cuFloatComplex* fwd_result = run_fft_power2_stockham_stream(
            d_signal, d_work, fft_len, tile_rows, false, stream);
        cuFloatComplex* fwd_other = (fwd_result == d_signal) ? d_work : d_signal;
        cuFloatComplex* inv_result = run_fft_power2_stockham_mul_inverse_stream(
            fwd_result, fwd_other, d_b_fft, fft_len, tile_rows, stream);

        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block, 0, stream>>>(
            inv_result, d_chirp, d_out_chunk,
            tile_rows, output_stride, output_cols, fft_len, 0, tile_rows);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_pinned + (size_t)u_base * output_stride,
                                   d_out_chunk,
                                   (size_t)tile_rows * output_stride * sizeof(cuFloatComplex),
                                   cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    std::vector<cuFloatComplex> h_output(h_pinned, h_pinned + (size_t)rows * output_stride);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_pinned);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_work);
    cudaFree(d_out_chunk);

    const size_t input_bytes = (size_t)(n_sp + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)coo.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)coo.nnz + (size_t)n_sp) * sizeof(int));
    return {nullptr, output_stride, ms, input_bytes
                     + (size_t)cols    * sizeof(cuFloatComplex)   // d_chirp
                     + (size_t)fft_len * sizeof(cuFloatComplex)   // d_b_fft (d_b eliminated)
                     + 2 * (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)   // d_signal + d_work
                     + (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex), // d_out_chunk
            std::move(h_output)};
}

SparseFFTResult sparse_fft_csr_2pass(const CSRMatrix& csr) {
    const int rows = csr.rows;
    const int cols = csr.cols;
    const int output_cols = cols / 2 + 1;
    const bool use_packed = can_pack_u16(rows, cols);

    // Distinct column indices — same logic as COO pass
    std::vector<int> sp_cols = make_sparse_cols(csr.col_idx.data(), csr.nnz);
    std::vector<int> col_ids = make_sparse_col_ids(csr.col_idx.data(), csr.nnz, sp_cols, cols);
    const int n_sp = (int)sp_cols.size();

    int *d_row_ptr, *d_col_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_col_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr,  csr.row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (use_packed) {
        std::vector<uint16_t> col_idx_packed = pack_cols_u16(col_ids.data(), csr.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(sp_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_col_idx_packed, csr.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp    * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_col_idx_packed, col_idx_packed.data(), csr.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(), n_sp    * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_col_idx,  csr.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols,  n_sp    * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_col_idx,  col_ids.data(),      csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols,  sp_cols.data(),       n_sp    * sizeof(int), cudaMemcpyHostToDevice));
    }

    cuFloatComplex *d_G, *d_out;
    CUDA_CHECK(cudaMalloc(&d_G,   (size_t)rows * n_sp * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)rows * output_cols * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemset(d_G, 0, (size_t)rows * n_sp * sizeof(cuFloatComplex)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    // Pass 1: CSR scatter into G
    dim3 p1_block(CSR_BU, CSR_BR);
    dim3 p1_grid((rows + CSR_BR - 1) / CSR_BR,
                 (rows + CSR_BU - 1) / CSR_BU);
    if (use_packed) {
        csr_build_intermediate_packed<<<p1_grid, p1_block>>>(
            d_row_ptr, d_col_idx_packed, d_G, rows, cols, 1.f / (float)rows);
    } else {
        csr_build_intermediate<<<p1_grid, p1_block>>>(
            d_row_ptr, d_col_idx, d_G, rows, cols, 1.f / (float)rows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: custom column DFT over sparse columns of G (same kernel as COO)
    dim3 p2_block(P2_BV, P2_BU);
    dim3 p2_grid((output_cols + P2_BV - 1) / P2_BV,
                 (rows + P2_BU - 1) / P2_BU);
    if (use_packed) {
        col_dft_kernel_packed<<<p2_grid, p2_block>>>(
            d_G, d_sp_cols_packed, n_sp, d_out, rows, cols, output_cols, 1.f / (float)cols);
    } else {
        col_dft_kernel<<<p2_grid, p2_block>>>(
            d_G, d_sp_cols, n_sp, d_out, rows, cols, output_cols, 1.f / (float)cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_col_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_G);

    const size_t input_bytes = (size_t)(rows + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)csr.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)csr.nnz + (size_t)n_sp) * sizeof(int));
    return {d_out, output_cols, ms, input_bytes
                     + (size_t)rows * n_sp * sizeof(cuFloatComplex)
                     + (size_t)rows * output_cols * sizeof(cuFloatComplex)};
}
