#include "sparse_fft_internal.cuh"

// ---------------------------------------------------------------------------
// CSC pass-1 build_intermediate kernels
// ---------------------------------------------------------------------------
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
// Pass-2 chunked/tiled column DFT kernels  (CSC 2-pass + streaming)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Twiddle matrix (row-major) for precompute_twiddle path  (sparse_fft_csc_2pass)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
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

