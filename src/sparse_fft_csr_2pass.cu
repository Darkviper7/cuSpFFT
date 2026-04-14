#include "sparse_fft_internal.cuh"

// ---------------------------------------------------------------------------
// CSR pass-1 build_intermediate kernels  (sparse_fft_csr_2pass)
// ---------------------------------------------------------------------------
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
// Pass-2 column DFT  (shared with COO, static linkage for this TU)
// ---------------------------------------------------------------------------
static __global__ void col_dft_kernel(
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

static __global__ void col_dft_kernel_packed(
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

// ---------------------------------------------------------------------------
// Public API: sparse_fft_csr_2pass
// ---------------------------------------------------------------------------
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

