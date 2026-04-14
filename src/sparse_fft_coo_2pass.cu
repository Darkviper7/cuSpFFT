#include "sparse_fft_internal.cuh"

// ---------------------------------------------------------------------------
// COO pass-1: build_intermediate  (sparse_fft_2pass)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Pass-2 column DFT  (sparse_fft_2pass / sparse_fft_csr_2pass)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// GEMM path: twiddle matrix (col-major) + transpose kernels  (sparse_fft_2pass_gemm)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
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

