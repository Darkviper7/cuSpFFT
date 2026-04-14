#include "sparse_fft_internal.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: tiled direct DFT  (sparse_fft)
// ---------------------------------------------------------------------------
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
// Public API: sparse_fft
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

