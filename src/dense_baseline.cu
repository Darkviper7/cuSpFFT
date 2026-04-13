#include "dense_baseline.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <cstring>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
            throw std::runtime_error(std::string("CUDA error: ") +          \
                                     cudaGetErrorString(err));               \
    } while (0)

#define CUFFT_CHECK(call)                                                    \
    do {                                                                     \
        cufftResult r = (call);                                              \
        if (r != CUFFT_SUCCESS)                                              \
            throw std::runtime_error("cuFFT error code: " +                 \
                                     std::to_string(r));                     \
    } while (0)

// Returns the smallest 7-smooth number (factors only from {2,3,5,7}) >= n.
// Used to find a cuFFT-friendly padded transform size for matrices whose
// dimensions contain large prime factors (e.g. 52329 = 3 × 17443).
static int next_smooth(int n) {
    for (int m = n; ; ++m) {
        int x = m;
        for (int p : {2, 3, 5, 7})
            while (x % p == 0) x /= p;
        if (x == 1) return m;
    }
}

// Kernel: scatter COO entries into a dense float array (value = 1.0f).
// Index must use 64-bit arithmetic: for large matrices (rows > ~46k) the product
// row * cols overflows int32, causing out-of-bounds writes and hardware errors.
__global__ void coo_to_dense(const int* row_idx, const int* col_idx,
                              float* dense, int cols, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz)
        dense[(size_t)row_idx[i] * cols + col_idx[i]] = 1.0f;
}

DenseFFTResult dense_fft(const COOMatrix& coo) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const size_t dense_floats = (size_t)rows * cols;
    // R2C output: rows × (cols/2+1) complex values (half-spectrum).
    const size_t output_complex = (size_t)rows * (cols / 2 + 1);

    // Estimate peak memory: dense input (float) + output (cuFloatComplex) + COO indices
    size_t peak = dense_floats * sizeof(float)
                + output_complex * sizeof(cuFloatComplex)
                + (size_t)coo.nnz * 2 * sizeof(int);

    // All pointers initialised to null so the cleanup block can safely free
    // whichever subset was successfully allocated before an exception.
    int *d_row = nullptr, *d_col = nullptr;
    float* d_dense = nullptr;
    cuFloatComplex* d_out = nullptr;
    cufftHandle plan = 0;
    cudaEvent_t t0 = nullptr, t1 = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_row, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row, coo.row_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col, coo.col_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_dense, dense_floats * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dense, 0, dense_floats * sizeof(float)));

        int threads = 256;
        int blocks = (coo.nnz + threads - 1) / threads;
        coo_to_dense<<<blocks, threads>>>(d_row, d_col, d_dense, cols, coo.nnz);
        CUDA_CHECK(cudaGetLastError());

        cudaFree(d_row);  d_row = nullptr;
        cudaFree(d_col);  d_col = nullptr;

        CUDA_CHECK(cudaMalloc(&d_out, output_complex * sizeof(cuFloatComplex)));

        CUFFT_CHECK(cufftPlan2d(&plan, rows, cols, CUFFT_R2C));

        // cuFFT allocates its workspace lazily at execution time, not during plan
        // creation.  If the workspace won't fit, cufftExecR2C will fail mid-kernel
        // leaving a sticky cudaErrorIllegalAddress on the context that
        // cudaGetLastError() cannot clear.  Query the required workspace size now
        // and throw a clean OOM before any kernel is launched.
        {
            size_t ws = 0, free_mem = 0, total_mem = 0;
            CUFFT_CHECK(cufftGetSize(plan, &ws));
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            if (ws > free_mem)
                throw std::runtime_error("CUDA error: out of memory"
                                         " (cuFFT workspace exceeds available device memory)");
        }

        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));

        CUFFT_CHECK(cufftExecR2C(plan, d_dense, d_out));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cufftDestroy(plan);
        cudaFree(d_dense);

        DenseFFTResult result;
        result.d_output  = d_out;
        result.ms        = ms;
        result.mem_bytes = peak;
        return result;

    } catch (...) {
        // Free every allocation that succeeded before the failure, then
        // clear any sticky CUDA error so the next benchmark starts clean.
        cudaFree(d_row);
        cudaFree(d_col);
        cudaFree(d_dense);
        cudaFree(d_out);
        if (plan) cufftDestroy(plan);
        if (t0) cudaEventDestroy(t0);
        if (t1) cudaEventDestroy(t1);
        cudaGetLastError();  // consume + clear sticky device error
        throw;
    }
}

DenseFFTResult dense_fft_padded(const COOMatrix& coo) {
    const int rows  = coo.rows;
    const int cols  = coo.cols;
    const int prows = next_smooth(rows);
    const int pcols = next_smooth(cols);

    // In-place R2C: a single buffer serves as both real input and complex output.
    // cuFFT requires each row to be padded to inembed = 2*(pcols/2+1) floats so
    // the complex output (pcols/2+1 values) fits in the same row allocation.
    // This halves peak memory vs out-of-place (one ~11 GB buffer instead of two).
    const int    inembed    = 2 * (pcols / 2 + 1);
    const size_t buf_floats = (size_t)prows * inembed;

    size_t peak = buf_floats * sizeof(float)
                + (size_t)coo.nnz * 2 * sizeof(int);

    int *d_row = nullptr, *d_col = nullptr;
    float* d_buf = nullptr;
    cufftHandle plan = 0;
    cudaEvent_t t0 = nullptr, t1 = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_row, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row, coo.row_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col, coo.col_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_buf, buf_floats * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_buf, 0, buf_floats * sizeof(float)));

        // Scatter with stride inembed (not pcols) so real values land in the
        // first pcols slots of each row; the trailing 2 padding floats remain 0.
        int threads = 256;
        int blocks  = (coo.nnz + threads - 1) / threads;
        coo_to_dense<<<blocks, threads>>>(d_row, d_col, d_buf, inembed, coo.nnz);
        CUDA_CHECK(cudaGetLastError());

        cudaFree(d_row);  d_row = nullptr;
        cudaFree(d_col);  d_col = nullptr;

        CUFFT_CHECK(cufftPlan2d(&plan, prows, pcols, CUFFT_R2C));

        {
            size_t ws = 0, free_mem = 0, total_mem = 0;
            CUFFT_CHECK(cufftGetSize(plan, &ws));
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            if (ws > free_mem)
                throw std::runtime_error(
                    "CUDA error: out of memory (cuFFT workspace needs " +
                    std::to_string(ws >> 20) + " MB, only " +
                    std::to_string(free_mem >> 20) + " MB free)");
        }

        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));

        // In-place: same pointer for real input and complex output.
        CUFFT_CHECK(cufftExecR2C(plan, (cufftReal*)d_buf, (cufftComplex*)d_buf));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cufftDestroy(plan);

        // d_buf now holds the complex R2C output; caller must cudaFree it.
        DenseFFTResult result;
        result.d_output    = (cuFloatComplex*)d_buf;
        result.ms          = ms;
        result.mem_bytes   = peak;
        result.padded_rows = prows;
        result.padded_cols = pcols;
        return result;

    } catch (...) {
        cudaFree(d_row);
        cudaFree(d_col);
        cudaFree(d_buf);
        if (plan) cufftDestroy(plan);
        if (t0) cudaEventDestroy(t0);
        if (t1) cudaEventDestroy(t1);
        cudaGetLastError();
        throw;
    }
}
