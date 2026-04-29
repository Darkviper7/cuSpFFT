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

    // Capture baseline free device memory; peak = baseline - min_free_during_run,
    // measured right before cleanup.
    size_t base_free = 0, _tot = 0;
    cudaMemGetInfo(&base_free, &_tot);
    size_t peak = 0;

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

        // Sample free memory at peak (cuFFT workspace lazily allocated by cufftExecR2C
        // is included; everything is still live).
        size_t peak_free = 0;
        cudaMemGetInfo(&peak_free, &_tot);
        peak = (peak_free < base_free) ? base_free - peak_free : 0;

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
