#include "cpu_reference.h"
#include <fftw3.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

CPUReferenceResult cpu_fft_r2c_2d(const COOMatrix& coo) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int freq_cols = cols / 2 + 1;
    const size_t in_size  = (size_t)rows * (size_t)cols;
    const size_t out_size = (size_t)rows * (size_t)freq_cols;

    // FFTW-aligned buffers for SIMD performance.
    float*          in  = (float*)         fftwf_malloc(in_size  * sizeof(float));
    fftwf_complex*  out = (fftwf_complex*) fftwf_malloc(out_size * sizeof(fftwf_complex));
    if (!in || !out) {
        if (in)  fftwf_free(in);
        if (out) fftwf_free(out);
        throw std::runtime_error("CPU reference: fftwf_malloc failed (need ~"
            + std::to_string((in_size * sizeof(float)
                              + out_size * sizeof(fftwf_complex)) / (1024 * 1024))
            + " MB)");
    }

    // Zero the dense input then scatter binarized COO entries.
    std::memset(in, 0, in_size * sizeof(float));
    for (int i = 0; i < coo.nnz; i++) {
        const int r = coo.row_idx[i];
        const int c = coo.col_idx[i];
        in[(size_t)r * (size_t)cols + (size_t)c] = 1.0f;
    }

    // Plan + execute. FFTW_ESTIMATE keeps planning cheap; FFTW_MEASURE would
    // tune kernels but takes seconds and isn't worth it for a one-shot reference.
    fftwf_plan plan = fftwf_plan_dft_r2c_2d(rows, cols, in, out, FFTW_ESTIMATE);
    if (!plan) {
        fftwf_free(in);
        fftwf_free(out);
        throw std::runtime_error("CPU reference: fftwf_plan_dft_r2c_2d returned null");
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    fftwf_execute(plan);
    auto t1 = std::chrono::high_resolution_clock::now();

    fftwf_destroy_plan(plan);

    // fftwf_complex is binary-compatible with cuFloatComplex (both are
    // {float real, float imag}).  std::memcpy is safe.
    CPUReferenceResult result;
    result.output.resize(out_size);
    std::memcpy(result.output.data(), out, out_size * sizeof(fftwf_complex));

    fftwf_free(in);
    fftwf_free(out);

    result.rows      = rows;
    result.freq_cols = freq_cols;
    result.ms        = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.mem_bytes = in_size  * sizeof(float)
                     + out_size * sizeof(fftwf_complex);
    return result;
}
