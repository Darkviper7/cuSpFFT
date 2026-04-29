#pragma once
#include "mtx_reader.h"
#include <cuComplex.h>

// Result of a dense cuFFT 2-D FFT run.
struct DenseFFTResult {
    cuFloatComplex* d_output;   // device pointer, size rows*(cols/2+1), caller must free
    float           ms;         // kernel time in milliseconds
    size_t          mem_bytes;  // peak device memory allocated (bytes)
};

// Run a dense 2-D FFT via cuFFT on the binarized matrix.
// Internally converts COO -> dense float array on device, then calls cuFFT R2C.
// Returns DenseFFTResult; caller is responsible for freeing d_output.
// Throws std::runtime_error if device memory is insufficient or cuFFT plan fails.
DenseFFTResult dense_fft(const COOMatrix& coo);
