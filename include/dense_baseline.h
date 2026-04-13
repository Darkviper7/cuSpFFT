#pragma once
#include "mtx_reader.h"
#include <cuComplex.h>

// Result of a dense cuFFT 2-D FFT run.
struct DenseFFTResult {
    cuFloatComplex* d_output;   // device pointer, size padded_rows*(padded_cols/2+1), caller must free
    float           ms;         // kernel time in milliseconds
    size_t          mem_bytes;  // peak device memory allocated (bytes)
    int             padded_rows = 0; // 0 = same as input (dense_fft); set by dense_fft_padded
    int             padded_cols = 0;
};

// Run a dense 2-D FFT via cuFFT on the binarized matrix.
// Internally converts COO -> dense float array on device, then calls cuFFT R2C.
// Returns DenseFFTResult; caller is responsible for freeing d_output.
// Throws std::runtime_error if device memory is insufficient or cuFFT plan fails.
DenseFFTResult dense_fft(const COOMatrix& coo);

// Padded in-place variant: rounds rows and cols up to the nearest 7-smooth number
// so that cufftPlan2d succeeds for matrices with large prime factors.
// Uses a single in-place buffer (real input + complex output share storage),
// halving peak memory vs out-of-place (~11 GB instead of ~22 GB for 52k nodes).
// The output is an R2C half-spectrum of the zero-padded matrix — it is NOT the
// same transform as dense_fft and cannot be used as a correctness reference for
// sparse methods (different frequency grid).
// padded_rows and padded_cols in the result record the actual transform size.
DenseFFTResult dense_fft_padded(const COOMatrix& coo);
