#pragma once
#include "mtx_reader.h"
#include <cuComplex.h>
#include <vector>

// CPU-only ground-truth reference using FFTW3 (single precision).
// Computes a 2-D R2C FFT on the binarized COO, returning the rows × (cols/2+1)
// host-side complex output and the wall-clock time of the FFTW execute call.
// Used to validate the CUDA dense cuFFT, SpFFT, and all sparse variants
// against an independent third-party CPU FFT — particularly important for
// large matrices where the GPU dense baseline cannot run.

struct CPUReferenceResult {
    std::vector<cuFloatComplex> output;  // rows × (cols/2+1) row-major
    int rows = 0;
    int freq_cols = 0;
    double ms = 0.0;          // FFTW execute wall-time (host clock, single-threaded)
    size_t mem_bytes = 0;     // peak host allocation for input + output
};

CPUReferenceResult cpu_fft_r2c_2d(const COOMatrix& coo);
