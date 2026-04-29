cmake_minimum_required(VERSION 3.20)
project(cuSpFFT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Detect GPU architecture (fallback to sm_80 Ampere if not detected)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

# Gather sources
set(SOURCES
    src/main.cu
    src/mtx_reader.cu
    src/dense_baseline.cu
    src/sparse_fft_bluestein.cu      # active benchmarks (Bluestein/Stockham/cuFFT)
)

add_executable(cuSpFFT ${SOURCES})

target_include_directories(cuSpFFT PRIVATE include)

target_link_libraries(cuSpFFT PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
    CUDA::cusparse
    CUDA::nvtx3
)

target_compile_options(cuSpFFT PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

# Enable assertions in debug builds
target_compile_definitions(cuSpFFT PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
)

# Optional SpFFT reference baseline
option(ENABLE_SPFFT "Enable SpFFT GPU baseline (requires libSpFFT with SPFFT_SINGLE_PRECISION=ON)" OFF)
if(ENABLE_SPFFT)
    find_package(SpFFT CONFIG REQUIRED)
    target_compile_definitions(cuSpFFT PRIVATE HAS_SPFFT)
    target_link_libraries(cuSpFFT PRIVATE SpFFT::spfft)
endif()

# Optional CPU reference baseline using FFTW3 single-precision.
# Provides an independent CPU ground-truth FFT for correctness-checking every
# CUDA variant (dense cuFFT, SpFFT, sparse paths).  Especially useful when the
# GPU dense baseline cannot run (e.g. pct20stif).
option(ENABLE_CPU_REFERENCE "Build CPU FFT reference using FFTW3 single-precision" OFF)
if(ENABLE_CPU_REFERENCE)
    find_path(FFTW3F_INCLUDE_DIR fftw3.h
        HINTS $ENV{HOME}/local/include /usr/local/include /usr/include)
    find_library(FFTW3F_LIBRARY fftw3f
        HINTS $ENV{HOME}/local/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
    if(NOT FFTW3F_INCLUDE_DIR OR NOT FFTW3F_LIBRARY)
        message(FATAL_ERROR
            "ENABLE_CPU_REFERENCE=ON but FFTW3 single-precision not found. "
            "Install fftw-3.x with --enable-float and pass -DCMAKE_PREFIX_PATH "
            "to point at its install prefix.")
    endif()
    target_sources(cuSpFFT PRIVATE src/cpu_reference.cpp)
    target_include_directories(cuSpFFT PRIVATE ${FFTW3F_INCLUDE_DIR})
    target_link_libraries(cuSpFFT PRIVATE ${FFTW3F_LIBRARY})
    target_compile_definitions(cuSpFFT PRIVATE HAS_CPU_REF)
endif()
