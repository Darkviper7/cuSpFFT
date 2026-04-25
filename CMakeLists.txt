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
    src/sparse_fft_direct_dft.cu     # direct tiled COO DFT
    src/sparse_fft_coo_2pass.cu      # COO 2-pass (custom DFT, cuFFT, GEMM)
    src/sparse_fft_csc_2pass.cu      # CSC 2-pass (tiled col DFT + streaming)
    src/sparse_fft_csr_2pass.cu      # CSR 2-pass
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
