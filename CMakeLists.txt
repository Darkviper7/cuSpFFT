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
    src/sparse_fft.cu
    src/dense_baseline.cu
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
