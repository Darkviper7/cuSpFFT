# cuSpFFT

Exact 2-D FFT of binary sparse matrices on GPU. Input is a Matrix Market
`.mtx` file; all nonzero values are binarized to 1 on read, so the code
exploits only the sparsity pattern. The output is the same R2C
half-spectrum (`rows x (cols/2 + 1)` complex) that dense cuFFT produces,
computed without ever materializing the dense grid.

## Requirements

- NVIDIA GPU, CUDA Toolkit 12.x (cuFFT included), CMake 3.20+, C++17 compiler.
- Optional: FFTW3f for the CPU ground-truth check (`-DENABLE_CPU_REFERENCE=ON`),
  SpFFT for the external GPU baseline (`-DENABLE_SPFFT=ON`).

## Build

On this machine the system `nvcc` is broken — always point CMake at the
CUDA 12.5 install explicitly. Use `Release` for benchmarking.

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc

cmake --build build -j$(nproc)
```

On a machine with a working CUDA install, the `-DCMAKE_CUDA_COMPILER` line
can be dropped.

## The main method: `--csc-cufft-smooth-sym`

This is the project's default/best variant
(`sparse_fft_csc_bluestein_cufft_smooth_sym` in `src/sparse_fft_bluestein.cu`).
It walks the binary input in compact active-column CSC form and computes
only the first `rows/2 + 1` row-frequencies; the input is real, so the
remaining rows are filled in by conjugate symmetry. The column transform
is a single batched cuFFT of length `cols` when `cols` is 7-smooth, and
otherwise a Bluestein convolution at a 7-smooth FFT length; either way the
output rows are processed in tiles through a double-buffered two-stream
pipeline so one tile's build overlaps the previous tile's FFT.

## Getting data

```bash
./scripts/download_matrices.sh   # sstmodel, benzene, pct20stif from SuiteSparse
```

`dataset_lists/sampled.txt` lists the 100-matrix sweep set (paths under
`dataset_sampled/`).

## Run

```bash
./build/cuSpFFT dataset/benzene/benzene.mtx --csc-cufft-smooth-sym --repeat 10
```

The dense cuFFT baseline runs by default and the sparse result is checked
against it automatically. `--repeat 10` does 1 warmup + 10 timed runs.
Useful extras:

- `--cpu-reference` — FFTW3 ground truth instead of dense cuFFT as the
  reference (needs the `-DENABLE_CPU_REFERENCE=ON` build).
- `--sparse-only` — skip the dense baseline (needed when dense would OOM).
- `--csc-tile N` — output-row tile size (CLI default 1024; benchmark runs
  use `--csc-tile 128`).
- `./build/cuSpFFT --help` — full flag list.

### Reading the output

Each method prints one line:

```
Sparse CSC (cufft smooth sym)  9.123  317.00  [min 9.1, max 9.3]  n=10  prep 12.4
```

That is: median kernel time (ms), peak device memory (MB), min/max over
the timed runs, and `prep` = host preprocessing time (ms, CSC build etc.,
not included in the kernel time). The line below it,
`check ... max_abs ... max_rel ... rms_abs ...`, is the correctness check
against the reference; `max_rel` is the worst relative error.

## Sweeps

```bash
MATRIX_LIST=dataset_lists/sampled.txt LOG_DIR=logs_sampled \
    FLAGS="--csc-cufft-smooth-sym --repeat 10" TIMEOUT=1800 \
    ./scripts/run_sweep.sh
```

`run_sweep.sh` runs the binary on every listed matrix (env knobs:
`MATRIX_LIST`, `LOG_DIR`, `FLAGS`, `TIMEOUT`, `BIN`) and appends one row
per (matrix, method) to `<LOG_DIR>/results.csv`. Pivot that into a
per-matrix comparison sheet with:

```bash
python3 scripts/make_sheet.py logs_sampled/results.csv results_sampled/sheet
```

## Layout

| Path | Purpose |
|---|---|
| `src/main.cu` | CLI driver: flag parsing, benchmark loop, correctness check |
| `src/sparse_fft_bluestein.cu` | All sparse variants, including the sym method |
| `src/dense_baseline.cu` | Dense cuFFT baseline (scatter + `cufftExecR2C`) |
| `src/mtx_reader.cu` | Matrix Market parser (binarizes values) |
| `src/cpu_reference.cpp` | Optional FFTW3 CPU reference |
| `include/` | Public APIs and shared device helpers |
| `scripts/` | Data download, sweeps, sheets, plots, profiling |
| `dataset/`, `dataset_sampled/` | `.mtx` inputs |

Many other `--csc-*` flags exist (`--csc-mixed-radix`, `--csc-stockham-*`,
`--csc-binary-lut`, ...). They are experimental ablations kept for
comparison; `--csc-cufft-smooth-sym` is the method that matters.
