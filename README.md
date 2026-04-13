# cuSpFFT — Efficient GPU FFT Leveraging Binary Sparsity

A CUDA implementation of 2-D FFT optimized for binary sparse matrices, targeting graph adjacency matrices and similar inputs where dense FFT is too slow or runs out of memory.

## Motivation

Standard GPU FFT libraries (cuFFT) treat all inputs as dense. For large sparse matrices — such as graph adjacency matrices with millions of nodes — this means:
- Allocating a full `rows × cols` float array, often causing OOM.
- Performing useless arithmetic on zeros.

cuSpFFT exploits two properties of the input simultaneously:
1. **Sparsity** — only nonzero entries contribute to the DFT sum.
2. **Binarization** — all values are 0 or 1, enabling bit-packing and simplified arithmetic.

## Project Structure

```
cuSpFFT/
├── CMakeLists.txt
├── include/
│   ├── mtx_reader.h        # COO/CSR matrix structs and parser interface
│   ├── dense_baseline.h    # Dense cuFFT baseline interface
│   └── sparse_fft.h        # Sparse FFT interface (all variants)
├── src/
│   ├── main.cu             # CLI entry point, benchmarking, correctness checking
│   ├── mtx_reader.cu       # Matrix Market .mtx parser (binarizes values)
│   ├── dense_baseline.cu   # COO → dense scatter + cuFFT R2C baseline
│   └── sparse_fft.cu       # All sparse FFT kernels and host functions (~3200 lines)
├── dataset/                # Place .mtx files here
└── scripts/
    ├── download_matrices.sh  # Downloads all benchmark matrices
    └── profile.sh            # Nsight Compute (ncu) profiling script
```

## Requirements

- CUDA 12.x (`/usr/local/cuda-12.5/bin/nvcc`)
- CMake >= 3.20
- cuFFT, cuBLAS, cuSPARSE, NVTX3 (included with CUDA Toolkit)

## Build

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc

cmake --build build -j$(nproc)
```

Use `Debug` for development builds with assertions enabled.

## Usage

```bash
# Standard benchmark run (benzene)
./build/cuSpFFT dataset/benzene/benzene.mtx \
    --csc-stockham-smem \
    --csc-stockham-cufft \
    --csc-stockham-cufft-stream \
    --csc-tile 128

# Include padded dense cuFFT comparison with correctness check
./build/cuSpFFT dataset/benzene/benzene.mtx \
    --csc-stockham-smem \
    --csc-stockham-cufft \
    --csc-stockham-cufft-stream \
    --csc-tile 128 \
    --dense-padded

# Run streaming cuFFT on 50k+ node graph (~329 MB device, dense OOM expected)
./build/cuSpFFT dataset/pct20stif/pct20stif.mtx \
    --sparse-only \
    --csc-stockham-cufft-stream \
    --csc-tile 128
```

### CLI Options

| Flag | Effect |
|---|---|
| `--sparse-only` | Skip dense cuFFT baseline |
| `--dense-only` | Skip all sparse FFT methods |
| `--csc-stockham-smem` | Bluestein+Stockham with smem row FFT (auto-fallback for fft_len > 4096) |
| `--csc-stockham-cufft` | Bluestein with cuFFT C2C batched FFT (~2× faster than custom Stockham) |
| `--csc-stockham-cufft-stream` | Streaming Bluestein+cuFFT — lowest device footprint, best for large matrices |
| `--dense-padded` | Padded dense cuFFT (next 7-smooth size, in-place R2C) with correctness check vs exact baseline |
| `--csc-tile N` | Output-row tile size for streaming CSC (default: 1024) |

## Benchmark Matrices

Download all three with:
```bash
./scripts/download_matrices.sh
```

| Matrix | Nodes | Use |
|---|---|---|
| HB/sstmodel | ~1–5k | Development and correctness checks |
| PARSEC/benzene | ~8–10k | Benchmark against dense cuFFT |
| Boeing/pct20stif | 52329 | Large-scale: dense OOM expected; `--csc-stockham-cufft-stream` runs in ~917 ms / 329 MB |

Source: [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)

## Approach

### Problem Statement

For a binary sparse matrix A with nonzero set `{(r, c)}`, the 2-D DFT is:

```
F[u, v] = sum_{(r, c) in nnz} exp(-2πi · (r·u/rows + c·v/cols))
```

All methods exploit the fact that the sum has only `nnz` terms instead of `rows × cols`.

### Dense Baseline

Converts COO to a full float matrix on the GPU, then calls `cufftExecR2C` for a standard 2-D real-to-complex FFT. Output is `rows × (cols/2 + 1)` (R2C half-spectrum). Serves as the correctness reference for all sparse methods.

The scatter kernel uses 64-bit index arithmetic (`(size_t)row * cols + col`) to avoid int32 overflow for matrices with more than ~46k rows. A cuFFT workspace size check (`cufftGetSize` + `cudaMemGetInfo`) runs before execution so that workspace OOM throws a clean exception rather than triggering a sticky hardware memory fault.

### Dense Baseline (Padded, `--dense-padded`)

When `cufftPlan2d` fails because the matrix dimensions contain large prime factors, `dense_fft_padded` rounds rows and cols up to the nearest 7-smooth number (factors only from {2, 3, 5, 7}) and runs in-place R2C. In-place mode uses a single buffer (`2 × (pcols/2+1)` floats per row) shared by the real input and complex output, halving static memory vs out-of-place (~11 GB instead of ~22 GB for 52k nodes).

**The padded output is NOT equivalent to the original transform** — frequency bins shift from `k/N` to `k/N'`, so it cannot serve as a correctness reference for sparse methods. A correctness check is printed against the exact dense baseline anyway to quantify the difference; expect large errors (e.g. `max_abs ~2e5` for benzene) since the two outputs are at different frequencies.

### Sparse FFT: Tiled Direct DFT (COO) *(disabled)*

One thread per output bin `(u, v)` loops over all `nnz` entries. COO arrays are tiled through shared memory to reduce global-memory traffic. Full complex output (`output_cols = cols`). Currently commented out in the benchmark CLI.

- **Complexity:** O(nnz × rows × cols)
- **Memory:** O(nnz + rows × cols)

### Two-Pass Decomposition

All other sparse methods use a two-pass factorization that separates the row and column exponentials:

```
Pass 1:  G[col_id][u] = sum_{r : (r,c) in nnz}  exp(-2πi · r·u/rows)
Pass 2:  F[u, v]      = sum_i  G[i][u] · exp(-2πi · sparse_cols[i]·v/cols)
```

G is compact: only `n_sparse_cols` columns (those with at least one nonzero) are stored, stored **column-major** (`G[col_id * rows + u]`) so pass-1 writes are coalesced.

Output is the R2C half-spectrum: `output_cols = cols/2 + 1`.

### Sparse FFT: Two-Pass COO *(disabled)*

Pass 1 uses atomic adds to scatter each nonzero `(r, c)` into `G[col_id][u]` for all output rows `u`. Pass 2 uses a custom `col_dft_kernel` that tiles sparse column indices through shared memory. Currently commented out in the benchmark CLI.

- **Complexity:** O(nnz × rows + rows × output_cols × n_sparse_cols)
- **Memory:** O(nnz + rows × n_sparse_cols + rows × output_cols)

### Sparse FFT: Two-Pass COO + cuFFT Pass 2 *(disabled)*

Same pass 1 as COO, but uses a full-sized dense G (`cols` columns) so cuFFT can run a batched C2C transform for pass 2. Currently commented out in the benchmark CLI.

### Sparse FFT: Two-Pass COO + GEMM Pass 2 *(disabled)*

Same compact COO pass 1. Pass 2 materializes the twiddle matrix `W[i, v] = exp(-2πi·sparse_cols[i]·v/cols)` and calls `cublasCgemm` for `F = G^T × W`. Currently commented out in the benchmark CLI.

- **Memory:** O(nnz + rows × n_sparse_cols + n_sparse_cols × output_cols + rows × output_cols)

### Sparse FFT: Two-Pass CSC

Builds a compact CSC representation from the COO input. Pass 1 assigns one thread block per active column, eliminating all atomics. Pass 2 is the same custom `col_dft_kernel`.

All `sparse_fft_csc_2pass` benchmark blocks are currently commented out, including cached-W and tiled-p2 sub-variants.

- **Complexity:** O(nnz × rows + rows × output_cols × n_sparse_cols)
- **Memory:** O(nnz + col_ptr + rows × n_sparse_cols + rows × output_cols)

#### Tiled Pass-2 (`--csc-tiled-p2`)

`col_dft_chunk_tiled_kernel` loads `CSC_TILED_C × (CSC_TILED_U × RBU)` G tiles and
`CSC_TILED_C × (CSC_TILED_V × RBV)` W tiles into shared memory, then accumulates into
`RBU × RBV = 2 × 4` register arrays per thread. Key tuning decisions:

- `CSC_TILED_C = 32` — halved from 64 to fit 16 KB smem and lift occupancy on V100.
- `RBV = 4` — each v-block covers 4× more output columns, reducing G_chunk DRAM reads by ~4×.
- Stride-coalesced v-layout ensures all output stores remain coalesced regardless of RBV.
- `fmaf()` for all complex multiply-accumulates in the inner loop.
- Output stride padded to next multiple of 4 complex values for L2 sector alignment.

### Sparse FFT: CSC + Bluestein Column FFT (`--csc-bluestein`, `--csc-stockham`, `--csc-stockham-smem`, `--csc-stockham-cufft`)

Instead of a direct DFT for pass 2, `sparse_fft_csc_bluestein` computes the column transform
via the Bluestein chirp-z convolution, which reduces arbitrary-length DFT to a power-of-2 FFT:

```
1. signal[u][c] = G[c][u] · chirp[c]       (chirp modulation)
2. A = FFT(signal)                           (forward power-of-2 FFT, length fft_len)
3. C = A ⊙ B_fft                            (pointwise multiply with precomputed B = FFT(b))
4. c = IFFT(C)                               (inverse FFT)
5. F[u][v] = c[u][v] · chirp_conj[v]        (finalize)
```

where `fft_len = next_pow2(2·cols - 1)` and `chirp[k] = exp(-iπk²/cols)`.

Three FFT backends are available via the `fft_backend` parameter:

- **`fft_backend=0` (default):** Out-of-place Stockham FFT with per-stage kernel launches.
  Step 3 is fused into the first IFFT stage via `stockham_stage_mul_kernel` to save one pass.
  Ping-pong buffers; stream helpers return the result pointer (no D2D copy).
  *(benchmark block currently disabled)*

- **`fft_backend=1` (`--csc-stockham-smem`):** Shared-memory staged Stockham — entire row held
  in smem across all butterfly stages, eliminating O(log₂ N) round-trips to DRAM.
  Automatically falls back to `fft_backend=0` when `fft_len > 4096` (smem limit ≈ 96 KB V100).

- **`fft_backend=3` (`--csc-stockham-cufft`):** cuFFT C2C batched — replaces the custom
  Stockham with `cufftExecC2C` (forward + inverse). cuFFT's internal smem-staged butterflies
  collapse log₂(N) global passes to O(1), giving ~2× speedup over custom Stockham on
  typical graph matrices (benzene: ~28.5 ms → ~13.4 ms; sstmodel: ~2.6 ms → ~1.0 ms).

Stockham kernel optimizations (backends 0 and 1):
- `fmaf()` in all butterfly outputs and chirp products.
- Branchless `stockham_two_stage_kernel`: fused radix-4 with predicated selects instead of divergent warp branches.
- 1-D build kernel layout for chirp-modulated signal construction: `blockIdx.x = col_id`, `threadIdx.x = u_local`. All threads in a block share the same column, so `row_idx[p]` reads hit the L1 broadcast path rather than scattering across 32 different offsets (the 2-D coalesced layout had 91% excessive sector traffic on this access).
- Stream functions return the result buffer pointer — no device-to-device copy when stage count is odd.

- **Output:** R2C half-spectrum, `output_cols = cols/2 + 1`.
- **Memory:** O(nnz + rows × fft_len × 2 + rows × output_cols) per tile.

### Sparse FFT: Streaming and Graph Variants (`--csc-stockham-stream`, `--csc-stockham-graph`)

- **Streaming Stockham (`sparse_fft_csc_stockham_streaming`):** double-buffered with two CUDA streams, overlapping FFT on one tile with D2H transfer of the previous. Output to host-pinned memory; `d_output` is `nullptr`. Peak device memory: `O(tile × fft_len)`.

- **Streaming cuFFT (`sparse_fft_csc_bluestein_cufft_streaming`, `--csc-stockham-cufft-stream`):** same streaming layout as above but uses cuFFT C2C for the FFT stages. Skips `d_work` (in-place cuFFT on `d_signal`) — saves `tile × fft_len × 8` bytes vs the Stockham streaming variant. Suitable for 50k+ node graphs where even the Bluestein non-streaming variant OOMs on `d_out`. `mem_bytes` includes the cuFFT internal workspace via `cufftGetSize`. On pct20stif (52329×52329): **~917 ms, 329 MB** on an RTX 4090 (24 GB).

- **Graph (`sparse_fft_csc_stockham_graph`):** captures the entire tile-loop body as a CUDA Graph during a dry run, then replays the instantiated graph. Eliminates per-tile kernel-launch overhead at the cost of fixed tile size for all iterations.

### Sparse FFT: Two-Pass CSR *(disabled)*

Converts COO to CSR. Pass 1 assigns `(output_row u, source_row r)` pairs to threads, computing `exp(-2πi·r·u/rows)` once per pair and distributing over all nonzero columns in that row. Reduces `sincosf` calls from `nnz × rows` (COO) to `rows²`. Currently commented out in the benchmark CLI.

- **Complexity:** O(rows² + nnz × rows + rows × output_cols × n_sparse_cols)
- **Memory:** O(nnz + row_ptr + rows × n_sparse_cols + rows × output_cols)

### 16-Bit Packing

For `rows ≤ 65536` and `cols ≤ 65536`, all kernels use 16-bit packed storage:
- COO: `uint32_t` packing `(row << 16 | col_id)`
- CSR/CSC column indices: `uint16_t`
- Sparse column list: `uint16_t`

This halves memory bandwidth for index arrays on typical graph matrices.

## Output and Correctness Checking

All sparse methods produce an R2C-style half-spectrum output (`SparseFFTResult.output_cols = cols/2+1`), except the COO+cuFFT variant which produces full complex output (`output_cols = cols`). For Bluestein and streaming variants, `output_cols` reflects the padded stride which may be slightly larger than `cols/2+1`.

When `--sparse-only` is not specified, the dense cuFFT output is copied to host and used as a correctness reference. For each sparse method, the benchmark prints:

```
  check <method>  max_abs <value>  max_rel <value>  rms_abs <value>
```

## Profiling

```bash
./scripts/profile.sh [matrix.mtx] [extra args...]
```

Runs an Nsight Compute (ncu) pass and writes a `.ncu-rep` report file to `profiles/`:
- `ncu_<name>_sparse.ncu-rep` — custom sparse kernels matched by name regex:
  `csc_build_bluestein_input_coalesced_packed_kernel`, `stockham_smem_row.*`,
  `pointwise_mul_batched_kernel`, `scale_complex_kernel`, `bluestein_finalize_kernel`

Extra arguments after the matrix path are forwarded to the binary (e.g. `--csc-stockham-smem --csc-tile 128`).

Open with `ncu-ui profiles/<file>.ncu-rep`. Requires sudo for hardware counters (or set `NVreg_RestrictProfilingToAdminUsers=0` once).

## Observations

Empirical findings from running benchmarks on the three matrix datasets (RTX 4090, 24 GB).

### pct20stif: dense cuFFT cannot run regardless of hardware

`cufftPlan2d` for pct20stif (52329×52329) fails with `CUFFT_INTERNAL_ERROR` (code 5) on every GPU. The root cause is arithmetic: **52329 = 3 × 17443**, and 17443 is prime. cuFFT cannot factor a size-17443 transform efficiently and fails to construct the plan — this is not an OOM, it is a hard algorithmic limitation independent of how much device memory is available.

### Padding to a smooth size does not help

The next 7-smooth number ≥ 52329 is **52488 = 2³ × 3⁸**. Even with in-place R2C (one ~11 GB buffer instead of separate 11 GB input + 11 GB output), `cufftPlan2d(52488, 52488)` succeeds but then **`cufftGetSize` reports a 10.5 GB internal workspace requirement**. After the 11 GB data buffer, only ~2.5 GB of the 24 GB card is free — execution fails before a single kernel launches.

The 10.5 GB workspace is cuFFT's row-column scratch buffer: the 2D algorithm transposes the full complex intermediate between the row-pass and column-pass, allocating a scratch array equal to the full output size.

| Dense cuFFT attempt | Static data | cuFFT workspace | Total needed | Fits in 24 GB? |
|---|---|---|---|---|
| Original size (52329²), out-of-place | ~22 GB | — (plan fails) | — | No (plan fails) |
| Padded size (52488²), out-of-place | ~22 GB | 10.5 GB | ~32.5 GB | No |
| Padded size (52488²), in-place | ~11 GB | 10.5 GB | ~21.5 GB | No |
| **Sparse CSC cufft stream** | **0.33 GB** | **none** | **0.33 GB** | **Yes** |

### Why sparse streaming succeeds where dense cannot

The streaming sparse method never materialises a dense grid. Its memory depends on `tile × fft_len`, not `rows × cols`:

- **Pass 1 (build G):** only touches `nnz = 2.7M` entries, not `rows × cols = 2.74B`.
- **Bluestein pass 2:** pads internally to `next_pow2(2 × 52329 - 1) = 131072` — a clean power of 2 that cuFFT C2C handles with a tiny workspace.
- **Streaming output:** only one tile of `d_signal` (`128 × 131072 × 8 = 134 MB`) lives on device at a time; completed tiles are copied to host-pinned memory asynchronously.
- **No row-column scratch:** cuFFT's 1D `cufftPlanMany` for batched 1D C2C transforms does not require the full 2D transpose scratch buffer.

### Memory accounting note

The reported 329 MB is **peak device memory only**. The full output (`rows × output_cols × 8 ≈ 10.9 GB`) accumulates in host-pinned memory via streaming D2H transfers. Host memory is more abundant and cheaper to allocate, but it is not free — a complete accounting includes both.

### Correctness scope

Correctness is verified against the dense cuFFT reference on sstmodel and benzene (matrices small enough for dense to run). For pct20stif, no reference exists; results are extrapolated from algorithm correctness on smaller matrices. The padded dense output cannot serve as a reference because it computes at different frequency bins.

### cuFFT's 2D workspace scales with transform size

For any `cufftPlan2d` of size `M × N`, the internal workspace is approximately `M × (N/2+1) × 8` bytes (the full complex half-spectrum). This is the cost of the transpose in the row-column algorithm. For 52488²: `52488 × 26245 × 8 ≈ 11 GB`. On any GPU with ≤ 22 GB free after loading the transform data, this will OOM.

## References

- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuSPARSE Storage Formats](https://docs.nvidia.com/cuda/cusparse/storage-formats.html)
- [SpFFT — open-source sparse FFT for CUDA](https://github.com/eth-cscs/SpFFT)
- [Matrix Market Format](https://math.nist.gov/MatrixMarket/formats.html)
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [Bluestein chirp-z transform](https://en.wikipedia.org/wiki/Chirp_Z-transform)
- [Stockham FFT](https://www.fftw.org/newsgroup-archives/na-digest/v95/na.95_05.01#v95_05_13)
