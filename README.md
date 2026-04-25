# cuSpFFT — Efficient GPU FFT Leveraging Binary Sparsity

A CUDA implementation of 2-D FFT optimized for **binary sparse matrices**,
targeting graph adjacency matrices and similar inputs where dense FFT is
either too slow or runs out of memory.

The codebase explores how far a custom GPU pipeline can go by exploiting
two structural properties of the input:
1. **Sparsity** — only the ~`nnz` nonzero entries contribute to the DFT sum;
   the implementation never materializes the dense `rows × cols` grid.
2. **Binarization** — every nonzero is exactly 1, so the values array is
   eliminated entirely and indices are bit-packed (`uint16_t` when
   `rows, cols ≤ 65k`).

## Repository layout

```
cuSpFFT/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── mtx_reader.h            # COO/CSR parser interface
│   ├── dense_baseline.h        # Dense cuFFT baseline interface
│   ├── cpu_reference.h         # CPU FFT (FFTW3) ground-truth interface
│   ├── sparse_fft.h            # All sparse FFT public APIs
│   └── sparse_fft_internal.cuh # Shared device helpers, constants, CSC builder
├── src/
│   ├── main.cu                 # CLI entry, benchmark loop, correctness check
│   ├── mtx_reader.cu           # Matrix Market .mtx parser (binarizes values)
│   ├── dense_baseline.cu       # Dense scatter + cufftExecR2C baseline
│   ├── cpu_reference.cpp       # CPU FFTW3 reference (built when ENABLE_CPU_REFERENCE=ON)
│   └── sparse_fft_bluestein.cu # All variants used in the headline benchmark
├── dataset/                    # Place .mtx files here
└── scripts/
    ├── download_matrices.sh    # Downloads the three benchmark matrices
    └── profile.sh              # Nsight Compute profiling helper
```

> Inside `src/main.cu`, benchmark blocks are grouped under HEADLINE /
> DIAGNOSTIC / EXPERIMENTAL banner comments; see the **Appendix** for the
> experimental flags and additional sparse paths preserved for ablation.

## Machine configuration used for the reported numbers

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (24 GB, compute capability 8.9) |
| Driver | 555.42.02 |
| CUDA Toolkit | 12.5.82 (`/usr/local/cuda-12.5`) |
| CMake | 3.20+ |
| OS | Linux x86_64 |

## Build

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc

cmake --build build -j$(nproc)
```

### Optional baselines

The two reference baselines (SpFFT and the CPU FFT) are independent CMake
options. Both are off by default; the binary still builds and runs without
them, the corresponding flags simply become no-ops.

**SpFFT (`-DENABLE_SPFFT=ON`)** — install SpFFT (CUDA backend, single
precision) somewhere accessible and pass `-DCMAKE_PREFIX_PATH=/path/to/spfft/install`.
Build SpFFT with:

```bash
cmake -S /path/to/spfft -B /tmp/spfft-build \
    -DSPFFT_GPU_BACKEND=CUDA -DSPFFT_SINGLE_PRECISION=ON \
    -DSPFFT_MPI=OFF -DSPFFT_OMP=OFF \
    -DCMAKE_INSTALL_PREFIX=$HOME/local \
    -DCMAKE_INSTALL_RPATH=$HOME/local/lib \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build /tmp/spfft-build --target install -j$(nproc)
```
SpFFT requires both `libfftw3` and `libfftw3f`; install both before
configuring SpFFT.

**CPU FFT reference (`-DENABLE_CPU_REFERENCE=ON`)** — uses FFTW3
single-precision (`libfftw3f`) to compute an independent CPU ground-truth
FFT. CMake searches `$HOME/local`, `/usr/local`, and `/usr` for
`fftw3.h` and `libfftw3f`. To build FFTW3 single-precision locally:

```bash
cd /tmp && curl -L -O https://www.fftw.org/fftw-3.3.10.tar.gz
tar xzf fftw-3.3.10.tar.gz && cd fftw-3.3.10
./configure --prefix=$HOME/local --enable-float --enable-shared
make -j$(nproc) && make install
```

Full build with both baselines enabled:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc \
    -DENABLE_SPFFT=ON \
    -DENABLE_CPU_REFERENCE=ON \
    -DCMAKE_PREFIX_PATH=$HOME/local
cmake --build build -j$(nproc)
```

## Benchmark matrices

```bash
./scripts/download_matrices.sh   # downloads all three from SuiteSparse
```

| Matrix | Source | Size | Role |
|---|---|---|---|
| `HB/sstmodel` | SuiteSparse | 3,345 × 3,345 (~22.7k nnz) | Dev / correctness checks (small) |
| `PARSEC/benzene` | SuiteSparse | 8,219 × 8,219 (~242k nnz) | Headline comparison vs dense cuFFT |
| `Boeing/pct20stif` | SuiteSparse | 52,329 × 52,329 (~2.7M nnz) | Large-scale OOM test (>50k nodes) |

All values are binarized to 1 on read — the algorithms exploit only the
sparse pattern, not the original numeric values.

---

# Variants implemented

The headline benchmark compares **four** variants — two baselines and two
of our optimized binary-sparse paths — plus an optional CPU FFT ground
truth used for independent correctness validation. Every other path in
the codebase is moved into a clearly-marked `EXPERIMENTAL / ABLATION`
section in `main.cu` and exists only for ablation studies; nothing has
been deleted.

## 0. CPU FFT reference (FFTW3 single-precision, optional)

**Source:** `src/cpu_reference.cpp`, function `cpu_fft_r2c_2d()`.
**Flag:** `--cpu-reference` (requires `-DENABLE_CPU_REFERENCE=ON` build).

Not a competitor — a *ground truth*. Provides independent third-party
validation of the entire CUDA pipeline by computing the same 2-D R2C FFT
on the host CPU using FFTW3 single-precision. When `--cpu-reference` is
set, this runs **first** and its output becomes the reference vector that
every subsequent variant — including dense cuFFT and SpFFT — is checked
against.

Pipeline:
1. Allocate FFTW-aligned buffers; scatter the binarized COO into the
   `rows × cols` real input.
2. `fftwf_plan_dft_r2c_2d(rows, cols, ..., FFTW_ESTIMATE)` → `fftwf_execute`.
3. Host-side complex output (`rows × (cols/2+1)`, binary-compatible with
   `cuFloatComplex`) becomes the reference.

## 1. Dense cuFFT (baseline)

**Source:** `src/dense_baseline.cu`, function `dense_fft()`.

The conventional GPU FFT route:
1. Scatter the COO entries into a dense `rows × cols` `float` array on device
   (`coo_to_dense` kernel; uses 64-bit index arithmetic to avoid `int32`
   overflow at 50k+ rows).
2. Run `cufftExecR2C` for an out-of-place real-to-complex 2-D FFT.
3. Output is `rows × (cols/2 + 1)` complex (R2C half-spectrum).

This is the **correctness reference** for the sparse paths. Its peak memory
is dominated by:
- Dense input: `rows × cols × 4` bytes.
- Complex output: `rows × (cols/2+1) × 8` bytes.
- cuFFT row-column transpose scratch: roughly equal to the output size.

For pct20stif (52,329²), `cufftPlan2d` fails with `CUFFT_INTERNAL_ERROR`
because `52329 = 3 × 17443` (17443 is prime — cuFFT cannot factor it).
Padding to the next 7-smooth size (52488²) makes the plan succeed but the
in-place workspace alone needs 10.5 GB on top of an 11 GB data buffer — over
21 GB total, OOM on a 24 GB card. Conclusion: dense cuFFT does not produce
a result for the >50k-node test, *regardless* of GPU memory.

## 2. SpFFT (baseline reference)

**Source:** inline in `src/main.cu` (under `#ifdef HAS_SPFFT`).
**External library:** [eth-cscs/SpFFT](https://github.com/eth-cscs/SpFFT)
(MIT-licensed, plane-wave DFT library by CSCS).

SpFFT is a CUDA-aware sparse-frequency FFT library used in plane-wave
electronic-structure codes. Its sparsity model is the *opposite* of ours:
SpFFT computes a 3-D FFT of dense spatial data evaluated at a specified
sparse subset of frequency points. We use it as a baseline by:
1. Scattering the COO entries into the SpFFT space-domain GPU buffer.
2. Configuring SpFFT with **all** R2C frequency triplets `(x, y, 0)` for
   `x ∈ [0, cols/2]`, `y ∈ [0, rows−1]`, `z = 0` — i.e. the full output.
3. Running `transform.forward(SPFFT_PU_GPU, ...)` and reading the result.

This is the closest external reference: a CUDA sparse FFT library, called
on the same matrices, that explicitly does **not** exploit the binary
nature of the input. SpFFT's overheads (3-D layout via `dimZ = 1`, sparse-
frequency indexing infrastructure, single-rank MPI scaffolding) prevent it
from beating cuFFT on this dense-output use case — see the Performance
section.

## 3. CSC mixed-radix, custom 1-D FFT (`--csc-mixed-radix`)

**Source:** `src/sparse_fft_bluestein.cu`, function
`sparse_fft_csc_bluestein_mixed_radix()`.
**This is the variant that demonstrates "binary-sparse helps on its own,
without calling cuFFT anywhere in the pipeline."**

### Pipeline

```
COO  ─▶ host-side CompactCSC (active columns, packed uint16 row indices)
        │
        ▼
        binary CSC pass-1 build  ───▶  signal[u_local, c]  (per u-tile)
        │
        ▼
        custom radix-{2,3,9} Stockham FFT  (length fft_len = next_3_smooth(2·cols−1))
        │
        ▼
        pointwise multiply by precomputed B = FFT(b)
        │
        ▼
        custom radix-{2,3,9} inverse Stockham FFT + 1/N scale
        │
        ▼
        bluestein_finalize_kernel (chirp_conj multiply)  ───▶  d_out
```

### Optimizations specific to the binary-sparse + custom-FFT story

#### CompactCSC representation (binary + sparsity)

The host builds a *compact* CSC where columns with no nonzeros are skipped:
- `sparse_cols`: the list of active column indices.
- `col_ptr`: prefix-sum offsets into `row_idx` (length `n_sparse_cols + 1`).
- `row_idx`: row indices of nonzeros.

For matrices with `rows, cols ≤ 65,536`, both `sparse_cols` and `row_idx`
are stored as `uint16_t` on device — halving the index bandwidth. There is
no `values` array because every nonzero is 1.

#### Pass-1 build kernel (`csc_build_bluestein_input_packed_kernel`)

For each output-row tile `[u_base, u_base + tile_rows)`, build the
chirp-modulated signal:

```
signal[u_local, c] = chirp[c] · Σ_{r ∈ rows(c)} exp(−2πi · r · u / rows)
```

Layout: `blockIdx.x = col_id`, `blockIdx.y` = u-block index,
`threadIdx.x = u_local within the block`. All threads in a block share the
same `col_id`, so `row_idx[p]` reads hit the L1 broadcast path instead of
scattering across 32 distinct `col_id` rows. (The earlier 2-D coalesced
layout had 91% wasted L2 sector traffic from the strided `row_idx` access
pattern; switching to this 1-D layout was a major speedup.)

The kernel iterates only the `nnz` row indices in each column — never
visits zero entries. No `values` are loaded because every nonzero is 1.

#### Bluestein with `next_3_smooth(2·cols−1)` FFT length

The Bluestein chirp-z transform converts an arbitrary-length-`cols` DFT
into a length-`fft_len` convolution, where `fft_len ≥ 2·cols − 1`. We pick
the **smallest 3-smooth size** ≥ `2·cols − 1` (factors only from {2, 3}).

- For `benzene` (`cols = 8219`): `2·cols−1 = 16437` → `fft_len = 17496 = 2³·3⁷`.
  The naive `next_pow2 = 32768` would do ~1.87× more FFT work for no gain.
- For `sstmodel` (`cols = 3345`): `fft_len = 6912 = 2⁸·3³` (vs 8192 pow2).
- For `pct20stif` (`cols = 52329`): `fft_len = 110592 = 2¹²·3³` (vs 131072).

#### Custom radix-{2, 3, 9} Stockham kernels

Three kernels in `sparse_fft_bluestein.cu` make up the FFT engine:
- `stockham_stage_kernel` — radix-2 Stockham butterfly, one stage per launch.
- `stockham_radix3_stage_kernel` — radix-3 Stockham butterfly with output
  twiddle.
- `stockham_radix9_stage_kernel` — **fused radix-9** = two radix-3 stages
  in registers via the 3×3 Kronecker decomposition. Replaces 2
  global-memory roundtrips with 1.

The host driver `run_fft_mixed_23` decomposes `fft_len` into a factor list
that prefers radix-9 over pairs of radix-3:

| `fft_len` | Factor list | # global passes |
|---|---|---|
| 6912 = 2⁸·3³ | `[9, 3, 2, 2, 2, 2, 2, 2, 2, 2]` | 10 |
| 17496 = 2³·3⁷ | `[9, 9, 9, 3, 2, 2, 2]` | 7 |
| 110592 = 2¹²·3³ | `[9, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]` | 12 |

Compared to a pure radix-2 Stockham at `next_pow2`, this is **roughly half**
the FFT work on benzene (smaller `fft_len` × fewer stages). Profiling with
ncu confirms each stage is memory-bandwidth-bound at ~85% of DRAM peak;
fusing two radix-3 stages into radix-9 directly shaves 2× on those
roundtrips, which is the dominant cost.

### Memory

- One tile of `signal[tile_rows, fft_len]` and one ping-pong work buffer of
  the same size — `2 × tile × fft_len × 8` bytes.
- Full output `d_out[rows, output_stride]` on device.
- Negligible: `chirp[cols]`, `B_fft[fft_len]`, packed CSC indices.

## 4. CSC mixed-radix, cuFFT 1-D FFT (`--csc-cufft-smooth`)

**Source:** `src/sparse_fft_bluestein.cu`, function
`sparse_fft_csc_bluestein_cufft_smooth()`.

Identical pipeline to variant 3 except the 1-D FFT engine is **cuFFT**
instead of our custom radix-{2,3,9} kernels:

```
... binary CSC pass-1 ─▶ signal[u_local, c]
                ▼
        cufftExecC2C(FORWARD)   (length fft_len = next_7_smooth(2·cols−1))
                ▼
        pointwise multiply by B_fft
                ▼
        cufftExecC2C(INVERSE) + 1/N scale
                ▼
        bluestein_finalize_kernel ─▶ d_out
```

Two parameter changes vs variant 3:
- `fft_len = next_7_smooth(2·cols − 1)` (factors from {2, 3, 5, 7}). cuFFT
  has hand-tuned mixed-radix kernels for any 7-smooth size, so we use a
  smaller fft_len than the 3-smooth one our custom kernels need.
  - benzene: 16464 = 2⁴·3·7³ (vs 17496 for 3-smooth, 32768 for pow2)
  - sstmodel: 6720 = 2⁶·3·5·7 (vs 6912, 8192)
  - pct20stif: 110592 (same as 3-smooth here — already smooth enough)
- `cufftPlanMany` + `cufftExecC2C` replace `run_fft_mixed_23`.

The pass-1 build, chirp/B precompute, pointwise multiply, scale, and
finalize kernels are *bit-identical reuses* of variant 3.

### Why this exists

It isolates exactly one question: **holding the binary-sparse pass-1 and
Bluestein scaffolding fixed, does cuFFT's mixed-radix beat our hand-rolled
mixed-radix?** The answer (see Performance section) is yes — by ~30% on
benzene — but the gap is closer than the comparison against power-of-2 cuFFT
would suggest, and it does not undo variant 3's central claim ("binary
sparsity beats dense cuFFT *without* calling cuFFT").

---

# Usage

```bash
# Headline benchmark on benzene with all four variants + CPU ground truth, 10-run medians
./build/cuSpFFT dataset/benzene/benzene.mtx \
    --cpu-reference \
    --spfft \
    --csc-mixed-radix \
    --csc-cufft-smooth \
    --repeat 10
```

Dense cuFFT runs by default. The other four are opt-in.

## CLI flags relevant to the headline variants

| Flag | Effect |
|---|---|
| (default) | Dense cuFFT baseline runs unless `--sparse-only` |
| `--cpu-reference` | Run CPU FFT (FFTW3) **first** and use it as the correctness reference for ALL variants (requires `-DENABLE_CPU_REFERENCE=ON` build) |
| `--spfft` | Run the SpFFT baseline (requires `-DENABLE_SPFFT=ON` build) |
| `--csc-mixed-radix` | Variant 3 — binary CSC + custom radix-{2,3,9} Stockham |
| `--csc-cufft-smooth` | Variant 4 — binary CSC + cuFFT at 7-smooth fft_len |
| `--csc-tile N` | Output-row tile size for the tiled CSC pipelines (default: 1024 for compatibility; tile=128 is what we use in the headline runs) |
| `--repeat N` | Run each benchmark `N + 1` times (1 warmup + N timed); report median plus min/max bracket. Default 1 (single-run, no warmup). |
| `--sparse-only` | Skip the dense cuFFT baseline |
| `--dense-only` | Skip the sparse variants |

Additional flags exist for ablation studies; see the **Appendix** for the
list. Run `./build/cuSpFFT --help` for the complete reference.

---

# Performance

All numbers are medians of 10 timed runs (`--repeat 10`) after one warmup.
Time is the kernel-loop wall time measured with CUDA events. Memory is the
device-side **runtime peak** measured via `cudaMemGetInfo` between baseline
capture and cleanup — it includes cuFFT/SpFFT internal workspace and graph
state, not just our explicit allocations.

Correctness (`max_abs vs CPU`) is the worst element-wise absolute error
versus the CPU FFTW3 reference, captured with `--cpu-reference`.

## Headline: benzene (8,219 × 8,219, density 0.36%, `--csc-tile 128`)

| Variant | Median (ms) | Min / Max | Memory (MB) | max_abs vs CPU | vs Dense cuFFT |
|---|---|---|---|---|---|
| CPU FFT reference (FFTW3) | — | — | — | (correctness reference; not benchmarked) | — |
| Dense cuFFT | 23.64 | 23.59 / 23.68 | 3,305 | 4.7e-02 | — |
| SpFFT | ~32 | (single-run) | ~4,400 | 4.7e-02 | 0.74× speed |
| **CSC mixed-radix (custom 1-D FFT)** | **14.65** | **14.61 / 14.66** | **310** | **2.0e-01** | **1.61× faster, 10.7× less memory** |
| CSC mixed-radix (cuFFT 1-D FFT) | 9.60 | 9.60 / 9.62 | 317 | 6.3e-02 | 2.46× faster, 10.4× less memory |

## Small matrix: sstmodel (3,345 × 3,345, density 0.20%)

| Variant | Median (ms) | Memory (MB) | max_abs vs CPU |
|---|---|---|---|
| CPU FFT reference (FFTW3) | — | — | (correctness reference; not benchmarked) |
| Dense cuFFT | 0.85 | 289 | 3.1e-03 |
| SpFFT | 2.08 | 457 | 3.1e-03 |
| CSC mixed-radix (custom) | 3.62 | 65 | 2.0e-02 |
| CSC mixed-radix (cuFFT) | 1.39 | 59 | 6.0e-03 |

At small N, dense cuFFT wins on speed because the sparse path's per-tile
launch overhead (~700 kernel launches even for the small fft_len) and CSC
preprocessing dominate the kernel time. The crossover where sparse beats
dense is around the benzene size on this hardware.

## Large matrix: pct20stif (52,329 × 52,329, density 0.10%)

| Variant | Status |
|---|---|
| Dense cuFFT (baseline) | **`CUFFT_INTERNAL_ERROR`** — `52329 = 3 × 17443` (prime); cuFFT cannot factor it |
| SpFFT | OOM — single-rank limitation at this size |
| CSC mixed-radix (custom) | 2,569 ms / 11.2 GB (fits but memory-tight at >50k nodes) |
| CSC mixed-radix (cuFFT) | 893 ms / 11.3 GB (same memory class, faster) |

Both headline sparse variants run on pct20stif but are memory-tight
because they hold the full output on device. See the **Appendix** for the
streaming approach that bounds device memory by tile size and runs
pct20stif at ~474 MB.

## How sparsity and binary structure are exploited

| Lever | Where it shows up | Effect on the numbers |
|---|---|---|
| **No values array** | Pass-1 kernel never loads or multiplies by a value field; the input is fully described by indices | Halves the input bandwidth vs a non-binary equivalent |
| **uint16-packed indices** | `pack_cols_u16` produces packed `row_idx` and `sparse_cols` whenever `rows, cols ≤ 65,536`; both CSC variants pick the packed kernel automatically via `can_pack_u16()` | Halves index bandwidth for benzene and sstmodel |
| **CompactCSC: skip empty columns** | `make_compact_csc` prefixes only active columns; pass-2 work scales with `n_sparse_cols`, not `cols` | For graph-style matrices where many columns are inactive, this drops pass-2 work directly |
| **Dense grid never materialized** | Pass-1 writes only `(u, c)` slots that have at least one nonzero; the rest of `signal[u, c]` stays zero from the per-tile `cudaMemset` | Memory peak scales with `tile × fft_len`, not `rows × cols` — the reason variant 3 fits in 310 MB on a problem dense cuFFT needs 3.3 GB for |
| **Bluestein + smooth `fft_len`** | Frees us from "size must be a power of 2 because the FFT kernel requires it" — variants 3 and 4 each pick a smooth size suited to their FFT engine | Halves FFT work on benzene compared to power-of-2 |
| **Custom radix-9 fusion** | `stockham_radix9_stage_kernel` — two radix-3 stages in registers | One global-memory roundtrip instead of two; 30% fewer total stages on benzene |
| **CUDA event timing with peak-mem capture** | `DeviceMemPeak` RAII helper in `sparse_fft_bluestein.cu` brackets each variant; `--repeat N` runs N+1 timed iterations and reports median + min/max | Trustworthy benchmark numbers; close-margin rankings (mr vs cufft-smooth) become statistically resolvable |
| **CPU FFTW3 ground truth** | `--cpu-reference` runs FFTW3 first; every CUDA variant (dense cuFFT, SpFFT, custom mr, cufft-smooth) is checked against the same reference | Independent third-party correctness validation against the CPU result |

---

# Limitations and remaining issues

- **Small matrices (sstmodel-class, ≤ ~5k nodes):** dense cuFFT wins by a
  large margin. The sparse pipeline's per-tile launch overhead and CSC
  preprocessing don't amortize. A useful future direction is a single-tile
  fast path for matrices small enough to fit `fft_len × rows` in
  reasonable device memory.
- **Mixed radix only goes to {2, 3}:** adding radix-5 and radix-7 kernels
  would let the custom path use the same 7-smooth `fft_len` cuFFT picks
  (16464 instead of 17496 on benzene); this would close ~5% of the gap to
  the cuFFT variant.
- **SpFFT comparison is approximate:** SpFFT's setup cost (Grid +
  Transform creation, sparse-frequency index registration) is bundled into
  the reported time on small matrices. We time only the `transform.forward()`
  call but the first call still pays cold-start costs.

---

# Profiling

```bash
./scripts/profile.sh dataset/benzene/benzene.mtx --csc-mixed-radix --csc-tile 128
```

Wraps `ncu --set full` with a kernel-name regex covering the variant's
kernels. Writes a `.ncu-rep` file under `profiles/` for inspection in
`ncu-ui`. Requires sudo for hardware counters (or set
`NVreg_RestrictProfilingToAdminUsers=0` once on the host).

The full-set profile of variant 3 on benzene shows ~83% of GPU time in
the Stockham FFT stages (radix-3 dominant at 58%, radix-2 at 25%) — every
stage memory-bandwidth-bound at ~85% of DRAM peak. Pass-1 is only 7.4% of
total time despite the `sincosf`-per-nonzero pattern, so further work
would target FFT stage fusion, not the build kernel.

---

# Appendix: Experimental variants and future directions

These paths are preserved in the codebase but are not part of the
headline four-variant story. Each is callable via its individual
`--csc-*` flag; the implementations live in `src/sparse_fft_bluestein.cu`
(or other `src/sparse_fft_*.cu` files for the disabled variants) and
remain compiled in.

## Streaming — future direction for very-large matrices

For matrices large enough that the full output (`rows × output_stride × 8`
bytes) does not comfortably fit on a single GPU — pct20stif at 52,329²
needs ~10.7 GB just for the output — a streaming pipeline that
double-buffers tiles and writes completed slices to host-pinned memory
keeps the device footprint bounded by `tile × fft_len`.

The codebase already includes an experimental cuFFT-based streaming path
(`--csc-stockham-cufft-stream`) that demonstrates the approach on
pct20stif at ~474 MB device peak. **Future work** is to port the
headline variant 3's custom radix-{2,3,9} kernels to the same streaming
driver, yielding a no-cuFFT path that also scales to >50k matrices.

## Other experimental flags (preserved, ablation only)

- `--csc-stockham-smem` — Bluestein with shared-memory Stockham (radix-2 only; falls back to per-stage for `fft_len > 4096`)
- `--csc-stockham-cufft` — Bluestein + cuFFT, non-streaming, at `next_pow2(2·cols−1)` fft_len (variant 4 supersedes this)
- `--csc-stockham-cufft-stream` — same as above, streaming (the variant that runs pct20stif today)
- `--csc-binary-lut` — Pass-1 with byte-mask + per-tile LUT instead of sincosf-per-nonzero; cuFFT, streaming
- `--csc-binary-stockham-smem` — Same byte-mask LUT pass-1 with the Stockham smem FFT path
- `--csc-mixed-radix-graph` — Variant 3 wrapped in a CUDA Graph for reduced launch overhead
- `--dense-padded` — Dense cuFFT zero-padded to next 7-smooth size (note: padded transform uses a different frequency grid)

## Disabled paths still in source tree

The codebase also contains direct-DFT, COO 2-pass (custom DFT / cuFFT /
GEMM), CSC 2-pass (cached-W, tiled-p2, half-G, streaming), and CSR
2-pass kernels in `src/sparse_fft_direct_dft.cu`, `sparse_fft_coo_2pass.cu`,
`sparse_fft_csc_2pass.cu`, and `sparse_fft_csr_2pass.cu`. Their
benchmark CLI blocks in `main.cu` are commented out.

---

# References

- [cuFFT documentation](https://docs.nvidia.com/cuda/cufft/)
- [SpFFT — open-source sparse FFT for CUDA](https://github.com/eth-cscs/SpFFT)
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [Matrix Market format](https://math.nist.gov/MatrixMarket/formats.html)
- [Bluestein chirp-z transform](https://en.wikipedia.org/wiki/Chirp_Z-transform)
- Temperton (1983), *Self-sorting in-place fast Fourier transforms* — Stockham auto-sort algorithm.
