#include "sparse_fft_internal.cuh"

// ---------------------------------------------------------------------------
// FFT kernel infrastructure (private to this TU)
// bit_reverse, fft_stage, scale_complex, all Stockham variants, pointwise_mul
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned bit_reverse_u32(unsigned x, int bits) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    x = (x << 16) | (x >> 16);
    return x >> (32 - bits);
}

__global__ void bit_reverse_copy_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int n, int log_n, int batch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * batch;
    if (idx >= total) return;
    const int b = idx / n;
    const int i = idx - b * n;
    const int r = (int)bit_reverse_u32((unsigned)i, log_n);
    out[(size_t)b * n + r] = in[(size_t)b * n + i];
}

__global__ void fft_stage_kernel(
    cuFloatComplex* __restrict__ data,
    int n, int half, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = (n / 2) * batch;
    if (idx >= butterflies) return;

    const int b = idx / (n / 2);
    const int k = idx - b * (n / 2);
    const int j = k & (half - 1);
    const int group = k / half;
    const int i0 = group * (half * 2) + j;
    const int i1 = i0 + half;

    const float two_pi = 6.28318530717958647f;
    const float angle = (inverse ? two_pi : -two_pi) * (float)j / (float)(half * 2);
    float s, c;
    __sincosf(angle, &s, &c);

    cuFloatComplex a = data[(size_t)b * n + i0];
    cuFloatComplex z = data[(size_t)b * n + i1];
    const float tx = z.x * c - z.y * s;
    const float ty = z.x * s + z.y * c;
    data[(size_t)b * n + i0] = make_cuFloatComplex(a.x + tx, a.y + ty);
    data[(size_t)b * n + i1] = make_cuFloatComplex(a.x - tx, a.y - ty);
}

__global__ void scale_complex_kernel(cuFloatComplex* data, int total, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

// ---------------------------------------------------------------------------
// Improvement 2: Shared-memory Stockham FFT (one block per row)
//
// When 2 * fft_len * 8 bytes fits in shared memory (≤ SMEM_MAX_FFT_N = 4096),
// load the entire row into smem ping-pong buffers and do ALL butterfly stages
// there.  This replaces log2(n) global-memory round-trips with a single
// global load + single global store — a log2(n)× reduction in DRAM traffic
// (11× for fft_len=2048, 12× for fft_len=4096).
//
// Applicable to sstmodel (fft_len = 2048).  Falls back to the per-stage stream
// variant for larger fft_len (benzene/pct20stif).
// ---------------------------------------------------------------------------
// In-place: reads from inout into smem ping-pong, runs all stages, writes back.
// d_tmp is not touched; can be nullptr for smem path.
__global__ void stockham_smem_row_kernel(
    cuFloatComplex* __restrict__ inout,
    int n, int batch, int inverse)
{
    extern __shared__ cuFloatComplex smem[];   // layout: [0,n) = buf A, [n,2n) = buf B
    const int row = blockIdx.x;
    if (row >= batch) return;

    cuFloatComplex* sa = smem;
    cuFloatComplex* sb = smem + n;

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sa[i] = inout[(size_t)row * n + i];
    __syncthreads();

    cuFloatComplex* src = sa, *dst = sb;
    const float sign = inverse ? 6.28318530717958647f : -6.28318530717958647f;

    for (int l = 1; l < n; l <<= 1) {
        const int m = n / (2 * l);
        for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
            const int j = t / m, k = t - j * m;
            const int in0 = j*(2*m)+k, in1 = in0+m;
            const int out0 = j*m+k,    out1 = out0+n/2;
            float s, c;
            __sincosf(sign * (float)k / (float)(2*m), &s, &c);
            cuFloatComplex a = src[in0], z = src[in1];
            const float dx = a.x-z.x, dy = a.y-z.y;
            dst[out0] = make_cuFloatComplex(a.x+z.x, a.y+z.y);
            dst[out1] = make_cuFloatComplex(fmaf(dx,c,-dy*s), fmaf(dx,s,dy*c));
        }
        __syncthreads();
        cuFloatComplex* tmp = src; src = dst; dst = tmp;
    }

    if (inverse) {
        const float sc = 1.f / (float)n;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            src[i] = make_cuFloatComplex(src[i].x * sc, src[i].y * sc);
        __syncthreads();
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        inout[(size_t)row * n + i] = src[i];
}

// Mul-inverse variant: pre-multiplies each element by mul[i % n] (the B_fft
// pointwise factor) before the IFFT.  Mathematically equivalent to fusing the
// multiply into the first butterfly stage, but implemented as a single smem pass.
__global__ void stockham_smem_row_mul_inverse_kernel(
    cuFloatComplex* __restrict__ inout,
    const cuFloatComplex* __restrict__ mul,
    int n, int batch)
{
    extern __shared__ cuFloatComplex smem[];
    const int row = blockIdx.x;
    if (row >= batch) return;

    cuFloatComplex* sa = smem, *sb = smem + n;

    // Load and pre-multiply by B_fft element-wise (all n positions).
    // This is equivalent to what stockham_stage_mul_kernel does at l=1, m=n/2:
    // in0=k, in1=k+n/2 covers all [0,n), so pre-multiplying all elements first
    // then running the regular butterfly gives identical results.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        cuFloatComplex x = inout[(size_t)row * n + i];
        cuFloatComplex m = mul[i];
        sa[i] = make_cuFloatComplex(fmaf(x.x,m.x,-x.y*m.y), fmaf(x.x,m.y,x.y*m.x));
    }
    __syncthreads();

    cuFloatComplex* src = sa, *dst = sb;
    const float sign = 6.28318530717958647f; // inverse

    for (int l = 1; l < n; l <<= 1) {
        const int m = n / (2 * l);
        for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
            const int j = t / m, k = t - j * m;
            const int in0 = j*(2*m)+k, in1 = in0+m;
            const int out0 = j*m+k,    out1 = out0+n/2;
            float s, c;
            __sincosf(sign * (float)k / (float)(2*m), &s, &c);
            cuFloatComplex a = src[in0], z = src[in1];
            const float dx = a.x-z.x, dy = a.y-z.y;
            dst[out0] = make_cuFloatComplex(a.x+z.x, a.y+z.y);
            dst[out1] = make_cuFloatComplex(fmaf(dx,c,-dy*s), fmaf(dx,s,dy*c));
        }
        __syncthreads();
        cuFloatComplex* tmp = src; src = dst; dst = tmp;
    }

    const float sc = 1.f / (float)n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        src[i] = make_cuFloatComplex(src[i].x * sc, src[i].y * sc);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        inout[(size_t)row * n + i] = src[i];
}

__global__ void stockham_stage_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int n, int l, int m, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = (n / 2) * batch;
    if (idx >= butterflies) return;

    const int b = idx / (n / 2);
    const int t = idx - b * (n / 2);
    const int j = t / m;
    const int k = t - j * m;

    const int in0 = j * (2 * m) + k;
    const int in1 = in0 + m;
    const int out0 = j * m + k;
    const int out1 = out0 + n / 2;

    const float two_pi = 6.28318530717958647f;
    const float angle = (inverse ? two_pi : -two_pi) * (float)k / (float)(2 * m);
    float s, c;
    __sincosf(angle, &s, &c);

    cuFloatComplex a = in[(size_t)b * n + in0];
    cuFloatComplex z = in[(size_t)b * n + in1];
    const float dx = a.x - z.x;
    const float dy = a.y - z.y;
    out[(size_t)b * n + out0] = make_cuFloatComplex(a.x + z.x, a.y + z.y);
    out[(size_t)b * n + out1] = make_cuFloatComplex(fmaf(dx, c, -dy * s),
                                                    fmaf(dx, s,  dy * c));
}

__global__ void stockham_stage_mul_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    const cuFloatComplex* __restrict__ mul,
    int n, int l, int m, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = (n / 2) * batch;
    if (idx >= butterflies) return;

    const int b = idx / (n / 2);
    const int t = idx - b * (n / 2);
    const int j = t / m;
    const int k = t - j * m;

    const int in0 = j * (2 * m) + k;
    const int in1 = in0 + m;
    const int out0 = j * m + k;
    const int out1 = out0 + n / 2;

    const float two_pi = 6.28318530717958647f;
    const float angle = (inverse ? two_pi : -two_pi) * (float)k / (float)(2 * m);
    float s, c;
    __sincosf(angle, &s, &c);

    cuFloatComplex x0 = in[(size_t)b * n + in0];
    cuFloatComplex x1 = in[(size_t)b * n + in1];
    cuFloatComplex w0 = mul[in0];
    cuFloatComplex w1 = mul[in1];
    cuFloatComplex a = make_cuFloatComplex(fmaf(x0.x, w0.x, -x0.y * w0.y),
                                           fmaf(x0.x, w0.y,  x0.y * w0.x));
    cuFloatComplex z = make_cuFloatComplex(fmaf(x1.x, w1.x, -x1.y * w1.y),
                                           fmaf(x1.x, w1.y,  x1.y * w1.x));

    const float dx = a.x - z.x;
    const float dy = a.y - z.y;
    out[(size_t)b * n + out0] = make_cuFloatComplex(a.x + z.x, a.y + z.y);
    out[(size_t)b * n + out1] = make_cuFloatComplex(fmaf(dx, c, -dy * s),
                                                    fmaf(dx, s,  dy * c));
}

__global__ void stockham_two_stage_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int n, int l, int m, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = (n / 2) * batch;
    if (idx >= butterflies) return;

    const int b = idx / (n / 2);
    const int t = idx - b * (n / 2);
    const int m2 = m >> 1;
    const int j2 = t / m2;
    const int k = t - j2 * m2;
    const int j = j2 & (l - 1);
    const int lower = j2 >= l;

    const int base = j * (2 * m);
    const int i0 = base + k;
    const int i1 = i0 + m2;
    const int i2 = i0 + m;
    const int i3 = i2 + m2;

    const float two_pi = 6.28318530717958647f;
    const float sign = inverse ? two_pi : -two_pi;
    float s0, c0, s1, c1, sb, cb;
    __sincosf(sign * (float)k / (float)(2 * m), &s0, &c0);
    __sincosf(sign * (float)(k + m2) / (float)(2 * m), &s1, &c1);
    __sincosf(sign * (float)k / (float)m, &sb, &cb);

    const size_t row = (size_t)b * n;
    cuFloatComplex x0 = in[row + i0];
    cuFloatComplex x1 = in[row + i1];
    cuFloatComplex x2 = in[row + i2];
    cuFloatComplex x3 = in[row + i3];

    // Branchless: upper outputs (lower==false) use twiddle=(1,0) so d*c-d*s == d.
    // Both paths share the same instruction structure — compiler emits predicated
    // selects instead of a divergent BRA.
    const float d0x = x0.x - x2.x;
    const float d0y = x0.y - x2.y;
    const float d1x = x1.x - x3.x;
    const float d1y = x1.y - x3.y;
    // upper: a = sum; lower: a = twiddle(diff)
    const float a0x = lower ? fmaf(d0x, c0, -d0y * s0) : x0.x + x2.x;
    const float a0y = lower ? fmaf(d0x, s0,  d0y * c0) : x0.y + x2.y;
    const float a1x = lower ? fmaf(d1x, c1, -d1y * s1) : x1.x + x3.x;
    const float a1y = lower ? fmaf(d1x, s1,  d1y * c1) : x1.y + x3.y;

    const float dx = a0x - a1x;
    const float dy = a0y - a1y;
    const int out0 = j2 * m2 + k;
    const int out1 = out0 + n / 2;
    out[row + out0] = make_cuFloatComplex(a0x + a1x, a0y + a1y);
    out[row + out1] = make_cuFloatComplex(fmaf(dx, cb, -dy * sb),
                                          fmaf(dx, sb,  dy * cb));
}

__global__ void pointwise_mul_batched_kernel(
    cuFloatComplex* __restrict__ a,
    const cuFloatComplex* __restrict__ b,
    int n, int batch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * batch;
    if (idx >= total) return;
    const int k = idx % n;
    cuFloatComplex x = a[idx];
    cuFloatComplex y = b[k];
    a[idx] = make_cuFloatComplex(fmaf(x.x, y.x, -x.y * y.y),
                                 fmaf(x.x, y.y,  x.y * y.x));
}

// ---------------------------------------------------------------------------
// Bluestein input build kernels: 1-D layout (active fast path)
// ---------------------------------------------------------------------------
__global__ void csc_build_bluestein_input_kernel(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    const int* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = sparse_cols[col_id];

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];
    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(re * q.x - im * q.y, re * q.y + im * q.x);
}

__global__ void csc_build_bluestein_input_packed_kernel(
    const int* __restrict__ col_ptr,
    const uint16_t* __restrict__ row_idx,
    const uint16_t* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = (int)sparse_cols[col_id];

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];
    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, (int)row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(fmaf(re, q.x, -im * q.y),
                            fmaf(re, q.y,  im * q.x));
}

// ---------------------------------------------------------------------------
// Bluestein input build kernels: 2-D coalesced layout (alternative variant)
// ---------------------------------------------------------------------------
__global__ void csc_build_bluestein_input_coalesced_kernel(
    const int* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    const int* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x * CSC_BUILD_COLS + threadIdx.x;
    const int u_local = blockIdx.y * CSC_BUILD_U + threadIdx.y;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = sparse_cols[col_id];

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];
    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(re * q.x - im * q.y, re * q.y + im * q.x);
}

__global__ void csc_build_bluestein_input_coalesced_packed_kernel(
    const int* __restrict__ col_ptr,
    const uint16_t* __restrict__ row_idx,
    const uint16_t* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, float inv_rows)
{
    const int col_id = blockIdx.x * CSC_BUILD_COLS + threadIdx.x;
    const int u_local = blockIdx.y * CSC_BUILD_U + threadIdx.y;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;

    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = (int)sparse_cols[col_id];

    const int start = col_ptr[col_id];
    const int end = col_ptr[col_id + 1];
    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        float s, co;
        sincos_twiddle(u, (int)row_idx[p], rows, inv_rows, &s, &co);
        re += co;
        im += s;
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(re * q.x - im * q.y, re * q.y + im * q.x);
}

// ---------------------------------------------------------------------------
// Bluestein finalize kernel
// ---------------------------------------------------------------------------
__global__ void bluestein_finalize_kernel(
    const cuFloatComplex* __restrict__ conv,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ out,
    int rows, int output_stride, int output_cols,
    int fft_len, int u_base, int tile_rows)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u_local = blockIdx.y * blockDim.y + threadIdx.y;
    const int u = u_base + u_local;
    if (v >= output_cols || u_local >= tile_rows || u >= rows) return;

    cuFloatComplex y = conv[(size_t)u_local * fft_len + v];
    cuFloatComplex q = chirp[v];
    out[(size_t)u * output_stride + v] =
        make_cuFloatComplex(fmaf(y.x, q.x, -y.y * q.y),
                            fmaf(y.x, q.y,  y.y * q.x));
}

// ---------------------------------------------------------------------------
// Stockham FFT host helpers (private to this TU)
// ---------------------------------------------------------------------------
static void run_fft_power2(cuFloatComplex* d_in,
                           cuFloatComplex* d_tmp,
                           int n, int batch, bool inverse)
{
    const int total = n * batch;
    const int threads = 256;
    const int log_n = log2_exact(n);
    bit_reverse_copy_kernel<<<(total + threads - 1) / threads, threads>>>(
        d_in, d_tmp, n, log_n, batch);
    CUDA_CHECK(cudaGetLastError());

    for (int half = 1; half < n; half <<= 1) {
        const int butterflies = (n / 2) * batch;
        fft_stage_kernel<<<(butterflies + threads - 1) / threads, threads>>>(
            d_tmp, n, half, batch, inverse ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
    }

    if (inverse) {
        scale_complex_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_tmp, total, 1.f / (float)n);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void run_fft_power2_stockham(cuFloatComplex* d_in,
                                    cuFloatComplex* d_tmp,
                                    int n, int batch, bool inverse)
{
    const int threads = 256;
    const int butterflies = (n / 2) * batch;
    cuFloatComplex* src = d_in;
    cuFloatComplex* dst = d_tmp;

    for (int l = 1; l < n; ) {
        const int m = n / (2 * l);
        if ((l << 1) < n) {
            stockham_two_stage_kernel<<<(butterflies + threads - 1) / threads, threads>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            l <<= 2;
        } else {
            stockham_stage_kernel<<<(butterflies + threads - 1) / threads, threads>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            l <<= 1;
        }
        cuFloatComplex* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (src != d_tmp) {
        CUDA_CHECK(cudaMemcpy(d_tmp, src,
                              (size_t)n * batch * sizeof(cuFloatComplex),
                              cudaMemcpyDeviceToDevice));
    }

    if (inverse) {
        const int total = n * batch;
        scale_complex_kernel<<<(total + threads - 1) / threads, threads>>>(
            d_tmp, total, 1.f / (float)n);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Returns pointer to whichever buffer (d_in or d_tmp) holds the FFT result,
// eliminating the D2D copy needed when the stage count is odd.
static cuFloatComplex* run_fft_power2_stockham_stream(cuFloatComplex* d_in,
                                                      cuFloatComplex* d_tmp,
                                                      int n, int batch, bool inverse,
                                                      cudaStream_t stream)
{
    const int threads = 256;
    const int butterflies = (n / 2) * batch;
    cuFloatComplex* src = d_in;
    cuFloatComplex* dst = d_tmp;

    for (int l = 1; l < n; ) {
        const int m = n / (2 * l);
        if ((l << 1) < n) {
            stockham_two_stage_kernel<<<(butterflies + threads - 1) / threads,
                                        threads, 0, stream>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            l <<= 2;
        } else {
            stockham_stage_kernel<<<(butterflies + threads - 1) / threads,
                                    threads, 0, stream>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            l <<= 1;
        }
        cuFloatComplex* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (inverse) {
        const int total = n * batch;
        scale_complex_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
            src, total, 1.f / (float)n);
        CUDA_CHECK(cudaGetLastError());
    }
    return src;
}

// Returns pointer to whichever buffer (d_in or d_tmp) holds the IFFT result.
static cuFloatComplex* run_fft_power2_stockham_mul_inverse_stream(
    cuFloatComplex* d_in,
    cuFloatComplex* d_tmp,
    const cuFloatComplex* d_mul,
    int n, int batch, cudaStream_t stream)
{
    const int threads = 256;
    const int butterflies = (n / 2) * batch;

    // First stage: multiply by chirp and do first butterfly d_in → d_tmp.
    int l = 1;
    int m = n / 2;
    stockham_stage_mul_kernel<<<(butterflies + threads - 1) / threads,
                                threads, 0, stream>>>(
        d_in, d_tmp, d_mul, n, l, m, batch, 1);
    CUDA_CHECK(cudaGetLastError());

    cuFloatComplex* src = d_tmp;
    cuFloatComplex* dst = d_in;
    for (l = 2; l < n; ) {
        m = n / (2 * l);
        if ((l << 1) < n) {
            stockham_two_stage_kernel<<<(butterflies + threads - 1) / threads,
                                        threads, 0, stream>>>(
                src, dst, n, l, m, batch, 1);
            CUDA_CHECK(cudaGetLastError());
            l <<= 2;
        } else {
            stockham_stage_kernel<<<(butterflies + threads - 1) / threads,
                                    threads, 0, stream>>>(
                src, dst, n, l, m, batch, 1);
            CUDA_CHECK(cudaGetLastError());
            l <<= 1;
        }
        cuFloatComplex* tmp = src;
        src = dst;
        dst = tmp;
    }

    const int total = n * batch;
    scale_complex_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        src, total, 1.f / (float)n);
    CUDA_CHECK(cudaGetLastError());
    return src;
}

// ---------------------------------------------------------------------------
// Improvement 2 host runners — smem path
//
// Returns d_in (in-place).  When fft_len > SMEM_MAX_FFT_N the call transparently
// falls back to the per-stage stream variant so callers need not branch.
// ---------------------------------------------------------------------------
static cuFloatComplex* run_fft_power2_stockham_smem_stream(
    cuFloatComplex* d_in, cuFloatComplex* d_tmp,
    int n, int batch, bool inverse, cudaStream_t stream)
{
    const size_t smem_needed = 2 * (size_t)n * sizeof(cuFloatComplex);
    if (n > SMEM_MAX_FFT_N || smem_needed > 96 * 1024UL) {
        return run_fft_power2_stockham_stream(d_in, d_tmp, n, batch, inverse, stream);
    }
    const int bdim = std::min(512, n / 2);
    stockham_smem_row_kernel<<<batch, bdim, smem_needed, stream>>>(
        d_in, n, batch, inverse ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    return d_in; // in-place: result always in d_in
}

static cuFloatComplex* run_fft_power2_stockham_smem_mul_inverse_stream(
    cuFloatComplex* d_in, cuFloatComplex* d_tmp,
    const cuFloatComplex* d_mul, int n, int batch, cudaStream_t stream)
{
    const size_t smem_needed = 2 * (size_t)n * sizeof(cuFloatComplex);
    if (n > SMEM_MAX_FFT_N || smem_needed > 96 * 1024UL) {
        return run_fft_power2_stockham_mul_inverse_stream(d_in, d_tmp, d_mul, n, batch, stream);
    }
    const int bdim = std::min(512, n / 2);
    stockham_smem_row_mul_inverse_kernel<<<batch, bdim, smem_needed, stream>>>(
        d_in, d_mul, n, batch);
    CUDA_CHECK(cudaGetLastError());
    return d_in; // in-place: result always in d_in
}

// ---------------------------------------------------------------------------
// Public API: Bluestein / Stockham / cuFFT variants
// ---------------------------------------------------------------------------
SparseFFTResult sparse_fft_csc_bluestein(const COOMatrix& coo, int u_tile,
                                         bool use_stockham, int fft_backend) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len = next_power_of_two(2 * cols - 1);
    const bool use_packed = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> row_idx_packed = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, row_idx_packed.data(),
                              coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp,  cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft,  fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(), cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // fft_backend==3 (cuFFT) operates in-place on d_signal — d_work is not needed.
    // Backends 0 and 1 use d_work as a ping-pong scratch buffer.
    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr, *d_work = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal, (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    if (fft_backend != 3)
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out,    (size_t)rows * output_stride * sizeof(cuFloatComplex)));

    // Stage h_b through d_work for the one-time d_b_fft precompute.
    // fft_backend==3 (cuFFT) uses an in-place plan directly on d_b_fft; all others
    // stage through d_work so the tile loop can overwrite it freely.
    if (fft_backend == 3) {
        // cuFFT backend: copy h_b directly to d_b_fft and FFT in-place.
        CUDA_CHECK(cudaMemcpy(d_b_fft, h_b.data(), fft_len * sizeof(cuFloatComplex),
                              cudaMemcpyHostToDevice));
        int fl = fft_len;
        cufftHandle plan_b;
        CUFFT_CHECK(cufftPlanMany(&plan_b, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, 1));
        CUFFT_CHECK(cufftExecC2C(plan_b,
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 CUFFT_FORWARD));
        CUFFT_CHECK(cufftDestroy(plan_b));
    } else {
        CUDA_CHECK(cudaMemcpy(d_work, h_b.data(), fft_len * sizeof(cuFloatComplex),
                              cudaMemcpyHostToDevice));
        if (use_stockham)
            run_fft_power2_stockham(d_work, d_b_fft, fft_len, 1, false);
        else
            run_fft_power2(d_work, d_b_fft, fft_len, 1, false);
    }

    // Build cuFFT plans for the tile loop (fft_backend==3 only).
    // Create one plan for full tiles and, if needed, a second for the last partial tile.
    // Query cufftGetSize so the internal workspace is included in mem_bytes.
    cufftHandle plan_cufft = 0, plan_cufft_last = 0;
    size_t cufft_workspace = 0;
    if (fft_backend == 3) {
        int fl = fft_len;
        const int full_tile = std::min(u_tile, rows);
        CUFFT_CHECK(cufftPlanMany(&plan_cufft, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, full_tile));
        size_t ws = 0;
        CUFFT_CHECK(cufftGetSize(plan_cufft, &ws));
        cufft_workspace = ws;
        const int last_tile = rows % u_tile;
        if (last_tile != 0 && last_tile != full_tile) {
            CUFFT_CHECK(cufftPlanMany(&plan_cufft_last, 1, &fl, nullptr, 1, fft_len,
                                      nullptr, 1, fft_len, CUFFT_C2C, last_tile));
            CUFFT_CHECK(cufftGetSize(plan_cufft_last, &ws));
            cufft_workspace = std::max(cufft_workspace, ws);
        }
    }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);
        CUDA_CHECK(cudaMemset(d_signal, 0, signal_bytes));

        if (use_stockham) {
            // 1D layout: blockIdx.x = col_id, threadIdx.x = u_local.
            // All u-threads share the same col_id → row_idx[p] is the same address
            // for every thread in the block → L1 broadcast, near-zero excess sectors.
            dim3 p1_block(CSC_BU);
            dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
            if (use_packed) {
                csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block>>>(
                    d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, d_signal,
                    rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
            } else {
                csc_build_bluestein_input_kernel<<<p1_grid, p1_block>>>(
                    d_col_ptr, d_row_idx, d_sp_cols, d_chirp, d_signal,
                    rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
            }
        } else {
            dim3 p1_block(CSC_BU);
            dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
            if (use_packed) {
                csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block>>>(
                    d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, d_signal,
                    rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
            } else {
                csc_build_bluestein_input_kernel<<<p1_grid, p1_block>>>(
                    d_col_ptr, d_row_idx, d_sp_cols, d_chirp, d_signal,
                    rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
            }
        }
        CUDA_CHECK(cudaGetLastError());

        cuFloatComplex* conv_result;
        if (use_stockham) {
            if (fft_backend == 1) {
                // smem row kernel — entire FFT in shared memory (improvement 2).
                // In-place on d_signal; d_work is ping-pong scratch (may be unused
                // if smem path activates, used as fallback otherwise).
                cuFloatComplex* fwd = run_fft_power2_stockham_smem_stream(
                    d_signal, d_work, fft_len, tile_rows, false, 0);
                cuFloatComplex* fwd_other = (fwd == d_signal) ? d_work : d_signal;
                conv_result = run_fft_power2_stockham_smem_mul_inverse_stream(
                    fwd, fwd_other, d_b_fft, fft_len, tile_rows, 0);
            } else if (fft_backend == 3) {
                // cuFFT C2C backend: uses cuFFT's internal smem-staged butterfly,
                // replacing O(N log N) global-memory passes with O(N) per direction.
                // Forward FFT in-place, pointwise ×B_fft, inverse FFT in-place, scale.
                cufftHandle tile_plan =
                    (plan_cufft_last && tile_rows != std::min(u_tile, rows))
                    ? plan_cufft_last : plan_cufft;
                CUFFT_CHECK(cufftExecC2C(tile_plan,
                    reinterpret_cast<cufftComplex*>(d_signal),
                    reinterpret_cast<cufftComplex*>(d_signal),
                    CUFFT_FORWARD));
                const int total_elems = tile_rows * fft_len;
                pointwise_mul_batched_kernel<<<(total_elems + 255) / 256, 256>>>(
                    d_signal, d_b_fft, fft_len, tile_rows);
                CUDA_CHECK(cudaGetLastError());
                CUFFT_CHECK(cufftExecC2C(tile_plan,
                    reinterpret_cast<cufftComplex*>(d_signal),
                    reinterpret_cast<cufftComplex*>(d_signal),
                    CUFFT_INVERSE));
                // cuFFT does not normalize the inverse; divide by fft_len.
                scale_complex_kernel<<<(total_elems + 255) / 256, 256>>>(
                    d_signal, total_elems, 1.f / (float)fft_len);
                CUDA_CHECK(cudaGetLastError());
                conv_result = d_signal;
            } else {
                // Default: per-stage two-stage Stockham loop.
                run_fft_power2_stockham(d_signal, d_work, fft_len, tile_rows, false);
                conv_result = run_fft_power2_stockham_mul_inverse_stream(
                    d_work, d_signal, d_b_fft, fft_len, tile_rows, 0);
            }
        } else {
            run_fft_power2(d_signal, d_work, fft_len, tile_rows, false);
            pointwise_mul_batched_kernel<<<((int)((size_t)tile_rows * fft_len) + 255) / 256, 256>>>(
                d_work, d_b_fft, fft_len, tile_rows);
            CUDA_CHECK(cudaGetLastError());
            run_fft_power2(d_work, d_signal, fft_len, tile_rows, true);
            conv_result = d_signal;
        }

        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block>>>(
            conv_result, d_chirp, d_out,
            rows, output_stride, output_cols, fft_len, u_base, tile_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (plan_cufft)      cufftDestroy(plan_cufft);
    if (plan_cufft_last) cufftDestroy(plan_cufft_last);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_work);  // nullptr-safe when fft_backend==3

    const size_t input_bytes = (size_t)(n_sp + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)coo.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)coo.nnz + (size_t)n_sp) * sizeof(int));
    const size_t work_bytes = (fft_backend != 3)
                            ? (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex) : 0;
    return {d_out, output_stride, ms, input_bytes
                     + (size_t)cols    * sizeof(cuFloatComplex)   // d_chirp
                     + (size_t)fft_len * sizeof(cuFloatComplex)   // d_b_fft
                     + (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)  // d_signal
                     + work_bytes                                                 // d_work (0 for backend 3)
                     + cufft_workspace                                            // cuFFT internal scratch
                     + (size_t)rows * output_stride * sizeof(cuFloatComplex)};   // d_out
}

SparseFFTResult sparse_fft_csc_bluestein_smem(const COOMatrix& coo, int u_tile) {
    return sparse_fft_csc_bluestein(coo, u_tile, true, 1);
}

SparseFFTResult sparse_fft_csc_bluestein_cufft(const COOMatrix& coo, int u_tile) {
    return sparse_fft_csc_bluestein(coo, u_tile, true, 3);
}

SparseFFTResult sparse_fft_csc_stockham_streaming(const COOMatrix& coo, int u_tile,
                                                   int fft_backend) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len = next_power_of_two(2 * cols - 1);
    const bool use_packed = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> row_idx_packed = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, row_idx_packed.data(),
                              coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp,  cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft,  fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(), cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // fft_backend==3 (cuFFT) operates in-place on d_signal — d_work not needed.
    // Backends 0/1 use d_work as ping-pong scratch across Stockham stages.
    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr, *d_work = nullptr;
    cuFloatComplex *d_out_chunk[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_signal, (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    if (fft_backend != 3)
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk[0], (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk[1], (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));

    // Precompute d_b_fft on the null stream before the tile loop.
    // Backend 3: copy h_b directly to d_b_fft and FFT in-place with a one-shot cuFFT plan.
    // Others: stage through d_signal (avoids a separate d_b allocation).
    if (fft_backend == 3) {
        CUDA_CHECK(cudaMemcpy(d_b_fft, h_b.data(), fft_len * sizeof(cuFloatComplex),
                              cudaMemcpyHostToDevice));
        int fl = fft_len;
        cufftHandle plan_b;
        CUFFT_CHECK(cufftPlanMany(&plan_b, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, 1));
        CUFFT_CHECK(cufftExecC2C(plan_b,
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 CUFFT_FORWARD));
        CUFFT_CHECK(cufftDestroy(plan_b));
    } else {
        CUDA_CHECK(cudaMemcpy(d_signal, h_b.data(), fft_len * sizeof(cuFloatComplex),
                              cudaMemcpyHostToDevice));
        run_fft_power2_stockham(d_signal, d_b_fft, fft_len, 1, false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cuFloatComplex* h_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pinned,
                             (size_t)rows * output_stride * sizeof(cuFloatComplex),
                             cudaHostAllocDefault));

    // compute_stream: build + FFTs + finalize
    // copy_stream:    async D2H transfers (overlaps with compute_stream)
    cudaStream_t compute_stream, copy_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&copy_stream));

    // Build cuFFT tile plans (fft_backend==3 only) now that compute_stream exists.
    cufftHandle plan_cufft = 0, plan_cufft_last = 0;
    size_t cufft_workspace = 0;
    if (fft_backend == 3) {
        int fl = fft_len;
        const int full_tile = std::min(u_tile, rows);
        CUFFT_CHECK(cufftPlanMany(&plan_cufft, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, full_tile));
        CUFFT_CHECK(cufftSetStream(plan_cufft, compute_stream));
        size_t ws = 0;
        CUFFT_CHECK(cufftGetSize(plan_cufft, &ws));
        cufft_workspace = ws;
        const int last_tile = rows % u_tile;
        if (last_tile != 0 && last_tile != full_tile) {
            CUFFT_CHECK(cufftPlanMany(&plan_cufft_last, 1, &fl, nullptr, 1, fft_len,
                                      nullptr, 1, fft_len, CUFFT_C2C, last_tile));
            CUFFT_CHECK(cufftSetStream(plan_cufft_last, compute_stream));
            CUFFT_CHECK(cufftGetSize(plan_cufft_last, &ws));
            cufft_workspace = std::max(cufft_workspace, ws);
        }
    }

    // compute_done[b]: signaled after finalize writes d_out_chunk[b]
    // d2h_done[b]:     signaled after D2H of d_out_chunk[b] completes
    // Initialise d2h_done as pre-signaled so the first two tiles don't stall.
    cudaEvent_t compute_done[2], d2h_done[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaEventCreate(&compute_done[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&d2h_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(d2h_done[i], compute_stream)); // immediately satisfied
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaEventSynchronize(t0));

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const int buf = (u_base / u_tile) & 1;
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);

        // Don't overwrite d_out_chunk[buf] until its in-flight D2H (if any) is done.
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, d2h_done[buf]));

        CUDA_CHECK(cudaMemsetAsync(d_signal, 0, signal_bytes, compute_stream));

        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block, 0, compute_stream>>>(
                d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_bluestein_input_kernel<<<p1_grid, p1_block, 0, compute_stream>>>(
                d_col_ptr, d_row_idx, d_sp_cols, d_chirp, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        cuFloatComplex* conv_result;
        if (fft_backend == 3) {
            cufftHandle tile_plan =
                (plan_cufft_last && tile_rows != std::min(u_tile, rows))
                ? plan_cufft_last : plan_cufft;
            CUFFT_CHECK(cufftExecC2C(tile_plan,
                reinterpret_cast<cufftComplex*>(d_signal),
                reinterpret_cast<cufftComplex*>(d_signal),
                CUFFT_FORWARD));
            const int total_elems = tile_rows * fft_len;
            pointwise_mul_batched_kernel<<<(total_elems + 255) / 256, 256, 0, compute_stream>>>(
                d_signal, d_b_fft, fft_len, tile_rows);
            CUDA_CHECK(cudaGetLastError());
            CUFFT_CHECK(cufftExecC2C(tile_plan,
                reinterpret_cast<cufftComplex*>(d_signal),
                reinterpret_cast<cufftComplex*>(d_signal),
                CUFFT_INVERSE));
            scale_complex_kernel<<<(total_elems + 255) / 256, 256, 0, compute_stream>>>(
                d_signal, total_elems, 1.f / (float)fft_len);
            CUDA_CHECK(cudaGetLastError());
            conv_result = d_signal;
        } else {
            cuFloatComplex* fwd_result = run_fft_power2_stockham_stream(
                d_signal, d_work, fft_len, tile_rows, false, compute_stream);
            cuFloatComplex* fwd_other = (fwd_result == d_signal) ? d_work : d_signal;
            conv_result = run_fft_power2_stockham_mul_inverse_stream(
                fwd_result, fwd_other, d_b_fft, fft_len, tile_rows, compute_stream);
        }

        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block, 0, compute_stream>>>(
            conv_result, d_chirp, d_out_chunk[buf],
            tile_rows, output_stride, output_cols, fft_len, 0, tile_rows);
        CUDA_CHECK(cudaGetLastError());

        // Signal that d_out_chunk[buf] is ready for D2H.
        CUDA_CHECK(cudaEventRecord(compute_done[buf], compute_stream));

        // Kick off async D2H on the copy stream — runs concurrently with next tile's compute.
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, compute_done[buf]));
        CUDA_CHECK(cudaMemcpyAsync(h_pinned + (size_t)u_base * output_stride,
                                   d_out_chunk[buf],
                                   (size_t)tile_rows * output_stride * sizeof(cuFloatComplex),
                                   cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaEventRecord(d2h_done[buf], copy_stream));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    std::vector<cuFloatComplex> h_output(h_pinned, h_pinned + (size_t)rows * output_stride);

    for (int i = 0; i < 2; i++) {
        cudaEventDestroy(compute_done[i]);
        cudaEventDestroy(d2h_done[i]);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    if (plan_cufft)      cufftDestroy(plan_cufft);
    if (plan_cufft_last) cufftDestroy(plan_cufft_last);
    cudaFreeHost(h_pinned);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_work);  // nullptr-safe when fft_backend==3
    cudaFree(d_out_chunk[0]);
    cudaFree(d_out_chunk[1]);

    const size_t input_bytes = (size_t)(n_sp + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)coo.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)coo.nnz + (size_t)n_sp) * sizeof(int));
    const size_t work_bytes = (fft_backend != 3)
                            ? (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex) : 0;
    return {nullptr, output_stride, ms, input_bytes
                     + (size_t)cols    * sizeof(cuFloatComplex)   // d_chirp
                     + (size_t)fft_len * sizeof(cuFloatComplex)   // d_b_fft
                     + (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)  // d_signal
                     + work_bytes                                                 // d_work (0 for backend 3)
                     + cufft_workspace                                            // cuFFT internal scratch
                     + 2 * (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex), // d_out_chunk[2]
            std::move(h_output)};
}

SparseFFTResult sparse_fft_csc_bluestein_cufft_streaming(const COOMatrix& coo, int u_tile) {
    return sparse_fft_csc_stockham_streaming(coo, u_tile, 3);
}


// ---------------------------------------------------------------------------
// CUDA Graph variant (same Bluestein kernel family)
// ---------------------------------------------------------------------------
SparseFFTResult sparse_fft_csc_stockham_graph(const COOMatrix& coo, int u_tile) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len = next_power_of_two(2 * cols - 1);
    const bool use_packed = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> row_idx_packed = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp_cols_packed = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, row_idx_packed.data(),
                              coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_cols_packed.data(),
                              n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp,  cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft,  fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(), cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr, *d_work = nullptr, *d_out_chunk = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal,    (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_work,      (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk, (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));

    // Stage h_b through d_signal for the one-time d_b_fft precompute (eliminates d_b).
    CUDA_CHECK(cudaMemcpy(d_signal, h_b.data(), fft_len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    run_fft_power2_stockham(d_signal, d_b_fft, fft_len, 1, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    cuFloatComplex* h_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pinned,
                             (size_t)rows * output_stride * sizeof(cuFloatComplex),
                             cudaHostAllocDefault));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);

        CUDA_CHECK(cudaMemsetAsync(d_signal, 0, signal_bytes, stream));

        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_bluestein_input_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx, d_sp_cols, d_chirp, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        cuFloatComplex* fwd_result = run_fft_power2_stockham_stream(
            d_signal, d_work, fft_len, tile_rows, false, stream);
        cuFloatComplex* fwd_other = (fwd_result == d_signal) ? d_work : d_signal;
        cuFloatComplex* inv_result = run_fft_power2_stockham_mul_inverse_stream(
            fwd_result, fwd_other, d_b_fft, fft_len, tile_rows, stream);

        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block, 0, stream>>>(
            inv_result, d_chirp, d_out_chunk,
            tile_rows, output_stride, output_cols, fft_len, 0, tile_rows);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_pinned + (size_t)u_base * output_stride,
                                   d_out_chunk,
                                   (size_t)tile_rows * output_stride * sizeof(cuFloatComplex),
                                   cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    std::vector<cuFloatComplex> h_output(h_pinned, h_pinned + (size_t)rows * output_stride);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_pinned);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_work);
    cudaFree(d_out_chunk);

    const size_t input_bytes = (size_t)(n_sp + 1) * sizeof(int)
                             + (use_packed
                                ? ((size_t)coo.nnz + (size_t)n_sp) * sizeof(uint16_t)
                                : ((size_t)coo.nnz + (size_t)n_sp) * sizeof(int));
    return {nullptr, output_stride, ms, input_bytes
                     + (size_t)cols    * sizeof(cuFloatComplex)   // d_chirp
                     + (size_t)fft_len * sizeof(cuFloatComplex)   // d_b_fft (d_b eliminated)
                     + 2 * (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)   // d_signal + d_work
                     + (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex), // d_out_chunk
            std::move(h_output)};
}


