#include "sparse_fft_internal.cuh"
#include <chrono>
#include <cstdlib>
#include <cstdio>

// Tracks peak device memory used between construction and finish().
struct DeviceMemPeak {
    size_t base_free = 0;
    DeviceMemPeak() {
        size_t total = 0;
        if (cudaMemGetInfo(&base_free, &total) != cudaSuccess)
            base_free = 0;
    }
    size_t finish() const {
        size_t now_free = 0, total = 0;
        if (base_free == 0) return 0;
        if (cudaMemGetInfo(&now_free, &total) != cudaSuccess) return 0;
        return (now_free < base_free) ? base_free - now_free : 0;
    }
};


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
// Per-stage FFT kernel launchers Shared-memory Stockham FFT (one block per row)
//
// When 2 * fft_len * 8 bytes fits in shared memory
// load the entire row into smem ping-pong buffers and do ALL butterfly stages
// there. 
// ---------------------------------------------------------------------------
// In-place: reads from inout into smem ping-pong, runs all stages, writes back.
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

// Stockham FFT, row-major, in-place, out-of-place, batched, with shared memory
__global__ void stockham_smem_row_mul_inverse_kernel(
    cuFloatComplex* __restrict__ inout,
    const cuFloatComplex* __restrict__ mul,
    int n, int batch)
{
    extern __shared__ cuFloatComplex smem[];
    const int row = blockIdx.x;
    if (row >= batch) return;

    cuFloatComplex* sa = smem, *sb = smem + n;

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
// Bluestein input build kernels: 1-D layout 
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
// Bluestein input build kernels: 2-D coalesced layout
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
// Bluestein finalize kernel — conjugate-symmetric dual write.
// The input matrix is real, so F[rows-u][v] = conj(F[u][(cols-v) mod cols]).
// Only rows u in [0, rows/2] are computed by the column transform; this
// kernel writes both F[u][v] (as the plain finalize does) and the mirror row
// F[(rows-u)%rows][v] = conj(conv[u][(cols-v)%cols] * chirp[(cols-v)%cols]).
// The conv buffer has width fft_len >= 2*cols-1, so index w = (cols-v)%cols
// (< cols) is always valid Bluestein output.
// ---------------------------------------------------------------------------
__global__ void bluestein_finalize_sym_kernel(
    const cuFloatComplex* __restrict__ conv,
    const cuFloatComplex* __restrict__ chirp,
    cuFloatComplex* __restrict__ out,
    int rows, int cols, int output_stride, int output_cols,
    int fft_len, int u_base, int tile_rows)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u_local = blockIdx.y * blockDim.y + threadIdx.y;
    const int u = u_base + u_local;
    const int rows_half = rows / 2 + 1;
    if (v >= output_cols || u_local >= tile_rows || u >= rows_half) return;

    cuFloatComplex y = conv[(size_t)u_local * fft_len + v];
    cuFloatComplex q = chirp[v];
    out[(size_t)u * output_stride + v] =
        make_cuFloatComplex(fmaf(y.x, q.x, -y.y * q.y),
                            fmaf(y.x, q.y,  y.y * q.x));

    const int m = (rows - u) % rows;          // mirror row (u==0 -> 0)
    if (m != u) {
        const int w = (cols - v) % cols;      // mirror column, always < cols
        cuFloatComplex ym = conv[(size_t)u_local * fft_len + w];
        cuFloatComplex qm = chirp[w];
        // F[u][w] = ym * qm; mirror entry is conj(F[u][w]).
        out[(size_t)m * output_stride + v] =
            make_cuFloatComplex( fmaf(ym.x, qm.x, -ym.y * qm.y),
                                -fmaf(ym.x, qm.y,  ym.y * qm.x));
    }
}

// ---------------------------------------------------------------------------
// Stockham FFT host helpers
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
// after performing the FFT.
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


// Returns d_in (in-place).  When fft_len > SMEM_MAX_FFT_N the call transparently
// falls back to the per-stage stream variant so callers need not branch.
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

    DeviceMemPeak peak_tracker;
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
                // smem row kernel — entire FFT in shared memory 
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
    const size_t mem_bytes = peak_tracker.finish();

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

    return {d_out, output_stride, ms, mem_bytes, {}};
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

    DeviceMemPeak peak_tracker;
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
    const size_t mem_bytes = peak_tracker.finish();

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

    return {nullptr, output_stride, ms, mem_bytes, std::move(h_output)};
}

SparseFFTResult sparse_fft_csc_bluestein_cufft_streaming(const COOMatrix& coo, int u_tile) {
    return sparse_fft_csc_stockham_streaming(coo, u_tile, 3);
}


// ---------------------------------------------------------------------------
// CUDA Graph variant + Bluestein
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

    DeviceMemPeak peak_tracker;
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
    const size_t mem_bytes = peak_tracker.finish();

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

    return {nullptr, output_stride, ms, mem_bytes, std::move(h_output)};
}


// ===========================================================================
// binary CSC + byte-mask LUT pass-1 + cuFFT Bluestein streaming
//
// Pass-1 cost in the existing cuFFT-streaming path is dominated by sincosf:
// one call per (nonzero, u_local) pair — i.e. nnz × tile_rows × num_tiles.
// This variant exploits the binary-only structure of the input by packing
// each column's row indices into (byte_pos, byte_mask) pairs (8 rows / byte)
// and replacing sincosf-per-row with two table lookups + one complex multiply.
//
// Tables (rebuilt per u-tile):
//   LUT_inner[u_local][mask]    = Σ_{j∈bits(mask)} exp(-2πi·j·u/rows)   (256 entries)
//   byte_phase[u_local][byte]   = exp(-2πi·8·byte·u/rows)                (num_byte_pos entries)
//
// Pass-1 inner: for each (byte_pos, mask) entry of column,
//     accum += byte_phase[u_local][byte_pos] · LUT_inner[u_local][mask]
// All sincosf calls move into the (small) precompute kernels.
// ===========================================================================

// Compressed byte-mask representation of a binary CSC matrix.
struct ByteMaskCSC {
    int n_sparse_cols = 0;
    std::vector<int> sparse_cols;
    std::vector<int> byte_col_ptr;     // size n_sparse_cols + 1
    std::vector<uint16_t> byte_pos;    // size total_bytes
    std::vector<uint8_t>  byte_mask;   // size total_bytes
};

static ByteMaskCSC make_byte_mask_csc(const CompactCSC& csc) {
    ByteMaskCSC bm;
    bm.n_sparse_cols = csc.n_sparse_cols;
    bm.sparse_cols   = csc.sparse_cols;
    bm.byte_col_ptr.assign(bm.n_sparse_cols + 1, 0);

    // For each column: sort row indices, group consecutive rows by byte_pos = row >> 3.
    std::vector<int> sorted_rows;
    for (int col = 0; col < bm.n_sparse_cols; col++) {
        const int start = csc.col_ptr[col];
        const int end   = csc.col_ptr[col + 1];
        sorted_rows.assign(csc.row_idx.begin() + start, csc.row_idx.begin() + end);
        std::sort(sorted_rows.begin(), sorted_rows.end());

        int cur_bp = -1;
        uint8_t cur_mask = 0;
        for (int r : sorted_rows) {
            const int bp  = r >> 3;
            const int bit = r & 7;
            if (bp != cur_bp) {
                if (cur_bp >= 0) {
                    bm.byte_pos.push_back((uint16_t)cur_bp);
                    bm.byte_mask.push_back(cur_mask);
                }
                cur_bp = bp;
                cur_mask = 0;
            }
            cur_mask |= (uint8_t)(1u << bit);
        }
        if (cur_bp >= 0) {
            bm.byte_pos.push_back((uint16_t)cur_bp);
            bm.byte_mask.push_back(cur_mask);
        }
        bm.byte_col_ptr[col + 1] = (int)bm.byte_pos.size();
    }
    return bm;
}

// Precompute LUT_inner[u_local * 256 + mask] for the current u-tile.
// 2-D launch: blockIdx.x sweeps mask (0..255), blockIdx.y / threadIdx.y sweeps u_local.
__global__ void precompute_lut_inner_kernel(
    cuFloatComplex* __restrict__ lut_inner,
    int rows, int u_base, int tile_rows, float inv_rows)
{
    const int mask    = blockIdx.x * blockDim.x + threadIdx.x;
    const int u_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (mask >= 256 || u_local >= tile_rows) return;
    const int u = u_base + u_local;
    if (u >= rows) return;

    float re = 0.f, im = 0.f;
    unsigned m = (unsigned)mask;
    while (m) {
        const int j = __ffs(m) - 1;            // 0..7
        float s, c;
        sincos_twiddle(u, j, rows, inv_rows, &s, &c);
        re += c;
        im += s;
        m &= m - 1;                            // clear lowest set bit
    }
    lut_inner[(size_t)u_local * 256 + mask] = make_cuFloatComplex(re, im);
}

// Precompute byte_phase[u_local * num_byte_pos + byte_pos] = exp(-2πi · 8·byte_pos · u / rows).
__global__ void precompute_byte_phase_kernel(
    cuFloatComplex* __restrict__ byte_phase,
    int rows, int u_base, int tile_rows,
    int num_byte_pos, float inv_rows)
{
    const int byte_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int u_local  = blockIdx.y * blockDim.y + threadIdx.y;
    if (byte_pos >= num_byte_pos || u_local >= tile_rows) return;
    const int u = u_base + u_local;
    if (u >= rows) return;

    float s, c;
    sincos_twiddle(u, 8 * byte_pos, rows, inv_rows, &s, &c);
    byte_phase[(size_t)u_local * num_byte_pos + byte_pos] = make_cuFloatComplex(c, s);
}

// Pass-1 build kernel: byte-mask + LUT, mirror of csc_build_bluestein_input_kernel.
// One block per (col_id, tile-row group); threadIdx.x indexes u_local within CSC_BU.
__global__ void csc_build_bluestein_input_binary_lut_kernel(
    const int*      __restrict__ byte_col_ptr,
    const uint16_t* __restrict__ byte_pos_arr,
    const uint8_t*  __restrict__ byte_mask_arr,
    const int*      __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    const cuFloatComplex* __restrict__ lut_inner,
    const cuFloatComplex* __restrict__ byte_phase,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, int num_byte_pos)
{
    const int col_id  = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;
    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = sparse_cols[col_id];

    const int start = byte_col_ptr[col_id];
    const int end   = byte_col_ptr[col_id + 1];

    const cuFloatComplex* lut_row   = lut_inner  + (size_t)u_local * 256;
    const cuFloatComplex* phase_row = byte_phase + (size_t)u_local * num_byte_pos;

    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        const int      byte_pos = (int)byte_pos_arr[p];
        const unsigned mask     = (unsigned)byte_mask_arr[p];
        if (mask == 0) continue;                         // defensive: should not occur

        const cuFloatComplex inner = lut_row[mask];
        const cuFloatComplex bp    = phase_row[byte_pos];
        // accum += bp * inner  (one complex MAC via fmaf)
        re = fmaf(bp.x, inner.x, fmaf(-bp.y, inner.y, re));
        im = fmaf(bp.x, inner.y, fmaf( bp.y, inner.x, im));
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(fmaf(re, q.x, -im * q.y),
                            fmaf(re, q.y,  im * q.x));
}

__global__ void csc_build_bluestein_input_binary_lut_packed_kernel(
    const int*      __restrict__ byte_col_ptr,
    const uint16_t* __restrict__ byte_pos_arr,
    const uint8_t*  __restrict__ byte_mask_arr,
    const uint16_t* __restrict__ sparse_cols,
    const cuFloatComplex* __restrict__ chirp,
    const cuFloatComplex* __restrict__ lut_inner,
    const cuFloatComplex* __restrict__ byte_phase,
    cuFloatComplex* __restrict__ signal,
    int rows, int n_sparse_cols, int fft_len,
    int u_base, int tile_rows, int num_byte_pos)
{
    const int col_id  = blockIdx.x;
    const int u_local = blockIdx.y * CSC_BU + threadIdx.x;
    if (col_id >= n_sparse_cols || u_local >= tile_rows) return;
    const int u = u_base + u_local;
    if (u >= rows) return;
    const int c = (int)sparse_cols[col_id];

    const int start = byte_col_ptr[col_id];
    const int end   = byte_col_ptr[col_id + 1];

    const cuFloatComplex* lut_row   = lut_inner  + (size_t)u_local * 256;
    const cuFloatComplex* phase_row = byte_phase + (size_t)u_local * num_byte_pos;

    float re = 0.f, im = 0.f;
    for (int p = start; p < end; p++) {
        const int      byte_pos = (int)byte_pos_arr[p];
        const unsigned mask     = (unsigned)byte_mask_arr[p];
        if (mask == 0) continue;

        const cuFloatComplex inner = lut_row[mask];
        const cuFloatComplex bp    = phase_row[byte_pos];
        re = fmaf(bp.x, inner.x, fmaf(-bp.y, inner.y, re));
        im = fmaf(bp.x, inner.y, fmaf( bp.y, inner.x, im));
    }

    cuFloatComplex q = chirp[c];
    signal[(size_t)u_local * fft_len + c] =
        make_cuFloatComplex(fmaf(re, q.x, -im * q.y),
                            fmaf(re, q.y,  im * q.x));
}

SparseFFTResult sparse_fft_csc_bluestein_binary_lut(const COOMatrix& coo, int u_tile) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols   = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len       = next_power_of_two(2 * cols - 1);
    const int num_byte_pos  = (rows + 7) / 8;
    const bool use_packed   = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    DeviceMemPeak peak_tracker;
    // Host-side preprocessing: CompactCSC → ByteMaskCSC.
    CompactCSC csc = make_compact_csc(coo);
    ByteMaskCSC bm = make_byte_mask_csc(csc);
    const int n_sp        = bm.n_sparse_cols;
    const int total_bytes = (int)bm.byte_pos.size();

    // Device byte-mask CSC.
    int      *d_byte_col_ptr = nullptr;
    uint16_t *d_byte_pos     = nullptr;
    uint8_t  *d_byte_mask    = nullptr;
    int      *d_sp_cols      = nullptr;
    uint16_t *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_byte_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_byte_pos,     total_bytes * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_byte_mask,    total_bytes * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_byte_col_ptr, bm.byte_col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_byte_pos, bm.byte_pos.data(),
                          total_bytes * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_byte_mask, bm.byte_mask.data(),
                          total_bytes * sizeof(uint8_t), cudaMemcpyHostToDevice));

    if (use_packed) {
        std::vector<uint16_t> sp_packed = pack_cols_u16(bm.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_packed.data(),
                              n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, bm.sparse_cols.data(),
                              n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Bluestein chirp + B FFT (one-shot cuFFT plan).
    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b     = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp,  (size_t)cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft,  (size_t)fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(), (size_t)cols    * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_fft, h_b.data(),     (size_t)fft_len * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));
    {
        int fl = fft_len;
        cufftHandle plan_b;
        CUFFT_CHECK(cufftPlanMany(&plan_b, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, 1));
        CUFFT_CHECK(cufftExecC2C(plan_b,
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 CUFFT_FORWARD));
        CUFFT_CHECK(cufftDestroy(plan_b));
    }

    // Per-tile scratch and output ring buffers.
    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr;
    cuFloatComplex *d_lut_inner = nullptr, *d_byte_phase = nullptr;
    cuFloatComplex *d_out_chunk[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_signal,
        (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_lut_inner,
        (size_t)max_tile_rows * 256 * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_byte_phase,
        (size_t)max_tile_rows * num_byte_pos * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk[0],
        (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out_chunk[1],
        (size_t)max_tile_rows * output_stride * sizeof(cuFloatComplex)));

    cuFloatComplex* h_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pinned,
        (size_t)rows * output_stride * sizeof(cuFloatComplex), cudaHostAllocDefault));

    cudaStream_t compute_stream, copy_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&copy_stream));

    // cuFFT tile plans.
    cufftHandle plan_cufft = 0, plan_cufft_last = 0;
    size_t cufft_workspace = 0;
    {
        int fl = fft_len;
        const int full_tile = max_tile_rows;
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

    cudaEvent_t compute_done[2], d2h_done[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaEventCreate(&compute_done[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&d2h_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(d2h_done[i], compute_stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaEventSynchronize(t0));

    const float inv_rows = 1.f / (float)rows;

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const int buf = (u_base / u_tile) & 1;
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);

        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, d2h_done[buf]));
        CUDA_CHECK(cudaMemsetAsync(d_signal, 0, signal_bytes, compute_stream));

        // Precompute per-tile LUTs.
        {
            dim3 lut_block(32, 8);
            dim3 lut_grid((256 + lut_block.x - 1) / lut_block.x,
                          (tile_rows + lut_block.y - 1) / lut_block.y);
            precompute_lut_inner_kernel<<<lut_grid, lut_block, 0, compute_stream>>>(
                d_lut_inner, rows, u_base, tile_rows, inv_rows);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            dim3 ph_block(64, 4);
            dim3 ph_grid((num_byte_pos + ph_block.x - 1) / ph_block.x,
                         (tile_rows    + ph_block.y - 1) / ph_block.y);
            precompute_byte_phase_kernel<<<ph_grid, ph_block, 0, compute_stream>>>(
                d_byte_phase, rows, u_base, tile_rows, num_byte_pos, inv_rows);
            CUDA_CHECK(cudaGetLastError());
        }

        // Pass 1: byte-mask + LUT build.
        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_binary_lut_packed_kernel<<<p1_grid, p1_block, 0, compute_stream>>>(
                d_byte_col_ptr, d_byte_pos, d_byte_mask, d_sp_cols_packed,
                d_chirp, d_lut_inner, d_byte_phase, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, num_byte_pos);
        } else {
            csc_build_bluestein_input_binary_lut_kernel<<<p1_grid, p1_block, 0, compute_stream>>>(
                d_byte_col_ptr, d_byte_pos, d_byte_mask, d_sp_cols,
                d_chirp, d_lut_inner, d_byte_phase, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, num_byte_pos);
        }
        CUDA_CHECK(cudaGetLastError());

        // Bluestein convolution: cuFFT FWD → pointwise mul → cuFFT INV → scale.
        cufftHandle tile_plan =
            (plan_cufft_last && tile_rows != max_tile_rows)
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

        // Finalize: chirp_conj multiply → d_out_chunk[buf].
        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block, 0, compute_stream>>>(
            d_signal, d_chirp, d_out_chunk[buf],
            tile_rows, output_stride, output_cols, fft_len, 0, tile_rows);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(compute_done[buf], compute_stream));
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
    const size_t mem_bytes = peak_tracker.finish();

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
    cudaFree(d_byte_col_ptr);
    cudaFree(d_byte_pos);
    cudaFree(d_byte_mask);
    cudaFree(d_sp_cols);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_lut_inner);
    cudaFree(d_byte_phase);
    cudaFree(d_out_chunk[0]);
    cudaFree(d_out_chunk[1]);

    return {nullptr, output_stride, ms, mem_bytes, std::move(h_output)};
}


// ===========================================================================
// : binary CSC + byte-mask LUT pass-1 + Stockham smem (NON-streaming)
//
// Same pass-1 as sparse_fft_csc_bluestein_binary_lut, but Bluestein convolution
// runs through run_fft_power2_stockham_smem_stream (which auto-falls-back to the
// per-stage Stockham loop when fft_len > SMEM_MAX_FFT_N) and the fused
// stockham_smem_row_mul_inverse_kernel — no cuFFT anywhere.  Allocates the full
// d_out = rows × output_stride on device.
//
// ===========================================================================

static bool stockham_binary_breakdown_enabled() {
    static bool init = false, val = false;
    if (!init) {
        const char* env = std::getenv("CUSPFFT_BREAKDOWN");
        val = env && env[0] && env[0] != '0';
        init = true;
    }
    return val;
}

SparseFFTResult sparse_fft_csc_stockham_binary_smem(const COOMatrix& coo, int u_tile) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols   = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len       = next_power_of_two(2 * cols - 1);
    const int num_byte_pos  = (rows + 7) / 8;
    const bool use_packed   = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    DeviceMemPeak peak_tracker;
    const bool breakdown = stockham_binary_breakdown_enabled();

    // ----- Host preprocessing: CSC + byte-mask CSC -----
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    CompactCSC csc = make_compact_csc(coo);
    ByteMaskCSC bm = make_byte_mask_csc(csc);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    const double prep_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
    const int n_sp        = bm.n_sparse_cols;
    const int total_bytes = (int)bm.byte_pos.size();

    // ----- Device buffers -----
    int      *d_byte_col_ptr = nullptr;
    uint16_t *d_byte_pos     = nullptr;
    uint8_t  *d_byte_mask    = nullptr;
    int      *d_sp_cols      = nullptr;
    uint16_t *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_byte_col_ptr, (n_sp + 1)  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_byte_pos,     total_bytes * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_byte_mask,    total_bytes * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_byte_col_ptr, bm.byte_col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_byte_pos, bm.byte_pos.data(),
                          total_bytes * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_byte_mask, bm.byte_mask.data(),
                          total_bytes * sizeof(uint8_t), cudaMemcpyHostToDevice));
    if (use_packed) {
        std::vector<uint16_t> sp_packed = pack_cols_u16(bm.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, n_sp * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp_packed.data(),
                              n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_sp_cols, n_sp * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, bm.sparse_cols.data(),
                              n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    // ----- Bluestein chirp + B FFT (precomputed via Stockham, NOT cuFFT) -----
    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b     = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp, (size_t)cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft, (size_t)fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(),
                          (size_t)cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr, *d_work = nullptr, *d_out = nullptr;
    cuFloatComplex *d_lut_inner = nullptr, *d_byte_phase = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal,
        (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_work,
        (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out,
        (size_t)rows * output_stride * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_lut_inner,
        (size_t)max_tile_rows * 256 * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_byte_phase,
        (size_t)max_tile_rows * num_byte_pos * sizeof(cuFloatComplex)));

    // Stage h_b through d_work for the one-time d_b_fft precompute.
    CUDA_CHECK(cudaMemcpy(d_work, h_b.data(),
                          (size_t)fft_len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    run_fft_power2_stockham(d_work, d_b_fft, fft_len, 1, false);

    // ----- Tile loop -----
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Per-phase events (only used when breakdown == true)
    cudaEvent_t e_lut0 = 0, e_lut1 = 0, e_p1_1 = 0, e_fft_1 = 0, e_fin_1 = 0;
    float t_lut = 0.f, t_p1 = 0.f, t_fft = 0.f, t_finalize = 0.f;
    if (breakdown) {
        CUDA_CHECK(cudaEventCreate(&e_lut0));
        CUDA_CHECK(cudaEventCreate(&e_lut1));
        CUDA_CHECK(cudaEventCreate(&e_p1_1));
        CUDA_CHECK(cudaEventCreate(&e_fft_1));
        CUDA_CHECK(cudaEventCreate(&e_fin_1));
    }

    CUDA_CHECK(cudaEventRecord(t0));

    const float inv_rows = 1.f / (float)rows;

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);
        CUDA_CHECK(cudaMemset(d_signal, 0, signal_bytes));

        if (breakdown) CUDA_CHECK(cudaEventRecord(e_lut0));

        // ---- Per-tile LUT precompute ----
        {
            dim3 lut_block(32, 8);
            dim3 lut_grid((256 + lut_block.x - 1) / lut_block.x,
                          (tile_rows + lut_block.y - 1) / lut_block.y);
            precompute_lut_inner_kernel<<<lut_grid, lut_block>>>(
                d_lut_inner, rows, u_base, tile_rows, inv_rows);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            dim3 ph_block(64, 4);
            dim3 ph_grid((num_byte_pos + ph_block.x - 1) / ph_block.x,
                         (tile_rows    + ph_block.y - 1) / ph_block.y);
            precompute_byte_phase_kernel<<<ph_grid, ph_block>>>(
                d_byte_phase, rows, u_base, tile_rows, num_byte_pos, inv_rows);
            CUDA_CHECK(cudaGetLastError());
        }

        if (breakdown) CUDA_CHECK(cudaEventRecord(e_lut1));

        // ---- Pass 1: byte-mask + LUT build ----
        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_binary_lut_packed_kernel<<<p1_grid, p1_block>>>(
                d_byte_col_ptr, d_byte_pos, d_byte_mask, d_sp_cols_packed,
                d_chirp, d_lut_inner, d_byte_phase, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, num_byte_pos);
        } else {
            csc_build_bluestein_input_binary_lut_kernel<<<p1_grid, p1_block>>>(
                d_byte_col_ptr, d_byte_pos, d_byte_mask, d_sp_cols,
                d_chirp, d_lut_inner, d_byte_phase, d_signal,
                rows, n_sp, fft_len, u_base, tile_rows, num_byte_pos);
        }
        CUDA_CHECK(cudaGetLastError());

        if (breakdown) CUDA_CHECK(cudaEventRecord(e_p1_1));

        // ---- Stockham smem FFT + pointwise mul + inverse FFT ----
        // Auto-falls-back to run_fft_power2_stockham_stream when fft_len > SMEM_MAX_FFT_N.
        cuFloatComplex* fwd = run_fft_power2_stockham_smem_stream(
            d_signal, d_work, fft_len, tile_rows, false, 0);
        cuFloatComplex* fwd_other = (fwd == d_signal) ? d_work : d_signal;
        cuFloatComplex* conv_result = run_fft_power2_stockham_smem_mul_inverse_stream(
            fwd, fwd_other, d_b_fft, fft_len, tile_rows, 0);

        if (breakdown) CUDA_CHECK(cudaEventRecord(e_fft_1));

        // ---- Finalize: chirp_conj × conv → d_out region [u_base..u_base+tile_rows) ----
        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block>>>(
            conv_result, d_chirp, d_out,
            rows, output_stride, output_cols, fft_len, u_base, tile_rows);
        CUDA_CHECK(cudaGetLastError());

        if (breakdown) {
            CUDA_CHECK(cudaEventRecord(e_fin_1));
            CUDA_CHECK(cudaEventSynchronize(e_fin_1));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e_lut0, e_lut1)); t_lut      += ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e_lut1, e_p1_1)); t_p1       += ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e_p1_1, e_fft_1)); t_fft     += ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e_fft_1, e_fin_1)); t_finalize += ms;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    const size_t mem_bytes = peak_tracker.finish();

    if (breakdown) {
        printf("    [breakdown] CSC+ByteMask host: %8.3f ms\n", prep_ms);
        printf("    [breakdown] LUT precompute    : %8.3f ms\n", t_lut);
        printf("    [breakdown] Pass-1 byte-mask  : %8.3f ms\n", t_p1);
        printf("    [breakdown] Stockham FFT+mul  : %8.3f ms\n", t_fft);
        printf("    [breakdown] Finalize          : %8.3f ms\n", t_finalize);
        printf("    [breakdown] Total (loop only) : %8.3f ms\n", ms);
        cudaEventDestroy(e_lut0); cudaEventDestroy(e_lut1);
        cudaEventDestroy(e_p1_1); cudaEventDestroy(e_fft_1); cudaEventDestroy(e_fin_1);
    }

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_byte_col_ptr);
    cudaFree(d_byte_pos);
    cudaFree(d_byte_mask);
    cudaFree(d_sp_cols);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_work);
    cudaFree(d_lut_inner);
    cudaFree(d_byte_phase);

    return {d_out, output_stride, ms, mem_bytes, {}};
}


// ===========================================================================
// : mixed-radix {2, 3} Stockham FFT for Bluestein convolution.
// ===========================================================================

// Smallest 3-smooth integer ≥ n (factors only from {2, 3}).
static int next_3_smooth(int n) {
    for (int m = n; ; m++) {
        int x = m;
        while (x % 2 == 0) x /= 2;
        while (x % 3 == 0) x /= 3;
        if (x == 1) return m;
    }
}

// Smallest 7-smooth integer ≥ n (factors only from {2, 3, 5, 7}).  cuFFT
// efficiently handles any 7-smooth size, so for the Bluestein convolution
// `fft_len = next_7_smooth(2·cols−1)` is much tighter than `next_pow2`.
static int next_7_smooth(int n) {
    for (int m = n; ; ++m) {
        int x = m;
        for (int p : {2, 3, 5, 7})
            while (x % p == 0) x /= p;
        if (x == 1) return m;
    }
}

// Decompose 3-smooth n into a sequence of radices.  Pairs of 3's are emitted
// as a single radix-9 stage (one fused kernel, one global-memory roundtrip
// instead of two).  Trailing odd factor of 3, if any, becomes a radix-3 stage.
// Then all 2's.
static std::vector<int> factor_3_smooth(int n) {
    std::vector<int> factors;
    int count_3 = 0;
    while (n % 3 == 0) { count_3++; n /= 3; }
    int count_9 = count_3 / 2;
    int leftover_3 = count_3 % 2;
    for (int i = 0; i < count_9; i++)    factors.push_back(9);
    if (leftover_3)                      factors.push_back(3);
    while (n % 2 == 0) { factors.push_back(2); n /= 2; }
    if (n != 1) throw std::runtime_error("factor_3_smooth: not 3-smooth");
    return factors;
}

// Stockham radix-3 stage. Layout matches stockham_stage_kernel:
//   in[s] at j·(3m)+s·m+k for s ∈ {0,1,2}
//   out[s] at j·m+k + s·(n/3)
// Order of operations: 3-point DFT → twiddle outputs s=1,2 by exp(±2πi·s·k/(3m)).
__global__ void stockham_radix3_stage_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int n, int /*l_pre*/, int m, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (n / 3) * batch;
    if (idx >= total) return;

    const int b = idx / (n / 3);
    const int t = idx - b * (n / 3);
    const int j = t / m;
    const int k = t - j * m;

    const int in0 = j * (3 * m) + k;
    const int in1 = in0 + m;
    const int in2 = in0 + 2 * m;

    const cuFloatComplex x0 = in[(size_t)b * n + in0];
    const cuFloatComplex x1 = in[(size_t)b * n + in1];
    const cuFloatComplex x2 = in[(size_t)b * n + in2];

    // 3-point DFT (no inner twiddles). W3 = exp(-2πi/3) for forward.
    // y[0] = x0 + x1 + x2
    // y[1] = x0 + W3   * x1 + W3^2 * x2  (W3^2 = conj(W3))
    // y[2] = x0 + W3^2 * x1 + W3   * x2
    constexpr float W3_re = -0.5f;
    const     float W3_im = inverse ? 0.86602540378443864676f : -0.86602540378443864676f;

    const float s12_x = x1.x + x2.x;
    const float s12_y = x1.y + x2.y;
    const float d12_x = x1.x - x2.x;
    const float d12_y = x1.y - x2.y;

    cuFloatComplex y0 = make_cuFloatComplex(x0.x + s12_x, x0.y + s12_y);
    cuFloatComplex y1 = make_cuFloatComplex(
        x0.x + W3_re * s12_x - W3_im * d12_y,
        x0.y + W3_re * s12_y + W3_im * d12_x);
    cuFloatComplex y2 = make_cuFloatComplex(
        x0.x + W3_re * s12_x + W3_im * d12_y,
        x0.y + W3_re * s12_y - W3_im * d12_x);

    // Output twiddles: y[s] *= exp(±2πi·s·k/(3m))   (forward: minus sign).
    if (k != 0) {
        const float two_pi = 6.28318530717958647f;
        const float ang1 = (inverse ? two_pi : -two_pi) * (float)k / (float)(3 * m);
        const float ang2 = ang1 + ang1;
        float s1, c1, s2, c2;
        __sincosf(ang1, &s1, &c1);
        __sincosf(ang2, &s2, &c2);
        const cuFloatComplex y1_old = y1;
        const cuFloatComplex y2_old = y2;
        y1 = make_cuFloatComplex(fmaf(y1_old.x, c1, -y1_old.y * s1),
                                 fmaf(y1_old.x, s1,  y1_old.y * c1));
        y2 = make_cuFloatComplex(fmaf(y2_old.x, c2, -y2_old.y * s2),
                                 fmaf(y2_old.x, s2,  y2_old.y * c2));
    }

    const int n3 = n / 3;
    const int out0_idx = j * m + k;
    out[(size_t)b * n + out0_idx]          = y0;
    out[(size_t)b * n + out0_idx + n3]     = y1;
    out[(size_t)b * n + out0_idx + 2 * n3] = y2;
}

// Stockham radix-9 stage = two fused radix-3 stages in one kernel.
// Reads 9 inputs, computes a 9-point DFT via 3×3 Kronecker decomposition with
// inter-stage twiddle absorbed in registers, applies outer twiddle, writes 9
// outputs.  Replaces 2 global-memory roundtrips with 1 — the primary lever
// against the memory-bandwidth ceiling identified in the ncu profile.
//
// l_pre = product of previous radices.  m = n / (9 · l_pre) = m_fused.
// Stage A's m_A = 3·m, stage B's m_B = m.  Stage A operates on sub_A ∈ [0,3)
// (stride 3·m), stage B on sub_B ∈ [0,3) (stride m).  Per-thread layout:
//   inputs  in[s] at j·9·m + s·m + k   for s = 3·sub_A + b ∈ [0,9).
//   outputs out at c'·(n/3) + c·(n/9) + j·m + k   for c, c' ∈ [0,3).
__global__ void stockham_radix9_stage_kernel(
    const cuFloatComplex* __restrict__ in,
    cuFloatComplex* __restrict__ out,
    int n, int l_pre, int m, int batch, int inverse)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (n / 9) * batch;
    if (idx >= total) return;

    const int b_idx = idx / (n / 9);
    const int t = idx - b_idx * (n / 9);
    const int j = t / m;
    const int k = t - j * m;

    constexpr float W3_re = -0.5f;
    const     float W3_im = inverse ? 0.86602540378443864676f : -0.86602540378443864676f;
    const     float two_pi = 6.28318530717958647f;
    const     float sign = inverse ? two_pi : -two_pi;

    const size_t row = (size_t)b_idx * n;
    const int base = j * 9 * m + k;

    // Read 9 inputs.  Index s = 3·sub_A + b.
    cuFloatComplex x[9];
    #pragma unroll
    for (int s = 0; s < 9; s++) x[s] = in[row + base + s * m];

    // Stage A: for each b ∈ [0,3), 3-point DFT over sub_A ∈ [0,3).
    // y_preA[b][c] = x[b] + W3^c · x[b+3] + W3^(2c) · x[b+6].
    // Then twiddle T1[b][c] = exp(sign · c · (b·m + k) / (9m))
    //                       = W9^(c·b) · exp(sign · c · k / (9m)).
    cuFloatComplex y[3][3];
    #pragma unroll
    for (int b = 0; b < 3; b++) {
        const cuFloatComplex p0 = x[b];
        const cuFloatComplex p1 = x[b + 3];
        const cuFloatComplex p2 = x[b + 6];
        const float s12_x = p1.x + p2.x;
        const float s12_y = p1.y + p2.y;
        const float d12_x = p1.x - p2.x;
        const float d12_y = p1.y - p2.y;
        cuFloatComplex y0 = make_cuFloatComplex(p0.x + s12_x, p0.y + s12_y);
        cuFloatComplex y1 = make_cuFloatComplex(
            p0.x + W3_re * s12_x - W3_im * d12_y,
            p0.y + W3_re * s12_y + W3_im * d12_x);
        cuFloatComplex y2 = make_cuFloatComplex(
            p0.x + W3_re * s12_x + W3_im * d12_y,
            p0.y + W3_re * s12_y - W3_im * d12_x);

        // Apply T1[b][c=1,2].  Identity when (b·m + k) == 0.
        const int kA = b * m + k;
        if (kA != 0) {
            const float ang1 = sign * (float)kA / (float)(9 * m);
            const float ang2 = ang1 + ang1;
            float s1, c1, s2, c2;
            __sincosf(ang1, &s1, &c1);
            __sincosf(ang2, &s2, &c2);
            const cuFloatComplex y1_old = y1;
            const cuFloatComplex y2_old = y2;
            y1 = make_cuFloatComplex(fmaf(y1_old.x, c1, -y1_old.y * s1),
                                     fmaf(y1_old.x, s1,  y1_old.y * c1));
            y2 = make_cuFloatComplex(fmaf(y2_old.x, c2, -y2_old.y * s2),
                                     fmaf(y2_old.x, s2,  y2_old.y * c2));
        }
        y[b][0] = y0;
        y[b][1] = y1;
        y[b][2] = y2;
    }

    // Stage B: for each c ∈ [0,3), 3-point DFT over sub_B (= b) ∈ [0,3).
    // w_preB[c][c'] = y[0][c] + W3^c' · y[1][c] + W3^(2c') · y[2][c].
    // Then twiddle T2[c'] = exp(sign · c' · k / (3m)).  Identity when k == 0.
    const int n9 = n / 9;
    const int n3 = n / 3;
    const int out_base = j * m + k;

    // Precompute T2 once (independent of c)
    float t2_s1 = 0.f, t2_c1 = 1.f, t2_s2 = 0.f, t2_c2 = 1.f;
    const bool apply_t2 = (k != 0);
    if (apply_t2) {
        const float ang1 = sign * (float)k / (float)(3 * m);
        const float ang2 = ang1 + ang1;
        __sincosf(ang1, &t2_s1, &t2_c1);
        __sincosf(ang2, &t2_s2, &t2_c2);
    }

    #pragma unroll
    for (int c = 0; c < 3; c++) {
        const cuFloatComplex p0 = y[0][c];
        const cuFloatComplex p1 = y[1][c];
        const cuFloatComplex p2 = y[2][c];
        const float s12_x = p1.x + p2.x;
        const float s12_y = p1.y + p2.y;
        const float d12_x = p1.x - p2.x;
        const float d12_y = p1.y - p2.y;
        cuFloatComplex w0 = make_cuFloatComplex(p0.x + s12_x, p0.y + s12_y);
        cuFloatComplex w1 = make_cuFloatComplex(
            p0.x + W3_re * s12_x - W3_im * d12_y,
            p0.y + W3_re * s12_y + W3_im * d12_x);
        cuFloatComplex w2 = make_cuFloatComplex(
            p0.x + W3_re * s12_x + W3_im * d12_y,
            p0.y + W3_re * s12_y - W3_im * d12_x);

        if (apply_t2) {
            const cuFloatComplex w1_old = w1;
            const cuFloatComplex w2_old = w2;
            w1 = make_cuFloatComplex(fmaf(w1_old.x, t2_c1, -w1_old.y * t2_s1),
                                     fmaf(w1_old.x, t2_s1,  w1_old.y * t2_c1));
            w2 = make_cuFloatComplex(fmaf(w2_old.x, t2_c2, -w2_old.y * t2_s2),
                                     fmaf(w2_old.x, t2_s2,  w2_old.y * t2_c2));
        }

        out[row + 0 * n3 + c * n9 + out_base] = w0;
        out[row + 1 * n3 + c * n9 + out_base] = w1;
        out[row + 2 * n3 + c * n9 + out_base] = w2;
    }
}

// Mixed-radix {2,3,9} Stockham FFT host driver.
// `factors` lists radices in stage order (radix-9 first, then any leftover
// radix-3, then radix-2).  Returns pointer to whichever ping-pong buffer
// holds the final result.
static cuFloatComplex* run_fft_mixed_23(
    cuFloatComplex* d_in, cuFloatComplex* d_tmp,
    int n, const std::vector<int>& factors, int batch, bool inverse,
    cudaStream_t stream)
{
    cuFloatComplex* src = d_in;
    cuFloatComplex* dst = d_tmp;
    int l = 1;

    for (int radix : factors) {
        const int m = n / (l * radix);
        const int total = (n / radix) * batch;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        if (radix == 2) {
            stockham_stage_kernel<<<blocks, threads, 0, stream>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
        } else if (radix == 3) {
            stockham_radix3_stage_kernel<<<blocks, threads, 0, stream>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
        } else if (radix == 9) {
            stockham_radix9_stage_kernel<<<blocks, threads, 0, stream>>>(
                src, dst, n, l, m, batch, inverse ? 1 : 0);
        } else {
            throw std::runtime_error("run_fft_mixed_23: unsupported radix");
        }
        CUDA_CHECK(cudaGetLastError());
        l *= radix;
        cuFloatComplex* tmp = src; src = dst; dst = tmp;
    }

    if (inverse) {
        const int total = n * batch;
        const int threads = 256;
        scale_complex_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
            src, total, 1.f / (float)n);
        CUDA_CHECK(cudaGetLastError());
    }
    return src;
}

SparseFFTResult sparse_fft_csc_bluestein_mixed_radix(const COOMatrix& coo, int u_tile) {
    const auto _pre0 = std::chrono::steady_clock::now();
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols   = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len       = next_3_smooth(2 * cols - 1);
    const std::vector<int> factors = factor_3_smooth(fft_len);
    const bool use_packed   = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");
    // Adapt tile size to fft_len so each (tile_rows × fft_len) batch lands in
    // the ~1.5M complex-element sweet spot. Two concurrent streams at this
    // batch keep SM/HBM near saturation without contention. Caps at the
    // user-supplied u_tile (so explicit --csc-tile is respected); floors at 8.
    const int eff_tile = std::min(u_tile, std::max(8, 1500000 / fft_len));

    DeviceMemPeak peak_tracker;
    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (use_packed) {
        std::vector<uint16_t> ri = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, (size_t)coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, (size_t)n_sp    * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, ri.data(),
                              (size_t)coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp.data(),
                              (size_t)n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, (size_t)coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, (size_t)n_sp    * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              (size_t)coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              (size_t)n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b     = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp, (size_t)cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft, (size_t)fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(),
                          (size_t)cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    const int max_tile_rows = std::min(eff_tile, rows);
    // Double-buffered tile pipeline: two d_signal/d_work slabs and two streams so
    // tile N's build/FFT/multiply/IFFT runs on stream[buf] while tile (N-1)'s
    // finalize/output write completes on the opposite stream. d_out stays single-
    // buffered — tiles write disjoint row stripes [u_base, u_base+tile_rows).
    cuFloatComplex *d_signal[2] = {nullptr, nullptr};
    cuFloatComplex *d_work[2]   = {nullptr, nullptr};
    cuFloatComplex *d_out       = nullptr;
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&d_signal[i],
            (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
        CUDA_CHECK(cudaMalloc(&d_work[i],
            (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    }
    CUDA_CHECK(cudaMalloc(&d_out,
        (size_t)rows * output_stride * sizeof(cuFloatComplex)));

    // Precompute d_b_fft via mixed-radix FFT (no cuFFT). One-shot setup on the
    // default stream using slab 0; loop allocations sync below before reuse.
    CUDA_CHECK(cudaMemcpy(d_signal[0], h_b.data(),
                          (size_t)fft_len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    {
        cuFloatComplex* res = run_fft_mixed_23(d_signal[0], d_work[0], fft_len, factors, 1, false, 0);
        if (res != d_b_fft)
            CUDA_CHECK(cudaMemcpy(d_b_fft, res, (size_t)fft_len * sizeof(cuFloatComplex),
                                  cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Finish all setup GPU work (d_b_fft precompute, H2D) so it is charged to
    // preprocessing, not to the kernel timer.
    CUDA_CHECK(cudaDeviceSynchronize());
    const float preprocess_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - _pre0).count();

    CUDA_CHECK(cudaEventRecord(t0));

    for (int u_base = 0; u_base < rows; u_base += eff_tile) {
        const int tile_rows = std::min(eff_tile, rows - u_base);
        const int buf = (u_base / eff_tile) & 1;
        cudaStream_t stream = streams[buf];
        cuFloatComplex* sig = d_signal[buf];
        cuFloatComplex* wrk = d_work[buf];
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);
        CUDA_CHECK(cudaMemsetAsync(sig, 0, signal_bytes, stream));

        // Pass 1: standard binary CSC build (sincosf-per-nonzero, packed indices).
        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, sig,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_bluestein_input_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx, d_sp_cols, d_chirp, sig,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        // Forward FFT (mixed-radix).
        cuFloatComplex* fwd = run_fft_mixed_23(
            sig, wrk, fft_len, factors, tile_rows, false, stream);
        cuFloatComplex* fwd_other = (fwd == sig) ? wrk : sig;

        // Pointwise multiply by d_b_fft.
        const int total_elems = tile_rows * fft_len;
        pointwise_mul_batched_kernel<<<(total_elems + 255) / 256, 256, 0, stream>>>(
            fwd, d_b_fft, fft_len, tile_rows);
        CUDA_CHECK(cudaGetLastError());

        // Inverse FFT (mixed-radix). Result lands in fwd_other or fwd depending on parity;
        // run_fft_mixed_23 returns the right pointer.
        cuFloatComplex* conv_result = run_fft_mixed_23(
            fwd, fwd_other, fft_len, factors, tile_rows, true, stream);

        // Finalize: multiply by chirp_conj, write into d_out.
        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block, 0, stream>>>(
            conv_result, d_chirp, d_out,
            rows, output_stride, output_cols, fft_len, u_base, tile_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    const size_t mem_bytes = peak_tracker.finish();

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal[0]);
    cudaFree(d_signal[1]);
    cudaFree(d_work[0]);
    cudaFree(d_work[1]);

    return {d_out, output_stride, ms, mem_bytes, {}, preprocess_ms};
}


// ===========================================================================
// CUDA Graph wrapper around the mixed-radix Bluestein algorithm.
// Captures the entire tile loop into a single graph, then replays it as one
// runtime call — cuts kernel-launch overhead from ~5 µs × N down to one launch.
// Non-streaming: writes directly to full d_out, identical output to
// sparse_fft_csc_bluestein_mixed_radix.
// ===========================================================================
SparseFFTResult sparse_fft_csc_bluestein_mixed_radix_graph(const COOMatrix& coo, int u_tile) {
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols   = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len       = next_3_smooth(2 * cols - 1);
    const std::vector<int> factors = factor_3_smooth(fft_len);
    const bool use_packed   = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");

    DeviceMemPeak peak_tracker;
    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (use_packed) {
        std::vector<uint16_t> ri = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, (size_t)coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, (size_t)n_sp    * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, ri.data(),
                              (size_t)coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp.data(),
                              (size_t)n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, (size_t)coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, (size_t)n_sp    * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              (size_t)coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              (size_t)n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b     = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp, (size_t)cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft, (size_t)fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(),
                          (size_t)cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    const int max_tile_rows = std::min(u_tile, rows);
    cuFloatComplex *d_signal = nullptr, *d_work = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal,
        (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_work,
        (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_out,
        (size_t)rows * output_stride * sizeof(cuFloatComplex)));

    // Precompute d_b_fft via mixed-radix FFT on null stream (one-shot, before capture).
    CUDA_CHECK(cudaMemcpy(d_signal, h_b.data(),
                          (size_t)fft_len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    {
        cuFloatComplex* res = run_fft_mixed_23(d_signal, d_work, fft_len, factors, 1, false, 0);
        if (res != d_b_fft)
            CUDA_CHECK(cudaMemcpy(d_b_fft, res, (size_t)fft_len * sizeof(cuFloatComplex),
                                  cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int u_base = 0; u_base < rows; u_base += u_tile) {
        const int tile_rows = std::min(u_tile, rows - u_base);
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);
        CUDA_CHECK(cudaMemsetAsync(d_signal, 0, signal_bytes, stream));

        // Pass 1: binary CSC build (sincosf-per-nonzero, packed indices).
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

        // Forward FFT (mixed-radix).
        cuFloatComplex* fwd = run_fft_mixed_23(
            d_signal, d_work, fft_len, factors, tile_rows, false, stream);
        cuFloatComplex* fwd_other = (fwd == d_signal) ? d_work : d_signal;

        // Pointwise multiply by d_b_fft.
        const int total_elems = tile_rows * fft_len;
        pointwise_mul_batched_kernel<<<(total_elems + 255) / 256, 256, 0, stream>>>(
            fwd, d_b_fft, fft_len, tile_rows);
        CUDA_CHECK(cudaGetLastError());

        // Inverse FFT (mixed-radix).
        cuFloatComplex* conv_result = run_fft_mixed_23(
            fwd, fwd_other, fft_len, factors, tile_rows, true, stream);

        // Finalize: chirp_conj × conv → d_out at u_base offset.
        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block, 0, stream>>>(
            conv_result, d_chirp, d_out,
            rows, output_stride, output_cols, fft_len, u_base, tile_rows);
        CUDA_CHECK(cudaGetLastError());
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
    const size_t mem_bytes = peak_tracker.finish();

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal);
    cudaFree(d_work);

    return {d_out, output_stride, ms, mem_bytes, {}};
}


// ===========================================================================
// same Bluestein scaffolding as the existing
// cuFFT path, but with fft_len = next_7_smooth(2·cols−1)
// Non-streaming; allocates full d_out.
// ===========================================================================
SparseFFTResult sparse_fft_csc_bluestein_cufft_smooth(const COOMatrix& coo, int u_tile) {
    const auto _pre0 = std::chrono::steady_clock::now();
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int output_cols   = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len       = next_7_smooth(2 * cols - 1);
    const bool use_packed   = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");
    // Adapt tile size to fft_len so each (tile_rows × fft_len) batch lands in
    // the ~1.5M complex-element sweet spot. Two concurrent streams at this
    // batch keep SM/HBM near saturation without contention. Caps at the
    // user-supplied u_tile (so explicit --csc-tile is respected); floors at 8.
    const int eff_tile = std::min(u_tile, std::max(8, 1500000 / fft_len));

    DeviceMemPeak peak_tracker;
    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (use_packed) {
        std::vector<uint16_t> ri = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, (size_t)coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, (size_t)n_sp    * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, ri.data(),
                              (size_t)coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp.data(),
                              (size_t)n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, (size_t)coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, (size_t)n_sp    * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              (size_t)coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              (size_t)n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b     = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp, (size_t)cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft, (size_t)fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(),
                          (size_t)cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // Precompute d_b_fft via one-shot cuFFT plan at the smooth size.
    CUDA_CHECK(cudaMemcpy(d_b_fft, h_b.data(), (size_t)fft_len * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));
    {
        int fl = fft_len;
        cufftHandle plan_b;
        CUFFT_CHECK(cufftPlanMany(&plan_b, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, 1));
        CUFFT_CHECK(cufftExecC2C(plan_b,
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 CUFFT_FORWARD));
        CUFFT_CHECK(cufftDestroy(plan_b));
        // Fold the Bluestein 1/fft_len normalization into B_fft once here, so
        // each tile's inverse-FFT output is already normalized and the per-tile
        // full-buffer scale_complex pass can be dropped (kernel fusion).
        {
            const int th = 256;
            scale_complex_kernel<<<(fft_len + th - 1) / th, th>>>(
                d_b_fft, fft_len, 1.f / (float)fft_len);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    const int max_tile_rows = std::min(eff_tile, rows);
    // Double-buffered tile pipeline: two in-place d_signal slabs (cuFFT C2C runs
    // in-place) and two streams so tile N's forward-FFT/multiply/inverse-FFT can
    // overlap with tile (N-1)'s finalize/d_out write. Each full-tile cuFFT plan
    // is bound to its own stream via cufftSetStream.
    cuFloatComplex *d_signal[2] = {nullptr, nullptr};
    cuFloatComplex *d_out       = nullptr;
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&d_signal[i],
            (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    }
    CUDA_CHECK(cudaMalloc(&d_out,
        (size_t)rows * output_stride * sizeof(cuFloatComplex)));

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    // Tile-loop cuFFT plans: one full-tile plan per stream, one shared last-tile
    // plan whose stream is rebound dynamically (last tile runs only once).
    cufftHandle plan_cufft[2] = {0, 0};
    cufftHandle plan_cufft_last = 0;
    size_t cufft_workspace = 0;
    {
        int fl = fft_len;
        const int full_tile = max_tile_rows;
        for (int i = 0; i < 2; ++i) {
            CUFFT_CHECK(cufftPlanMany(&plan_cufft[i], 1, &fl, nullptr, 1, fft_len,
                                      nullptr, 1, fft_len, CUFFT_C2C, full_tile));
            CUFFT_CHECK(cufftSetStream(plan_cufft[i], streams[i]));
            size_t ws = 0;
            CUFFT_CHECK(cufftGetSize(plan_cufft[i], &ws));
            cufft_workspace = std::max(cufft_workspace, ws);
        }
        const int last_tile = rows % eff_tile;
        if (last_tile != 0 && last_tile != full_tile) {
            CUFFT_CHECK(cufftPlanMany(&plan_cufft_last, 1, &fl, nullptr, 1, fft_len,
                                      nullptr, 1, fft_len, CUFFT_C2C, last_tile));
            size_t ws = 0;
            CUFFT_CHECK(cufftGetSize(plan_cufft_last, &ws));
            cufft_workspace = std::max(cufft_workspace, ws);
        }
    }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Finish all setup GPU work (d_b_fft precompute, H2D, cuFFT plan workspace)
    // so it is charged to preprocessing, not to the kernel timer.
    CUDA_CHECK(cudaDeviceSynchronize());
    const float preprocess_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - _pre0).count();

    CUDA_CHECK(cudaEventRecord(t0));

    for (int u_base = 0; u_base < rows; u_base += eff_tile) {
        const int tile_rows = std::min(eff_tile, rows - u_base);
        const int buf = (u_base / eff_tile) & 1;
        cudaStream_t stream = streams[buf];
        cuFloatComplex* sig = d_signal[buf];
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);
        CUDA_CHECK(cudaMemsetAsync(sig, 0, signal_bytes, stream));

        // Pass 1: standard binary CSC build (sincosf-per-nonzero, packed indices).
        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, sig,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_bluestein_input_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx, d_sp_cols, d_chirp, sig,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        // Bluestein convolution via cuFFT (forward + multiply + inverse + scale).
        cufftHandle tile_plan;
        if (plan_cufft_last && tile_rows != max_tile_rows) {
            CUFFT_CHECK(cufftSetStream(plan_cufft_last, stream));
            tile_plan = plan_cufft_last;
        } else {
            tile_plan = plan_cufft[buf];
        }
        CUFFT_CHECK(cufftExecC2C(tile_plan,
            reinterpret_cast<cufftComplex*>(sig),
            reinterpret_cast<cufftComplex*>(sig),
            CUFFT_FORWARD));
        const int total_elems = tile_rows * fft_len;
        pointwise_mul_batched_kernel<<<(total_elems + 255) / 256, 256, 0, stream>>>(
            sig, d_b_fft, fft_len, tile_rows);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecC2C(tile_plan,
            reinterpret_cast<cufftComplex*>(sig),
            reinterpret_cast<cufftComplex*>(sig),
            CUFFT_INVERSE));
        // (scale_complex dropped: 1/fft_len is pre-folded into d_b_fft above.)

        // Finalize: chirp_conj × conv → d_out at u_base offset.
        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_kernel<<<f_grid, f_block, 0, stream>>>(
            sig, d_chirp, d_out,
            rows, output_stride, output_cols, fft_len, u_base, tile_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    const size_t mem_bytes = peak_tracker.finish();

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    if (plan_cufft[0]) cufftDestroy(plan_cufft[0]);
    if (plan_cufft[1]) cufftDestroy(plan_cufft[1]);
    if (plan_cufft_last) cufftDestroy(plan_cufft_last);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal[0]);
    cudaFree(d_signal[1]);

    return {d_out, output_stride, ms, mem_bytes, {}, preprocess_ms};
}

// ---------------------------------------------------------------------------
// CSC Bluestein + cuFFT smooth, conjugate-symmetric (Lever B).
// Identical to sparse_fft_csc_bluestein_cufft_smooth except that the column
// transform (build + Bluestein convolution) runs only for output rows
// u in [0, rows/2] — the input matrix is real, so the remaining rows follow
// from F[rows-u][v] = conj(F[u][(cols-v) mod cols]) and are filled by a
// dual-write finalize kernel. Halves every per-tile kernel.
// ---------------------------------------------------------------------------
SparseFFTResult sparse_fft_csc_bluestein_cufft_smooth_sym(const COOMatrix& coo, int u_tile) {
    const auto _pre0 = std::chrono::steady_clock::now();
    const int rows = coo.rows;
    const int cols = coo.cols;
    const int rows_half     = rows / 2 + 1;   // computed rows; rest by conjugation
    const int output_cols   = cols / 2 + 1;
    const int output_stride = (output_cols + 3) / 4 * 4;
    const int fft_len       = next_7_smooth(2 * cols - 1);
    const bool use_packed   = can_pack_u16(rows, cols);
    if (u_tile <= 0)
        throw std::runtime_error("CSC tile size must be positive");
    // Adapt tile size to fft_len so each (tile_rows × fft_len) batch lands in
    // the ~1.5M complex-element sweet spot. Two concurrent streams at this
    // batch keep SM/HBM near saturation without contention. Caps at the
    // user-supplied u_tile (so explicit --csc-tile is respected); floors at 8.
    const int eff_tile = std::min(u_tile, std::max(8, 1500000 / fft_len));

    DeviceMemPeak peak_tracker;
    CompactCSC csc = make_compact_csc(coo);
    const int n_sp = csc.n_sparse_cols;

    int *d_col_ptr = nullptr, *d_row_idx = nullptr, *d_sp_cols = nullptr;
    uint16_t *d_row_idx_packed = nullptr, *d_sp_cols_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col_ptr, (n_sp + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_col_ptr, csc.col_ptr.data(),
                          (n_sp + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (use_packed) {
        std::vector<uint16_t> ri = pack_cols_u16(csc.row_idx.data(), coo.nnz);
        std::vector<uint16_t> sp = pack_cols_u16(csc.sparse_cols.data(), n_sp);
        CUDA_CHECK(cudaMalloc(&d_row_idx_packed, (size_t)coo.nnz * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols_packed, (size_t)n_sp    * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_row_idx_packed, ri.data(),
                              (size_t)coo.nnz * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols_packed, sp.data(),
                              (size_t)n_sp * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_row_idx, (size_t)coo.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sp_cols, (size_t)n_sp    * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_row_idx, csc.row_idx.data(),
                              (size_t)coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sp_cols, csc.sparse_cols.data(),
                              (size_t)n_sp * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::vector<cuFloatComplex> h_chirp = make_bluestein_chirp(cols);
    std::vector<cuFloatComplex> h_b     = make_bluestein_b(cols, fft_len);
    cuFloatComplex *d_chirp = nullptr, *d_b_fft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chirp, (size_t)cols    * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_b_fft, (size_t)fft_len * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMemcpy(d_chirp, h_chirp.data(),
                          (size_t)cols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // Precompute d_b_fft via one-shot cuFFT plan at the smooth size.
    CUDA_CHECK(cudaMemcpy(d_b_fft, h_b.data(), (size_t)fft_len * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));
    {
        int fl = fft_len;
        cufftHandle plan_b;
        CUFFT_CHECK(cufftPlanMany(&plan_b, 1, &fl, nullptr, 1, fft_len,
                                  nullptr, 1, fft_len, CUFFT_C2C, 1));
        CUFFT_CHECK(cufftExecC2C(plan_b,
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 reinterpret_cast<cufftComplex*>(d_b_fft),
                                 CUFFT_FORWARD));
        CUFFT_CHECK(cufftDestroy(plan_b));
        // Fold the Bluestein 1/fft_len normalization into B_fft once here, so
        // each tile's inverse-FFT output is already normalized and the per-tile
        // full-buffer scale_complex pass can be dropped (kernel fusion).
        {
            const int th = 256;
            scale_complex_kernel<<<(fft_len + th - 1) / th, th>>>(
                d_b_fft, fft_len, 1.f / (float)fft_len);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    const int max_tile_rows = std::min(eff_tile, rows_half);
    // Double-buffered tile pipeline: two in-place d_signal slabs (cuFFT C2C runs
    // in-place) and two streams so tile N's forward-FFT/multiply/inverse-FFT can
    // overlap with tile (N-1)'s finalize/d_out write. Each full-tile cuFFT plan
    // is bound to its own stream via cufftSetStream.
    cuFloatComplex *d_signal[2] = {nullptr, nullptr};
    cuFloatComplex *d_out       = nullptr;
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&d_signal[i],
            (size_t)max_tile_rows * fft_len * sizeof(cuFloatComplex)));
    }
    CUDA_CHECK(cudaMalloc(&d_out,
        (size_t)rows * output_stride * sizeof(cuFloatComplex)));

    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    // Tile-loop cuFFT plans: one full-tile plan per stream, one shared last-tile
    // plan whose stream is rebound dynamically (last tile runs only once).
    cufftHandle plan_cufft[2] = {0, 0};
    cufftHandle plan_cufft_last = 0;
    size_t cufft_workspace = 0;
    {
        int fl = fft_len;
        const int full_tile = max_tile_rows;
        for (int i = 0; i < 2; ++i) {
            CUFFT_CHECK(cufftPlanMany(&plan_cufft[i], 1, &fl, nullptr, 1, fft_len,
                                      nullptr, 1, fft_len, CUFFT_C2C, full_tile));
            CUFFT_CHECK(cufftSetStream(plan_cufft[i], streams[i]));
            size_t ws = 0;
            CUFFT_CHECK(cufftGetSize(plan_cufft[i], &ws));
            cufft_workspace = std::max(cufft_workspace, ws);
        }
        const int last_tile = rows_half % eff_tile;
        if (last_tile != 0 && last_tile != full_tile) {
            CUFFT_CHECK(cufftPlanMany(&plan_cufft_last, 1, &fl, nullptr, 1, fft_len,
                                      nullptr, 1, fft_len, CUFFT_C2C, last_tile));
            size_t ws = 0;
            CUFFT_CHECK(cufftGetSize(plan_cufft_last, &ws));
            cufft_workspace = std::max(cufft_workspace, ws);
        }
    }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Finish all setup GPU work (d_b_fft precompute, H2D, cuFFT plan workspace)
    // so it is charged to preprocessing, not to the kernel timer.
    CUDA_CHECK(cudaDeviceSynchronize());
    const float preprocess_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - _pre0).count();

    CUDA_CHECK(cudaEventRecord(t0));

    for (int u_base = 0; u_base < rows_half; u_base += eff_tile) {
        const int tile_rows = std::min(eff_tile, rows_half - u_base);
        const int buf = (u_base / eff_tile) & 1;
        cudaStream_t stream = streams[buf];
        cuFloatComplex* sig = d_signal[buf];
        const size_t signal_bytes = (size_t)tile_rows * fft_len * sizeof(cuFloatComplex);
        CUDA_CHECK(cudaMemsetAsync(sig, 0, signal_bytes, stream));

        // Pass 1: standard binary CSC build (sincosf-per-nonzero, packed indices).
        dim3 p1_block(CSC_BU);
        dim3 p1_grid(n_sp, (tile_rows + CSC_BU - 1) / CSC_BU);
        if (use_packed) {
            csc_build_bluestein_input_packed_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx_packed, d_sp_cols_packed, d_chirp, sig,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        } else {
            csc_build_bluestein_input_kernel<<<p1_grid, p1_block, 0, stream>>>(
                d_col_ptr, d_row_idx, d_sp_cols, d_chirp, sig,
                rows, n_sp, fft_len, u_base, tile_rows, 1.f / (float)rows);
        }
        CUDA_CHECK(cudaGetLastError());

        // Bluestein convolution via cuFFT (forward + multiply + inverse).
        cufftHandle tile_plan;
        if (plan_cufft_last && tile_rows != max_tile_rows) {
            CUFFT_CHECK(cufftSetStream(plan_cufft_last, stream));
            tile_plan = plan_cufft_last;
        } else {
            tile_plan = plan_cufft[buf];
        }
        CUFFT_CHECK(cufftExecC2C(tile_plan,
            reinterpret_cast<cufftComplex*>(sig),
            reinterpret_cast<cufftComplex*>(sig),
            CUFFT_FORWARD));
        const int total_elems = tile_rows * fft_len;
        pointwise_mul_batched_kernel<<<(total_elems + 255) / 256, 256, 0, stream>>>(
            sig, d_b_fft, fft_len, tile_rows);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecC2C(tile_plan,
            reinterpret_cast<cufftComplex*>(sig),
            reinterpret_cast<cufftComplex*>(sig),
            CUFFT_INVERSE));
        // (scale_complex dropped: 1/fft_len is pre-folded into d_b_fft above.)

        // Finalize: chirp_conj × conv → d_out, plus the conjugate-mirror row.
        dim3 f_block(32, 8);
        dim3 f_grid((output_cols + f_block.x - 1) / f_block.x,
                    (tile_rows + f_block.y - 1) / f_block.y);
        bluestein_finalize_sym_kernel<<<f_grid, f_block, 0, stream>>>(
            sig, d_chirp, d_out,
            rows, cols, output_stride, output_cols, fft_len, u_base, tile_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    const size_t mem_bytes = peak_tracker.finish();

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    if (plan_cufft[0]) cufftDestroy(plan_cufft[0]);
    if (plan_cufft[1]) cufftDestroy(plan_cufft[1]);
    if (plan_cufft_last) cufftDestroy(plan_cufft_last);
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_sp_cols);
    cudaFree(d_row_idx_packed);
    cudaFree(d_sp_cols_packed);
    cudaFree(d_chirp);
    cudaFree(d_b_fft);
    cudaFree(d_signal[0]);
    cudaFree(d_signal[1]);

    return {d_out, output_stride, ms, mem_bytes, {}, preprocess_ms};
}
