/**
 * MnemoCUDA GEMM Q4K — Tensor core matrix multiply for Q4K × Q8_1
 *
 * Ported from llama.cpp mmq.cuh, stripped to NVIDIA Ampere+ and Q4_K only.
 * Supports batch=1..128 via template mmq_x parameter.
 *
 * Flow: Q4K weights [M × K] × Q8_1 activations [K × N] → FP32 output [M × N]
 *   where K is the reduction dimension (hidden_size), M is output rows, N is batch.
 */

#include "mma.cuh"
#include <cuda_fp16.h>
#include <stdint.h>
#include <cstdio>

using namespace mnemo_mma;

// ── Q4K block (same as in kernels.cu) ──
struct __align__(2) GemmBlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};

// ── Q8_1 MMQ block: 128 quantized values + 4 scale/sum pairs ──
// Padded layout for bank-conflict-free shared memory access.
struct block_q8_1_mmq {
    union {
        half2 ds4[4];   // d0,s0, d1,s1, d2,s2, d3,s3 (scale + partial sum per 32 values)
    };
    int8_t qs[128];     // 4 × 32 quantized values
};

// Constants
#define QK4_K  256
#define QI4_K  (QK4_K / (4 * 2))  // = 32 ints per Q4K block for qs
#define QR4_K  2
#define QI8_1  16   // MMA k-tile for Q8_1
#define MMQ_Y  64   // rows per tile (Ada Lovelace optimal)
#define NWARPS 8    // warps per block
#define MMQ_TILE_NE_K  128   // K elements per tile iteration
#define MMQ_ITER_K     256   // K elements per Q4K block = 1 block per iteration
#define MMQ_MMA_TILE_X_K_Q8_1  16  // stride for x_qs in shared memory (int units)
#define MMQ_TILE_Y_K  (sizeof(block_q8_1_mmq) / sizeof(int))  // = 36

// ── Helper: read 4 bytes from Q4K qs ──
static __device__ __forceinline__ int get_int_b4(const uint8_t * x, int i) {
    return ((const int *)x)[i];
}

// ── Helper: unpack Q4K 6-bit scales ──
static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}

// ── Load Q4K weights tile into shared memory as INT8 ──
template <bool need_check>
static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile,
    const int kbx0, const int i_max, const int stride
) {
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);  // = 32
    constexpr int nrows = 32 / threads_per_row;  // = 1
    const int txi = threadIdx.x % threads_per_row;

    // Load quantized values: expand 4-bit → 8-bit with MMA-compatible interleaving
    #pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * NWARPS) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;
        if (need_check) i = min(i, i_max);

        const GemmBlockQ4K * bxi = (const GemmBlockQ4K *)x + kbx0 + i * stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

        // Low nibbles → positions [16*(txi/8) + txi%8 + 0]
        // High nibbles → positions [16*(txi/8) + txi%8 + 8]
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 16 * (txi / 8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 16 * (txi / 8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
    }

    // Load and pre-multiply scales
    constexpr int rows_per_warp = 32 / 2;  // = 16
    #pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + threadIdx.x / 2) % MMQ_Y;
        if (need_check) i = min(i, i_max);

        const GemmBlockQ4K * bxi = (const GemmBlockQ4K *)x + kbx0 + i * stride;
        const int * scales = (const int *) bxi->scales;
        const int ksc = threadIdx.x % 2;

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);
        const uint8_t * sc8 = (const uint8_t *)&sc32;
        const uint8_t *  m8 = (const uint8_t *) &m32;

        // d × scale, -dmin × min packed as half2
        half d_h, dmin_h;
        memcpy(&d_h, &bxi->d, 2);
        memcpy(&dmin_h, &bxi->dmin, 2);
        half2 dm = __hmul2(make_half2(d_h, dmin_h), make_half2(__float2half(1.0f), __float2half(-1.0f)));

        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + 4 * ksc + l] =
                __hmul2(dm, make_half2(__float2half((float)sc8[l]), __float2half((float)m8[l])));
        }
    }
}

// ── Compute: MMA dot product with scale application ──
template <int mmq_x>
static __device__ __forceinline__ void vec_dot_q4k_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y,
    float * __restrict__ sum, const int k00
) {
    typedef tile<16, 8, int>  tile_A;  // weight fragment
    typedef tile< 8, 8, int>  tile_B;  // activation fragment
    typedef tile<16, 8, int>  tile_C;  // accumulator

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2 * MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;  // skip ds4 header (4 ints = 16 bytes)
    const half2 * y_dm = (const half2 *) y;

    constexpr int rows_per_warp = 2 * 16;  // 2 × tile_C::I for NVIDIA
    constexpr int ntx = rows_per_warp / tile_C::I;  // = 2
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;

        // Load A tiles (weights) from shared memory
        tile_A A[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n * 16) * MMQ_MMA_TILE_X_K_Q8_1 + k0,
                          MMQ_MMA_TILE_X_K_Q8_1);
        }

        // Process each batch column tile
        #pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
            // Load B tile (activations) from shared memory
            tile_B B;
            load_ldmatrix(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                    float2 dmA = __half22float2(x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + k0 / QI8_1]);
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.x * dsB.x * C.x[l];
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += dmA.y * dsB.y;
                }
            }
        }
    }
}

// ── Write back results ──
template <int mmq_x, bool need_check>
static __device__ __forceinline__ void gemm_write_back(
    const float * __restrict__ sum, float * __restrict__ dst,
    const int stride_dst, const int i_max, const int j_max
) {
    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * 16;
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                if (j > j_max) continue;
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (need_check && i > i_max) continue;
                dst[j * stride_dst + i] = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
            }
        }
    }
}

// ── Main GEMM kernel ──
template <int mmq_x, bool need_check>
__global__ void gemm_q4k_kernel(
    const char  * __restrict__ weights,   // Q4K [M × K] as raw bytes
    const int   * __restrict__ act_q8,    // Q8_1 pre-quantized [N × K/128 × block_q8_1_mmq]
    float       * __restrict__ output,    // [M × N] column-major
    const int M, const int K, const int N,
    const int stride_row_x  // Q4K blocks per row
) {
    extern __shared__ int smem[];
    int * tile_y = smem;
    int * tile_x = tile_y + mmq_x * MMQ_TILE_Y_K + NWARPS * 32;  // pad for alignment

    const int tile_i = blockIdx.x;  // row tile
    const int tile_j = blockIdx.y;  // column (batch) tile
    const int i_max = min(MMQ_Y - 1, M - 1 - tile_i * MMQ_Y);
    const int j_max = min(mmq_x - 1, N - 1 - tile_j * mmq_x);

    constexpr int ne_sum = mmq_x * MMQ_Y / (NWARPS * 32);
    float sum[ne_sum] = {0.0f};

    const int blocks_per_row = K / QK4_K;

    for (int kb = 0; kb < blocks_per_row; kb++) {
        // Load Q4K weight tile
        load_tiles_q4_K<need_check>(
            weights, tile_x, kb + tile_i * MMQ_Y * stride_row_x,
            i_max, stride_row_x);

        // Load first half of Q8_1 activation tile (128 values)
        {
            const int * by0 = act_q8 + N * (kb * QK4_K / MMQ_TILE_NE_K) * MMQ_TILE_Y_K
                              + tile_j * mmq_x * MMQ_TILE_Y_K;
            #pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += NWARPS * 32) {
                int l = l0 + threadIdx.y * 32 + threadIdx.x;
                if (l < mmq_x * MMQ_TILE_Y_K)
                    tile_y[l] = by0[l];
            }
        }
        __syncthreads();

        vec_dot_q4k_q8_1_mma<mmq_x>(tile_x, tile_y, sum, 0);
        __syncthreads();

        // Load second half (next 128 values)
        {
            const int * by1 = act_q8 + N * ((kb * QK4_K / MMQ_TILE_NE_K) * MMQ_TILE_Y_K + MMQ_TILE_Y_K)
                              + tile_j * mmq_x * MMQ_TILE_Y_K;
            #pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += NWARPS * 32) {
                int l = l0 + threadIdx.y * 32 + threadIdx.x;
                if (l < mmq_x * MMQ_TILE_Y_K)
                    tile_y[l] = by1[l];
            }
        }
        __syncthreads();

        vec_dot_q4k_q8_1_mma<mmq_x>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    // Write results
    gemm_write_back<mmq_x, need_check>(
        sum, output + tile_j * mmq_x * M + tile_i * MMQ_Y,
        M, i_max, j_max);
}

// ── Host wrapper ──

extern "C" {

void cuda_gemm_q4k(const void *weights, const void *act_q8, float *output,
                    int M, int K, int N, cudaStream_t stream) {
    const int blocks_per_row = K / QK4_K;
    const dim3 block(32, NWARPS);
    const bool need_check = (M % MMQ_Y != 0);

    // Select mmq_x based on batch size N
    int mmq_x = 8;
    int best_tiles = (N + 7) / 8;
    for (int mx = 16; mx <= 128; mx += 8) {
        int tiles = (N + mx - 1) / mx;
        if (tiles < best_tiles) { best_tiles = tiles; mmq_x = mx; }
    }

    const dim3 grid((M + MMQ_Y - 1) / MMQ_Y, (N + mmq_x - 1) / mmq_x);

    // Shared memory: tile_y + pad + tile_x
    // tile_x needs: MMQ_Y * MMQ_MMA_TILE_X_K_Q8_1 ints for qs + MMQ_Y * MMQ_MMA_TILE_X_K_Q8_1 half2 for dm
    size_t smem_x = MMQ_Y * MMQ_MMA_TILE_X_K_Q8_1 * sizeof(int) * 2 +
                    MMQ_Y * MMQ_MMA_TILE_X_K_Q8_1 * sizeof(half2);

    #define LAUNCH(MX) do { \
        size_t smem = (MX) * MMQ_TILE_Y_K * sizeof(int) + NWARPS * 32 * sizeof(int) + smem_x; \
        if (need_check) \
            gemm_q4k_kernel<MX, true><<<grid, block, smem, stream>>>( \
                (const char *)weights, (const int *)act_q8, output, M, K, N, blocks_per_row); \
        else \
            gemm_q4k_kernel<MX, false><<<grid, block, smem, stream>>>( \
                (const char *)weights, (const int *)act_q8, output, M, K, N, blocks_per_row); \
    } while(0)

    switch (mmq_x) {
        case   8: LAUNCH(8);   break;
        case  16: LAUNCH(16);  break;
        case  24: LAUNCH(24);  break;
        case  32: LAUNCH(32);  break;
        case  40: LAUNCH(40);  break;
        case  48: LAUNCH(48);  break;
        case  56: LAUNCH(56);  break;
        case  64: LAUNCH(64);  break;
        case  72: LAUNCH(72);  break;
        case  80: LAUNCH(80);  break;
        case  88: LAUNCH(88);  break;
        case  96: LAUNCH(96);  break;
        case 104: LAUNCH(104); break;
        case 112: LAUNCH(112); break;
        case 120: LAUNCH(120); break;
        case 128: LAUNCH(128); break;
        default:  LAUNCH(8);   break;
    }

    #undef LAUNCH
}

// Q8_1 pre-quantization kernel for activations
__global__ void quantize_mmq_q8_1_kernel(
    const float * __restrict__ src,  // [N × K]
    block_q8_1_mmq * __restrict__ dst,  // [N × K/128]
    int K, int N
) {
    int col = blockIdx.y;  // batch column
    int sb = blockIdx.x * blockDim.x + threadIdx.x;  // super-block index (128 values each)
    int n_sb = K / 128;
    if (sb >= n_sb || col >= N) return;

    const float * src_sb = src + col * K + sb * 128;
    block_q8_1_mmq * out = dst + col * n_sb + sb;

    // Quantize 4 sub-blocks of 32 values
    for (int sub = 0; sub < 4; sub++) {
        const float * xb = src_sb + sub * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xb[i]));

        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        float sum = 0.0f;

        for (int i = 0; i < 32; i++) {
            int v = __float2int_rn(xb[i] * id);
            v = max(-127, min(127, v));
            out->qs[sub * 32 + i] = (int8_t)v;
            sum += xb[i];
        }
        out->ds4[sub] = make_half2(__float2half(d), __float2half(d * sum));
    }
}

void cuda_quantize_mmq_q8_1(const float *src, void *dst, int K, int N, cudaStream_t stream) {
    int n_sb = K / 128;
    dim3 grid((n_sb + 3) / 4, N);
    quantize_mmq_q8_1_kernel<<<grid, 4, 0, stream>>>(src, (block_q8_1_mmq *)dst, K, N);
}

} // extern "C"
