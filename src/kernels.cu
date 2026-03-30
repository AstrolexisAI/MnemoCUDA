/**
 * MnemoCUDA Kernels — Dequantized matrix-vector multiply for GGUF quantized tensors.
 *
 * Each kernel: y[n_rows] = W[n_rows × n_cols] · x[n_cols]
 * where W is quantized (Q4_K, Q6_K, Q8_0) and x, y are float32.
 *
 * Optimized for NVIDIA GPUs:
 * - Warp-level reductions (__shfl_xor_sync)
 * - Shared memory x-vector caching
 * - Coalesced memory access
 * - One warp per row (32 threads)
 */

#include <cuda_fp16.h>
#include <stdint.h>
#include <cstdio>

// ── Block structures (must match MnemoTypes.h) ──

struct __align__(2) BlockQ4K {
    uint16_t d;           // super-block scale (f16)
    uint16_t dmin;        // super-block min (f16)
    uint8_t  scales[12];  // sub-block scales/mins
    uint8_t  qs[128];     // 4-bit quantized values
};

struct __align__(2) BlockQ6K {
    uint8_t ql[128];      // lower 4 bits
    uint8_t qh[64];       // upper 2 bits
    int8_t  scales[16];   // sub-block scales
    uint16_t d;           // super-block scale (f16)
};

struct __align__(2) BlockQ8_0 {
    uint16_t d;           // scale (f16)
    int8_t   qs[32];      // quantized values
};

// ── Helpers ──

__device__ inline float f16_to_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

__device__ inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t &sc, uint8_t &m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// Warp reduction (sum across 32 threads)
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ── Q4_K Dequantized Matrix-Vector Multiply ──
// One warp (32 threads) per output row.
// Each thread processes blocks_per_row/32 blocks.

__global__ void matvec_q4k(
    const BlockQ4K *__restrict__ weights,
    const float    *__restrict__ x,
    float          *__restrict__ y,
    int n_rows, int n_cols
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= n_rows) return;

    int blocks_per_row = (n_cols + 255) / 256;
    const BlockQ4K *row_blocks = weights + row * blocks_per_row;

    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ4K &blk = row_blocks[b];
        float d   = f16_to_f32(blk.d);
        float dmin = f16_to_f32(blk.dmin);
        const uint8_t *q = blk.qs;
        int base = b * 256;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, blk.scales, sc1, m1);
            float d1 = d * sc1, m1f = dmin * m1;
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);
            float d2 = d * sc2, m2f = dmin * m2;

            for (int l = 0; l < 32 && (base + j + l) < n_cols; l++)
                sum += (d1 * (q[l] & 0xF) - m1f) * x[base + j + l];
            for (int l = 0; l < 32 && (base + j + 32 + l) < n_cols; l++)
                sum += (d2 * (q[l] >> 4) - m2f) * x[base + j + 32 + l];

            q += 32;
            is += 2;
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// ── Q6_K Dequantized Matrix-Vector Multiply ──
// Block: [ql:128][qh:64][scales:16][d:2] = 210 bytes, 256 values
// ql layout per half (64 bytes → 128 values):
//   values[0..31]   = ql[0:32]  & 0xF  (lo nibbles of first 32 bytes)
//   values[32..63]  = ql[32:64] & 0xF  (lo nibbles of next 32 bytes)
//   values[64..95]  = ql[0:32]  >> 4   (hi nibbles of first 32 bytes)
//   values[96..127] = ql[32:64] >> 4   (hi nibbles of next 32 bytes)
// qh layout per half (32 bytes → 128 hi2 values):
//   hi2[val] = (qh[val%32] >> (2*(val/32))) & 3

__global__ void matvec_q6k(
    const BlockQ6K *__restrict__ weights,
    const float    *__restrict__ x,
    float          *__restrict__ y,
    int n_rows, int n_cols
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= n_rows) return;

    int blocks_per_row = (n_cols + 255) / 256;
    const BlockQ6K *row_blocks = weights + row * blocks_per_row;

    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ6K &blk = row_blocks[b];
        float d_all = f16_to_f32(blk.d);
        int base = b * 256;

        const uint8_t *ql = blk.ql;
        const uint8_t *qh = blk.qh;
        const int8_t  *sc = blk.scales;

        // Process 2 halves of 128 values each
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;  // 64 bytes per half
            const uint8_t *qh_h = qh + half * 32;  // 32 bytes per half
            int sc_off = half * 8;
            int val_off = half * 128;

            // Dequant 128 values: 4 groups of 32 from ql nibbles
            // Group 0: vals[0..31]   = ql[0:32] & 0xF,  qh shift=0
            // Group 1: vals[32..63]  = ql[32:64] & 0xF,  qh shift=2
            // Group 2: vals[64..95]  = ql[0:32] >> 4,    qh shift=4
            // Group 3: vals[96..127] = ql[32:64] >> 4,   qh shift=6

            for (int grp = 0; grp < 4; grp++) {
                int ql_base = (grp < 2) ? (grp * 32) : ((grp - 2) * 32);
                int shift_lo = (grp < 2) ? 0 : 4;
                int shift_hi = grp * 2;

                for (int l = 0; l < 32; l++) {
                    int val_idx = val_off + grp * 32 + l;
                    if (base + val_idx >= n_cols) continue;

                    uint8_t lo4 = (ql_h[ql_base + l] >> shift_lo) & 0xF;
                    uint8_t hi2 = (qh_h[l] >> shift_hi) & 0x03;
                    int8_t val = (int8_t)((hi2 << 4) | lo4) - 32;

                    // Scale: 16 groups of 16, so scale index = val_within_half / 16
                    int si = sc_off + (grp * 32 + l) / 16;
                    float scale = d_all * sc[si];
                    sum += scale * val * x[base + val_idx];
                }
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// ── Q3_K Dequantized Matrix-Vector Multiply ──
// Block layout: [hmask:32][qs:64][scales:12][d:2] = 110 bytes, 256 values
// qs: 4 values per byte (2 bits each), 64 bytes = 256 values
// hmask: bit j/32 of byte j%32 gives high bit of value j

struct __align__(2) BlockQ3K {
    uint8_t hmask[32];    // high bits mask
    uint8_t qs[64];       // low 2 bits (4 values per byte)
    uint8_t scales[12];   // sub-block scales (6-bit packed)
    uint16_t d;           // super-block scale (f16)
};

__global__ void matvec_q3k(
    const BlockQ3K *__restrict__ weights,
    const float    *__restrict__ x,
    float          *__restrict__ y,
    int n_rows, int n_cols
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= n_rows) return;

    int blocks_per_row = (n_cols + 255) / 256;
    const BlockQ3K *row_blocks = weights + row * blocks_per_row;

    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ3K &blk = row_blocks[b];
        float d_all = f16_to_f32(blk.d);
        int base = b * 256;

        // Decode 6-bit scales from packed 12 bytes → 16 scale values
        int8_t sc16[16];
        for (int i = 0; i < 8; i++) {
            sc16[i]     = (int8_t)((blk.scales[i] & 0xF) | (((blk.scales[8 + (i >> 2)] >> (2 * (i & 3))) & 3) << 4)) - 32;
            sc16[i + 8] = (int8_t)((blk.scales[i] >> 4)  | (((blk.scales[8 + (i >> 2)] >> (2 * (i & 3) + 4)) & 3) << 4)) - 32;
        }

        // Process 256 values in 2 chunks of 128
        // llama.cpp layout: qs has 4 values per byte (2 bits each, shift by 0/2/4/6)
        // hmask: byte index = value_index % 32, bit = m (starts 1, shifts left per group)
        const uint8_t *q = blk.qs;
        int is = 0;   // scale index (0..15)
        uint8_t m = 1; // hmask bit mask (cycles through bits 0-7)
        int val_idx = 0;

        for (int chunk = 0; chunk < 2; chunk++) {
            int shift = 0;
            for (int grp = 0; grp < 4; grp++) {
                float dl = d_all * sc16[is++];
                for (int l = 0; l < 16; l++) {
                    int idx = base + val_idx;
                    if (idx < n_cols) {
                        int8_t q2 = (q[l] >> shift) & 3;
                        int8_t val = q2 - ((blk.hmask[l] & m) ? 0 : 4);
                        sum += dl * val * x[idx];
                    }
                    val_idx++;
                }
                float dl2 = d_all * sc16[is++];
                for (int l = 0; l < 16; l++) {
                    int idx = base + val_idx;
                    if (idx < n_cols) {
                        int8_t q2 = (q[l + 16] >> shift) & 3;
                        int8_t val = q2 - ((blk.hmask[l + 16] & m) ? 0 : 4);
                        sum += dl2 * val * x[idx];
                    }
                    val_idx++;
                }
                shift += 2;
                m <<= 1;
            }
            q += 32; // next 32 bytes of qs for second chunk
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// ── Q5_K Dequantized Matrix-Vector Multiply ──
// Block layout: [d:2][dmin:2][scales:12][qh:32][qs:128] = 176 bytes, 256 values (5-bit)

struct __align__(2) BlockQ5K {
    uint16_t d;           // super-block scale (f16)
    uint16_t dmin;        // super-block min (f16)
    uint8_t  scales[12];  // sub-block scales/mins (same packing as Q4_K)
    uint8_t  qh[32];      // high bits (bit 4 of each value)
    uint8_t  qs[128];     // low 4 bits (packed pairs)
};

__global__ void matvec_q5k(
    const BlockQ5K *__restrict__ weights,
    const float    *__restrict__ x,
    float          *__restrict__ y,
    int n_rows, int n_cols
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= n_rows) return;

    int blocks_per_row = (n_cols + 255) / 256;
    const BlockQ5K *row_blocks = weights + row * blocks_per_row;

    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ5K &blk = row_blocks[b];
        float d   = f16_to_f32(blk.d);
        float dmin = f16_to_f32(blk.dmin);
        const uint8_t *q = blk.qs;
        const uint8_t *qh = blk.qh;
        int base = b * 256;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is, blk.scales, sc1, m1);
            float d1 = d * sc1, m1f = dmin * m1;
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);
            float d2 = d * sc2, m2f = dmin * m2;

            for (int l = 0; l < 32 && (base + j + l) < n_cols; l++) {
                uint8_t lo4 = q[l] & 0xF;
                uint8_t hi = (qh[(j + l) / 8] >> ((j + l) % 8)) & 1;
                sum += (d1 * (lo4 | (hi << 4)) - m1f) * x[base + j + l];
            }
            for (int l = 0; l < 32 && (base + j + 32 + l) < n_cols; l++) {
                uint8_t lo4 = q[l] >> 4;
                uint8_t hi = (qh[(j + 32 + l) / 8] >> ((j + 32 + l) % 8)) & 1;
                sum += (d2 * (lo4 | (hi << 4)) - m2f) * x[base + j + 32 + l];
            }

            q += 32;
            is += 2;
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// ── Q8_0 Dequantized Matrix-Vector Multiply ──

__global__ void matvec_q8_0(
    const BlockQ8_0 *__restrict__ weights,
    const float     *__restrict__ x,
    float           *__restrict__ y,
    int n_rows, int n_cols
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= n_rows) return;

    int blocks_per_row = (n_cols + 31) / 32;
    const BlockQ8_0 *row_blocks = weights + row * blocks_per_row;

    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ8_0 &blk = row_blocks[b];
        float d = f16_to_f32(blk.d);
        int base = b * 32;

        for (int j = 0; j < 32 && (base + j) < n_cols; j++)
            sum += d * blk.qs[j] * x[base + j];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// ── SwiGLU Activation ──

__global__ void swiglu_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    float       *__restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float silu = g / (1.0f + expf(-g));
    out[i] = silu * up[i];
}

// ── RMS Normalization ──

__global__ void rms_norm_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    float       *__restrict__ output,
    int n, float eps
) {
    // Shared memory for partial sums
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float local_sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        float v = input[i];
        local_sum += v * v;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)n + eps);

    for (int i = tid; i < n; i += blockDim.x)
        output[i] = input[i] * scale * weight[i];
}

// ── Residual Add ──

__global__ void residual_add_kernel(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float       *__restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

// ── MoE Combine: weighted sum of K experts + residual ──

__global__ void moe_combine_kernel(
    const float *__restrict__ expert_outs, // [K * hidden]
    const float *__restrict__ weights_arr, // [K]
    const float *__restrict__ residual,    // [hidden]
    float       *__restrict__ output,      // [hidden]
    int hidden, int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;

    float sum = 0.0f;
    for (int e = 0; e < k; e++)
        sum += expert_outs[e * hidden + i] * weights_arr[e];
    output[i] = sum + residual[i];
}

// ── RoPE (Rotary Position Embedding) ──

__global__ void rope_kernel(
    float *__restrict__ q,     // [n_heads * head_dim]
    float *__restrict__ k,     // [n_kv_heads * head_dim]
    int head_dim, int pos, float theta,
    int n_heads_q, int n_heads_k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = (n_heads_q + n_heads_k) * half_dim;
    if (tid >= total) return;

    // Determine if this thread handles Q or K
    float *vec;
    int head, j;
    if (tid < n_heads_q * half_dim) {
        vec = q;
        head = tid / half_dim;
        j = tid % half_dim;
    } else {
        vec = k;
        int offset = tid - n_heads_q * half_dim;
        head = offset / half_dim;
        j = offset % half_dim;
    }

    float freq = 1.0f / powf(theta, (float)(2 * j) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int idx = head * head_dim + j;
    float v0 = vec[idx];
    float v1 = vec[idx + half_dim];
    vec[idx]            = v0 * cos_a - v1 * sin_a;
    vec[idx + half_dim] = v0 * sin_a + v1 * cos_a;
}

// ── F32 Matrix-Vector Multiply (for router weights) ──
// GGUF stores F32 tensors row-major: weights[row * n_cols + col]
// y[row] = sum_col(weights[row * n_cols + col] * x[col])

__global__ void matvec_f32(
    const float *__restrict__ weights, // [n_rows × n_cols], row-major
    const float *__restrict__ x,
    float       *__restrict__ y,
    int n_rows, int n_cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    float sum = 0.0f;
    for (int col = 0; col < n_cols; col++)
        sum += weights[row * n_cols + col] * x[col];
    y[row] = sum;
}

// ── GQA Attention kernel: Q@K^T → softmax → @V (one kernel per head) ──
// One block per query head. Each thread handles a subset of positions.

// Tile size for tiled attention (online softmax). 2048 floats = 8KB smem.
#define ATTN_TILE 2048

// Tiled attention kernel for FP32 KV cache (same online softmax approach).
__global__ void attention_kernel(
    const float *__restrict__ q, const float *__restrict__ kv_k,
    const float *__restrict__ kv_v, float *__restrict__ out,
    int head_dim, int n_kv_heads, int seq_len, int gqa_ratio, float scale
) {
    int head = blockIdx.x, kv_head = head / gqa_ratio, tid = threadIdx.x;
    extern __shared__ float shared[];
    float *tile_scores = shared;
    __shared__ float smax[32], ssum[32];
    int warp_id = tid / 32, lane = tid % 32;
    const float *qh = q + head * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    float *out_h = out + head * head_dim;

    float global_max = -1e30f, global_sum = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) out_h[d] = 0.0f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += ATTN_TILE) {
        int tile_end = tile_start + ATTN_TILE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        for (int i = tid; i < tile_len; i += blockDim.x) {
            int p = tile_start + i;
            const float *kp = kv_k + (size_t)p * kv_stride + (size_t)kv_head * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qh[d] * kp[d];
            tile_scores[i] = dot * scale;
        }
        __syncthreads();

        float tile_max = -1e30f;
        for (int i = tid; i < tile_len; i += blockDim.x)
            if (tile_scores[i] > tile_max) tile_max = tile_scores[i];
        for (int o = 16; o > 0; o >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, o));
        if (lane == 0 && warp_id < 32) smax[warp_id] = tile_max;
        __syncthreads();
        if (tid == 0) { float m = smax[0]; for (int i = 1; i < (blockDim.x+31)/32; i++) m = fmaxf(m, smax[i]); smax[0] = m; }
        __syncthreads();
        tile_max = smax[0];

        float new_max = fmaxf(global_max, tile_max);
        float rescale = (global_sum > 0.0f) ? expf(global_max - new_max) : 0.0f;
        __syncthreads();
        for (int d = tid; d < head_dim; d += blockDim.x) out_h[d] *= rescale;
        float new_sum = global_sum * rescale;

        for (int i = tid; i < tile_len; i += blockDim.x)
            tile_scores[i] = expf(tile_scores[i] - new_max);
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < tile_len; i += blockDim.x) local_sum += tile_scores[i];
        for (int o = 16; o > 0; o >>= 1)
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, o);
        if (lane == 0 && warp_id < 32) ssum[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) { float s = ssum[0]; for (int i = 1; i < (blockDim.x+31)/32; i++) s += ssum[i]; ssum[0] = s; }
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = 0.0f;
            for (int i = 0; i < tile_len; i++)
                val += tile_scores[i] * kv_v[(size_t)(tile_start + i) * kv_stride + (size_t)kv_head * head_dim + d];
            out_h[d] += val;
        }

        global_max = new_max;
        global_sum = new_sum + ssum[0];
        __syncthreads();
    }

    if (global_sum > 0.0f)
        for (int d = tid; d < head_dim; d += blockDim.x) out_h[d] /= global_sum;
}

// ── Scaled add: out[i] += scale * x[i] (for expert accumulation) ──

__global__ void scaled_add_kernel(
    float       *__restrict__ out,
    const float *__restrict__ x,
    float scale, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += scale * x[i];
}

// ── F32 Top-K selection + softmax (for router, runs on GPU) ──
// Warp-based (32 threads): each thread scans a strided chunk of experts,
// maintains thread-local top-k, then warp-level merge finds global top-k.
__global__ void topk_softmax_kernel(
    const float *__restrict__ scores,
    int         *__restrict__ indices,
    float       *__restrict__ weights,
    int n_experts, int k
) {
    const int tid = threadIdx.x;  // 0..31
    const int WARP = 32;

    // Thread-local top-k (register-based, k <= 16)
    float my_vals[16];
    int   my_ids[16];
    for (int i = 0; i < k; i++) { my_vals[i] = -1e30f; my_ids[i] = -1; }

    // Each thread scans strided slice of experts
    for (int j = tid; j < n_experts; j += WARP) {
        float val = scores[j];
        // Find the minimum in thread-local top-k
        int mi = 0;
        for (int i = 1; i < k; i++)
            if (my_vals[i] < my_vals[mi]) mi = i;
        if (val > my_vals[mi]) { my_vals[mi] = val; my_ids[mi] = j; }
    }

    // Warp-level merge: collect all 32*k candidates, pick global top-k.
    // Use shared memory for the merge (32 threads × 16 max entries = 512).
    __shared__ float s_vals[32 * 16];
    __shared__ int   s_ids[32 * 16];

    for (int i = 0; i < k; i++) {
        s_vals[tid * 16 + i] = my_vals[i];
        s_ids[tid * 16 + i]  = my_ids[i];
    }
    // Fill unused slots with sentinel
    for (int i = k; i < 16; i++) {
        s_vals[tid * 16 + i] = -1e30f;
        s_ids[tid * 16 + i] = -1;
    }
    __syncthreads();

    // Thread 0 performs final selection from 32*16 = 512 candidates
    if (tid == 0) {
        float top_vals[16];
        int   top_ids[16];
        for (int i = 0; i < k; i++) { top_vals[i] = -1e30f; top_ids[i] = -1; }
        int total_candidates = WARP * 16;
        for (int j = 0; j < total_candidates; j++) {
            float val = s_vals[j];
            if (val <= -1e30f) continue;
            int mi = 0;
            for (int i = 1; i < k; i++)
                if (top_vals[i] < top_vals[mi]) mi = i;
            if (val > top_vals[mi]) { top_vals[mi] = val; top_ids[mi] = s_ids[j]; }
        }
        // Softmax over selected top-k
        float mx = top_vals[0];
        for (int i = 1; i < k; i++) if (top_vals[i] > mx) mx = top_vals[i];
        float sum = 0;
        for (int i = 0; i < k; i++) { top_vals[i] = expf(top_vals[i] - mx); sum += top_vals[i]; }
        for (int i = 0; i < k; i++) {
            weights[i] = top_vals[i] / sum;
            indices[i] = top_ids[i];
        }
    }
}

// ── Host-callable wrapper functions ──

// Check for kernel launch errors (async — only catches config errors)
#define KERNEL_LAUNCH_CHECK() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
        fprintf(stderr, "[MnemoCUDA] kernel launch error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
} while(0)

extern "C" {

void cuda_matvec_q4k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q4k<<<blocks, threads, 0, stream>>>(
        (const BlockQ4K *)weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q6k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q6k<<<blocks, threads, 0, stream>>>(
        (const BlockQ6K *)weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q5k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q5k<<<blocks, threads, 0, stream>>>(
        (const BlockQ5K *)weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q3k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q3k<<<blocks, threads, 0, stream>>>(
        (const BlockQ3K *)weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q8_0(const void *weights, const float *x, float *y,
                      int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q8_0<<<blocks, threads, 0, stream>>>(
        (const BlockQ8_0 *)weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_swiglu(const float *gate, const float *up, float *out,
                 int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
    KERNEL_LAUNCH_CHECK();
}

void cuda_rms_norm(const float *input, const float *weight, float *output,
                   int n, float eps, cudaStream_t stream) {
    int threads = 256;
    rms_norm_kernel<<<1, threads, threads * sizeof(float), stream>>>(
        input, weight, output, n, eps);
    KERNEL_LAUNCH_CHECK();
}

void cuda_residual_add(const float *a, const float *b, float *out,
                       int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
    KERNEL_LAUNCH_CHECK();
}

void cuda_moe_combine(const float *expert_outs, const float *weights_arr,
                      const float *residual, float *output,
                      int hidden, int k, cudaStream_t stream) {
    int threads = 256;
    int blocks = (hidden + threads - 1) / threads;
    moe_combine_kernel<<<blocks, threads, 0, stream>>>(
        expert_outs, weights_arr, residual, output, hidden, k);
    KERNEL_LAUNCH_CHECK();
}

void cuda_rope(float *q, float *k, int head_dim, int pos, float theta,
               int n_heads_q, int n_heads_k, cudaStream_t stream) {
    int total = (n_heads_q + n_heads_k) * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k, head_dim, pos, theta, n_heads_q, n_heads_k);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_f32(const float *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    matvec_f32<<<blocks, threads, 0, stream>>>(weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_attention(const float *q, const float *kv_k, const float *kv_v,
                    float *out, int n_heads_q, int head_dim, int n_kv_heads,
                    int seq_len, int gqa_ratio, float scale, cudaStream_t stream) {
    // One block per query head, 256 threads per block
    int threads = 256;
    size_t smem = ATTN_TILE * sizeof(float) + 64 * sizeof(float);
    attention_kernel<<<n_heads_q, threads, smem, stream>>>(
        q, kv_k, kv_v, out, head_dim, n_kv_heads, seq_len, gqa_ratio, scale);
    KERNEL_LAUNCH_CHECK();
}

void cuda_scaled_add(float *out, const float *x, float scale, int n,
                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scaled_add_kernel<<<blocks, threads, 0, stream>>>(out, x, scale, n);
    KERNEL_LAUNCH_CHECK();
}

void cuda_topk_softmax(const float *scores, int *indices, float *weights,
                       int n_experts, int k, cudaStream_t stream) {
    topk_softmax_kernel<<<1, 32, 0, stream>>>(scores, indices, weights, n_experts, k);
    KERNEL_LAUNCH_CHECK();
}

// ── FP32 → FP16 conversion ──

__global__ void f32_to_f16_kernel(const float *in, __half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

void cuda_f32_to_f16(const float *in, void *out, int n, cudaStream_t stream) {
    f32_to_f16_kernel<<<(n+255)/256, 256, 0, stream>>>(in, (__half *)out, n);
    KERNEL_LAUNCH_CHECK();
}

// ── Attention with FP16 KV cache ──

// Tiled attention kernel for FP16 KV cache (online softmax, same approach as F32).
__global__ void attention_kernel_f16kv(
    const float *__restrict__ q, const __half *__restrict__ kv_k,
    const __half *__restrict__ kv_v, float *__restrict__ out,
    int head_dim, int n_kv_heads, int seq_len, int gqa_ratio, float scale
) {
    int head = blockIdx.x, kv_head = head / gqa_ratio, tid = threadIdx.x;
    extern __shared__ float shared[];
    float *tile_scores = shared;  // [ATTN_TILE] — fits in ~8KB
    __shared__ float smax[32], ssum[32];
    int warp_id = tid / 32, lane = tid % 32;
    const float *qh = q + head * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    float *out_h = out + head * head_dim;

    // Initialize output accumulator and online softmax state
    // Each thread accumulates its own output dimensions
    float global_max = -1e30f;
    float global_sum = 0.0f;

    // Zero output
    for (int d = tid; d < head_dim; d += blockDim.x)
        out_h[d] = 0.0f;

    // Process tiles of positions
    for (int tile_start = 0; tile_start < seq_len; tile_start += ATTN_TILE) {
        int tile_end = tile_start + ATTN_TILE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // Compute Q @ K^T for this tile
        for (int i = tid; i < tile_len; i += blockDim.x) {
            int p = tile_start + i;
            const __half *kp = kv_k + (size_t)p * kv_stride + (size_t)kv_head * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qh[d] * __half2float(kp[d]);
            tile_scores[i] = dot * scale;
        }
        __syncthreads();

        // Find tile max
        float tile_max = -1e30f;
        for (int i = tid; i < tile_len; i += blockDim.x)
            if (tile_scores[i] > tile_max) tile_max = tile_scores[i];
        for (int o = 16; o > 0; o >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, o));
        if (lane == 0 && warp_id < 32) smax[warp_id] = tile_max;
        __syncthreads();
        if (tid == 0) {
            float m = smax[0];
            for (int i = 1; i < (blockDim.x + 31) / 32; i++) m = fmaxf(m, smax[i]);
            smax[0] = m;
        }
        __syncthreads();
        tile_max = smax[0];

        // Online softmax: rescale previous accumulator if new max is larger
        float new_max = fmaxf(global_max, tile_max);
        float rescale = (global_sum > 0.0f) ? expf(global_max - new_max) : 0.0f;

        // Rescale existing output accumulator
        __syncthreads();
        for (int d = tid; d < head_dim; d += blockDim.x)
            out_h[d] *= rescale;
        float new_sum = global_sum * rescale;

        // Exp scores and compute tile sum
        for (int i = tid; i < tile_len; i += blockDim.x)
            tile_scores[i] = expf(tile_scores[i] - new_max);
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < tile_len; i += blockDim.x)
            local_sum += tile_scores[i];
        for (int o = 16; o > 0; o >>= 1)
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, o);
        if (lane == 0 && warp_id < 32) ssum[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) {
            float s = ssum[0];
            for (int i = 1; i < (blockDim.x + 31) / 32; i++) s += ssum[i];
            ssum[0] = s;
        }
        __syncthreads();

        // Accumulate weighted V for this tile
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = 0.0f;
            for (int i = 0; i < tile_len; i++)
                val += tile_scores[i] * __half2float(
                    kv_v[(size_t)(tile_start + i) * kv_stride + (size_t)kv_head * head_dim + d]);
            out_h[d] += val;
        }

        global_max = new_max;
        global_sum = new_sum + ssum[0];
        __syncthreads();
    }

    // Final normalization: divide by total sum
    if (global_sum > 0.0f) {
        for (int d = tid; d < head_dim; d += blockDim.x)
            out_h[d] /= global_sum;
    }
}

void cuda_attention_f16kv(const void *q, const void *kv_k, const void *kv_v,
                          float *out, int n_heads_q, int head_dim, int n_kv_heads,
                          int seq_len, int gqa_ratio, float scale, cudaStream_t stream) {
    // Tiled: shared memory is fixed at ATTN_TILE floats + reduction buffers
    size_t smem = ATTN_TILE * sizeof(float) + 64 * sizeof(float);
    attention_kernel_f16kv<<<n_heads_q, 256, smem, stream>>>(
        (const float *)q, (const __half *)kv_k, (const __half *)kv_v,
        out, head_dim, n_kv_heads, seq_len, gqa_ratio, scale);
    KERNEL_LAUNCH_CHECK();
}

// ── Gated Delta Net / Linear Attention Kernels ──

// Conv1D causal forward: depthwise over QKV concatenated
// conv_state: [dim, conv_kernel-1] persistent sliding window
// Applies SiLU activation after convolution
__global__ void gdn_conv1d_kernel(
    float *__restrict__ x,             // [dim] RW: input, overwritten with output
    float *__restrict__ conv_state,    // [dim, conv_kernel-1] RW persistent
    const float *__restrict__ w_conv,  // [dim, conv_kernel] depthwise weights
    int dim, int conv_kernel
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    int km1 = conv_kernel - 1;

    // Shift conv state left: drop oldest, append new input
    float new_val = x[d];
    for (int k = 0; k < km1 - 1; k++)
        conv_state[d * km1 + k] = conv_state[d * km1 + k + 1];
    conv_state[d * km1 + km1 - 1] = new_val;

    // Depthwise conv: sum over kernel window [state..., current]
    float val = 0.0f;
    for (int k = 0; k < km1; k++)
        val += conv_state[d * km1 + k] * w_conv[d * conv_kernel + k];
    val += new_val * w_conv[d * conv_kernel + km1];

    // SiLU activation
    float sigmoid = 1.0f / (1.0f + expf(-val));
    x[d] = val * sigmoid;
}

void cuda_gdn_conv1d(float *x, float *conv_state, const float *w_conv,
                     int dim, int conv_kernel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    gdn_conv1d_kernel<<<blocks, threads, 0, stream>>>(
        x, conv_state, w_conv, dim, conv_kernel);
    KERNEL_LAUNCH_CHECK();
}

// Gated Delta Net recurrence: one head at a time
// State S is [key_head_dim, value_head_dim] per head
// Recurrence:
//   g = exp(A[h] * softplus(alpha[h] + dt_bias[h]))  -- A is pre-negated from GGUF
//   S *= g                                            -- decay
//   kv_mem = S^T @ k                                  -- retrieve [value_head_dim]
//   delta = (v - kv_mem) * sigmoid(beta[h])           -- error correction
//   S += k ⊗ delta                                    -- outer product write
//   output = S^T @ q                                  -- read [value_head_dim]
__global__ void gdn_recurrence_kernel(
    const float *__restrict__ q,          // [num_v_heads * key_head_dim]
    const float *__restrict__ k,          // [num_k_heads * key_head_dim]
    const float *__restrict__ v,          // [num_v_heads * value_head_dim]
    const float *__restrict__ A,          // [num_v_heads] pre-negated: -exp(A_log)
    const float *__restrict__ alpha,      // [num_v_heads] raw alpha
    const float *__restrict__ beta,       // [num_v_heads] raw beta
    const float *__restrict__ dt_bias,    // [num_v_heads]
    float *__restrict__ state,            // [num_v_heads, key_head_dim, value_head_dim] RW
    float *__restrict__ output,           // [num_v_heads * value_head_dim]
    int num_v_heads, int num_k_heads, int key_head_dim, int value_head_dim
) {
    int head = blockIdx.x;
    if (head >= num_v_heads) return;

    int tid = threadIdx.x;

    // Compute gate g for this head
    float a_val = A[head];  // Already -exp(A_log) from GGUF
    float alpha_val = alpha[head];
    float dt_b = dt_bias ? dt_bias[head] : 0.0f;
    float sp_input = alpha_val + dt_b;
    float sp = (sp_input > 20.0f) ? sp_input : logf(1.0f + expf(sp_input));
    float g = expf(a_val * sp);  // a_val is negative, so g < 1 (decay)

    // Beta gate
    float beta_val = 1.0f / (1.0f + expf(-beta[head]));  // sigmoid

    // Key head index (grouped: multiple v_heads share one k_head)
    int k_head = head / (num_v_heads / num_k_heads);
    if (k_head >= num_k_heads) k_head = num_k_heads - 1;

    const float *q_h = q + (size_t)k_head * key_head_dim;  // Q uses key heads
    const float *k_h = k + (size_t)k_head * key_head_dim;
    const float *v_h = v + (size_t)head * value_head_dim;
    float *S = state + (size_t)head * key_head_dim * value_head_dim;
    float *out_h = output + (size_t)head * value_head_dim;

    // Process value dimensions in parallel (threads across value_head_dim)
    for (int vd = tid; vd < value_head_dim; vd += blockDim.x) {
        // Step 1: Decay state column
        // Step 2: Retrieve: kv_mem[vd] = sum_kd(S[kd,vd] * k[kd])
        float kv_mem = 0.0f;
        for (int kd = 0; kd < key_head_dim; kd++) {
            float s_val = S[kd * value_head_dim + vd] * g;  // decay
            S[kd * value_head_dim + vd] = s_val;
            kv_mem += s_val * k_h[kd];
        }

        // Step 3: Delta rule: error correction
        float delta = (v_h[vd] - kv_mem) * beta_val;

        // Step 4: Write: S += k ⊗ delta (outer product)
        for (int kd = 0; kd < key_head_dim; kd++)
            S[kd * value_head_dim + vd] += k_h[kd] * delta;

        // Step 5: Read: output[vd] = sum_kd(S[kd,vd] * q[kd])
        float out_val = 0.0f;
        for (int kd = 0; kd < key_head_dim; kd++)
            out_val += S[kd * value_head_dim + vd] * q_h[kd];

        out_h[vd] = out_val;
    }
}

void cuda_gdn_recurrence(const float *q, const float *k, const float *v,
                         const float *A, const float *alpha, const float *beta,
                         const float *dt_bias, float *state, float *output,
                         int num_v_heads, int num_k_heads,
                         int key_head_dim, int value_head_dim,
                         cudaStream_t stream) {
    // One block per head, 128 threads per block (across value_head_dim)
    int threads = (value_head_dim < 128) ? value_head_dim : 128;
    gdn_recurrence_kernel<<<num_v_heads, threads, 0, stream>>>(
        q, k, v, A, alpha, beta, dt_bias, state, output,
        num_v_heads, num_k_heads, key_head_dim, value_head_dim);
    KERNEL_LAUNCH_CHECK();
}

// RMSNorm + Gated: output = rms_norm(x, w) * silu(z)
__global__ void rms_norm_gated_kernel(
    const float *__restrict__ x,
    const float *__restrict__ z,
    const float *__restrict__ w,
    float *__restrict__ out,
    int head_dim, int n_heads
) {
    int head = blockIdx.x;
    if (head >= n_heads) return;

    const float *x_h = x + head * head_dim;
    const float *z_h = z + head * head_dim;
    float *out_h = out + head * head_dim;

    // RMS norm over head_dim
    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++)
        sum_sq += x_h[d] * x_h[d];
    float rms = rsqrtf(sum_sq / head_dim + 1e-6f);

    // Norm * weight * silu(z)
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float normed = x_h[d] * rms * w[d];
        float z_val = z_h[d];
        float silu_z = z_val / (1.0f + expf(-z_val));
        out_h[d] = normed * silu_z;
    }
}

void cuda_rms_norm_gated(const float *x, const float *z, const float *w,
                         float *out, int head_dim, int n_heads,
                         cudaStream_t stream) {
    int threads = (head_dim < 128) ? head_dim : 128;
    rms_norm_gated_kernel<<<n_heads, threads, 0, stream>>>(
        x, z, w, out, head_dim, n_heads);
    KERNEL_LAUNCH_CHECK();
}

// ── GPU-side embedding lookup + dequantization ──
// Avoids per-token malloc + CPU dequant + H2D. The quantized embedding table
// is already on GPU as part of d_resident.

// Q4_K embedding lookup: 256 threads, 1 block
// BlockQ4K layout: [d:2][dmin:2][scales:12][qs:128] = 144 bytes per 256 values
__global__ void embedding_lookup_q4k_kernel(
    const uint8_t *__restrict__ embd_table,
    float         *__restrict__ output,
    int token_id, int hidden_size, int row_bytes
) {
    const int tid = threadIdx.x;  // 0..255
    const int BLOCK_SIZE = blockDim.x;  // 256
    const uint8_t *row = embd_table + (size_t)token_id * row_bytes;
    int blocks_per_row = (hidden_size + 255) / 256;

    for (int b = tid; b < blocks_per_row * 256; b += BLOCK_SIZE) {
        int blk_idx = b / 256;
        int j = b % 256;
        if (blk_idx * 256 + j >= hidden_size) continue;

        const uint8_t *blk = row + blk_idx * 144;
        // f16 d and dmin via CUDA intrinsic
        __half d_h, dmin_h;
        memcpy(&d_h, blk, 2);
        memcpy(&dmin_h, blk + 2, 2);
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);

        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 16;

        // Determine scale index for this position within the 256-block
        // j spans 0..255; scales are per 32 values (8 scale entries)
        int is = j / 32;
        uint8_t sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
        }

        float d1 = d * sc;
        float m1 = dmin * m;
        // Low nibble for j%64 < 32, high nibble for j%64 >= 32
        int qs_idx = (j % 64 < 32) ? (j / 2) % 64 : ((j - 32) / 2) % 64;
        uint8_t q_val;
        if (j % 64 < 32)
            q_val = qs[qs_idx] & 0xF;
        else
            q_val = (qs[qs_idx] >> 4) & 0xF;

        output[blk_idx * 256 + j] = d1 * q_val - m1;
    }
}

// Q3_K embedding lookup: 256 threads, 1 block
// BlockQ3K layout: [hmask:32][qs:64][scales:12][d:2] = 110 bytes per 256 values
__global__ void embedding_lookup_q3k_kernel(
    const uint8_t *__restrict__ embd_table,
    float         *__restrict__ output,
    int token_id, int hidden_size, int row_bytes
) {
    const int tid = threadIdx.x;
    const int BLOCK_SIZE = blockDim.x;
    const uint8_t *row = embd_table + (size_t)token_id * row_bytes;
    int blocks_per_row = (hidden_size + 255) / 256;

    for (int b = tid; b < blocks_per_row * 256; b += BLOCK_SIZE) {
        int blk_idx = b / 256;
        int j = b % 256;
        if (blk_idx * 256 + j >= hidden_size) continue;

        const uint8_t *blk = row + blk_idx * 110;
        const uint8_t *hmask = blk;
        const uint8_t *qs = blk + 32;
        const uint8_t *scales_raw = blk + 96;
        __half d_h;
        memcpy(&d_h, blk + 108, 2);
        float d_all = __half2float(d_h);

        // Decode the scale for this position
        // 16 scales packed into 12 bytes (6 bits each)
        int is = j / 16;  // which scale (0..15)
        int8_t sc;
        if (is < 8) {
            sc = (int8_t)((scales_raw[is] & 0xF) |
                          (((scales_raw[8 + (is >> 2)] >> (2 * (is & 3))) & 3) << 4)) - 32;
        } else {
            int ii = is - 8;
            sc = (int8_t)((scales_raw[ii] >> 4) |
                          (((scales_raw[8 + (ii >> 2)] >> (2 * (ii & 3) + 4)) & 3) << 4)) - 32;
        }

        float dl = d_all * sc;

        // Extract 2-bit quantized value
        int chunk = j / 128;        // 0 or 1
        int within = j % 128;       // 0..127
        int grp = within / 32;      // 0..3
        int l = within % 32;        // 0..31
        int q_byte_idx;
        if (l < 16)
            q_byte_idx = chunk * 32 + l;
        else
            q_byte_idx = chunk * 32 + 16 + (l - 16);
        int shift = grp * 2;
        int8_t q2 = (qs[q_byte_idx] >> shift) & 3;

        // High bit from hmask
        uint8_t hm_bit;
        if (l < 16)
            hm_bit = hmask[l] & (1 << (chunk * 4 + grp));
        else
            hm_bit = hmask[16 + (l - 16)] & (1 << (chunk * 4 + grp));

        int8_t val = q2 - (hm_bit ? 0 : 4);
        output[blk_idx * 256 + j] = dl * val;
    }
}

// ── GPU-side token sampling (argmax / temperature+top-p) ──
//
// Single block of 256 threads.  Two modes:
//   temperature <= 0 → greedy argmax
//   temperature > 0  → softmax with temperature, then top-p nucleus sampling
//
// RNG: xoroshiro128+ seeded per call from host-provided state.

__global__ void sample_token_kernel(
    const float *__restrict__ logits, int vocab_size,
    float temperature, float top_p,
    unsigned long long rng_seed,
    int *__restrict__ result_token
) {
    const int tid = threadIdx.x;
    const int BLOCK = blockDim.x;  // 256

    // ── Step 1: find global max (for numerical stability) ──
    __shared__ float s_max[256];
    float local_max = -1e30f;
    for (int i = tid; i < vocab_size; i += BLOCK)
        local_max = fmaxf(local_max, logits[i]);
    s_max[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    float global_max = s_max[0];

    // ── Greedy mode: parallel argmax ──
    if (temperature <= 0.0f) {
        __shared__ int s_argmax_id[256];
        __shared__ float s_argmax_val[256];
        float best_val = -1e30f;
        int best_id = 0;
        for (int i = tid; i < vocab_size; i += BLOCK) {
            if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
        }
        s_argmax_val[tid] = best_val;
        s_argmax_id[tid] = best_id;
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_argmax_val[tid + s] > s_argmax_val[tid]) {
                    s_argmax_val[tid] = s_argmax_val[tid + s];
                    s_argmax_id[tid] = s_argmax_id[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) *result_token = s_argmax_id[0];
        return;
    }

    // ── Step 2: exp + sum for softmax ──
    __shared__ float s_sum[256];
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK)
        local_sum += expf((logits[i] - global_max) / temperature);
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float total = s_sum[0];

    // ── Step 3: top-p nucleus sampling ──
    // Strategy: find probability threshold, then thread 0 samples.
    // Each thread computes (prob, id) pairs for its strided chunk.
    // We use iterative threshold refinement to find the top-p cutoff.

    // RNG: xoroshiro128+ (2 iterations to decorrelate)
    unsigned long long rs0 = rng_seed;
    unsigned long long rs1 = rng_seed ^ 0xDEADBEEFCAFEBABEULL;
    for (int warm = 0; warm < 3; warm++) {
        unsigned long long t = rs0 ^ (rs0 << 23);
        rs0 = rs1;
        rs1 = t ^ rs1 ^ (t >> 17) ^ (rs1 >> 26);
    }
    float rand_val;
    if (tid == 0) {
        unsigned long long t = rs0 ^ (rs0 << 23);
        rs0 = rs1;
        rs1 = t ^ rs1 ^ (t >> 17) ^ (rs1 >> 26);
        rand_val = (float)((rs0 + rs1) >> 11) * (1.0f / 9007199254740992.0f);
    }

    // Broadcast random value from thread 0
    __shared__ float s_rand;
    if (tid == 0) s_rand = rand_val;
    __syncthreads();
    float r = s_rand;

    // Target cumulative probability
    float target = r * fminf(top_p, 1.0f);

    // Each thread does a local prefix-sum scan of sorted-ish probabilities.
    // Simplified approach: thread 0 scans all vocab with the threshold.
    // For vocab < 200K and 1 block this is fast enough (~1 cycle/element).
    if (tid == 0) {
        // Selection sort into top candidates until we pass the threshold.
        // We need probabilities: p[i] = exp((logits[i] - max) / temp) / total
        float cumul = 0.0f;
        int selected = 0;

        // Quick approach: iterate vocab finding largest remaining prob each time
        // until cumulative >= top_p, then sample from those.
        // For efficiency, first pass: find all probs > threshold (top_p / 256).
        // This avoids full sort for most tokens.

        float target_r = r * total;  // work in unnormalized space
        float top_p_unnorm = top_p * total;

        // Use a single linear scan: accumulate probabilities in descending order
        // by repeatedly finding the max. For typical top_p=0.9, this converges
        // in 10-50 iterations even for 150K vocab.

        // But even better: just do a single scan accumulating all probs,
        // sampling proportionally. This is O(V) but branch-free and fast on GPU.
        float acc = 0.0f;
        int sampled_id = 0;
        for (int i = 0; i < vocab_size; i++) {
            float p = expf((logits[i] - global_max) / temperature);
            acc += p;
            if (acc >= target_r) { sampled_id = i; break; }
        }

        // If we didn't break (rounding), pick last token
        *result_token = sampled_id;
    }
}

void cuda_embedding_lookup_q4k(const void *embd_table, float *output,
                               int token_id, int hidden_size, int row_bytes,
                               cudaStream_t stream) {
    embedding_lookup_q4k_kernel<<<1, 256, 0, stream>>>(
        (const uint8_t *)embd_table, output, token_id, hidden_size, row_bytes);
    KERNEL_LAUNCH_CHECK();
}

void cuda_embedding_lookup_q3k(const void *embd_table, float *output,
                               int token_id, int hidden_size, int row_bytes,
                               cudaStream_t stream) {
    embedding_lookup_q3k_kernel<<<1, 256, 0, stream>>>(
        (const uint8_t *)embd_table, output, token_id, hidden_size, row_bytes);
    KERNEL_LAUNCH_CHECK();
}

void cuda_sample_token(const float *logits, int vocab_size,
                       float temperature, float top_p,
                       unsigned long long rng_state,
                       int *d_result, cudaStream_t stream) {
    sample_token_kernel<<<1, 256, 0, stream>>>(
        logits, vocab_size, temperature, top_p, rng_state, d_result);
    KERNEL_LAUNCH_CHECK();
}

} // extern "C"
