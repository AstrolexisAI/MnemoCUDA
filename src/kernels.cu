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

__global__ void attention_kernel(
    const float *__restrict__ q,        // [n_heads_q * head_dim]
    const float *__restrict__ kv_k,     // [seq_len * n_kv_heads * head_dim]
    const float *__restrict__ kv_v,     // [seq_len * n_kv_heads * head_dim]
    float       *__restrict__ out,      // [n_heads_q * head_dim]
    int head_dim, int n_kv_heads, int seq_len, int gqa_ratio, float scale
) {
    int head = blockIdx.x;          // query head index
    int kv_head = head / gqa_ratio; // GQA: multiple Q heads share one KV head
    int tid = threadIdx.x;

    extern __shared__ float shared[];  // [seq_len] for scores + [seq_len] for softmax
    float *scores = shared;

    const float *qh = q + head * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    // Phase 1: Q @ K^T — each thread handles positions tid, tid+blockDim, ...
    for (int p = tid; p < seq_len; p += blockDim.x) {
        const float *kp = kv_k + (size_t)p * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += qh[d] * kp[d];
        scores[p] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Softmax over scores[0..seq_len-1]
    // Find max (parallel reduction in shared mem)
    float local_max = -1e30f;
    for (int p = tid; p < seq_len; p += blockDim.x)
        if (scores[p] > local_max) local_max = scores[p];

    // Reduce max across threads (use warp shuffles + shared)
    __shared__ float smax[32]; // one per warp
    int warp_id = tid / 32;
    int lane = tid % 32;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    if (lane == 0) smax[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = smax[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; i++)
            m = fmaxf(m, smax[i]);
        smax[0] = m;
    }
    __syncthreads();
    float max_val = smax[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int p = tid; p < seq_len; p += blockDim.x) {
        scores[p] = expf(scores[p] - max_val);
        local_sum += scores[p];
    }
    // Reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    __shared__ float ssum[32];
    if (lane == 0) ssum[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = ssum[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; i++)
            s += ssum[i];
        ssum[0] = s;
    }
    __syncthreads();
    float total_sum = ssum[0];

    // Normalize
    for (int p = tid; p < seq_len; p += blockDim.x)
        scores[p] /= total_sum;
    __syncthreads();

    // Phase 3: Weighted sum of V — each thread handles some dims
    float *out_h = out + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int p = 0; p < seq_len; p++) {
            const float *vp = kv_v + (size_t)p * kv_stride + kv_head * head_dim;
            val += scores[p] * vp[d];
        }
        out_h[d] = val;
    }
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
// Single-thread top-K + softmax (no race conditions)
__global__ void topk_softmax_kernel(
    const float *__restrict__ scores,
    int         *__restrict__ indices,
    float       *__restrict__ weights,
    int n_experts, int k
) {
    float top_vals[16];
    int   top_ids[16];
    for (int i = 0; i < k; i++) { top_vals[i] = -1e30f; top_ids[i] = -1; }
    for (int j = 0; j < n_experts; j++) {
        float val = scores[j];
        int mi = 0;
        for (int i = 1; i < k; i++) if (top_vals[i] < top_vals[mi]) mi = i;
        if (val > top_vals[mi]) { top_vals[mi] = val; top_ids[mi] = j; }
    }
    float mx = top_vals[0];
    for (int i = 1; i < k; i++) if (top_vals[i] > mx) mx = top_vals[i];
    float sum = 0;
    for (int i = 0; i < k; i++) { top_vals[i] = expf(top_vals[i] - mx); sum += top_vals[i]; }
    for (int i = 0; i < k; i++) { weights[i] = top_vals[i] / sum; indices[i] = top_ids[i]; }
}

// ── Host-callable wrapper functions ──

extern "C" {

void cuda_matvec_q4k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q4k<<<blocks, threads, 0, stream>>>(
        (const BlockQ4K *)weights, x, y, n_rows, n_cols);
}

void cuda_matvec_q6k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q6k<<<blocks, threads, 0, stream>>>(
        (const BlockQ6K *)weights, x, y, n_rows, n_cols);
}

void cuda_matvec_q5k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q5k<<<blocks, threads, 0, stream>>>(
        (const BlockQ5K *)weights, x, y, n_rows, n_cols);
}

void cuda_matvec_q3k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q3k<<<blocks, threads, 0, stream>>>(
        (const BlockQ3K *)weights, x, y, n_rows, n_cols);
}

void cuda_matvec_q8_0(const void *weights, const float *x, float *y,
                      int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q8_0<<<blocks, threads, 0, stream>>>(
        (const BlockQ8_0 *)weights, x, y, n_rows, n_cols);
}

void cuda_swiglu(const float *gate, const float *up, float *out,
                 int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

void cuda_rms_norm(const float *input, const float *weight, float *output,
                   int n, float eps, cudaStream_t stream) {
    int threads = 256;
    rms_norm_kernel<<<1, threads, threads * sizeof(float), stream>>>(
        input, weight, output, n, eps);
}

void cuda_residual_add(const float *a, const float *b, float *out,
                       int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

void cuda_moe_combine(const float *expert_outs, const float *weights_arr,
                      const float *residual, float *output,
                      int hidden, int k, cudaStream_t stream) {
    int threads = 256;
    int blocks = (hidden + threads - 1) / threads;
    moe_combine_kernel<<<blocks, threads, 0, stream>>>(
        expert_outs, weights_arr, residual, output, hidden, k);
}

void cuda_rope(float *q, float *k, int head_dim, int pos, float theta,
               int n_heads_q, int n_heads_k, cudaStream_t stream) {
    int total = (n_heads_q + n_heads_k) * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k, head_dim, pos, theta, n_heads_q, n_heads_k);
}

void cuda_matvec_f32(const float *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    matvec_f32<<<blocks, threads, 0, stream>>>(weights, x, y, n_rows, n_cols);
}

void cuda_attention(const float *q, const float *kv_k, const float *kv_v,
                    float *out, int n_heads_q, int head_dim, int n_kv_heads,
                    int seq_len, int gqa_ratio, float scale, cudaStream_t stream) {
    // One block per query head, 256 threads per block
    int threads = 256;
    size_t smem = seq_len * sizeof(float) + 32 * sizeof(float) * 2; // scores + smax + ssum
    attention_kernel<<<n_heads_q, threads, smem, stream>>>(
        q, kv_k, kv_v, out, head_dim, n_kv_heads, seq_len, gqa_ratio, scale);
}

void cuda_scaled_add(float *out, const float *x, float scale, int n,
                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scaled_add_kernel<<<blocks, threads, 0, stream>>>(out, x, scale, n);
}

void cuda_topk_softmax(const float *scores, int *indices, float *weights,
                       int n_experts, int k, cudaStream_t stream) {
    topk_softmax_kernel<<<1, 1, 0, stream>>>(scores, indices, weights, n_experts, k);
}

// ── FP32 → FP16 conversion ──

__global__ void f32_to_f16_kernel(const float *in, __half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

void cuda_f32_to_f16(const float *in, void *out, int n, cudaStream_t stream) {
    f32_to_f16_kernel<<<(n+255)/256, 256, 0, stream>>>(in, (__half *)out, n);
}

// ── Attention with FP16 KV cache ──

__global__ void attention_kernel_f16kv(
    const float *__restrict__ q, const __half *__restrict__ kv_k,
    const __half *__restrict__ kv_v, float *__restrict__ out,
    int head_dim, int n_kv_heads, int seq_len, int gqa_ratio, float scale
) {
    int head = blockIdx.x, kv_head = head / gqa_ratio, tid = threadIdx.x;
    extern __shared__ float shared[];
    float *scores = shared;
    const float *qh = q + head * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    for (int p = tid; p < seq_len; p += blockDim.x) {
        const __half *kp = kv_k + (size_t)p * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qh[d] * __half2float(kp[d]);
        scores[p] = dot * scale;
    }
    __syncthreads();

    // Softmax reduction
    float local_max = -1e30f;
    for (int p = tid; p < seq_len; p += blockDim.x)
        if (scores[p] > local_max) local_max = scores[p];
    __shared__ float smax[32], ssum[32];
    int warp_id = tid / 32, lane = tid % 32;
    for (int o = 16; o > 0; o >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    if (lane == 0) smax[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) { float m = smax[0]; for (int i = 1; i < (blockDim.x+31)/32; i++) m = fmaxf(m, smax[i]); smax[0] = m; }
    __syncthreads();
    float local_sum = 0.0f;
    for (int p = tid; p < seq_len; p += blockDim.x) { scores[p] = expf(scores[p] - smax[0]); local_sum += scores[p]; }
    for (int o = 16; o > 0; o >>= 1) local_sum += __shfl_xor_sync(0xffffffff, local_sum, o);
    if (lane == 0) ssum[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) { float s = ssum[0]; for (int i = 1; i < (blockDim.x+31)/32; i++) s += ssum[i]; ssum[0] = s; }
    __syncthreads();
    for (int p = tid; p < seq_len; p += blockDim.x) scores[p] /= ssum[0];
    __syncthreads();

    // Weighted V sum
    float *out_h = out + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int p = 0; p < seq_len; p++)
            val += scores[p] * __half2float(kv_v[(size_t)p * kv_stride + kv_head * head_dim + d]);
        out_h[d] = val;
    }
}

void cuda_attention_f16kv(const void *q, const void *kv_k, const void *kv_v,
                          float *out, int n_heads_q, int head_dim, int n_kv_heads,
                          int seq_len, int gqa_ratio, float scale, cudaStream_t stream) {
    size_t smem = seq_len * sizeof(float) + 64 * sizeof(float);
    attention_kernel_f16kv<<<n_heads_q, 256, smem, stream>>>(
        (const float *)q, (const __half *)kv_k, (const __half *)kv_v,
        out, head_dim, n_kv_heads, seq_len, gqa_ratio, scale);
}

} // extern "C"
