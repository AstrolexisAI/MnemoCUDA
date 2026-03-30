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
    // Vectorized Q4K matvec: 32 lanes load qs[128] as 32×uint32_t in 1 coalesced
    // transaction (128 bytes). Each lane unpacks 4 bytes = 8 Q4K values (lo+hi nibbles).
    // No branches in inner loop, fully unrolled per group.
    __shared__ float s_tile[256];

    int warps_per_block = blockDim.x / 32;
    int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    int blocks_per_row = (n_cols + 255) / 256;

    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        int base = b * 256;
        for (int i = threadIdx.x; i < 256 && (base + i) < n_cols; i += blockDim.x)
            s_tile[i] = x[base + i];
        __syncthreads();

        if (row < n_rows) {
            const BlockQ4K *blk_ptr = &weights[(size_t)row * blocks_per_row + b];
            float d   = f16_to_f32(blk_ptr->d);
            float dmin = f16_to_f32(blk_ptr->dmin);

            // Vectorized load: 32 lanes × 4 bytes = 128 bytes in 1 transaction
            uint32_t my_qs = ((const uint32_t *)blk_ptr->qs)[lane];
            uint8_t q0 = my_qs & 0xFF;
            uint8_t q1 = (my_qs >> 8) & 0xFF;
            uint8_t q2 = (my_qs >> 16) & 0xFF;
            uint8_t q3 = (my_qs >> 24) & 0xFF;

            // Lane k owns qs[4k..4k+3]. Group = lane/8, pos_in_group = (lane%8)*4
            int grp = lane / 8;
            int pos = (lane % 8) * 4;

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(grp * 2,     blk_ptr->scales, sc_lo, m_lo);
            get_scale_min_k4(grp * 2 + 1, blk_ptr->scales, sc_hi, m_hi);
            float d_lo = d * sc_lo, mf_lo = dmin * m_lo;
            float d_hi = d * sc_hi, mf_hi = dmin * m_hi;

            // Low nibbles: values at grp*64 + pos + {0,1,2,3}
            int lo_base = grp * 64 + pos;
            if (lo_base + 3 < n_cols - base) {
                sum += (d_lo * (q0 & 0xF) - mf_lo) * s_tile[lo_base];
                sum += (d_lo * (q1 & 0xF) - mf_lo) * s_tile[lo_base + 1];
                sum += (d_lo * (q2 & 0xF) - mf_lo) * s_tile[lo_base + 2];
                sum += (d_lo * (q3 & 0xF) - mf_lo) * s_tile[lo_base + 3];
            }

            // High nibbles: values at grp*64 + 32 + pos + {0,1,2,3}
            int hi_base = grp * 64 + 32 + pos;
            if (hi_base + 3 < n_cols - base) {
                sum += (d_hi * (q0 >> 4) - mf_hi) * s_tile[hi_base];
                sum += (d_hi * (q1 >> 4) - mf_hi) * s_tile[hi_base + 1];
                sum += (d_hi * (q2 >> 4) - mf_hi) * s_tile[hi_base + 2];
                sum += (d_hi * (q3 >> 4) - mf_hi) * s_tile[hi_base + 3];
            }
        }
        __syncthreads();
    }

    if (row < n_rows) {
        sum = warp_reduce_sum(sum);
        if (lane == 0) y[row] = sum;
    }
}

// ── Q4_K Matvec + Residual Add (O_proj + skip connection fused) ──
// y[row] = dot(weights[row], x) + residual[row]
__global__ void matvec_q4k_add(
    const BlockQ4K *__restrict__ weights,
    const float    *__restrict__ x,
    const float    *__restrict__ residual,
    float          *__restrict__ y,
    int n_rows, int n_cols
) {
    __shared__ float s_tile[256];
    int warps_per_block = blockDim.x / 32;
    int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    int blocks_per_row = (n_cols + 255) / 256;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        int base = b * 256;
        for (int i = threadIdx.x; i < 256 && (base + i) < n_cols; i += blockDim.x)
            s_tile[i] = x[base + i];
        __syncthreads();
        if (row < n_rows) {
            const BlockQ4K *blk_ptr = &weights[(size_t)row * blocks_per_row + b];
            float d = f16_to_f32(blk_ptr->d), dmin = f16_to_f32(blk_ptr->dmin);
            uint32_t my_qs = ((const uint32_t *)blk_ptr->qs)[lane];
            int grp = lane / 8, pos = (lane % 8) * 4;
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(grp * 2,     blk_ptr->scales, sc_lo, m_lo);
            get_scale_min_k4(grp * 2 + 1, blk_ptr->scales, sc_hi, m_hi);
            float d_lo = d * sc_lo, mf_lo = dmin * m_lo;
            float d_hi = d * sc_hi, mf_hi = dmin * m_hi;
            int lo_base = grp * 64 + pos;
            if (lo_base + 3 < n_cols - base) {
                sum += (d_lo * ((my_qs)       & 0xF) - mf_lo) * s_tile[lo_base];
                sum += (d_lo * ((my_qs >> 8)  & 0xF) - mf_lo) * s_tile[lo_base + 1];
                sum += (d_lo * ((my_qs >> 16) & 0xF) - mf_lo) * s_tile[lo_base + 2];
                sum += (d_lo * ((my_qs >> 24) & 0xF) - mf_lo) * s_tile[lo_base + 3];
            }
            int hi_base = grp * 64 + 32 + pos;
            if (hi_base + 3 < n_cols - base) {
                sum += (d_hi * ((my_qs >> 4)  & 0xF) - mf_hi) * s_tile[hi_base];
                sum += (d_hi * ((my_qs >> 12) & 0xF) - mf_hi) * s_tile[hi_base + 1];
                sum += (d_hi * ((my_qs >> 20) & 0xF) - mf_hi) * s_tile[hi_base + 2];
                sum += (d_hi * ((my_qs >> 28) & 0xF) - mf_hi) * s_tile[hi_base + 3];
            }
        }
        __syncthreads();
    }
    if (row < n_rows) {
        sum = warp_reduce_sum(sum);
        if (lane == 0) y[row] = sum + residual[row];
    }
}

// ── Q4_K Dual Matvec (gate + up projections in one kernel) ──
// Vectorized uint32_t loads, same pattern as matvec_q4k.
__global__ void matvec_q4k_dual(
    const BlockQ4K *__restrict__ weights_a,
    const BlockQ4K *__restrict__ weights_b,
    const float    *__restrict__ x,
    float          *__restrict__ y_a,
    float          *__restrict__ y_b,
    int n_rows, int n_cols
) {
    __shared__ float s_tile[256];
    int warps_per_block = blockDim.x / 32;
    int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    int blocks_per_row = (n_cols + 255) / 256;

    float sum_a = 0.0f, sum_b = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        int base = b * 256;
        for (int i = threadIdx.x; i < 256 && (base + i) < n_cols; i += blockDim.x)
            s_tile[i] = x[base + i];
        __syncthreads();

        if (row < n_rows) {
            const BlockQ4K *ba = &weights_a[(size_t)row * blocks_per_row + b];
            const BlockQ4K *bb = &weights_b[(size_t)row * blocks_per_row + b];
            float da = f16_to_f32(ba->d), dmin_a = f16_to_f32(ba->dmin);
            float db = f16_to_f32(bb->d), dmin_b = f16_to_f32(bb->dmin);

            uint32_t qa4 = ((const uint32_t *)ba->qs)[lane];
            uint32_t qb4 = ((const uint32_t *)bb->qs)[lane];

            int grp = lane / 8;
            int pos = (lane % 8) * 4;

            uint8_t sca_lo, ma_lo, sca_hi, ma_hi;
            uint8_t scb_lo, mb_lo, scb_hi, mb_hi;
            get_scale_min_k4(grp * 2,     ba->scales, sca_lo, ma_lo);
            get_scale_min_k4(grp * 2 + 1, ba->scales, sca_hi, ma_hi);
            get_scale_min_k4(grp * 2,     bb->scales, scb_lo, mb_lo);
            get_scale_min_k4(grp * 2 + 1, bb->scales, scb_hi, mb_hi);
            float da_lo = da * sca_lo, mfa_lo = dmin_a * ma_lo;
            float da_hi = da * sca_hi, mfa_hi = dmin_a * ma_hi;
            float db_lo = db * scb_lo, mfb_lo = dmin_b * mb_lo;
            float db_hi = db * scb_hi, mfb_hi = dmin_b * mb_hi;

            int lo_base = grp * 64 + pos;
            if (lo_base + 3 < n_cols - base) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    uint8_t va = (qa4 >> (k * 8)) & 0xF;
                    uint8_t vb = (qb4 >> (k * 8)) & 0xF;
                    float xv = s_tile[lo_base + k];
                    sum_a += (da_lo * va - mfa_lo) * xv;
                    sum_b += (db_lo * vb - mfb_lo) * xv;
                }
            }
            int hi_base = grp * 64 + 32 + pos;
            if (hi_base + 3 < n_cols - base) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    uint8_t va = (qa4 >> (k * 8 + 4)) & 0xF;
                    uint8_t vb = (qb4 >> (k * 8 + 4)) & 0xF;
                    float xv = s_tile[hi_base + k];
                    sum_a += (da_hi * va - mfa_hi) * xv;
                    sum_b += (db_hi * vb - mfb_hi) * xv;
                }
            }
        }
        __syncthreads();
    }

    if (row < n_rows) {
        sum_a = warp_reduce_sum(sum_a);
        sum_b = warp_reduce_sum(sum_b);
        if (lane == 0) { y_a[row] = sum_a; y_b[row] = sum_b; }
    }
}

// ── Q4_K Matvec + Scaled Accumulate (fused down-projection + weighted add) ──
// Vectorized uint32_t loads, same pattern as matvec_q4k.
__global__ void matvec_q4k_scaled_add(
    const BlockQ4K *__restrict__ weights,
    const float    *__restrict__ x,
    float          *__restrict__ y,
    int n_rows, int n_cols, float scale
) {
    __shared__ float s_tile[256];
    int warps_per_block = blockDim.x / 32;
    int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    int blocks_per_row = (n_cols + 255) / 256;

    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        int base = b * 256;
        for (int i = threadIdx.x; i < 256 && (base + i) < n_cols; i += blockDim.x)
            s_tile[i] = x[base + i];
        __syncthreads();

        if (row < n_rows) {
            const BlockQ4K *blk_ptr = &weights[(size_t)row * blocks_per_row + b];
            float d   = f16_to_f32(blk_ptr->d);
            float dmin = f16_to_f32(blk_ptr->dmin);

            uint32_t my_qs = ((const uint32_t *)blk_ptr->qs)[lane];
            int grp = lane / 8;
            int pos = (lane % 8) * 4;

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(grp * 2,     blk_ptr->scales, sc_lo, m_lo);
            get_scale_min_k4(grp * 2 + 1, blk_ptr->scales, sc_hi, m_hi);
            float d_lo = d * sc_lo, mf_lo = dmin * m_lo;
            float d_hi = d * sc_hi, mf_hi = dmin * m_hi;

            int lo_base = grp * 64 + pos;
            if (lo_base + 3 < n_cols - base) {
                sum += (d_lo * ((my_qs)       & 0xF) - mf_lo) * s_tile[lo_base];
                sum += (d_lo * ((my_qs >> 8)  & 0xF) - mf_lo) * s_tile[lo_base + 1];
                sum += (d_lo * ((my_qs >> 16) & 0xF) - mf_lo) * s_tile[lo_base + 2];
                sum += (d_lo * ((my_qs >> 24) & 0xF) - mf_lo) * s_tile[lo_base + 3];
            }
            int hi_base = grp * 64 + 32 + pos;
            if (hi_base + 3 < n_cols - base) {
                sum += (d_hi * ((my_qs >> 4)  & 0xF) - mf_hi) * s_tile[hi_base];
                sum += (d_hi * ((my_qs >> 12) & 0xF) - mf_hi) * s_tile[hi_base + 1];
                sum += (d_hi * ((my_qs >> 20) & 0xF) - mf_hi) * s_tile[hi_base + 2];
                sum += (d_hi * ((my_qs >> 28) & 0xF) - mf_hi) * s_tile[hi_base + 3];
            }
        }
        __syncthreads();
    }

    if (row < n_rows) {
        sum = warp_reduce_sum(sum);
        if (lane == 0) y[row] += scale * sum;
    }
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
    extern __shared__ float s_x[];
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) s_x[i] = x[i];
    __syncthreads();
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
                    sum += scale * val * s_x[base + val_idx];
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
    extern __shared__ float s_x[];
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) s_x[i] = x[i];
    __syncthreads();
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
                        sum += dl * val * s_x[idx];
                    }
                    val_idx++;
                }
                float dl2 = d_all * sc16[is++];
                for (int l = 0; l < 16; l++) {
                    int idx = base + val_idx;
                    if (idx < n_cols) {
                        int8_t q2 = (q[l + 16] >> shift) & 3;
                        int8_t val = q2 - ((blk.hmask[l + 16] & m) ? 0 : 4);
                        sum += dl2 * val * s_x[idx];
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
    extern __shared__ float s_x[];
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) s_x[i] = x[i];
    __syncthreads();
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
                sum += (d1 * (lo4 | (hi << 4)) - m1f) * s_x[base + j + l];
            }
            for (int l = 0; l < 32 && (base + j + 32 + l) < n_cols; l++) {
                uint8_t lo4 = q[l] >> 4;
                uint8_t hi = (qh[(j + 32 + l) / 8] >> ((j + 32 + l) % 8)) & 1;
                sum += (d2 * (lo4 | (hi << 4)) - m2f) * s_x[base + j + 32 + l];
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
    extern __shared__ float s_x[];
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) s_x[i] = x[i];
    __syncthreads();
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
            sum += d * blk.qs[j] * s_x[base + j];
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

// ── RMS Norm + Residual Save (fused: saves input to residual during norm pass) ──
__global__ void rms_norm_residual_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    float       *__restrict__ normed_output,
    float       *__restrict__ residual_output,
    int n, float eps
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = input[i];
        local_sum += v * v;
        residual_output[i] = v;  // save residual during same read
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        normed_output[i] = input[i] * scale * weight[i];
}

// ── Batched RMS Norm (all heads in one launch) ──
// Each block processes one head: input[block_id * head_dim .. (block_id+1) * head_dim]
// Weight is shared across all heads (same norm weight vector of length head_dim).
__global__ void rms_norm_batched_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    float       *__restrict__ output,
    int head_dim, float eps
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int head = blockIdx.x;
    const float *in  = input  + head * head_dim;
    float       *out = output + head * head_dim;

    float local_sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = in[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x)
        out[i] = in[i] * scale * weight[i];
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
// Frequencies precalculated in constant memory (avoids powf per thread)
__constant__ float d_rope_freqs[256];  // max head_dim/2 = 128
static int d_rope_freqs_init = 0;

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

    float freq = d_rope_freqs[j];
    float angle = (float)pos * freq;
    float sin_a, cos_a;
    __sincosf(angle, &sin_a, &cos_a);

    int idx = head * head_dim + j;
    float v0 = vec[idx];
    float v1 = vec[idx + half_dim];
    vec[idx]            = v0 * cos_a - v1 * sin_a;
    vec[idx + half_dim] = v0 * sin_a + v1 * cos_a;
}

// ── F32 Matrix-Vector Multiply (for router weights) ──
// Warp-parallel: 1 warp (32 threads) per row, each thread accumulates
// n_cols/32 elements, then warp-level shuffle reduction.
// Block = 4 warps = 128 threads → 4 rows per block.

__global__ void matvec_f32(
    const float *__restrict__ weights, // [n_rows × n_cols], row-major
    const float *__restrict__ x,
    float       *__restrict__ y,
    int n_rows, int n_cols
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 4 + warp_id;  // 4 warps per block
    if (row >= n_rows) return;

    const float *row_ptr = weights + (size_t)row * n_cols;
    float sum = 0.0f;
    for (int col = lane; col < n_cols; col += 32)
        sum += row_ptr[col] * x[col];

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0) y[row] = sum;
}

// ── GQA Attention kernel: Q@K^T → softmax → @V (one kernel per head) ──
// One block per query head. Each thread handles a subset of positions.

// Tile size for tiled attention (online softmax). 2048 floats = 8KB smem.
#define ATTN_TILE 2048

// Tiled attention kernel for FP32 KV cache (same online softmax approach).
// Q vector cached in shared memory to avoid repeated global memory reads.
__global__ void attention_kernel(
    const float *__restrict__ q, const float *__restrict__ kv_k,
    const float *__restrict__ kv_v, float *__restrict__ out,
    int head_dim, int n_kv_heads, int seq_len, int gqa_ratio, float scale
) {
    int head = blockIdx.x, kv_head = head / gqa_ratio, tid = threadIdx.x;
    extern __shared__ float shared[];
    float *tile_scores = shared;
    float *s_q = shared + ATTN_TILE;  // Q vector cache
    __shared__ float smax[32], ssum[32];
    int warp_id = tid / 32, lane = tid % 32;
    const float *qh = q + head * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    float *out_h = out + head * head_dim;

    // Cache Q in shared memory
    for (int d = tid; d < head_dim; d += blockDim.x)
        s_q[d] = qh[d];
    __syncthreads();

    float global_max = -1e30f, global_sum = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) out_h[d] = 0.0f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += ATTN_TILE) {
        int tile_end = tile_start + ATTN_TILE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // Q @ K^T: warp-cooperative
        {
            int n_warps = blockDim.x / 32;
            for (int pos_idx = warp_id; pos_idx < tile_len; pos_idx += n_warps) {
                int p = tile_start + pos_idx;
                const float *kp = kv_k + (size_t)p * kv_stride + (size_t)kv_head * head_dim;
                float partial = 0.0f;
                for (int d = lane; d < head_dim; d += 32) partial += s_q[d] * kp[d];
                partial = warp_reduce_sum(partial);
                if (lane == 0) tile_scores[pos_idx] = partial * scale;
            }
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

void cuda_matvec_q4k_add(const void *weights, const float *x, const float *residual,
                         float *y, int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q4k_add<<<blocks, threads, 0, stream>>>(
        (const BlockQ4K *)weights, x, residual, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q4k_dual(const void *wa, const void *wb, const float *x,
                          float *ya, float *yb,
                          int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q4k_dual<<<blocks, threads, 0, stream>>>(
        (const BlockQ4K *)wa, (const BlockQ4K *)wb, x, ya, yb, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q4k_scaled_add(const void *weights, const float *x, float *y,
                                int n_rows, int n_cols, float scale,
                                cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    matvec_q4k_scaled_add<<<blocks, threads, 0, stream>>>(
        (const BlockQ4K *)weights, x, y, n_rows, n_cols, scale);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q6k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    size_t smem = n_cols * sizeof(float);
    matvec_q6k<<<blocks, threads, smem, stream>>>(
        (const BlockQ6K *)weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q5k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    size_t smem = n_cols * sizeof(float);
    matvec_q5k<<<blocks, threads, smem, stream>>>(
        (const BlockQ5K *)weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q3k(const void *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    size_t smem = n_cols * sizeof(float);
    matvec_q3k<<<blocks, threads, smem, stream>>>(
        (const BlockQ3K *)weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_q8_0(const void *weights, const float *x, float *y,
                      int n_rows, int n_cols, cudaStream_t stream) {
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int blocks = (n_rows + warps_per_block - 1) / warps_per_block;
    size_t smem = n_cols * sizeof(float);
    matvec_q8_0<<<blocks, threads, smem, stream>>>(
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

void cuda_rms_norm_residual(const float *input, const float *weight,
                            float *normed_output, float *residual_output,
                            int n, float eps, cudaStream_t stream) {
    int threads = 256;
    rms_norm_residual_kernel<<<1, threads, threads * sizeof(float), stream>>>(
        input, weight, normed_output, residual_output, n, eps);
    KERNEL_LAUNCH_CHECK();
}

void cuda_rms_norm(const float *input, const float *weight, float *output,
                   int n, float eps, cudaStream_t stream) {
    int threads = 256;
    rms_norm_kernel<<<1, threads, threads * sizeof(float), stream>>>(
        input, weight, output, n, eps);
    KERNEL_LAUNCH_CHECK();
}

// Fused Q+K norm: blocks 0..n_q-1 normalize Q, blocks n_q..n_q+n_k-1 normalize K
__global__ void rms_norm_qk_kernel(
    float *__restrict__ q, float *__restrict__ k,
    const float *__restrict__ q_weight, const float *__restrict__ k_weight,
    int head_dim, int n_q, int n_k, float eps
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int head = blockIdx.x;
    float *data;
    const float *weight;
    if (head < n_q) { data = q + head * head_dim; weight = q_weight; }
    else { data = k + (head - n_q) * head_dim; weight = k_weight; }

    float local_sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) { float v = data[i]; local_sum += v * v; }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x)
        data[i] = data[i] * scale * weight[i];
}

void cuda_rms_norm_batched(const float *input, const float *weight, float *output,
                           int head_dim, int n_heads, float eps, cudaStream_t stream) {
    int threads = (head_dim < 256) ? head_dim : 256;
    rms_norm_batched_kernel<<<n_heads, threads, threads * sizeof(float), stream>>>(
        input, weight, output, head_dim, eps);
    KERNEL_LAUNCH_CHECK();
}

void cuda_rms_norm_qk(float *q, float *k,
                       const float *q_weight, const float *k_weight,
                       int head_dim, int n_q, int n_k, float eps, cudaStream_t stream) {
    int threads = (head_dim < 256) ? head_dim : 256;
    rms_norm_qk_kernel<<<n_q + n_k, threads, threads * sizeof(float), stream>>>(
        q, k, q_weight, k_weight, head_dim, n_q, n_k, eps);
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
    // Lazy init: precompute frequencies in constant memory once
    if (!d_rope_freqs_init) {
        float freqs[256];
        int half = head_dim / 2;
        for (int j = 0; j < half && j < 256; j++)
            freqs[j] = 1.0f / powf(theta, (float)(2 * j) / (float)head_dim);
        cudaMemcpyToSymbol(d_rope_freqs, freqs, half * sizeof(float));
        d_rope_freqs_init = 1;
    }
    int total = (n_heads_q + n_heads_k) * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k, head_dim, pos, theta, n_heads_q, n_heads_k);
    KERNEL_LAUNCH_CHECK();
}

void cuda_matvec_f32(const float *weights, const float *x, float *y,
                     int n_rows, int n_cols, cudaStream_t stream) {
    int threads = 128;  // 4 warps, 1 warp per row
    int blocks = (n_rows + 3) / 4;
    matvec_f32<<<blocks, threads, 0, stream>>>(weights, x, y, n_rows, n_cols);
    KERNEL_LAUNCH_CHECK();
}

void cuda_attention(const float *q, const float *kv_k, const float *kv_v,
                    float *out, int n_heads_q, int head_dim, int n_kv_heads,
                    int seq_len, int gqa_ratio, float scale, cudaStream_t stream) {
    int threads = 256;
    size_t smem = (ATTN_TILE + head_dim + 64) * sizeof(float);
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

// Fused dual: converts K and V to FP16 in one launch
__global__ void f32_to_f16_dual_kernel(
    const float *__restrict__ a, const float *__restrict__ b,
    __half *__restrict__ oa, __half *__restrict__ ob, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { oa[i] = __float2half(a[i]); ob[i] = __float2half(b[i]); }
}

void cuda_f32_to_f16_dual(const float *a, const float *b,
                           void *oa, void *ob, int n, cudaStream_t stream) {
    f32_to_f16_dual_kernel<<<(n+255)/256, 256, 0, stream>>>(
        a, b, (__half *)oa, (__half *)ob, n);
    KERNEL_LAUNCH_CHECK();
}

// ── FP32 → INT8 conversion with per-head absmax scaling ──
// For each head: find absmax, scale = absmax / 127, store int8 + scale.
// scales[pos * n_kv_heads + head] = absmax / 127

__global__ void f32_to_int8_kv_kernel(
    const float *__restrict__ in,  // [n_kv_heads * head_dim]
    int8_t      *__restrict__ out, // [n_kv_heads * head_dim]
    float       *__restrict__ scales, // [n_kv_heads] — one scale per head
    int head_dim, int n_kv_heads
) {
    int head = blockIdx.x;
    if (head >= n_kv_heads) return;
    int tid = threadIdx.x;
    const float *head_in = in + head * head_dim;
    int8_t *head_out = out + head * head_dim;

    // Find absmax within this head
    extern __shared__ float smem[];
    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x)
        local_max = fmaxf(local_max, fabsf(head_in[d]));
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float absmax = smem[0];
    float scale_inv = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;

    // Store scale
    if (tid == 0) scales[head] = (absmax > 0.0f) ? absmax / 127.0f : 0.0f;

    // Quantize
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = head_in[d] * scale_inv;
        int iv = __float2int_rn(v);
        if (iv > 127) iv = 127;
        if (iv < -127) iv = -127;
        head_out[d] = (int8_t)iv;
    }
}

// ── Attention with INT8 KV cache ──
// Same online softmax tiled approach as FP16 version, but dequantizes
// K and V from INT8 using per-position per-head scale factors.

__global__ void attention_kernel_int8kv(
    const float  *__restrict__ q,
    const int8_t *__restrict__ kv_k,     // [seq_len, n_kv_heads, head_dim] int8
    const int8_t *__restrict__ kv_v,     // [seq_len, n_kv_heads, head_dim] int8
    const float  *__restrict__ k_scales, // [seq_len, n_kv_heads]
    const float  *__restrict__ v_scales, // [seq_len, n_kv_heads]
    float *__restrict__ out,
    int head_dim, int n_kv_heads, int seq_len, int gqa_ratio, float scale
) {
    int head = blockIdx.x, kv_head = head / gqa_ratio, tid = threadIdx.x;
    extern __shared__ float shared[];
    float *tile_scores = shared;
    float *s_q = shared + ATTN_TILE;
    __shared__ float smax[32], ssum[32];
    int warp_id = tid / 32, lane = tid % 32;
    const float *qh = q + head * head_dim;
    int kv_elem_stride = n_kv_heads * head_dim;
    int kv_scale_stride = n_kv_heads;
    float *out_h = out + head * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x)
        s_q[d] = qh[d];
    __syncthreads();

    float global_max = -1e30f, global_sum = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) out_h[d] = 0.0f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += ATTN_TILE) {
        int tile_end = tile_start + ATTN_TILE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // Q @ K^T with INT8 dequant: warp-cooperative
        {
            int n_warps = blockDim.x / 32;
            for (int pos_idx = warp_id; pos_idx < tile_len; pos_idx += n_warps) {
                int p = tile_start + pos_idx;
                const int8_t *kp = kv_k + (size_t)p * kv_elem_stride + (size_t)kv_head * head_dim;
                float k_sc = k_scales[p * kv_scale_stride + kv_head];
                float partial = 0.0f;
                for (int d = lane; d < head_dim; d += 32)
                    partial += s_q[d] * ((float)kp[d] * k_sc);
                partial = warp_reduce_sum(partial);
                if (lane == 0) tile_scores[pos_idx] = partial * scale;
            }
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
        for (int o = 16; o > 0; o >>= 1) local_sum += __shfl_xor_sync(0xffffffff, local_sum, o);
        if (lane == 0 && warp_id < 32) ssum[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) { float s = ssum[0]; for (int i = 1; i < (blockDim.x+31)/32; i++) s += ssum[i]; ssum[0] = s; }
        __syncthreads();

        // V accumulation with INT8 dequant
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = 0.0f;
            for (int i = 0; i < tile_len; i++) {
                int p = tile_start + i;
                float v_sc = v_scales[p * kv_scale_stride + kv_head];
                val += tile_scores[i] * ((float)kv_v[(size_t)p * kv_elem_stride + (size_t)kv_head * head_dim + d] * v_sc);
            }
            out_h[d] += val;
        }

        global_max = new_max;
        global_sum = new_sum + ssum[0];
        __syncthreads();
    }

    if (global_sum > 0.0f) {
        for (int d = tid; d < head_dim; d += blockDim.x)
            out_h[d] /= global_sum;
    }
}

// ── INT8 KV cache wrappers ──

void cuda_f32_to_int8_kv(const float *in, void *out, float *scales,
                         int head_dim, int n_kv_heads, cudaStream_t stream) {
    int threads = (head_dim < 256) ? head_dim : 256;
    f32_to_int8_kv_kernel<<<n_kv_heads, threads, threads * sizeof(float), stream>>>(
        in, (int8_t *)out, scales, head_dim, n_kv_heads);
    KERNEL_LAUNCH_CHECK();
}

// Fused dual INT8 KV store: K and V in one launch (NKV*2 blocks)
__global__ void f32_to_int8_kv_dual_kernel(
    const float *__restrict__ in_a, const float *__restrict__ in_b,
    int8_t *__restrict__ out_a, int8_t *__restrict__ out_b,
    float *__restrict__ scales_a, float *__restrict__ scales_b,
    int head_dim, int n_kv_heads
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    bool is_b = (head_idx >= n_kv_heads);
    int head = is_b ? (head_idx - n_kv_heads) : head_idx;
    const float *head_in = (is_b ? in_b : in_a) + head * head_dim;
    int8_t *head_out = (is_b ? out_b : out_a) + head * head_dim;
    float *scale_out = is_b ? scales_b : scales_a;

    extern __shared__ float smem[];
    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x)
        local_max = fmaxf(local_max, fabsf(head_in[d]));
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float absmax = smem[0];
    float scale_inv = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;
    if (tid == 0) scale_out[head] = (absmax > 0.0f) ? absmax / 127.0f : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int iv = __float2int_rn(head_in[d] * scale_inv);
        head_out[d] = (int8_t)(iv > 127 ? 127 : (iv < -127 ? -127 : iv));
    }
}

void cuda_f32_to_int8_kv_dual(const float *k, const float *v,
                               void *out_k, void *out_v,
                               float *k_scales, float *v_scales,
                               int head_dim, int n_kv_heads, cudaStream_t stream) {
    int threads = (head_dim < 256) ? head_dim : 256;
    f32_to_int8_kv_dual_kernel<<<n_kv_heads * 2, threads, threads * sizeof(float), stream>>>(
        k, v, (int8_t *)out_k, (int8_t *)out_v, k_scales, v_scales, head_dim, n_kv_heads);
    KERNEL_LAUNCH_CHECK();
}

void cuda_attention_int8kv(const float *q, const void *kv_k, const void *kv_v,
                           const float *k_scales, const float *v_scales,
                           float *out, int n_heads_q, int head_dim, int n_kv_heads,
                           int seq_len, int gqa_ratio, float scale, cudaStream_t stream) {
    size_t smem = (ATTN_TILE + head_dim + 64) * sizeof(float);
    attention_kernel_int8kv<<<n_heads_q, 256, smem, stream>>>(
        q, (const int8_t *)kv_k, (const int8_t *)kv_v, k_scales, v_scales,
        out, head_dim, n_kv_heads, seq_len, gqa_ratio, scale);
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
    float *tile_scores = shared;  // [ATTN_TILE]
    float *s_q = shared + ATTN_TILE;  // [head_dim] — Q vector cached in shared mem
    __shared__ float smax[32], ssum[32];
    int warp_id = tid / 32, lane = tid % 32;
    const float *qh = q + head * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    float *out_h = out + head * head_dim;

    // Cache Q vector in shared memory (avoids repeated global reads)
    for (int d = tid; d < head_dim; d += blockDim.x)
        s_q[d] = qh[d];
    __syncthreads();

    float global_max = -1e30f;
    float global_sum = 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x)
        out_h[d] = 0.0f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += ATTN_TILE) {
        int tile_end = tile_start + ATTN_TILE;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // Q @ K^T: warp-cooperative (32 threads per position, shuffle reduce)
        {
            int n_warps = blockDim.x / 32;
            for (int pos_idx = warp_id; pos_idx < tile_len; pos_idx += n_warps) {
                int p = tile_start + pos_idx;
                const __half *kp = kv_k + (size_t)p * kv_stride + (size_t)kv_head * head_dim;
                float partial = 0.0f;
                for (int d = lane; d < head_dim; d += 32)
                    partial += s_q[d] * __half2float(kp[d]);
                partial = warp_reduce_sum(partial);
                if (lane == 0) tile_scores[pos_idx] = partial * scale;
            }
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
    // Shared: ATTN_TILE (scores) + head_dim (Q cache) + 64 (reduction)
    size_t smem = (ATTN_TILE + head_dim + 64) * sizeof(float);
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
// Parallel reduction for sum-of-squares across all threads.
__global__ void rms_norm_gated_kernel(
    const float *__restrict__ x,
    const float *__restrict__ z,
    const float *__restrict__ w,
    float *__restrict__ out,
    int head_dim, int n_heads
) {
    extern __shared__ float smem[];
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int tid = threadIdx.x;

    const float *x_h = x + head * head_dim;
    const float *z_h = z + head * head_dim;
    float *out_h = out + head * head_dim;

    // Parallel RMS norm reduction
    float local_sum = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x)
        local_sum += x_h[d] * x_h[d];
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(smem[0] / head_dim + 1e-6f);

    // Norm * weight * silu(z)
    for (int d = tid; d < head_dim; d += blockDim.x) {
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
    rms_norm_gated_kernel<<<n_heads, threads, threads * sizeof(float), stream>>>(
        x, z, w, out, head_dim, n_heads);
    KERNEL_LAUNCH_CHECK();
}

// ── GPU-side expert cache classification ──
// One thread per expert index (K threads, K <= 16).
// Checks each expert against the VRAM cache slot table.
// Outputs: hit_slots[e] = slot index if hit (-1 if miss),
//          miss_mask[e] = 1 if miss, 0 if hit.
__global__ void classify_experts_kernel(
    const int *__restrict__ expert_indices,  // K expert IDs from router
    const int *__restrict__ cache_layer,     // [n_slots] — layer per slot
    const int *__restrict__ cache_expert,    // [n_slots] — expert per slot
    int        *__restrict__ hit_slots,      // [K] output: slot index or -1
    int        *__restrict__ miss_mask,      // [K] output: 1=miss, 0=hit
    int layer, int n_slots, int K, int n_experts
) {
    int e = threadIdx.x;
    if (e >= K) return;
    int eid = expert_indices[e];
    if (eid < 0 || eid >= n_experts) {
        hit_slots[e] = -1;
        miss_mask[e] = 0;  // invalid expert, skip
        return;
    }

    // Linear scan of cache slots (n_slots typically 1000-4000)
    for (int s = 0; s < n_slots; s++) {
        if (cache_layer[s] == layer && cache_expert[s] == eid) {
            hit_slots[e] = s;
            miss_mask[e] = 0;
            return;
        }
    }
    hit_slots[e] = -1;
    miss_mask[e] = 1;
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
    // Each thread maintains a local top-8 min-heap from its strided partition,
    // then all 256×8 = 2048 candidates are collected in shared memory.
    // Thread 0 does selection sort (with early stop at top_p) and samples.
    // For typical LLM distributions, top-2048 holds >99.9% of probability mass.

    #define TOPP_LOCAL_K 8
    #define TOPP_TOTAL_CANDS (256 * TOPP_LOCAL_K)  // 2048

    // Thread-local top-8 (registers): min-heap by probability
    float my_vals[TOPP_LOCAL_K];
    int   my_ids[TOPP_LOCAL_K];
    for (int i = 0; i < TOPP_LOCAL_K; i++) { my_vals[i] = -1.0f; my_ids[i] = -1; }

    for (int i = tid; i < vocab_size; i += BLOCK) {
        float p = expf((logits[i] - global_max) / temperature);
        // Find min in local heap
        int mi = 0;
        for (int k = 1; k < TOPP_LOCAL_K; k++)
            if (my_vals[k] < my_vals[mi]) mi = k;
        if (p > my_vals[mi]) { my_vals[mi] = p; my_ids[mi] = i; }
    }

    // Write 8 candidates per thread to shared memory (2048 total)
    // 2048 * (4+4) = 16KB shared memory — within limits
    __shared__ float s_cand_prob[TOPP_TOTAL_CANDS];
    __shared__ int   s_cand_id[TOPP_TOTAL_CANDS];
    for (int k = 0; k < TOPP_LOCAL_K; k++) {
        s_cand_prob[tid * TOPP_LOCAL_K + k] = my_vals[k];
        s_cand_id[tid * TOPP_LOCAL_K + k] = my_ids[k];
    }
    __syncthreads();

    // Thread 0: selection sort descending + early stop at top_p, then sample
    if (tid == 0) {
        // RNG: xoroshiro128+
        unsigned long long rs0 = rng_seed;
        unsigned long long rs1 = rng_seed ^ 0xDEADBEEFCAFEBABEULL;
        for (int warm = 0; warm < 3; warm++) {
            unsigned long long t = rs0 ^ (rs0 << 23);
            rs0 = rs1;
            rs1 = t ^ rs1 ^ (t >> 17) ^ (rs1 >> 26);
        }
        unsigned long long t = rs0 ^ (rs0 << 23);
        rs0 = rs1;
        rs1 = t ^ rs1 ^ (t >> 17) ^ (rs1 >> 26);
        float r = (float)((rs0 + rs1) >> 11) * (1.0f / 9007199254740992.0f);

        float top_p_mass = top_p * total;
        float cumul = 0.0f;
        int nucleus_size = 0;

        // Selection sort: pick largest, swap to front, accumulate, stop at top_p.
        // Worst case scans 2048 candidates per pick, but early stop at nucleus.
        // For top_p=0.9, nucleus is typically 10-100 tokens → 10-100 passes.
        for (int pick = 0; pick < TOPP_TOTAL_CANDS; pick++) {
            // Find largest remaining
            int best = pick;
            for (int j = pick + 1; j < TOPP_TOTAL_CANDS; j++)
                if (s_cand_prob[j] > s_cand_prob[best]) best = j;

            if (s_cand_prob[best] <= 0.0f) break;  // no more valid candidates

            // Swap to front
            float tp = s_cand_prob[pick]; int ti = s_cand_id[pick];
            s_cand_prob[pick] = s_cand_prob[best]; s_cand_id[pick] = s_cand_id[best];
            s_cand_prob[best] = tp; s_cand_id[best] = ti;

            cumul += s_cand_prob[pick];
            nucleus_size = pick + 1;
            if (cumul >= top_p_mass) break;
        }

        // Renormalize nucleus and sample
        float nsum = 0.0f;
        for (int i = 0; i < nucleus_size; i++) nsum += s_cand_prob[i];
        float target = r * nsum;
        float acc = 0.0f;
        int sampled_id = s_cand_id[0];
        for (int i = 0; i < nucleus_size; i++) {
            acc += s_cand_prob[i];
            if (acc >= target) { sampled_id = s_cand_id[i]; break; }
        }
        *result_token = sampled_id;
    }

    #undef TOPP_LOCAL_K
    #undef TOPP_TOTAL_CANDS
}

void cuda_classify_experts(const int *expert_indices,
                           const int *cache_layer, const int *cache_expert,
                           int *hit_slots, int *miss_mask,
                           int layer, int n_slots, int K, int n_experts,
                           cudaStream_t stream) {
    classify_experts_kernel<<<1, K, 0, stream>>>(
        expert_indices, cache_layer, cache_expert,
        hit_slots, miss_mask, layer, n_slots, K, n_experts);
    KERNEL_LAUNCH_CHECK();
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

void cuda_kernels_init(void) {
    // Allow up to 100KB dynamic shared memory for bandwidth-bound matvec kernels.
    // Default is 48KB which limits occupancy to 25% with H=4096 x-vector (16KB).
    cudaFuncSetAttribute(matvec_q4k,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(matvec_q4k_scaled_add,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(matvec_q4k_dual,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(matvec_q6k,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(matvec_q3k,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(matvec_q5k,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(matvec_q8_0,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(attention_kernel_f16kv,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
    cudaFuncSetAttribute(attention_kernel_int8kv,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
}

} // extern "C"
