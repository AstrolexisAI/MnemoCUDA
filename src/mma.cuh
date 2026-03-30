/**
 * MnemoCUDA MMA Primitives — Ported from llama.cpp mma.cuh
 *
 * Stripped to NVIDIA Turing+ only (sm_75+). No AMD, no Volta, no Blackwell.
 * Provides tile struct, ldmatrix loads, and mma.sync wrappers for INT8.
 */

#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

namespace mnemo_mma {

// ── Tile: matrix fragment held in warp registers ──
// I = rows, J = cols (in 32-bit register units), T = element type
// For NVIDIA Turing+: DATA_LAYOUT_I_MAJOR is the only layout needed.

template<int I_, int J_, typename T>
struct tile {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I_ * J_ / 32;  // elements per thread
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        if constexpr (I == 16 && J == 8) {
            return ((l / 2) * 8) + (threadIdx.x / 4);
        } else if constexpr (I == 16 && J == 16) {
            return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
        } else if constexpr (I == 8 && J == 4) {
            return threadIdx.x / 4;
        } else if constexpr (I == 8 && J == 8) {
            return threadIdx.x / 4;
        } else {
            return -1;
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 16 && J == 8) {
            return ((threadIdx.x % 4) * 2) + (l % 2);
        } else if constexpr (I == 16 && J == 16) {
            return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
        } else if constexpr (I == 8 && J == 4) {
            return threadIdx.x % 4;
        } else if constexpr (I == 8 && J == 8) {
            return (l * 4) + (threadIdx.x % 4);
        } else {
            return -1;
        }
    }
};

// ── Load from shared memory via ldmatrix.sync ──

// 16x8 tile load (4 registers)
template <typename T>
static __device__ __forceinline__ void load_ldmatrix(
        tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
}

// 8x8 tile load (2 registers)
template <typename T>
static __device__ __forceinline__ void load_ldmatrix(
        tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % 8) * stride + ((threadIdx.x / 8) * 4) % 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(xs));
}

// 16x4 tile load (2 registers) — for A operand of m16n8k16
template <typename T>
static __device__ __forceinline__ void load_ldmatrix(
        tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(xi[0]), "=r"(xi[1])
        : "l"(xs));
}

// 8x4 tile load (1 register) — for B operand of m16n8k16
template <typename T>
static __device__ __forceinline__ void load_ldmatrix(
        tile<8, 4, T> & t, const T * __restrict__ xs0, const int stride) {
    int * xi = (int *) t.x;
    const int * xs = (const int *) xs0 + (threadIdx.x % 8) * stride + (threadIdx.x / 8) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];"
        : "=r"(xi[0])
        : "l"(xs));
}

// Generic load fallback (no ldmatrix)
template <int I, int J, typename T>
static __device__ __forceinline__ void load_generic(
        tile<I, J, T> & t, const T * __restrict__ xs0, const int stride) {
    #pragma unroll
    for (int l = 0; l < t.ne; ++l) {
        t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
    }
}

// ── MMA instructions ──

// m16n8k16: C[16x8] += A[16x16] * B[16x8], INT8 → INT32
// A = tile<16, 4, int> (2 registers), B = tile<8, 4, int> (1 register)
// D = tile<16, 8, int> (4 registers)
static __device__ __forceinline__ void mma(
        tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
#if __CUDA_ARCH__ >= 800  // Ampere+
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#elif __CUDA_ARCH__ >= 750  // Turing: use 2x m8n8k16
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1])
        : "r"(A.x[0]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[1]), "r"(B.x[0]));
#endif
}

// m16n8k32: C[16x8] += A[16x32] * B[32x8], INT8 → INT32
// A = tile<16, 8, int> (4 registers), B = tile<8, 8, int> (2 registers)
static __device__ __forceinline__ void mma(
        tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
#if __CUDA_ARCH__ >= 800  // Ampere+
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#elif __CUDA_ARCH__ >= 750  // Turing: use 4x m8n8k16
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1]) : "r"(A.x[0]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3]) : "r"(A.x[1]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1]) : "r"(A.x[2]), "r"(B.x[1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3]) : "r"(A.x[3]), "r"(B.x[1]));
#endif
}

// FP16 MMA: m16n8k16 f32 accumulate
// A = tile<16, 4, half2>, B = tile<8, 4, half2>, D = tile<16, 8, float>
static __device__ __forceinline__ void mma(
        tile<16, 8, float> & D, const tile<16, 4, half2> & A, const tile<8, 4, half2> & B) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(D.x[0]), "+f"(D.x[1]), "+f"(D.x[2]), "+f"(D.x[3])
        : "r"(*((int*)&A.x[0])), "r"(*((int*)&A.x[1])), "r"(*((int*)&A.x[2])), "r"(*((int*)&A.x[3]))
        , "r"(*((int*)&B.x[0])), "r"(*((int*)&B.x[1])));
#elif __CUDA_ARCH__ >= 750
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(D.x[0]), "+f"(D.x[1]), "+f"(D.x[2]), "+f"(D.x[3])
        : "r"(*((int*)&A.x[0])), "r"(*((int*)&A.x[1])), "r"(*((int*)&B.x[0])));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(D.x[0]), "+f"(D.x[1]), "+f"(D.x[2]), "+f"(D.x[3])
        : "r"(*((int*)&A.x[2])), "r"(*((int*)&A.x[3])), "r"(*((int*)&B.x[1])));
#endif
}

} // namespace mnemo_mma
