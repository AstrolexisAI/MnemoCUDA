/**
 * MnemoCUDA Engine — Expert streaming inference for NVIDIA GPUs.
 *
 * Runs MoE models larger than VRAM by:
 * 1. Loading resident weights (non-expert) to GPU VRAM
 * 2. Streaming expert weights from NVMe via pread to pinned memory
 * 3. Async cudaMemcpy from pinned → device per active expert
 * 4. CUDA kernels for dequant matvec on device
 *
 * Uses same split model format as MnemoMetal (iOS).
 */

#ifndef MNEMO_CUDA_ENGINE_H
#define MNEMO_CUDA_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[MnemoCUDA] CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RET(call, retval) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[MnemoCUDA] CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return (retval); \
    } \
} while(0)

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    MNEMO_OK              =  0,
    MNEMO_ERR_BAD_CONFIG  = -1,  // Invalid config or missing model_dir
    MNEMO_ERR_TOKENIZER   = -2,  // Tokenizer load/encode failure
    MNEMO_ERR_NO_GPU      = -3,  // No CUDA GPUs detected or invalid gpu_ids
    MNEMO_ERR_CONTEXT_FULL= -4,  // Position exceeds context length
    MNEMO_ERR_CUDA        = -5,  // CUDA runtime error (OOM, invalid device, etc.)
    MNEMO_ERR_IO          = -6,  // File I/O error (missing model files, pread failure)
    MNEMO_ERR_CANCELLED   = -7,  // Generation cancelled via mnemo_cuda_cancel()
} MnemoError;

const char *mnemo_cuda_strerror(int err);

// Opaque context
typedef struct MnemoCudaCtx MnemoCudaCtx;

// Token callback
typedef void (*MnemoCudaTokenCB)(const char *text, bool is_done, void *userdata);

// Configuration
typedef struct {
    const char *model_dir;    // Split model directory
    int context_length;       // Max context (default 8192)
    int expert_k;             // Override active experts (0 = use config, max 16)
    int gpu_ids[8];           // CUDA device IDs (default {0})
    int n_gpus;               // Number of GPUs to use (0 = auto-detect, max 8)
    int io_threads;           // pread threads (default 8, max 16)
    int cache_percent;        // VRAM % for expert cache (0 = auto, 1-90)
    int extra_prefetch;       // Prefetch depth: 0=normal, 1=also prefetch layer+2
    int kv_int8;              // 1 = INT8 KV cache (half bandwidth vs FP16)
    bool use_pinned_memory;   // Use cudaMallocHost for DMA (default true)
} MnemoCudaConfig;

MnemoCudaConfig mnemo_cuda_config_default(void);

// Lifecycle
MnemoCudaCtx *mnemo_cuda_create(void);
int mnemo_cuda_load(MnemoCudaCtx *ctx, MnemoCudaConfig config);
void mnemo_cuda_unload(MnemoCudaCtx *ctx);
void mnemo_cuda_destroy(MnemoCudaCtx *ctx);

// Generation
// raw_prompt: if true, prompt is passed as-is (no ChatML wrapping).
//             if false, prompt is wrapped in ChatML single-turn format.
int mnemo_cuda_generate(MnemoCudaCtx *ctx, const char *prompt, int max_tokens,
                        float temperature, bool raw_prompt,
                        MnemoCudaTokenCB callback, void *userdata);
void mnemo_cuda_cancel(MnemoCudaCtx *ctx);

// Batch benchmark: compares batch=1 vs batch=2 decode throughput.
// Runs N decode steps for each mode and logs results.
void mnemo_cuda_batch_bench(MnemoCudaCtx *ctx, int n_steps);

// Info
typedef struct {
    int tokens_generated;
    int prompt_tokens;
    double tokens_per_second;
    double ttft_seconds;         // Time to first token (prefill time)
    double total_seconds;        // Total request time
    size_t vram_used_bytes;      // Total VRAM: resident + KV + expert cache + buffers
    size_t resident_size_bytes;  // Host-side resident weights file size
    int n_gpus_active;
} MnemoCudaStats;

MnemoCudaStats mnemo_cuda_get_stats(MnemoCudaCtx *ctx);
const char *mnemo_cuda_get_info(MnemoCudaCtx *ctx);

// Expert heat profiling
#define MNEMO_MAX_EXPERT_K 16  // Max active experts per token (buffer limit in forward/kernels)
#define HEAT_TOP_N 20
#define HEAT_MIN_TOKENS_FOR_PINNING 1000  // Don't pin experts until heat map has enough signal

typedef struct {
    uint64_t total_tokens;       // Tokens profiled
    uint64_t total_activations;  // Sum of all expert activations
    int active_experts;          // Experts activated at least once
    int n_layers;
    int n_experts_per_layer;
    bool pinning_active;

    // Top N hottest experts
    int top_layer[HEAT_TOP_N];
    int top_expert[HEAT_TOP_N];
    uint32_t top_count[HEAT_TOP_N];

    // Per-GPU cache stats
    int cache_slots[8];
    int cache_used[8];
    int cache_pinned[8];

    // RAM cache stats
    int ram_slots;
    int ram_used;
    int ram_hits;
    int ram_misses;
} MnemoCudaHeatStats;

void mnemo_cuda_heat_pin(MnemoCudaCtx *ctx);
void mnemo_cuda_heat_save(MnemoCudaCtx *ctx);
MnemoCudaHeatStats mnemo_cuda_get_heat_stats(MnemoCudaCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif
