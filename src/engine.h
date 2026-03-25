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

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context
typedef struct MnemoCudaCtx MnemoCudaCtx;

// Token callback
typedef void (*MnemoCudaTokenCB)(const char *text, bool is_done, void *userdata);

// Configuration
typedef struct {
    const char *model_dir;    // Split model directory
    int context_length;       // Max context (default 2048)
    int expert_k;             // Override active experts (0 = use config)
    int gpu_ids[8];           // CUDA device IDs (default {0})
    int n_gpus;               // Number of GPUs to use (default 1, max 8)
    int io_threads;           // pread threads (default 8)
    bool use_pinned_memory;   // Use cudaMallocHost for DMA (default true)
} MnemoCudaConfig;

MnemoCudaConfig mnemo_cuda_config_default(void);

// Lifecycle
MnemoCudaCtx *mnemo_cuda_create(void);
int mnemo_cuda_load(MnemoCudaCtx *ctx, MnemoCudaConfig config);
void mnemo_cuda_unload(MnemoCudaCtx *ctx);
void mnemo_cuda_destroy(MnemoCudaCtx *ctx);

// Generation
int mnemo_cuda_generate(MnemoCudaCtx *ctx, const char *prompt, int max_tokens,
                        float temperature, MnemoCudaTokenCB callback, void *userdata);
void mnemo_cuda_cancel(MnemoCudaCtx *ctx);

// Info
typedef struct {
    int tokens_generated;
    double tokens_per_second;
    double avg_expert_io_ms;
    double avg_gpu_compute_ms;
    size_t vram_used_bytes;
    size_t resident_size_bytes;
    int n_gpus_active;
} MnemoCudaStats;

MnemoCudaStats mnemo_cuda_get_stats(MnemoCudaCtx *ctx);
const char *mnemo_cuda_get_info(MnemoCudaCtx *ctx);

// Expert heat profiling
#define HEAT_TOP_N 20

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
