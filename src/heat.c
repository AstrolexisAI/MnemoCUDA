/**
 * MnemoCUDA Heat Profiling — Expert activation tracking and hot-expert pinning.
 *
 * Tracks which experts are activated most frequently, pins hot experts in
 * VRAM cache to prevent eviction, and persists heat maps to disk.
 */

#include "engine_internal.h"
#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

typedef struct { int layer; int expert; uint32_t count; } HeatEntry;

static int heat_entry_cmp_desc(const void *a, const void *b) {
    uint32_t ca = ((const HeatEntry *)a)->count;
    uint32_t cb = ((const HeatEntry *)b)->count;
    return (cb > ca) - (cb < ca); // descending
}

// Pin the hottest experts in VRAM cache so they never get evicted.
// Fills up to 50% of each GPU's cache slots with pinned hot experts.
void mnemo_cuda_heat_pin(MnemoCudaCtx *ctx) {
    if (!ctx || !ctx->loaded || !ctx->heat_map) return;

    ModelConfig *cfg = &ctx->config;
    int NL = cfg->num_hidden_layers;
    int NE = cfg->num_experts;
    int total = NL * NE;

    // Build sorted index of (layer, expert) by activation count (descending)
    HeatEntry *sorted = malloc(total * sizeof(HeatEntry));
    if (!sorted) return;
    for (int i = 0; i < total; i++) {
        sorted[i].layer = i / NE;
        sorted[i].expert = i % NE;
        sorted[i].count = ctx->heat_map[i];
    }

    // qsort descending by count (handles 48K+ entries for large MoE models)
    qsort(sorted, total, sizeof(HeatEntry), heat_entry_cmp_desc);

    // For each GPU, pin the hottest experts that belong to its layer range
    int total_pinned = 0;
    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        if (!gpu->d_expert_cache || gpu->expert_cache_slots == 0) continue;

        // Pin up to 50% of slots (leave room for dynamic LRU)
        int max_pin = gpu->expert_cache_slots / 2;
        if (max_pin < 1) max_pin = 1;
        int pinned = 0;

        // Clear previous pins
        if (gpu->cache_pinned)
            memset(gpu->cache_pinned, 0, gpu->expert_cache_slots * sizeof(bool));

        // Walk sorted list, pin experts that belong to this GPU's layers
        for (int i = 0; i < total && pinned < max_pin; i++) {
            if (sorted[i].count == 0) break;
            int layer = sorted[i].layer;
            int eid = sorted[i].expert;

            // Check if this layer belongs to this GPU
            if (layer < gpu->layer_start || layer >= gpu->layer_end) continue;

            // Check if already in cache
            void *cached = expert_cache_lookup(gpu, layer, eid);
            if (cached) {
                // Find its slot and pin it
                for (int s = 0; s < gpu->expert_cache_slots; s++) {
                    if (gpu->cache_layer[s] == layer && gpu->cache_expert[s] == eid) {
                        gpu->cache_pinned[s] = true;
                        pinned++;
                        break;
                    }
                }
            } else {
                // Load from disk and insert as pinned
                ExpertLayerFile *elf = &ctx->expert_layers[layer];
                if (elf->fd <= 0 && !elf->mmap_data) continue;

                size_t expert_size = elf->expert_size;
                void *h_buf = malloc(expert_size);
                if (!h_buf) continue;

                if (elf->mmap_data) {
                    memcpy(h_buf, (char *)elf->mmap_data + (size_t)eid * expert_size, expert_size);
                } else {
                    pread(elf->fd, h_buf, expert_size, (off_t)eid * (off_t)expert_size);
                }

                void *d = expert_cache_insert(gpu, layer, eid, h_buf, expert_size,
                                               gpu->stream_compute);
                cudaStreamSynchronize(gpu->stream_compute);
                free(h_buf);

                if (d) {
                    // Find and pin the slot
                    for (int s = 0; s < gpu->expert_cache_slots; s++) {
                        if (gpu->cache_layer[s] == layer && gpu->cache_expert[s] == eid) {
                            gpu->cache_pinned[s] = true;
                            pinned++;
                            break;
                        }
                    }
                }
            }
        }

        total_pinned += pinned;
        LOG_INFO("GPU %d: pinned %d/%d hot experts (%.0f%% of cache)",
                gpu->gpu_id, pinned, gpu->expert_cache_slots,
                100.0 * pinned / gpu->expert_cache_slots);
    }

    ctx->heat_pinning_active = true;
    LOG_INFO("Heat pinning active: %d experts pinned across %d GPUs "
            "(based on %lu tokens of profiling)",
            total_pinned, ctx->n_gpus, (unsigned long)ctx->heat_total_tokens);

    free(sorted);
}

// Save heat map to disk for persistence across restarts
void mnemo_cuda_heat_save(MnemoCudaCtx *ctx) {
    if (!ctx || !ctx->loaded || !ctx->heat_map) return;

    ModelConfig *cfg = &ctx->config;
    char heat_path[1200];
    snprintf(heat_path, sizeof(heat_path), "%s/expert_heat.bin", ctx->model_dir);

    FILE *f = fopen(heat_path, "wb");
    if (!f) {
        LOG_ERROR("Failed to save heat map: %s", heat_path);
        return;
    }

    uint32_t magic = 0x48454154; // "HEAT"
    uint32_t n_layers = (uint32_t)cfg->num_hidden_layers;
    uint32_t n_experts = (uint32_t)cfg->num_experts;
    fwrite(&magic, 4, 1, f);
    fwrite(&n_layers, 4, 1, f);
    fwrite(&n_experts, 4, 1, f);
    fwrite(&ctx->heat_total_tokens, 8, 1, f);
    fwrite(ctx->heat_map, sizeof(uint32_t), n_layers * n_experts, f);
    fclose(f);

    LOG_INFO("Heat map saved: %s (%lu tokens)",
            heat_path, (unsigned long)ctx->heat_total_tokens);
}

// Get heat statistics: top N hottest experts and overall stats
MnemoCudaHeatStats mnemo_cuda_get_heat_stats(MnemoCudaCtx *ctx) {
    MnemoCudaHeatStats hs = {0};
    if (!ctx || !ctx->loaded || !ctx->heat_map) return hs;

    ModelConfig *cfg = &ctx->config;
    int NL = cfg->num_hidden_layers;
    int NE = cfg->num_experts;
    int total = NL * NE;

    hs.total_tokens = ctx->heat_total_tokens;
    hs.pinning_active = ctx->heat_pinning_active;
    hs.n_layers = NL;
    hs.n_experts_per_layer = NE;

    // Count active experts (activated at least once)
    uint64_t total_activations = 0;
    uint32_t max_count = 0;
    int active = 0;
    for (int i = 0; i < total; i++) {
        if (ctx->heat_map[i] > 0) active++;
        total_activations += ctx->heat_map[i];
        if (ctx->heat_map[i] > max_count) max_count = ctx->heat_map[i];
    }
    hs.active_experts = active;
    hs.total_activations = total_activations;

    // Find top HEAT_TOP_N experts
    for (int t = 0; t < HEAT_TOP_N; t++) {
        uint32_t best = 0;
        int best_idx = -1;
        for (int i = 0; i < total; i++) {
            if (ctx->heat_map[i] > best) {
                // Check not already in top list
                bool dup = false;
                for (int p = 0; p < t; p++) {
                    if (hs.top_layer[p] == i / NE && hs.top_expert[p] == i % NE) {
                        dup = true; break;
                    }
                }
                if (!dup) {
                    best = ctx->heat_map[i];
                    best_idx = i;
                }
            }
        }
        if (best_idx >= 0) {
            hs.top_layer[t] = best_idx / NE;
            hs.top_expert[t] = best_idx % NE;
            hs.top_count[t] = best;
        }
    }

    // Cache stats per GPU
    for (int g = 0; g < ctx->n_gpus && g < 8; g++) {
        GPUState *gpu = &ctx->gpus[g];
        hs.cache_slots[g] = gpu->expert_cache_slots;
        int pinned = 0, used = 0;
        for (int s = 0; s < gpu->expert_cache_slots; s++) {
            if (gpu->cache_layer[s] >= 0) used++;
            if (gpu->cache_pinned && gpu->cache_pinned[s]) pinned++;
        }
        hs.cache_used[g] = used;
        hs.cache_pinned[g] = pinned;
    }

    // RAM cache stats
    RAMCache *rc = &ctx->ram_cache;
    hs.ram_slots = rc->n_slots;
    hs.ram_hits = rc->hits;
    hs.ram_misses = rc->misses;
    int ram_used = 0;
    for (int s = 0; s < rc->n_slots; s++)
        if (rc->slot_layer[s] >= 0) ram_used++;
    hs.ram_used = ram_used;

    return hs;
}
