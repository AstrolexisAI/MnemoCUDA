/**
 * MnemoCUDA Engine Internal — Shared types for engine submodules.
 *
 * This header is NOT part of the public API. It exposes internal types
 * needed by the extracted modules (heat, config, cache, forward).
 */

#ifndef MNEMO_ENGINE_INTERNAL_H
#define MNEMO_ENGINE_INTERNAL_H

#include "engine.h"
#include "tokenizer.h"
#include "io_pool.h"

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <cuda_runtime.h>

// ── Model config ──

typedef struct {
    int hidden_size;
    int intermediate_size;
    int moe_intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int num_hidden_layers;
    int vocab_size;
    int num_experts;
    int num_experts_per_tok;
    float rope_theta;
    float rms_norm_eps;
    int max_position_embeddings;
    int full_attention_interval;  // 0 = all layers have attention, N = every Nth layer (hybrid SSM+MoE)
    // SSM/Mamba config (hybrid models)
    int ssm_inner_size;           // Inner dimension for SSM (e.g., 8192)
    int ssm_state_size;           // State dimension per feature (e.g., 128)
    int ssm_conv_kernel;          // Causal conv kernel size (e.g., 4)
    int ssm_group_count;          // Number of groups for B/C sharing (e.g., 16)
    int ssm_dt_rank;              // Time-step projection rank (e.g., 64)
} ModelConfig;

// ── Expert layer file ──

typedef struct {
    int fd;
    int n_experts;
    size_t expert_size;
    size_t gate_size, up_size, down_size;
    void *mmap_data;
    size_t mmap_size;
} ExpertLayerFile;

// ── Per-GPU state ──

typedef struct {
    int gpu_id;
    cudaStream_t stream_compute;
    cudaStream_t stream_io;

    void *d_resident;
    size_t resident_size;
    size_t resident_offset;

    // Per-tensor offset remapping for partitioned resident weights.
    // tensor_offsets[tensor_index] = offset into this GPU's d_resident,
    // or (size_t)-1 if the tensor is not on this GPU.
    size_t *tensor_offsets;
    int n_tensor_offsets;

    int layer_start;
    int layer_end;

    float *d_hidden;
    float *d_residual;
    float *d_q, *d_k, *d_v;
    float *d_attn_out;
    float *d_normed;
    float *d_router_logits;
    float *d_expert_gate, *d_expert_up, *d_expert_act, *d_expert_out;
    float *d_moe_out;
    float *d_logits;

    void *h_expert_buf;
    void *d_expert_buf;
    size_t expert_buf_size;

    void *d_kv_k, *d_kv_v;

    float *d_router_out;
    int   *d_expert_indices;
    float *d_expert_weights;

    void *d_expert_cache;
    size_t expert_cache_size;
    int expert_cache_slots;
    size_t expert_slot_size;

    int *cache_layer;
    int *cache_expert;
    uint64_t *cache_lru;
    uint32_t *cache_hits;
    uint64_t cache_clock;
    bool *cache_pinned;

    bool expert_buf_pinned;
    bool prefetch_buf_pinned;

    void *h_prefetch_buf;
    size_t prefetch_buf_size;
    int prefetch_layer;
    int prefetch_eids[16];
    int n_prefetched;
    volatile int prefetch_ready;

    // SSM/Mamba state (hybrid models only, NULL if pure attention)
    float *d_ssm_state;      // [n_ssm_layers, inner_size, state_size] persistent
    float *d_conv_state;     // [n_ssm_layers, inner_size, conv_kernel-1] persistent
    float *d_ssm_x;          // [inner_size] temp: input projection x
    float *d_ssm_z;          // [inner_size] temp: gate z
    float *d_ssm_y;          // [inner_size] temp: SSM output
    int n_ssm_layers;        // number of SSM layers on this GPU
    int *ssm_layer_map;      // [n_ssm_layers] maps SSM index -> global layer index
} GPUState;

// ── Tensor hash table ──

#define TENSOR_HASH_BUCKETS 2048

typedef struct TensorEntry {
    char name[128];
    void *data;
    size_t offset;
    size_t size;
    int type_id;
    int dims[4];
    int n_dims;
    struct TensorEntry *next;
} TensorEntry;

typedef struct {
    TensorEntry *buckets[TENSOR_HASH_BUCKETS];
    TensorEntry *entries;
    int n_entries;
} TensorTable;

// ── RAM cache ──

#define RAM_CACHE_DEFAULT_GB 0

typedef struct {
    void *data;
    size_t slot_size;
    int n_slots;
    int *slot_layer;
    int *slot_expert;
    uint64_t *slot_lru;
    uint64_t clock;
    int hits;
    int misses;
} RAMCache;

// ── Main context (opaque in public API, exposed here for submodules) ──

struct MnemoCudaCtx {
    ModelConfig config;
    char model_dir[1024];

    GPUState gpus[8];
    int n_gpus;

    void *h_resident_mmap;
    size_t resident_size;

    TensorTable tensor_table;

    ExpertLayerFile *expert_layers;
    int n_expert_layers;

    Tokenizer *tokenizer;

    float *h_hidden_transfer;

    int kv_pos;

    RAMCache ram_cache;

    uint32_t *heat_map;
    uint64_t heat_total_tokens;
    bool heat_pinning_active;

    uint32_t *layer_hits;
    uint32_t *layer_misses;

    int last_activated[16];
    int n_last_activated;

    volatile bool cancelled;
    bool loaded;
    MnemoCudaStats stats;
    char info[256];
};

// ── Internal functions shared between modules ──

// Tensor table operations
static inline uint32_t fnv1a(const char *s) {
    uint32_t h = 0x811c9dc5;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 0x01000193;
    }
    return h;
}

static inline void tensor_table_insert(TensorTable *t, TensorEntry *e) {
    uint32_t idx = fnv1a(e->name) % TENSOR_HASH_BUCKETS;
    e->next = t->buckets[idx];
    t->buckets[idx] = e;
}

static inline TensorEntry *tensor_find(TensorTable *t, const char *name) {
    uint32_t idx = fnv1a(name) % TENSOR_HASH_BUCKETS;
    for (TensorEntry *e = t->buckets[idx]; e; e = e->next)
        if (strcmp(e->name, name) == 0) return e;
    return NULL;
}

// RAM cache operations
int ram_cache_init(RAMCache *rc, size_t slot_size, int target_gb);
void *ram_cache_lookup(RAMCache *rc, int layer, int expert_id);
void *ram_cache_insert(RAMCache *rc, int layer, int expert_id,
                       const void *src, size_t size);
void ram_cache_free(RAMCache *rc);

// Expert VRAM cache operations
void *expert_cache_lookup(GPUState *gpu, int layer, int expert_id);
bool expert_cache_has(GPUState *gpu, int layer, int expert_id);
void *expert_cache_insert(GPUState *gpu, int layer, int expert_id,
                          const void *host_data, size_t data_size,
                          cudaStream_t stream);
void *expert_cache_insert_ctx(struct MnemoCudaCtx *ctx, GPUState *gpu,
                              int layer, int expert_id,
                              const void *host_data, size_t data_size,
                              cudaStream_t stream);

// JSON helpers
int json_get_int(const char *json, const char *key, int default_val);
float json_get_float(const char *json, const char *key, float default_val);

#endif
