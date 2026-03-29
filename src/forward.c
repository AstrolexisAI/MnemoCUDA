/**
 * MnemoCUDA Forward Pass — Layer computation, expert cache, and prefetch.
 */

#include "forward.h"
#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <cuda_runtime.h>

// Kernel declarations (extern "C" from kernels.cu)
extern void cuda_matvec_q3k(const void *weights, const float *x, float *y,
                            int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_matvec_q5k(const void *weights, const float *x, float *y,
                            int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_matvec_q8_0(const void *weights, const float *x, float *y,
                             int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_matvec_f32(const float *weights, const float *x, float *y,
                            int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_matvec_q4k(const void *weights, const float *x, float *y,
                            int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_matvec_q6k(const void *weights, const float *x, float *y,
                            int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_swiglu(const float *gate, const float *up, float *out,
                        int n, cudaStream_t stream);
extern void cuda_rms_norm(const float *input, const float *weight, float *output,
                          int n, float eps, cudaStream_t stream);
extern void cuda_residual_add(const float *a, const float *b, float *out,
                              int n, cudaStream_t stream);
extern void cuda_rope(float *q, float *k, int head_dim, int pos, float theta,
                      int n_heads_q, int n_heads_k, cudaStream_t stream);
extern void cuda_scaled_add(float *out, const float *x, float scale, int n,
                            cudaStream_t stream);
extern void cuda_topk_softmax(const float *scores, int *indices, float *weights,
                              int n_experts, int k, cudaStream_t stream);
extern void cuda_f32_to_f16(const float *in, void *out, int n, cudaStream_t stream);
extern void cuda_attention_f16kv(const void *q, const void *kv_k, const void *kv_v,
                                 float *out, int n_heads_q, int head_dim, int n_kv_heads,
                                 int seq_len, int gqa_ratio, float scale, cudaStream_t stream);
// Gated Delta Net kernels
extern void cuda_gdn_conv1d(float *x, float *conv_state, const float *w_conv,
                            int dim, int conv_kernel, cudaStream_t stream);
extern void cuda_gdn_recurrence(const float *q, const float *k, const float *v,
                                const float *A, const float *alpha, const float *beta,
                                const float *dt_bias, float *state, float *output,
                                int num_v_heads, int num_k_heads,
                                int key_head_dim, int value_head_dim,
                                cudaStream_t stream);
extern void cuda_rms_norm_gated(const float *x, const float *z, const float *w,
                                float *out, int head_dim, int n_heads,
                                cudaStream_t stream);

// ── Helper: get tensor data on specific GPU ──

static void *tensor_data_on_gpu(MnemoCudaCtx *ctx, const char *name, int gpu_idx) {
    TensorEntry *e = tensor_find(&ctx->tensor_table, name);
    if (!e) return NULL;

    GPUState *gpu = &ctx->gpus[gpu_idx];
    // Use per-GPU tensor offset if partitioned, else fall back to global offset
    if (gpu->tensor_offsets) {
        int idx = (int)(e - ctx->tensor_table.entries);
        if (idx >= 0 && idx < gpu->n_tensor_offsets &&
            gpu->tensor_offsets[idx] != (size_t)-1)
            return (char *)gpu->d_resident + gpu->tensor_offsets[idx];
        return NULL;  // tensor not on this GPU
    }
    return (char *)gpu->d_resident + e->offset;
}

static TensorEntry *tensor_get(MnemoCudaCtx *ctx, const char *name) {
    return tensor_find(&ctx->tensor_table, name);
}

// Resolve a TensorEntry to its device pointer on a specific GPU
static void *tensor_ptr_on_gpu(MnemoCudaCtx *ctx, TensorEntry *e, int gpu_idx) {
    if (!e) return NULL;
    GPUState *gpu = &ctx->gpus[gpu_idx];
    if (gpu->tensor_offsets) {
        int idx = (int)(e - ctx->tensor_table.entries);
        if (idx >= 0 && idx < gpu->n_tensor_offsets &&
            gpu->tensor_offsets[idx] != (size_t)-1)
            return (char *)gpu->d_resident + gpu->tensor_offsets[idx];
        return NULL;
    }
    return (char *)gpu->d_resident + e->offset;
}

// ── Matvec dispatch by quantization type ──

static void matvec(const void *w, const float *x, float *y,
                   int rows, int cols, int type_id, cudaStream_t stream) {
    switch (type_id) {
        case 0:  cuda_matvec_f32((const float *)w, x, y, rows, cols, stream); break;
        case 8:  cuda_matvec_q8_0(w, x, y, rows, cols, stream); break;
        case 11: cuda_matvec_q3k(w, x, y, rows, cols, stream); break;
        case 12: cuda_matvec_q4k(w, x, y, rows, cols, stream); break;
        case 13: cuda_matvec_q5k(w, x, y, rows, cols, stream); break;
        case 14: cuda_matvec_q6k(w, x, y, rows, cols, stream); break;
        default:
            LOG_WARN("Unsupported type_id %d for matvec", type_id);
            break;
    }
}

// ── Forward one layer on a specific GPU ──

// (cache forward declarations in engine_internal.h)

// ── Prefetch: start reading hot experts for a future layer ──

static void prefetch_submit(MnemoCudaCtx *ctx, GPUState *gpu, int layer) {
    if (!ctx->heat_map || !gpu->h_prefetch_buf) return;

    ModelConfig *cfg = &ctx->config;
    int NE = cfg->num_experts;
    int K = cfg->num_experts_per_tok;
    if (layer < 0 || layer >= cfg->num_hidden_layers) return;

    ExpertLayerFile *elf = &ctx->expert_layers[layer];
    if (elf->fd <= 0 && !elf->mmap_data) return;

    // Adaptive prefetch count: fetch MORE for layers with low hit rate
    int max_prefetch = K; // default: K experts
    if (ctx->layer_hits && ctx->layer_misses) {
        uint32_t lh = ctx->layer_hits[layer];
        uint32_t lm = ctx->layer_misses[layer];
        if (lh + lm > 20) { // enough data to judge
            float hit_rate = (float)lh / (float)(lh + lm);
            if (hit_rate < 0.80f) max_prefetch = K * 2; // aggressive for bad layers
            if (hit_rate < 0.70f) max_prefetch = K * 3; // very aggressive
        }
    }
    if (max_prefetch > 16) max_prefetch = 16; // cap at buffer size
    int pool_sz = io_pool_size();
    if (pool_sz > 0 && max_prefetch > pool_sz) max_prefetch = pool_sz;

    // Score each non-cached expert: heat + cross-layer correlation
    uint32_t *layer_heat = &ctx->heat_map[layer * NE];
    int n_to_prefetch = 0;

    // Build candidates with composite scoring
    typedef struct { int eid; float score; } PrefetchCandidate;
    PrefetchCandidate candidates[16];
    int n_candidates = 0;

    for (int e = 0; e < NE && n_candidates < 16; e++) {
        if (expert_cache_has(gpu, layer, e)) continue;
        if (layer_heat[e] == 0) continue;

        float score = (float)layer_heat[e]; // base: heat

        // Cross-layer boost: if this expert_id was active in the previous layer,
        // boost its score (experts with same ID tend to co-activate across layers)
        for (int a = 0; a < ctx->n_last_activated; a++) {
            if (ctx->last_activated[a] == e) {
                score *= 2.0f; // 2x boost for cross-layer correlation
                break;
            }
        }

        candidates[n_candidates].eid = e;
        candidates[n_candidates].score = score;
        n_candidates++;
    }

    // Sort candidates by score descending (simple insertion sort, n_candidates <= 16)
    for (int i = 1; i < n_candidates; i++) {
        PrefetchCandidate tmp = candidates[i];
        int j = i - 1;
        while (j >= 0 && candidates[j].score < tmp.score) {
            candidates[j + 1] = candidates[j]; j--;
        }
        candidates[j + 1] = tmp;
    }

    // Pick top max_prefetch
    for (int i = 0; i < n_candidates && n_to_prefetch < max_prefetch; i++)
        gpu->prefetch_eids[n_to_prefetch++] = candidates[i].eid;

    if (n_to_prefetch == 0) {
        gpu->prefetch_layer = -1;
        gpu->n_prefetched = 0;
        gpu->prefetch_ready = 1;
        return;
    }

    // Submit I/O tasks to read these experts into prefetch buffer
    IOTask io_tasks[16];
    for (int i = 0; i < n_to_prefetch; i++) {
        io_tasks[i].dst = (char *)gpu->h_prefetch_buf + (size_t)i * elf->expert_size;
        io_tasks[i].size = elf->expert_size;
        io_tasks[i].done = 0;
        if (elf->mmap_data) {
            io_tasks[i].fd = -1;
            io_tasks[i].src = (char *)elf->mmap_data + (size_t)gpu->prefetch_eids[i] * elf->expert_size;
        } else {
            io_tasks[i].fd = elf->fd;
            io_tasks[i].src = NULL;
            io_tasks[i].offset = (off_t)gpu->prefetch_eids[i] * (off_t)elf->expert_size;
        }
    }

    gpu->prefetch_layer = layer;
    gpu->n_prefetched = n_to_prefetch;
    gpu->prefetch_ready = 0;
    io_pool_submit(io_tasks, n_to_prefetch);
}

static void prefetch_wait(GPUState *gpu, int n) {
    if (n > 0) io_pool_wait(n);
    gpu->prefetch_ready = 1;
}

// Check if an expert is in the prefetch buffer; returns host pointer or NULL.
// Returns NULL if the I/O for that slot had an error.
static void *prefetch_lookup(GPUState *gpu, int layer, int expert_id, size_t expert_size) {
    if (gpu->prefetch_layer != layer || !gpu->prefetch_ready) return NULL;
    for (int i = 0; i < gpu->n_prefetched; i++) {
        if (gpu->prefetch_eids[i] == expert_id) {
            if (io_pool_task_error(i) != 0) return NULL;  // I/O failed
            return (char *)gpu->h_prefetch_buf + (size_t)i * expert_size;
        }
    }
    return NULL;
}

// ── Expert VRAM cache: lookup / insert ──

// Returns device pointer to expert data in cache, or NULL if miss
void *expert_cache_lookup(GPUState *gpu, int layer, int expert_id) {
    if (!gpu->d_expert_cache || gpu->expert_cache_slots == 0) return NULL;
    for (int s = 0; s < gpu->expert_cache_slots; s++) {
        if (gpu->cache_layer[s] == layer && gpu->cache_expert[s] == expert_id) {
            gpu->cache_lru[s] = ++gpu->cache_clock;
            if (gpu->cache_hits) gpu->cache_hits[s]++;
            return (char *)gpu->d_expert_cache + (size_t)s * gpu->expert_slot_size;
        }
    }
    return NULL;
}

// Check-only: returns true if expert is cached, without bumping LRU
bool expert_cache_has(GPUState *gpu, int layer, int expert_id) {
    if (!gpu->d_expert_cache || gpu->expert_cache_slots == 0) return false;
    for (int s = 0; s < gpu->expert_cache_slots; s++)
        if (gpu->cache_layer[s] == layer && gpu->cache_expert[s] == expert_id) return true;
    return false;
}

// Insert expert into cache (evict LRU if full), returns device pointer.
// If ram_demote is non-NULL, evicted experts are demoted to RAM cache.
void *expert_cache_insert_ctx(struct MnemoCudaCtx *ctx, GPUState *gpu,
                                      int layer, int expert_id,
                                      const void *host_data, size_t data_size,
                                      cudaStream_t stream) {
    if (!gpu->d_expert_cache || gpu->expert_cache_slots == 0) return NULL;

    // Find empty slot or LRU slot (skip pinned slots)
    // NOTE: Tested composite scoring (heat + frequency + recency) — it performed WORSE
    // than pure LRU for MoE expert streaming (78% vs 88% hit rate). LRU is optimal here
    // because expert access patterns are highly temporal: what was used last token is
    // most likely needed next token. Heat/frequency protect stale experts.
    int target = -1;
    uint64_t min_lru = UINT64_MAX;
    for (int s = 0; s < gpu->expert_cache_slots; s++) {
        if (gpu->cache_layer[s] == -1) { target = s; break; } // empty
        if (gpu->cache_pinned && gpu->cache_pinned[s]) continue; // pinned — don't evict
        if (gpu->cache_lru[s] < min_lru) { min_lru = gpu->cache_lru[s]; target = s; }
    }
    if (target < 0) return NULL; // all slots pinned, can't evict

    // ── VRAM→RAM demotion: save evicted expert to RAM cache ──
    if (ctx && gpu->cache_layer[target] >= 0) {
        // Read evicted expert data back from VRAM to a temp pinned buffer, then insert into RAM cache
        // Optimization: use the host_data buffer as temp (it's about to be overwritten anyway)
        // Instead, just insert from the incoming host_data's neighbor — but that's the NEW data.
        // Simplest: demote using the incoming host_data as the source is wrong.
        // Real demotion: copy VRAM slot → RAM. But cudaMemcpy D2H is expensive (~1ms for 12MB).
        // Better approach: when the expert was loaded, it passed through host memory.
        // If it's in the mmap page cache, re-reading is free. Just record it in RAM cache
        // from the mmap data (which is already in page cache if recently used).
        int evict_layer = gpu->cache_layer[target];
        int evict_expert = gpu->cache_expert[target];
        if (ctx->expert_layers && evict_layer < ctx->config.num_hidden_layers) {
            ExpertLayerFile *elf = &ctx->expert_layers[evict_layer];
            if (elf->mmap_data) {
                void *src = (char *)elf->mmap_data + (size_t)evict_expert * elf->expert_size;
                ram_cache_insert(&ctx->ram_cache, evict_layer, evict_expert,
                                 src, elf->expert_size);
            }
        }
    }

    gpu->cache_layer[target] = layer;
    gpu->cache_expert[target] = expert_id;
    gpu->cache_lru[target] = ++gpu->cache_clock;
    if (gpu->cache_hits) gpu->cache_hits[target] = 0; // reset frequency for new entry

    void *slot = (char *)gpu->d_expert_cache + (size_t)target * gpu->expert_slot_size;
    cudaMemcpyAsync(slot, host_data, data_size, cudaMemcpyHostToDevice, stream);
    return slot;
}

// Simple wrapper without context (for heat_pin and prefetch where we don't need demotion)
void *expert_cache_insert(GPUState *gpu, int layer, int expert_id,
                                  const void *host_data, size_t data_size,
                                  cudaStream_t stream) {
    return expert_cache_insert_ctx(NULL, gpu, layer, expert_id, host_data, data_size, stream);
}

// ── SSM/Mamba forward for hybrid layers ──

// Gated Delta Net forward for one layer (replaces attention in hybrid models)
static void forward_gdn_block(MnemoCudaCtx *ctx, int layer, int gpu_idx) {
    GPUState *gpu = &ctx->gpus[gpu_idx];
    ModelConfig *cfg = &ctx->config;
    int H = cfg->hidden_size;
    int V_DIM = cfg->ssm_inner_size;        // value_dim (8192)
    int K_HD = cfg->ssm_state_size;          // key_head_dim (128)
    int V_HD = K_HD;                         // value_head_dim = key_head_dim
    int N_VH = cfg->ssm_dt_rank;             // num_value_heads (64)
    int N_KH = cfg->ssm_group_count;         // num_key_heads (16)
    int K_DIM = K_HD * N_KH;                 // key_dim (2048)
    int CONV_DIM = K_DIM + K_DIM + V_DIM;    // conv1d dim (12288)
    int CK = cfg->ssm_conv_kernel;
    float eps = cfg->rms_norm_eps;
    cudaStream_t cs = gpu->stream_compute;
    char tname[128];

    if (!gpu->d_gdn_state || !gpu->d_gdn_qkv) return;

    // Find GDN layer index
    int gdn_idx = -1;
    for (int i = 0; i < gpu->n_gdn_layers; i++) {
        if (gpu->gdn_layer_map[i] == layer) { gdn_idx = i; break; }
    }
    if (gdn_idx < 0) return;

    float *layer_state = gpu->d_gdn_state + (size_t)gdn_idx * N_VH * K_HD * V_HD;
    float *layer_conv = gpu->d_conv_state + (size_t)gdn_idx * CONV_DIM * (CK - 1);

    // ── 1. Pre-layer RMS norm ──
    snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", layer);
    void *norm_w = tensor_data_on_gpu(ctx, tname, gpu_idx);
    if (norm_w)
        cuda_rms_norm(gpu->d_hidden, (float *)norm_w, gpu->d_normed, H, eps, cs);

    // Save residual
    cudaMemcpyAsync(gpu->d_residual, gpu->d_hidden, H * sizeof(float),
                    cudaMemcpyDeviceToDevice, cs);

    // ── 2. QKV projection: [H] → [K_DIM + K_DIM + V_DIM] = [12288] ──
    snprintf(tname, sizeof(tname), "blk.%d.attn_qkv.weight", layer);
    TensorEntry *w_qkv = tensor_get(ctx, tname);
    if (!w_qkv) return;
    matvec(tensor_ptr_on_gpu(ctx, w_qkv, gpu_idx),
           gpu->d_normed, gpu->d_gdn_qkv, CONV_DIM, H, w_qkv->type_id, cs);

    // ── 3. Alpha/Beta projections: [H] → [N_VH] each ──
    snprintf(tname, sizeof(tname), "blk.%d.ssm_alpha.weight", layer);
    TensorEntry *w_alpha = tensor_get(ctx, tname);
    if (w_alpha)
        matvec(tensor_ptr_on_gpu(ctx, w_alpha, gpu_idx),
               gpu->d_normed, gpu->d_gdn_alpha, N_VH, H, w_alpha->type_id, cs);

    snprintf(tname, sizeof(tname), "blk.%d.ssm_beta.weight", layer);
    TensorEntry *w_beta = tensor_get(ctx, tname);
    if (w_beta)
        matvec(tensor_ptr_on_gpu(ctx, w_beta, gpu_idx),
               gpu->d_normed, gpu->d_gdn_beta, N_VH, H, w_beta->type_id, cs);

    // ── 4. Gate Z projection: [H] → [V_DIM] ──
    snprintf(tname, sizeof(tname), "blk.%d.attn_gate.weight", layer);
    TensorEntry *w_gate = tensor_get(ctx, tname);
    if (w_gate)
        matvec(tensor_ptr_on_gpu(ctx, w_gate, gpu_idx),
               gpu->d_normed, gpu->d_gdn_z, V_DIM, H, w_gate->type_id, cs);

    // ── 5. Causal conv1d over concatenated QKV ──
    snprintf(tname, sizeof(tname), "blk.%d.ssm_conv1d.weight", layer);
    TensorEntry *w_conv = tensor_get(ctx, tname);
    if (w_conv) {
        cuda_gdn_conv1d(gpu->d_gdn_qkv, layer_conv,
                        (float *)tensor_ptr_on_gpu(ctx, w_conv, gpu_idx),
                        CONV_DIM, CK, cs);
    }

    // ── 6. Split QKV after conv ──
    // Layout: [Q=K_DIM | K=K_DIM | V=V_DIM]
    float *q_ptr = gpu->d_gdn_qkv;                  // [K_DIM]
    float *k_ptr = gpu->d_gdn_qkv + K_DIM;          // [K_DIM]
    float *v_ptr = gpu->d_gdn_qkv + K_DIM + K_DIM;  // [V_DIM]

    // ── 7. A and dt_bias ──
    snprintf(tname, sizeof(tname), "blk.%d.ssm_a", layer);
    TensorEntry *w_a = tensor_get(ctx, tname);
    snprintf(tname, sizeof(tname), "blk.%d.ssm_dt.bias", layer);
    TensorEntry *w_dt = tensor_get(ctx, tname);

    float *a_ptr = w_a ? (float *)tensor_ptr_on_gpu(ctx, w_a, gpu_idx) : NULL;
    float *dt_ptr = w_dt ? (float *)tensor_ptr_on_gpu(ctx, w_dt, gpu_idx) : NULL;

    // ── 8. Gated Delta Net recurrence ──
    if (a_ptr) {
        cuda_gdn_recurrence(q_ptr, k_ptr, v_ptr,
                            a_ptr, gpu->d_gdn_alpha, gpu->d_gdn_beta, dt_ptr,
                            layer_state, gpu->d_gdn_out,
                            N_VH, N_KH, K_HD, V_HD, cs);
    }

    // ── 9. RMSNormGated: output = rms_norm(output, ssm_norm) * silu(z) ──
    snprintf(tname, sizeof(tname), "blk.%d.ssm_norm.weight", layer);
    TensorEntry *w_norm = tensor_get(ctx, tname);
    if (w_norm) {
        cuda_rms_norm_gated(gpu->d_gdn_out, gpu->d_gdn_z,
                            (float *)tensor_ptr_on_gpu(ctx, w_norm, gpu_idx),
                            gpu->d_gdn_out, V_HD, N_VH, cs);
    }

    // ── 10. Output projection: [V_DIM] → [H] ──
    snprintf(tname, sizeof(tname), "blk.%d.ssm_out.weight", layer);
    TensorEntry *w_out = tensor_get(ctx, tname);
    if (w_out) {
        matvec(tensor_ptr_on_gpu(ctx, w_out, gpu_idx),
               gpu->d_gdn_out, gpu->d_hidden, H, V_DIM, w_out->type_id, cs);
    }

    // ── 11. Residual connection ──
    cuda_residual_add(gpu->d_hidden, gpu->d_residual, gpu->d_hidden, H, cs);
}

void forward_layer(MnemoCudaCtx *ctx, int layer, int gpu_idx, int pos) {
    GPUState *gpu = &ctx->gpus[gpu_idx];
    cudaSetDevice(gpu->gpu_id);
    ModelConfig *cfg = &ctx->config;
    int H = cfg->hidden_size;
    int NH = cfg->num_attention_heads;
    int NKV = cfg->num_key_value_heads;
    int HD = cfg->head_dim;
    int NE = cfg->num_experts;
    int K = cfg->num_experts_per_tok;
    int EFF = cfg->moe_intermediate_size;
    float eps = cfg->rms_norm_eps;
    cudaStream_t cs = gpu->stream_compute;

    char tname[128];

    // Detect if this is an attention layer or SSM layer (hybrid MoE+SSM models)
    // full_attention_interval=4 means layers 0,4,8,... have attention, rest are SSM
    int is_attention_layer = 1;
    if (cfg->full_attention_interval > 0) {
        is_attention_layer = (layer % cfg->full_attention_interval == 0);
    }

    if (is_attention_layer) {
        // ── 1. Pre-attention RMS norm ──
        snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", layer);
        void *norm_w = tensor_data_on_gpu(ctx, tname, gpu_idx);
        if (norm_w) {
            cuda_rms_norm(gpu->d_hidden, (float *)norm_w, gpu->d_normed, H, eps, cs);
        }

        // Save residual
        cudaMemcpyAsync(gpu->d_residual, gpu->d_hidden, H * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs);

        // ── 2. Q/K/V projections (GPU dequant matvec) ──
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", layer);
        TensorEntry *wq = tensor_get(ctx, tname);
        if (wq) matvec(tensor_ptr_on_gpu(ctx, wq, gpu_idx),
                        gpu->d_normed, gpu->d_q, NH * HD, H, wq->type_id, cs);

        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", layer);
        TensorEntry *wk = tensor_get(ctx, tname);
        if (wk) matvec(tensor_ptr_on_gpu(ctx, wk, gpu_idx),
                        gpu->d_normed, gpu->d_k, NKV * HD, H, wk->type_id, cs);

        snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", layer);
        TensorEntry *wv = tensor_get(ctx, tname);
        if (wv) matvec(tensor_ptr_on_gpu(ctx, wv, gpu_idx),
                        gpu->d_normed, gpu->d_v, NKV * HD, H, wv->type_id, cs);

        // ── 3. QK norms (optional, Qwen3 uses them) ──
        snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", layer);
        TensorEntry *qn = tensor_get(ctx, tname);
        if (qn) {
            for (int h = 0; h < NH; h++)
                cuda_rms_norm(gpu->d_q + h * HD,
                             (float *)tensor_ptr_on_gpu(ctx, qn, gpu_idx),
                             gpu->d_q + h * HD, HD, eps, cs);
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", layer);
        TensorEntry *kn = tensor_get(ctx, tname);
        if (kn) {
            for (int h = 0; h < NKV; h++)
                cuda_rms_norm(gpu->d_k + h * HD,
                             (float *)tensor_ptr_on_gpu(ctx, kn, gpu_idx),
                             gpu->d_k + h * HD, HD, eps, cs);
        }

        // ── 4. RoPE (GPU kernel) ──
        cuda_rope(gpu->d_q, gpu->d_k, HD, pos, cfg->rope_theta, NH, NKV, cs);

        // ── 5. KV cache store (FP32 → FP16 on GPU) ──
        int local_layer = layer - gpu->layer_start;
        int ctx_len = cfg->max_position_embeddings;
        size_t kv_layer_stride = (size_t)ctx_len * NKV * HD;

        void *kv_k_dst = (char *)gpu->d_kv_k + (local_layer * kv_layer_stride + (size_t)pos * NKV * HD) * sizeof(uint16_t);
        void *kv_v_dst = (char *)gpu->d_kv_v + (local_layer * kv_layer_stride + (size_t)pos * NKV * HD) * sizeof(uint16_t);
        cuda_f32_to_f16(gpu->d_k, kv_k_dst, NKV * HD, cs);
        cuda_f32_to_f16(gpu->d_v, kv_v_dst, NKV * HD, cs);

        // ── 6. Attention: Q@K^T → softmax → @V (FP16 KV — half bandwidth) ──
        void *kv_k_layer = (char *)gpu->d_kv_k + local_layer * kv_layer_stride * sizeof(uint16_t);
        void *kv_v_layer = (char *)gpu->d_kv_v + local_layer * kv_layer_stride * sizeof(uint16_t);
        int gqa_ratio = NH / NKV;
        float attn_scale = 1.0f / sqrtf((float)HD);
        int seq_len = pos + 1;

        cuda_attention_f16kv(gpu->d_q, kv_k_layer, kv_v_layer, gpu->d_attn_out,
                             NH, HD, NKV, seq_len, gqa_ratio, attn_scale, cs);

        // ── 7. Output projection + residual (GPU) ──
        snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", layer);
        TensorEntry *wo = tensor_get(ctx, tname);
        if (wo) matvec(tensor_ptr_on_gpu(ctx, wo, gpu_idx),
                        gpu->d_attn_out, gpu->d_hidden, H, NH * HD, wo->type_id, cs);

        cuda_residual_add(gpu->d_hidden, gpu->d_residual, gpu->d_hidden, H, cs);
    } else {
        // Gated Delta Net layer: linear attention with delta rule
        forward_gdn_block(ctx, layer, gpu_idx);
    }

    // ── 8. Pre-FFN RMS norm ──
    snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", layer);
    void *ffn_norm_w = tensor_data_on_gpu(ctx, tname, gpu_idx);
    if (!ffn_norm_w) return;  // no FFN on this layer (pure SSM without MoE)
    cuda_rms_norm(gpu->d_hidden, (float *)ffn_norm_w, gpu->d_normed, H, eps, cs);

    // Save residual for MoE skip connection
    cudaMemcpyAsync(gpu->d_residual, gpu->d_hidden, H * sizeof(float),
                    cudaMemcpyDeviceToDevice, cs);

    // ── 9. Router: F32 matvec on GPU ──
    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.weight", layer);
    TensorEntry *wr = tensor_get(ctx, tname);
    if (!wr) return;

    // Router is F32 column-major [H, NE] — use F32 matvec kernel on GPU
    matvec(tensor_ptr_on_gpu(ctx, wr, gpu_idx),
           gpu->d_normed, gpu->d_router_out, NE, H, wr->type_id, cs);

    // ── 10. Top-K expert selection (GPU — no explicit stream sync) ──
    cuda_topk_softmax(gpu->d_router_out, gpu->d_expert_indices,
                      gpu->d_expert_weights, NE, K, cs);
    int expert_indices[16];
    float expert_weights_h[16];
    cudaMemcpy(expert_indices, gpu->d_expert_indices, K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(expert_weights_h, gpu->d_expert_weights, K * sizeof(float), cudaMemcpyDeviceToHost);

    // ── 11 + 12. Expert load (cache-first) + forward on GPU ──
    ExpertLayerFile *elf = &ctx->expert_layers[layer];
    if (elf->fd <= 0 && !elf->mmap_data) return;

    size_t expert_size = elf->expert_size;
    size_t gate_sz = elf->gate_size > 0 ? elf->gate_size : expert_size / 3;
    size_t up_sz = elf->up_size > 0 ? elf->up_size : expert_size / 3;
    size_t down_sz = elf->down_size > 0 ? elf->down_size : expert_size - gate_sz - up_sz;

    size_t expected_q4k_down = (size_t)((H * EFF + 255) / 256) * 144;
    int down_type_id = (down_sz > expected_q4k_down * 1.2) ? 14 : 12;

    // Zero MoE accumulator on GPU
    cudaMemsetAsync(gpu->d_moe_out, 0, H * sizeof(float), cs);

    // ── Wait for any pending prefetch from previous layer ──
    if (gpu->prefetch_layer == layer && !gpu->prefetch_ready)
        prefetch_wait(gpu, gpu->n_prefetched);

    // ── Classify experts: VRAM hit → prefetch hit → RAM hit → NVMe miss ──
    static int total_lookups = 0, total_hits = 0;
    static int total_prefetch_hits = 0, total_ram_hits = 0;

    void *hit_ptrs[16]; float hit_weights[16]; int n_hits = 0;
    int miss_eids[16]; float miss_weights[16]; int n_misses = 0;
    // RAM-hit experts: already in pinned host, just need GPU upload
    void *ram_hit_ptrs[16]; float ram_hit_weights[16]; int ram_hit_eids[16]; int n_ram_hits = 0;

    for (int e = 0; e < K && e < 16; e++) {
        int eid = expert_indices[e];
        if (eid < 0 || eid >= NE) continue;

        // Heat profiling: count this activation
        if (ctx->heat_map)
            ctx->heat_map[layer * NE + eid]++;

        total_lookups++;

        // Level 1: VRAM cache (instant, on-GPU)
        void *cached = expert_cache_lookup(gpu, layer, eid);
        if (cached) {
            total_hits++;
            if (ctx->layer_hits) ctx->layer_hits[layer]++;
            hit_ptrs[n_hits] = cached;
            hit_weights[n_hits++] = expert_weights_h[e];
            continue;
        }

        // Level 2: Prefetch buffer (pinned host, predicted)
        void *prefetched = prefetch_lookup(gpu, layer, eid, expert_size);
        if (prefetched) {
            total_prefetch_hits++;
            total_hits++;
            void *d = expert_cache_insert_ctx(ctx, gpu, layer, eid, prefetched,
                                               expert_size, gpu->stream_io);
            if (d) {
                cudaStreamSynchronize(gpu->stream_io);
                hit_ptrs[n_hits] = d;
                hit_weights[n_hits++] = expert_weights_h[e];
            } else {
                miss_eids[n_misses] = eid;
                miss_weights[n_misses++] = expert_weights_h[e];
            }
            continue;
        }

        // Level 3: RAM cache (pinned host, fast PCIe upload)
        void *ram_cached = ram_cache_lookup(&ctx->ram_cache, layer, eid);
        if (ram_cached) {
            total_ram_hits++;
            total_hits++;
            ram_hit_ptrs[n_ram_hits] = ram_cached;
            ram_hit_weights[n_ram_hits] = expert_weights_h[e];
            ram_hit_eids[n_ram_hits] = eid;
            n_ram_hits++;
            continue;
        }

        // Level 4: NVMe miss — must read from disk
        miss_eids[n_misses] = eid;
        miss_weights[n_misses++] = expert_weights_h[e];
        if (ctx->layer_misses) ctx->layer_misses[layer]++;
    }

    // Save activated experts for cross-layer prefetch correlation
    ctx->n_last_activated = 0;
    for (int e = 0; e < K && e < 16; e++) {
        if (expert_indices[e] >= 0 && expert_indices[e] < NE)
            ctx->last_activated[ctx->n_last_activated++] = expert_indices[e];
    }

    if (total_lookups % 5000 == 0)
        LOG_INFO("L1:%d L2(pf):%d L3(ram):%d L4(nvme):%d / %d (%.0f%% hit)",
                total_hits - total_prefetch_hits - total_ram_hits,
                total_prefetch_hits, total_ram_hits,
                total_lookups - total_hits,
                total_lookups, 100.0 * total_hits / total_lookups);

    // ── Upload RAM-hit experts to VRAM (fast: pinned → device DMA) ──
    for (int r = 0; r < n_ram_hits; r++) {
        void *d = expert_cache_insert_ctx(ctx, gpu, layer, ram_hit_eids[r],
                                           ram_hit_ptrs[r], expert_size, gpu->stream_io);
        if (d) {
            cudaStreamSynchronize(gpu->stream_io);
            hit_ptrs[n_hits] = d;
            hit_weights[n_hits++] = ram_hit_weights[r];
        } else {
            // VRAM full and all pinned — compute directly from device temp buffer
            cudaMemcpyAsync(gpu->d_expert_buf, ram_hit_ptrs[r], expert_size,
                            cudaMemcpyHostToDevice, gpu->stream_io);
            cudaStreamSynchronize(gpu->stream_io);
            hit_ptrs[n_hits] = gpu->d_expert_buf;
            hit_weights[n_hits++] = ram_hit_weights[r];
        }
    }

    // ── Load NVMe misses: batch I/O in chunks of io_pool_size() ──
    if (n_misses > 0) {
        int pool_sz = io_pool_size();
        if (pool_sz < 1) pool_sz = 1;
        int hits_computed = 0;

        // Process misses in batches that fit the I/O pool
        for (int batch_start = 0; batch_start < n_misses; batch_start += pool_sz) {
            int batch_end = batch_start + pool_sz;
            if (batch_end > n_misses) batch_end = n_misses;
            int batch_n = batch_end - batch_start;

            // Submit this batch of I/O tasks
            IOTask io_tasks[16];
            for (int b = 0; b < batch_n; b++) {
                int i = batch_start + b;
                io_tasks[b].dst = (char *)gpu->h_expert_buf + (size_t)i * expert_size;
                io_tasks[b].size = expert_size;
                io_tasks[b].done = 0;
                io_tasks[b].error = 0;
                if (elf->mmap_data) {
                    io_tasks[b].fd = -1;
                    io_tasks[b].src = (char *)elf->mmap_data + (size_t)miss_eids[i] * expert_size;
                } else {
                    io_tasks[b].fd = elf->fd;
                    io_tasks[b].src = NULL;
                    io_tasks[b].offset = (off_t)miss_eids[i] * (off_t)expert_size;
                }
            }
            io_pool_submit(io_tasks, batch_n);

            // Overlap: compute cache hits while first batch is loading
            if (!hits_computed) {
                for (int h = 0; h < n_hits; h++) {
                    void *d = hit_ptrs[h];
                    matvec(d, gpu->d_normed, gpu->d_expert_gate, EFF, H, 12, cs);
                    matvec((char *)d + gate_sz, gpu->d_normed, gpu->d_expert_up, EFF, H, 12, cs);
                    cuda_swiglu(gpu->d_expert_gate, gpu->d_expert_up, gpu->d_expert_act, EFF, cs);
                    matvec((char *)d + gate_sz + up_sz, gpu->d_expert_act, gpu->d_expert_out, H, EFF, down_type_id, cs);
                    cuda_scaled_add(gpu->d_moe_out, gpu->d_expert_out, hit_weights[h], H, cs);
                }
                cudaStreamSynchronize(cs);
                hits_computed = 1;
            }

            // Wait for this batch
            io_pool_wait(batch_n);

            // Process loaded experts
            for (int b = 0; b < batch_n; b++) {
                int i = batch_start + b;

                // Skip experts with I/O errors
                if (io_pool_task_error(b) != 0) {
                    LOG_ERROR("I/O error loading layer %d expert %d, skipping",
                            layer, miss_eids[i]);
                    continue;
                }

                void *h_buf = (char *)gpu->h_expert_buf + (size_t)i * expert_size;

                ram_cache_insert(&ctx->ram_cache, layer, miss_eids[i], h_buf, expert_size);

                void *d = expert_cache_insert_ctx(ctx, gpu, layer, miss_eids[i], h_buf,
                                                   expert_size, gpu->stream_io);
                if (!d) {
                    cudaMemcpyAsync(gpu->d_expert_buf, h_buf, expert_size,
                                    cudaMemcpyHostToDevice, gpu->stream_io);
                    d = gpu->d_expert_buf;
                }

                cudaEvent_t upload_done;
                cudaEventCreateWithFlags(&upload_done, cudaEventDisableTiming);
                cudaEventRecord(upload_done, gpu->stream_io);
                cudaStreamWaitEvent(cs, upload_done, 0);
                cudaEventDestroy(upload_done);

                matvec(d, gpu->d_normed, gpu->d_expert_gate, EFF, H, 12, cs);
                matvec((char *)d + gate_sz, gpu->d_normed, gpu->d_expert_up, EFF, H, 12, cs);
                cuda_swiglu(gpu->d_expert_gate, gpu->d_expert_up, gpu->d_expert_act, EFF, cs);
                matvec((char *)d + gate_sz + up_sz, gpu->d_expert_act, gpu->d_expert_out, H, EFF, down_type_id, cs);
                cuda_scaled_add(gpu->d_moe_out, gpu->d_expert_out, miss_weights[i], H, cs);
            }
        }

        // If there were no misses batches but we have hits, compute them now
        if (!hits_computed) {
            for (int h = 0; h < n_hits; h++) {
                void *d = hit_ptrs[h];
                matvec(d, gpu->d_normed, gpu->d_expert_gate, EFF, H, 12, cs);
                matvec((char *)d + gate_sz, gpu->d_normed, gpu->d_expert_up, EFF, H, 12, cs);
                cuda_swiglu(gpu->d_expert_gate, gpu->d_expert_up, gpu->d_expert_act, EFF, cs);
                matvec((char *)d + gate_sz + up_sz, gpu->d_expert_act, gpu->d_expert_out, H, EFF, down_type_id, cs);
                cuda_scaled_add(gpu->d_moe_out, gpu->d_expert_out, hit_weights[h], H, cs);
            }
        }
    } else {
        // All hits — just compute
        for (int h = 0; h < n_hits; h++) {
            void *d = hit_ptrs[h];
            matvec(d, gpu->d_normed, gpu->d_expert_gate, EFF, H, 12, cs);
            matvec((char *)d + gate_sz, gpu->d_normed, gpu->d_expert_up, EFF, H, 12, cs);
            cuda_swiglu(gpu->d_expert_gate, gpu->d_expert_up, gpu->d_expert_act, EFF, cs);
            matvec((char *)d + gate_sz + up_sz, gpu->d_expert_act, gpu->d_expert_out, H, EFF, down_type_id, cs);
            cuda_scaled_add(gpu->d_moe_out, gpu->d_expert_out, hit_weights[h], H, cs);
        }
    }

    // ── 12b. Start prefetching hot experts for NEXT layer (overlaps with GPU compute) ──
    {
        int next_layer = layer + 1;
        if (next_layer < cfg->num_hidden_layers &&
            next_layer >= gpu->layer_start && next_layer < gpu->layer_end) {
            // For SSM layers (no MoE), skip to the next MoE layer
            if (cfg->full_attention_interval > 0) {
                // Find next layer that has MoE (all layers have MoE in Qwen3.5)
                prefetch_submit(ctx, gpu, next_layer);
            } else {
                prefetch_submit(ctx, gpu, next_layer);
            }
        }

        // Extra prefetch: also prefetch layer+2 for deeper pipelining
        if (ctx->extra_prefetch) {
            int layer2 = layer + 2;
            if (layer2 < cfg->num_hidden_layers &&
                layer2 >= gpu->layer_start && layer2 < gpu->layer_end) {
                // Only submit if pool has capacity (don't block current I/O)
                // We rely on the fact that prefetch_submit for layer+1 already
                // submitted and this will be queued after it completes
                prefetch_submit(ctx, gpu, layer2);
            }
        }
    }

    // ── 13. MoE output + residual (GPU) ──
    cuda_residual_add(gpu->d_moe_out, gpu->d_residual, gpu->d_hidden, H, cs);

}

// ── Full forward pass: embedding → layers → lm_head ──

int forward_pass(MnemoCudaCtx *ctx, int token_id, int pos, float *h_logits) {
    ModelConfig *cfg = &ctx->config;
    int H = cfg->hidden_size;
    int V = cfg->vocab_size;

    if (pos >= cfg->max_position_embeddings) {
        LOG_ERROR("pos %d exceeds context length %d",
                pos, cfg->max_position_embeddings);
        return -4;
    }

    // ── 1. Token embedding lookup ──
    TensorEntry *embd = tensor_get(ctx, "token_embd.weight");
    if (!embd) { LOG_ERROR("Missing token_embd.weight"); return -1; }

    // Embedding is quantized — extract one row on host, then upload
    // Row size depends on quant type
    // For Q4_K: block_size = 256 values, block_bytes = sizeof(BlockQ4K) = 144
    // Row bytes = (H / 256) * 144
    int block_values, block_bytes_size;
    switch (embd->type_id) {
        case 11: block_values = 256; block_bytes_size = 110; break; // Q3_K
        case 12: block_values = 256; block_bytes_size = 144; break; // Q4_K
        case 14: block_values = 256; block_bytes_size = 210; break; // Q6_K
        case 8:  block_values = 32;  block_bytes_size = 34; break;  // Q8_0
        default: block_values = 1; block_bytes_size = 4; break;     // F32
    }

    size_t row_bytes = (size_t)((H + block_values - 1) / block_values) * block_bytes_size;
    size_t row_offset = (size_t)token_id * row_bytes;

    // Upload embedding row to GPU 0 and dequant via matvec with identity
    // Actually: for embedding lookup, we need a dequant kernel, not matvec.
    // Simpler approach: dequant on host, upload float vector.
    const void *embd_row = (const char *)ctx->h_resident_mmap + embd->offset + row_offset;

    // Host-side dequant of one embedding row
    float *h_hidden = (float *)malloc(H * sizeof(float));

    if (embd->type_id == 12) {
        // Q4_K dequant
        int blocks_per_row = (H + 255) / 256;
        const uint8_t *raw = (const uint8_t *)embd_row;
        for (int b = 0; b < blocks_per_row; b++) {
            // BlockQ4K: [d:2][dmin:2][scales:12][qs:128] = 144 bytes
            const uint8_t *blk = raw + b * 144;
            uint16_t d_bits = *(const uint16_t *)blk;
            uint16_t dmin_bits = *(const uint16_t *)(blk + 2);
            // f16 → f32 (handles denormals correctly)
            float d, dmin;
            { uint32_t exp = (d_bits & 0x7C00);
              uint32_t mant = (d_bits & 0x03FF);
              uint32_t sign = (uint32_t)(d_bits & 0x8000) << 16;
              uint32_t t;
              if (exp == 0) {
                  if (mant == 0) { t = sign; }
                  else { float f = (float)mant / 1024.0f; f *= (1.0f / 16384.0f); memcpy(&t, &f, 4); t = (t & 0x7FFFFFFF) | sign; }
              } else {
                  t = sign | ((exp + 0x1C000) << 13) | (mant << 13);
              }
              memcpy(&d, &t, 4); }
            { uint32_t exp = (dmin_bits & 0x7C00);
              uint32_t mant = (dmin_bits & 0x03FF);
              uint32_t sign = (uint32_t)(dmin_bits & 0x8000) << 16;
              uint32_t t;
              if (exp == 0) {
                  if (mant == 0) { t = sign; }
                  else { float f = (float)mant / 1024.0f; f *= (1.0f / 16384.0f); memcpy(&t, &f, 4); t = (t & 0x7FFFFFFF) | sign; }
              } else {
                  t = sign | ((exp + 0x1C000) << 13) | (mant << 13);
              }
              memcpy(&dmin, &t, 4); }

            const uint8_t *scales = blk + 4;
            const uint8_t *qs = blk + 16;
            int base = b * 256;

            for (int j = 0; j < 256 && (base + j) < H; j += 64) {
                int is = j / 32;  // scale index (0..7 across the 256-value block)
                uint8_t sc, m;
                // get_scale_min_k4: use is directly (NOT is/2)
                if (is < 4) {
                    sc = scales[is] & 63;
                    m = scales[is + 4] & 63;
                } else {
                    sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                    m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
                }

                float d1 = d * sc, m1 = dmin * m;
                // First 32 values: LOW nibbles of qs[qs_off..qs_off+31]
                int qs_off = j / 2;
                for (int l = 0; l < 32 && (base + j + l) < H; l++) {
                    uint8_t q_val = qs[qs_off + l] & 0xF;
                    h_hidden[base + j + l] = d1 * q_val - m1;
                }

                // Second half of the 64-block: HIGH nibbles of SAME bytes
                is++;
                if (is < 4) {
                    sc = scales[is] & 63;
                    m = scales[is + 4] & 63;
                } else {
                    sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                    m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
                }
                float d2 = d * sc, m2 = dmin * m;
                for (int l = 0; l < 32 && (base + j + 32 + l) < H; l++) {
                    uint8_t q_val = (qs[qs_off + l] >> 4) & 0xF;
                    h_hidden[base + j + 32 + l] = d2 * q_val - m2;
                }
            }
        }
    } else if (embd->type_id == 11) {
        // Q3_K dequant: [hmask:32][qs:64][scales:12][d:2] = 110 bytes per block of 256
        int blocks_per_row = (H + 255) / 256;
        const uint8_t *raw = (const uint8_t *)embd_row;
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *blk = raw + b * 110;
            const uint8_t *hmask = blk;       // 32 bytes
            const uint8_t *qs = blk + 32;     // 64 bytes
            const uint8_t *scales_raw = blk + 96; // 12 bytes
            uint16_t d_bits = *(const uint16_t *)(blk + 108);
            float d_all;
            { uint32_t t = ((uint32_t)(d_bits & 0x8000) << 16) |
                           (((uint32_t)(d_bits & 0x7C00) + 0x1C000) << 13) |
                           ((uint32_t)(d_bits & 0x03FF) << 13);
              memcpy(&d_all, &t, 4); }

            // Decode 6-bit packed scales → 16 int8 values
            int8_t sc16[16];
            for (int i = 0; i < 8; i++) {
                sc16[i]     = (int8_t)((scales_raw[i] & 0xF) | (((scales_raw[8 + (i >> 2)] >> (2 * (i & 3))) & 3) << 4)) - 32;
                sc16[i + 8] = (int8_t)((scales_raw[i] >> 4)  | (((scales_raw[8 + (i >> 2)] >> (2 * (i & 3) + 4)) & 3) << 4)) - 32;
            }

            int base = b * 256;
            const uint8_t *q = qs;
            int is = 0;
            uint8_t m = 1;
            int val_idx = 0;

            for (int chunk = 0; chunk < 2; chunk++) {
                int shift = 0;
                for (int grp = 0; grp < 4; grp++) {
                    float dl = d_all * sc16[is++];
                    for (int l = 0; l < 16; l++) {
                        if (base + val_idx < H) {
                            int8_t q2 = (q[l] >> shift) & 3;
                            int8_t val = q2 - ((hmask[l] & m) ? 0 : 4);
                            h_hidden[base + val_idx] = dl * val;
                        }
                        val_idx++;
                    }
                    float dl2 = d_all * sc16[is++];
                    for (int l = 0; l < 16; l++) {
                        if (base + val_idx < H) {
                            int8_t q2 = (q[l + 16] >> shift) & 3;
                            int8_t val = q2 - ((hmask[l + 16] & m) ? 0 : 4);
                            h_hidden[base + val_idx] = dl2 * val;
                        }
                        val_idx++;
                    }
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }
    } else if (embd->type_id == 0) {
        // F32 — direct copy
        memcpy(h_hidden, embd_row, H * sizeof(float));
    } else {
        // Fallback: zero (shouldn't happen)
        memset(h_hidden, 0, H * sizeof(float));
    }

    // Upload to GPU 0
    GPUState *gpu0 = &ctx->gpus[0];
    cudaSetDevice(gpu0->gpu_id);
    cudaMemcpyAsync(gpu0->d_hidden, h_hidden, H * sizeof(float),
                    cudaMemcpyHostToDevice, gpu0->stream_compute);
    free(h_hidden);

    // ── 2. Layer loop (multi-GPU pipeline) ──
    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        cudaSetDevice(gpu->gpu_id);

        // If not first GPU, transfer hidden state from previous GPU
        if (g > 0) {
            GPUState *prev = &ctx->gpus[g - 1];
            // prev GPU → pinned host → this GPU
            cudaSetDevice(prev->gpu_id);
            cudaMemcpy(ctx->h_hidden_transfer, prev->d_hidden, H * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaSetDevice(gpu->gpu_id);
            cudaMemcpyAsync(gpu->d_hidden, ctx->h_hidden_transfer, H * sizeof(float),
                            cudaMemcpyHostToDevice, gpu->stream_compute);
        }

        // Run all layers owned by this GPU
        for (int layer = gpu->layer_start; layer < gpu->layer_end; layer++)
            forward_layer(ctx, layer, g, pos);
    }

    // ── 3. Final RMS norm + LM head (on last GPU) ──
    int last_g = ctx->n_gpus - 1;
    GPUState *last_gpu = &ctx->gpus[last_g];
    cudaSetDevice(last_gpu->gpu_id);
    cudaStream_t cs = last_gpu->stream_compute;

    TensorEntry *out_norm = tensor_get(ctx, "output_norm.weight");
    if (out_norm) {
        void *norm_w = tensor_ptr_on_gpu(ctx, out_norm, last_g);
        cuda_rms_norm(last_gpu->d_hidden, (float *)norm_w, last_gpu->d_normed,
                      H, cfg->rms_norm_eps, cs);
    }

    TensorEntry *lm_head = tensor_get(ctx, "output.weight");
    if (lm_head && last_gpu->d_logits) {
        void *lm_w = tensor_ptr_on_gpu(ctx, lm_head, last_g);
        matvec(lm_w, last_gpu->d_normed, last_gpu->d_logits,
               V, H, lm_head->type_id, cs);
    }

    // Copy logits to host
    cudaStreamSynchronize(cs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        LOG_ERROR("CUDA error before logits copy: %s", cudaGetErrorString(err));
    cudaMemcpy(h_logits, last_gpu->d_logits, V * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

