/**
 * MnemoCUDA Engine — Core implementation.
 *
 * Memory layout:
 *   GPU VRAM: resident weights (attention, norms, routing, embeddings, lm_head)
 *   Pinned host: double-buffered expert staging (pread → pinned → cudaMemcpyAsync → device)
 *   GPU VRAM: compute buffers (hidden, q/k/v, expert intermediates)
 *   GPU VRAM: KV cache
 *
 * I/O pipeline:
 *   Thread pool (8 threads) → pread expert data → pinned buffer A
 *   CUDA stream 1: compute on expert buffer B (from previous layer)
 *   Overlap: I/O for next layer happens while GPU computes current layer
 */

#include "engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <errno.h>

// CUDA runtime
#include <cuda_runtime.h>

#include "engine_internal.h"
#include "forward.h"
#include "log.h"

// Default log level: INFO
MnemoLogLevel mnemo_log_level = MNEMO_LOG_INFO;

// ── I/O helpers (retry on partial read/write) ──

static ssize_t write_full(int fd, const void *buf, size_t count) {
    size_t written = 0;
    while (written < count) {
        ssize_t n = write(fd, (const char *)buf + written, count - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) break;
        written += (size_t)n;
    }
    return (ssize_t)written;
}

static ssize_t read_full(int fd, void *buf, size_t count) {
    size_t total = 0;
    while (total < count) {
        ssize_t n = read(fd, (char *)buf + total, count - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) break;  // EOF
        total += (size_t)n;
    }
    return (ssize_t)total;
}

static const char *json_next_string(const char *p, char *out, int maxlen) {
    p = strchr(p, '"');
    if (!p) return NULL;
    p++; // skip opening quote
    int i = 0;
    while (*p && *p != '"' && i < maxlen - 4) {  // -4: room for worst-case \uXXXX (3 bytes + NUL)
        if (*p == '\\' && *(p+1)) {
            p++;
            switch (*p) {
                case 'n': out[i++] = '\n'; break;
                case 't': out[i++] = '\t'; break;
                case '\\': out[i++] = '\\'; break;
                case '"': out[i++] = '"'; break;
                case '/': out[i++] = '/'; break;
                case 'u': {
                    unsigned cp = 0;
                    int j;
                    for (j = 0; j < 4 && p[1+j]; j++) {
                        char c = p[1+j];
                        cp = cp * 16 + (c >= 'a' ? c-'a'+10 : c >= 'A' ? c-'A'+10 : c-'0');
                    }
                    p += j;
                    if (cp < 0x80) {
                        out[i++] = (char)cp;
                    } else if (cp < 0x800) {
                        out[i++] = (char)(0xC0 | (cp >> 6));
                        out[i++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        out[i++] = (char)(0xE0 | (cp >> 12));
                        out[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[i++] = (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: out[i++] = *p; break;
            }
        } else {
            out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    return (*p == '"') ? p + 1 : p;
}

// ── Sampling (temperature + top-p) ──

// xoroshiro128+ RNG
static uint64_t rng_state[2] = {0x12345678DEADBEEF, 0xFEDCBA9876543210};

static uint64_t rng_next(void) {
    uint64_t s0 = rng_state[0], s1 = rng_state[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    rng_state[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
    rng_state[1] = (s1 << 37) | (s1 >> 27);
    return result;
}

static float rng_float(void) {
    return (float)(rng_next() >> 11) * (1.0f / 9007199254740992.0f);
}

static void rng_seed(uint64_t seed) {
    rng_state[0] = seed;
    rng_state[1] = seed ^ 0xDEADBEEFCAFEBABE;
    for (int i = 0; i < 10; i++) rng_next(); // warm up
}

static int sample_token(const float *logits, int vocab_size,
                        float temperature, float top_p) {
    // Temperature 0 or greedy
    if (temperature <= 0.0f || top_p <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    // Softmax with temperature
    float *probs = malloc(vocab_size * sizeof(float));
    if (!probs) return 0;  // OOM fallback: greedy
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > max_logit) max_logit = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) probs[i] /= sum;

    int sampled;
    if (top_p < 1.0f) {
        // Top-p (nucleus) sampling — sort descending, find cutoff
        typedef struct { float p; int id; } ProbId;
        ProbId *sorted = malloc(vocab_size * sizeof(ProbId));
        if (!sorted) { free(probs); return 0; }  // OOM fallback: greedy
        for (int i = 0; i < vocab_size; i++) { sorted[i].p = probs[i]; sorted[i].id = i; }

        // Partial sort: only need tokens until cumulative >= top_p
        // Use simple selection for top candidates
        float cumul = 0.0f;
        int n_nucleus = 0;
        while (cumul < top_p && n_nucleus < vocab_size) {
            int best = n_nucleus;
            for (int j = n_nucleus + 1; j < vocab_size; j++)
                if (sorted[j].p > sorted[best].p) best = j;
            ProbId tmp = sorted[n_nucleus];
            sorted[n_nucleus] = sorted[best];
            sorted[best] = tmp;
            cumul += sorted[n_nucleus].p;
            n_nucleus++;
        }

        // Renormalize and sample
        float nsum = 0.0f;
        for (int i = 0; i < n_nucleus; i++) nsum += sorted[i].p;
        float r = rng_float() * nsum;
        float acc = 0.0f;
        sampled = sorted[0].id;
        for (int i = 0; i < n_nucleus; i++) {
            acc += sorted[i].p;
            if (acc >= r) { sampled = sorted[i].id; break; }
        }
        free(sorted);
    } else {
        // Full distribution sampling
        float r = rng_float();
        float acc = 0.0f;
        sampled = 0;
        for (int i = 0; i < vocab_size; i++) {
            acc += probs[i];
            if (acc >= r) { sampled = i; break; }
        }
    }

    free(probs);
    return sampled;
}

// (I/O pool moved to io_pool.c/h)
#include "io_pool.h"



// ── RAM cache implementation (type in engine_internal.h) ──

int ram_cache_init(RAMCache *rc, size_t slot_size, int target_gb) {
    memset(rc, 0, sizeof(*rc));
    rc->slot_size = slot_size;
    rc->n_slots = (int)((size_t)target_gb * 1024 * 1024 * 1024 / slot_size);
    if (rc->n_slots < 16) rc->n_slots = 16;

    // Use regular malloc — NOT cudaMallocHost, which steals from OS page cache
    // On Linux with mmap'd expert files, the page cache IS the primary RAM cache.
    // This explicit cache only covers the DMA staging gap.
    size_t total = (size_t)rc->n_slots * slot_size;
    rc->data = malloc(total);
    if (!rc->data) {
        LOG_ERROR("RAM cache: allocation failed");
        rc->n_slots = 0;
        return -1;
    }

    rc->slot_layer = (int *)malloc(rc->n_slots * sizeof(int));
    rc->slot_expert = (int *)malloc(rc->n_slots * sizeof(int));
    rc->slot_lru = (uint64_t *)calloc(rc->n_slots, sizeof(uint64_t));
    for (int i = 0; i < rc->n_slots; i++) {
        rc->slot_layer[i] = -1;
        rc->slot_expert[i] = -1;
    }
    rc->clock = 0;
    rc->hits = 0;
    rc->misses = 0;

    LOG_INFO("RAM cache: %.1f GB (%d slots × %.1f MB)",
            (double)total / (1024*1024*1024), rc->n_slots,
            (double)slot_size / (1024*1024));
    return 0;
}

// Returns pointer to expert data in RAM cache, or NULL
void *ram_cache_lookup(RAMCache *rc, int layer, int expert_id) {
    if (!rc->data || rc->n_slots == 0) return NULL;
    for (int s = 0; s < rc->n_slots; s++) {
        if (rc->slot_layer[s] == layer && rc->slot_expert[s] == expert_id) {
            rc->slot_lru[s] = ++rc->clock;
            rc->hits++;
            return (char *)rc->data + (size_t)s * rc->slot_size;
        }
    }
    rc->misses++;
    return NULL;
}

// Insert expert into RAM cache (evict LRU if full), returns host pointer
void *ram_cache_insert(RAMCache *rc, int layer, int expert_id,
                               const void *src, size_t size) {
    if (!rc->data || rc->n_slots == 0) return NULL;

    // Find empty or LRU slot
    int target = -1;
    uint64_t min_lru = UINT64_MAX;
    for (int s = 0; s < rc->n_slots; s++) {
        if (rc->slot_layer[s] == -1) { target = s; break; }
        if (rc->slot_lru[s] < min_lru) { min_lru = rc->slot_lru[s]; target = s; }
    }
    if (target < 0) return NULL;

    rc->slot_layer[target] = layer;
    rc->slot_expert[target] = expert_id;
    rc->slot_lru[target] = ++rc->clock;

    void *dst = (char *)rc->data + (size_t)target * rc->slot_size;
    if (src) memcpy(dst, src, size > rc->slot_size ? rc->slot_size : size);
    return dst;
}

void ram_cache_free(RAMCache *rc) {
    free(rc->data);
    free(rc->slot_layer);
    free(rc->slot_expert);
    free(rc->slot_lru);
    memset(rc, 0, sizeof(*rc));
}

// ── Implementation ──

const char *mnemo_cuda_strerror(int err) {
    switch (err) {
        case MNEMO_OK:              return "success";
        case MNEMO_ERR_BAD_CONFIG:  return "invalid configuration or missing model directory";
        case MNEMO_ERR_TOKENIZER:   return "tokenizer load or encode failure";
        case MNEMO_ERR_NO_GPU:      return "no CUDA GPUs detected or invalid GPU IDs";
        case MNEMO_ERR_CONTEXT_FULL:return "position exceeds context length";
        case MNEMO_ERR_CUDA:        return "CUDA runtime error";
        case MNEMO_ERR_IO:          return "file I/O error";
        case MNEMO_ERR_CANCELLED:   return "generation cancelled";
        default:                    return "unknown error";
    }
}

MnemoCudaConfig mnemo_cuda_config_default(void) {
    MnemoCudaConfig c = {0};
    c.context_length = 2048;  // 2K default — keeps KV cache small, maximizes expert cache slots
    c.n_gpus = 0;             // 0 = auto-detect
    c.io_threads = 8;
    c.use_pinned_memory = true;
    return c;
}

MnemoCudaCtx *mnemo_cuda_create(void) {
    MnemoCudaCtx *ctx = calloc(1, sizeof(MnemoCudaCtx));
    return ctx;
}

// Minimal JSON value extractor — finds "key": value in config.json
int json_get_int(const char *json, const char *key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}
float json_get_float(const char *json, const char *key, float default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return (float)atof(p);
}

// Helper: read a required int from config, return -1 if missing
static int cfg_require_int(const char *json, const char *key, int *out) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    *out = (int)strtol(p, NULL, 10);
    return 0;
}

static int load_config_json(MnemoCudaCtx *ctx, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        LOG_ERROR("Cannot open config: %s", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 0 || size > 10 * 1024 * 1024) {
        LOG_ERROR("Config file invalid size: %ld", size);
        fclose(f); return -1;
    }
    fseek(f, 0, SEEK_SET);
    char *json = malloc(size + 1);
    if (!json) { fclose(f); return -1; }
    if (fread(json, 1, size, f) != (size_t)size) {
        LOG_ERROR("Config read error");
        free(json); fclose(f); return -1;
    }
    json[size] = '\0';
    fclose(f);

    // Auto-detect architecture prefix (order matters: longer prefixes first)
    const char *prefix = "qwen3moe";
    if (strstr(json, "\"qwen35moe.")) prefix = "qwen35moe";
    else if (strstr(json, "\"glm4moe.")) prefix = "glm4moe";
    else if (strstr(json, "\"qwen3next.")) prefix = "qwen3next";
    else if (strstr(json, "\"llama.")) prefix = "llama";
    else if (strstr(json, "\"qwen2.")) prefix = "qwen2";
    LOG_INFO("Config prefix: %s", prefix);

    char key[256];
    int missing = 0;

    #define CFG_INT(field, suffix, def) \
        snprintf(key, sizeof(key), "%s.%s", prefix, suffix); \
        ctx->config.field = json_get_int(json, key, def)
    #define CFG_FLOAT(field, suffix, def) \
        snprintf(key, sizeof(key), "%s.%s", prefix, suffix); \
        ctx->config.field = json_get_float(json, key, def)
    #define CFG_REQUIRE(field, suffix) do { \
        snprintf(key, sizeof(key), "%s.%s", prefix, suffix); \
        if (cfg_require_int(json, key, &ctx->config.field) != 0) { \
            LOG_ERROR("Missing required config key: %s", key); \
            missing++; \
        } \
    } while(0)

    // Required fields — fail if any missing
    CFG_REQUIRE(hidden_size, "embedding_length");
    CFG_REQUIRE(num_hidden_layers, "block_count");
    CFG_REQUIRE(num_attention_heads, "attention.head_count");
    CFG_REQUIRE(num_key_value_heads, "attention.head_count_kv");

    if (missing > 0) {
        LOG_ERROR("Config missing %d required field(s), cannot load", missing);
        free(json);
        return -1;
    }

    // Optional fields with defaults
    CFG_INT(moe_intermediate_size, "expert_feed_forward_length", 1536);
    CFG_INT(head_dim, "attention.key_length", 128);
    CFG_INT(num_experts, "expert_count", 128);
    CFG_INT(num_experts_per_tok, "expert_used_count", 8);
    CFG_FLOAT(rope_theta, "rope.freq_base", 1000000.0f);
    CFG_FLOAT(rms_norm_eps, "attention.layer_norm_rms_epsilon", 1e-6f);
    CFG_INT(max_position_embeddings, "context_length", 40960);
    CFG_INT(full_attention_interval, "full_attention_interval", 0);
    CFG_INT(ssm_inner_size, "ssm.inner_size", 0);
    CFG_INT(ssm_state_size, "ssm.state_size", 0);
    CFG_INT(ssm_conv_kernel, "ssm.conv_kernel", 4);
    CFG_INT(ssm_group_count, "ssm.group_count", 1);
    CFG_INT(ssm_dt_rank, "ssm.time_step_rank", 64);

    // Read vocab_size from metadata (try multiple common keys)
    int vs = json_get_int(json, "tokenizer.ggml.tokens_count", 0);
    if (vs <= 0) vs = json_get_int(json, "vocab_size", 0);
    if (vs <= 0) {
        // Fallback: read from tokenizer if loaded later, use common default for now
        vs = 151936;
        LOG_WARN("vocab_size not in config, using default %d", vs);
    }
    ctx->config.vocab_size = vs;

    #undef CFG_INT
    #undef CFG_FLOAT
    #undef CFG_REQUIRE

    // Sanity checks on loaded values
    if (ctx->config.hidden_size <= 0 || ctx->config.hidden_size > 65536) {
        LOG_ERROR("Invalid hidden_size: %d", ctx->config.hidden_size);
        free(json); return -1;
    }
    if (ctx->config.num_hidden_layers <= 0 || ctx->config.num_hidden_layers > 1024) {
        LOG_ERROR("Invalid num_hidden_layers: %d", ctx->config.num_hidden_layers);
        free(json); return -1;
    }

    free(json);
    return 0;
}

// Fail-fast macro for CUDA calls during load.
// Cleans up partial state via mnemo_cuda_unload (idempotent) before returning.
#define CUDA_LOAD_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        LOG_ERROR("CUDA error in load at %s:%d: %s", \
                  __FILE__, __LINE__, cudaGetErrorString(_err)); \
        mnemo_cuda_unload(ctx); \
        return -5; \
    } \
} while(0)

int mnemo_cuda_load(MnemoCudaCtx *ctx, MnemoCudaConfig config) {
    if (!ctx || !config.model_dir) return -1;

    strncpy(ctx->model_dir, config.model_dir, sizeof(ctx->model_dir) - 1);

    // Load config
    char path[1200];
    snprintf(path, sizeof(path), "%s/config.json", config.model_dir);
    if (load_config_json(ctx, path) != 0) {
        LOG_ERROR("Failed to load model config");
        mnemo_cuda_unload(ctx);
        return -1;
    }

    if (config.expert_k > 0)
        ctx->config.num_experts_per_tok = config.expert_k;

    if (ctx->config.num_experts_per_tok > MNEMO_MAX_EXPERT_K) {
        LOG_WARN("expert_k=%d exceeds maximum %d, clamping",
                ctx->config.num_experts_per_tok, MNEMO_MAX_EXPERT_K);
        ctx->config.num_experts_per_tok = MNEMO_MAX_EXPERT_K;
    }

    if (config.context_length < 128) {
        LOG_WARN("context_length=%d too small, using 128",
                config.context_length);
        config.context_length = 128;
    }
    if (config.context_length > 131072) {
        LOG_WARN("context_length=%d too large, capping at 131072",
                config.context_length);
        config.context_length = 131072;
    }
    ctx->config.max_position_embeddings = config.context_length;

    ModelConfig *cfg = &ctx->config;

    // Load tokenizer
    ctx->tokenizer = tokenizer_load(config.model_dir);
    if (!ctx->tokenizer) {
        LOG_WARN("Tokenizer not loaded, inference will fail");
    } else if (ctx->tokenizer->vocab_size > 0 && ctx->config.vocab_size != ctx->tokenizer->vocab_size) {
        LOG_INFO("Using tokenizer vocab_size=%d (config had %d)",
                ctx->tokenizer->vocab_size, ctx->config.vocab_size);
        ctx->config.vocab_size = ctx->tokenizer->vocab_size;
    }

    // Load resident_manifest.json → build tensor hash table
    snprintf(path, sizeof(path), "%s/resident_manifest.json", config.model_dir);
    FILE *mf2 = fopen(path, "r");
    if (mf2) {
        fseek(mf2, 0, SEEK_END);
        long mf2_size = ftell(mf2);
        fseek(mf2, 0, SEEK_SET);
        char *mf2_json = malloc(mf2_size + 1);
        if (!mf2_json) { fclose(mf2); LOG_ERROR("OOM: manifest json"); return MNEMO_ERR_CUDA; }
        if ((long)fread(mf2_json, 1, mf2_size, mf2) != mf2_size) {
            LOG_WARN("Short read on resident_manifest.json");
        }
        mf2_json[mf2_size] = '\0';
        fclose(mf2);

        // Count tensors (count "name" occurrences)
        int tensor_count = 0;
        const char *cnt = mf2_json;
        while ((cnt = strstr(cnt, "\"name\"")) != NULL) { tensor_count++; cnt += 6; }

        memset(&ctx->tensor_table, 0, sizeof(TensorTable));
        ctx->tensor_table.entries = calloc(tensor_count, sizeof(TensorEntry));
        if (!ctx->tensor_table.entries) { free(mf2_json); LOG_ERROR("OOM: tensor table"); return MNEMO_ERR_CUDA; }
        ctx->tensor_table.n_entries = 0;

        // Parse each tensor entry
        const char *tp = strstr(mf2_json, "\"tensors\"");
        if (tp) tp = strchr(tp, '[');
        if (tp) tp++;

        while (tp && ctx->tensor_table.n_entries < tensor_count) {
            tp = strstr(tp, "\"name\"");
            if (!tp) break;

            TensorEntry *e = &ctx->tensor_table.entries[ctx->tensor_table.n_entries];
            char val[256];

            // Name — skip past "name" (6 chars) then ": " to find the value string
            const char *after = json_next_string(tp + 6, val, sizeof(val));
            if (!after) break;
            strncpy(e->name, val, sizeof(e->name) - 1);
            tp = after;

            // Offset
            const char *off_p = strstr(tp, "\"offset\"");
            if (off_p && off_p < tp + 300) {
                off_p += 8;
                while (*off_p && *off_p != ':') off_p++;
                if (*off_p) e->offset = (size_t)strtoul(off_p + 1, NULL, 10);
            }

            // Size
            const char *sz_p = strstr(tp, "\"size\"");
            if (sz_p && sz_p < tp + 300) {
                sz_p += 6;
                while (*sz_p && *sz_p != ':') sz_p++;
                if (*sz_p) e->size = (size_t)strtoul(sz_p + 1, NULL, 10);
            }

            // Type
            const char *ty_p = strstr(tp, "\"type_id\"");
            if (ty_p && ty_p < tp + 300) {
                ty_p += 9;
                while (*ty_p && *ty_p != ':') ty_p++;
                if (*ty_p) e->type_id = atoi(ty_p + 1);
            }

            // Dims
            const char *dim_p = strstr(tp, "\"dims\"");
            if (dim_p && dim_p < tp + 400) {
                dim_p = strchr(dim_p, '[');
                if (dim_p) {
                    dim_p++;
                    e->n_dims = 0;
                    for (int d = 0; d < 4; d++) {
                        while (*dim_p == ' ' || *dim_p == '\t') dim_p++;
                        if (*dim_p == ']') break;
                        e->dims[d] = atoi(dim_p);
                        e->n_dims++;
                        while (*dim_p && *dim_p != ',' && *dim_p != ']') dim_p++;
                        if (*dim_p == ',') dim_p++;
                    }
                }
            }

            tensor_table_insert(&ctx->tensor_table, e);
            ctx->tensor_table.n_entries++;
        }

        free(mf2_json);
        LOG_INFO("Resident manifest: %d tensors indexed",
                ctx->tensor_table.n_entries);
    } else {
        LOG_WARN("No resident_manifest.json");
    }

    // Validate and set up GPUs
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0) {
        LOG_ERROR("No CUDA GPUs detected");
        mnemo_cuda_unload(ctx);
        return -3;
    }

    if (config.n_gpus == 0) {
        // Auto-detect
        ctx->n_gpus = device_count > 8 ? 8 : device_count;
        for (int i = 0; i < ctx->n_gpus; i++)
            ctx->gpus[i].gpu_id = i;
    } else {
        if (config.n_gpus < 0 || config.n_gpus > 8) {
            LOG_ERROR("n_gpus=%d out of range [1,8]", config.n_gpus);
            mnemo_cuda_unload(ctx);
            return -3;
        }
        ctx->n_gpus = config.n_gpus;
        for (int i = 0; i < ctx->n_gpus; i++) {
            int gid = config.gpu_ids[i];
            if (gid < 0 || gid >= device_count) {
                LOG_ERROR("gpu_ids[%d]=%d invalid (have %d devices)",
                        i, gid, device_count);
                mnemo_cuda_unload(ctx);
                return -3;
            }
            // Check for duplicate GPU IDs
            for (int j = 0; j < i; j++) {
                if (ctx->gpus[j].gpu_id == gid) {
                    LOG_ERROR("Duplicate gpu_id %d", gid);
                    mnemo_cuda_unload(ctx);
                    return -3;
                }
            }
            ctx->gpus[i].gpu_id = gid;
        }
    }

    LOG_INFO("Loading: hidden=%d, layers=%d, experts=%d (K=%d), GPUs=%d",
            cfg->hidden_size, cfg->num_hidden_layers,
            cfg->num_experts, cfg->num_experts_per_tok, ctx->n_gpus);

    if (cfg->full_attention_interval > 0) {
        int attn_layers = cfg->num_hidden_layers / cfg->full_attention_interval;
        int ssm_layers = cfg->num_hidden_layers - attn_layers;
        LOG_WARN("Hybrid MoE+SSM model: %d attention layers, %d SSM layers (interval=%d). "
                 "SSM layers will be skipped (not yet implemented)",
                 attn_layers, ssm_layers, cfg->full_attention_interval);
    }

    // Open expert files BEFORE CUDA allocations (CUDA may invalidate fds)
    ctx->expert_layers = calloc(cfg->num_hidden_layers, sizeof(ExpertLayerFile));
    if (!ctx->expert_layers) { LOG_ERROR("OOM: expert_layers"); return MNEMO_ERR_CUDA; }
    ctx->n_expert_layers = 0;
    {
        snprintf(path, sizeof(path), "%s/expert_manifest.json", config.model_dir);
        FILE *emf = fopen(path, "r");
        char *em_json = NULL;
        if (emf) {
            fseek(emf, 0, SEEK_END); long emsz = ftell(emf); fseek(emf, 0, SEEK_SET);
            em_json = malloc(emsz + 1);
            if (em_json) {
                if ((long)fread(em_json, 1, emsz, emf) != emsz)
                    LOG_WARN("Short read on expert_manifest.json");
                em_json[emsz] = '\0';
            }
            fclose(emf);
        }
        for (int i = 0; i < cfg->num_hidden_layers; i++) {
            snprintf(path, sizeof(path), "%s/experts/layer_%02d.bin", config.model_dir, i);
            int efd = open(path, O_RDONLY);
            if (efd < 0) continue;
            ctx->expert_layers[i].fd = efd;
            ctx->expert_layers[i].n_experts = cfg->num_experts;
            struct stat est;
            if (fstat(efd, &est) == 0)
                ctx->expert_layers[i].expert_size = est.st_size / cfg->num_experts;
            if (em_json) {
                char key[64]; snprintf(key, sizeof(key), "\"%d\"", i);
                const char *lp = strstr(em_json, key);
                if (lp) {
                    ctx->expert_layers[i].expert_size = json_get_int(lp, "expert_size", (int)ctx->expert_layers[i].expert_size);
                    ctx->expert_layers[i].gate_size = json_get_int(lp, "gate_size", 0);
                    ctx->expert_layers[i].up_size = json_get_int(lp, "up_size", 0);
                    ctx->expert_layers[i].down_size = json_get_int(lp, "down_size", 0);
                }
            }
            ctx->n_expert_layers++;

            // mmap the expert file for fast access via page cache
            struct stat mst;
            if (fstat(efd, &mst) == 0 && mst.st_size > 0) {
                ctx->expert_layers[i].mmap_size = mst.st_size;
                ctx->expert_layers[i].mmap_data = mmap(NULL, mst.st_size, PROT_READ,
                                                        MAP_SHARED, efd, 0);
                if (ctx->expert_layers[i].mmap_data == MAP_FAILED) {
                    ctx->expert_layers[i].mmap_data = NULL;
                    ctx->expert_layers[i].mmap_size = 0;
                }
            }
        }
        free(em_json);

        // Pre-warm OS page cache: hint kernel to read expert files in background
        size_t total_mmap = 0;
        for (int i = 0; i < cfg->num_hidden_layers; i++) {
            if (ctx->expert_layers[i].mmap_data) {
                madvise(ctx->expert_layers[i].mmap_data, ctx->expert_layers[i].mmap_size, MADV_WILLNEED);
                total_mmap += ctx->expert_layers[i].mmap_size;
            }
        }
        LOG_INFO("Expert files: %d/%d mmap'd (%.1f GB, pre-warming page cache)",
                ctx->n_expert_layers, cfg->num_hidden_layers, (double)total_mmap / (1024*1024*1024));
    }

    // Warn if expert layers don't match expected count
    if (ctx->n_expert_layers > 0 && ctx->n_expert_layers < cfg->num_hidden_layers) {
        LOG_WARN("Only %d/%d expert layer files found — some layers will have no MoE",
                 ctx->n_expert_layers, cfg->num_hidden_layers);
    }

    // Validate expert file sizes match manifest
    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        ExpertLayerFile *elf = &ctx->expert_layers[i];
        if (elf->fd <= 0 && !elf->mmap_data) continue;
        if (elf->expert_size == 0) {
            LOG_WARN("Layer %d expert_size is 0, skipping validation", i);
            continue;
        }
        size_t expected = (size_t)elf->n_experts * elf->expert_size;
        if (elf->mmap_size > 0 && elf->mmap_size < expected) {
            LOG_ERROR("Layer %d expert file too small: %zu bytes, expected %zu (%d experts x %zu)",
                      i, elf->mmap_size, expected, elf->n_experts, elf->expert_size);
            mnemo_cuda_unload(ctx);
            return MNEMO_ERR_BAD_CONFIG;
        }
    }

    // Assign layer ranges to GPUs (pipeline parallelism)
    int layers_per_gpu = cfg->num_hidden_layers / ctx->n_gpus;
    int extra_layers = cfg->num_hidden_layers % ctx->n_gpus;
    int layer_offset = 0;

    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        gpu->layer_start = layer_offset;
        gpu->layer_end = layer_offset + layers_per_gpu + (g < extra_layers ? 1 : 0);
        layer_offset = gpu->layer_end;

        LOG_INFO("GPU %d: layers %d-%d (%d layers)",
                gpu->gpu_id, gpu->layer_start, gpu->layer_end - 1,
                gpu->layer_end - gpu->layer_start);
    }

    // Load resident weights: mmap file on host
    snprintf(path, sizeof(path), "%s/resident_weights.bin", config.model_dir);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { LOG_ERROR("Cannot open %s", path); mnemo_cuda_unload(ctx); return -2; }

    struct stat st;
    fstat(fd, &st);
    ctx->resident_size = st.st_size;

    ctx->h_resident_mmap = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (ctx->h_resident_mmap == MAP_FAILED) { close(fd); mnemo_cuda_unload(ctx); return -2; }
    close(fd);

    // Cross-validate: resident manifest offsets must fit within the file
    {
        int bad_tensors = 0;
        for (int i = 0; i < ctx->tensor_table.n_entries; i++) {
            TensorEntry *e = &ctx->tensor_table.entries[i];
            if (e->size > ctx->resident_size ||
                e->offset > ctx->resident_size - e->size) {
                LOG_ERROR("Tensor '%s' offset+size (%zu+%zu) exceeds resident file (%zu)",
                          e->name, e->offset, e->size, ctx->resident_size);
                bad_tensors++;
            }
        }
        if (bad_tensors > 0) {
            LOG_ERROR("%d tensor(s) have invalid offsets", bad_tensors);
            mnemo_cuda_unload(ctx);
            return MNEMO_ERR_BAD_CONFIG;
        }
    }

    // Sanitize NaN/Inf in F32 resident tensors (some GGUFs have corrupt weights)
    int nan_fixed = 0;
    for (int i = 0; i < ctx->tensor_table.n_entries; i++) {
        TensorEntry *e = &ctx->tensor_table.entries[i];
        if (e->type_id == 0) { // F32
            float *fdata = (float *)((char *)ctx->h_resident_mmap + e->offset);
            int n_floats = (int)(e->size / 4);
            for (int j = 0; j < n_floats; j++) {
                if (fdata[j] != fdata[j] || fdata[j] > 65504.0f || fdata[j] < -65504.0f) {
                    fdata[j] = 0.0f;
                    nan_fixed++;
                }
            }
        }
    }
    if (nan_fixed > 0)
        LOG_WARN("Sanitized %d NaN/Inf values in resident F32 tensors", nan_fixed);

    // ── Partitioned resident weight loading ──
    // Each GPU only gets the tensors it needs:
    //   - Global tensors (token_embd, output_norm, output.weight): GPU 0 + last GPU
    //   - Per-layer tensors (blk.N.*): only the GPU owning layer N
    // This saves VRAM proportional to (n_gpus - 1) * per_layer_weight_size.
    int n_tensors = ctx->tensor_table.n_entries;
    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        cudaSetDevice(gpu->gpu_id);

        CUDA_LOAD_CHECK(cudaStreamCreate(&gpu->stream_compute));
        CUDA_LOAD_CHECK(cudaStreamCreate(&gpu->stream_io));

        // Classify which tensors this GPU needs
        gpu->tensor_offsets = (size_t *)malloc(n_tensors * sizeof(size_t));
        gpu->n_tensor_offsets = n_tensors;
        for (int i = 0; i < n_tensors; i++)
            gpu->tensor_offsets[i] = (size_t)-1;  // not present by default

        size_t gpu_resident_size = 0;
        for (int i = 0; i < n_tensors; i++) {
            TensorEntry *e = &ctx->tensor_table.entries[i];
            bool need = false;

            // Parse layer number from tensor name (blk.N.*)
            int tensor_layer = -1;
            if (strncmp(e->name, "blk.", 4) == 0)
                tensor_layer = atoi(e->name + 4);

            if (tensor_layer >= 0) {
                // Per-layer tensor: only on the owning GPU
                need = (tensor_layer >= gpu->layer_start && tensor_layer < gpu->layer_end);
            } else {
                // Global tensor (not blk.N.*):
                //   GPU 0: gets all global tensors (token_embd, etc.)
                //   Last GPU: gets output_norm.weight + output.weight for lm_head
                //   Single-GPU: GPU 0 == last GPU, gets all
                if (g == 0) {
                    need = true;
                } else if (g == ctx->n_gpus - 1) {
                    need = (strcmp(e->name, "output_norm.weight") == 0 ||
                            strcmp(e->name, "output.weight") == 0);
                }
            }

            if (need) {
                // Align to 256 bytes for CUDA efficiency
                size_t aligned = (gpu_resident_size + 255) & ~(size_t)255;
                gpu->tensor_offsets[i] = aligned;
                gpu_resident_size = aligned + e->size;
            }
        }

        // Allocate and copy only needed tensors
        if (gpu_resident_size == 0) gpu_resident_size = 256;  // avoid zero alloc
        CUDA_LOAD_CHECK(cudaMalloc(&gpu->d_resident, gpu_resident_size));
        gpu->resident_size = gpu_resident_size;

        for (int i = 0; i < n_tensors; i++) {
            if (gpu->tensor_offsets[i] == (size_t)-1) continue;
            TensorEntry *e = &ctx->tensor_table.entries[i];
            const void *src = (const char *)ctx->h_resident_mmap + e->offset;
            void *dst = (char *)gpu->d_resident + gpu->tensor_offsets[i];
            cudaMemcpy(dst, src, e->size, cudaMemcpyHostToDevice);
        }

        LOG_INFO("GPU %d: %.1f MB resident → VRAM (partitioned from %.1f MB total)",
                gpu->gpu_id, (double)gpu_resident_size / (1024*1024),
                (double)st.st_size / (1024*1024));
    }

    // Set tensor data pointers (GPU 0 for backward compat with any direct access)
    for (int i = 0; i < n_tensors; i++) {
        TensorEntry *e = &ctx->tensor_table.entries[i];
        if (ctx->gpus[0].tensor_offsets[i] != (size_t)-1)
            e->data = (char *)ctx->gpus[0].d_resident + ctx->gpus[0].tensor_offsets[i];
        else
            e->data = NULL;
    }

    // Allocate pinned host buffer for inter-GPU hidden state transfer
    CUDA_LOAD_CHECK(cudaMallocHost((void**)&ctx->h_hidden_transfer, cfg->hidden_size * sizeof(float)));

    // Allocate per-GPU compute buffers
    int H = cfg->hidden_size;
    int NH = cfg->num_attention_heads;
    int NKV = cfg->num_key_value_heads;
    int HD = cfg->head_dim;
    int EFF = cfg->moe_intermediate_size;
    int V = cfg->vocab_size;
    int K = cfg->num_experts_per_tok;

    // Determine max expert size across ALL layers (buffers must fit the largest)
    size_t max_expert_sz = 0;
    for (int i = 0; i < cfg->num_hidden_layers; i++)
        if (ctx->expert_layers[i].expert_size > max_expert_sz)
            max_expert_sz = ctx->expert_layers[i].expert_size;
    if (max_expert_sz == 0)
        max_expert_sz = 16UL * 1024 * 1024;  // fallback default
    LOG_INFO("Max expert size: %.1f MB", (double)max_expert_sz / (1024*1024));

    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        cudaSetDevice(gpu->gpu_id);

        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_hidden,   H * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_residual, H * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_q,        NH * HD * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_k,        NKV * HD * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_v,        NKV * HD * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_attn_out, NH * HD * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_normed,   H * sizeof(float)));
        // GEMM Q8_1 buffer: (H/128) blocks × 144 bytes × MAX_BATCH
        CUDA_LOAD_CHECK(cudaMalloc(&gpu->d_gemm_q8, (size_t)(H / 128) * 144 * MAX_BATCH));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_router_logits, cfg->num_experts * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_router_out, cfg->num_experts * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_expert_indices, K * sizeof(int)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_expert_weights, K * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMallocHost((void**)&gpu->h_expert_indices, 16 * sizeof(int)));
        CUDA_LOAD_CHECK(cudaMallocHost((void**)&gpu->h_expert_weights, 16 * sizeof(float)));
        cudaEventCreateWithFlags(&gpu->ev_router, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&gpu->ev_upload, cudaEventDisableTiming);
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_expert_gate, EFF * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_expert_up,   EFF * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_expert_act,  EFF * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_expert_out,  H * sizeof(float)));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_moe_out,     (size_t)H * K * sizeof(float)));

        // Logits only on last GPU
        if (g == ctx->n_gpus - 1) {
            CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_logits, V * sizeof(float)));
            CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_sampled_token, sizeof(int)));
            CUDA_LOAD_CHECK(cudaMallocHost((void**)&gpu->h_sampled_token, sizeof(int)));
        }

        // Expert I/O buffer (pinned host, K experts for parallel pread)
        gpu->expert_buf_size = (size_t)max_expert_sz * (size_t)K;
        gpu->expert_buf_pinned = config.use_pinned_memory;
        if (config.use_pinned_memory) {
            CUDA_LOAD_CHECK(cudaMallocHost(&gpu->h_expert_buf, gpu->expert_buf_size));
        } else {
            gpu->h_expert_buf = malloc(gpu->expert_buf_size);
            if (!gpu->h_expert_buf) { LOG_ERROR("OOM: expert buf"); mnemo_cuda_unload(ctx); return -5; }
        }
        CUDA_LOAD_CHECK(cudaMalloc(&gpu->d_expert_buf, gpu->expert_buf_size));

        // KV cache — FP16 (half the VRAM of FP32)
        int n_layers = gpu->layer_end - gpu->layer_start;
        size_t kv_bytes_per_elem = config.kv_int8 ? sizeof(int8_t) : sizeof(uint16_t);
        size_t kv_per_tok = (size_t)NKV * HD * kv_bytes_per_elem;
        size_t kv_size = (size_t)n_layers * config.context_length * kv_per_tok;
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_kv_k, kv_size));
        CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_kv_v, kv_size));
        CUDA_LOAD_CHECK(cudaMemset(gpu->d_kv_k, 0, kv_size));
        CUDA_LOAD_CHECK(cudaMemset(gpu->d_kv_v, 0, kv_size));
        if (config.kv_int8) {
            size_t scales_size = (size_t)n_layers * config.context_length * NKV * sizeof(float);
            CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_kv_k_scales, scales_size));
            CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_kv_v_scales, scales_size));
            CUDA_LOAD_CHECK(cudaMemset(gpu->d_kv_k_scales, 0, scales_size));
            CUDA_LOAD_CHECK(cudaMemset(gpu->d_kv_v_scales, 0, scales_size));
        }

        LOG_INFO("GPU %d: KV cache %.1f MB FP16 (%d ctx), expert buf %.1f MB",
                gpu->gpu_id, (double)kv_size*2 / (1024*1024), config.context_length,
                (double)gpu->expert_buf_size / (1024*1024));

        // Expert VRAM cache — fill remaining VRAM with expert slots
        gpu->expert_slot_size = max_expert_sz;
        gpu->expert_cache_slots = 0;
        gpu->d_expert_cache = NULL;
        gpu->cache_layer = NULL;
        gpu->cache_expert = NULL;
        gpu->cache_lru = NULL;
        gpu->cache_hits = NULL;
        gpu->cache_pinned = NULL;
        gpu->cache_clock = 0;
        gpu->cache_state = NULL;
        gpu->cache_pending_layer = NULL;
        gpu->cache_pending_expert = NULL;
        gpu->cache_ready_event = NULL;

        // Prefetch buffer — holds up to K experts for speculative next-layer reads
        gpu->prefetch_buf_size = max_expert_sz * K;
        gpu->prefetch_buf_pinned = config.use_pinned_memory;
        if (config.use_pinned_memory) {
            CUDA_LOAD_CHECK(cudaMallocHost(&gpu->h_prefetch_buf, gpu->prefetch_buf_size));
        } else {
            gpu->h_prefetch_buf = malloc(gpu->prefetch_buf_size);
            if (!gpu->h_prefetch_buf) { LOG_ERROR("OOM: prefetch buf"); mnemo_cuda_unload(ctx); return -5; }
        }
        gpu->prefetch_layer = -1;
        gpu->n_prefetched = 0;
        gpu->prefetch_ready = 0;

        // Gated Delta Net state buffers (hybrid models only)
        gpu->d_gdn_state = NULL;
        gpu->d_conv_state = NULL;
        gpu->d_gdn_qkv = NULL;
        gpu->d_gdn_alpha = NULL;
        gpu->d_gdn_beta = NULL;
        gpu->d_gdn_z = NULL;
        gpu->d_gdn_out = NULL;
        gpu->n_gdn_layers = 0;
        gpu->gdn_layer_map = NULL;

        if (cfg->full_attention_interval > 0 && cfg->ssm_inner_size > 0) {
            int V_DIM = cfg->ssm_inner_size;       // value_dim (8192)
            int K_HD = cfg->ssm_state_size;         // key_head_dim (128)
            int V_HD = K_HD;                        // value_head_dim = key_head_dim
            int N_VH = cfg->ssm_dt_rank;            // num_value_heads (64)
            int N_KH = cfg->ssm_group_count;        // num_key_heads (16)
            int K_DIM = K_HD * N_KH;                // key_dim (2048)
            int CONV_DIM = K_DIM + K_DIM + V_DIM;   // conv1d dimension (12288)
            int CK = cfg->ssm_conv_kernel;

            // Count GDN layers on this GPU
            int n_gdn = 0;
            for (int l = gpu->layer_start; l < gpu->layer_end; l++)
                if (l % cfg->full_attention_interval != 0) n_gdn++;

            if (n_gdn > 0) {
                gpu->n_gdn_layers = n_gdn;
                gpu->gdn_layer_map = (int *)malloc(n_gdn * sizeof(int));
                int idx = 0;
                for (int l = gpu->layer_start; l < gpu->layer_end; l++)
                    if (l % cfg->full_attention_interval != 0)
                        gpu->gdn_layer_map[idx++] = l;

                // Persistent: state[n_gdn][N_VH][K_HD][V_HD] + conv[n_gdn][CONV_DIM][CK-1]
                size_t state_sz = (size_t)n_gdn * N_VH * K_HD * V_HD * sizeof(float);
                size_t conv_sz = (size_t)n_gdn * CONV_DIM * (CK - 1) * sizeof(float);
                CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_gdn_state, state_sz));
                CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_conv_state, conv_sz));
                cudaMemset(gpu->d_gdn_state, 0, state_sz);
                cudaMemset(gpu->d_conv_state, 0, conv_sz);

                // Temp buffers
                CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_gdn_qkv, CONV_DIM * sizeof(float)));
                CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_gdn_alpha, N_VH * sizeof(float)));
                CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_gdn_beta, N_VH * sizeof(float)));
                CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_gdn_z, V_DIM * sizeof(float)));
                CUDA_LOAD_CHECK(cudaMalloc((void**)&gpu->d_gdn_out, V_DIM * sizeof(float)));

                LOG_INFO("GPU %d: %d GDN layers, state %.1f MB, conv %.1f MB",
                         gpu->gpu_id, n_gdn,
                         (double)state_sz / (1024*1024),
                         (double)conv_sz / (1024*1024));
            }
        }

        size_t vram_free = 0, vram_total = 0;
        cudaMemGetInfo(&vram_free, &vram_total);
        size_t headroom = 256UL * 1024 * 1024; // 256 MB safety margin

        if (vram_free > headroom + max_expert_sz * 4) {
            size_t cache_budget;
            if (config.cache_percent > 0 && config.cache_percent <= 90) {
                // Use explicit percentage of total VRAM
                cache_budget = (size_t)vram_total * config.cache_percent / 100;
                if (cache_budget > vram_free - headroom)
                    cache_budget = vram_free - headroom;
            } else {
                cache_budget = vram_free - headroom;
            }
            gpu->expert_cache_slots = (int)(cache_budget / max_expert_sz);
            gpu->expert_cache_size = (size_t)gpu->expert_cache_slots * max_expert_sz;

            cudaError_t err = cudaMalloc(&gpu->d_expert_cache, gpu->expert_cache_size);
            if (err != cudaSuccess) {
                // Allocation failed — try half the size
                gpu->expert_cache_slots /= 2;
                gpu->expert_cache_size = (size_t)gpu->expert_cache_slots * max_expert_sz;
                err = cudaMalloc(&gpu->d_expert_cache, gpu->expert_cache_size);
                if (err != cudaSuccess) {
                    LOG_WARN("GPU %d: VRAM cache alloc failed, running without cache", gpu->gpu_id);
                    gpu->d_expert_cache = NULL;
                    gpu->expert_cache_slots = 0;
                    gpu->expert_cache_size = 0;
                }
            }

            if (gpu->d_expert_cache) {
                int ns = gpu->expert_cache_slots;
                gpu->cache_layer = (int *)calloc(ns, sizeof(int));
                gpu->cache_expert = (int *)calloc(ns, sizeof(int));
                gpu->cache_lru = (uint64_t *)calloc(ns, sizeof(uint64_t));
                gpu->cache_hits = (uint32_t *)calloc(ns, sizeof(uint32_t));
                gpu->cache_pinned = (bool *)calloc(ns, sizeof(bool));
                gpu->cache_state = (ExpertSlotState *)calloc(ns, sizeof(ExpertSlotState));
                gpu->cache_pending_layer = (int *)calloc(ns, sizeof(int));
                gpu->cache_pending_expert = (int *)calloc(ns, sizeof(int));
                gpu->cache_ready_event = (cudaEvent_t *)calloc(ns, sizeof(cudaEvent_t));
                for (int s = 0; s < ns; s++) {
                    gpu->cache_layer[s] = -1;
                    gpu->cache_expert[s] = -1;
                    gpu->cache_state[s] = SLOT_EMPTY;
                    gpu->cache_pending_layer[s] = -1;
                    gpu->cache_pending_expert[s] = -1;
                    cudaEventCreateWithFlags(&gpu->cache_ready_event[s], cudaEventDisableTiming);
                }
                LOG_INFO("GPU %d: VRAM cache %.1f MB (%d slots × %.1f MB)",
                        gpu->gpu_id, (double)gpu->expert_cache_size / (1024*1024),
                        gpu->expert_cache_slots, (double)max_expert_sz / (1024*1024));

                // Device-side cache state for GPU classification kernel
                cudaMalloc((void**)&gpu->d_cache_layer, ns * sizeof(int));
                cudaMalloc((void**)&gpu->d_cache_expert, ns * sizeof(int));
                cudaMalloc((void**)&gpu->d_class_hit_slots, 16 * sizeof(int));
                cudaMalloc((void**)&gpu->d_class_miss_mask, 16 * sizeof(int));
                cudaMallocHost((void**)&gpu->h_class_miss_mask, 16 * sizeof(int));
                cudaMalloc((void**)&gpu->d_n_hits, sizeof(int));
                cudaMallocHost((void**)&gpu->h_n_hits, sizeof(int));
                // Initialize device cache state to -1 (empty)
                cudaMemset(gpu->d_cache_layer, 0xFF, ns * sizeof(int));
                cudaMemset(gpu->d_cache_expert, 0xFF, ns * sizeof(int));
            }
        }
    }

    ctx->kv_pos = 0;
    ctx->prof_token_count = 0;

    // RAM-tier cache (disabled by default — OS page cache handles this via mmap)
    if (RAM_CACHE_DEFAULT_GB > 0)
        ram_cache_init(&ctx->ram_cache, max_expert_sz, RAM_CACHE_DEFAULT_GB);
    else
        memset(&ctx->ram_cache, 0, sizeof(ctx->ram_cache));

    // Expert heat profiling — allocate and zero (or load from disk)
    size_t heat_size = (size_t)cfg->num_hidden_layers * cfg->num_experts;
    ctx->heat_map = (uint32_t *)calloc(heat_size, sizeof(uint32_t));
    ctx->heat_total_tokens = 0;
    ctx->heat_pinning_active = false;
    ctx->layer_hits = (uint32_t *)calloc(cfg->num_hidden_layers, sizeof(uint32_t));
    ctx->layer_misses = (uint32_t *)calloc(cfg->num_hidden_layers, sizeof(uint32_t));
    if (!ctx->heat_map || !ctx->layer_hits || !ctx->layer_misses) {
        LOG_ERROR("OOM: heat profiling buffers");
        mnemo_cuda_unload(ctx);
        return MNEMO_ERR_CUDA;
    }
    ctx->n_last_activated = 0;

    // Try to load saved heat map from previous sessions
    {
        char heat_path[1200];
        snprintf(heat_path, sizeof(heat_path), "%s/expert_heat.bin", config.model_dir);
        FILE *hf = fopen(heat_path, "rb");
        if (hf) {
            uint32_t magic, n_layers, n_experts;
            uint64_t saved_tokens;
            if (fread(&magic, 4, 1, hf) == 1 && magic == 0x48454154 &&
                fread(&n_layers, 4, 1, hf) == 1 && (int)n_layers == cfg->num_hidden_layers &&
                fread(&n_experts, 4, 1, hf) == 1 && (int)n_experts == cfg->num_experts &&
                fread(&saved_tokens, 8, 1, hf) == 1) {
                if (fread(ctx->heat_map, sizeof(uint32_t), heat_size, hf) != heat_size)
                    LOG_WARN("Short read on heat map data");
                ctx->heat_total_tokens = saved_tokens;
                LOG_INFO("Loaded heat map (%lu tokens of history)",
                        (unsigned long)saved_tokens);
            }
            fclose(hf);
        }
    }

    // Model info
    snprintf(ctx->info, sizeof(ctx->info),
             "MnemoCUDA: %dB MoE (K=%d), %d GPUs, %zu MB resident/GPU",
             cfg->hidden_size > 3000 ? (cfg->num_hidden_layers > 60 ? 235 : 30) : 5,
             cfg->num_experts_per_tok,
             ctx->n_gpus,
             ctx->resident_size / (1024*1024));

    // Init kernel attributes (100KB shared memory for bandwidth-bound matvec)
    {
        extern void cuda_kernels_init(void);
        cuda_kernels_init();
    }

    io_pool_init(config.io_threads);
    ctx->extra_prefetch = config.extra_prefetch;
    // Auto-enable deep prefetch for large MoE models with sufficient I/O threads
    if (!ctx->extra_prefetch && cfg->num_experts >= 32 && config.io_threads >= 4)
        ctx->extra_prefetch = 1;
    cfg->kv_int8 = config.kv_int8;
    ctx->profiling_enabled = (getenv("MNEMO_PROFILE") != NULL);

    // Pre-cache per-layer tensor pointers (avoids snprintf+hash per layer per token)
    {
        int NL = cfg->num_hidden_layers;
        ctx->layer_tensors = calloc(NL, sizeof(*ctx->layer_tensors));
        char tn[128];
        for (int l = 0; l < NL; l++) {
            #define LT(field, name) do { snprintf(tn, sizeof(tn), "blk.%d." name, l); \
                ctx->layer_tensors[l].field = tensor_find(&ctx->tensor_table, tn); } while(0)
            LT(attn_norm, "attn_norm.weight");
            LT(attn_q, "attn_q.weight");
            LT(attn_k, "attn_k.weight");
            LT(attn_v, "attn_v.weight");
            LT(attn_q_norm, "attn_q_norm.weight");
            LT(attn_k_norm, "attn_k_norm.weight");
            LT(attn_output, "attn_output.weight");
            LT(ffn_norm, "ffn_norm.weight");
            LT(ffn_gate_inp, "ffn_gate_inp.weight");
            #undef LT
        }
    }

    // Probe and enable P2P GPU access for direct GPU-to-GPU transfers
    if (ctx->n_gpus > 1) {
        for (int i = 0; i < ctx->n_gpus; i++) {
            for (int j = 0; j < ctx->n_gpus; j++) {
                if (i == j) continue;
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, ctx->gpus[i].gpu_id, ctx->gpus[j].gpu_id);
                if (can_access) {
                    cudaSetDevice(ctx->gpus[i].gpu_id);
                    cudaError_t err = cudaDeviceEnablePeerAccess(ctx->gpus[j].gpu_id, 0);
                    if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                        ctx->p2p_enabled[i][j] = true;
                        if (err == cudaErrorPeerAccessAlreadyEnabled)
                            cudaGetLastError();  // clear the error
                    }
                }
            }
        }
        int p2p_count = 0;
        for (int i = 0; i < ctx->n_gpus; i++)
            for (int j = 0; j < ctx->n_gpus; j++)
                if (ctx->p2p_enabled[i][j]) p2p_count++;
        if (p2p_count > 0)
            LOG_INFO("P2P enabled for %d GPU pairs", p2p_count);
    }

    // Build CUDA graphs for attention path (done in forward.c where kernels are visible)
    {
        extern void build_attention_graphs(MnemoCudaCtx *ctx);
        if (!cfg->kv_int8 && cfg->full_attention_interval == 0)
            build_attention_graphs(ctx);
    }

    // Allocate extra batch slots for continuous batching (1 extra slot = batch=2)
    {
        int NKV_l = cfg->num_key_value_heads, HD_l = cfg->head_dim;
        int NH_l = cfg->num_attention_heads, V_l = cfg->vocab_size;
        for (int g = 0; g < ctx->n_gpus; g++) {
            GPUState *gpu = &ctx->gpus[g];
            cudaSetDevice(gpu->gpu_id);
            int n_layers = gpu->layer_end - gpu->layer_start;
            size_t kv_elem_sz = cfg->kv_int8 ? 1 : 2;
            size_t kv_size = (size_t)n_layers * config.context_length * NKV_l * HD_l * kv_elem_sz;

            BatchSlot *s = &gpu->extra_slots[0];
            memset(s, 0, sizeof(*s));
            cudaError_t e = cudaSuccess;
            e |= cudaMalloc((void**)&s->d_hidden,  H * sizeof(float));
            e |= cudaMalloc((void**)&s->d_residual, H * sizeof(float));
            e |= cudaMalloc((void**)&s->d_normed,   H * sizeof(float));
            e |= cudaMalloc((void**)&s->d_q,        NH_l * HD_l * sizeof(float));
            e |= cudaMalloc((void**)&s->d_k,        NKV_l * HD_l * sizeof(float));
            e |= cudaMalloc((void**)&s->d_v,        NKV_l * HD_l * sizeof(float));
            e |= cudaMalloc((void**)&s->d_attn_out,  NH_l * HD_l * sizeof(float));
            e |= cudaMalloc((void**)&s->d_moe_out,   (size_t)H * K * sizeof(float));
            if (g == ctx->n_gpus - 1)
                e |= cudaMalloc((void**)&s->d_logits, V_l * sizeof(float));
            e |= cudaMalloc((void**)&s->d_kv_k, kv_size);
            e |= cudaMalloc((void**)&s->d_kv_v, kv_size);
            if (e == cudaSuccess) {
                cudaMemset(s->d_kv_k, 0, kv_size);
                cudaMemset(s->d_kv_v, 0, kv_size);
                s->allocated = true;
                gpu->n_extra_slots = 1;
                LOG_INFO("GPU %d: batch slot 1 allocated (KV %.1f MB)",
                         gpu->gpu_id, (double)kv_size * 2 / (1024*1024));
            } else {
                cudaFree(s->d_hidden); cudaFree(s->d_residual); cudaFree(s->d_normed);
                cudaFree(s->d_q); cudaFree(s->d_k); cudaFree(s->d_v);
                cudaFree(s->d_attn_out); cudaFree(s->d_moe_out); cudaFree(s->d_logits);
                cudaFree(s->d_kv_k); cudaFree(s->d_kv_v);
                memset(s, 0, sizeof(*s));
                cudaGetLastError();
                LOG_WARN("GPU %d: batch slot 1 OOM — single-request only", gpu->gpu_id);
            }
        }
    }

    ctx->loaded = true;
    LOG_INFO("Ready: %s", ctx->info);
    return 0;
}

void mnemo_cuda_unload(MnemoCudaCtx *ctx) {
    if (!ctx) return;

    // Only do runtime operations if fully loaded
    if (ctx->loaded) {
        mnemo_cuda_heat_save(ctx);
        io_pool_shutdown();
    }

    // Free all resources regardless of loaded state (handles partial init cleanup)

    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        cudaSetDevice(gpu->gpu_id);

        cudaFree(gpu->d_resident);
        cudaFree(gpu->d_hidden); cudaFree(gpu->d_residual);
        cudaFree(gpu->d_q); cudaFree(gpu->d_k); cudaFree(gpu->d_v);
        cudaFree(gpu->d_attn_out); cudaFree(gpu->d_normed);
        cudaFree(gpu->d_router_logits); cudaFree(gpu->d_router_out);
        cudaFree(gpu->d_gemm_q8);
        cudaFree(gpu->d_expert_indices); cudaFree(gpu->d_expert_weights);
        if (gpu->h_expert_indices) cudaFreeHost(gpu->h_expert_indices);
        if (gpu->h_expert_weights) cudaFreeHost(gpu->h_expert_weights);
        cudaEventDestroy(gpu->ev_router);
        cudaEventDestroy(gpu->ev_upload);
        cudaFree(gpu->d_expert_gate); cudaFree(gpu->d_expert_up);
        cudaFree(gpu->d_expert_act); cudaFree(gpu->d_expert_out);
        cudaFree(gpu->d_moe_out); cudaFree(gpu->d_logits);
        cudaFree(gpu->d_sampled_token);
        if (gpu->h_sampled_token) cudaFreeHost(gpu->h_sampled_token);
        cudaFree(gpu->d_kv_k); cudaFree(gpu->d_kv_v);
        cudaFree(gpu->d_kv_k_scales); cudaFree(gpu->d_kv_v_scales);
        cudaFree(gpu->d_expert_buf);
        cudaFree(gpu->d_expert_cache);
        cudaFree(gpu->d_cache_layer);
        cudaFree(gpu->d_cache_expert);
        cudaFree(gpu->d_class_hit_slots);
        cudaFree(gpu->d_class_miss_mask);
        if (gpu->h_class_miss_mask) cudaFreeHost(gpu->h_class_miss_mask);
        cudaFree(gpu->d_n_hits);
        if (gpu->h_n_hits) cudaFreeHost(gpu->h_n_hits);

        // Free CUDA graphs
        if (gpu->attn_graph_exec) {
            int nl = gpu->layer_end - gpu->layer_start;
            for (int li = 0; li < nl; li++)
                if (gpu->attn_graph_exec[li]) cudaGraphExecDestroy(gpu->attn_graph_exec[li]);
            free(gpu->attn_graph_exec);
            free(gpu->attn_rope_node);
            free(gpu->attn_kvst_node);
            free(gpu->attn_attn_node);
            free(gpu->attn_pos_buf);
            free(gpu->attn_seqlen_buf);
            free(gpu->attn_kvk_dst_buf);
            free(gpu->attn_kvv_dst_buf);
        }

        if (gpu->h_expert_buf) {
            if (gpu->expert_buf_pinned) cudaFreeHost(gpu->h_expert_buf);
            else free(gpu->h_expert_buf);
        }
        if (gpu->h_prefetch_buf) {
            if (gpu->prefetch_buf_pinned) cudaFreeHost(gpu->h_prefetch_buf);
            else free(gpu->h_prefetch_buf);
        }
        if (gpu->cache_ready_event) {
            for (int s = 0; s < gpu->expert_cache_slots; s++)
                cudaEventDestroy(gpu->cache_ready_event[s]);
        }
        free(gpu->cache_layer);
        free(gpu->cache_expert);
        free(gpu->cache_lru);
        free(gpu->cache_hits);
        free(gpu->cache_pinned);
        free(gpu->cache_state);
        free(gpu->cache_pending_layer);
        free(gpu->cache_pending_expert);
        free(gpu->cache_ready_event);
        free(gpu->tensor_offsets);

        // GDN buffers
        cudaFree(gpu->d_gdn_state);
        cudaFree(gpu->d_conv_state);
        cudaFree(gpu->d_gdn_qkv);
        cudaFree(gpu->d_gdn_alpha);
        cudaFree(gpu->d_gdn_beta);
        cudaFree(gpu->d_gdn_z);
        cudaFree(gpu->d_gdn_out);
        free(gpu->gdn_layer_map);

        cudaStreamDestroy(gpu->stream_compute);
        cudaStreamDestroy(gpu->stream_io);

        // Zero the entire GPUState so a second unload() is safe
        memset(gpu, 0, sizeof(GPUState));
    }
    ctx->n_gpus = 0;

    if (ctx->h_resident_mmap) {
        munmap(ctx->h_resident_mmap, ctx->resident_size);
        ctx->h_resident_mmap = NULL;
        ctx->resident_size = 0;
    }
    if (ctx->h_hidden_transfer) {
        cudaFreeHost(ctx->h_hidden_transfer);
        ctx->h_hidden_transfer = NULL;
    }

    if (ctx->expert_layers) {
        for (int i = 0; i < ctx->config.num_hidden_layers; i++) {
            if (ctx->expert_layers[i].mmap_data)
                munmap(ctx->expert_layers[i].mmap_data, ctx->expert_layers[i].mmap_size);
            if (ctx->expert_layers[i].fd > 0)
                close(ctx->expert_layers[i].fd);
        }
        free(ctx->expert_layers);
        ctx->expert_layers = NULL;
    }

    free(ctx->tensor_table.entries);
    memset(&ctx->tensor_table, 0, sizeof(TensorTable));

    ram_cache_free(&ctx->ram_cache);

    free(ctx->heat_map);
    ctx->heat_map = NULL;
    free(ctx->layer_tensors);
    ctx->layer_tensors = NULL;
    free(ctx->layer_hits);
    ctx->layer_hits = NULL;
    free(ctx->layer_misses);
    ctx->layer_misses = NULL;

    tokenizer_free(ctx->tokenizer);
    ctx->tokenizer = NULL;

    ctx->loaded = false;
}

void mnemo_cuda_destroy(MnemoCudaCtx *ctx) {
    if (!ctx) return;
    mnemo_cuda_unload(ctx);
    free(ctx);
}

void mnemo_cuda_cancel(MnemoCudaCtx *ctx) {
    if (ctx) ctx->cancelled = true;
}


// (forward pass, expert cache, prefetch moved to forward.c/h)
#include "forward.h"

// ── Generate: tokenize → prefill → autoregressive decode ──

void mnemo_cuda_batch_bench(MnemoCudaCtx *ctx, int n_steps) {
    if (!ctx || !ctx->loaded) return;
    ModelConfig *cfg = &ctx->config;

    // Use a dummy token for decode steps
    int dummy_token = 1;  // typically a common token
    struct timespec t0, t1;

    // Batch=1: N decode steps
    LOG_INFO("Batch=1: %d decode steps...", n_steps);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int t = 0; t < n_steps; t++) {
        int out;
        int token_ids[1] = { dummy_token };
        int positions[1] = { t };
        forward_pass_batch(ctx, token_ids, positions, 1, 0.0f, 0.9f, 42 + t, &out);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double b1 = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    LOG_INFO("Batch=1: %d tokens in %.2fs = %.1f tok/s", n_steps, b1, n_steps / b1);

    // Check if batch=2 is available
    bool batch_ok = true;
    for (int g = 0; g < ctx->n_gpus; g++)
        if (!ctx->gpus[g].extra_slots[0].allocated) batch_ok = false;

    if (!batch_ok) {
        LOG_INFO("Batch=2 not available (slot 1 not allocated)");
        return;
    }

    // Batch=2: N decode steps × 2 requests
    LOG_INFO("Batch=2: %d decode steps × 2 requests...", n_steps);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int t = 0; t < n_steps; t++) {
        int out[2];
        int token_ids[2] = { dummy_token, dummy_token };
        int positions[2] = { t, t };
        forward_pass_batch(ctx, token_ids, positions, 2, 0.0f, 0.9f, 42 + t, out);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double b2 = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    LOG_INFO("Batch=2: %d tokens (2×%d) in %.2fs = %.1f tok/s total, %.1f per request",
             n_steps * 2, n_steps, b2, n_steps * 2 / b2, n_steps / b2);

    LOG_INFO("Batch throughput: %.1f tok/s (batch=1) vs %.1f tok/s (batch=2) = %.2fx",
             n_steps / b1, n_steps * 2 / b2, (n_steps * 2 / b2) / (n_steps / b1));
}

int mnemo_cuda_generate(MnemoCudaCtx *ctx, const char *prompt, int max_tokens,
                        float temperature, bool raw_prompt,
                        MnemoCudaTokenCB callback, void *userdata) {
    if (!ctx || !ctx->loaded) return MNEMO_ERR_BAD_CONFIG;
    if (!ctx->tokenizer) { callback("Error: tokenizer not loaded", true, userdata); return MNEMO_ERR_TOKENIZER; }

    ctx->cancelled = false;
    rng_seed((uint64_t)time(NULL));

    ModelConfig *cfg = &ctx->config;
    int V = cfg->vocab_size;

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // ── 1. Tokenize prompt ──
    // raw_prompt=true: pass prompt as-is (already ChatML-formatted)
    // raw_prompt=false: wrap in ChatML single-turn format
    // Also auto-detect if prompt starts with <|im_start|> for backward compat
    char *chatml;
    if (raw_prompt || strncmp(prompt, "<|im_start|>", 12) == 0) {
        chatml = strdup(prompt);
        if (!chatml) { LOG_ERROR("OOM: chatml strdup"); return MNEMO_ERR_CUDA; }
    } else {
        size_t chatml_size = strlen(prompt) + 128;
        chatml = malloc(chatml_size);
        if (!chatml) { LOG_ERROR("OOM: chatml alloc"); return MNEMO_ERR_CUDA; }
        snprintf(chatml, chatml_size, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", prompt);
    }

    // Try to find tokenize.py
    const char *tok_script = NULL;
    char tok_path[1200];
    // 1. Check in model_dir
    snprintf(tok_path, sizeof(tok_path), "%s/tokenize.py", ctx->model_dir);
    if (access(tok_path, F_OK) == 0) {
        tok_script = tok_path;
    } else {
        // 2. Check tools/ relative to executable (for installed binaries)
        {
            char exe_dir[1024] = {0};
            ssize_t len = readlink("/proc/self/exe", exe_dir, sizeof(exe_dir) - 1);
            if (len > 0) {
                exe_dir[len] = '\0';
                char *slash = strrchr(exe_dir, '/');
                if (slash) *slash = '\0';
                snprintf(tok_path, sizeof(tok_path), "%s/../tools/tokenize.py", exe_dir);
            }
        }
        if (access(tok_path, F_OK) == 0) {
            tok_script = tok_path;
        } else {
            // 3. Check tools/ relative to cwd (common when running from repo root)
            if (access("tools/tokenize.py", F_OK) == 0) {
                tok_script = "tools/tokenize.py";
            } else if (access("tokenize.py", F_OK) == 0) {
                // 4. Check cwd directly
                tok_script = "tokenize.py";
            }
        }
    }

    int *tokens = NULL;
    int n_tokens = 0;

    // Pipe chatml to tokenize.py via fork+exec (stdin→stdout binary protocol)
    if (tok_script) {
        int pipe_in[2], pipe_out[2];
        if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0) {
            LOG_WARN("pipe() failed: %s, skipping Python tokenizer", strerror(errno));
            goto fallback_tokenizer;
        }

        pid_t pid = fork();
        if (pid < 0) {
            LOG_WARN("fork() failed: %s, skipping Python tokenizer", strerror(errno));
            close(pipe_in[0]); close(pipe_in[1]);
            close(pipe_out[0]); close(pipe_out[1]);
            goto fallback_tokenizer;
        }
        if (pid == 0) {
            // Child: run tokenize.py
            close(pipe_in[1]);
            close(pipe_out[0]);
            dup2(pipe_in[0], STDIN_FILENO);
            dup2(pipe_out[1], STDOUT_FILENO);
            close(pipe_in[0]);
            close(pipe_out[1]);
            execlp("python3", "python3", tok_script, ctx->model_dir, "--pipe", NULL);
            _exit(1);
        }

        // Parent: write chatml, read tokens
        close(pipe_in[0]);
        close(pipe_out[1]);

        write_full(pipe_in[1], chatml, strlen(chatml));
        close(pipe_in[1]);

        // Read binary response: [4B n_tokens] [4B token_id]*n
        uint32_t n_tok;
        if (read_full(pipe_out[0], &n_tok, 4) == 4 && n_tok > 0 && n_tok < 1000000) {
            n_tokens = (int)n_tok;
            tokens = malloc(n_tokens * sizeof(int));
            if (!tokens) { n_tokens = 0; close(pipe_out[0]); waitpid(pid, NULL, 0); free(chatml); LOG_ERROR("OOM: tokens"); return MNEMO_ERR_CUDA; }
            ssize_t expected = (ssize_t)(n_tokens * sizeof(int));
            if (read_full(pipe_out[0], tokens, expected) != expected) {
                free(tokens);
                tokens = NULL;
                n_tokens = 0;
            }
        }
        close(pipe_out[0]);

        int status;
        waitpid(pid, &status, 0);
    }

    free(chatml);

fallback_tokenizer:
    if (!tokens || n_tokens == 0) {
        // Fallback to built-in BPE tokenizer
        LOG_WARN("Python tokenizer failed, using built-in BPE");
        if (raw_prompt || strncmp(prompt, "<|im_start|>", 12) == 0) {
            chatml = strdup(prompt);
            if (!chatml) { LOG_ERROR("OOM: chatml strdup"); return MNEMO_ERR_CUDA; }
        } else {
            size_t chatml_size = strlen(prompt) + 128;
            chatml = malloc(chatml_size);
            if (!chatml) { LOG_ERROR("OOM: chatml alloc"); return MNEMO_ERR_CUDA; }
            snprintf(chatml, chatml_size, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", prompt);
        }
        int max_prompt_tokens = cfg->max_position_embeddings - max_tokens;
        if (max_prompt_tokens < 32) max_prompt_tokens = 32;
        tokens = malloc(max_prompt_tokens * sizeof(int));
        if (!tokens) { free(chatml); LOG_ERROR("OOM: tokens alloc"); return MNEMO_ERR_CUDA; }
        n_tokens = tokenizer_encode(ctx->tokenizer, chatml, tokens, max_prompt_tokens);
        free(chatml);
    }

    // Cap prompt tokens to fit within context window (leave room for generation)
    int max_ctx = cfg->max_position_embeddings;
    int max_prompt = max_ctx - max_tokens;
    if (max_prompt < 1) max_prompt = 1;
    if (n_tokens > max_prompt) {
        LOG_WARN("Prompt too long (%d tokens), truncating to %d",
                n_tokens, max_prompt);
        n_tokens = max_prompt;
    }

    LOG_INFO("Prompt: %d tokens, generating up to %d", n_tokens, max_tokens);

    // ── 2. Prefill: process all prompt tokens ──
    // Batched prefill for all but the last token (no lm_head needed).
    // Last prefill token uses forward_pass_sample to get the first generated token.
    ctx->kv_pos = 0;
    if (n_tokens > 1) {
        if (ctx->cancelled) { free(tokens); return MNEMO_ERR_CANCELLED; }
        int rc = forward_prefill_batch(ctx, tokens, n_tokens - 1, 0);
        if (rc != 0) { free(tokens); return rc; }
        ctx->kv_pos = n_tokens - 1;
    }

    // Last prefill token: need to sample from it
    int first_token_id = -1;
    if (n_tokens > 0) {
        if (ctx->cancelled) { free(tokens); return MNEMO_ERR_CANCELLED; }
        int rc = forward_pass_sample(ctx, tokens[n_tokens - 1], ctx->kv_pos,
                                     temperature, 0.9f, rng_next(),
                                     &first_token_id);
        if (rc != 0) { free(tokens); return rc; }
        ctx->kv_pos++;
    }
    free(tokens);
    tokens = NULL;

    // ── 3. Autoregressive generation ──
    int tokens_generated = 0;
    int gen_error = 0;
    struct timespec t_gen_start;
    clock_gettime(CLOCK_MONOTONIC, &t_gen_start);

    int next_id = first_token_id;
    for (int t = 0; t < max_tokens; t++) {
        if (ctx->cancelled) break;
        if (ctx->kv_pos >= max_ctx) {
            LOG_INFO("Context full at %d tokens", ctx->kv_pos);
            break;
        }
        if (next_id < 0) break;

        // Check for EOS
        if (next_id == ctx->tokenizer->eos_id ||
            next_id == ctx->tokenizer->im_end_id) {
            LOG_INFO("EOS token %d at position %d", next_id, t);
            break;
        }

        // Decode and emit (with UTF-8 accumulation for multi-byte chars)
        {
            const char *text = tokenizer_decode(ctx->tokenizer, next_id);
            char utf8_buf[32];
            int utf8_len = 0;

            for (int ci = 0; text[ci]; ci++) {
                if (utf8_len >= (int)sizeof(utf8_buf) - 1) {
                    callback(utf8_buf, false, userdata);
                    utf8_len = 0;
                }
                utf8_buf[utf8_len++] = text[ci];
                utf8_buf[utf8_len] = '\0';

                uint8_t first = (uint8_t)utf8_buf[0];
                int expected;
                if (first < 0x80) expected = 1;
                else if (first < 0xE0) expected = 2;
                else if (first < 0xF0) expected = 3;
                else expected = 4;

                if (utf8_len >= expected) {
                    callback(utf8_buf, false, userdata);
                    utf8_len = 0;
                }
            }
        }
        tokens_generated++;

        // Forward pass + GPU sampling for next token
        int sampled = -1;
        int rc = forward_pass_sample(ctx, next_id, ctx->kv_pos,
                                     temperature, 0.9f, rng_next(), &sampled);
        if (rc != 0) { gen_error = rc; break; }
        ctx->kv_pos++;
        next_id = sampled;
    }

    // Signal completion
    callback("", true, userdata);

    // Update stats
    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double gen_secs = (t_end.tv_sec - t_gen_start.tv_sec) +
                      (t_end.tv_nsec - t_gen_start.tv_nsec) / 1e9;
    double total_secs = (t_end.tv_sec - t_start.tv_sec) +
                        (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    double ttft_secs = (t_gen_start.tv_sec - t_start.tv_sec) +
                       (t_gen_start.tv_nsec - t_start.tv_nsec) / 1e9;

    ctx->stats.tokens_generated = tokens_generated;
    ctx->stats.prompt_tokens = n_tokens;
    ctx->stats.tokens_per_second = gen_secs > 0 ? tokens_generated / gen_secs : 0;
    ctx->stats.ttft_seconds = ttft_secs;
    ctx->stats.total_seconds = total_secs;
    ctx->stats.n_gpus_active = ctx->n_gpus;
    ctx->heat_total_tokens += (uint64_t)(n_tokens + tokens_generated);

    // Auto-pin hot experts after sufficient statistical signal
    if (!ctx->heat_pinning_active && ctx->heat_total_tokens >= HEAT_MIN_TOKENS_FOR_PINNING)
        mnemo_cuda_heat_pin(ctx);

    // Calculate total VRAM used across all GPUs
    size_t vram = 0;
    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        vram += gpu->resident_size;
        vram += gpu->expert_buf_size;  // device expert staging buffer
        vram += gpu->expert_cache_size; // device expert LRU cache
        // Note: prefetch_buf is pinned HOST memory, not VRAM
        // KV cache: 2 * n_layers * ctx_len * NKV * HD * sizeof(fp16)
        int n_layers = gpu->layer_end - gpu->layer_start;
        size_t kv_per_tok = (size_t)cfg->num_key_value_heads * cfg->head_dim * sizeof(uint16_t);
        vram += (size_t)n_layers * cfg->max_position_embeddings * kv_per_tok * 2;
    }
    ctx->stats.vram_used_bytes = vram;
    ctx->stats.resident_size_bytes = ctx->resident_size;

    LOG_INFO("Done: %d tokens in %.1fs (%.1f tok/s, prefill %.1fs)",
            tokens_generated, total_secs, ctx->stats.tokens_per_second,
            (t_gen_start.tv_sec - t_start.tv_sec) +
            (t_gen_start.tv_nsec - t_start.tv_nsec) / 1e9);

    // Per-layer cache stats — show worst 5 layers
    if (ctx->layer_hits && ctx->layer_misses) {
        int NL = cfg->num_hidden_layers;
        // Find worst layers by hit rate
        int worst[5] = {-1,-1,-1,-1,-1};
        for (int w = 0; w < 5 && w < NL; w++) {
            float worst_rate = 2.0f;
            for (int l = 0; l < NL; l++) {
                uint32_t total = ctx->layer_hits[l] + ctx->layer_misses[l];
                if (total < 10) continue;
                float rate = (float)ctx->layer_hits[l] / (float)total;
                // Skip if already in worst list
                int dup = 0;
                for (int p = 0; p < w; p++) if (worst[p] == l) { dup = 1; break; }
                if (dup) continue;
                if (rate < worst_rate) { worst_rate = rate; worst[w] = l; }
            }
        }
        char worst_buf[256];
        int wpos = 0;
        for (int w = 0; w < 5 && worst[w] >= 0; w++) {
            int l = worst[w];
            uint32_t t = ctx->layer_hits[l] + ctx->layer_misses[l];
            wpos += snprintf(worst_buf + wpos, sizeof(worst_buf) - wpos,
                    " L%d=%.0f%%", l,
                    t > 0 ? 100.0 * ctx->layer_hits[l] / t : 0);
        }
        LOG_INFO("Worst layers by hit rate:%s", worst_buf);
    }

    return gen_error;
}

MnemoCudaStats mnemo_cuda_get_stats(MnemoCudaCtx *ctx) {
    return ctx ? ctx->stats : (MnemoCudaStats){0};
}

const char *mnemo_cuda_get_info(MnemoCudaCtx *ctx) {
    return ctx ? ctx->info : "Not loaded";
}


// (heat profiling moved to heat.c)
