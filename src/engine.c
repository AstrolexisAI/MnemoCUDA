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

// Additional kernel declarations
extern void cuda_matvec_q3k(const void *weights, const float *x, float *y,
                            int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_matvec_q5k(const void *weights, const float *x, float *y,
                            int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_matvec_q8_0(const void *weights, const float *x, float *y,
                             int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_matvec_f32(const float *weights, const float *x, float *y,
                            int n_rows, int n_cols, cudaStream_t stream);
extern void cuda_attention(const float *q, const float *kv_k, const float *kv_v,
                           float *out, int n_heads_q, int head_dim, int n_kv_heads,
                           int seq_len, int gqa_ratio, float scale, cudaStream_t stream);
extern void cuda_scaled_add(float *out, const float *x, float scale, int n,
                            cudaStream_t stream);
extern void cuda_topk_softmax(const float *scores, int *indices, float *weights,
                              int n_experts, int k, cudaStream_t stream);
extern void cuda_f32_to_f16(const float *in, void *out, int n, cudaStream_t stream);
extern void cuda_attention_f16kv(const void *q, const void *kv_k, const void *kv_v,
                                 float *out, int n_heads_q, int head_dim, int n_kv_heads,
                                 int seq_len, int gqa_ratio, float scale, cudaStream_t stream);

// Our CUDA kernels (extern "C" wrappers)
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
extern void cuda_moe_combine(const float *expert_outs, const float *weights_arr,
                             const float *residual, float *output,
                             int hidden, int k, cudaStream_t stream);
extern void cuda_rope(float *q, float *k, int head_dim, int pos, float theta,
                      int n_heads_q, int n_heads_k, cudaStream_t stream);

// ── Model config (same as MnemoMetal) ──

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
} ModelConfig;

// ── Expert layer file ──

typedef struct {
    int fd;
    int n_experts;
    size_t expert_size;
    size_t gate_size, up_size, down_size;
    void *mmap_data;    // mmap'd expert file (NULL if not mapped)
    size_t mmap_size;   // total file size
} ExpertLayerFile;

// ── Per-GPU state (pipeline parallelism: each GPU owns a range of layers) ──

typedef struct {
    int gpu_id;                  // CUDA device ID
    cudaStream_t stream_compute; // Compute stream
    cudaStream_t stream_io;      // Expert I/O stream (async memcpy)

    // Resident weights for this GPU's layers
    void *d_resident;            // Device pointer to resident weights
    size_t resident_size;
    size_t resident_offset;      // Offset into host mmap for this GPU's portion

    // Layer range this GPU owns
    int layer_start;             // First layer (inclusive)
    int layer_end;               // Last layer (exclusive)

    // Compute buffers (per-GPU)
    float *d_hidden;
    float *d_residual;
    float *d_q, *d_k, *d_v;
    float *d_attn_out;
    float *d_normed;
    float *d_router_logits;
    float *d_expert_gate, *d_expert_up, *d_expert_act, *d_expert_out;
    float *d_moe_out;
    float *d_logits;             // Only on last GPU

    // Expert I/O buffers (pinned host + device)
    void *h_expert_buf;          // Pinned host for pread
    void *d_expert_buf;          // Device for GPU compute
    size_t expert_buf_size;

    // KV cache for this GPU's layers (FP16)
    void *d_kv_k, *d_kv_v;

    // Router + top-K on GPU
    float *d_router_out;     // [num_experts]
    int   *d_expert_indices; // [K]
    float *d_expert_weights; // [K]

    // Expert VRAM cache — LRU cache of expert weights on GPU
    void *d_expert_cache;        // Contiguous VRAM slab for cached experts
    size_t expert_cache_size;    // Total bytes allocated
    int expert_cache_slots;      // Number of expert slots that fit
    size_t expert_slot_size;     // Bytes per expert slot (= max expert_size)

    // Cache index: slot → (layer, expert_id), -1 = empty
    int *cache_layer;            // [expert_cache_slots]
    int *cache_expert;           // [expert_cache_slots]
    uint64_t *cache_lru;         // [expert_cache_slots] — access counter for LRU
    uint64_t cache_clock;        // Global clock for LRU
    bool *cache_pinned;          // [expert_cache_slots] — true = hot expert, don't evict

    // Prefetch buffer for next-layer prediction
    void *h_prefetch_buf;        // Pinned host buffer for speculative reads
    size_t prefetch_buf_size;
    int prefetch_layer;          // Which layer was prefetched (-1 = none)
    int prefetch_eids[16];       // Expert IDs in prefetch buffer
    int n_prefetched;            // Number of experts prefetched
    volatile int prefetch_ready; // 1 when prefetch I/O is complete
} GPUState;

// ── Tensor hash table (FNV-1a, O(1) lookup) ──

#define TENSOR_HASH_BUCKETS 2048

typedef struct TensorEntry {
    char name[128];
    void *data;             // Pointer into mmap'd resident (host) OR d_resident (device)
    size_t offset;          // Byte offset into resident_weights.bin
    size_t size;            // Total byte size
    int type_id;            // 0=F32, 12=Q4_K, 14=Q6_K, 8=Q8_0
    int dims[4];
    int n_dims;
    struct TensorEntry *next;
} TensorEntry;

typedef struct {
    TensorEntry *buckets[TENSOR_HASH_BUCKETS];
    TensorEntry *entries;
    int n_entries;
} TensorTable;

static uint32_t fnv1a(const char *s) {
    uint32_t h = 0x811c9dc5;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 0x01000193;
    }
    return h;
}

static void tensor_table_insert(TensorTable *t, TensorEntry *e) {
    uint32_t idx = fnv1a(e->name) % TENSOR_HASH_BUCKETS;
    e->next = t->buckets[idx];
    t->buckets[idx] = e;
}

static TensorEntry *tensor_find(TensorTable *t, const char *name) {
    uint32_t idx = fnv1a(name) % TENSOR_HASH_BUCKETS;
    for (TensorEntry *e = t->buckets[idx]; e; e = e->next)
        if (strcmp(e->name, name) == 0) return e;
    return NULL;
}

// ── Tokenizer (BPE with special tokens) ──

typedef struct {
    char **vocab;       // id → string
    int *vocab_len;     // string lengths
    int vocab_size;

    // Merge pairs: "tokenA tokenB" → priority (lower = higher priority)
    char **merges;
    int n_merges;

    // Special tokens
    char **special_tokens;
    int *special_ids;
    int n_special;

    int eos_id;
    int im_end_id;
    int im_start_id;
} Tokenizer;

// Minimal JSON string extractor — finds next quoted string after pos
static const char *json_next_string(const char *p, char *out, int maxlen) {
    p = strchr(p, '"');
    if (!p) return NULL;
    p++; // skip opening quote
    int i = 0;
    while (*p && *p != '"' && i < maxlen - 1) {
        if (*p == '\\' && *(p+1)) {
            p++;
            switch (*p) {
                case 'n': out[i++] = '\n'; break;
                case 't': out[i++] = '\t'; break;
                case '\\': out[i++] = '\\'; break;
                case '"': out[i++] = '"'; break;
                case '/': out[i++] = '/'; break;
                case 'u': {
                    // Parse \uXXXX
                    unsigned cp = 0;
                    for (int j = 0; j < 4 && p[1+j]; j++) {
                        char c = p[1+j];
                        cp = cp * 16 + (c >= 'a' ? c-'a'+10 : c >= 'A' ? c-'A'+10 : c-'0');
                    }
                    p += 4;
                    // UTF-8 encode
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

static Tokenizer *tokenizer_load(const char *model_dir) {
    // Try binary format first (prep_tokenizer.py output), fall back to JSON
    char path[1200];
    snprintf(path, sizeof(path), "%s/tokenizer.bin", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[MnemoCUDA] No tokenizer.bin — run prep_tokenizer.py first\n");
        return NULL;
    }

    // Read header
    uint32_t magic, vocab_size, n_merges, n_special, eos_id, im_start_id, im_end_id;
    fread(&magic, 4, 1, f);
    if (magic != 0x4D544F4B) {
        fprintf(stderr, "[MnemoCUDA] Bad tokenizer magic: 0x%08X\n", magic);
        fclose(f); return NULL;
    }
    fread(&vocab_size, 4, 1, f);
    fread(&n_merges, 4, 1, f);
    fread(&n_special, 4, 1, f);
    fread(&eos_id, 4, 1, f);
    fread(&im_start_id, 4, 1, f);
    fread(&im_end_id, 4, 1, f);

    Tokenizer *tok = calloc(1, sizeof(Tokenizer));
    tok->vocab_size = (int)vocab_size;
    tok->eos_id = (int)eos_id;
    tok->im_start_id = (int)im_start_id;
    tok->im_end_id = (int)im_end_id;

    // Read vocab
    tok->vocab = calloc(vocab_size, sizeof(char*));
    tok->vocab_len = calloc(vocab_size, sizeof(int));
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint16_t len;
        fread(&len, 2, 1, f);
        if (len > 0) {
            tok->vocab[i] = malloc(len + 1);
            fread(tok->vocab[i], 1, len, f);
            tok->vocab[i][len] = '\0';
            tok->vocab_len[i] = len;
        }
    }

    // Read merges
    tok->n_merges = (int)n_merges;
    tok->merges = calloc(n_merges, sizeof(char*));
    for (uint32_t i = 0; i < n_merges; i++) {
        uint16_t len;
        fread(&len, 2, 1, f);
        tok->merges[i] = malloc(len + 1);
        fread(tok->merges[i], 1, len, f);
        tok->merges[i][len] = '\0';
    }

    // Read special tokens
    tok->n_special = (int)n_special;
    tok->special_tokens = calloc(n_special, sizeof(char*));
    tok->special_ids = calloc(n_special, sizeof(int));
    for (uint32_t i = 0; i < n_special; i++) {
        uint32_t sid;
        uint16_t len;
        fread(&sid, 4, 1, f);
        fread(&len, 2, 1, f);
        tok->special_ids[i] = (int)sid;
        tok->special_tokens[i] = malloc(len + 1);
        fread(tok->special_tokens[i], 1, len, f);
        tok->special_tokens[i][len] = '\0';
    }

    fclose(f);
    fprintf(stderr, "[MnemoCUDA] Tokenizer: %d vocab, %d merges, %d special\n",
            tok->vocab_size, tok->n_merges, tok->n_special);
    return tok;
}

static void tokenizer_free(Tokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i]);
    free(tok->vocab); free(tok->vocab_len);
    for (int i = 0; i < tok->n_merges; i++) free(tok->merges[i]);
    free(tok->merges);
    for (int i = 0; i < tok->n_special; i++) free(tok->special_tokens[i]);
    free(tok->special_tokens); free(tok->special_ids);
    free(tok);
}

// Find vocab ID for a string (linear scan — only used during tokenization)
static int tokenizer_find_token(Tokenizer *tok, const char *str, int len) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i] && tok->vocab_len[i] == len &&
            memcmp(tok->vocab[i], str, len) == 0)
            return i;
    }
    return -1;
}

// BPE encode a non-special segment into token IDs
static int bpe_encode_segment(Tokenizer *tok, const char *text, int text_len,
                              int *out_ids, int max_ids) {
    if (text_len == 0) return 0;

    // Start with character-level tokens
    // Each "symbol" is a substring of text
    typedef struct { int start; int len; } Sym;
    Sym *syms = malloc(text_len * sizeof(Sym));
    int n_syms = 0;

    // Greedy initial tokenization (longest match per character)
    int pos = 0;
    while (pos < text_len && n_syms < text_len) {
        int best_len = 1;
        // Try increasingly longer matches
        for (int l = text_len - pos; l >= 1; l--) {
            if (tokenizer_find_token(tok, text + pos, l) >= 0) {
                best_len = l;
                break;
            }
        }
        syms[n_syms].start = pos;
        syms[n_syms].len = best_len;
        n_syms++;
        pos += best_len;
    }

    // BPE merge loop: apply merges in priority order
    for (int m = 0; m < tok->n_merges && n_syms > 1; m++) {
        const char *merge = tok->merges[m];
        // Parse "tokenA tokenB"
        const char *space = strchr(merge, ' ');
        if (!space) continue;
        int a_len = (int)(space - merge);
        const char *b_str = space + 1;
        int b_len = strlen(b_str);

        for (int i = 0; i < n_syms - 1; i++) {
            if (syms[i].len == a_len && memcmp(text + syms[i].start, merge, a_len) == 0 &&
                syms[i+1].len == b_len && memcmp(text + syms[i+1].start, b_str, b_len) == 0) {
                // Merge: extend sym[i] to cover both, remove sym[i+1]
                syms[i].len += syms[i+1].len;
                memmove(&syms[i+1], &syms[i+2], (n_syms - i - 2) * sizeof(Sym));
                n_syms--;
                i--; // Re-check at same position
            }
        }
    }

    // Convert symbols to IDs
    int n_ids = 0;
    for (int i = 0; i < n_syms && n_ids < max_ids; i++) {
        int id = tokenizer_find_token(tok, text + syms[i].start, syms[i].len);
        if (id >= 0) {
            out_ids[n_ids++] = id;
        } else {
            // Fallback: encode as bytes (shouldn't happen with good vocab)
            for (int j = 0; j < syms[i].len && n_ids < max_ids; j++) {
                // Byte fallback token
                out_ids[n_ids++] = (unsigned char)text[syms[i].start + j];
            }
        }
    }

    free(syms);
    return n_ids;
}

// Full tokenize: split on special tokens first, then BPE each segment
static int tokenizer_encode(Tokenizer *tok, const char *text,
                            int *out_ids, int max_ids) {
    int n_ids = 0;
    int text_len = strlen(text);
    int pos = 0;

    while (pos < text_len && n_ids < max_ids) {
        // Find earliest special token match
        int best_sp = -1, best_pos = text_len;
        for (int s = 0; s < tok->n_special; s++) {
            const char *found = strstr(text + pos, tok->special_tokens[s]);
            if (found && (int)(found - text) < best_pos) {
                best_pos = (int)(found - text);
                best_sp = s;
            }
        }

        // Encode text before the special token (BPE)
        if (best_pos > pos) {
            n_ids += bpe_encode_segment(tok, text + pos, best_pos - pos,
                                         out_ids + n_ids, max_ids - n_ids);
        }

        if (best_sp >= 0) {
            // Add special token
            if (n_ids < max_ids)
                out_ids[n_ids++] = tok->special_ids[best_sp];
            pos = best_pos + strlen(tok->special_tokens[best_sp]);
        } else {
            // No more special tokens — encode remainder
            if (best_pos < text_len) {
                n_ids += bpe_encode_segment(tok, text + pos, text_len - pos,
                                             out_ids + n_ids, max_ids - n_ids);
            }
            break;
        }
    }
    return n_ids;
}

// Decode a single token ID to string
// ByteLevel decode: BPE vocab stores unicode chars that map to raw bytes.
// This table converts unicode codepoints (0-511) back to the original byte value.
static const int16_t bytelevel_to_byte[512] = {
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
     64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
     80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
     96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,  -1, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 173,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
};

static char decode_buf[1024];

static const char *tokenizer_decode(Tokenizer *tok, int id) {
    const char *raw = NULL;
    if (id >= 0 && id < tok->vocab_size && tok->vocab[id])
        raw = tok->vocab[id];
    else {
        for (int i = 0; i < tok->n_special; i++)
            if (tok->special_ids[i] == id) return "";
        return "";
    }

    // Decode UTF-8 string from vocab, convert each unicode codepoint via ByteLevel table
    // Output: raw bytes (which form valid UTF-8 text)
    int j = 0;
    for (int i = 0; raw[i] && j < (int)sizeof(decode_buf) - 4; ) {
        // Decode one UTF-8 codepoint from raw
        uint32_t cp;
        uint8_t c = (uint8_t)raw[i];
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if (c < 0xE0) { cp = (c & 0x1F) << 6 | ((uint8_t)raw[i+1] & 0x3F); len = 2; }
        else if (c < 0xF0) { cp = (c & 0x0F) << 12 | ((uint8_t)raw[i+1] & 0x3F) << 6 | ((uint8_t)raw[i+2] & 0x3F); len = 3; }
        else { cp = (c & 0x07) << 18 | ((uint8_t)raw[i+1] & 0x3F) << 12 | ((uint8_t)raw[i+2] & 0x3F) << 6 | ((uint8_t)raw[i+3] & 0x3F); len = 4; }
        i += len;

        // Look up in ByteLevel table
        if (cp < 512 && bytelevel_to_byte[cp] >= 0) {
            decode_buf[j++] = (char)(uint8_t)bytelevel_to_byte[cp];
        } else {
            // Not in ByteLevel mapping — pass through as UTF-8
            i -= len;
            for (int k = 0; k < len && j < (int)sizeof(decode_buf) - 1; k++)
                decode_buf[j++] = raw[i++];
        }
    }
    decode_buf[j] = '\0';
    return decode_buf;
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

// ── CPU helpers for attention (host-side softmax, top-K) ──

static void softmax_inplace(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void top_k_experts(const float *scores, int n, int k,
                          int *out_indices, float *out_weights) {
    for (int i = 0; i < k; i++) { out_indices[i] = -1; out_weights[i] = -INFINITY; }
    for (int j = 0; j < n; j++) {
        int min_idx = 0;
        for (int i = 1; i < k; i++)
            if (out_weights[i] < out_weights[min_idx]) min_idx = i;
        if (scores[j] > out_weights[min_idx]) {
            out_weights[min_idx] = scores[j];
            out_indices[min_idx] = j;
        }
    }
    // Softmax over selected
    float max_w = out_weights[0];
    for (int i = 1; i < k; i++) if (out_weights[i] > max_w) max_w = out_weights[i];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) { out_weights[i] = expf(out_weights[i] - max_w); sum += out_weights[i]; }
    for (int i = 0; i < k; i++) out_weights[i] /= sum;
}

// ── Persistent I/O thread pool (avoids 752 pthread_create/join per token) ──

#include <semaphore.h>

#define IO_POOL_SIZE 8

typedef struct {
    int fd;
    void *dst;
    const void *src;
    size_t size;
    off_t offset;
    volatile int done;
} IOTask;

typedef struct {
    pthread_t threads[IO_POOL_SIZE];
    IOTask tasks[IO_POOL_SIZE];
    sem_t task_ready[IO_POOL_SIZE];  // signal worker: new task
    sem_t task_done[IO_POOL_SIZE];   // signal main: task complete
    volatile int shutdown;
} IOPool;

static IOPool g_io_pool;
static int g_io_pool_init = 0;

static void *io_pool_worker(void *arg) {
    int id = (int)(intptr_t)arg;
    IOPool *pool = &g_io_pool;
    while (!pool->shutdown) {
        sem_wait(&pool->task_ready[id]);
        if (pool->shutdown) break;
        IOTask *t = &pool->tasks[id];
        if (t->fd >= 0) {
            pread(t->fd, t->dst, t->size, t->offset);
        } else if (t->src) {
            memcpy(t->dst, t->src, t->size);
        }
        t->done = 1;
        sem_post(&pool->task_done[id]);
    }
    return NULL;
}

static void io_pool_init(void) {
    if (g_io_pool_init) return;
    memset(&g_io_pool, 0, sizeof(g_io_pool));
    for (int i = 0; i < IO_POOL_SIZE; i++) {
        sem_init(&g_io_pool.task_ready[i], 0, 0);
        sem_init(&g_io_pool.task_done[i], 0, 0);
        pthread_create(&g_io_pool.threads[i], NULL, io_pool_worker, (void *)(intptr_t)i);
    }
    g_io_pool_init = 1;
}

static void io_pool_shutdown(void) {
    if (!g_io_pool_init) return;
    g_io_pool.shutdown = 1;
    for (int i = 0; i < IO_POOL_SIZE; i++)
        sem_post(&g_io_pool.task_ready[i]);
    for (int i = 0; i < IO_POOL_SIZE; i++)
        pthread_join(g_io_pool.threads[i], NULL);
    for (int i = 0; i < IO_POOL_SIZE; i++) {
        sem_destroy(&g_io_pool.task_ready[i]);
        sem_destroy(&g_io_pool.task_done[i]);
    }
    g_io_pool_init = 0;
}

// Submit N tasks and wait for all to complete
static void io_pool_submit_wait(IOTask *tasks, int n) {
    if (n > IO_POOL_SIZE) n = IO_POOL_SIZE;
    for (int i = 0; i < n; i++) {
        g_io_pool.tasks[i] = tasks[i];
        g_io_pool.tasks[i].done = 0;
        sem_post(&g_io_pool.task_ready[i]);
    }
    for (int i = 0; i < n; i++)
        sem_wait(&g_io_pool.task_done[i]);
}

// Submit N tasks WITHOUT waiting (for overlap with GPU compute)
static void io_pool_submit(IOTask *tasks, int n) {
    if (n > IO_POOL_SIZE) n = IO_POOL_SIZE;
    for (int i = 0; i < n; i++) {
        g_io_pool.tasks[i] = tasks[i];
        g_io_pool.tasks[i].done = 0;
        sem_post(&g_io_pool.task_ready[i]);
    }
}

static void io_pool_wait(int n) {
    if (n > IO_POOL_SIZE) n = IO_POOL_SIZE;
    for (int i = 0; i < n; i++)
        sem_wait(&g_io_pool.task_done[i]);
}

// (parallel_pread removed — replaced by io_pool)

// ── RAM-tier expert cache (pinned host memory, between VRAM and NVMe) ──

#define RAM_CACHE_DEFAULT_GB 0  // Disabled: OS page cache already serves as RAM tier for mmap'd experts

typedef struct {
    void *data;              // Pinned host memory slab
    size_t slot_size;        // Bytes per expert slot
    int n_slots;             // Total expert slots
    int *slot_layer;         // [n_slots] layer ID (-1 = empty)
    int *slot_expert;        // [n_slots] expert ID
    uint64_t *slot_lru;      // [n_slots] LRU counter
    uint64_t clock;          // Global LRU counter
    int hits;                // Stats
    int misses;
} RAMCache;

static int ram_cache_init(RAMCache *rc, size_t slot_size, int target_gb) {
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
        fprintf(stderr, "[MnemoCUDA] RAM cache: allocation failed\n");
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

    fprintf(stderr, "[MnemoCUDA] RAM cache: %.1f GB (%d slots × %.1f MB)\n",
            (double)total / (1024*1024*1024), rc->n_slots,
            (double)slot_size / (1024*1024));
    return 0;
}

// Returns pointer to expert data in RAM cache, or NULL
static void *ram_cache_lookup(RAMCache *rc, int layer, int expert_id) {
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
static void *ram_cache_insert(RAMCache *rc, int layer, int expert_id,
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

static void ram_cache_free(RAMCache *rc) {
    free(rc->data);
    free(rc->slot_layer);
    free(rc->slot_expert);
    free(rc->slot_lru);
    memset(rc, 0, sizeof(*rc));
}

// ── Main context ──

struct MnemoCudaCtx {
    ModelConfig config;
    char model_dir[1024];

    // Multi-GPU state
    GPUState gpus[8];
    int n_gpus;

    // Full resident weights (mmap'd on host for initial copy)
    void *h_resident_mmap;
    size_t resident_size;

    // Tensor hash table (name → offset/size/type in resident)
    TensorTable tensor_table;

    // Expert files (shared across GPUs — each GPU reads its own layers)
    ExpertLayerFile *expert_layers;
    int n_expert_layers;

    // Tokenizer
    Tokenizer *tokenizer;

    // Hidden state transfer buffer (for inter-GPU communication)
    float *h_hidden_transfer;    // Pinned host buffer [hidden_size]

    int kv_pos;

    // RAM-tier expert cache (pinned host, between VRAM and NVMe)
    RAMCache ram_cache;

    // Expert heat profiling
    uint32_t *heat_map;          // [num_hidden_layers * num_experts] activation counts
    uint64_t heat_total_tokens;  // Total tokens processed (for normalization)
    bool heat_pinning_active;    // Whether hot experts are pinned in cache

    // State
    volatile bool cancelled;
    bool loaded;
    MnemoCudaStats stats;
    char info[256];
};

// ── Implementation ──

MnemoCudaConfig mnemo_cuda_config_default(void) {
    MnemoCudaConfig c = {0};
    c.context_length = 65536; // 65K context — fits in 24+32 GB dual-GPU
    c.n_gpus = 0; // 0 = auto-detect
    c.io_threads = 8;
    c.use_pinned_memory = true;
    return c;
}

MnemoCudaCtx *mnemo_cuda_create(void) {
    MnemoCudaCtx *ctx = calloc(1, sizeof(MnemoCudaCtx));
    return ctx;
}

// Minimal JSON value extractor — finds "key": value in config.json
static int json_get_int(const char *json, const char *key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}
static float json_get_float(const char *json, const char *key, float default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return (float)atof(p);
}

static int load_config_json(MnemoCudaCtx *ctx, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = malloc(size + 1);
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    // Auto-detect architecture prefix (qwen3moe, qwen3next, llama, etc.)
    const char *prefix = "qwen3moe";
    if (strstr(json, "\"qwen3next.")) prefix = "qwen3next";
    else if (strstr(json, "\"llama.")) prefix = "llama";
    else if (strstr(json, "\"qwen2.")) prefix = "qwen2";
    fprintf(stderr, "[MnemoCUDA] Config prefix: %s\n", prefix);

    char key[256];
    #define CFG_INT(field, suffix, def) \
        snprintf(key, sizeof(key), "%s.%s", prefix, suffix); \
        ctx->config.field = json_get_int(json, key, def)
    #define CFG_FLOAT(field, suffix, def) \
        snprintf(key, sizeof(key), "%s.%s", prefix, suffix); \
        ctx->config.field = json_get_float(json, key, def)

    CFG_INT(hidden_size, "embedding_length", 4096);
    CFG_INT(moe_intermediate_size, "expert_feed_forward_length", 1536);
    CFG_INT(num_attention_heads, "attention.head_count", 64);
    CFG_INT(num_key_value_heads, "attention.head_count_kv", 4);
    CFG_INT(head_dim, "attention.key_length", 128);
    CFG_INT(num_hidden_layers, "block_count", 94);
    ctx->config.vocab_size = 151936;
    CFG_INT(num_experts, "expert_count", 128);
    CFG_INT(num_experts_per_tok, "expert_used_count", 8);
    CFG_FLOAT(rope_theta, "rope.freq_base", 1000000.0f);
    CFG_FLOAT(rms_norm_eps, "attention.layer_norm_rms_epsilon", 1e-6f);
    CFG_INT(max_position_embeddings, "context_length", 40960);

    #undef CFG_INT
    #undef CFG_FLOAT

    free(json);
    return 0;
}

int mnemo_cuda_load(MnemoCudaCtx *ctx, MnemoCudaConfig config) {
    if (!ctx || !config.model_dir) return -1;

    strncpy(ctx->model_dir, config.model_dir, sizeof(ctx->model_dir) - 1);

    // Load config
    char path[1200];
    snprintf(path, sizeof(path), "%s/config.json", config.model_dir);
    load_config_json(ctx, path);

    if (config.expert_k > 0)
        ctx->config.num_experts_per_tok = config.expert_k;

    ctx->config.max_position_embeddings = config.context_length;

    ModelConfig *cfg = &ctx->config;

    // Load tokenizer
    ctx->tokenizer = tokenizer_load(config.model_dir);
    if (!ctx->tokenizer) {
        fprintf(stderr, "[MnemoCUDA] Warning: tokenizer not loaded, inference will fail\n");
    }

    // Load resident_manifest.json → build tensor hash table
    snprintf(path, sizeof(path), "%s/resident_manifest.json", config.model_dir);
    FILE *mf2 = fopen(path, "r");
    if (mf2) {
        fseek(mf2, 0, SEEK_END);
        long mf2_size = ftell(mf2);
        fseek(mf2, 0, SEEK_SET);
        char *mf2_json = malloc(mf2_size + 1);
        fread(mf2_json, 1, mf2_size, mf2);
        mf2_json[mf2_size] = '\0';
        fclose(mf2);

        // Count tensors (count "name" occurrences)
        int tensor_count = 0;
        const char *cnt = mf2_json;
        while ((cnt = strstr(cnt, "\"name\"")) != NULL) { tensor_count++; cnt += 6; }

        memset(&ctx->tensor_table, 0, sizeof(TensorTable));
        ctx->tensor_table.entries = calloc(tensor_count, sizeof(TensorEntry));
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
        fprintf(stderr, "[MnemoCUDA] Resident manifest: %d tensors indexed\n",
                ctx->tensor_table.n_entries);
    } else {
        fprintf(stderr, "[MnemoCUDA] Warning: no resident_manifest.json\n");
    }

    // Auto-detect GPUs if n_gpus == 0
    if (config.n_gpus == 0) {
        cudaGetDeviceCount(&ctx->n_gpus);
        if (ctx->n_gpus > 8) ctx->n_gpus = 8;
        for (int i = 0; i < ctx->n_gpus; i++)
            ctx->gpus[i].gpu_id = i;
    } else {
        ctx->n_gpus = config.n_gpus;
        for (int i = 0; i < ctx->n_gpus; i++)
            ctx->gpus[i].gpu_id = config.gpu_ids[i];
    }

    fprintf(stderr, "[MnemoCUDA] Loading: hidden=%d, layers=%d, experts=%d (K=%d), GPUs=%d\n",
            cfg->hidden_size, cfg->num_hidden_layers,
            cfg->num_experts, cfg->num_experts_per_tok, ctx->n_gpus);

    // Open expert files BEFORE CUDA allocations (CUDA may invalidate fds)
    ctx->expert_layers = calloc(cfg->num_hidden_layers, sizeof(ExpertLayerFile));
    ctx->n_expert_layers = 0;
    {
        snprintf(path, sizeof(path), "%s/expert_manifest.json", config.model_dir);
        FILE *emf = fopen(path, "r");
        char *em_json = NULL;
        if (emf) {
            fseek(emf, 0, SEEK_END); long emsz = ftell(emf); fseek(emf, 0, SEEK_SET);
            em_json = malloc(emsz + 1); fread(em_json, 1, emsz, emf); em_json[emsz] = '\0'; fclose(emf);
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
                if (ctx->expert_layers[i].mmap_data == MAP_FAILED)
                    ctx->expert_layers[i].mmap_data = NULL;
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
        fprintf(stderr, "[MnemoCUDA] Expert files: %d/%d mmap'd (%.1f GB, pre-warming page cache)\n",
                ctx->n_expert_layers, cfg->num_hidden_layers, (double)total_mmap / (1024*1024*1024));
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

        fprintf(stderr, "[MnemoCUDA] GPU %d: layers %d-%d (%d layers)\n",
                gpu->gpu_id, gpu->layer_start, gpu->layer_end - 1,
                gpu->layer_end - gpu->layer_start);
    }

    // Load resident weights: mmap file on host
    snprintf(path, sizeof(path), "%s/resident_weights.bin", config.model_dir);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[MnemoCUDA] Cannot open %s\n", path); return -2; }

    struct stat st;
    fstat(fd, &st);
    ctx->resident_size = st.st_size;

    ctx->h_resident_mmap = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (ctx->h_resident_mmap == MAP_FAILED) { close(fd); return -2; }
    close(fd);

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
        fprintf(stderr, "[MnemoCUDA] Sanitized %d NaN/Inf values in resident F32 tensors\n", nan_fixed);

    // Copy resident weights to EACH GPU (each GPU gets full copy for global tensors)
    // TODO: in future, partition per-layer tensors to only the owning GPU
    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        cudaSetDevice(gpu->gpu_id);

        cudaStreamCreate(&gpu->stream_compute);
        cudaStreamCreate(&gpu->stream_io);

        cudaMalloc(&gpu->d_resident, st.st_size);
        cudaMemcpy(gpu->d_resident, ctx->h_resident_mmap, st.st_size, cudaMemcpyHostToDevice);
        gpu->resident_size = st.st_size;

        fprintf(stderr, "[MnemoCUDA] GPU %d: %.1f MB resident → VRAM\n",
                gpu->gpu_id, (double)st.st_size / (1024*1024));
    }

    // Set tensor data pointers (GPU 0's resident for now — all GPUs have full copy)
    for (int i = 0; i < ctx->tensor_table.n_entries; i++) {
        TensorEntry *e = &ctx->tensor_table.entries[i];
        e->data = (char *)ctx->gpus[0].d_resident + e->offset;
    }

    // Allocate pinned host buffer for inter-GPU hidden state transfer
    cudaMallocHost((void**)&ctx->h_hidden_transfer, cfg->hidden_size * sizeof(float));

    // Allocate per-GPU compute buffers
    int H = cfg->hidden_size;
    int NH = cfg->num_attention_heads;
    int NKV = cfg->num_key_value_heads;
    int HD = cfg->head_dim;
    int EFF = cfg->moe_intermediate_size;
    int V = cfg->vocab_size;
    int K = cfg->num_experts_per_tok;

    // Determine max expert size from already-opened expert files
    size_t max_expert_sz = 16UL * 1024 * 1024;
    for (int i = 0; i < cfg->num_hidden_layers; i++)
        if (ctx->expert_layers[i].expert_size > 0 &&
            ctx->expert_layers[i].expert_size < max_expert_sz)
            max_expert_sz = ctx->expert_layers[i].expert_size; // use actual, not default
    for (int i = 0; i < cfg->num_hidden_layers; i++)
        if (ctx->expert_layers[i].expert_size > max_expert_sz)
            max_expert_sz = ctx->expert_layers[i].expert_size;

    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        cudaSetDevice(gpu->gpu_id);

        cudaMalloc((void**)&gpu->d_hidden,   H * sizeof(float));
        cudaMalloc((void**)&gpu->d_residual, H * sizeof(float));
        cudaMalloc((void**)&gpu->d_q,        NH * HD * sizeof(float));
        cudaMalloc((void**)&gpu->d_k,        NKV * HD * sizeof(float));
        cudaMalloc((void**)&gpu->d_v,        NKV * HD * sizeof(float));
        cudaMalloc((void**)&gpu->d_attn_out, NH * HD * sizeof(float));
        cudaMalloc((void**)&gpu->d_normed,   H * sizeof(float));
        cudaMalloc((void**)&gpu->d_router_logits, cfg->num_experts * sizeof(float));
        cudaMalloc((void**)&gpu->d_router_out, cfg->num_experts * sizeof(float));
        cudaMalloc((void**)&gpu->d_expert_indices, K * sizeof(int));
        cudaMalloc((void**)&gpu->d_expert_weights, K * sizeof(float));
        cudaMalloc((void**)&gpu->d_expert_gate, EFF * sizeof(float));
        cudaMalloc((void**)&gpu->d_expert_up,   EFF * sizeof(float));
        cudaMalloc((void**)&gpu->d_expert_act,  EFF * sizeof(float));
        cudaMalloc((void**)&gpu->d_expert_out,  H * sizeof(float));
        cudaMalloc((void**)&gpu->d_moe_out,     H * K * sizeof(float));

        // Logits only on last GPU
        if (g == ctx->n_gpus - 1)
            cudaMalloc((void**)&gpu->d_logits, V * sizeof(float));

        // Expert I/O buffer (pinned host, K experts for parallel pread)
        gpu->expert_buf_size = max_expert_sz * K;
        if (config.use_pinned_memory)
            cudaMallocHost(&gpu->h_expert_buf, gpu->expert_buf_size);
        else
            gpu->h_expert_buf = malloc(gpu->expert_buf_size);
        cudaMalloc(&gpu->d_expert_buf, gpu->expert_buf_size);

        // KV cache — FP16 (half the VRAM of FP32)
        int n_layers = gpu->layer_end - gpu->layer_start;
        size_t kv_per_tok = (size_t)NKV * HD * sizeof(uint16_t);
        size_t kv_size = (size_t)n_layers * config.context_length * kv_per_tok;
        cudaMalloc((void**)&gpu->d_kv_k, kv_size);
        cudaMalloc((void**)&gpu->d_kv_v, kv_size);
        cudaMemset(gpu->d_kv_k, 0, kv_size);
        cudaMemset(gpu->d_kv_v, 0, kv_size);

        fprintf(stderr, "[MnemoCUDA] GPU %d: KV cache %.1f MB FP16 (%d ctx), expert buf %.1f MB\n",
                gpu->gpu_id, (double)kv_size*2 / (1024*1024), config.context_length,
                (double)gpu->expert_buf_size / (1024*1024));

        // Expert VRAM cache — fill remaining VRAM with expert slots
        gpu->expert_slot_size = max_expert_sz;
        gpu->expert_cache_slots = 0;
        gpu->d_expert_cache = NULL;
        gpu->cache_layer = NULL;
        gpu->cache_expert = NULL;
        gpu->cache_lru = NULL;
        gpu->cache_pinned = NULL;
        gpu->cache_clock = 0;

        // Prefetch buffer — holds up to K experts for speculative next-layer reads
        gpu->prefetch_buf_size = max_expert_sz * K;
        if (config.use_pinned_memory)
            cudaMallocHost(&gpu->h_prefetch_buf, gpu->prefetch_buf_size);
        else
            gpu->h_prefetch_buf = malloc(gpu->prefetch_buf_size);
        gpu->prefetch_layer = -1;
        gpu->n_prefetched = 0;
        gpu->prefetch_ready = 0;

        size_t vram_free = 0, vram_total = 0;
        cudaMemGetInfo(&vram_free, &vram_total);
        size_t headroom = 256UL * 1024 * 1024; // 256 MB safety margin

        if (vram_free > headroom + max_expert_sz * 4) {
            size_t cache_budget = vram_free - headroom;
            gpu->expert_cache_slots = (int)(cache_budget / max_expert_sz);
            gpu->expert_cache_size = (size_t)gpu->expert_cache_slots * max_expert_sz;

            cudaError_t err = cudaMalloc(&gpu->d_expert_cache, gpu->expert_cache_size);
            if (err != cudaSuccess) {
                // Allocation failed — try half the size
                gpu->expert_cache_slots /= 2;
                gpu->expert_cache_size = (size_t)gpu->expert_cache_slots * max_expert_sz;
                err = cudaMalloc(&gpu->d_expert_cache, gpu->expert_cache_size);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[MnemoCUDA] GPU %d: VRAM cache alloc failed, running without cache\n", gpu->gpu_id);
                    gpu->d_expert_cache = NULL;
                    gpu->expert_cache_slots = 0;
                    gpu->expert_cache_size = 0;
                }
            }

            if (gpu->d_expert_cache) {
                gpu->cache_layer = (int *)calloc(gpu->expert_cache_slots, sizeof(int));
                gpu->cache_expert = (int *)calloc(gpu->expert_cache_slots, sizeof(int));
                gpu->cache_lru = (uint64_t *)calloc(gpu->expert_cache_slots, sizeof(uint64_t));
                gpu->cache_pinned = (bool *)calloc(gpu->expert_cache_slots, sizeof(bool));
                for (int s = 0; s < gpu->expert_cache_slots; s++) {
                    gpu->cache_layer[s] = -1;
                    gpu->cache_expert[s] = -1;
                }
                fprintf(stderr, "[MnemoCUDA] GPU %d: VRAM cache %.1f MB (%d slots × %.1f MB)\n",
                        gpu->gpu_id, (double)gpu->expert_cache_size / (1024*1024),
                        gpu->expert_cache_slots, (double)max_expert_sz / (1024*1024));
            }
        }
    }

    ctx->kv_pos = 0;

    // RAM-tier cache (disabled by default — OS page cache handles this via mmap)
    if (RAM_CACHE_DEFAULT_GB > 0)
        ram_cache_init(&ctx->ram_cache, max_expert_sz, RAM_CACHE_DEFAULT_GB);
    else
        memset(&ctx->ram_cache, 0, sizeof(ctx->ram_cache));

    // Expert heat profiling — allocate and zero (or load from disk)
    int heat_size = cfg->num_hidden_layers * cfg->num_experts;
    ctx->heat_map = (uint32_t *)calloc(heat_size, sizeof(uint32_t));
    ctx->heat_total_tokens = 0;
    ctx->heat_pinning_active = false;

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
                fread(ctx->heat_map, sizeof(uint32_t), heat_size, hf);
                ctx->heat_total_tokens = saved_tokens;
                fprintf(stderr, "[MnemoCUDA] Loaded heat map (%lu tokens of history)\n",
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

    io_pool_init();
    ctx->loaded = true;
    fprintf(stderr, "[MnemoCUDA] Ready: %s\n", ctx->info);
    return 0;
}

void mnemo_cuda_unload(MnemoCudaCtx *ctx) {
    if (!ctx || !ctx->loaded) return;

    // Save heat map FIRST (before freeing anything, while ctx->loaded is still true)
    mnemo_cuda_heat_save(ctx);

    for (int g = 0; g < ctx->n_gpus; g++) {
        GPUState *gpu = &ctx->gpus[g];
        cudaSetDevice(gpu->gpu_id);

        cudaFree(gpu->d_resident);
        cudaFree(gpu->d_hidden); cudaFree(gpu->d_residual);
        cudaFree(gpu->d_q); cudaFree(gpu->d_k); cudaFree(gpu->d_v);
        cudaFree(gpu->d_attn_out); cudaFree(gpu->d_normed);
        cudaFree(gpu->d_router_logits); cudaFree(gpu->d_router_out);
        cudaFree(gpu->d_expert_indices); cudaFree(gpu->d_expert_weights);
        cudaFree(gpu->d_expert_gate); cudaFree(gpu->d_expert_up);
        cudaFree(gpu->d_expert_act); cudaFree(gpu->d_expert_out);
        cudaFree(gpu->d_moe_out); cudaFree(gpu->d_logits);
        cudaFree(gpu->d_kv_k); cudaFree(gpu->d_kv_v);
        cudaFree(gpu->d_expert_buf);
        cudaFree(gpu->d_expert_cache);

        if (gpu->h_expert_buf) cudaFreeHost(gpu->h_expert_buf);
        if (gpu->h_prefetch_buf) cudaFreeHost(gpu->h_prefetch_buf);
        free(gpu->cache_layer);
        free(gpu->cache_expert);
        free(gpu->cache_lru);
        free(gpu->cache_pinned);

        cudaStreamDestroy(gpu->stream_compute);
        cudaStreamDestroy(gpu->stream_io);
    }

    if (ctx->h_resident_mmap) {
        munmap(ctx->h_resident_mmap, ctx->resident_size);
        ctx->h_resident_mmap = NULL;
    }
    if (ctx->h_hidden_transfer) cudaFreeHost(ctx->h_hidden_transfer);

    for (int i = 0; i < ctx->config.num_hidden_layers; i++) {
        if (ctx->expert_layers && ctx->expert_layers[i].mmap_data)
            munmap(ctx->expert_layers[i].mmap_data, ctx->expert_layers[i].mmap_size);
        if (ctx->expert_layers && ctx->expert_layers[i].fd > 0)
            close(ctx->expert_layers[i].fd);
    }
    free(ctx->expert_layers);

    free(ctx->tensor_table.entries);
    memset(&ctx->tensor_table, 0, sizeof(TensorTable));

    ram_cache_free(&ctx->ram_cache);

    free(ctx->heat_map);
    ctx->heat_map = NULL;

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

// ── Helper: get tensor data on specific GPU ──

static void *tensor_data_on_gpu(MnemoCudaCtx *ctx, const char *name, int gpu_idx) {
    TensorEntry *e = tensor_find(&ctx->tensor_table, name);
    if (!e) return NULL;
    return (char *)ctx->gpus[gpu_idx].d_resident + e->offset;
}

static TensorEntry *tensor_get(MnemoCudaCtx *ctx, const char *name) {
    return tensor_find(&ctx->tensor_table, name);
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
            fprintf(stderr, "[MnemoCUDA] Warning: unsupported type_id %d for matvec\n", type_id);
            break;
    }
}

// ── Forward one layer on a specific GPU ──

// ── Forward declarations for cache operations ──
static void *expert_cache_lookup(GPUState *gpu, int layer, int expert_id);
static bool expert_cache_has(GPUState *gpu, int layer, int expert_id);
static void *expert_cache_insert(GPUState *gpu, int layer, int expert_id,
                                  const void *host_data, size_t data_size,
                                  cudaStream_t stream);

// ── Prefetch: start reading hot experts for a future layer ──

static void prefetch_submit(MnemoCudaCtx *ctx, GPUState *gpu, int layer) {
    if (!ctx->heat_map || !gpu->h_prefetch_buf) return;

    ModelConfig *cfg = &ctx->config;
    int NE = cfg->num_experts;
    int K = cfg->num_experts_per_tok;
    if (layer < 0 || layer >= cfg->num_hidden_layers) return;

    ExpertLayerFile *elf = &ctx->expert_layers[layer];
    if (elf->fd <= 0 && !elf->mmap_data) return;

    // Find top K hot experts for this layer that are NOT in VRAM cache
    uint32_t *layer_heat = &ctx->heat_map[layer * NE];
    int n_to_prefetch = 0;

    // Build candidates sorted by heat (simple: pick top K non-cached)
    for (int pass = 0; pass < K && n_to_prefetch < K; pass++) {
        uint32_t best = 0;
        int best_eid = -1;
        for (int e = 0; e < NE; e++) {
            if (layer_heat[e] <= best) continue;
            // Skip if already selected
            int dup = 0;
            for (int p = 0; p < n_to_prefetch; p++)
                if (gpu->prefetch_eids[p] == e) { dup = 1; break; }
            if (dup) continue;
            // Skip if already in VRAM cache (no LRU bump)
            if (expert_cache_has(gpu, layer, e)) continue;
            best = layer_heat[e];
            best_eid = e;
        }
        if (best_eid < 0 || best == 0) break;
        gpu->prefetch_eids[n_to_prefetch++] = best_eid;
    }

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

// Check if an expert is in the prefetch buffer; returns host pointer or NULL
static void *prefetch_lookup(GPUState *gpu, int layer, int expert_id, size_t expert_size) {
    if (gpu->prefetch_layer != layer || !gpu->prefetch_ready) return NULL;
    for (int i = 0; i < gpu->n_prefetched; i++) {
        if (gpu->prefetch_eids[i] == expert_id) {
            return (char *)gpu->h_prefetch_buf + (size_t)i * expert_size;
        }
    }
    return NULL;
}

// ── Expert VRAM cache: lookup / insert ──

// Returns device pointer to expert data in cache, or NULL if miss
static void *expert_cache_lookup(GPUState *gpu, int layer, int expert_id) {
    if (!gpu->d_expert_cache || gpu->expert_cache_slots == 0) return NULL;
    for (int s = 0; s < gpu->expert_cache_slots; s++) {
        if (gpu->cache_layer[s] == layer && gpu->cache_expert[s] == expert_id) {
            gpu->cache_lru[s] = ++gpu->cache_clock;
            return (char *)gpu->d_expert_cache + (size_t)s * gpu->expert_slot_size;
        }
    }
    return NULL;
}

// Check-only: returns true if expert is cached, without bumping LRU
static bool expert_cache_has(GPUState *gpu, int layer, int expert_id) {
    if (!gpu->d_expert_cache || gpu->expert_cache_slots == 0) return false;
    for (int s = 0; s < gpu->expert_cache_slots; s++)
        if (gpu->cache_layer[s] == layer && gpu->cache_expert[s] == expert_id) return true;
    return false;
}

// Forward declaration for RAM demotion
struct MnemoCudaCtx;

// Insert expert into cache (evict LRU if full), returns device pointer.
// If ram_demote is non-NULL, evicted experts are demoted to RAM cache.
static void *expert_cache_insert_ctx(struct MnemoCudaCtx *ctx, GPUState *gpu,
                                      int layer, int expert_id,
                                      const void *host_data, size_t data_size,
                                      cudaStream_t stream) {
    if (!gpu->d_expert_cache || gpu->expert_cache_slots == 0) return NULL;

    // Find empty slot or LRU slot (skip pinned slots)
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

    void *slot = (char *)gpu->d_expert_cache + (size_t)target * gpu->expert_slot_size;
    cudaMemcpyAsync(slot, host_data, data_size, cudaMemcpyHostToDevice, stream);
    return slot;
}

// Simple wrapper without context (for heat_pin and prefetch where we don't need demotion)
static void *expert_cache_insert(GPUState *gpu, int layer, int expert_id,
                                  const void *host_data, size_t data_size,
                                  cudaStream_t stream) {
    return expert_cache_insert_ctx(NULL, gpu, layer, expert_id, host_data, data_size, stream);
}

static void forward_layer(MnemoCudaCtx *ctx, int layer, int gpu_idx, int pos) {
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

    // ── 1. Pre-attention RMS norm ──
    snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", layer);
    void *norm_w = tensor_data_on_gpu(ctx, tname, gpu_idx);
    cuda_rms_norm(gpu->d_hidden, (float *)norm_w, gpu->d_normed, H, eps, cs);

    // Save residual
    cudaMemcpyAsync(gpu->d_residual, gpu->d_hidden, H * sizeof(float),
                    cudaMemcpyDeviceToDevice, cs);

    // Debug checkpoint: after RMS norm
    if (0 /* layer==15 */) {
        static int layer0_dbg = 0;
        if (layer0_dbg < 2) {
            cudaStreamSynchronize(cs);
            float dbg_n[4], dbg_h[4];
            cudaMemcpy(dbg_n, gpu->d_normed, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(dbg_h, gpu->d_hidden, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L15] hidden[0..3]=%.6f %.6f %.6f %.6f\n",
                    dbg_h[0], dbg_h[1], dbg_h[2], dbg_h[3]);
            fprintf(stderr, "[DEBUG L15] normed[0..3]=%.6f %.6f %.6f %.6f\n",
                    dbg_n[0], dbg_n[1], dbg_n[2], dbg_n[3]);

            // Check norm weight
            TensorEntry *nw = tensor_get(ctx, "blk.0.attn_norm.weight");
            if (nw) {
                float nw_h[4];
                cudaMemcpy(nw_h, (char *)gpu->d_resident + nw->offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[DEBUG L0] attn_norm.weight[0..3]=%.6f %.6f %.6f %.6f (type=%d)\n",
                        nw_h[0], nw_h[1], nw_h[2], nw_h[3], nw->type_id);
            }
            layer0_dbg++;
        }
    }

    // ── 2. Q/K/V projections (GPU dequant matvec) ──
    snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", layer);
    TensorEntry *wq = tensor_get(ctx, tname);
    if (wq) matvec((char *)ctx->gpus[gpu_idx].d_resident + wq->offset,
                    gpu->d_normed, gpu->d_q, NH * HD, H, wq->type_id, cs);

    snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", layer);
    TensorEntry *wk = tensor_get(ctx, tname);
    if (wk) matvec((char *)ctx->gpus[gpu_idx].d_resident + wk->offset,
                    gpu->d_normed, gpu->d_k, NKV * HD, H, wk->type_id, cs);

    snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", layer);
    TensorEntry *wv = tensor_get(ctx, tname);
    if (wv) matvec((char *)ctx->gpus[gpu_idx].d_resident + wv->offset,
                    gpu->d_normed, gpu->d_v, NKV * HD, H, wv->type_id, cs);

    // Debug: check Q/K/V after projection (layer 0 only)
    if (0) {
        static int qkv_dbg = 0;
        if (qkv_dbg < 1) {
            cudaStreamSynchronize(cs);
            float dbg[4];
            cudaMemcpy(dbg, gpu->d_q, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L0] Q[0..3]=%.6f %.6f %.6f %.6f (type=%d)\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], wq ? wq->type_id : -1);
            cudaMemcpy(dbg, gpu->d_k, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L0] K[0..3]=%.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            cudaMemcpy(dbg, gpu->d_v, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L0] V[0..3]=%.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            qkv_dbg++;
        }
    }

    // ── 3. QK norms (optional, Qwen3 uses them) ──
    snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", layer);
    TensorEntry *qn = tensor_get(ctx, tname);
    if (qn) {
        for (int h = 0; h < NH; h++)
            cuda_rms_norm(gpu->d_q + h * HD,
                         (float *)((char *)ctx->gpus[gpu_idx].d_resident + qn->offset),
                         gpu->d_q + h * HD, HD, eps, cs);
    }

    snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", layer);
    TensorEntry *kn = tensor_get(ctx, tname);
    if (kn) {
        for (int h = 0; h < NKV; h++)
            cuda_rms_norm(gpu->d_k + h * HD,
                         (float *)((char *)ctx->gpus[gpu_idx].d_resident + kn->offset),
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

    if (0) {
        static int attn_dbg = 0;
        if (attn_dbg < 1) {
            cudaStreamSynchronize(cs);
            float dbg[4];
            cudaMemcpy(dbg, gpu->d_attn_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L0] attn_out[0..3]=%.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            attn_dbg++;
        }
    }

    // ── 7. Output projection + residual (GPU) ──
    snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", layer);
    TensorEntry *wo = tensor_get(ctx, tname);
    if (wo) matvec((char *)ctx->gpus[gpu_idx].d_resident + wo->offset,
                    gpu->d_attn_out, gpu->d_hidden, H, NH * HD, wo->type_id, cs);

    cuda_residual_add(gpu->d_hidden, gpu->d_residual, gpu->d_hidden, H, cs);

    if (0) {
        static int oproj_dbg = 0;
        if (oproj_dbg < 1) {
            cudaStreamSynchronize(cs);
            float dbg[4];
            cudaMemcpy(dbg, gpu->d_hidden, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L%d] post-attn hidden[0..3]=%.6f %.6f %.6f %.6f\n",
                    layer, dbg[0], dbg[1], dbg[2], dbg[3]);
            oproj_dbg++;
        }
    }

    // ── 8. Pre-FFN RMS norm ──
    snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", layer);
    void *ffn_norm_w = tensor_data_on_gpu(ctx, tname, gpu_idx);
    cuda_rms_norm(gpu->d_hidden, (float *)ffn_norm_w, gpu->d_normed, H, eps, cs);

    // Save residual for MoE skip connection
    cudaMemcpyAsync(gpu->d_residual, gpu->d_hidden, H * sizeof(float),
                    cudaMemcpyDeviceToDevice, cs);

    // ── 9. Router: F32 matvec on GPU ──
    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.weight", layer);
    TensorEntry *wr = tensor_get(ctx, tname);
    if (!wr) return;

    // Router is F32 column-major [H, NE] — use F32 matvec kernel on GPU
    matvec((char *)ctx->gpus[gpu_idx].d_resident + wr->offset,
           gpu->d_normed, gpu->d_router_out, NE, H, wr->type_id, cs);

    // ── 10. Top-K expert selection (GPU — no explicit stream sync) ──
    cuda_topk_softmax(gpu->d_router_out, gpu->d_expert_indices,
                      gpu->d_expert_weights, NE, K, cs);
    int expert_indices[16];
    float expert_weights_h[16];
    cudaMemcpy(expert_indices, gpu->d_expert_indices, K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(expert_weights_h, gpu->d_expert_weights, K * sizeof(float), cudaMemcpyDeviceToHost);

    if (0) {
        static int router_dbg = 0;
        if (router_dbg < 1) {
            float rout[4];
            cudaMemcpy(rout, gpu->d_router_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L0] router_out[0..3]=%.4f %.4f %.4f %.4f\n",
                    rout[0], rout[1], rout[2], rout[3]);
            fprintf(stderr, "[DEBUG L0] top-K experts: ");
            for (int i = 0; i < K; i++)
                fprintf(stderr, "%d(%.4f) ", expert_indices[i], expert_weights_h[i]);
            fprintf(stderr, "\n");
            router_dbg++;
        }
    }

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

        if (total_lookups % 5000 == 0)
            fprintf(stderr, "[MnemoCUDA] L1(VRAM):%d L2(prefetch):%d L3(RAM):%d L4(NVMe):%d / %d (%.0f%% hit)\n",
                    total_hits - total_prefetch_hits - total_ram_hits,
                    total_prefetch_hits, total_ram_hits,
                    total_lookups - total_hits,
                    total_lookups, 100.0 * total_hits / total_lookups);
    }

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

    // ── Load NVMe misses: mmap memcpy (page cache) or pread ──
    if (n_misses > 0) {
        IOTask io_tasks[16];
        for (int i = 0; i < n_misses; i++) {
            io_tasks[i].dst = (char *)gpu->h_expert_buf + (size_t)i * expert_size;
            io_tasks[i].size = expert_size;
            io_tasks[i].done = 0;
            if (elf->mmap_data) {
                io_tasks[i].fd = -1;
                io_tasks[i].src = (char *)elf->mmap_data + (size_t)miss_eids[i] * expert_size;
            } else {
                io_tasks[i].fd = elf->fd;
                io_tasks[i].src = NULL;
                io_tasks[i].offset = (off_t)miss_eids[i] * (off_t)expert_size;
            }
        }
        io_pool_submit(io_tasks, n_misses);

        // ── While I/O runs, compute ALL cache hits on GPU (free!) ──
        for (int h = 0; h < n_hits; h++) {
            void *d = hit_ptrs[h];
            matvec(d, gpu->d_normed, gpu->d_expert_gate, EFF, H, 12, cs);
            matvec((char *)d + gate_sz, gpu->d_normed, gpu->d_expert_up, EFF, H, 12, cs);
            cuda_swiglu(gpu->d_expert_gate, gpu->d_expert_up, gpu->d_expert_act, EFF, cs);
            matvec((char *)d + gate_sz + up_sz, gpu->d_expert_act, gpu->d_expert_out, H, EFF, down_type_id, cs);
            cuda_scaled_add(gpu->d_moe_out, gpu->d_expert_out, hit_weights[h], H, cs);
        }

        // ── Wait for I/O ──
        io_pool_wait(n_misses);

        // ── Pipeline: upload miss[i] + compute, overlap upload[i+1] with compute[i] ──
        // Use CUDA events for stream sync instead of blocking cudaStreamSynchronize
        cudaStreamSynchronize(cs); // ensure hits are done before touching cache slots

        for (int i = 0; i < n_misses; i++) {
            void *h_buf = (char *)gpu->h_expert_buf + (size_t)i * expert_size;

            // Insert NVMe miss into RAM cache (so next time it's a RAM hit, not NVMe)
            ram_cache_insert(&ctx->ram_cache, layer, miss_eids[i], h_buf, expert_size);

            // Upload to VRAM cache (with demotion of evicted expert to RAM)
            void *d = expert_cache_insert_ctx(ctx, gpu, layer, miss_eids[i], h_buf,
                                               expert_size, gpu->stream_io);
            if (!d) {
                cudaMemcpyAsync(gpu->d_expert_buf, h_buf, expert_size,
                                cudaMemcpyHostToDevice, gpu->stream_io);
                d = gpu->d_expert_buf;
            }

            // Use CUDA event to sync: compute stream waits for this upload
            // WITHOUT blocking the CPU (allows next upload to start immediately)
            cudaEvent_t upload_done;
            cudaEventCreateWithFlags(&upload_done, cudaEventDisableTiming);
            cudaEventRecord(upload_done, gpu->stream_io);
            cudaStreamWaitEvent(cs, upload_done, 0);
            cudaEventDestroy(upload_done);

            // Compute this miss on compute stream (will wait for upload via event)
            matvec(d, gpu->d_normed, gpu->d_expert_gate, EFF, H, 12, cs);
            matvec((char *)d + gate_sz, gpu->d_normed, gpu->d_expert_up, EFF, H, 12, cs);
            cuda_swiglu(gpu->d_expert_gate, gpu->d_expert_up, gpu->d_expert_act, EFF, cs);
            matvec((char *)d + gate_sz + up_sz, gpu->d_expert_act, gpu->d_expert_out, H, EFF, down_type_id, cs);
            cuda_scaled_add(gpu->d_moe_out, gpu->d_expert_out, miss_weights[i], H, cs);
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
            prefetch_submit(ctx, gpu, next_layer);
        }
    }

    // ── 13. MoE output + residual (GPU) ──
    cuda_residual_add(gpu->d_moe_out, gpu->d_residual, gpu->d_hidden, H, cs);

    if (0) {
        static int moe_dbg = 0;
        if (moe_dbg < 1) {
            cudaStreamSynchronize(cs);
            float dbg[4];
            cudaMemcpy(dbg, gpu->d_hidden, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L0] post-MoE hidden[0..3]=%.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            cudaMemcpy(dbg, gpu->d_moe_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[DEBUG L0] moe_out[0..3]=%.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            moe_dbg++;
        }
    }
}

// ── Full forward pass: embedding → layers → lm_head ──

static int forward_pass(MnemoCudaCtx *ctx, int token_id, int pos, float *h_logits) {
    ModelConfig *cfg = &ctx->config;
    int H = cfg->hidden_size;
    int V = cfg->vocab_size;

    // ── 1. Token embedding lookup ──
    TensorEntry *embd = tensor_get(ctx, "token_embd.weight");
    if (!embd) { fprintf(stderr, "[MnemoCUDA] Missing token_embd.weight\n"); return -1; }

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

    // Debug: verify embedding upload (disabled)
    static int fwd_dbg = 0;
    if (0 && fwd_dbg < 2) {
        cudaStreamSynchronize(gpu0->stream_compute);
        float dbg[4];
        cudaMemcpy(dbg, gpu0->d_hidden, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DEBUG] post-embed hidden[0..3]: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
    }

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

        // Debug: check hidden after transfer
        if (0) {
            static int xfer_dbg = 0;
            if (xfer_dbg < 2) {
                cudaStreamSynchronize(gpu->stream_compute);
                float dbg[4];
                cudaMemcpy(dbg, gpu->d_hidden, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[DEBUG] GPU%d after transfer hidden[0..3]=%.6f %.6f %.6f %.6f\n",
                        g, dbg[0], dbg[1], dbg[2], dbg[3]);
                xfer_dbg++;
            }
        }

        // Run all layers owned by this GPU
        for (int layer = gpu->layer_start; layer < gpu->layer_end; layer++) {
            forward_layer(ctx, layer, g, pos);

            // Debug: detect first NaN layer (pos=1 only)
            if (pos == 1 && g == 0) {
                static int nan_layer_found = 0;
                if (!nan_layer_found) {
                    cudaStreamSynchronize(gpu->stream_compute);
                    float chk[1];
                    cudaMemcpy(chk, gpu->d_hidden, sizeof(float), cudaMemcpyDeviceToHost);
                    if (chk[0] != chk[0]) {
                        fprintf(stderr, "[DEBUG] NaN first appears after layer %d\n", layer);
                        nan_layer_found = 1;
                    }
                }
            }
        }
    }

    // ── 3. Final RMS norm + LM head (on last GPU) ──
    int last_g = ctx->n_gpus - 1;
    GPUState *last_gpu = &ctx->gpus[last_g];
    cudaSetDevice(last_gpu->gpu_id);
    cudaStream_t cs = last_gpu->stream_compute;

    TensorEntry *out_norm = tensor_get(ctx, "output_norm.weight");
    if (out_norm) {
        void *norm_w = (char *)last_gpu->d_resident + out_norm->offset;
        cuda_rms_norm(last_gpu->d_hidden, (float *)norm_w, last_gpu->d_normed,
                      H, cfg->rms_norm_eps, cs);
    }

    TensorEntry *lm_head = tensor_get(ctx, "output.weight");
    if (lm_head && last_gpu->d_logits) {
        void *lm_w = (char *)last_gpu->d_resident + lm_head->offset;
        matvec(lm_w, last_gpu->d_normed, last_gpu->d_logits,
               V, H, lm_head->type_id, cs);
    }

    // Copy logits to host
    cudaStreamSynchronize(cs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[MnemoCUDA] CUDA error before logits copy: %s\n", cudaGetErrorString(err));
    cudaMemcpy(h_logits, last_gpu->d_logits, V * sizeof(float), cudaMemcpyDeviceToHost);

    // Debug: dump logit stats on first call
    static int dbg_count = 0;
    if (dbg_count < 2) {
        float lmin = h_logits[0], lmax = h_logits[0], lsum = 0;
        int nzero = 0, nnan = 0;
        for (int i = 0; i < V; i++) {
            if (h_logits[i] != h_logits[i]) { nnan++; continue; }
            if (h_logits[i] == 0.0f) nzero++;
            if (h_logits[i] < lmin) lmin = h_logits[i];
            if (h_logits[i] > lmax) lmax = h_logits[i];
            lsum += h_logits[i];
        }
        int argmax = 0;
        for (int i = 1; i < V; i++) if (h_logits[i] > h_logits[argmax]) argmax = i;
        fprintf(stderr, "[DEBUG] pos=%d logits: min=%.4f max=%.4f mean=%.6f zeros=%d nans=%d argmax=%d (%.4f)\n",
                pos, lmin, lmax, lsum/V, nzero, nnan, argmax, h_logits[argmax]);

        // Also check hidden state before lm_head
        float h_normed_dbg[8];
        cudaMemcpy(h_normed_dbg, last_gpu->d_normed, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DEBUG] normed[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                h_normed_dbg[0], h_normed_dbg[1], h_normed_dbg[2], h_normed_dbg[3],
                h_normed_dbg[4], h_normed_dbg[5], h_normed_dbg[6], h_normed_dbg[7]);

        // Check hidden state on GPU 0 after first layer
        float h_hidden_dbg[8];
        cudaSetDevice(ctx->gpus[0].gpu_id);
        cudaMemcpy(h_hidden_dbg, ctx->gpus[0].d_hidden, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DEBUG] gpu0 hidden[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                h_hidden_dbg[0], h_hidden_dbg[1], h_hidden_dbg[2], h_hidden_dbg[3],
                h_hidden_dbg[4], h_hidden_dbg[5], h_hidden_dbg[6], h_hidden_dbg[7]);
        cudaSetDevice(last_gpu->gpu_id);
        dbg_count++;
    }

    return 0;
}

// ── Generate: tokenize → prefill → autoregressive decode ──

int mnemo_cuda_generate(MnemoCudaCtx *ctx, const char *prompt, int max_tokens,
                        float temperature, MnemoCudaTokenCB callback, void *userdata) {
    if (!ctx || !ctx->loaded) return -1;
    if (!ctx->tokenizer) { callback("Error: tokenizer not loaded", true, userdata); return -2; }

    ctx->cancelled = false;
    rng_seed((uint64_t)time(NULL));

    ModelConfig *cfg = &ctx->config;
    int V = cfg->vocab_size;

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // ── 1. Tokenize prompt ──
    // If prompt already starts with <|im_start|>, treat as raw ChatML (multi-turn)
    // Otherwise, wrap in ChatML single-turn format
    char *chatml;
    if (strncmp(prompt, "<|im_start|>", 12) == 0) {
        chatml = strdup(prompt); // raw ChatML from client
    } else {
        chatml = malloc(strlen(prompt) + 128);
        sprintf(chatml, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", prompt);
    }

    // Call Python tokenizer (subprocess) for 100% correct BPE encoding
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
             "python3 %s/../tokenize.py %s '%s'",
             __FILE__, ctx->model_dir,
             chatml); // Note: single quotes in prompt could break this

    // Safer approach: use popen and write to stdin
    char popen_cmd[2048];
    snprintf(popen_cmd, sizeof(popen_cmd),
             "python3 \"%s/tokenize.py\" \"%s\" --pipe",
             // Find tokenize.py relative to model_dir or current dir
             ".", ctx->model_dir);

    // Try to find tokenize.py
    const char *tok_script = NULL;
    char tok_path[1200];
    // Check next to the engine binary
    snprintf(tok_path, sizeof(tok_path), "%s/tokenize.py", ctx->model_dir);
    if (access(tok_path, F_OK) == 0) {
        tok_script = tok_path;
    } else {
        // Check in current directory
        if (access("tokenize.py", F_OK) == 0) {
            tok_script = "tokenize.py";
        } else {
            // Hardcoded fallback path
            tok_script = "/home/curly/jarvis/core/mnemo_cuda/tokenize.py";
        }
    }

    int *tokens = NULL;
    int n_tokens = 0;

    // Pipe chatml string to tokenize.py via stdin
    snprintf(cmd, sizeof(cmd), "printf '%%s' | python3 '%s' '%s' --pipe",
             tok_script, ctx->model_dir);

    // Use popen with explicit stdin write instead
    {
        int pipe_in[2], pipe_out[2];
        pipe(pipe_in);
        pipe(pipe_out);

        pid_t pid = fork();
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

        write(pipe_in[1], chatml, strlen(chatml));
        close(pipe_in[1]);

        // Read binary response: [4B n_tokens] [4B token_id]*n
        uint32_t n_tok;
        if (read(pipe_out[0], &n_tok, 4) == 4) {
            n_tokens = (int)n_tok;
            tokens = malloc(n_tokens * sizeof(int));
            read(pipe_out[0], tokens, n_tokens * sizeof(int));
        }
        close(pipe_out[0]);

        int status;
        waitpid(pid, &status, 0);
    }

    free(chatml);

    if (!tokens || n_tokens == 0) {
        // Fallback to built-in BPE tokenizer
        fprintf(stderr, "[MnemoCUDA] Python tokenizer failed, using built-in BPE\n");
        chatml = malloc(strlen(prompt) + 128);
        sprintf(chatml, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", prompt);
        int max_prompt_tokens = cfg->max_position_embeddings - max_tokens;
        if (max_prompt_tokens < 32) max_prompt_tokens = 32;
        tokens = malloc(max_prompt_tokens * sizeof(int));
        n_tokens = tokenizer_encode(ctx->tokenizer, chatml, tokens, max_prompt_tokens);
        free(chatml);
    }

    fprintf(stderr, "[MnemoCUDA] Prompt: %d tokens, generating up to %d\n", n_tokens, max_tokens);

    // Allocate host logits buffer
    float *h_logits = malloc(V * sizeof(float));

    // Debug: verify embedding for first token
    {
        TensorEntry *embd_dbg = tensor_get(ctx, "token_embd.weight");
        if (embd_dbg && n_tokens > 0) {
            int bv = 256, bbs = 144;
            if (embd_dbg->type_id == 11) { bbs = 110; }
            size_t rb = (size_t)((cfg->hidden_size + bv - 1) / bv) * bbs;
            size_t ro = (size_t)tokens[0] * rb;
            const uint8_t *raw = (const uint8_t *)ctx->h_resident_mmap + embd_dbg->offset + ro;
            // debug disabled
        }
    }

    // Debug: print first few token IDs
    // fprintf(stderr, "[MnemoCUDA] First tokens: ");
    for (int i = 0; i < n_tokens && i < 12; i++) fprintf(stderr, "%d ", tokens[i]);
    fprintf(stderr, "\n");

    // ── 2. Prefill: process all prompt tokens ──
    ctx->kv_pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (ctx->cancelled) { free(tokens); free(h_logits); return -3; }
        int rc = forward_pass(ctx, tokens[i], ctx->kv_pos, h_logits);
        if (rc != 0) { free(tokens); free(h_logits); return rc; }
        ctx->kv_pos++;
    }

    // ── 3. Autoregressive generation ──
    int tokens_generated = 0;
    struct timespec t_gen_start;
    clock_gettime(CLOCK_MONOTONIC, &t_gen_start);

    for (int t = 0; t < max_tokens; t++) {
        if (ctx->cancelled) break;

        // Sample next token
        int next_id = sample_token(h_logits, V, temperature, 0.9f);

        // Check for EOS
        if (next_id == ctx->tokenizer->eos_id ||
            next_id == ctx->tokenizer->im_end_id) {
            fprintf(stderr, "[MnemoCUDA] EOS token %d at position %d\n", next_id, t);
            break;
        }
        // if (t < 5) fprintf(stderr, "[MnemoCUDA] token[%d] = %d '%s'\n",
        //                     t, next_id, tokenizer_decode(ctx->tokenizer, next_id));

        // Decode and emit (with UTF-8 accumulation for multi-byte chars)
        {
            const char *text = tokenizer_decode(ctx->tokenizer, next_id);
            static char utf8_buf[32];
            static int utf8_len = 0;

            for (int ci = 0; text[ci]; ci++) {
                utf8_buf[utf8_len++] = text[ci];
                utf8_buf[utf8_len] = '\0';

                // Check if we have a complete UTF-8 sequence
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

        // Forward pass for next token
        int rc = forward_pass(ctx, next_id, ctx->kv_pos, h_logits);
        if (rc != 0) break;
        ctx->kv_pos++;
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

    ctx->stats.tokens_generated = tokens_generated;
    ctx->stats.tokens_per_second = gen_secs > 0 ? tokens_generated / gen_secs : 0;
    ctx->stats.n_gpus_active = ctx->n_gpus;
    ctx->heat_total_tokens += (uint64_t)(n_tokens + tokens_generated);

    // Auto-pin hot experts after enough data (100 tokens warmup)
    if (!ctx->heat_pinning_active && ctx->heat_total_tokens >= 100)
        mnemo_cuda_heat_pin(ctx);

    // Calculate VRAM used
    size_t vram = 0;
    for (int g = 0; g < ctx->n_gpus; g++) {
        vram += ctx->gpus[g].resident_size;
        vram += ctx->gpus[g].expert_buf_size;
    }
    ctx->stats.vram_used_bytes = vram;
    ctx->stats.resident_size_bytes = ctx->resident_size;

    fprintf(stderr, "[MnemoCUDA] Done: %d tokens in %.1fs (%.1f tok/s, prefill %.1fs)\n",
            tokens_generated, total_secs, ctx->stats.tokens_per_second,
            (t_gen_start.tv_sec - t_start.tv_sec) +
            (t_gen_start.tv_nsec - t_start.tv_nsec) / 1e9);

    free(tokens);
    free(h_logits);
    return 0;
}

MnemoCudaStats mnemo_cuda_get_stats(MnemoCudaCtx *ctx) {
    return ctx ? ctx->stats : (MnemoCudaStats){0};
}

const char *mnemo_cuda_get_info(MnemoCudaCtx *ctx) {
    return ctx ? ctx->info : "Not loaded";
}

// ── Expert Heat Profiling ──

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
        fprintf(stderr, "[MnemoCUDA] GPU %d: pinned %d/%d hot experts (%.0f%% of cache)\n",
                gpu->gpu_id, pinned, gpu->expert_cache_slots,
                100.0 * pinned / gpu->expert_cache_slots);
    }

    ctx->heat_pinning_active = true;
    fprintf(stderr, "[MnemoCUDA] Heat pinning active: %d experts pinned across %d GPUs "
            "(based on %lu tokens of profiling)\n",
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
        fprintf(stderr, "[MnemoCUDA] Failed to save heat map: %s\n", heat_path);
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

    fprintf(stderr, "[MnemoCUDA] Heat map saved: %s (%lu tokens)\n",
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
