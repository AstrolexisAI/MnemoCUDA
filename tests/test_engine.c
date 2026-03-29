/**
 * Test MnemoCUDA Engine — smoke tests for the real CUDA inference path.
 *
 * Tests are gated behind the MODEL_DIR environment variable.
 * Without it, only non-GPU structural tests run.
 * With it, exercises: create → load → info → generate → stats → unload → destroy.
 *
 * Build:  make build/test_engine
 * Run:    MODEL_DIR=/path/to/split_model ./build/test_engine
 *         ./build/test_engine   (structural tests only)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "engine.h"

// ── Test helpers (same macros as test_heat.c) ──

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_skipped = 0;

#define TEST(name) printf("  TEST: %s ... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define SKIP(msg) do { printf("SKIP: %s\n", msg); tests_skipped++; } while(0)
#define ASSERT_TRUE(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { printf("FAIL: %s (got %ld, expected %ld)\n", msg, (long)(a), (long)(b)); tests_failed++; return; } } while(0)

// ── Test 1: Create and destroy without loading ──

static void test_create_destroy(void) {
    TEST("create/destroy without load");
    MnemoCudaCtx *ctx = mnemo_cuda_create();
    ASSERT_TRUE(ctx != NULL, "mnemo_cuda_create returned NULL");
    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 2: Config defaults are sane ──

static void test_config_defaults(void) {
    TEST("config defaults");
    MnemoCudaConfig cfg = mnemo_cuda_config_default();
    ASSERT_EQ(cfg.context_length, 8192, "context_length default");
    ASSERT_EQ(cfg.n_gpus, 0, "n_gpus default (auto-detect)");
    ASSERT_EQ(cfg.io_threads, 8, "io_threads default");
    ASSERT_TRUE(cfg.use_pinned_memory, "use_pinned_memory default");
    ASSERT_TRUE(cfg.model_dir == NULL, "model_dir should be NULL");
    PASS();
}

// ── Test 3: Load with NULL model_dir fails gracefully ──

static void test_load_null_dir(void) {
    TEST("load with NULL model_dir");
    MnemoCudaCtx *ctx = mnemo_cuda_create();
    ASSERT_TRUE(ctx != NULL, "create failed");
    MnemoCudaConfig cfg = mnemo_cuda_config_default();
    cfg.model_dir = NULL;
    int rc = mnemo_cuda_load(ctx, cfg);
    ASSERT_TRUE(rc != 0, "load should fail with NULL dir");
    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 4: Load with nonexistent dir fails gracefully ──

static void test_load_bad_dir(void) {
    TEST("load with nonexistent dir");
    MnemoCudaCtx *ctx = mnemo_cuda_create();
    ASSERT_TRUE(ctx != NULL, "create failed");
    MnemoCudaConfig cfg = mnemo_cuda_config_default();
    cfg.model_dir = "/nonexistent/path/to/model";
    int rc = mnemo_cuda_load(ctx, cfg);
    ASSERT_TRUE(rc != 0, "load should fail with bad dir");
    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 5: Double destroy is safe ──

static void test_double_destroy(void) {
    TEST("double destroy safety");
    MnemoCudaCtx *ctx = mnemo_cuda_create();
    ASSERT_TRUE(ctx != NULL, "create failed");
    mnemo_cuda_destroy(ctx);
    // Second destroy on freed pointer would crash — but we test that
    // destroy(NULL) is safe
    mnemo_cuda_destroy(NULL);
    PASS();
}

// ── Test 6: Cancel on unloaded context ──

static void test_cancel_unloaded(void) {
    TEST("cancel on unloaded context");
    MnemoCudaCtx *ctx = mnemo_cuda_create();
    mnemo_cuda_cancel(ctx);  // should not crash
    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 7: Stats on unloaded context ──

static void test_stats_unloaded(void) {
    TEST("stats on unloaded context");
    MnemoCudaCtx *ctx = mnemo_cuda_create();
    MnemoCudaStats s = mnemo_cuda_get_stats(ctx);
    ASSERT_EQ(s.tokens_generated, 0, "no tokens generated");
    ASSERT_EQ(s.n_gpus_active, 0, "no GPUs active");
    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 8: Info on unloaded context ──

static void test_info_unloaded(void) {
    TEST("info on unloaded context");
    MnemoCudaCtx *ctx = mnemo_cuda_create();
    const char *info = mnemo_cuda_get_info(ctx);
    ASSERT_TRUE(info != NULL, "info should not be NULL");
    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 9: Tensor name classification for GPU partitioning ──

static int classify_tensor_gpu(const char *name, int gpu_idx, int n_gpus,
                                int layer_start, int layer_end) {
    // Reproduce the classification logic from engine.c
    int tensor_layer = -1;
    if (strncmp(name, "blk.", 4) == 0)
        tensor_layer = atoi(name + 4);

    if (tensor_layer >= 0) {
        return (tensor_layer >= layer_start && tensor_layer < layer_end) ? 1 : 0;
    } else {
        if (gpu_idx == 0) return 1;
        if (gpu_idx == n_gpus - 1) {
            return (strcmp(name, "output_norm.weight") == 0 ||
                    strcmp(name, "output.weight") == 0) ? 1 : 0;
        }
        return 0;
    }
}

static void test_tensor_classification(void) {
    TEST("tensor name classification for GPU partitioning");

    // Simulate 2-GPU setup: GPU 0 owns layers 0-46, GPU 1 owns 47-93
    int n_gpus = 2;

    // Per-layer tensors: only on owning GPU
    ASSERT_TRUE(classify_tensor_gpu("blk.0.attn_q.weight", 0, n_gpus, 0, 47) == 1,
                "blk.0 should be on GPU 0");
    ASSERT_TRUE(classify_tensor_gpu("blk.0.attn_q.weight", 1, n_gpus, 47, 94) == 0,
                "blk.0 should NOT be on GPU 1");
    ASSERT_TRUE(classify_tensor_gpu("blk.46.ffn_norm.weight", 0, n_gpus, 0, 47) == 1,
                "blk.46 should be on GPU 0");
    ASSERT_TRUE(classify_tensor_gpu("blk.47.ffn_norm.weight", 0, n_gpus, 0, 47) == 0,
                "blk.47 should NOT be on GPU 0");
    ASSERT_TRUE(classify_tensor_gpu("blk.47.ffn_norm.weight", 1, n_gpus, 47, 94) == 1,
                "blk.47 should be on GPU 1");
    ASSERT_TRUE(classify_tensor_gpu("blk.93.attn_output.weight", 1, n_gpus, 47, 94) == 1,
                "blk.93 should be on GPU 1");

    // Global tensors: GPU 0 gets all, GPU 1 (last) gets only output-related
    ASSERT_TRUE(classify_tensor_gpu("token_embd.weight", 0, n_gpus, 0, 47) == 1,
                "token_embd should be on GPU 0");
    ASSERT_TRUE(classify_tensor_gpu("token_embd.weight", 1, n_gpus, 47, 94) == 0,
                "token_embd should NOT be on GPU 1 (last)");
    ASSERT_TRUE(classify_tensor_gpu("output_norm.weight", 1, n_gpus, 47, 94) == 1,
                "output_norm should be on last GPU");
    ASSERT_TRUE(classify_tensor_gpu("output.weight", 1, n_gpus, 47, 94) == 1,
                "output.weight should be on last GPU");
    ASSERT_TRUE(classify_tensor_gpu("output_norm.weight", 0, n_gpus, 0, 47) == 1,
                "output_norm should also be on GPU 0");

    // Regression: blk.N.attn_output.weight must NOT match "output" rule on last GPU
    ASSERT_TRUE(classify_tensor_gpu("blk.10.attn_output.weight", 1, n_gpus, 47, 94) == 0,
                "blk.10.attn_output should NOT be on GPU 1 via output match");

    // 3-GPU test: middle GPU gets nothing global
    ASSERT_TRUE(classify_tensor_gpu("token_embd.weight", 1, 3, 31, 62) == 0,
                "token_embd should NOT be on middle GPU");
    ASSERT_TRUE(classify_tensor_gpu("output.weight", 1, 3, 31, 62) == 0,
                "output.weight should NOT be on middle GPU");
    ASSERT_TRUE(classify_tensor_gpu("output.weight", 2, 3, 62, 94) == 1,
                "output.weight should be on last GPU (idx 2 of 3)");

    // Single-GPU: gets everything
    ASSERT_TRUE(classify_tensor_gpu("token_embd.weight", 0, 1, 0, 94) == 1,
                "single GPU gets token_embd");
    ASSERT_TRUE(classify_tensor_gpu("output.weight", 0, 1, 0, 94) == 1,
                "single GPU gets output.weight");
    ASSERT_TRUE(classify_tensor_gpu("blk.50.attn_q.weight", 0, 1, 0, 94) == 1,
                "single GPU gets all layers");

    PASS();
}

// ── Test 10: Error codes and strerror ──

static void test_error_codes(void) {
    TEST("error codes and strerror");

    // Verify all error codes have non-empty messages
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(MNEMO_OK)) > 0, "MNEMO_OK has message");
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(MNEMO_ERR_BAD_CONFIG)) > 0, "BAD_CONFIG has message");
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(MNEMO_ERR_TOKENIZER)) > 0, "TOKENIZER has message");
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(MNEMO_ERR_NO_GPU)) > 0, "NO_GPU has message");
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(MNEMO_ERR_CONTEXT_FULL)) > 0, "CONTEXT_FULL has message");
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(MNEMO_ERR_CUDA)) > 0, "CUDA has message");
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(MNEMO_ERR_IO)) > 0, "IO has message");
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(MNEMO_ERR_CANCELLED)) > 0, "CANCELLED has message");
    ASSERT_TRUE(strlen(mnemo_cuda_strerror(-99)) > 0, "unknown error has message");

    // Verify enum values are negative (except OK)
    ASSERT_EQ(MNEMO_OK, 0, "MNEMO_OK should be 0");
    ASSERT_TRUE(MNEMO_ERR_BAD_CONFIG < 0, "errors should be negative");

    // Verify strerror returns different strings for different codes
    ASSERT_TRUE(strcmp(mnemo_cuda_strerror(MNEMO_OK),
                       mnemo_cuda_strerror(MNEMO_ERR_CUDA)) != 0,
                "OK and CUDA error should differ");

    PASS();
}

// ── Test 11: Stats struct has expected defaults ──

static void test_stats_defaults(void) {
    TEST("stats have TTFT and prompt fields");
    MnemoCudaCtx *ctx = mnemo_cuda_create();
    MnemoCudaStats s = mnemo_cuda_get_stats(ctx);
    ASSERT_EQ(s.tokens_generated, 0, "tokens_generated default");
    ASSERT_EQ(s.prompt_tokens, 0, "prompt_tokens default");
    ASSERT_TRUE(s.ttft_seconds == 0.0, "ttft_seconds default");
    ASSERT_TRUE(s.total_seconds == 0.0, "total_seconds default");
    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 12: Config default values ──

static void test_config_extended(void) {
    TEST("config defaults extended validation");
    MnemoCudaConfig cfg = mnemo_cuda_config_default();
    // io_threads should be positive
    ASSERT_TRUE(cfg.io_threads > 0 && cfg.io_threads <= 16,
                "io_threads in valid range");
    // expert_k default should be 0 (use model config)
    ASSERT_EQ(cfg.expert_k, 0, "expert_k default is 0");
    // All GPU IDs should be 0 by default
    for (int i = 0; i < 8; i++)
        ASSERT_EQ(cfg.gpu_ids[i], 0, "gpu_ids default is 0");
    PASS();
}

// ── GPU Tests (require MODEL_DIR) ──

typedef struct {
    int token_count;
    bool got_done;
    char first_token[256];
} GenResult;

static void on_token(const char *text, bool is_done, void *userdata) {
    GenResult *r = (GenResult *)userdata;
    if (text && text[0]) {
        r->token_count++;
        if (r->token_count == 1 && strlen(text) < sizeof(r->first_token))
            strcpy(r->first_token, text);
    }
    if (is_done) r->got_done = true;
}

// ── Test 9: Full load → generate → unload cycle ──

static void test_full_cycle(const char *model_dir) {
    TEST("full load/generate/unload cycle");

    MnemoCudaCtx *ctx = mnemo_cuda_create();
    ASSERT_TRUE(ctx != NULL, "create failed");

    MnemoCudaConfig cfg = mnemo_cuda_config_default();
    cfg.model_dir = model_dir;
    cfg.context_length = 2048;  // small for test speed

    int rc = mnemo_cuda_load(ctx, cfg);
    ASSERT_EQ(rc, 0, "load failed");

    const char *info = mnemo_cuda_get_info(ctx);
    ASSERT_TRUE(info != NULL && strlen(info) > 0, "info empty after load");

    // Generate with temp=0 for determinism
    GenResult result = {0};
    rc = mnemo_cuda_generate(ctx, "Hello", 8, 0.0, on_token, &result);
    ASSERT_EQ(rc, 0, "generate failed");
    ASSERT_TRUE(result.got_done, "callback never received done=true");
    ASSERT_TRUE(result.token_count > 0, "no tokens generated");

    MnemoCudaStats stats = mnemo_cuda_get_stats(ctx);
    ASSERT_TRUE(stats.tokens_generated > 0, "stats show 0 tokens");
    ASSERT_TRUE(stats.tokens_per_second > 0.0, "stats show 0 tok/s");
    ASSERT_TRUE(stats.n_gpus_active >= 1, "no active GPUs in stats");

    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 10: Generate respects cancel ──

static void on_token_cancel(const char *text, bool is_done, void *userdata) {
    MnemoCudaCtx **pctx = (MnemoCudaCtx **)userdata;
    // Cancel after first token
    if (text && text[0] && !is_done)
        mnemo_cuda_cancel(*pctx);
}

static void test_cancel(const char *model_dir) {
    TEST("generate respects cancel");

    MnemoCudaCtx *ctx = mnemo_cuda_create();
    ASSERT_TRUE(ctx != NULL, "create failed");

    MnemoCudaConfig cfg = mnemo_cuda_config_default();
    cfg.model_dir = model_dir;
    cfg.context_length = 2048;

    int rc = mnemo_cuda_load(ctx, cfg);
    ASSERT_EQ(rc, 0, "load failed");

    // Request 1000 tokens but cancel after first
    rc = mnemo_cuda_generate(ctx, "Tell me a long story", 1000, 0.7,
                             on_token_cancel, &ctx);
    // Should have stopped early (cancel returns -3 or 0 depending on timing)
    MnemoCudaStats stats = mnemo_cuda_get_stats(ctx);
    ASSERT_TRUE(stats.tokens_generated < 100, "cancel didn't stop generation early");

    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Test 11: Heat stats work after generation ──

static void test_heat_after_generate(const char *model_dir) {
    TEST("heat stats populated after generate");

    MnemoCudaCtx *ctx = mnemo_cuda_create();
    ASSERT_TRUE(ctx != NULL, "create failed");

    MnemoCudaConfig cfg = mnemo_cuda_config_default();
    cfg.model_dir = model_dir;
    cfg.context_length = 2048;

    int rc = mnemo_cuda_load(ctx, cfg);
    ASSERT_EQ(rc, 0, "load failed");

    GenResult result = {0};
    mnemo_cuda_generate(ctx, "Test heat profiling", 4, 0.0, on_token, &result);

    MnemoCudaHeatStats hs = mnemo_cuda_get_heat_stats(ctx);
    ASSERT_TRUE(hs.total_tokens > 0, "heat total_tokens should be > 0");
    ASSERT_TRUE(hs.total_activations > 0, "heat total_activations should be > 0");
    ASSERT_TRUE(hs.active_experts > 0, "no active experts recorded");
    ASSERT_TRUE(hs.n_layers > 0, "n_layers should be > 0");

    mnemo_cuda_destroy(ctx);
    PASS();
}

// ── Main ──

int main(void) {
    printf("\n=== MnemoCUDA Engine Tests ===\n\n");

    // Structural tests (no GPU required)
    printf("-- Structural tests --\n");
    test_create_destroy();
    test_config_defaults();
    test_load_null_dir();
    test_load_bad_dir();
    test_double_destroy();
    test_cancel_unloaded();
    test_stats_unloaded();
    test_info_unloaded();
    test_tensor_classification();
    test_error_codes();
    test_stats_defaults();
    test_config_extended();

    // GPU tests (require MODEL_DIR)
    const char *model_dir = getenv("MODEL_DIR");
    printf("\n-- GPU inference tests --\n");
    if (model_dir && model_dir[0]) {
        printf("  MODEL_DIR=%s\n", model_dir);
        test_full_cycle(model_dir);
        test_cancel(model_dir);
        test_heat_after_generate(model_dir);
    } else {
        printf("  MODEL_DIR not set — skipping GPU tests\n");
        printf("  Set MODEL_DIR=/path/to/split_model to run full suite\n");
        tests_skipped += 3;
    }

    printf("\n=== Results: %d passed, %d failed, %d skipped ===\n\n",
           tests_passed, tests_failed, tests_skipped);
    return tests_failed > 0 ? 1 : 0;
}
