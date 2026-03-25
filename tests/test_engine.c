/**
 * Test MnemoCUDA engine — load a split model and run basic inference.
 *
 * Usage: ./test_engine /path/to/split_model/
 */

#include <stdio.h>
#include <stdlib.h>
#include "engine.h"

static void on_token(const char *text, bool is_done, void *userdata) {
    if (text && text[0]) printf("%s", text);
    if (is_done) printf("\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir> [prompt]\n", argv[0]);
        return 1;
    }

    const char *model_dir = argv[1];
    const char *prompt = argc > 2 ? argv[2] : "Hola, ¿cómo estás?";

    printf("Loading model from %s...\n", model_dir);

    MnemoCudaCtx *ctx = mnemo_cuda_create();
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    MnemoCudaConfig config = mnemo_cuda_config_default();
    config.model_dir = model_dir;
    config.context_length = 2048;

    int result = mnemo_cuda_load(ctx, config);
    if (result != 0) {
        fprintf(stderr, "Load failed: %d\n", result);
        mnemo_cuda_destroy(ctx);
        return 1;
    }

    printf("Model: %s\n", mnemo_cuda_get_info(ctx));
    printf("Prompt: %s\n", prompt);
    printf("Response: ");

    mnemo_cuda_generate(ctx, prompt, 256, 0.7, on_token, NULL);

    MnemoCudaStats stats = mnemo_cuda_get_stats(ctx);
    printf("Stats: %d tokens, %.1f tok/s\n",
           stats.tokens_generated, stats.tokens_per_second);

    mnemo_cuda_destroy(ctx);
    return 0;
}
