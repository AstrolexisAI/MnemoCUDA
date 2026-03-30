/**
 * MnemoCUDA Forward Pass — Layer computation, expert cache, and prefetch.
 *
 * Contains the forward_pass entry point (embedding → layers → lm_head)
 * and all supporting infrastructure: matvec dispatch, expert VRAM/RAM
 * cache, speculative prefetch, and the per-layer MoE pipeline.
 */

#ifndef MNEMO_FORWARD_H
#define MNEMO_FORWARD_H

#include "engine_internal.h"

// Full forward pass: token embedding → transformer layers → logits
int forward_pass(MnemoCudaCtx *ctx, int token_id, int pos, float *h_logits);

// Forward pass with GPU-side sampling: returns sampled token ID.
// Avoids copying full logits vector to host (4 bytes D2H instead of V*4).
int forward_pass_sample(MnemoCudaCtx *ctx, int token_id, int pos,
                        float temperature, float top_p, uint64_t rng_state,
                        int *out_token);

// Forward pass without lm_head: used during prefill (no logits needed).
int forward_pass_no_logits(MnemoCudaCtx *ctx, int token_id, int pos);

// Batched prefill: process multiple tokens sequentially without returning
// to host between tokens. Reduces function call overhead and keeps GPU busy.
int forward_prefill_batch(MnemoCudaCtx *ctx, const int *token_ids, int n_tokens, int start_pos);

#endif
