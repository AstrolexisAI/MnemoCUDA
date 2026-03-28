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

#endif
