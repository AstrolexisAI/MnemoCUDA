/**
 * MnemoCUDA Tokenizer — BPE tokenizer with special token support.
 *
 * Loads a binary tokenizer format (tokenizer.bin) produced by prep_tokenizer.py.
 * Supports both internal BPE encoding and delegation to external Python tokenizer.
 */

#ifndef MNEMO_TOKENIZER_H
#define MNEMO_TOKENIZER_H

#include <stdbool.h>
#include <stdint.h>

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

    // Hash table for O(1) vocab lookup (replaces linear scan)
    int *ht_ids;            // token IDs (-1 = empty slot)
    uint32_t *ht_hashes;    // stored hashes for collision check
    int ht_cap;             // capacity (power of 2)
} Tokenizer;

#ifdef __cplusplus
extern "C" {
#endif

Tokenizer *tokenizer_load(const char *model_dir);
void tokenizer_free(Tokenizer *tok);

int tokenizer_encode(const Tokenizer *tok, const char *text,
                     int *out_ids, int max_ids);
const char *tokenizer_decode(const Tokenizer *tok, int id);

#ifdef __cplusplus
}
#endif

#endif
