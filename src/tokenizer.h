/**
 * MnemoCUDA Tokenizer — BPE tokenizer with special token support.
 *
 * Loads a binary tokenizer format (tokenizer.bin) produced by prep_tokenizer.py.
 * Supports both internal BPE encoding and delegation to external Python tokenizer.
 */

#ifndef MNEMO_TOKENIZER_H
#define MNEMO_TOKENIZER_H

#include <stdbool.h>

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

Tokenizer *tokenizer_load(const char *model_dir);
void tokenizer_free(Tokenizer *tok);

int tokenizer_encode(Tokenizer *tok, const char *text,
                     int *out_ids, int max_ids);
const char *tokenizer_decode(Tokenizer *tok, int id);

#endif
