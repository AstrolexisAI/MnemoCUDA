/**
 * MnemoCUDA Tokenizer — BPE tokenizer implementation.
 */

#include "tokenizer.h"
#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TOK_MAX_VOCAB   500000
#define TOK_MAX_MERGES  500000
#define TOK_MAX_SPECIAL 10000
#define TOK_MAX_STRLEN  65535

#define TOK_READ(ptr, sz, n, fp) do { \
    if (fread(ptr, sz, n, fp) != (size_t)(n)) goto tok_fail; \
} while(0)

Tokenizer *tokenizer_load(const char *model_dir) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/tokenizer.bin", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        LOG_WARN("No tokenizer.bin — run prep_tokenizer.py first");
        return NULL;
    }

    Tokenizer *tok = NULL;

    // Read and validate header
    uint32_t magic, vocab_size, n_merges, n_special, eos_id, im_start_id, im_end_id;
    TOK_READ(&magic, 4, 1, f);
    if (magic != 0x4D544F4B) {
        LOG_ERROR("Bad tokenizer magic: 0x%08X", magic);
        fclose(f); return NULL;
    }
    TOK_READ(&vocab_size, 4, 1, f);
    TOK_READ(&n_merges, 4, 1, f);
    TOK_READ(&n_special, 4, 1, f);
    TOK_READ(&eos_id, 4, 1, f);
    TOK_READ(&im_start_id, 4, 1, f);
    TOK_READ(&im_end_id, 4, 1, f);

    // Sanity checks on sizes
    if (vocab_size > TOK_MAX_VOCAB || n_merges > TOK_MAX_MERGES || n_special > TOK_MAX_SPECIAL) {
        LOG_ERROR("Tokenizer sizes out of range: vocab=%u merges=%u special=%u",
                vocab_size, n_merges, n_special);
        fclose(f); return NULL;
    }

    tok = calloc(1, sizeof(Tokenizer));
    if (!tok) { fclose(f); return NULL; }
    tok->vocab_size = (int)vocab_size;
    tok->eos_id = (int)eos_id;
    tok->im_start_id = (int)im_start_id;
    tok->im_end_id = (int)im_end_id;

    // Read vocab
    tok->vocab = calloc(vocab_size, sizeof(char*));
    tok->vocab_len = calloc(vocab_size, sizeof(int));
    if (!tok->vocab || !tok->vocab_len) goto tok_fail;
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint16_t len;
        TOK_READ(&len, 2, 1, f);
        if (len > 0) {
            if (len > TOK_MAX_STRLEN) goto tok_fail;
            tok->vocab[i] = malloc(len + 1);
            if (!tok->vocab[i]) goto tok_fail;
            TOK_READ(tok->vocab[i], 1, len, f);
            tok->vocab[i][len] = '\0';
            tok->vocab_len[i] = len;
        }
    }

    // Read merges
    tok->n_merges = (int)n_merges;
    tok->merges = calloc(n_merges, sizeof(char*));
    if (!tok->merges) goto tok_fail;
    for (uint32_t i = 0; i < n_merges; i++) {
        uint16_t len;
        TOK_READ(&len, 2, 1, f);
        if (len > TOK_MAX_STRLEN) goto tok_fail;
        tok->merges[i] = malloc(len + 1);
        if (!tok->merges[i]) goto tok_fail;
        TOK_READ(tok->merges[i], 1, len, f);
        tok->merges[i][len] = '\0';
    }

    // Read special tokens
    tok->n_special = (int)n_special;
    tok->special_tokens = calloc(n_special, sizeof(char*));
    tok->special_ids = calloc(n_special, sizeof(int));
    if (!tok->special_tokens || !tok->special_ids) goto tok_fail;
    for (uint32_t i = 0; i < n_special; i++) {
        uint32_t sid;
        uint16_t len;
        TOK_READ(&sid, 4, 1, f);
        TOK_READ(&len, 2, 1, f);
        if (len > TOK_MAX_STRLEN) goto tok_fail;
        tok->special_ids[i] = (int)sid;
        tok->special_tokens[i] = malloc(len + 1);
        if (!tok->special_tokens[i]) goto tok_fail;
        TOK_READ(tok->special_tokens[i], 1, len, f);
        tok->special_tokens[i][len] = '\0';
    }

    fclose(f);
    LOG_INFO("Tokenizer: %d vocab, %d merges, %d special",
            tok->vocab_size, tok->n_merges, tok->n_special);
    return tok;

tok_fail:
    LOG_ERROR("Tokenizer load failed: corrupt or truncated tokenizer.bin");
    fclose(f);
    tokenizer_free(tok);
    return NULL;
}

void tokenizer_free(Tokenizer *tok) {
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
int tokenizer_encode(Tokenizer *tok, const char *text,
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
// Generated from GPT-2 bytes_to_unicode() reference implementation.
// Maps Unicode codepoints (used in BPE vocab) back to raw byte values.
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

const char *tokenizer_decode(Tokenizer *tok, int id) {
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
