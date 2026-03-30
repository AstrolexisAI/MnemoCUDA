/**
 * MnemoCUDA JSON Helpers — Context-aware extraction for flat JSON objects.
 */

#include "json_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Skip a JSON string starting at p (pointing to opening '"').
// Returns pointer past the closing '"', or NULL on unterminated string.
static const char *json_skip_string(const char *p) {
    if (*p != '"') return NULL;
    p++;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p + 1)) p++;
        p++;
    }
    return *p == '"' ? p + 1 : NULL;
}

// Find the value position for a top-level key in a JSON object.
// Returns pointer to the first non-whitespace character of the value,
// or NULL if the key is not found at the top level.
static const char *json_find_top_level_key(const char *json, const char *key) {
    const char *p = json;
    int depth = 0;
    int keylen = strlen(key);

    while (*p) {
        if (*p == '{' || *p == '[') {
            depth++;
            p++;
        } else if (*p == '}' || *p == ']') {
            depth--;
            if (depth <= 0) break;
            p++;
        } else if (*p == '"') {
            if (depth == 1) {
                if (strncmp(p + 1, key, keylen) == 0 && p[1 + keylen] == '"') {
                    const char *after_key = p + 1 + keylen + 1;
                    while (*after_key == ' ' || *after_key == '\t') after_key++;
                    if (*after_key == ':') {
                        after_key++;
                        while (*after_key == ' ' || *after_key == '\t') after_key++;
                        return after_key;
                    }
                }
            }
            p = json_skip_string(p);
            if (!p) return NULL;
        } else {
            p++;
        }
    }
    return NULL;
}

// Decode a JSON string value starting at p (pointing to opening '"').
static char *json_decode_string(const char *p) {
    if (*p != '"') return NULL;
    p++;

    int cap = strlen(p) + 1;
    char *out = malloc(cap);
    if (!out) return NULL;

    int i = 0;
    while (*p && *p != '"' && i < cap - 1) {
        if (*p == '\\' && *(p + 1)) {
            p++;
            switch (*p) {
                case '"':  out[i++] = '"'; break;
                case '\\': out[i++] = '\\'; break;
                case 'n':  out[i++] = '\n'; break;
                case 't':  out[i++] = '\t'; break;
                case 'r':  out[i++] = '\r'; break;
                case '/':  out[i++] = '/'; break;
                case 'u': {
                    unsigned cp = 0;
                    int j;
                    for (j = 0; j < 4 && p[1 + j]; j++) {
                        char c = p[1 + j];
                        cp = cp * 16 + (c >= 'a' ? c - 'a' + 10 :
                                        c >= 'A' ? c - 'A' + 10 : c - '0');
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
    return out;
}

char *json_extract_string(const char *json, const char *key) {
    const char *val = json_find_top_level_key(json, key);
    if (!val || *val != '"') return NULL;
    return json_decode_string(val);
}

int json_extract_int(const char *json, const char *key, int def) {
    const char *val = json_find_top_level_key(json, key);
    if (!val) return def;
    return atoi(val);
}

float json_extract_float(const char *json, const char *key, float def) {
    const char *val = json_find_top_level_key(json, key);
    if (!val) return def;
    return (float)atof(val);
}

int json_extract_bool(const char *json, const char *key, int def) {
    const char *val = json_find_top_level_key(json, key);
    if (!val) return def;
    if (strncmp(val, "true", 4) == 0) return 1;
    if (strncmp(val, "false", 5) == 0) return 0;
    return def;
}

char *json_escape(const char *src, int len) {
    int cap = len * 6 + 1;
    char *out = malloc(cap);
    if (!out) return NULL;
    int j = 0;
    for (int i = 0; i < len; i++) {
        switch (src[i]) {
            case '"':  out[j++] = '\\'; out[j++] = '"'; break;
            case '\\': out[j++] = '\\'; out[j++] = '\\'; break;
            case '\n': out[j++] = '\\'; out[j++] = 'n'; break;
            case '\t': out[j++] = '\\'; out[j++] = 't'; break;
            case '\r': out[j++] = '\\'; out[j++] = 'r'; break;
            case '\b': out[j++] = '\\'; out[j++] = 'b'; break;
            case '\f': out[j++] = '\\'; out[j++] = 'f'; break;
            default:
                if ((unsigned char)src[i] < 0x20) {
                    j += snprintf(out + j, 7, "\\u%04x", (unsigned char)src[i]);
                } else {
                    out[j++] = src[i];
                }
        }
    }
    out[j] = '\0';
    return out;
}
