/**
 * Test MnemoCUDA Server Logic — structural tests for HTTP parsing,
 * JSON extraction, and escape functions.
 *
 * Runs WITHOUT GPU or model — tests server helper functions only.
 *
 * Build: make build/test_server
 * Run:   ./build/test_server
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// ── Test helpers ──

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  TEST: %s ... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT_TRUE(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { printf("FAIL: %s (got %ld, expected %ld)\n", msg, (long)(a), (long)(b)); tests_failed++; return; } } while(0)
#define ASSERT_STR_EQ(a, b, msg) do { if (strcmp(a, b) != 0) { printf("FAIL: %s (got '%s', expected '%s')\n", msg, a, b); tests_failed++; return; } } while(0)

// ── Re-implement server JSON helpers (no server dependency) ──

static char *json_extract_string(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
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
                    for (int j = 0; j < 4 && p[1 + j]; j++) {
                        char c = p[1 + j];
                        cp = cp * 16 + (c >= 'a' ? c - 'a' + 10 :
                                        c >= 'A' ? c - 'A' + 10 : c - '0');
                    }
                    p += 4;
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

static int json_extract_int(const char *json, const char *key, int def) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return def;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return atoi(p);
}

static float json_extract_float(const char *json, const char *key, float def) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return def;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return (float)atof(p);
}

static int json_extract_bool(const char *json, const char *key, int def) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return def;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def;
}

static char *json_escape(const char *src, int len) {
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

// ── Tests ──

static void test_json_extract_string_basic(void) {
    TEST("json_extract_string basic");
    char *s = json_extract_string("{\"prompt\": \"hello world\"}", "prompt");
    ASSERT_TRUE(s != NULL, "should find prompt");
    ASSERT_STR_EQ(s, "hello world", "value mismatch");
    free(s);
    PASS();
}

static void test_json_extract_string_escapes(void) {
    TEST("json_extract_string with escapes");
    char *s = json_extract_string("{\"text\": \"line1\\nline2\\t\\\"quoted\\\"\"}", "text");
    ASSERT_TRUE(s != NULL, "should find text");
    ASSERT_TRUE(strstr(s, "line1\nline2") != NULL, "should decode \\n");
    ASSERT_TRUE(strstr(s, "\t") != NULL, "should decode \\t");
    ASSERT_TRUE(strstr(s, "\"quoted\"") != NULL, "should decode \\\"");
    free(s);
    PASS();
}

static void test_json_extract_string_unicode(void) {
    TEST("json_extract_string with \\uXXXX");
    char *s = json_extract_string("{\"text\": \"caf\\u00e9\"}", "text");
    ASSERT_TRUE(s != NULL, "should find text");
    // \u00e9 = UTF-8 0xC3 0xA9
    ASSERT_TRUE(strlen(s) == 5, "should be 5 bytes (caf + 2-byte UTF-8)");
    free(s);
    PASS();
}

static void test_json_extract_string_missing(void) {
    TEST("json_extract_string missing key");
    char *s = json_extract_string("{\"other\": \"value\"}", "prompt");
    ASSERT_TRUE(s == NULL, "should return NULL for missing key");
    PASS();
}

static void test_json_extract_int(void) {
    TEST("json_extract_int");
    ASSERT_EQ(json_extract_int("{\"max_tokens\": 256}", "max_tokens", 0), 256, "should find 256");
    ASSERT_EQ(json_extract_int("{\"max_tokens\": 0}", "max_tokens", 99), 0, "should find 0");
    ASSERT_EQ(json_extract_int("{\"other\": 1}", "max_tokens", 42), 42, "should return default");
    PASS();
}

static void test_json_extract_float(void) {
    TEST("json_extract_float");
    float t = json_extract_float("{\"temperature\": 0.7}", "temperature", 1.0f);
    ASSERT_TRUE(t > 0.69f && t < 0.71f, "should find 0.7");
    t = json_extract_float("{\"other\": 1}", "temperature", 0.5f);
    ASSERT_TRUE(t > 0.49f && t < 0.51f, "should return default");
    PASS();
}

static void test_json_extract_bool(void) {
    TEST("json_extract_bool");
    ASSERT_EQ(json_extract_bool("{\"stream\": true}", "stream", 0), 1, "should find true");
    ASSERT_EQ(json_extract_bool("{\"stream\": false}", "stream", 1), 0, "should find false");
    ASSERT_EQ(json_extract_bool("{\"other\": 1}", "stream", -1), -1, "should return default");
    // Spacing variants
    ASSERT_EQ(json_extract_bool("{\"stream\":true}", "stream", 0), 1, "no space true");
    ASSERT_EQ(json_extract_bool("{\"stream\" : false}", "stream", 1), 0, "extra space false");
    PASS();
}

static void test_json_escape_basic(void) {
    TEST("json_escape basic");
    char *e = json_escape("hello \"world\"\n", 14);
    ASSERT_TRUE(e != NULL, "should not be NULL");
    ASSERT_TRUE(strstr(e, "\\\"") != NULL, "should escape quotes");
    ASSERT_TRUE(strstr(e, "\\n") != NULL, "should escape newline");
    free(e);
    PASS();
}

static void test_json_escape_control_chars(void) {
    TEST("json_escape control chars");
    char input[] = {0x01, 0x02, 0x1f, 0x00};
    char *e = json_escape(input, 3);
    ASSERT_TRUE(e != NULL, "should not be NULL");
    ASSERT_TRUE(strstr(e, "\\u0001") != NULL, "should escape 0x01");
    ASSERT_TRUE(strstr(e, "\\u001f") != NULL, "should escape 0x1f");
    ASSERT_EQ((int)strlen(e), 18, "3 control chars * 6 bytes each");
    free(e);
    PASS();
}

static void test_json_escape_empty(void) {
    TEST("json_escape empty string");
    char *e = json_escape("", 0);
    ASSERT_TRUE(e != NULL, "should not be NULL");
    ASSERT_EQ((int)strlen(e), 0, "should be empty");
    free(e);
    PASS();
}

static void test_json_extract_multikey(void) {
    TEST("json_extract multiple keys from same object");
    const char *json = "{\"prompt\":\"hi\",\"max_tokens\":100,\"temperature\":0.5,\"stream\":true}";
    char *p = json_extract_string(json, "prompt");
    ASSERT_TRUE(p != NULL, "prompt found");
    ASSERT_STR_EQ(p, "hi", "prompt value");
    free(p);
    ASSERT_EQ(json_extract_int(json, "max_tokens", 0), 100, "max_tokens");
    float t = json_extract_float(json, "temperature", 0.0f);
    ASSERT_TRUE(t > 0.49f && t < 0.51f, "temperature");
    ASSERT_EQ(json_extract_bool(json, "stream", 0), 1, "stream");
    PASS();
}

// ── Main ──

int main(void) {
    printf("\n=== MnemoCUDA Server Logic Tests ===\n\n");

    test_json_extract_string_basic();
    test_json_extract_string_escapes();
    test_json_extract_string_unicode();
    test_json_extract_string_missing();
    test_json_extract_int();
    test_json_extract_float();
    test_json_extract_bool();
    test_json_escape_basic();
    test_json_escape_control_chars();
    test_json_escape_empty();
    test_json_extract_multikey();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
