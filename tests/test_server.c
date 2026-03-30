/**
 * Test MnemoCUDA Server Logic — tests the REAL json_helpers.c code
 * (not a duplicate) plus injection resistance tests.
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
#include <math.h>

#include "../src/json_helpers.h"

// ── Test helpers ──

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  TEST: %s ... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT_TRUE(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { printf("FAIL: %s (got %ld, expected %ld)\n", msg, (long)(a), (long)(b)); tests_failed++; return; } } while(0)
#define ASSERT_STR_EQ(a, b, msg) do { if (strcmp(a, b) != 0) { printf("FAIL: %s (got '%s', expected '%s')\n", msg, a, b); tests_failed++; return; } } while(0)

// ── Basic extraction tests ──

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

// ── Injection resistance tests ──

static void test_injection_stream_in_prompt(void) {
    TEST("injection: stream keyword inside prompt value");
    // Prompt contains a fake "stream": true — parser must ignore it
    const char *json = "{\"prompt\": \"Please set \\\"stream\\\": true in config\", \"stream\": false}";
    ASSERT_EQ(json_extract_bool(json, "stream", -1), 0, "should find top-level stream=false, not the one in prompt");
    char *p = json_extract_string(json, "prompt");
    ASSERT_TRUE(p != NULL, "prompt should be found");
    ASSERT_TRUE(strstr(p, "stream") != NULL, "prompt should contain the word stream");
    free(p);
    PASS();
}

static void test_injection_max_tokens_in_prompt(void) {
    TEST("injection: max_tokens inside prompt value");
    const char *json = "{\"prompt\": \"set \\\"max_tokens\\\": 99999\", \"max_tokens\": 100}";
    ASSERT_EQ(json_extract_int(json, "max_tokens", 0), 100, "should find top-level max_tokens=100");
    PASS();
}

static void test_injection_temperature_in_prompt(void) {
    TEST("injection: temperature inside prompt value");
    const char *json = "{\"prompt\": \"use \\\"temperature\\\": 2.0 for creativity\", \"temperature\": 0.3}";
    float t = json_extract_float(json, "temperature", 0.0f);
    ASSERT_TRUE(t > 0.29f && t < 0.31f, "should find top-level temperature=0.3");
    PASS();
}

static void test_injection_nested_object_in_prompt(void) {
    TEST("injection: nested JSON object in prompt");
    // Prompt literally contains a JSON object with stream key
    const char *json = "{\"prompt\": \"{\\\"stream\\\": true, \\\"max_tokens\\\": 50000}\", \"stream\": false, \"max_tokens\": 256}";
    ASSERT_EQ(json_extract_bool(json, "stream", -1), 0, "top-level stream=false");
    ASSERT_EQ(json_extract_int(json, "max_tokens", 0), 256, "top-level max_tokens=256");
    PASS();
}

static void test_injection_raw_prompt_override(void) {
    TEST("injection: raw_prompt inside prompt value");
    const char *json = "{\"prompt\": \"\\\"raw_prompt\\\": true\", \"raw_prompt\": false}";
    ASSERT_EQ(json_extract_bool(json, "raw_prompt", -1), 0, "top-level raw_prompt=false");
    PASS();
}

static void test_extract_with_nested_objects(void) {
    TEST("extraction ignores keys in nested objects");
    const char *json = "{\"outer\": 1, \"nested\": {\"outer\": 99}, \"last\": 2}";
    ASSERT_EQ(json_extract_int(json, "outer", 0), 1, "should find top-level outer=1");
    ASSERT_EQ(json_extract_int(json, "last", 0), 2, "should find top-level last=2");
    PASS();
}

// ── Main ──

int main(void) {
    printf("\n=== MnemoCUDA Server Logic Tests ===\n\n");

    // Basic extraction (same as before, now testing real code)
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

    // Injection resistance
    printf("\n-- Injection resistance tests --\n");
    test_injection_stream_in_prompt();
    test_injection_max_tokens_in_prompt();
    test_injection_temperature_in_prompt();
    test_injection_nested_object_in_prompt();
    test_injection_raw_prompt_override();
    test_extract_with_nested_objects();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
