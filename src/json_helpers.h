/**
 * MnemoCUDA JSON Helpers — Context-aware extraction for flat JSON objects.
 *
 * Only matches keys at the top level (depth 1), so keys embedded inside
 * string values (e.g. in prompts) are correctly ignored.
 */

#ifndef MNEMO_JSON_HELPERS_H
#define MNEMO_JSON_HELPERS_H

#ifdef __cplusplus
extern "C" {
#endif

// Extract a top-level JSON string value. Returns malloc'd string, or NULL.
char *json_extract_string(const char *json, const char *key);

// Extract a top-level JSON int value. Returns def if key not found.
int json_extract_int(const char *json, const char *key, int def);

// Extract a top-level JSON float value. Returns def if key not found.
float json_extract_float(const char *json, const char *key, float def);

// Extract a top-level JSON bool (true/false). Returns def if key not found.
int json_extract_bool(const char *json, const char *key, int def);

// Escape a string for JSON output. Returns malloc'd buffer.
char *json_escape(const char *src, int len);

#ifdef __cplusplus
}
#endif

#endif
