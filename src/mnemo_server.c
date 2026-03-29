/**
 * MnemoCUDA Server — Persistent inference server with warm expert cache.
 *
 * Modes:
 *   REPL:   ./mnemo_server <model_dir> --repl
 *   HTTP:   ./mnemo_server <model_dir> --http 8095
 *   Single: ./mnemo_server <model_dir> "prompt"
 *
 * The model stays loaded between requests, so the expert VRAM cache
 * stays warm — subsequent requests are much faster than the first.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <signal.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h>
#include "engine.h"

static volatile int running = 1;

static void sigint_handler(int sig) {
    (void)sig;
    running = 0;
    fprintf(stderr, "\n[MnemoCUDA] Shutting down...\n");
}

// ── Token callback ──

typedef struct {
    int fd;         // socket fd for HTTP mode, -1 for stdout
    int streaming;  // SSE streaming
    char *buf;      // accumulation buffer
    int buf_len;
    int buf_cap;
} OutputCtx;

static char *json_escape(const char *src, int len);  // forward decl

static void on_token(const char *text, bool is_done, void *userdata) {
    OutputCtx *out = (OutputCtx *)userdata;

    if (text && text[0]) {
        if (out->fd == -2) {
            // Silent mode (warmup) — skip output, just accumulate below
        } else if (out->fd >= 0) {
            // HTTP SSE: data: {"token": "..."}\n\n
            char *escaped = json_escape(text, strlen(text));
            if (escaped) {
                char sse[4096];
                int n = snprintf(sse, sizeof(sse),
                    "data: {\"token\":\"%s\",\"done\":%s}\n\n",
                    escaped, is_done ? "true" : "false");
                if (n > 0 && n < (int)sizeof(sse))
                    write(out->fd, sse, n);
                free(escaped);
            }
        } else {
            printf("%s", text);
            fflush(stdout);
        }

        // Accumulate
        int tlen = strlen(text);
        while (out->buf_len + tlen + 1 > out->buf_cap) {
            out->buf_cap *= 2;
            void *new_buf = realloc(out->buf, out->buf_cap);
            if (!new_buf) { out->buf_cap /= 2; break; } // OOM: stop accumulating
            out->buf = new_buf;
        }
        if (out->buf_len + tlen + 1 <= out->buf_cap) {
            memcpy(out->buf + out->buf_len, text, tlen);
            out->buf_len += tlen;
            out->buf[out->buf_len] = '\0';
        }
    }

    if (is_done) {
        if (out->fd >= 0) {
            write(out->fd, "data: [DONE]\n\n", 14);
        } else if (out->fd != -2) {
            printf("\n");
        }
    }
}

// ── Generate with stats ──

static void generate(MnemoCudaCtx *ctx, const char *prompt, float temp, int max_tokens, OutputCtx *out) {
    out->buf_len = 0;
    out->buf[0] = '\0';

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    mnemo_cuda_generate(ctx, prompt, max_tokens, temp, on_token, out);

    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    MnemoCudaStats stats = mnemo_cuda_get_stats(ctx);
    fprintf(stderr, "[MnemoCUDA] %d tokens in %.1fs (%.1f tok/s, prefill %.1fs)\n",
            stats.tokens_generated, elapsed, stats.tokens_per_second,
            elapsed - (stats.tokens_generated > 0 ? stats.tokens_generated / stats.tokens_per_second : 0));
}

// ── REPL mode ──

static void run_repl(MnemoCudaCtx *ctx) {
    OutputCtx out = { .fd = -1, .buf = malloc(4096), .buf_len = 0, .buf_cap = 4096 };
    char line[4096];

    fprintf(stderr, "[MnemoCUDA] REPL ready. Type a prompt (Ctrl+C to exit):\n");
    while (running) {
        printf("\n> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        // Strip trailing newline
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;

        // Commands
        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) break;
        if (strcmp(line, "/stats") == 0) {
            MnemoCudaStats s = mnemo_cuda_get_stats(ctx);
            printf("Tokens: %d, Speed: %.1f tok/s, VRAM: %.1f MB, Resident: %.1f MB\n",
                   s.tokens_generated, s.tokens_per_second,
                   (double)s.vram_used_bytes / (1024*1024),
                   (double)s.resident_size_bytes / (1024*1024));
            continue;
        }
        if (strcmp(line, "/info") == 0) {
            printf("%s\n", mnemo_cuda_get_info(ctx));
            continue;
        }
        if (strcmp(line, "/heat") == 0) {
            MnemoCudaHeatStats hs = mnemo_cuda_get_heat_stats(ctx);
            printf("Expert Heat Map (%lu tokens profiled, %s)\n",
                   (unsigned long)hs.total_tokens,
                   hs.pinning_active ? "pinning ACTIVE" : "pinning OFF");
            printf("Active experts: %d/%d, Total activations: %lu\n",
                   hs.active_experts, hs.n_layers * hs.n_experts_per_layer,
                   (unsigned long)hs.total_activations);
            printf("\nTop %d hottest experts:\n", HEAT_TOP_N);
            for (int i = 0; i < HEAT_TOP_N && hs.top_count[i] > 0; i++)
                printf("  #%d: layer %d expert %d — %u activations\n",
                       i + 1, hs.top_layer[i], hs.top_expert[i], hs.top_count[i]);
            printf("\nVRAM cache per GPU:\n");
            for (int g = 0; g < 8 && hs.cache_slots[g] > 0; g++)
                printf("  GPU %d: %d/%d used, %d pinned\n",
                       g, hs.cache_used[g], hs.cache_slots[g], hs.cache_pinned[g]);
            printf("\nRAM cache: %d/%d used, %d hits, %d misses (%.0f%% hit rate)\n",
                   hs.ram_used, hs.ram_slots,
                   hs.ram_hits, hs.ram_misses,
                   hs.ram_hits + hs.ram_misses > 0
                       ? 100.0 * hs.ram_hits / (hs.ram_hits + hs.ram_misses) : 0);
            continue;
        }
        if (strcmp(line, "/pin") == 0) {
            mnemo_cuda_heat_pin(ctx);
            continue;
        }
        if (strcmp(line, "/save") == 0) {
            mnemo_cuda_heat_save(ctx);
            continue;
        }

        generate(ctx, line, 0.7, 512, &out);
    }
    free(out.buf);
}

// ── HTTP helpers ──

#define HTTP_MAX_HEADERS 8192
#define HTTP_MAX_BODY    (1024 * 1024)  // 1 MB max request body

static void http_respond_error(int fd, int code, const char *msg) {
    char buf[512];
    int blen = snprintf(buf, sizeof(buf),
        "{\"error\":\"%s\",\"code\":%d}", msg, code);
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n",
        code, msg, blen);
    write(fd, hdr, hlen);
    write(fd, buf, blen);
}

static void http_respond_json(int fd, const char *json, int json_len) {
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n", json_len);
    write(fd, hdr, hlen);
    write(fd, json, json_len);
}

static void http_respond_cors_preflight(int fd) {
    const char *resp =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Connection: close\r\n\r\n";
    write(fd, resp, strlen(resp));
}

// Read full HTTP request: headers + body (respects Content-Length).
// Returns dynamically allocated buffer containing full request, or NULL.
// Sets *body_out to point into the buffer at the start of the body.
// Caller must free the returned buffer.
static char *http_read_request(int fd, char **body_out, int *body_len_out) {
    *body_out = NULL;
    *body_len_out = 0;

    // Read headers first (up to HTTP_MAX_HEADERS)
    char *buf = malloc(HTTP_MAX_HEADERS + HTTP_MAX_BODY + 1);  // +1 for NUL
    if (!buf) return NULL;

    int total = 0;
    int headers_end = -1;

    // Read until we find \r\n\r\n (end of headers)
    while (total < HTTP_MAX_HEADERS) {
        int n = read(fd, buf + total, HTTP_MAX_HEADERS - total);
        if (n <= 0) { free(buf); return NULL; }
        total += n;
        buf[total] = '\0';

        char *hend = strstr(buf, "\r\n\r\n");
        if (hend) {
            headers_end = (int)(hend - buf) + 4;
            break;
        }
    }
    if (headers_end < 0) { free(buf); return NULL; }

    // Parse Content-Length from headers
    int content_length = 0;
    char *cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        cl += 15;
        while (*cl == ' ') cl++;
        content_length = atoi(cl);
    }
    if (content_length < 0) content_length = 0;
    if (content_length > HTTP_MAX_BODY) {
        free(buf);
        return NULL;  // body too large
    }

    // Read remaining body bytes if we haven't received them all
    int body_received = total - headers_end;
    while (body_received < content_length) {
        int want = content_length - body_received;
        int n = read(fd, buf + total, want);
        if (n <= 0) {
            // Connection closed before full body received — reject
            free(buf);
            return NULL;
        }
        total += n;
        body_received += n;
    }
    buf[total] = '\0';

    *body_out = buf + headers_end;
    *body_len_out = body_received;
    return buf;
}

// Extract a JSON string value for a given key from a flat JSON object.
// Handles basic escapes (\", \\, \n, \t, \/, \uXXXX).
// Returns dynamically allocated string, or NULL if key not found.
static char *json_extract_string(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p != '"') return NULL;
    p++;  // skip opening quote

    // Estimate max output size (input length is safe upper bound)
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
                    // UTF-8 encode
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

// Escape a string for JSON output. Returns malloc'd buffer.
static char *json_escape(const char *src, int len) {
    int cap = len * 6 + 1;  // worst case: every byte is \u00XX (6 chars each)
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

// ── HTTP server mode ──

static void run_http(MnemoCudaCtx *ctx, int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(port),
    };

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return;
    }
    listen(server_fd, 5);

    fprintf(stderr, "[MnemoCUDA] HTTP server on port %d\n", port);
    fprintf(stderr, "[MnemoCUDA] POST /v1/completions {\"prompt\":\"...\", \"max_tokens\":256, \"temperature\":0.7}\n");

    OutputCtx out = { .fd = -1, .buf = malloc(65536), .buf_len = 0, .buf_cap = 65536 };

    while (running) {
        struct sockaddr_in client;
        socklen_t clen = sizeof(client);
        int client_fd = accept(server_fd, (struct sockaddr *)&client, &clen);
        if (client_fd < 0) continue;

        // Set socket timeout to avoid blocking forever on slow clients
        struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        // Read full HTTP request (headers + body via Content-Length)
        char *body = NULL;
        int body_len = 0;
        char *req = http_read_request(client_fd, &body, &body_len);
        if (!req) {
            http_respond_error(client_fd, 400, "Bad Request");
            close(client_fd);
            continue;
        }

        // CORS preflight
        if (strncmp(req, "OPTIONS ", 8) == 0) {
            http_respond_cors_preflight(client_fd);
            free(req); close(client_fd);
            continue;
        }

        // GET /heat — return heat stats as JSON
        if (strncmp(req, "GET /heat", 9) == 0) {
            MnemoCudaHeatStats hs = mnemo_cuda_get_heat_stats(ctx);
            char json[8192];
            int off = snprintf(json, sizeof(json),
                "{\"total_tokens\":%lu,\"pinning_active\":%s,"
                "\"active_experts\":%d,\"total_experts\":%d,"
                "\"total_activations\":%lu,\"top\":[",
                (unsigned long)hs.total_tokens,
                hs.pinning_active ? "true" : "false",
                hs.active_experts,
                hs.n_layers * hs.n_experts_per_layer,
                (unsigned long)hs.total_activations);
            for (int i = 0; i < HEAT_TOP_N && hs.top_count[i] > 0; i++) {
                if (i > 0) json[off++] = ',';
                off += snprintf(json + off, sizeof(json) - off,
                    "{\"layer\":%d,\"expert\":%d,\"count\":%u}",
                    hs.top_layer[i], hs.top_expert[i], hs.top_count[i]);
            }
            off += snprintf(json + off, sizeof(json) - off, "],\"cache\":[");
            for (int g = 0; g < 8 && hs.cache_slots[g] > 0; g++) {
                if (g > 0) json[off++] = ',';
                off += snprintf(json + off, sizeof(json) - off,
                    "{\"gpu\":%d,\"slots\":%d,\"used\":%d,\"pinned\":%d}",
                    g, hs.cache_slots[g], hs.cache_used[g], hs.cache_pinned[g]);
            }
            off += snprintf(json + off, sizeof(json) - off,
                "],\"ram\":{\"slots\":%d,\"used\":%d,\"hits\":%d,\"misses\":%d}}",
                hs.ram_slots, hs.ram_used, hs.ram_hits, hs.ram_misses);

            http_respond_json(client_fd, json, off);
            free(req); close(client_fd);
            continue;
        }

        // Only POST /v1/completions accepted
        if (strncmp(req, "POST ", 5) != 0) {
            http_respond_error(client_fd, 405, "Method Not Allowed");
            free(req); close(client_fd);
            continue;
        }
        if (strncmp(req + 5, "/v1/completions", 15) != 0) {
            http_respond_error(client_fd, 404, "Not Found");
            free(req); close(client_fd);
            continue;
        }
        if (!body || body_len == 0) {
            http_respond_error(client_fd, 400, "Empty body");
            free(req); close(client_fd);
            continue;
        }

        // Extract fields from JSON body
        char *prompt = json_extract_string(body, "prompt");
        if (!prompt || prompt[0] == '\0') {
            http_respond_error(client_fd, 400, "Missing or empty prompt");
            free(prompt); free(req); close(client_fd);
            continue;
        }

        int max_tokens = json_extract_int(body, "max_tokens", 256);
        float temperature = json_extract_float(body, "temperature", 0.7f);
        int stream = json_extract_bool(body, "stream", 0);
        int raw_prompt = json_extract_bool(body, "raw_prompt", 0);

        if (max_tokens <= 0) max_tokens = 256;
        if (max_tokens > 32768) max_tokens = 32768;

        if (raw_prompt)
            fprintf(stderr, "[MnemoCUDA] raw_prompt=true, passing prompt as-is\n");

        if (stream) {
            const char *hdr = "HTTP/1.1 200 OK\r\n"
                              "Content-Type: text/event-stream; charset=utf-8\r\n"
                              "Cache-Control: no-cache\r\n"
                              "Access-Control-Allow-Origin: *\r\n"
                              "Connection: close\r\n\r\n";
            write(client_fd, hdr, strlen(hdr));
            out.fd = client_fd;
            generate(ctx, prompt, temperature, max_tokens, &out);
            out.fd = -1;
        } else {
            out.fd = -1;
            generate(ctx, prompt, temperature, max_tokens, &out);

            MnemoCudaStats stats = mnemo_cuda_get_stats(ctx);
            char *escaped = json_escape(out.buf, out.buf_len);
            if (!escaped) {
                http_respond_error(client_fd, 500, "Internal Server Error");
                free(prompt); free(req); close(client_fd);
                continue;
            }

            int resp_cap = strlen(escaped) + 128;
            char *response = malloc(resp_cap);
            int rlen = snprintf(response, resp_cap,
                "{\"text\":\"%s\",\"tokens\":%d,\"tok_per_sec\":%.1f}",
                escaped, stats.tokens_generated, stats.tokens_per_second);

            http_respond_json(client_fd, response, rlen);
            free(escaped);
            free(response);
        }

        free(prompt);
        free(req);
        close(client_fd);
    }

    close(server_fd);
    free(out.buf);
}

// ── Main ──

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "MnemoCUDA Server — Expert streaming inference for MoE models\n\n");
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s <model_dir> --repl              Interactive REPL\n", argv[0]);
        fprintf(stderr, "  %s <model_dir> --http <port>       HTTP API server\n", argv[0]);
        fprintf(stderr, "  %s <model_dir> \"prompt\"            Single generation\n", argv[0]);
        return 1;
    }

    signal(SIGINT, sigint_handler);
    signal(SIGPIPE, SIG_IGN);

    const char *model_dir = argv[1];

    // Context length from config_default(), override with --context N
    MnemoCudaConfig config = mnemo_cuda_config_default();
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--context") == 0) config.context_length = atoi(argv[i+1]);
    }

    fprintf(stderr, "[MnemoCUDA] Loading model from %s...\n", model_dir);

    MnemoCudaCtx *ctx = mnemo_cuda_create();
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    config.model_dir = model_dir;

    int result = mnemo_cuda_load(ctx, config);
    if (result != 0) {
        fprintf(stderr, "Load failed: %d\n", result);
        mnemo_cuda_destroy(ctx);
        return 1;
    }

    fprintf(stderr, "[MnemoCUDA] %s\n", mnemo_cuda_get_info(ctx));

    // Pre-warm caches: multi-round diverse prompts to fill VRAM + page cache.
    // Different topics activate different experts — diverse warmup = better coverage.
    {
        fprintf(stderr, "[MnemoCUDA] Warming caches (multi-round)...\n");
        struct timespec tw0, tw1;
        clock_gettime(CLOCK_MONOTONIC, &tw0);

        // Silent callback — discard generated tokens, just accumulate for cache warmth
        OutputCtx warmup_out = { .fd = -2, .buf = malloc(16384), .buf_len = 0, .buf_cap = 16384 };

        const char *warmup_prompts[] = {
            "Hello, how are you today?",
            "Explain quantum physics briefly.",
            "Escribe un poema corto en español sobre el mar.",
            "def fibonacci(n):\n    if n <= 1: return n\n    return",
            "What is the capital of France and why is it important?",
            "Describe the solar system and its planets.",
            NULL
        };
        int warmup_tokens[] = { 20, 30, 30, 20, 20, 30 };
        int n_rounds = 0;

        for (int i = 0; warmup_prompts[i]; i++) {
            warmup_out.buf_len = 0;
            warmup_out.buf[0] = '\0';
            n_rounds++;
            fprintf(stderr, "[MnemoCUDA] Warm-up %d/6...\n", n_rounds);
            mnemo_cuda_generate(ctx, warmup_prompts[i], warmup_tokens[i], 0.0,
                                on_token, &warmup_out);
        }

        clock_gettime(CLOCK_MONOTONIC, &tw1);
        double warm_secs = (tw1.tv_sec - tw0.tv_sec) + (tw1.tv_nsec - tw0.tv_nsec) / 1e9;
        fprintf(stderr, "[MnemoCUDA] Warm-up done in %.1fs (%d rounds, cache hot)\n",
                warm_secs, n_rounds);
        free(warmup_out.buf);
    }

    if (argc >= 3 && strcmp(argv[2], "--repl") == 0) {
        run_repl(ctx);
    } else if (argc >= 4 && strcmp(argv[2], "--http") == 0) {
        run_http(ctx, atoi(argv[3]));
    } else if (argc >= 3) {
        // Single prompt
        OutputCtx out = { .fd = -1, .buf = malloc(65536), .buf_len = 0, .buf_cap = 65536 };
        generate(ctx, argv[2], 0.7, 512, &out);
        free(out.buf);
    }

    fprintf(stderr, "[MnemoCUDA] Cleaning up...\n");
    mnemo_cuda_destroy(ctx);
    return 0;
}
