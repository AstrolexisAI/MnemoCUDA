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

static void on_token(const char *text, bool is_done, void *userdata) {
    OutputCtx *out = (OutputCtx *)userdata;

    if (text && text[0]) {
        if (out->fd >= 0) {
            // HTTP SSE: data: {"token": "..."}\n\n
            char sse[1024];
            // Escape JSON
            char escaped[512];
            int j = 0;
            for (int i = 0; text[i] && j < 500; i++) {
                if (text[i] == '"') { escaped[j++] = '\\'; escaped[j++] = '"'; }
                else if (text[i] == '\\') { escaped[j++] = '\\'; escaped[j++] = '\\'; }
                else if (text[i] == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
                else escaped[j++] = text[i];
            }
            escaped[j] = '\0';

            int n = snprintf(sse, sizeof(sse),
                "data: {\"token\":\"%s\",\"done\":%s}\n\n",
                escaped, is_done ? "true" : "false");
            write(out->fd, sse, n);
        } else {
            printf("%s", text);
            fflush(stdout);
        }

        // Accumulate
        int tlen = strlen(text);
        while (out->buf_len + tlen + 1 > out->buf_cap) {
            out->buf_cap *= 2;
            out->buf = realloc(out->buf, out->buf_cap);
        }
        memcpy(out->buf + out->buf_len, text, tlen);
        out->buf_len += tlen;
        out->buf[out->buf_len] = '\0';
    }

    if (is_done) {
        if (out->fd >= 0) {
            write(out->fd, "data: [DONE]\n\n", 14);
        } else {
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

        // Read HTTP request
        char req[8192];
        int n = read(client_fd, req, sizeof(req) - 1);
        if (n <= 0) { close(client_fd); continue; }
        req[n] = '\0';

        // Handle GET /heat — return heat stats as JSON
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

            char hdr[256];
            int hlen = snprintf(hdr, sizeof(hdr),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n", off);
            write(client_fd, hdr, hlen);
            write(client_fd, json, off);
            close(client_fd);
            continue;
        }

        // Parse prompt from JSON body (minimal)
        char *body = strstr(req, "\r\n\r\n");
        if (!body) { close(client_fd); continue; }
        body += 4;

        // Extract "prompt": "..."
        char *pp = strstr(body, "\"prompt\"");
        if (!pp) { close(client_fd); continue; }
        pp = strchr(pp + 8, '"');
        if (!pp) { close(client_fd); continue; }
        pp++;
        char prompt[4096];
        int pi = 0;
        while (*pp && *pp != '"' && pi < 4090) {
            if (*pp == '\\' && *(pp+1) == '"') { prompt[pi++] = '"'; pp += 2; }
            else if (*pp == '\\' && *(pp+1) == 'n') { prompt[pi++] = '\n'; pp += 2; }
            else prompt[pi++] = *pp++;
        }
        prompt[pi] = '\0';

        // Extract max_tokens and temperature
        int max_tokens = 256;
        float temperature = 0.7;
        char *mt = strstr(body, "\"max_tokens\"");
        if (mt) { mt = strchr(mt + 12, ':'); if (mt) max_tokens = atoi(mt + 1); }
        char *tp = strstr(body, "\"temperature\"");
        if (tp) { tp = strchr(tp + 13, ':'); if (tp) temperature = atof(tp + 1); }

        // Check if streaming requested
        int stream = (strstr(body, "\"stream\":true") || strstr(body, "\"stream\": true")) ? 1 : 0;

        if (stream) {
            // SSE streaming response
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
            // Non-streaming: generate full response, return JSON
            out.fd = -1;
            generate(ctx, prompt, temperature, max_tokens, &out);

            MnemoCudaStats stats = mnemo_cuda_get_stats(ctx);
            char response[65536];
            // Escape the output for JSON
            char escaped[65536];
            int j = 0;
            for (int i = 0; out.buf[i] && j < 65000; i++) {
                if (out.buf[i] == '"') { escaped[j++] = '\\'; escaped[j++] = '"'; }
                else if (out.buf[i] == '\\') { escaped[j++] = '\\'; escaped[j++] = '\\'; }
                else if (out.buf[i] == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
                else if (out.buf[i] == '\t') { escaped[j++] = '\\'; escaped[j++] = 't'; }
                else escaped[j++] = out.buf[i];
            }
            escaped[j] = '\0';

            int rlen = snprintf(response, sizeof(response),
                "{\"text\":\"%s\",\"tokens\":%d,\"tok_per_sec\":%.1f}",
                escaped, stats.tokens_generated, stats.tokens_per_second);

            char hdr[256];
            int hlen = snprintf(hdr, sizeof(hdr),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n", rlen);
            write(client_fd, hdr, hlen);
            write(client_fd, response, rlen);
        }

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

    // Context length: 8K default (maximizes expert VRAM cache)
    // Override with --context N
    int context_len = 8192;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--context") == 0) context_len = atoi(argv[i+1]);
    }

    fprintf(stderr, "[MnemoCUDA] Loading model from %s...\n", model_dir);

    MnemoCudaCtx *ctx = mnemo_cuda_create();
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    MnemoCudaConfig config = mnemo_cuda_config_default();
    config.model_dir = model_dir;
    config.context_length = context_len;

    int result = mnemo_cuda_load(ctx, config);
    if (result != 0) {
        fprintf(stderr, "Load failed: %d\n", result);
        mnemo_cuda_destroy(ctx);
        return 1;
    }

    fprintf(stderr, "[MnemoCUDA] %s\n", mnemo_cuda_get_info(ctx));

    // Pre-warm caches: run a short generation to fill VRAM cache + page cache
    {
        fprintf(stderr, "[MnemoCUDA] Warming caches...\n");
        struct timespec tw0, tw1;
        clock_gettime(CLOCK_MONOTONIC, &tw0);
        OutputCtx warmup_out = { .fd = -1, .buf = malloc(4096), .buf_len = 0, .buf_cap = 4096 };
        mnemo_cuda_generate(ctx, "Hello, how are you?", 3, 0.0, on_token, &warmup_out);
        clock_gettime(CLOCK_MONOTONIC, &tw1);
        double warm_secs = (tw1.tv_sec - tw0.tv_sec) + (tw1.tv_nsec - tw0.tv_nsec) / 1e9;
        MnemoCudaStats ws = mnemo_cuda_get_stats(ctx);
        fprintf(stderr, "[MnemoCUDA] Warm-up done in %.1fs (cache ready)\n", warm_secs);
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
