/**
 * Test Expert Heat Profiling — validates heat map, save/load, pinning logic.
 *
 * Runs WITHOUT GPU or model — tests data structures and algorithms only.
 *
 * Build: gcc-14 -O2 -Wall -o test_heat test_heat.c -lm
 * Run:   ./test_heat
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>

// ── Re-implement heat data structures (no CUDA dependency) ──

#define HEAT_TOP_N 20

typedef struct { int layer; int expert; uint32_t count; } HeatEntry;

typedef struct {
    uint64_t total_tokens;
    uint64_t total_activations;
    int active_experts;
    int n_layers;
    int n_experts_per_layer;
    bool pinning_active;
    int top_layer[HEAT_TOP_N];
    int top_expert[HEAT_TOP_N];
    uint32_t top_count[HEAT_TOP_N];
    int cache_slots[8];
    int cache_used[8];
    int cache_pinned[8];
} HeatStats;

static int heat_entry_cmp_desc(const void *a, const void *b) {
    uint32_t ca = ((const HeatEntry *)a)->count;
    uint32_t cb = ((const HeatEntry *)b)->count;
    return (cb > ca) - (cb < ca);
}

// ── Test helpers ──

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  TEST: %s ... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { printf("FAIL: %s (got %ld, expected %ld)\n", msg, (long)(a), (long)(b)); tests_failed++; return; } } while(0)
#define ASSERT_TRUE(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

// ── Test 1: Heat map allocation and counting ──

static void test_heat_map_counting(void) {
    TEST("heat_map counting");

    int n_layers = 4;
    int n_experts = 8;
    int total = n_layers * n_experts;

    uint32_t *heat_map = calloc(total, sizeof(uint32_t));
    ASSERT_TRUE(heat_map != NULL, "calloc failed");

    // All zeros initially
    for (int i = 0; i < total; i++)
        ASSERT_EQ(heat_map[i], 0, "initial zero");

    // Simulate expert activations: layer 0, experts 3 and 5 are hot
    for (int tok = 0; tok < 100; tok++) {
        heat_map[0 * n_experts + 3]++;  // layer 0, expert 3
        heat_map[0 * n_experts + 5]++;  // layer 0, expert 5
        heat_map[1 * n_experts + 2]++;  // layer 1, expert 2
        // layer 0 expert 7 only activated sometimes
        if (tok % 10 == 0) heat_map[0 * n_experts + 7]++;
    }

    ASSERT_EQ(heat_map[0 * n_experts + 3], 100, "expert 0:3 count");
    ASSERT_EQ(heat_map[0 * n_experts + 5], 100, "expert 0:5 count");
    ASSERT_EQ(heat_map[1 * n_experts + 2], 100, "expert 1:2 count");
    ASSERT_EQ(heat_map[0 * n_experts + 7], 10, "expert 0:7 count");
    ASSERT_EQ(heat_map[2 * n_experts + 0], 0, "cold expert");

    free(heat_map);
    PASS();
}

// ── Test 2: Heat sorting (qsort descending) ──

static void test_heat_sorting(void) {
    TEST("heat_map qsort descending");

    int n_layers = 4;
    int n_experts = 8;
    int total = n_layers * n_experts;

    uint32_t *heat_map = calloc(total, sizeof(uint32_t));
    heat_map[0 * n_experts + 3] = 500;   // hottest
    heat_map[1 * n_experts + 2] = 300;   // 2nd
    heat_map[0 * n_experts + 5] = 200;   // 3rd
    heat_map[3 * n_experts + 7] = 50;    // 4th
    heat_map[2 * n_experts + 1] = 10;    // 5th

    HeatEntry *sorted = malloc(total * sizeof(HeatEntry));
    for (int i = 0; i < total; i++) {
        sorted[i].layer = i / n_experts;
        sorted[i].expert = i % n_experts;
        sorted[i].count = heat_map[i];
    }

    qsort(sorted, total, sizeof(HeatEntry), heat_entry_cmp_desc);

    // Verify descending order
    ASSERT_EQ(sorted[0].count, 500, "1st hottest");
    ASSERT_EQ(sorted[0].layer, 0, "1st layer");
    ASSERT_EQ(sorted[0].expert, 3, "1st expert");

    ASSERT_EQ(sorted[1].count, 300, "2nd hottest");
    ASSERT_EQ(sorted[1].layer, 1, "2nd layer");
    ASSERT_EQ(sorted[1].expert, 2, "2nd expert");

    ASSERT_EQ(sorted[2].count, 200, "3rd hottest");
    ASSERT_EQ(sorted[3].count, 50, "4th hottest");
    ASSERT_EQ(sorted[4].count, 10, "5th hottest");

    // Rest should be 0
    for (int i = 5; i < total; i++)
        ASSERT_EQ(sorted[i].count, 0, "cold expert is 0");

    free(sorted);
    free(heat_map);
    PASS();
}

// ── Test 3: Save and load heat map binary format ──

static void test_heat_save_load(void) {
    TEST("heat_map save/load binary");

    int n_layers = 4;
    int n_experts = 8;
    int total = n_layers * n_experts;
    uint64_t tokens = 12345;

    uint32_t *heat_map = calloc(total, sizeof(uint32_t));
    heat_map[0] = 100;
    heat_map[5] = 200;
    heat_map[31] = 999;

    // Save
    const char *path = "/tmp/test_expert_heat.bin";
    FILE *f = fopen(path, "wb");
    ASSERT_TRUE(f != NULL, "fopen write");
    uint32_t magic = 0x48454154;
    uint32_t nl = (uint32_t)n_layers;
    uint32_t ne = (uint32_t)n_experts;
    fwrite(&magic, 4, 1, f);
    fwrite(&nl, 4, 1, f);
    fwrite(&ne, 4, 1, f);
    fwrite(&tokens, 8, 1, f);
    fwrite(heat_map, sizeof(uint32_t), total, f);
    fclose(f);

    // Load
    uint32_t *loaded = calloc(total, sizeof(uint32_t));
    f = fopen(path, "rb");
    ASSERT_TRUE(f != NULL, "fopen read");

    uint32_t rm, rn_layers, rn_experts;
    uint64_t rtokens;
    fread(&rm, 4, 1, f);
    fread(&rn_layers, 4, 1, f);
    fread(&rn_experts, 4, 1, f);
    fread(&rtokens, 8, 1, f);
    fread(loaded, sizeof(uint32_t), total, f);
    fclose(f);

    ASSERT_EQ(rm, 0x48454154, "magic");
    ASSERT_EQ(rn_layers, (uint32_t)n_layers, "n_layers");
    ASSERT_EQ(rn_experts, (uint32_t)n_experts, "n_experts");
    ASSERT_EQ(rtokens, tokens, "tokens");
    ASSERT_EQ(loaded[0], 100, "loaded[0]");
    ASSERT_EQ(loaded[5], 200, "loaded[5]");
    ASSERT_EQ(loaded[31], 999, "loaded[31]");
    ASSERT_EQ(loaded[1], 0, "cold expert loaded");

    free(heat_map);
    free(loaded);
    unlink(path);
    PASS();
}

// ── Test 4: Dimension mismatch on load ──

static void test_heat_load_mismatch(void) {
    TEST("heat_map load dimension mismatch");

    const char *path = "/tmp/test_expert_heat_mismatch.bin";
    FILE *f = fopen(path, "wb");
    ASSERT_TRUE(f != NULL, "fopen write");

    // Write with n_layers=4, n_experts=8
    uint32_t magic = 0x48454154;
    uint32_t nl = 4, ne = 8;
    uint64_t tokens = 100;
    fwrite(&magic, 4, 1, f);
    fwrite(&nl, 4, 1, f);
    fwrite(&ne, 4, 1, f);
    fwrite(&tokens, 8, 1, f);
    uint32_t data[32] = {0};
    data[0] = 42;
    fwrite(data, sizeof(uint32_t), 32, f);
    fclose(f);

    // Try to load with different dimensions (n_layers=6)
    f = fopen(path, "rb");
    uint32_t rm, rn_layers, rn_experts;
    uint64_t rtokens;
    fread(&rm, 4, 1, f);
    fread(&rn_layers, 4, 1, f);
    fread(&rn_experts, 4, 1, f);
    fread(&rtokens, 8, 1, f);

    // The engine checks: magic == 0x48454154 && n_layers == cfg->num_hidden_layers
    // Simulate: cfg has 6 layers, file has 4
    bool should_load = (rm == 0x48454154 && (int)rn_layers == 6 && (int)rn_experts == 8);
    ASSERT_TRUE(!should_load, "mismatch should reject");

    fclose(f);
    unlink(path);
    PASS();
}

// ── Test 5: Cache pinning logic (simulated) ──

static void test_cache_pinning_logic(void) {
    TEST("cache pinning eviction skip");

    // Simulate a cache with 6 slots
    int n_slots = 6;
    int cache_layer[6] = {0, 0, 1, 1, 2, 2};
    int cache_expert[6] = {3, 5, 2, 7, 1, 4};
    uint64_t cache_lru[6] = {10, 20, 5, 30, 1, 15};
    bool cache_pinned[6] = {false, true, false, true, false, false};

    // Find LRU slot that is NOT pinned
    int target = -1;
    uint64_t min_lru = UINT64_MAX;
    for (int s = 0; s < n_slots; s++) {
        if (cache_pinned[s]) continue; // skip pinned
        if (cache_lru[s] < min_lru) {
            min_lru = cache_lru[s];
            target = s;
        }
    }

    // Slot 4 has LRU=1 (lowest among non-pinned: 10, 5, 1, 15)
    ASSERT_EQ(target, 4, "evict slot 4 (lowest non-pinned LRU)");
    ASSERT_EQ(min_lru, 1, "min LRU value");

    // Verify pinned slots (1 and 3) were NOT considered
    ASSERT_TRUE(cache_pinned[1], "slot 1 is pinned");
    ASSERT_TRUE(cache_pinned[3], "slot 3 is pinned");

    PASS();
}

// ── Test 6: All slots pinned → eviction returns -1 ──

static void test_all_slots_pinned(void) {
    TEST("all slots pinned → no eviction");

    int n_slots = 3;
    bool cache_pinned[3] = {true, true, true};
    int cache_layer[3] = {0, 1, 2};  // no empty slots
    uint64_t cache_lru[3] = {10, 20, 30};
    (void)cache_layer; (void)cache_lru; // used conceptually

    int target = -1;
    uint64_t min_lru = UINT64_MAX;
    for (int s = 0; s < n_slots; s++) {
        if (cache_layer[s] == -1) { target = s; break; }
        if (cache_pinned[s]) continue;
        if (cache_lru[s] < min_lru) { min_lru = cache_lru[s]; target = s; }
    }

    ASSERT_EQ(target, -1, "no evictable slot");
    PASS();
}

// ── Test 7: Top-N heat stats extraction ──

static void test_top_n_stats(void) {
    TEST("top-N heat stats");

    int n_layers = 3;
    int n_experts = 4;
    int total = n_layers * n_experts;

    uint32_t heat_map[12] = {
        // Layer 0: experts 0-3
        10, 50, 5, 100,
        // Layer 1: experts 0-3
        200, 0, 30, 0,
        // Layer 2: experts 0-3
        0, 0, 0, 75
    };

    // Expected top order: L1:E0(200), L0:E3(100), L2:E3(75), L0:E1(50), L1:E2(30), L0:E0(10), L0:E2(5)

    HeatEntry sorted[12];
    for (int i = 0; i < total; i++) {
        sorted[i].layer = i / n_experts;
        sorted[i].expert = i % n_experts;
        sorted[i].count = heat_map[i];
    }
    qsort(sorted, total, sizeof(HeatEntry), heat_entry_cmp_desc);

    ASSERT_EQ(sorted[0].layer, 1, "top1 layer");
    ASSERT_EQ(sorted[0].expert, 0, "top1 expert");
    ASSERT_EQ(sorted[0].count, 200, "top1 count");

    ASSERT_EQ(sorted[1].layer, 0, "top2 layer");
    ASSERT_EQ(sorted[1].expert, 3, "top2 expert");
    ASSERT_EQ(sorted[1].count, 100, "top2 count");

    ASSERT_EQ(sorted[2].layer, 2, "top3 layer");
    ASSERT_EQ(sorted[2].expert, 3, "top3 expert");
    ASSERT_EQ(sorted[2].count, 75, "top3 count");

    // Count active
    int active = 0;
    uint64_t total_act = 0;
    for (int i = 0; i < total; i++) {
        if (heat_map[i] > 0) active++;
        total_act += heat_map[i];
    }
    ASSERT_EQ(active, 7, "active experts count");
    ASSERT_EQ(total_act, 470, "total activations");

    PASS();
}

// ── Test 8: Large-scale sort performance (simulate 235B: 94 layers × 128 experts) ──

static void test_large_scale_sort(void) {
    TEST("large-scale sort (94×128 = 12032 entries)");

    int n_layers = 94;
    int n_experts = 128;
    int total = n_layers * n_experts;

    HeatEntry *sorted = malloc(total * sizeof(HeatEntry));
    ASSERT_TRUE(sorted != NULL, "malloc");

    // Simulate realistic distribution: most experts cold, ~10% hot
    for (int i = 0; i < total; i++) {
        sorted[i].layer = i / n_experts;
        sorted[i].expert = i % n_experts;
        // Hot experts: ~10% get high counts, rest near 0
        if (i % 10 == 0) {
            sorted[i].count = (uint32_t)(1000 + (i * 7) % 5000);
        } else if (i % 3 == 0) {
            sorted[i].count = (uint32_t)((i * 13) % 100);
        } else {
            sorted[i].count = 0;
        }
    }

    qsort(sorted, total, sizeof(HeatEntry), heat_entry_cmp_desc);

    // Verify descending order
    for (int i = 1; i < total; i++) {
        ASSERT_TRUE(sorted[i].count <= sorted[i-1].count, "descending order broken");
    }

    // First entry should be the max
    ASSERT_TRUE(sorted[0].count > 0, "top entry nonzero");

    free(sorted);
    PASS();
}

// ── Test 9: Even larger MoE (512 experts × 94 layers = 48128) ──

static void test_huge_moe_sort(void) {
    TEST("huge MoE sort (94×512 = 48128 entries)");

    int n_layers = 94;
    int n_experts = 512;
    int total = n_layers * n_experts;

    HeatEntry *sorted = malloc(total * sizeof(HeatEntry));
    ASSERT_TRUE(sorted != NULL, "malloc");

    for (int i = 0; i < total; i++) {
        sorted[i].layer = i / n_experts;
        sorted[i].expert = i % n_experts;
        sorted[i].count = (uint32_t)((i * 31337) % 10000);
    }

    qsort(sorted, total, sizeof(HeatEntry), heat_entry_cmp_desc);

    for (int i = 1; i < total; i++)
        ASSERT_TRUE(sorted[i].count <= sorted[i-1].count, "descending order");

    free(sorted);
    PASS();
}

// ── Test 10: Pin ratio check (50% of cache) ──

static void test_pin_ratio(void) {
    TEST("pin ratio 50% cap");

    int cache_slots = 100;
    int max_pin = cache_slots / 2;

    ASSERT_EQ(max_pin, 50, "50% of 100");

    cache_slots = 1;
    max_pin = cache_slots / 2;
    if (max_pin < 1) max_pin = 1;
    ASSERT_EQ(max_pin, 1, "minimum 1 pin");

    cache_slots = 7;
    max_pin = cache_slots / 2;
    ASSERT_EQ(max_pin, 3, "50% of 7 = 3");

    PASS();
}

// ── Test 11: Binary file header validation ──

static void test_heat_file_bad_magic(void) {
    TEST("heat file bad magic rejected");

    const char *path = "/tmp/test_heat_bad_magic.bin";
    FILE *f = fopen(path, "wb");
    uint32_t bad_magic = 0xDEADBEEF;
    uint32_t nl = 4, ne = 8;
    uint64_t tokens = 100;
    fwrite(&bad_magic, 4, 1, f);
    fwrite(&nl, 4, 1, f);
    fwrite(&ne, 4, 1, f);
    fwrite(&tokens, 8, 1, f);
    fclose(f);

    f = fopen(path, "rb");
    uint32_t rm;
    fread(&rm, 4, 1, f);
    bool valid = (rm == 0x48454154);
    ASSERT_TRUE(!valid, "bad magic rejected");
    fclose(f);

    unlink(path);
    PASS();
}

// ── Test 12: Heat map overflow behavior at uint32_t boundary ──

static void test_heat_overflow(void) {
    TEST("heat_map uint32_t near overflow");

    uint32_t count = UINT32_MAX - 1;
    count++; // UINT32_MAX
    ASSERT_EQ(count, UINT32_MAX, "at max");

    count++; // wraps to 0
    ASSERT_EQ(count, 0, "overflow wraps to 0");

    // In practice: 4.29B increments at ~9400/sec = 127 hours.
    // This is acceptable for MoE inference (sessions are hours, not days).
    PASS();
}

// ── Main ──

int main(void) {
    printf("\n=== MnemoCUDA Expert Heat Profiling Tests ===\n\n");

    test_heat_map_counting();
    test_heat_sorting();
    test_heat_save_load();
    test_heat_load_mismatch();
    test_cache_pinning_logic();
    test_all_slots_pinned();
    test_top_n_stats();
    test_large_scale_sort();
    test_huge_moe_sort();
    test_pin_ratio();
    test_heat_file_bad_magic();
    test_heat_overflow();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
