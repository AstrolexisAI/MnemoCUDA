#!/bin/bash
# MnemoCUDA Benchmark — measures tok/s, TTFT, and hit rates
# Usage: MODEL_DIR=/path/to/model ./tools/benchmark.sh [--kv-int8] [--context N]

set -e

if [ -z "$MODEL_DIR" ]; then
    echo "Usage: MODEL_DIR=/path/to/model ./tools/benchmark.sh [--kv-int8] [--context N]"
    exit 1
fi

BUILD_DIR="$(dirname "$0")/../build"
SERVER="$BUILD_DIR/mnemo_server"

if [ ! -f "$SERVER" ]; then
    echo "Building..."
    make -C "$(dirname "$0")/.." -j$(nproc) 2>/dev/null
fi

EXTRA_FLAGS="$@"

SHORT_PROMPT="Explain quantum computing in one paragraph."
LONG_PROMPT="Write a detailed analysis of the economic implications of artificial intelligence adoption across different industries, covering manufacturing, healthcare, finance, education, and transportation. For each industry, discuss the potential benefits, risks, job displacement concerns, and timeline for significant impact. Consider both developed and developing economies."

echo "============================================"
echo "MnemoCUDA Benchmark"
echo "Model: $MODEL_DIR"
echo "Flags: $EXTRA_FLAGS"
echo "============================================"
echo ""

# Test 1: Short prompt, warm cache (2 runs — first warms, second measures)
echo "--- Test 1: Short prompt (warm cache) ---"
echo "Warming cache..."
MNEMO_PROFILE=1 "$SERVER" "$MODEL_DIR" --context 2048 $EXTRA_FLAGS "$SHORT_PROMPT" --max-tokens 8 2>&1 | grep -E 'PROF|tok/s|Done:|VRAM|cache' | tail -5
echo ""
echo "Measuring..."
MNEMO_PROFILE=1 "$SERVER" "$MODEL_DIR" --context 2048 $EXTRA_FLAGS "$SHORT_PROMPT" --max-tokens 32 2>&1 | tee /tmp/mnemo_bench_short.log | grep -E 'PROF|tok/s|Done:|VRAM|cache|hit|Worst'
echo ""

# Test 2: Long prompt (TTFT test)
echo "--- Test 2: Long prompt (TTFT) ---"
MNEMO_PROFILE=1 "$SERVER" "$MODEL_DIR" --context 2048 $EXTRA_FLAGS "$LONG_PROMPT" --max-tokens 32 2>&1 | tee /tmp/mnemo_bench_long.log | grep -E 'PROF|tok/s|Done:|prefill|TTFT|Prompt:'
echo ""

# Test 3: With extra prefetch
echo "--- Test 3: Extra prefetch ---"
MNEMO_PROFILE=1 "$SERVER" "$MODEL_DIR" --context 2048 --extra-prefetch $EXTRA_FLAGS "$SHORT_PROMPT" --max-tokens 32 2>&1 | tee /tmp/mnemo_bench_prefetch.log | grep -E 'PROF|tok/s|Done:|hit|Worst'
echo ""

# Summary
echo "============================================"
echo "Logs saved to /tmp/mnemo_bench_*.log"
echo "Full profiling: grep 'PROF' /tmp/mnemo_bench_short.log"
echo "============================================"
