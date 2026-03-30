# MnemoCUDA
> **Experimental** — Active research project by [AstroLexis](https://github.com/AstrolexisAI). API may change.

**Expert streaming inference engine for MoE models larger than VRAM.**

Run 235B+ parameter Mixture-of-Experts models on consumer GPUs by streaming expert weights from NVMe SSD with intelligent VRAM caching.

## The Problem

Large MoE models like Qwen3-235B have 128+ experts per layer but only activate 8 per token. The full expert weights (128 GB+) don't fit in GPU VRAM, so standard inference engines can't load them.

## The Solution

MnemoCUDA keeps only the small **resident weights** (attention, norms, embeddings — ~5 GB) in VRAM and **streams expert weights on demand** from NVMe SSD. A multi-level cache hierarchy minimizes disk I/O:

```
L1  VRAM Cache      — on-GPU, instant access, LRU + heat-pinned
L2  Prefetch Buffer — pinned host memory, predictively loaded
L3  OS Page Cache   — kernel-managed mmap of expert files
L4  NVMe SSD        — cold reads via pread / page fault
```

### Expert Heat Profiling

MnemoCUDA profiles which experts are activated most frequently and uses this data to:

- **Pin hot experts** in VRAM so they're never evicted (top 50% of cache)
- **Predict next-layer experts** and pre-read them while GPU computes the current layer
- **Persist heat maps** across restarts — the engine gets faster with every session

## Performance

Tested on **Qwen3-235B-A22B** (128 experts/layer, K=8, 94 layers, Q4_K_M):

| Hardware | VRAM Cache | tok/s | TTFT | Notes |
|----------|-----------|-------|------|-------|
| RTX 4090 + RTX 5090 | 46 GB (3,959 slots) | **14.0** | **1.6s** | Warm cache, optimized engine |
| RTX 4090 + RTX 5090 | 46 GB (3,959 slots) | ~12 | 2.5s | Sustained decode, warm cache |
| RTX 4090 + RTX 5090 | 46 GB (3,959 slots) | ~5 | 9.3s | Cold start, first request |

### Per-token breakdown (ncu-verified, warm cache)

| Component | Time | % | Notes |
|-----------|------|---|-------|
| Attention (Q/K/V/O proj + norms + RoPE + attn) | 52ms | 80% | Q4K matvec at 427 GB/s (44% peak BW) |
| MoE experts (gate+up+swiglu+down × 8) | 7ms | 11% | Fused dual+scaled_add kernels |
| Router (matvec + topk + classify + sync) | 3ms | 5% | GPU classification + event sync |
| Cache management | 3ms | 5% | Prefetch, deferred insert |
| Multi-GPU handoff | <1ms | <1% | P2P when available |

### Optimization history (5 → 14 tok/s)

The engine went through extensive optimization across ~20 commits:

- **Sync elimination**: Removed per-layer `cudaStreamSynchronize` barriers, replaced with events
- **GPU-side sampling**: `sample_token_kernel` (top-p nucleus, 2048 candidates) — copies 4 bytes instead of full vocab logits
- **Kernel fusions**: 7,500+ launches eliminated per token — batched QK norms, dual K+V projection, fused O_proj+residual, fused norm+residual-save, fused FP16 KV store, fused expert gate+up matvec
- **CUDA Graphs**: Attention path captured as graphs (8 kernels → 1 graph launch × 94 layers)
- **Vectorized Q4K**: `uint32_t` coalesced loads, branch-free inner loop, `#pragma unroll`
- **GPU embedding dequant**: Q4K/Q3K embedding lookup kernels eliminate per-token malloc/CPU-dequant
- **Warp-parallel kernels**: Router matvec_f32 (32 threads/row), attention Q@K^T (warp-cooperative shuffle)
- **Cache improvements**: GPU-side expert classification, process-as-ready I/O, auto extra-prefetch
- **Infrastructure**: Request queue with worker thread, hash-table tokenizer, RoPE constant memory, pre-created event pool, per-layer cached tensor pointers

**Theoretical floor**: ~25ms/token (12ms compute + 13ms overhead). The Q4K matvec kernel achieves 44% of peak DRAM bandwidth (427 of 1008 GB/s). Reaching 70%+ requires tensor-core dequant or fundamentally different memory access patterns.

## Supported Models

Any GGUF MoE model split with the preparation pipeline (see `tools/`):

- Qwen3-235B, Qwen3-30B-MoE
- **Qwen3.5-122B** (hybrid MoE+SSM with Mamba layers)
- **GLM-4.5-128x9.4B** (MoE)
- DeepSeek-V2/V3
- Mixtral 8x7B, 8x22B
- Any model with `expert_count` in GGUF metadata
- Hybrid MoE+SSM models with `full_attention_interval` are supported

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA (RTX 3060+ recommended, 12 GB+ VRAM)
- CUDA Toolkit 12.x
- Fast NVMe SSD (PCIe 4.0+ recommended, >3 GB/s)
- Linux (kernel 5.x+)
- GCC or compatible C compiler

### Build

```bash
git clone https://github.com/AstrolexisAI/MnemoCUDA.git
cd MnemoCUDA
make
```

The build auto-detects your GPU architecture. Override with `GPU_ARCH=sm_89 make`.

To override the compiler: `CC=gcc CXX=g++ make`.

### Prepare a Model

Split a GGUF MoE model into the streaming format using the tools in `tools/`:

1. `tools/prep_tokenizer.py` — Convert HuggingFace tokenizer to binary format
2. Split model weights into resident + per-layer expert files

See [docs/model-format.md](docs/model-format.md) for the complete file format specification.

The output structure:
```
output_dir/
├── config.json              # Model config (layers, experts, dimensions)
├── resident_weights.bin     # Non-expert tensors (~5 GB)
├── resident_manifest.json   # Tensor index for resident weights
├── expert_manifest.json     # Expert size/layout metadata
├── tokenizer.bin            # BPE tokenizer (binary format)
└── experts/
    ├── layer_00.bin          # Expert weights for layer 0
    ├── layer_01.bin
    └── ...                   # One file per layer
```

### Run

```bash
# HTTP API server (binds to localhost by default)
./build/mnemo_server /path/to/split_model --http 8095

# Public access (use behind reverse proxy in production)
./build/mnemo_server /path/to/split_model --http 8095 --bind 0.0.0.0

# With authentication (or set MNEMO_AUTH_TOKEN env var)
./build/mnemo_server /path/to/split_model --http 8095 --auth YOUR_SECRET_TOKEN

# Interactive REPL
./build/mnemo_server /path/to/split_model --repl

# Single prompt
./build/mnemo_server /path/to/split_model "Explain quantum computing"

# Skip warmup for faster startup
./build/mnemo_server /path/to/split_model --http 8095 --warmup off
```

### HTTP API

```bash
# Streaming generation (SSE)
curl -X POST http://localhost:8095/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 256, "temperature": 0.7, "stream": true}'

# Non-streaming
curl -X POST http://localhost:8095/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 256, "stream": false}'

# Raw prompt (pre-formatted ChatML, skip wrapping)
curl -X POST http://localhost:8095/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", "raw_prompt": true}'

# Health / readiness / status
curl http://localhost:8095/live     # Process alive
curl http://localhost:8095/ready    # Model loaded, ready/busy
curl http://localhost:8095/health   # Alias for /ready
curl http://localhost:8095/status   # Detailed: VRAM, cache, heat, TTFT, tok/s

# Expert heat profiling stats
curl http://localhost:8095/heat
```

With authentication:
```bash
curl -X POST http://localhost:8095/v1/completions \
  -H "Authorization: Bearer YOUR_SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 256}'
```

### Error Codes

The C API returns typed error codes (`MnemoError` enum in `engine.h`):

| Code | Name | Description |
|------|------|-------------|
| 0 | `MNEMO_OK` | Success |
| -1 | `MNEMO_ERR_BAD_CONFIG` | Invalid config or missing model directory |
| -2 | `MNEMO_ERR_TOKENIZER` | Tokenizer load or encode failure |
| -3 | `MNEMO_ERR_NO_GPU` | No CUDA GPUs or invalid GPU IDs |
| -4 | `MNEMO_ERR_CONTEXT_FULL` | Position exceeds context length |
| -5 | `MNEMO_ERR_CUDA` | CUDA runtime error (OOM, etc.) |
| -6 | `MNEMO_ERR_IO` | File I/O error |
| -7 | `MNEMO_ERR_CANCELLED` | Generation cancelled |

Use `mnemo_cuda_strerror(code)` for human-readable messages. The HTTP server maps these to appropriate status codes (400, 500, 503).

## Architecture

### How Expert Streaming Works

```
Token -> Embedding -> [Layer 0..N] -> LM Head -> Next Token
                         |
                    +----+-----+
                    | Attention |  <- Resident in VRAM (always loaded)
                    | RMS Norm  |
                    | Router    |
                    +----+-----+
                         |
                    +----+---------------------+
                    | Router selects K experts  |
                    | from 128 available         |
                    +----+---------------------+
                         |
              +----------+----------+
              v          v          v
         [Expert 42] [Expert 7] [Expert 103]  <- Streamed from NVMe
              |          |          |            on demand, cached in VRAM
              v          v          v
           gate+up -> SwiGLU -> down projection
              |          |          |
              +----------+----------+
                         |
                    Weighted sum -> Residual -> Next Layer
```

### Cache Hierarchy

The VRAM cache uses **heat-aware LRU eviction**:

1. Each expert activation is counted in a **heat map** (layer x expert_id)
2. After warmup, the **hottest 50%** of cache slots are **pinned** -- never evicted
3. A **prefetch engine** reads predicted experts for layer N+1 while GPU computes layer N
4. **CUDA events** (not `cudaStreamSynchronize`) allow upload and compute to truly overlap

### Multi-GPU Support

MnemoCUDA distributes layers across GPUs via pipeline parallelism with **partitioned resident weights** -- each GPU only loads the tensors for its assigned layers:

```
GPU 0 (RTX 4090, 24 GB): Layers 0-46   -- resident + embeddings
GPU 1 (RTX 5090, 32 GB): Layers 47-93  -- resident + output head
```

Hidden state transfers between GPUs use **P2P** (NVLink/PCIe peer access) when available, falling back to pinned host memory.

### Attention Pipeline

Each layer's attention is captured as a **CUDA Graph** at load time (8 fused kernels per graph):

1. `rms_norm_residual` — norm + residual save in one pass
2. `matvec_q4k` — Q projection (vectorized uint32 loads)
3. `matvec_q4k_dual` — K+V projection fused (single x-vector read)
4. `rms_norm_qk` — Q+K head norms in one launch
5. `rope_kernel` — RoPE with constant-memory frequencies + `__sincosf`
6. `f32_to_f16_dual` — K+V → FP16 KV cache in one launch
7. `attention_kernel_f16kv` — tiled online softmax with shared-memory Q cache
8. `matvec_q4k_add` — output projection + residual add fused

Per-token graph param updates (RoPE pos, KV destinations, seq_len) via `cudaGraphExecKernelNodeSetParams`.

### Expert Pipeline

Each expert runs 3 fused kernels (was 5):

1. `matvec_q4k_dual` — gate + up projections (shared x-vector)
2. `cuda_swiglu` — SiLU activation
3. `matvec_q4k_scaled_add` — down projection + weighted accumulation fused

## REPL Commands

| Command | Description |
|---------|-------------|
| `/heat` | Expert heat map: top-20 hottest experts, per-GPU cache stats |
| `/pin`  | Force re-pin hot experts to VRAM cache |
| `/save` | Save heat map to disk (`expert_heat.bin`) |
| `/stats`| Generation stats: tokens, tok/s, VRAM usage |
| `/info` | Model info |
| `/quit` | Exit |

## Configuration

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--http <port>` | -- | Run as HTTP server |
| `--repl` | -- | Interactive mode |
| `--context <n>` | 2048 | Max context length (128-131072) |
| `--bind <addr>` | 127.0.0.1 | Bind address for HTTP server |
| `--warmup <mode>` | full | Warmup mode: `off`, `light` (1 prompt), `full` (6 prompts) |
| `--auth <token>` | -- | Require Bearer token (also via `MNEMO_AUTH_TOKEN` env var) |
| `--kv-int8` | off | INT8 KV cache (halves KV bandwidth vs FP16) |
| `--extra-prefetch` | auto | 2-layer-ahead prefetch (auto-enabled for large MoE models) |

Lower context = more VRAM for expert cache = higher hit rate = faster generation.

## Project Structure

```
MnemoCUDA/
|-- src/
|   |-- engine.c              # Lifecycle, config, sampling, generate, CUDA graph build (~1,750 lines)
|   |-- engine.h              # Public C API (error codes, stats, config)
|   |-- engine_internal.h     # Shared internal types (ModelConfig, GPUState, cache, graphs)
|   |-- forward.c/h           # Forward pass, expert cache, prefetch, attention graphs (~1,700 lines)
|   |-- tokenizer.c/h         # BPE tokenizer with hash-table vocab lookup (~380 lines)
|   |-- io_pool.c/h           # I/O thread pool with wait_any completion (~150 lines)
|   |-- heat.c/h              # Expert heat profiling, pinning, persistence (~240 lines)
|   |-- json_helpers.c/h      # Minimal JSON parser with injection resistance
|   |-- log.h                 # Leveled logging macros (LOG_INFO/WARN/ERROR/DEBUG)
|   |-- kernels.cu            # CUDA kernels: vectorized Q4K, fused attention, sampling, INT8 KV (~2,150 lines)
|   |-- mnemo_server.c        # HTTP/REPL server with request queue + worker thread (~800 lines)
|   |-- async_client.py       # Python async HTTP/SSE client
|-- tests/
|   |-- test_heat.c           # Heat profiling unit tests (12 tests)
|   |-- test_engine.c         # Engine structural + GPU inference tests (12 + 4 tests)
|   |-- test_server.c         # JSON parsing, escaping, injection resistance (17 tests)
|-- tools/
|   |-- prep_tokenizer.py     # Convert HuggingFace tokenizer to binary
|   |-- tokenize.py           # Python tokenizer bridge (subprocess protocol)
|   |-- validate_dequant.py   # Dequantization validation
|   |-- validate_rope_norm.py # RoPE and RMS norm validation
|   |-- benchmark.sh          # Benchmark suite (tok/s, TTFT, hit rates, profiling)
|-- docs/
|   |-- deployment.md         # Production deployment guide (Nginx, systemd, security)
|   |-- troubleshooting.md    # Error codes, performance diagnosis, compatibility
|   |-- model-format.md       # Complete file format specification
|-- Makefile
|-- LICENSE                   # AGPL-3.0
|-- README.md
```

## Test Suite

```bash
make test   # Runs 41 tests (no GPU required for structural tests)

# With real model for GPU inference tests:
MODEL_DIR=/path/to/split_model make test
```

| Suite | Tests | GPU Required |
|-------|-------|:---:|
| `test_heat` | 12 | No |
| `test_engine` (structural) | 12 | No |
| `test_engine` (GPU inference) | 4 | Yes (`MODEL_DIR` env var) |
| `test_server` (JSON + injection) | 17 | No |

GPU smoke tests require `MODEL_DIR=/path/to/split_model` to run load/generate/cancel/heat cycles.

## Quantization Support

| Format | Block Size | Bits/Weight | Supported |
|--------|-----------|-------------|-----------|
| Q4_K   | 256 values | 4.5 bpw | Primary |
| Q6_K   | 256 values | 6.5 bpw | Yes |
| Q8_0   | 32 values  | 8.5 bpw | Yes |
| Q3_K   | 256 values | 3.4 bpw | Yes |
| Q5_K   | 256 values | 5.5 bpw | Yes |
| F32    | 1 value    | 32 bpw  | Yes |

## Deployment

See [docs/deployment.md](docs/deployment.md) for the full production deployment guide.

### Development / Demo

```bash
./build/mnemo_server /path/to/model --http 8095 --warmup light
```

### Production (behind reverse proxy)

```bash
# Set auth token via environment (avoids ps exposure)
export MNEMO_AUTH_TOKEN=$(openssl rand -hex 32)

# Start bound to localhost with auth
./build/mnemo_server /path/to/model --http 8095 --auth $MNEMO_AUTH_TOKEN

# Nginx/Envoy handles TLS, rate limiting, and public access
# upstream mnemo { server 127.0.0.1:8095; }
```

## How It Compares

| Engine | Can run 235B? | Consumer GPU? | Expert Streaming? |
|--------|:---:|:---:|:---:|
| **MnemoCUDA** | Yes | Yes | Heat-profiled, multi-level cache |
| llama.cpp | No (needs full VRAM) | Yes | No |
| vLLM | Yes (needs 8x A100) | No | No |
| Ollama | No (needs full VRAM) | Yes | No |
| ktransformers | Yes | Yes | Basic offloading |

## Known Limitations

- **Cold start is slow**: first prompt after loading processes at ~1 tok/s while VRAM cache fills. Subsequent prompts improve as cache warms up. Use `--warmup full` to pre-fill caches.
- **NVMe speed is critical**: on SATA SSDs or slow NVMe (<2 GB/s), performance degrades significantly. PCIe 4.0+ NVMe recommended.
- **Single-token generation only**: no batch prefill or parallel token processing yet. TTFT scales linearly with prompt length.
- **Single-client HTTP**: the server processes one request at a time and returns 503 when busy. Use a request queue or load balancer for multi-client scenarios.
- **Memory pressure**: large models with 8K+ context consume significant VRAM for KV cache, leaving less room for expert cache slots. Reduce context length for higher cache hit rates.
- **Linux only**: relies on mmap, pread, pthreads, and CUDA. No Windows or macOS support.

See [docs/troubleshooting.md](docs/troubleshooting.md) for common errors and fixes.

## License

AGPL-3.0 -- see [LICENSE](LICENSE).

Created by [AstroLexis](https://github.com/AstrolexisAI).

## Contributing

Contributions welcome. Areas of active research:

- **Speculative decoding** with small draft models on separate hardware
- **Dual-quant VRAM cache** -- Q2_K warm tier for 2x effective cache size
- **io_uring** for kernel-bypass NVMe reads
- **Batch prefill** -- process multiple prompt tokens simultaneously
- **Concurrent HTTP** -- worker threads or async I/O for multi-client serving
