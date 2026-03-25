# MnemoCUDA

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

| Hardware | VRAM Cache | Hit Rate | tok/s | TTFT |
|----------|-----------|----------|-------|------|
| RTX 4090 + RTX 5090 | 43 GB (3,690 slots) | 88% | ~5 | 9.3s |
| RTX 4090 (single) | 17 GB (1,478 slots) | ~75% | ~2.5 | ~18s |

For comparison, this model normally requires 8× A100 80GB ($100K+ of GPUs). MnemoCUDA runs it on **$2,600 of consumer hardware**.

## Supported Models

Any GGUF MoE model split with `tools/gguf_expert_split.py`:

- Qwen3-235B, Qwen3-30B-MoE
- DeepSeek-V2/V3
- Mixtral 8x7B, 8x22B
- Any model with `expert_count` in GGUF metadata

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA (RTX 3060+ recommended, 12 GB+ VRAM)
- CUDA Toolkit 12.x
- Fast NVMe SSD (PCIe 4.0+ recommended, >3 GB/s)
- Linux (kernel 5.x+)

### Build

```bash
git clone https://github.com/AstrolexisAI/MnemoCUDA.git
cd MnemoCUDA
make
```

The build auto-detects your GPU architecture. Override with `GPU_ARCH=sm_89 make`.

### Prepare a Model

Split a GGUF MoE model into the streaming format:

```bash
pip install gguf numpy
python tools/gguf_expert_split.py /path/to/model.gguf /path/to/output_dir/
```

This produces:
```
output_dir/
├── config.json              # Model config (layers, experts, dimensions)
├── resident_weights.bin     # Non-expert tensors (~5 GB)
├── tokenizer.bin            # BPE tokenizer (binary format)
└── experts/
    ├── layer_00.bin          # Expert weights for layer 0
    ├── layer_01.bin
    └── ...                   # One file per layer
```

### Run

```bash
# HTTP API server (production)
./build/mnemo_server /path/to/split_model --http 8095

# Interactive REPL
./build/mnemo_server /path/to/split_model --repl

# Single prompt
./build/mnemo_server /path/to/split_model "Explain quantum computing"
```

### HTTP API

```bash
# Streaming generation (SSE)
curl -X POST http://localhost:8095/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 256, "temperature": 0.7, "stream": true}'

# Expert heat profiling stats
curl http://localhost:8095/heat
```

## Architecture

### How Expert Streaming Works

```
Token → Embedding → [Layer 0..N] → LM Head → Next Token
                         │
                    ┌────┴─────┐
                    │ Attention │  ← Resident in VRAM (always loaded)
                    │ RMS Norm  │
                    │ Router    │
                    └────┬─────┘
                         │
                    ┌────┴─────────────────────┐
                    │ Router selects K experts  │
                    │ from 128 available         │
                    └────┬─────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         [Expert 42] [Expert 7] [Expert 103]  ← Streamed from NVMe
              │          │          │            on demand, cached in VRAM
              ▼          ▼          ▼
           gate+up → SwiGLU → down projection
              │          │          │
              └──────────┼──────────┘
                         │
                    Weighted sum → Residual → Next Layer
```

### Cache Hierarchy

The VRAM cache uses **heat-aware LRU eviction**:

1. Each expert activation is counted in a **heat map** (layer × expert_id)
2. After warmup, the **hottest 50%** of cache slots are **pinned** — never evicted
3. A **prefetch engine** reads predicted experts for layer N+1 while GPU computes layer N
4. **CUDA events** (not `cudaStreamSynchronize`) allow upload and compute to truly overlap

### Multi-GPU Support

MnemoCUDA distributes layers across GPUs via pipeline parallelism:

```
GPU 0 (RTX 4090, 24 GB): Layers 0-46   — 17 GB VRAM cache
GPU 1 (RTX 5090, 32 GB): Layers 47-93  — 26 GB VRAM cache
```

Hidden state transfers between GPUs use pinned host memory.

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
| `--http <port>` | — | Run as HTTP server |
| `--repl` | — | Interactive mode |
| `--context <n>` | 8192 | Max context length (affects KV cache VRAM usage) |

Lower context = more VRAM for expert cache = higher hit rate = faster generation.

## Files

```
MnemoCUDA/
├── src/
│   ├── engine.c           # Core inference engine (2,500+ lines)
│   ├── engine.h           # Public C API
│   ├── kernels.cu         # CUDA kernels (matvec, attention, SwiGLU, RoPE, topK)
│   ├── mnemo_server.c     # HTTP/REPL server
│   └── async_client.py    # Python async HTTP client
├── tests/
│   ├── test_heat.c        # Heat profiling unit tests (12 tests)
│   └── test_engine.c      # Engine integration test
├── tools/
│   ├── prep_tokenizer.py  # Convert HuggingFace tokenizer to binary
│   ├── tokenize.py        # Standalone tokenizer test
│   ├── validate_dequant.py    # Dequantization validation
│   └── validate_rope_norm.py  # RoPE and RMS norm validation
├── Makefile
├── LICENSE                # AGPL-3.0
└── README.md
```

## Quantization Support

| Format | Block Size | Bits/Weight | Supported |
|--------|-----------|-------------|-----------|
| Q4_K   | 256 values | 4.5 bpw | ✅ Primary |
| Q6_K   | 256 values | 6.5 bpw | ✅ |
| Q8_0   | 32 values  | 8.5 bpw | ✅ |
| Q3_K   | 256 values | 3.4 bpw | ✅ |
| Q5_K   | 256 values | 5.5 bpw | ✅ |
| F32    | 1 value    | 32 bpw  | ✅ |

## How It Compares

| Engine | Can run 235B? | Consumer GPU? | Expert Streaming? |
|--------|:---:|:---:|:---:|
| **MnemoCUDA** | ✅ | ✅ | ✅ Heat-profiled, multi-level cache |
| llama.cpp | ❌ (needs full VRAM) | ✅ | ❌ |
| vLLM | ✅ (needs 8× A100) | ❌ | ❌ |
| Ollama | ❌ (needs full VRAM) | ✅ | ❌ |
| ktransformers | ✅ | ✅ | ✅ Basic offloading |

## License

AGPL-3.0 — see [LICENSE](LICENSE).

Created by [AstroLexis](https://github.com/AstrolexisAI).

## Contributing

Contributions welcome. Areas of active research:

- **Speculative decoding** with small draft models on separate hardware
- **Dual-quant VRAM cache** — Q2_K warm tier for 2× effective cache size
- **io_uring** for kernel-bypass NVMe reads
- **Batch prefill** — process multiple prompt tokens simultaneously
- **Layer importance scoring** — skip redundant MoE layers
