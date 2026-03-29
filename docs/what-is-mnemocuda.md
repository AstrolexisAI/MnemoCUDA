# What Is MnemoCUDA?

## Executive overview
MnemoCUDA is an inference engine that lets very large Mixture-of-Experts (MoE)
language models run on consumer-grade NVIDIA GPUs by **streaming expert weights from
NVMe storage while keeping only the resident tensors in VRAM**. Rather than assuming
the entire model must be resident, MnemoCUDA treats experts like a working set that is
loaded on demand and cached in decreasingly fast tiers.

The project targets pilot/demo-grade and emerging enterprise deployments where low
latency is less important than being able to run huge models without datacenter
footprints. Productivity matters: it exposes a C API, a hardened HTTP server, tests,
and documentation covering deployment, troubleshooting, and model formats.

## Architecture at a glance

- **Resident tensors** (attention, norms, router, embeddings, output) are partitioned
  per GPU and bound into evenly sized buffers aligned to 256 bytes. Each GPU only
  loads what it needs, reducing VRAM use by roughly `(n_gpus - 1) * layer_weight`.
- **Expert tensors** stay on disk until the router selects them. The engine maintains
  a three-tier cache: VRAM (hot experts + pinned heat), host pinned memory (prefetch
  buffers), and the OS page cache / NVMe storage.
- **Heat profiling** tracks activation frequency for pinning and eviction decisions.
  Persisted heat data makes the cache smarter after warmup.
- **Forward pass** is extracted into `forward.c` with helper layers for expert cache
  lookup/insert, prefetch submission, and batched I/O. Tensor reads go through a
  unified `tensor_ptr_on_gpu()` that understands the new offset map.

## Expert streaming pipeline

1. Load resident weights, router, and tokenizer during initialization (`engine.c`).
2. For each token, run attention and routing on the partitions of each GPU it owns.
3. Determine the active experts, check VRAM cache, and if missing submit I/O tasks
   through the `io_pool` with retry-safe `read_full`/`pread_full` and batch-aware
   error reporting.
4. Prefetch predicted experts into host pinned memory while overlapping compute.
5. Upload or pin experts in VRAM, run expert kernels with tiled attention safeguards,
   and update heat statistics.
6. Evict cold experts via LRU/heat heuristics, clamp `expert_k` to `MNEMO_MAX_EXPERT_K`
   (default 16) to avoid OOB.

Every stage reacts to failures: CUDA calls are wrapped with `CUDA_LOAD_CHECK`, I/O
errors set flags that skip failing experts, and decode errors surface through a
`gen_error` returned by `mnemo_cuda_generate()`.

## Multi-GPU, observability, and telemetry

- GPUs own sequential layers (e.g., GPU 0 has layers 0..L0, GPU 1 has L0+1..L1).
  Global tensors go to GPU 0 while outputs go to the final GPU. Tensor offset bounds
  are validated against `resident_weights.bin` and `expert_size` entries from the
  manifest.
- Every GPU has its own KV cache, expert cache, heat stats, and `tensor_offsets`
  remapping table, so no GPU copies unneeded tensors.
- `/status`, `/health`, `/ready`, `/live`, and `/heat` expose TTFT, prompt tokens,
  cache slots, pinned counts, and generation metrics. Logs include structured `req=N`
  fields, ISO timestamps, and error strings via `mnemo_cuda_strerror()`.
- Numeric metrics cover TTFT, prompt tokens consumed, warmup progress, token/s, heat
  stats, and VRAM usage (excluding host-only prefetch buffers).

## Reliability, error handling, and tests

- Input validation enforces ranges for `n_gpus`, `context_length`, `gpu_ids`,
  `expert_k`, HTTP port, and bind address.
- `mnemo_cuda_unload()` is idempotent: resources are nulled after freeing, and `n_gpus`
  is zeroed to stop loops. Load errors clean up immediately and return specific
  `MnemoError` codes (`MNEMO_ERR_FATAL`, `MNEMO_ERR_UNSUPPORTED`, …
  `MNEMO_ERR_RESOURCE`). HTTP 500/503 reflect runtime failures, SSE sends terminal
  error events, and non-streaming requests get status-specific JSON.
- Legacy `fprintf` calls are replaced with structured `LOG_` macros (`LOG_ERROR`,
  `LOG_WARN`, `LOG_INFO`), and `write_full()` plus `read_full()` guarantee pipe
  integrity with retries for partial reads/writes.
- A suite of 24 tests covers engine lifecycle, config parsing, tensor classification,
  error code handling, stats defaults, config validation, and GPU/non-GPU paths.

## Security and deployment hardening

- HTTP server defaults to `127.0.0.1`, accepts `--bind`, enforces Bearer token auth via
  `MNEMO_AUTH_TOKEN`, rejects unknown paths, and handles `OPTIONS`/CORS.
- Prompts are parsed safely with `json_extract_string()`/`json_escape()` supporting
  full escape sequences and preventing truncation; inputs that start with `<|im_start|>`
  bypass ChatML wrapping via the `raw_prompt` flag.
- Socket timeouts (30s `SO_RCVTIMEO`) protect against slow clients, and `max_tokens`
  caps ensure KV cache limits are respected.
- Docs cover deployment patterns (Nginx, systemd, tuning, security checklist),
  troubleshooting (error codes + fixes), and the complete model format.

## Who should look at MnemoCUDA?

- Teams building pilots or demos of MoE models on constrained hardware.
- Researchers studying expert locality, heat profiling, or cache trade-offs.
- Organizations that need a C API or HTTP/SSE frontend with clear error codes,
  authentication, and observability hooks before committing to larger serving
  infrastructure.

MnemoCUDA is not a drop-in distributed serving system; it is instead a specialized
engine that **keeps hardware costs low by streaming MoE experts while still
providing enterprise-level validation, security, and logging**.
