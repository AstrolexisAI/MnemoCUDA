# What Is MnemoCUDA?

## Overview

MnemoCUDA is an **expert-streaming inference engine for Mixture-of-Experts (MoE) language models** that are too large to fit fully in GPU VRAM.

Its purpose is to make very large MoE models practical on **consumer NVIDIA GPUs** by keeping only the always-needed model weights in VRAM and streaming expert weights from fast NVMe storage on demand.

In short:

- **Resident weights** stay in GPU memory
- **Expert weights** stay on disk until needed
- **Hot experts** are cached in VRAM
- **Predicted experts** are prefetched into host memory
- **Heat profiling** improves cache behavior over time

This allows models that would normally require datacenter-scale GPU memory to run on much smaller and cheaper hardware.

## The Problem It Solves

Large MoE models activate only a small subset of experts per token, but traditional inference engines still assume the full model should be resident in memory.

That creates a mismatch:

- total model size is enormous
- per-token active working set is much smaller
- VRAM is the limiting resource

MnemoCUDA is built around the idea that **you do not need all expert weights in VRAM at once**.

## Core Idea

MnemoCUDA splits model execution into two classes of weights:

- **Resident weights**
  - attention
  - norms
  - router weights
  - embeddings
  - output head
- **Expert weights**
  - MoE FFN experts, loaded only when selected by the router

At runtime, the engine:

1. loads resident weights into GPU memory
2. runs attention and routing on GPU
3. finds the active experts for the current token
4. checks whether those experts are already in VRAM cache
5. if not, reads them from disk into pinned host memory
6. uploads them to GPU
7. runs expert computation
8. updates heat/cache statistics for future tokens

## Cache Hierarchy

MnemoCUDA uses a multi-level hierarchy:

- **L1: VRAM cache**
  - active and recently used experts
  - LRU eviction with heat-aware pinning
- **L2: Prefetch buffer**
  - pinned host memory for predicted next-layer experts
- **L3: OS page cache**
  - file-backed caching through `mmap`
- **L4: NVMe SSD**
  - cold expert storage

This is the heart of the design. The engine tries to keep the active working set close to the GPU while relying on SSD only when needed.

## What Makes It Different

MnemoCUDA is not just an HTTP wrapper around CUDA kernels. Its differentiator is the runtime strategy:

- expert streaming from disk
- VRAM expert cache
- persistent heat profiling
- prefetch based on recent and historical activation patterns
- partitioned resident weights for multi-GPU execution
- tiled attention that supports long contexts without shared-memory blowups

The point is not only “run a model,” but “run a model that normally would not fit.”

## Multi-GPU Model

MnemoCUDA supports multiple GPUs using pipeline-style layer partitioning.

Each GPU gets:

- the resident tensors for the layers it owns
- global tensors it actually needs
- its own KV cache
- its own expert cache

Hidden state is transferred between GPUs using pinned host memory.

This keeps the design simpler than a fully distributed inference engine while still giving meaningful scaling and VRAM savings.

## Heat Profiling

MnemoCUDA tracks which experts are used most often.

That data is used to:

- pin hot experts in VRAM
- improve eviction decisions
- improve prefetch decisions
- persist behavior across restarts

As a result, the engine can get faster after warmup and across repeated sessions.

## Current Interfaces

MnemoCUDA currently exposes:

- a C API in [`src/engine.h`](/home/curly/MnemoCUDA/src/engine.h)
- an HTTP server in [`src/mnemo_server.c`](/home/curly/MnemoCUDA/src/mnemo_server.c)
- an async Python client in [`src/async_client.py`](/home/curly/MnemoCUDA/src/async_client.py)

The HTTP server supports:

- `/v1/completions`
- `/live`
- `/ready`
- `/health`
- `/status`
- `/heat`

It also supports:

- localhost-only bind by default
- optional Bearer auth
- warmup modes
- structured request logging

## Model Format

MnemoCUDA expects a split model directory containing:

- `config.json`
- `resident_weights.bin`
- `resident_manifest.json`
- `tokenizer.bin`
- `experts/layer_XX.bin`
- optionally `expert_manifest.json`

See [`docs/model-format.md`](/home/curly/MnemoCUDA/docs/model-format.md) for the full specification.

## What It Is Good For

MnemoCUDA is well-suited for:

- very large MoE models
- cost-sensitive deployments
- single-model dedicated serving
- technical demos and pilots
- research into expert locality and cache behavior

It is especially attractive where:

- model size exceeds VRAM
- NVMe is fast
- latency is acceptable in exchange for much lower hardware cost

## What It Is Not

MnemoCUDA is not currently:

- a generic high-throughput multi-tenant inference platform
- a batch-optimized serving engine
- a drop-in replacement for large distributed serving systems
- a mature enterprise orchestration stack by itself

Today it is best understood as a **specialized MoE inference engine** with a strong cost/performance story.

## Current Maturity

MnemoCUDA is significantly more mature than an experimental prototype:

- modularized engine
- validated error codes
- health/readiness endpoints
- safer load-time validation
- improved CUDA failure handling
- structured logging
- deployment and troubleshooting docs

At the same time, it is still best described as:

- strong for demos
- strong for controlled pilots
- improving toward production hardening

## Why It Matters

The strategic value of MnemoCUDA is simple:

it makes it possible to run classes of models that are usually associated with expensive datacenter infrastructure on far cheaper hardware, by exploiting the sparse activation structure of MoE models instead of treating them like dense models.

That is the central idea, and everything else in the project exists to make that idea practical.
