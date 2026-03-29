# MnemoCUDA Troubleshooting

## Common Errors

### No CUDA GPUs detected (error -3)

```
[ERROR] No CUDA GPUs detected
Load failed: -3
```

**Causes:**
- CUDA drivers not installed or not loaded
- GPU not visible (container without `--gpus` flag)
- `nvidia-smi` returns no devices

**Fix:**
```bash
# Check GPU visibility
nvidia-smi

# In Docker, ensure GPU passthrough
docker run --gpus all ...

# Check CUDA version
nvcc --version
```

### Config missing required fields (error -1)

```
[ERROR] Missing required config key: qwen3moe.embedding_length
[ERROR] Config missing 1 required field(s), cannot load
```

**Causes:**
- config.json doesn't match expected GGUF metadata format
- Wrong architecture prefix (model is llama but detected as qwen3moe)

**Fix:**
- Verify config.json contains the required keys for the model architecture
- Required: `embedding_length`, `block_count`, `attention.head_count`, `attention.head_count_kv`
- Check the auto-detected prefix in the log output

### Tokenizer not loaded (error -2)

```
[WARN] No tokenizer.bin — run prep_tokenizer.py first
[WARN] Tokenizer not loaded, inference will fail
```

**Fix:**
```bash
python3 tools/prep_tokenizer.py /path/to/hf_model /path/to/output_dir/
```

The tokenizer.bin must be in the model directory alongside config.json.

### CUDA out of memory (error -5)

```
[ERROR] CUDA error in load at engine.c:XXX: out of memory
```

**Causes:**
- Context length too large for available VRAM
- Another process using GPU memory

**Fix:**
```bash
# Reduce context length
./build/mnemo_server model --http 8095 --context 4096

# Check GPU memory usage
nvidia-smi

# Kill other GPU processes if needed
```

### Context full (error -4)

```
[MnemoCUDA] Context full at 8192 tokens
```

The prompt + generated tokens exceeded `--context` limit.

**Fix:** Increase `--context` or use shorter prompts. Note: larger context reduces expert cache space.

### Tensor offset mismatch

```
[ERROR] Tensor 'blk.0.attn_q.weight' offset+size exceeds resident file
```

**Cause:** `resident_manifest.json` and `resident_weights.bin` are from different model preparations.

**Fix:** Re-run the model preparation pipeline to regenerate both files together.

### Expert file too small

```
[ERROR] Layer 5 expert file too small: 1234 bytes, expected 5678
```

**Cause:** Corrupted or truncated expert layer file.

**Fix:** Re-split the model. Verify disk space during preparation.

### I/O error loading expert

```
[MnemoCUDA] I/O error loading layer 42 expert 7, skipping
```

**Cause:** Disk read failure (bad sector, NVMe issue, filesystem error).

**Fix:**
- Check `dmesg` for disk errors
- Run filesystem check
- Replace faulty SSD

### Server returns 503 Busy

```json
{"error":"Server busy - generation in progress","code":503}
```

The server is single-threaded. Only one generation request is processed at a time.

**Fix:** Queue requests at the client or proxy level. Use `/ready` endpoint to check availability.

### Server returns 401 Unauthorized

```json
{"error":"Unauthorized","code":401}
```

**Fix:**
```bash
# Include Bearer token in request
curl -H "Authorization: Bearer YOUR_TOKEN" ...

# Or set via environment
export MNEMO_AUTH_TOKEN=your_token
```

## Performance Issues

### Low cache hit rate (<80%)

**Symptoms:** Slow tok/s, many L4 (NVMe) reads in logs.

**Fix:**
- Reduce `--context` to free VRAM for expert cache
- Use `--warmup full` to pre-fill caches
- Check `/status` for `cache_used` vs `cache_slots`
- Ensure NVMe is PCIe 4.0+ with >3 GB/s read speed

### Slow first token (high TTFT)

**Causes:**
- Cold expert cache after restart
- Large prompt (linear prefill time)
- Slow NVMe

**Fix:**
- Use `--warmup full` for diverse cache warming
- Check `/status` for `heat_tokens` — more profiled tokens = better predictions
- Monitor `last_ttft` in `/status`

### NVMe bottleneck

**Diagnosis:**
```bash
# Check NVMe speed
fio --filename=/path/to/model/experts/layer_00.bin \
    --rw=randread --bs=12M --numjobs=8 --size=1G \
    --runtime=10 --name=test
```

Target: >3 GB/s sequential, >1 GB/s random 12MB reads.

## Compatibility

### Tested Configurations

| OS | CUDA | Compiler | GPUs |
|----|------|----------|------|
| Ubuntu 22.04/24.04 | 12.x | gcc 12-14 | RTX 3060+ |
| Fedora 39+ | 12.x | gcc 13-14 | RTX 4090, 5090 |

### Requirements

- Linux with kernel 5.x+
- CUDA Toolkit 12.0+
- NVIDIA driver 535+
- GCC or compatible C compiler
- NVMe SSD (PCIe 4.0+ recommended)
