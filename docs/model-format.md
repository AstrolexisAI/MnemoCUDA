# MnemoCUDA Model Format

## Directory Structure

A split model directory must contain:

```
model_dir/
├── config.json              # Required: model architecture metadata
├── resident_weights.bin     # Required: non-expert weight tensors
├── resident_manifest.json   # Required: tensor index (name, offset, size, type)
├── tokenizer.bin            # Required: binary BPE tokenizer
├── expert_manifest.json     # Optional: expert size/layout metadata
├── expert_heat.bin          # Auto-generated: persistent heat map
└── experts/
    ├── layer_00.bin         # Required: expert weights for layer 0
    ├── layer_01.bin         # Required: expert weights for layer 1
    └── ...                  # One file per transformer layer with MoE
```

## config.json

GGUF-compatible metadata. Architecture prefix is auto-detected from key names.

### Required Keys

| Key | Example | Description |
|-----|---------|-------------|
| `{prefix}.embedding_length` | 4096 | Hidden size |
| `{prefix}.block_count` | 94 | Number of transformer layers |
| `{prefix}.attention.head_count` | 64 | Query attention heads |
| `{prefix}.attention.head_count_kv` | 4 | Key/value attention heads |

### Optional Keys (with defaults)

| Key | Default | Description |
|-----|---------|-------------|
| `{prefix}.expert_feed_forward_length` | 1536 | MoE intermediate size |
| `{prefix}.attention.key_length` | 128 | Head dimension |
| `{prefix}.expert_count` | 128 | Experts per layer |
| `{prefix}.expert_used_count` | 8 | Active experts per token (K) |
| `{prefix}.rope.freq_base` | 1000000 | RoPE theta |
| `{prefix}.attention.layer_norm_rms_epsilon` | 1e-6 | RMS norm epsilon |
| `{prefix}.context_length` | 40960 | Model's max context |
| `tokenizer.ggml.tokens_count` or `vocab_size` | 151936 | Vocabulary size |

Supported prefixes: `qwen3moe`, `qwen3next`, `llama`, `qwen2`.

## resident_manifest.json

JSON array of tensor entries:

```json
{
    "tensors": [
        {
            "name": "token_embd.weight",
            "offset": 0,
            "size": 12345678,
            "type_id": 12,
            "dims": [4096, 151936]
        },
        ...
    ]
}
```

### Type IDs

| ID | Format | Block Size | Bytes/Block |
|----|--------|-----------|-------------|
| 0  | F32    | 1         | 4           |
| 8  | Q8_0   | 32        | 34          |
| 11 | Q3_K   | 256       | 110         |
| 12 | Q4_K   | 256       | 144         |
| 13 | Q5_K   | 256       | —           |
| 14 | Q6_K   | 256       | 210         |

### Tensor Names

Per-layer tensors use the format `blk.{N}.{component}`:

| Tensor | Description |
|--------|-------------|
| `blk.N.attn_norm.weight` | Pre-attention RMS norm |
| `blk.N.attn_q.weight` | Query projection |
| `blk.N.attn_k.weight` | Key projection |
| `blk.N.attn_v.weight` | Value projection |
| `blk.N.attn_q_norm.weight` | QK norm (optional, Qwen3) |
| `blk.N.attn_k_norm.weight` | QK norm (optional, Qwen3) |
| `blk.N.attn_output.weight` | Output projection |
| `blk.N.ffn_norm.weight` | Pre-FFN RMS norm |
| `blk.N.ffn_gate_inp.weight` | Router weights |
| `token_embd.weight` | Token embeddings |
| `output_norm.weight` | Final RMS norm |
| `output.weight` | LM head |

### Validation

The loader validates:
- All tensor `offset + size` must fit within `resident_weights.bin`
- Required fields in config.json must be present
- Expert file sizes must match `n_experts * expert_size`

## Expert Files

Each `experts/layer_XX.bin` contains all experts for that layer concatenated:

```
[expert_0_data][expert_1_data]...[expert_N_data]
```

Expert data layout: `[gate_weights][up_weights][down_weights]`

Sizes from `expert_manifest.json` or computed as `file_size / n_experts`.

## tokenizer.bin

Binary format produced by `tools/prep_tokenizer.py`:

```
Header (28 bytes):
    magic:       uint32 = 0x4D544F4B ("MTOK")
    vocab_size:  uint32
    n_merges:    uint32
    n_special:   uint32
    eos_id:      uint32
    im_start_id: uint32
    im_end_id:   uint32

Vocab (variable):
    For each token: [uint16 len][bytes...]

Merges (variable):
    For each merge: [uint16 len]["tokenA tokenB"]

Special tokens (variable):
    For each: [uint32 id][uint16 len][bytes...]
```

## expert_heat.bin

Persistent heat map, auto-saved on unload:

```
Header (20 bytes):
    magic:       uint32 = 0x48454154 ("HEAT")
    n_layers:    uint32
    n_experts:   uint32
    total_tokens: uint64

Data:
    uint32[n_layers * n_experts] — activation counts
```
