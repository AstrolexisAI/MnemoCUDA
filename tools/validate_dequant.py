#!/usr/bin/env python3
"""
MnemoCUDA Dequantization Validation Script

Compares our CUDA kernel dequantization logic (reimplemented in Python) against
the gguf library's reference dequantization to find which kernel has bugs.

Validates: Q4_K, Q5_K, Q6_K, Q8_0 dequant + F32 column-major matvec.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

# gguf library (ground truth)
from gguf import GGUFReader, dequantize
from gguf.quants import Q4_K, Q5_K, Q6_K, Q8_0

# ── Paths ──

SPLIT_DIR = Path("/home/curly/kulvex-models/mnemo/mark5-mid-split")
GGUF_PATH = Path("/home/curly/kulvex-models/mnemo/mark5-mid.gguf")
RESIDENT_BIN = SPLIT_DIR / "resident_weights.bin"
MANIFEST_JSON = SPLIT_DIR / "resident_manifest.json"

# ── Load manifest ──

def load_manifest():
    with open(MANIFEST_JSON) as f:
        data = json.load(f)
    return {t["name"]: t for t in data["tensors"]}

# ── Read raw bytes from resident_weights.bin ──

def read_raw_tensor(manifest_entry):
    offset = manifest_entry["offset"]
    size = manifest_entry["size"]
    with open(RESIDENT_BIN, "rb") as f:
        f.seek(offset)
        return f.read(size)

# ── f16 to f32 (matches CUDA f16_to_f32) ──

def f16_to_f32(raw_u16):
    """Convert a uint16 holding an IEEE 754 half-float to float32."""
    return np.frombuffer(np.array([raw_u16], dtype=np.uint16).tobytes(), dtype=np.float16)[0].astype(np.float32)

# ── get_scale_min_k4 (matches CUDA exactly) ──

def get_scale_min_k4(j, scales):
    """Decode sub-block scale and min from the 12-byte packed scales array (Q4_K / Q5_K)."""
    if j < 4:
        sc = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, m


# ============================================================
# Q4_K dequantization — mirrors matvec_q4k kernel
# ============================================================

def dequant_q4k_cuda_style(raw_bytes, n_rows, n_cols):
    """
    Dequantize Q4_K raw bytes using the same algorithm as our CUDA kernel.
    Block layout: [d:2][dmin:2][scales:12][qs:128] = 144 bytes, 256 values.
    Returns float32 array of shape (n_rows, n_cols).
    """
    block_size = 144  # Q4_K type_size
    blocks_per_row = (n_cols + 255) // 256
    result = np.zeros((n_rows, n_cols), dtype=np.float32)

    for row in range(n_rows):
        for b in range(blocks_per_row):
            blk_offset = (row * blocks_per_row + b) * block_size
            blk = raw_bytes[blk_offset:blk_offset + block_size]

            d = f16_to_f32(struct.unpack_from("<H", blk, 0)[0])
            dmin = f16_to_f32(struct.unpack_from("<H", blk, 2)[0])
            scales = blk[4:16]   # 12 bytes
            qs = blk[16:144]     # 128 bytes

            base = b * 256
            q_ptr = 0  # index into qs
            is_idx = 0

            for j in range(0, 256, 64):
                sc1, m1 = get_scale_min_k4(is_idx, scales)
                d1 = d * sc1
                m1f = dmin * m1
                sc2, m2 = get_scale_min_k4(is_idx + 1, scales)
                d2 = d * sc2
                m2f = dmin * m2

                for l in range(32):
                    col = base + j + l
                    if col >= n_cols:
                        break
                    result[row, col] = d1 * (qs[q_ptr + l] & 0xF) - m1f

                for l in range(32):
                    col = base + j + 32 + l
                    if col >= n_cols:
                        break
                    result[row, col] = d2 * (qs[q_ptr + l] >> 4) - m2f

                q_ptr += 32
                is_idx += 2

    return result


# ============================================================
# Q5_K dequantization — mirrors matvec_q5k kernel
# ============================================================

def dequant_q5k_cuda_style(raw_bytes, n_rows, n_cols):
    """
    Dequantize Q5_K raw bytes using the same algorithm as our CUDA kernel.
    Block layout: [d:2][dmin:2][scales:12][qh:32][qs:128] = 176 bytes, 256 values.
    """
    block_size = 176
    blocks_per_row = (n_cols + 255) // 256
    result = np.zeros((n_rows, n_cols), dtype=np.float32)

    for row in range(n_rows):
        for b in range(blocks_per_row):
            blk_offset = (row * blocks_per_row + b) * block_size
            blk = raw_bytes[blk_offset:blk_offset + block_size]

            d = f16_to_f32(struct.unpack_from("<H", blk, 0)[0])
            dmin = f16_to_f32(struct.unpack_from("<H", blk, 2)[0])
            scales = blk[4:16]    # 12 bytes
            qh = blk[16:48]       # 32 bytes
            qs = blk[48:176]      # 128 bytes

            base = b * 256
            q_ptr = 0
            is_idx = 0

            for j in range(0, 256, 64):
                sc1, m1 = get_scale_min_k4(is_idx, scales)
                d1 = d * sc1
                m1f = dmin * m1
                sc2, m2 = get_scale_min_k4(is_idx + 1, scales)
                d2 = d * sc2
                m2f = dmin * m2

                for l in range(32):
                    col = base + j + l
                    if col >= n_cols:
                        break
                    lo4 = qs[q_ptr + l] & 0xF
                    hi_byte_idx = (j + l) // 8
                    hi_bit_idx = (j + l) % 8
                    hi = (qh[hi_byte_idx] >> hi_bit_idx) & 1
                    result[row, col] = d1 * (lo4 | (hi << 4)) - m1f

                for l in range(32):
                    col = base + j + 32 + l
                    if col >= n_cols:
                        break
                    lo4 = qs[q_ptr + l] >> 4
                    hi_byte_idx = (j + 32 + l) // 8
                    hi_bit_idx = (j + 32 + l) % 8
                    hi = (qh[hi_byte_idx] >> hi_bit_idx) & 1
                    result[row, col] = d2 * (lo4 | (hi << 4)) - m2f

                q_ptr += 32
                is_idx += 2

    return result


# ============================================================
# Q6_K dequantization — mirrors matvec_q6k kernel
# ============================================================

def dequant_q6k_cuda_style(raw_bytes, n_rows, n_cols):
    """
    Dequantize Q6_K raw bytes using the same algorithm as our CUDA kernel.
    Block layout: [ql:128][qh:64][scales:16][d:2] = 210 bytes, 256 values.

    This reproduces the EXACT indexing from kernels.cu matvec_q6k:
      lo4 = (l % 2 == 0) ? (ql[qli + group*8 + l/2] & 0xF)
                          : (ql[qli + group*8 + l/2] >> 4);
      hi2 = (qh[qhi + l] >> (group * 2)) & 0x03;
    """
    block_size = 210
    blocks_per_row = (n_cols + 255) // 256
    result = np.zeros((n_rows, n_cols), dtype=np.float32)

    for row in range(n_rows):
        for b in range(blocks_per_row):
            blk_offset = (row * blocks_per_row + b) * block_size
            blk = raw_bytes[blk_offset:blk_offset + block_size]

            ql = blk[0:128]
            qh = blk[128:192]
            scales_raw = blk[192:208]
            scales = np.frombuffer(scales_raw, dtype=np.int8)
            d = f16_to_f32(struct.unpack_from("<H", blk, 208)[0])

            base = b * 256

            for half in range(2):
                qli = half * 64
                qhi = half * 32
                sci = half * 8
                ei = half * 128

                for group in range(4):
                    scale = d * float(scales[sci + group])
                    for l in range(16):
                        idx = ei + group * 32 + l
                        col = base + idx
                        if col >= n_cols:
                            continue
                        # CUDA kernel indexing (verbatim from kernels.cu):
                        byte_idx = qli + group * 8 + l // 2
                        if l % 2 == 0:
                            lo4 = ql[byte_idx] & 0xF
                        else:
                            lo4 = ql[byte_idx] >> 4
                        hi2 = (qh[qhi + l] >> (group * 2)) & 0x03
                        val = np.int8(np.uint8((hi2 << 4) | lo4)) - 32
                        result[row, col] = scale * val

                    scale2 = d * float(scales[sci + group + 4])
                    for l in range(16):
                        idx = ei + group * 32 + 16 + l
                        col = base + idx
                        if col >= n_cols:
                            continue
                        nl = 16 + l
                        byte_idx = qli + group * 8 + nl // 2
                        if nl % 2 == 0:
                            lo4 = ql[byte_idx] & 0xF
                        else:
                            lo4 = ql[byte_idx] >> 4
                        hi2 = (qh[qhi + 16 + l] >> (group * 2)) & 0x03
                        val = np.int8(np.uint8((hi2 << 4) | lo4)) - 32
                        result[row, col] = scale2 * val

    return result


def dequant_q6k_reference(raw_bytes, n_rows, n_cols):
    """
    Dequantize Q6_K using the CORRECT algorithm (matching gguf library).
    Block layout: [ql:128][qh:64][scales:16][d:2] = 210 bytes, 256 values.

    CORRECT ql indexing (from gguf library):
      ql is reshaped (2, 64), then split into lo/hi nibbles:
        values[0..31]   = ql[0:32]   & 0xF   (lo nibbles of first 32 bytes)
        values[32..63]  = ql[32:64]  & 0xF   (lo nibbles of next 32 bytes)
        values[64..95]  = ql[0:32]   >> 4     (hi nibbles of first 32 bytes)
        values[96..127] = ql[32:64]  >> 4     (hi nibbles of next 32 bytes)
      (repeated for second half: ql[64:128])

    CORRECT qh indexing:
      qh is reshaped (2, 32), then shifted by [0,2,4,6]:
        hi2[val] = (qh[val%32] >> (2*(val//32))) & 3   (within each half)
    """
    block_size = 210
    blocks_per_row = (n_cols + 255) // 256
    result = np.zeros((n_rows, n_cols), dtype=np.float32)

    for row in range(n_rows):
        for b in range(blocks_per_row):
            blk_offset = (row * blocks_per_row + b) * block_size
            blk = raw_bytes[blk_offset:blk_offset + block_size]

            ql = np.frombuffer(blk[0:128], dtype=np.uint8)
            qh = np.frombuffer(blk[128:192], dtype=np.uint8)
            scales = np.frombuffer(blk[192:208], dtype=np.int8)
            d = f16_to_f32(struct.unpack_from("<H", blk, 208)[0])

            base = b * 256

            # Decode ql: 128 bytes -> 256 lo4 values
            # gguf layout: reshape(2, 1, 64) >> [0, 4] & 0xF -> reshape(8, 32)
            lo4_all = np.zeros(256, dtype=np.uint8)
            for half in range(2):
                ql_half = ql[half * 64:(half + 1) * 64]
                # lo nibbles: values 0..31 and 32..63
                lo4_all[half * 128 + 0:half * 128 + 32] = ql_half[0:32] & 0xF
                lo4_all[half * 128 + 32:half * 128 + 64] = ql_half[32:64] & 0xF
                # hi nibbles: values 64..95 and 96..127
                lo4_all[half * 128 + 64:half * 128 + 96] = ql_half[0:32] >> 4
                lo4_all[half * 128 + 96:half * 128 + 128] = ql_half[32:64] >> 4

            # Decode qh: 64 bytes -> 256 hi2 values
            hi2_all = np.zeros(256, dtype=np.uint8)
            for half in range(2):
                qh_half = qh[half * 32:(half + 1) * 32]
                for shift in range(4):
                    hi2_all[half * 128 + shift * 32:half * 128 + (shift + 1) * 32] = \
                        (qh_half >> (shift * 2)) & 0x03

            # Combine: q = (hi2 << 4 | lo4) as int8 - 32
            q_vals = (hi2_all.astype(np.uint8) << 4) | lo4_all.astype(np.uint8)
            q_vals = q_vals.view(np.int8).astype(np.float32) - 32.0

            # Apply scales: 16 groups of 16 values
            for g in range(16):
                scale = d * float(scales[g])
                for l in range(16):
                    idx = g * 16 + l
                    col = base + idx
                    if col < n_cols:
                        result[row, col] = scale * q_vals[idx]

    return result


# ============================================================
# Q8_0 dequantization — mirrors matvec_q8_0 kernel
# ============================================================

def dequant_q8_0_cuda_style(raw_bytes, n_rows, n_cols):
    """
    Dequantize Q8_0 raw bytes.
    Block layout: [d:2][qs:32] = 34 bytes, 32 values.
    """
    block_size = 34
    blocks_per_row = (n_cols + 31) // 32
    result = np.zeros((n_rows, n_cols), dtype=np.float32)

    for row in range(n_rows):
        for b in range(blocks_per_row):
            blk_offset = (row * blocks_per_row + b) * block_size
            blk = raw_bytes[blk_offset:blk_offset + block_size]

            d = f16_to_f32(struct.unpack_from("<H", blk, 0)[0])
            qs = np.frombuffer(blk[2:34], dtype=np.int8)
            base_col = b * 32

            for j in range(32):
                col = base_col + j
                if col >= n_cols:
                    break
                result[row, col] = d * float(qs[j])

    return result


# ============================================================
# F32 column-major matvec — mirrors matvec_f32 kernel
# ============================================================

def matvec_f32_colmajor(weights_bytes, n_rows, n_cols, x):
    """
    Column-major matvec: y[row] = sum_col(W[col * n_rows + row] * x[col])
    weights_bytes is the raw F32 data stored column-major.
    """
    W = np.frombuffer(weights_bytes, dtype=np.float32).reshape((n_cols, n_rows)).T
    # W is now (n_rows, n_cols) but laid out column-major, so W[row, col] = data[col * n_rows + row]
    # Actually: reshape((n_cols, n_rows)) gives W_cm[col, row], so .T gives (n_rows, n_cols) row-major
    # But that's the same as reading column-major. Let's just do the explicit sum:
    W_flat = np.frombuffer(weights_bytes, dtype=np.float32)
    y = np.zeros(n_rows, dtype=np.float32)
    for col in range(n_cols):
        y += W_flat[col * n_rows:(col + 1) * n_rows] * x[col]
    return y


# ============================================================
# Comparison helpers
# ============================================================

def compare_tensors(name, our_vals, ref_vals, max_show=10):
    """Compare two float arrays and report statistics."""
    assert our_vals.shape == ref_vals.shape, \
        f"Shape mismatch: ours={our_vals.shape} vs ref={ref_vals.shape}"

    diff = np.abs(our_vals.astype(np.float64) - ref_vals.astype(np.float64))
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    n_nonzero = np.count_nonzero(diff > 1e-7)
    total = diff.size

    passed = max_err < 1e-5
    status = "PASS" if passed else "FAIL"

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Status:         {status}")
    print(f"  Shape:          {our_vals.shape}")
    print(f"  Max abs error:  {max_err:.2e}")
    print(f"  Mean abs error: {mean_err:.2e}")
    print(f"  Elements differ (>1e-7): {n_nonzero} / {total}")

    if n_nonzero > 0 and not passed:
        # Show worst offenders
        flat_diff = diff.flatten()
        worst_indices = np.argsort(flat_diff)[-max_show:][::-1]
        print(f"\n  Top {min(max_show, n_nonzero)} worst elements:")
        print(f"  {'Index':<12} {'Ours':>14} {'Reference':>14} {'AbsDiff':>14}")
        for idx in worst_indices:
            if flat_diff[idx] < 1e-7:
                break
            multi_idx = np.unravel_index(idx, our_vals.shape)
            print(f"  {str(multi_idx):<12} {our_vals.flat[idx]:>14.6f} "
                  f"{ref_vals.flat[idx]:>14.6f} {flat_diff[idx]:>14.2e}")

    return passed


# ============================================================
# Main validation
# ============================================================

def main():
    print("MnemoCUDA Dequantization Validator")
    print("=" * 70)
    print(f"Split model:   {SPLIT_DIR}")
    print(f"Original GGUF: {GGUF_PATH}")
    print()

    # Load manifest
    manifest = load_manifest()

    # Load GGUF reader (ground truth)
    print("Loading GGUF reader (memory-mapped)...")
    reader = GGUFReader(str(GGUF_PATH))
    gguf_tensors = {t.name: t for t in reader.tensors}

    # We test a subset of rows to keep runtime reasonable
    # (full tensors can be huge; we validate the first N rows)
    MAX_ROWS = 8  # enough to catch systematic bugs

    results = []

    # ── Test 1: Q4_K — blk.0.attn_q.weight ──

    tensor_name = "blk.0.attn_q.weight"
    print(f"\n--- Testing {tensor_name} (Q4_K) ---")
    entry = manifest[tensor_name]
    n_cols, n_rows = entry["dims"]  # GGUF dims: [n_cols, n_rows] (column count first)
    print(f"  Manifest dims: {entry['dims']} -> n_rows={n_rows}, n_cols={n_cols}")

    # Reference: gguf library dequantization
    gt = gguf_tensors[tensor_name]
    ref_full = dequantize(gt.data, gt.tensor_type)  # shape: (n_rows, n_cols) after gguf's reshape
    print(f"  gguf dequantized shape: {ref_full.shape}")
    ref = ref_full[:MAX_ROWS]

    # Our dequant: read raw bytes from split, dequant first MAX_ROWS
    raw = read_raw_tensor(entry)
    blocks_per_row = (n_cols + 255) // 256
    bytes_per_row = blocks_per_row * 144  # Q4_K type_size
    raw_subset = raw[:MAX_ROWS * bytes_per_row]
    ours = dequant_q4k_cuda_style(raw_subset, MAX_ROWS, n_cols)

    passed = compare_tensors(f"{tensor_name} (Q4_K) — CUDA-style vs gguf", ours, ref)
    results.append(("Q4_K", tensor_name, passed))

    # ── Test 2: Q6_K — blk.0.attn_v.weight ──

    tensor_name = "blk.0.attn_v.weight"
    print(f"\n--- Testing {tensor_name} (Q6_K) ---")
    entry = manifest[tensor_name]
    n_cols, n_rows = entry["dims"]
    print(f"  Manifest dims: {entry['dims']} -> n_rows={n_rows}, n_cols={n_cols}")

    gt = gguf_tensors[tensor_name]
    ref_full = dequantize(gt.data, gt.tensor_type)
    print(f"  gguf dequantized shape: {ref_full.shape}")
    ref = ref_full[:MAX_ROWS]

    raw = read_raw_tensor(entry)
    blocks_per_row = (n_cols + 255) // 256
    bytes_per_row = blocks_per_row * 210  # Q6_K type_size
    raw_subset = raw[:MAX_ROWS * bytes_per_row]

    # Test CUDA-style (buggy) kernel
    ours_cuda = dequant_q6k_cuda_style(raw_subset, MAX_ROWS, n_cols)
    passed_cuda = compare_tensors(
        f"{tensor_name} (Q6_K) — CUDA kernel logic vs gguf", ours_cuda, ref)
    results.append(("Q6_K (CUDA)", tensor_name, passed_cuda))

    # Test correct reference implementation
    ours_fixed = dequant_q6k_reference(raw_subset, MAX_ROWS, n_cols)
    passed_fixed = compare_tensors(
        f"{tensor_name} (Q6_K) — FIXED dequant vs gguf", ours_fixed, ref)
    results.append(("Q6_K (fixed)", tensor_name, passed_fixed))

    if not passed_cuda and passed_fixed:
        print(f"\n  ROOT CAUSE: Q6_K CUDA kernel has WRONG ql indexing.")
        print(f"    CUDA kernel does: ql[qli + group*8 + l/2], nibble select by l%2")
        print(f"    This treats each byte as 2 consecutive values (nibble-packed pairs).")
        print(f"    CORRECT layout (gguf/llama.cpp):")
        print(f"      ql is split into 64-byte halves, each giving 128 values:")
        print(f"        vals[0:32]   = ql[0:32]  & 0xF")
        print(f"        vals[32:64]  = ql[32:64] & 0xF")
        print(f"        vals[64:96]  = ql[0:32]  >> 4")
        print(f"        vals[96:128] = ql[32:64] >> 4")
        print(f"    FIX: Rewrite CUDA ql indexing to match the interleaved layout.")

    # ── Test 3: Q6_K — output.weight (large tensor, test first rows) ──

    tensor_name = "output.weight"
    print(f"\n--- Testing {tensor_name} (Q6_K) ---")
    entry = manifest[tensor_name]
    n_cols, n_rows = entry["dims"]
    print(f"  Manifest dims: {entry['dims']} -> n_rows={n_rows}, n_cols={n_cols}")

    gt = gguf_tensors[tensor_name]
    ref_full = dequantize(gt.data, gt.tensor_type)
    print(f"  gguf dequantized shape: {ref_full.shape}")
    ref = ref_full[:MAX_ROWS]

    raw = read_raw_tensor(entry)
    blocks_per_row = (n_cols + 255) // 256
    bytes_per_row = blocks_per_row * 210
    raw_subset = raw[:MAX_ROWS * bytes_per_row]

    # Only test CUDA-style here (the bug is the same)
    ours = dequant_q6k_cuda_style(raw_subset, MAX_ROWS, n_cols)
    passed = compare_tensors(f"{tensor_name} (Q6_K) — CUDA kernel logic vs gguf", ours, ref)
    results.append(("Q6_K (CUDA)", tensor_name, passed))

    # ── Test 4: Find a Q5_K tensor ──

    q5k_name = None
    for tname, tentry in manifest.items():
        if tentry["type"] == "Q5_K":
            q5k_name = tname
            break

    if q5k_name:
        tensor_name = q5k_name
        print(f"\n--- Testing {tensor_name} (Q5_K) ---")
        entry = manifest[tensor_name]
        n_cols, n_rows = entry["dims"]
        print(f"  Manifest dims: {entry['dims']} -> n_rows={n_rows}, n_cols={n_cols}")

        gt = gguf_tensors[tensor_name]
        ref_full = dequantize(gt.data, gt.tensor_type)
        print(f"  gguf dequantized shape: {ref_full.shape}")
        ref = ref_full[:MAX_ROWS]

        raw = read_raw_tensor(entry)
        blocks_per_row = (n_cols + 255) // 256
        bytes_per_row = blocks_per_row * 176
        raw_subset = raw[:MAX_ROWS * bytes_per_row]
        ours = dequant_q5k_cuda_style(raw_subset, MAX_ROWS, n_cols)

        passed = compare_tensors(f"{tensor_name} (Q5_K) — CUDA-style vs gguf", ours, ref)
        results.append(("Q5_K", tensor_name, passed))
    else:
        # Also check GGUF directly (model may not use Q5_K at all)
        q5k_in_gguf = any(t.tensor_type == 13 for t in reader.tensors)
        if q5k_in_gguf:
            print("\n--- Q5_K found in GGUF but not in resident manifest, skipping ---")
        else:
            print("\n--- No Q5_K tensors in this model (Q5_K kernel untestable) ---")
            print("  Note: Q5_K kernel shares scale/min decoding with Q4_K (get_scale_min_k4).")
            print("  The qh bit extraction logic is unique and cannot be validated here.")

    # ── Test 5: F32 matvec — blk.0.ffn_gate_inp.weight ──

    tensor_name = "blk.0.ffn_gate_inp.weight"
    print(f"\n--- Testing {tensor_name} (F32 matvec) ---")
    entry = manifest[tensor_name]
    n_cols, n_rows = entry["dims"]
    print(f"  Manifest dims: {entry['dims']} -> n_rows={n_rows}, n_cols={n_cols}")

    # Read from both sources
    raw = read_raw_tensor(entry)
    gt = gguf_tensors[tensor_name]
    ref_data = dequantize(gt.data, gt.tensor_type)  # F32 -> identity dequant
    print(f"  gguf dequantized shape: {ref_data.shape}")

    # Diagnose storage layout
    our_f32 = np.frombuffer(raw, dtype=np.float32)
    ref_f32 = ref_data.flatten()

    is_rowmajor = np.allclose(our_f32.reshape(n_rows, n_cols), ref_data, atol=1e-7)
    is_colmajor = np.allclose(our_f32.reshape(n_cols, n_rows).T, ref_data, atol=1e-7)
    print(f"  Raw data is row-major (n_rows x n_cols)?   {is_rowmajor}")
    print(f"  Raw data is column-major (needs transpose)? {is_colmajor}")
    if is_rowmajor:
        print("  ** GGUF stores F32 in ROW-MAJOR order.")
        print("  ** CUDA kernel matvec_f32 assumes COLUMN-MAJOR -> BUG!")

    # Test matvec both ways
    np.random.seed(42)
    x = np.random.randn(n_cols).astype(np.float32)

    # Our column-major matvec (matches CUDA kernel logic)
    y_colmajor = matvec_f32_colmajor(raw, n_rows, n_cols, x)

    # Row-major matvec (correct for GGUF data)
    W_rowmajor = our_f32.reshape(n_rows, n_cols)
    y_rowmajor = W_rowmajor @ x

    # Reference
    W_ref = ref_data  # (n_rows, n_cols) row-major
    y_ref = W_ref @ x

    passed_colmajor = np.allclose(y_colmajor, y_ref, atol=1e-4)
    passed_rowmajor = np.allclose(y_rowmajor, y_ref, atol=1e-4)
    print(f"  Column-major matvec matches ref? {passed_colmajor}")
    print(f"  Row-major matvec matches ref?    {passed_rowmajor}")

    passed = compare_tensors(
        f"{tensor_name} (F32 matvec) — CUDA column-major kernel vs reference",
        y_colmajor, y_ref
    )
    results.append(("F32 matvec", tensor_name, passed))

    if passed_rowmajor and not passed_colmajor:
        print(f"\n  ROOT CAUSE: CUDA matvec_f32 uses column-major indexing")
        print(f"    weights[col * n_rows + row]")
        print(f"  but GGUF stores F32 tensors in row-major order:")
        print(f"    weights[row * n_cols + col]")
        print(f"  FIX: Change kernel to row-major, or transpose data at load time.")

    # ── Test 6: Verify raw bytes from split match original GGUF ──

    print(f"\n--- Byte-level verification: split vs original GGUF ---")
    byte_checks = [
        ("blk.0.attn_q.weight", "Q4_K"),
        ("blk.0.attn_v.weight", "Q6_K"),
        ("blk.0.ffn_gate_inp.weight", "F32"),
    ]
    for tname, ttype in byte_checks:
        entry = manifest[tname]
        raw = read_raw_tensor(entry)
        gt = gguf_tensors[tname]
        gt_raw = bytes(gt.data)
        # Compare first 4096 bytes (or less)
        check_len = min(len(raw), len(gt_raw), 4096)
        match = raw[:check_len] == gt_raw[:check_len]
        print(f"  {tname} ({ttype}): first {check_len} bytes match = {match}")
        if not match:
            # Find first mismatch
            for i in range(check_len):
                if raw[i] != gt_raw[i]:
                    print(f"    First mismatch at byte {i}: split=0x{raw[i]:02x} vs gguf=0x{gt_raw[i]:02x}")
                    break

    # ── Summary ──

    print("\n")
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_passed = True
    for kernel, tensor, passed in results:
        status = "PASS" if passed else "FAIL"
        marker = "  " if passed else ">>"
        print(f"  {marker} [{status}] {kernel:12s} — {tensor}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All kernels match the gguf reference. Dequantization is correct.")
    else:
        print("  BUGS FOUND: some kernels do NOT match the gguf reference.")
        print("  Check the detailed output above for which elements differ.")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
