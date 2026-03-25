#!/usr/bin/env python3
"""
Validate MnemoCUDA RoPE and RMS Norm kernels against reference implementations.

Tests the exact algorithms from kernels.cu (rms_norm_kernel, rope_kernel)
by reimplementing them in NumPy and comparing against known-correct reference
formulas.
"""

import numpy as np
import sys

# ─── Configuration (Qwen3-235B / Qwen3-Coder-Next) ───
HEAD_DIM = 128
N_HEADS_Q = 16     # query heads (GQA)
N_HEADS_K = 2      # KV heads
THETA = 1_000_000.0  # RoPE base frequency
HIDDEN_DIM = 2048  # for RMS norm test
EPS_DEFAULT = 1e-6
EPS_QWEN3 = 9.999999974752427e-07  # exact Qwen3 config value

PASS = 0
FAIL = 0


def report(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")


# ═══════════════════════════════════════════════════════════
# RMS NORM VALIDATION
# ═══════════════════════════════════════════════════════════

def reference_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Standard RMS norm: output[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def kernel_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """
    Exact replica of rms_norm_kernel from kernels.cu:
      scale = rsqrt(sum(x^2) / n + eps)
      output[i] = x[i] * scale * weight[i]

    Note: rsqrt(a) = 1/sqrt(a), and sum/n = mean, so this should be
    mathematically equivalent to the reference.
    """
    n = len(x)
    sum_sq = np.sum(x ** 2)
    scale = 1.0 / np.sqrt(sum_sq / float(n) + eps)
    return x * scale * weight


def test_rms_norm():
    print("\n══════ RMS NORM TESTS ══════")

    # Test 1: Sequential values [1, 2, 3, ..., 2048]
    print("\n── Test 1: Sequential values (dim=2048) ──")
    x = np.arange(1.0, HIDDEN_DIM + 1, dtype=np.float64)
    w = np.ones(HIDDEN_DIM, dtype=np.float64)

    ref = reference_rms_norm(x, w, EPS_DEFAULT)
    kern = kernel_rms_norm(x, w, EPS_DEFAULT)
    max_diff = np.max(np.abs(ref - kern))
    report("Reference vs kernel (unit weights)", max_diff < 1e-12,
           f"max_diff={max_diff:.2e}")

    # Test 2: Random weights
    print("\n── Test 2: Random input + random weights ──")
    rng = np.random.default_rng(42)
    x = rng.standard_normal(HIDDEN_DIM)
    w = rng.standard_normal(HIDDEN_DIM)

    ref = reference_rms_norm(x, w, EPS_DEFAULT)
    kern = kernel_rms_norm(x, w, EPS_DEFAULT)
    max_diff = np.max(np.abs(ref - kern))
    report("Reference vs kernel (random)", max_diff < 1e-12,
           f"max_diff={max_diff:.2e}")

    # Test 3: Near-zero input (eps matters)
    print("\n── Test 3: Near-zero input (eps dominates) ──")
    x = np.full(HIDDEN_DIM, 1e-10, dtype=np.float64)
    w = np.ones(HIDDEN_DIM, dtype=np.float64)

    ref = reference_rms_norm(x, w, EPS_DEFAULT)
    kern = kernel_rms_norm(x, w, EPS_DEFAULT)
    max_diff = np.max(np.abs(ref - kern))
    report("Near-zero input", max_diff < 1e-12,
           f"max_diff={max_diff:.2e}")

    # Test 4: Qwen3 exact eps
    print("\n── Test 4: Qwen3 eps (9.999999974752427e-07) vs 1e-6 ──")
    x = rng.standard_normal(HIDDEN_DIM)
    w = rng.standard_normal(HIDDEN_DIM)

    out_default = kernel_rms_norm(x, w, EPS_DEFAULT)
    out_qwen3 = kernel_rms_norm(x, w, EPS_QWEN3)
    diff = np.max(np.abs(out_default - out_qwen3))
    report(f"eps=1e-6 vs eps={EPS_QWEN3}", True,
           f"max_diff={diff:.2e} (informational — both are ~1e-6)")

    # Verify the eps values are practically identical
    eps_diff = abs(EPS_DEFAULT - EPS_QWEN3)
    report(f"Eps difference is negligible", eps_diff < 1e-12,
           f"|1e-6 - {EPS_QWEN3}| = {eps_diff:.2e}")

    # Test 5: f32 precision simulation (the kernel runs in f32)
    print("\n── Test 5: f32 precision (simulating CUDA kernel) ──")
    x_f32 = rng.standard_normal(HIDDEN_DIM).astype(np.float32)
    w_f32 = rng.standard_normal(HIDDEN_DIM).astype(np.float32)

    ref_f64 = reference_rms_norm(x_f32.astype(np.float64), w_f32.astype(np.float64), EPS_DEFAULT)
    kern_f32 = kernel_rms_norm(x_f32, w_f32, np.float32(EPS_DEFAULT))
    max_diff = np.max(np.abs(ref_f64 - kern_f32.astype(np.float64)))
    report("f32 kernel vs f64 reference", max_diff < 1e-4,
           f"max_diff={max_diff:.2e} (f32 precision loss expected)")

    # Test 6: All zeros (should not NaN/inf with eps)
    print("\n── Test 6: All-zero input (eps prevents division by zero) ──")
    x = np.zeros(HIDDEN_DIM, dtype=np.float64)
    w = np.ones(HIDDEN_DIM, dtype=np.float64)

    kern = kernel_rms_norm(x, w, EPS_DEFAULT)
    report("No NaN/Inf in output", not np.any(np.isnan(kern)) and not np.any(np.isinf(kern)),
           f"output range: [{kern.min():.2e}, {kern.max():.2e}]")
    report("Zero input → zero output", np.allclose(kern, 0.0),
           f"max_abs={np.max(np.abs(kern)):.2e}")


# ═══════════════════════════════════════════════════════════
# RoPE VALIDATION
# ═══════════════════════════════════════════════════════════

def reference_rope(vec: np.ndarray, n_heads: int, head_dim: int,
                   pos: int, theta: float) -> np.ndarray:
    """
    Standard RoPE reference (llama.cpp / HuggingFace convention):
    For each head, for each pair index j in [0, head_dim/2):
      freq = 1 / theta^(2j / head_dim)
      angle = pos * freq
      new[j]            = old[j] * cos(angle) - old[j + half] * sin(angle)
      new[j + half]     = old[j] * sin(angle) + old[j + half] * cos(angle)
    """
    out = vec.copy()
    half = head_dim // 2
    for h in range(n_heads):
        base = h * head_dim
        for j in range(half):
            freq = 1.0 / (theta ** (2.0 * j / head_dim))
            angle = pos * freq
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            v0 = vec[base + j]
            v1 = vec[base + j + half]
            out[base + j]        = v0 * cos_a - v1 * sin_a
            out[base + j + half] = v0 * sin_a + v1 * cos_a
    return out


def kernel_rope(vec: np.ndarray, n_heads: int, head_dim: int,
                pos: int, theta: float) -> np.ndarray:
    """
    Exact replica of rope_kernel from kernels.cu.
    The kernel processes threads tid in [0, n_heads * half_dim).
    Each thread:
      head = tid / half_dim
      j = tid % half_dim
      freq = 1 / pow(theta, 2*j / head_dim)
      angle = pos * freq
      idx = head * head_dim + j
      v0 = vec[idx], v1 = vec[idx + half_dim]
      vec[idx] = v0 * cos - v1 * sin
      vec[idx + half] = v0 * sin + v1 * cos
    """
    out = vec.copy()
    half_dim = head_dim // 2
    for tid in range(n_heads * half_dim):
        head = tid // half_dim
        j = tid % half_dim
        freq = 1.0 / (theta ** ((2.0 * j) / head_dim))
        angle = pos * freq
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        idx = head * head_dim + j
        v0 = out[idx]
        v1 = out[idx + half_dim]
        out[idx]            = v0 * cos_a - v1 * sin_a
        out[idx + half_dim] = v0 * sin_a + v1 * cos_a
    return out


def test_rope():
    print("\n══════ RoPE TESTS ══════")
    rng = np.random.default_rng(123)

    # Test 1: Reference vs kernel implementation (position 0)
    print("\n── Test 1: Position 0 (should be identity for freq=0 pair) ──")
    q = rng.standard_normal(N_HEADS_Q * HEAD_DIM)
    k = rng.standard_normal(N_HEADS_K * HEAD_DIM)

    q_ref = reference_rope(q, N_HEADS_Q, HEAD_DIM, 0, THETA)
    q_kern = kernel_rope(q, N_HEADS_Q, HEAD_DIM, 0, THETA)
    max_diff_q = np.max(np.abs(q_ref - q_kern))
    report("Q vectors at pos=0 (ref vs kernel)", max_diff_q < 1e-12,
           f"max_diff={max_diff_q:.2e}")

    k_ref = reference_rope(k, N_HEADS_K, HEAD_DIM, 0, THETA)
    k_kern = kernel_rope(k, N_HEADS_K, HEAD_DIM, 0, THETA)
    max_diff_k = np.max(np.abs(k_ref - k_kern))
    report("K vectors at pos=0 (ref vs kernel)", max_diff_k < 1e-12,
           f"max_diff={max_diff_k:.2e}")

    # At position 0, angle=0 for all freqs → cos=1, sin=0 → output should equal input
    max_diff_identity = np.max(np.abs(q_ref - q))
    report("Pos=0 is identity transform", max_diff_identity < 1e-12,
           f"max_diff_from_input={max_diff_identity:.2e}")

    # Test 2: Position 5 (non-trivial rotation)
    print("\n── Test 2: Position 5 ──")
    q = rng.standard_normal(N_HEADS_Q * HEAD_DIM)
    k = rng.standard_normal(N_HEADS_K * HEAD_DIM)

    q_ref = reference_rope(q, N_HEADS_Q, HEAD_DIM, 5, THETA)
    q_kern = kernel_rope(q, N_HEADS_Q, HEAD_DIM, 5, THETA)
    max_diff_q = np.max(np.abs(q_ref - q_kern))
    report("Q vectors at pos=5 (ref vs kernel)", max_diff_q < 1e-12,
           f"max_diff={max_diff_q:.2e}")

    k_ref = reference_rope(k, N_HEADS_K, HEAD_DIM, 5, THETA)
    k_kern = kernel_rope(k, N_HEADS_K, HEAD_DIM, 5, THETA)
    max_diff_k = np.max(np.abs(k_ref - k_kern))
    report("K vectors at pos=5 (ref vs kernel)", max_diff_k < 1e-12,
           f"max_diff={max_diff_k:.2e}")

    # Verify it's NOT identity (values should have changed)
    changed = np.max(np.abs(q_ref - q))
    report("Pos=5 actually modifies values", changed > 1e-6,
           f"max_change={changed:.4e}")

    # Test 3: Multi-head correctness (heads are independent)
    print("\n── Test 3: Multi-head independence ──")
    q_full = rng.standard_normal(N_HEADS_Q * HEAD_DIM)

    # Apply RoPE to all heads
    q_all = kernel_rope(q_full, N_HEADS_Q, HEAD_DIM, 5, THETA)

    # Apply RoPE to each head individually
    all_match = True
    max_head_diff = 0.0
    for h in range(N_HEADS_Q):
        single = q_full[h * HEAD_DIM:(h + 1) * HEAD_DIM].copy()
        single_rope = kernel_rope(single, 1, HEAD_DIM, 5, THETA)
        head_slice = q_all[h * HEAD_DIM:(h + 1) * HEAD_DIM]
        diff = np.max(np.abs(single_rope - head_slice))
        max_head_diff = max(max_head_diff, diff)
        if diff > 1e-12:
            all_match = False
    report("Each head independent (single vs batch)", all_match,
           f"max_diff={max_head_diff:.2e}")

    # Test 4: The kernel handles Q and K in one launch
    print("\n── Test 4: Combined Q+K processing (kernel dispatch) ──")
    q = rng.standard_normal(N_HEADS_Q * HEAD_DIM)
    k = rng.standard_normal(N_HEADS_K * HEAD_DIM)
    pos = 42

    # Simulate the kernel's combined processing:
    # tid in [0, (N_HEADS_Q + N_HEADS_K) * half_dim)
    # First N_HEADS_Q * half_dim threads → Q
    # Remaining N_HEADS_K * half_dim threads → K
    q_out = q.copy()
    k_out = k.copy()
    half_dim = HEAD_DIM // 2
    total = (N_HEADS_Q + N_HEADS_K) * half_dim

    for tid in range(total):
        if tid < N_HEADS_Q * half_dim:
            vec = q_out
            head = tid // half_dim
            j = tid % half_dim
        else:
            vec = k_out
            offset = tid - N_HEADS_Q * half_dim
            head = offset // half_dim
            j = offset % half_dim

        freq = 1.0 / (THETA ** ((2.0 * j) / HEAD_DIM))
        angle = pos * freq
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        idx = head * HEAD_DIM + j
        v0 = vec[idx]
        v1 = vec[idx + half_dim]
        vec[idx]            = v0 * cos_a - v1 * sin_a
        vec[idx + half_dim] = v0 * sin_a + v1 * cos_a

    # Compare with separate application
    q_ref = reference_rope(q, N_HEADS_Q, HEAD_DIM, pos, THETA)
    k_ref = reference_rope(k, N_HEADS_K, HEAD_DIM, pos, THETA)

    q_diff = np.max(np.abs(q_out - q_ref))
    k_diff = np.max(np.abs(k_out - k_ref))
    report("Combined Q dispatch matches reference", q_diff < 1e-12,
           f"max_diff={q_diff:.2e}")
    report("Combined K dispatch matches reference", k_diff < 1e-12,
           f"max_diff={k_diff:.2e}")

    # Test 5: Rotation preserves vector magnitude
    print("\n── Test 5: RoPE preserves magnitude (rotation property) ──")
    q = rng.standard_normal(N_HEADS_Q * HEAD_DIM)
    q_rotated = kernel_rope(q, N_HEADS_Q, HEAD_DIM, 100, THETA)

    for h in range(N_HEADS_Q):
        orig_norm = np.linalg.norm(q[h * HEAD_DIM:(h + 1) * HEAD_DIM])
        rot_norm = np.linalg.norm(q_rotated[h * HEAD_DIM:(h + 1) * HEAD_DIM])
        rel_diff = abs(orig_norm - rot_norm) / orig_norm
        if rel_diff > 1e-10:
            report(f"Head {h} magnitude preservation", False,
                   f"orig={orig_norm:.6f} rot={rot_norm:.6f} rel_diff={rel_diff:.2e}")
            break
    else:
        report("All heads preserve magnitude", True,
               f"checked {N_HEADS_Q} heads, all within 1e-10 relative")

    # Test 6: f32 precision simulation
    print("\n── Test 6: f32 precision (simulating CUDA) ──")
    q_f32 = rng.standard_normal(N_HEADS_Q * HEAD_DIM).astype(np.float32)
    test_pos = 500

    # f64 reference (using f32 input values but f64 math — best possible answer)
    q_ref_f64 = reference_rope(q_f32.astype(np.float64), N_HEADS_Q, HEAD_DIM, test_pos, THETA)

    # f32 kernel simulation (matching CUDA's powf, cosf, sinf)
    q_kern_f32 = q_f32.copy()
    for tid in range(N_HEADS_Q * half_dim):
        head = tid // half_dim
        j = tid % half_dim
        # CUDA: powf(theta, (float)(2*j) / (float)head_dim) — all in f32
        exponent = np.float32(np.float32(2 * j) / np.float32(HEAD_DIM))
        denom = np.float32(np.power(np.float32(THETA), np.float64(exponent)))
        freq = np.float32(np.float32(1.0) / denom)
        angle = np.float32(np.float32(test_pos) * freq)
        cos_a = np.float32(np.cos(np.float32(angle)))
        sin_a = np.float32(np.sin(np.float32(angle)))

        idx = head * HEAD_DIM + j
        v0 = q_kern_f32[idx]
        v1 = q_kern_f32[idx + half_dim]
        q_kern_f32[idx]            = np.float32(np.float32(v0 * cos_a) - np.float32(v1 * sin_a))
        q_kern_f32[idx + half_dim] = np.float32(np.float32(v0 * sin_a) + np.float32(v1 * cos_a))

    max_diff = np.max(np.abs(q_ref_f64 - q_kern_f32.astype(np.float64)))
    # f32 has ~7 decimal digits; for values of order 1, expect ~1e-6 error
    # For pos=500 with theta=1e6, high-freq dims have tiny angles (well-behaved)
    # Low-freq dims (j=0) have angle=500, which wraps — f32 cos/sin still OK
    report("f32 kernel vs f64 reference (pos=500)", max_diff < 5e-3,
           f"max_diff={max_diff:.2e} (f32 trig precision at large angles)")

    # Test 7: Large position (stress test frequency precision)
    print("\n── Test 7: Large position (pos=100000) ──")
    q = rng.standard_normal(N_HEADS_Q * HEAD_DIM)
    q_ref = reference_rope(q, N_HEADS_Q, HEAD_DIM, 100000, THETA)
    q_kern = kernel_rope(q, N_HEADS_Q, HEAD_DIM, 100000, THETA)
    max_diff = np.max(np.abs(q_ref - q_kern))
    report("Large position (ref vs kernel)", max_diff < 1e-10,
           f"max_diff={max_diff:.2e}")

    # Magnitude preservation at large pos
    for h in range(N_HEADS_Q):
        orig = np.linalg.norm(q[h * HEAD_DIM:(h + 1) * HEAD_DIM])
        rot = np.linalg.norm(q_kern[h * HEAD_DIM:(h + 1) * HEAD_DIM])
        if abs(orig - rot) / orig > 1e-8:
            report("Magnitude at pos=100000", False, f"head {h} drifted")
            break
    else:
        report("Magnitude preserved at pos=100000", True)


# ═══════════════════════════════════════════════════════════
# MATHEMATICAL EQUIVALENCE CHECK
# ═══════════════════════════════════════════════════════════

def test_math_equivalence():
    print("\n══════ MATHEMATICAL EQUIVALENCE ══════")

    # RMS Norm: verify kernel formula = reference formula algebraically
    # Reference: x / sqrt(mean(x^2) + eps) * w
    # Kernel:    x * rsqrt(sum(x^2)/n + eps) * w
    # These are identical since mean = sum/n and 1/sqrt = rsqrt
    print("\n── RMS Norm formula equivalence ──")
    rng = np.random.default_rng(999)
    x = rng.standard_normal(HIDDEN_DIM)
    w = rng.standard_normal(HIDDEN_DIM)

    # Explicit computation both ways
    n = len(x)
    sum_sq = np.sum(x ** 2)
    mean_sq = sum_sq / n

    # Reference path
    rms = np.sqrt(mean_sq + EPS_DEFAULT)
    ref_out = (x / rms) * w

    # Kernel path
    scale = 1.0 / np.sqrt(sum_sq / float(n) + EPS_DEFAULT)
    kern_out = x * scale * w

    diff = np.max(np.abs(ref_out - kern_out))
    report("x/sqrt(mean+eps)*w == x*rsqrt(sum/n+eps)*w", diff < 1e-14,
           f"max_diff={diff:.2e}")

    # RoPE: verify the rotation matrix is correct
    # Standard 2D rotation: [cos -sin; sin cos] applied to [v0, v1]
    print("\n── RoPE rotation matrix check ──")
    v0, v1 = 3.0, 4.0
    angle = 0.7
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Kernel formula
    new_v0 = v0 * cos_a - v1 * sin_a
    new_v1 = v0 * sin_a + v1 * cos_a

    # Matrix form
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    mat_result = R @ np.array([v0, v1])

    report("Kernel matches rotation matrix",
           abs(new_v0 - mat_result[0]) < 1e-15 and abs(new_v1 - mat_result[1]) < 1e-15)

    # Verify norm preservation of rotation
    orig_norm = np.sqrt(v0**2 + v1**2)
    new_norm = np.sqrt(new_v0**2 + new_v1**2)
    report("Rotation preserves 2D norm", abs(orig_norm - new_norm) < 1e-14,
           f"|{orig_norm} - {new_norm}| = {abs(orig_norm - new_norm):.2e}")


# ═══════════════════════════════════════════════════════════
# EDGE CASES & POTENTIAL BUGS
# ═══════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n══════ EDGE CASES & POTENTIAL ISSUES ══════")

    # RMS Norm: single block limitation
    # The kernel launches with <<<1, 256, ...>>> — only ONE block!
    # For dim > 256, threads stride: for(i=tid; i<n; i+=blockDim.x)
    # Tree reduction in shared memory with 256 entries — this is correct
    # as long as n is handled by the stride loop.
    print("\n── RMS Norm: Single block with dim=2048 ──")
    # 256 threads, each processes 2048/256 = 8 elements in the sum
    # This is correct but may lose precision for very large dims
    rng = np.random.default_rng(777)
    x = rng.standard_normal(8192).astype(np.float32)  # larger than typical
    w = np.ones(8192, dtype=np.float32)

    ref = reference_rms_norm(x.astype(np.float64), w.astype(np.float64), EPS_DEFAULT)
    kern = kernel_rms_norm(x, w, np.float32(EPS_DEFAULT))
    max_diff = np.max(np.abs(ref - kern.astype(np.float64)))
    report("dim=8192 with 256 threads (stride loop)", max_diff < 1e-3,
           f"max_diff={max_diff:.2e}")

    # RoPE: frequency precision at high j values
    print("\n── RoPE: Frequency precision at high j ──")
    # For j near head_dim/2 - 1, freq = 1/theta^(2*63/128) = 1/theta^0.984
    # With theta=1e6, this is ~1/(1e6^0.984) ≈ 1.1e-6 — very small
    # powf precision matters here
    j_max = HEAD_DIM // 2 - 1  # j=63
    freq_f64 = 1.0 / (THETA ** (2.0 * j_max / HEAD_DIM))
    freq_f32 = np.float32(1.0) / np.float32(np.float32(THETA) ** np.float32(2.0 * j_max / HEAD_DIM))
    rel_err = abs(float(freq_f64) - float(freq_f32)) / abs(float(freq_f64))
    report(f"Freq at j={j_max}: f32 vs f64 relative error", rel_err < 1e-5,
           f"f64={freq_f64:.6e} f32={float(freq_f32):.6e} rel_err={rel_err:.2e}")

    # RoPE: j=0 should have freq = 1/theta^0 = 1.0
    freq_j0 = 1.0 / (THETA ** (0.0 / HEAD_DIM))
    report("Freq at j=0 is exactly 1.0", abs(freq_j0 - 1.0) < 1e-15,
           f"freq={freq_j0}")

    # RoPE: position 0 angle = 0 for all freqs
    print("\n── RoPE: Position 0 produces zero angles ──")
    for j in range(HEAD_DIM // 2):
        freq = 1.0 / (THETA ** (2.0 * j / HEAD_DIM))
        angle = 0 * freq
        if angle != 0.0:
            report("Pos=0 all angles zero", False, f"j={j} angle={angle}")
            break
    else:
        report("Pos=0 all angles zero", True)

    # Check Qwen3 eps is effectively 1e-6
    print("\n── Qwen3 eps precision ──")
    # 9.999999974752427e-07 is the float32 representation of 1e-6
    f32_1e6 = np.float32(1e-6)
    report("Qwen3 eps == float32(1e-6)",
           abs(float(f32_1e6) - EPS_QWEN3) < 1e-18,
           f"float32(1e-6) = {float(f32_1e6):.16e}, config = {EPS_QWEN3:.16e}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MnemoCUDA Kernel Validation: RoPE + RMS Norm")
    print("=" * 60)
    print(f"Config: head_dim={HEAD_DIM}, n_heads_q={N_HEADS_Q}, "
          f"n_heads_k={N_HEADS_K}, theta={THETA:.0e}")
    print(f"        hidden_dim={HIDDEN_DIM}, eps={EPS_DEFAULT}, "
          f"qwen3_eps={EPS_QWEN3}")

    test_rms_norm()
    test_rope()
    test_math_equivalence()
    test_edge_cases()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
