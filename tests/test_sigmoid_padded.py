"""Correctness tests for padded sigmoid attention kernel.

This module validates the Triton-optimized padded sigmoid attention implementation
against a high-precision PyTorch reference. Tests cover:

- Multiple batch sizes (1, 4)
- Multiple head counts (2, 8)
- Variable sequence lengths (128-4096)
- Different head dimensions (64, 128)
- All supported dtypes (float16, bfloat16, float32)
- Both forward and backward passes
- Random padding patterns (80-100% sequence utilization)
- Self-attention and cross-attention configurations

Test Strategy:
    - Reference: Full FP32 computation (gold standard)
    - Baseline: Mixed precision PyTorch (numerical behavior match)
    - Target: Triton kernel (should match baseline within tolerance)

Tolerances:
    Accounts for accurate sigmoid by default:
    - FP32: 1e-5 (accurate sigmoid with FP32 precision)
    - FP16: 1e-3 (FP16 quantization error)
    - BF16: 7e-3 (BF16 lower mantissa precision)

Notes:
    - Uses PYTEST_VERSION env var to enable minimal autotuning during tests
    - Compares only valid (non-padded) positions
    - Tests gradient computation via backward pass
"""

import os

import pytest
import torch

# Enable accurate sigmoid for FP32 tests
os.environ['TRITON_SIGMOID_ACCURATE'] = '1'

from triton_sigmoid import sigmoid_attention_padded, sigmoid_attention_ref  # noqa: E402
from conftest import get_tolerance  # noqa: E402

# Get current CUDA device
device_id = torch.cuda.current_device()
DEVICE = torch.device(f'cuda:{device_id}')


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("H", [2, 8])
@pytest.mark.parametrize("N_CTX_Q,N_CTX_K", [(128, 128), (256, 512), (512, 256), (1024, 1024), (4096, 4096)])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_op(B, H, N_CTX_Q, N_CTX_K, HEAD_DIM, mode, dtype):
    """Test padded sigmoid attention correctness against reference implementation.

    Tests random padding patterns (80-100% utilization) for both self-attention
    and cross-attention. Validates forward/backward passes on valid positions only.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tol = get_tolerance(dtype)

    # Create Q, K, V tensors (BTHD format)
    q = torch.empty((B, N_CTX_Q, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, N_CTX_K, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, N_CTX_K, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()

    # Create copies for reference comparison
    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    ref_v = v.detach().clone().requires_grad_(True)

    pt_q = q.detach().clone().requires_grad_(True)
    pt_k = k.detach().clone().requires_grad_(True)
    pt_v = v.detach().clone().requires_grad_(True)

    # Generate random sequence lengths (80-100% of max length)
    seq_lens_q = torch.randint(int(0.8 * N_CTX_Q), N_CTX_Q + 1, (B,), device=DEVICE, dtype=torch.int32)
    seq_lens_k = torch.randint(int(0.8 * N_CTX_K), N_CTX_K + 1, (B,), device=DEVICE, dtype=torch.int32)

    # Validity masks for error computation
    query_positions = torch.arange(N_CTX_Q, device=DEVICE).unsqueeze(0)
    key_positions = torch.arange(N_CTX_K, device=DEVICE).unsqueeze(0)
    query_valid_mask = query_positions < seq_lens_q.unsqueeze(1)
    key_valid_mask = key_positions < seq_lens_k.unsqueeze(1)
    query_valid_mask_expanded = query_valid_mask.view(B, N_CTX_Q, 1, 1).to(dtype)
    key_valid_mask_expanded = key_valid_mask.view(B, N_CTX_K, 1, 1).to(dtype)

    # Forward pass
    ref_out = sigmoid_attention_ref(
        ref_q,
        ref_k,
        ref_v,
        seq_lens_k=seq_lens_k,
        seq_lens_q=seq_lens_q,
        upcast=True,
        reorder_ops=False,
    )

    # PyTorch baseline: standard precision
    pt_out = sigmoid_attention_ref(
        pt_q,
        pt_k,
        pt_v,
        seq_lens_k=seq_lens_k,
        seq_lens_q=seq_lens_q,
        upcast=False,
        reorder_ops=True,
    )

    # Triton implementation
    tri_out = sigmoid_attention_padded(
        q,
        k,
        v,
        seq_lens_q=seq_lens_q,
        seq_lens_k=seq_lens_k,
    )

    # Compute errors on valid positions only
    triton_error = ((tri_out - ref_out) * query_valid_mask_expanded).abs().max().item()
    pytorch_error = ((pt_out - ref_out) * query_valid_mask_expanded).abs().max().item()

    print(
        f"\nForward (fast sigmoid, {dtype}): Triton={triton_error:.6e}, PyTorch={pytorch_error:.6e}, "
        f"Ratio={triton_error/(pytorch_error+1e-10):.2f}x, Tol={tol:.6e}"
    )

    # Assertions
    assert (
        triton_error <= 2 * pytorch_error + tol
    ), f"Triton relative error {triton_error:.6e} > 2x PyTorch {pytorch_error:.6e} + {tol:.6e}"
    assert triton_error < tol, f"Triton absolute error {triton_error:.6e} exceeds threshold {tol:.6e}"

    if mode == "fwd":
        return

    # Backward pass
    dout = torch.randn_like(ref_out)

    ref_out.backward(dout)
    ref_dq = ref_q.grad.clone() * query_valid_mask_expanded
    ref_dk = ref_k.grad.clone() * key_valid_mask_expanded
    ref_dv = ref_v.grad.clone() * key_valid_mask_expanded
    ref_q.grad = ref_k.grad = ref_v.grad = None

    pt_out.backward(dout)
    pt_dq = pt_q.grad.clone() * query_valid_mask_expanded
    pt_dk = pt_k.grad.clone() * key_valid_mask_expanded
    pt_dv = pt_v.grad.clone() * key_valid_mask_expanded
    pt_q.grad = pt_k.grad = pt_v.grad = None

    tri_out.backward(dout)
    tri_dq = q.grad.clone() * query_valid_mask_expanded
    tri_dk = k.grad.clone() * key_valid_mask_expanded
    tri_dv = v.grad.clone() * key_valid_mask_expanded
    q.grad = k.grad = v.grad = None
    dq_triton_error = (tri_dq - ref_dq).abs().max().item()
    dq_pytorch_error = (pt_dq - ref_dq).abs().max().item()
    dk_triton_error = (tri_dk - ref_dk).abs().max().item()
    dk_pytorch_error = (pt_dk - ref_dk).abs().max().item()
    dv_triton_error = (tri_dv - ref_dv).abs().max().item()
    dv_pytorch_error = (pt_dv - ref_dv).abs().max().item()

    print(
        f"Backward (fast sigmoid, {dtype}): dQ={dq_triton_error:.6e}/{dq_pytorch_error:.6e}, "
        f"dK={dk_triton_error:.6e}/{dk_pytorch_error:.6e}, "
        f"dV={dv_triton_error:.6e}/{dv_pytorch_error:.6e}, Tol={tol:.6e}"
    )

    # Assertions
    assert (
        dq_triton_error <= 3 * dq_pytorch_error + tol
    ), f"dQ relative error {dq_triton_error:.6e} > 3x PyTorch {dq_pytorch_error:.6e} + {tol:.6e}"
    assert (
        dk_triton_error <= 3 * dk_pytorch_error + tol
    ), f"dK relative error {dk_triton_error:.6e} > 3x PyTorch {dk_pytorch_error:.6e} + {tol:.6e}"
    assert (
        dv_triton_error <= 3 * dv_pytorch_error + tol
    ), f"dV relative error {dv_triton_error:.6e} > 3x PyTorch {dv_pytorch_error:.6e} + {tol:.6e}"

    assert dq_triton_error < tol, f"dQ absolute error {dq_triton_error:.6e} exceeds threshold {tol:.6e}"
    assert dk_triton_error < tol, f"dK absolute error {dk_triton_error:.6e} exceeds threshold {tol:.6e}"
    assert dv_triton_error < tol, f"dV absolute error {dv_triton_error:.6e} exceeds threshold {tol:.6e}"


def test_zero_length_sequences():
    """Test handling of batches with zero valid tokens (all padding)."""
    torch.manual_seed(42)
    B, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.float16

    q = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)
    k = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)
    v = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)

    # Batch 0: all padding (seq_len=0), Batch 1: normal (seq_len=N_CTX)
    seq_lens_k = torch.tensor([0, N_CTX], dtype=torch.int32, device=DEVICE)

    # Should not crash
    out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)

    # Backward should not crash
    dout = torch.randn_like(out)
    out.backward(dout)

    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isnan(q.grad).any(), "dQ contains NaN"
    assert not torch.isnan(k.grad).any(), "dK contains NaN"
    assert not torch.isnan(v.grad).any(), "dV contains NaN"


def test_no_padding():
    """Test that kernel works correctly when no padding is present."""
    torch.manual_seed(42)
    B, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.float16

    q = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)
    k = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)
    v = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)

    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    ref_v = v.detach().clone().requires_grad_(True)

    # No padding (None or all sequences at max length)
    tri_out = sigmoid_attention_padded(q, k, v, seq_lens_k=None)
    ref_out = sigmoid_attention_ref(
        ref_q,
        ref_k,
        ref_v,
        seq_lens_k=None,
        upcast=True,
    )

    tol = get_tolerance(dtype)
    error = (tri_out - ref_out).abs().max().item()
    assert error < tol, f"Error {error:.6e} exceeds tolerance {tol:.6e}"

    # Test backward
    dout = torch.randn_like(tri_out)
    tri_out.backward(dout)
    ref_out.backward(dout)

    dq_error = (q.grad - ref_q.grad).abs().max().item()
    assert dq_error < tol, f"dQ error {dq_error:.6e} exceeds tolerance {tol:.6e}"


def test_gradient_numerical_stability():
    """Test that gradients don't explode or vanish with extreme values."""
    torch.manual_seed(42)
    B, H, N_CTX, HEAD_DIM = 1, 1, 64, 64
    dtype = torch.float16

    # Test with large values
    q = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE) * 10.0
    k = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE) * 10.0
    v = torch.randn(B, N_CTX, H, HEAD_DIM, dtype=dtype, device=DEVICE) * 10.0
    q.requires_grad = k.requires_grad = v.requires_grad = True

    # No padding - all sequences at max length
    out = sigmoid_attention_padded(q, k, v, seq_lens_k=None)
    loss = out.sum()
    loss.backward()

    # Check gradients are finite
    assert torch.isfinite(q.grad).all(), "dQ contains inf/nan"
    assert torch.isfinite(k.grad).all(), "dK contains inf/nan"
    assert torch.isfinite(v.grad).all(), "dV contains inf/nan"

    # Check gradients are not too large
    assert q.grad.abs().max() < 1e6, "dQ gradients exploded"
    assert k.grad.abs().max() < 1e6, "dK gradients exploded"
    assert v.grad.abs().max() < 1e6, "dV gradients exploded"


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seqlen", [256, 512])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("dtype_str", ["float16", "float32"])
def test_causal_masking(batch_size, n_heads, seqlen, head_dim, dtype_str):
    """Test that causal masking works correctly with padded attention."""

    dtype = getattr(torch, dtype_str)

    # Create inputs with some padding
    q = torch.randn(batch_size, seqlen, n_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
    k = torch.randn(batch_size, seqlen, n_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
    v = torch.randn(batch_size, seqlen, n_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)

    # Create sequence lengths (last 20% is padding)
    padding_len = seqlen // 5
    valid_len = seqlen - padding_len
    seq_lens_k = torch.full((batch_size,), valid_len, dtype=torch.int32, device=DEVICE)

    # Compute with causal masking
    out_causal = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, is_causal=True)

    # Verify causality by checking that changing future keys doesn't affect past outputs
    # For a specific query position, modify all future key positions
    test_batch = 0
    test_pos = seqlen // 2

    # Get the output at test position
    out_test_orig = out_causal[test_batch, test_pos, :].clone()

    # Modify future keys (but not padding positions)
    k_modified = k.clone()
    k_modified[test_batch, test_pos + 1 : valid_len, :] += 10.0  # Large perturbation

    # Recompute
    out_modified = sigmoid_attention_padded(q, k_modified, v, seq_lens_k=seq_lens_k, is_causal=True)

    # Output at test position should be unchanged
    out_test_new = out_modified[test_batch, test_pos, :]
    assert torch.allclose(
        out_test_orig, out_test_new, atol=1e-5, rtol=1e-5
    ), "Causal masking failed: future keys affected past outputs"


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("seqlen", [256])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("dtype_str", ["float16"])
def test_causal_with_padding_correctness(batch_size, n_heads, seqlen, head_dim, dtype_str):
    """Test that causal + padding gives correct results compared to reference."""

    dtype = getattr(torch, dtype_str)

    # Create inputs
    q = torch.randn(batch_size, seqlen, n_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
    k = torch.randn(batch_size, seqlen, n_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
    v = torch.randn(batch_size, seqlen, n_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)

    # Create sequence lengths (variable lengths)
    seq_lens_k = torch.tensor([200, 220], dtype=torch.int32, device=DEVICE)

    # Reference
    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    ref_v = v.detach().clone().requires_grad_(True)

    # Compute with triton kernel
    tri_out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, is_causal=True)

    # Compute reference with causal mask
    ref_out = sigmoid_attention_ref(
        ref_q,
        ref_k,
        ref_v,
        seq_lens_k=seq_lens_k,
        upcast=True,
        causal=True,
    )

    # Compare valid positions only
    tol = get_tolerance(dtype)
    for b in range(batch_size):
        # Determine valid length for this sequence
        valid_len = seq_lens_k[b].item()
        tri_slice = tri_out[b, :, :valid_len, :]
        ref_slice = ref_out[b, :, :valid_len, :]

        error = (tri_slice - ref_slice).abs().max().item()
        assert error < tol, f"Batch {b}: Error {error:.6e} exceeds tolerance {tol:.6e}"

    # Test backward pass
    dout = torch.randn_like(tri_out)
    tri_out.backward(dout)
    ref_out.backward(dout)

    # Compare gradients at valid positions
    for b in range(batch_size):
        valid_len = seq_lens_k[b].item()

        dq_tri = q.grad[b, :, :valid_len, :]
        dq_ref = ref_q.grad[b, :, :valid_len, :]
        dq_error = (dq_tri - dq_ref).abs().max().item()
        assert dq_error < tol, f"Batch {b} dQ: Error {dq_error:.6e} exceeds tolerance {tol:.6e}"

        dk_tri = k.grad[b, :, :valid_len, :]
        dk_ref = ref_k.grad[b, :, :valid_len, :]
        dk_error = (dk_tri - dk_ref).abs().max().item()
        assert dk_error < tol, f"Batch {b} dK: Error {dk_error:.6e} exceeds tolerance {tol:.6e}"

        dv_tri = v.grad[b, :, :valid_len, :]
        dv_ref = ref_v.grad[b, :, :valid_len, :]
        dv_error = (dv_tri - dv_ref).abs().max().item()
        assert dv_error < tol, f"Batch {b} dV: Error {dv_error:.6e} exceeds tolerance {tol:.6e}"


def test_score_mod_padded():
    """Test that score_mod parameter works correctly with padded sequences."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, L, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.float16
    tol = get_tolerance(dtype)

    # Create Q, K, V tensors (BTHD format)
    q = torch.empty((B, L, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, L, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, L, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    score_mod = torch.randn(B, L, dtype=dtype, device=DEVICE)

    # Create sequence lengths
    seq_lens_k = torch.tensor([100, 110], dtype=torch.int32, device=DEVICE)

    # Create copies for reference and PyTorch implementations
    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    ref_v = v.detach().clone().requires_grad_(True)

    pt_q = q.detach().clone().requires_grad_(True)
    pt_k = k.detach().clone().requires_grad_(True)
    pt_v = v.detach().clone().requires_grad_(True)

    # Reference: high-precision
    ref_out = sigmoid_attention_ref(
        ref_q, ref_k, ref_v, seq_lens_k=seq_lens_k, score_mod=score_mod, upcast=True, reorder_ops=False
    )

    # PyTorch baseline: standard precision
    pt_out = sigmoid_attention_ref(
        pt_q, pt_k, pt_v, seq_lens_k=seq_lens_k, score_mod=score_mod, upcast=False, reorder_ops=True
    )

    # Triton implementation
    tri_out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, score_mod=score_mod)

    # Forward pass errors (only at valid positions)
    for b in range(B):
        valid_len = seq_lens_k[b].item()
        tri_slice = tri_out[b, :valid_len, :, :]
        ref_slice = ref_out[b, :valid_len, :, :]
        pt_slice = pt_out[b, :valid_len, :, :]
        triton_error = (tri_slice - ref_slice).abs().max().item()
        pytorch_error = (pt_slice - ref_slice).abs().max().item()
        print(f"Batch {b} score_mod forward: Triton={triton_error:.6e}, PyTorch={pytorch_error:.6e}")
        assert triton_error <= 2 * pytorch_error + tol
        assert triton_error < tol

    # Backward pass
    dout = torch.randn_like(ref_out)
    tri_out.backward(dout)
    ref_out.backward(dout)
    pt_out.backward(dout)

    # Compute gradient errors (only at valid positions)
    for b in range(B):
        valid_len = seq_lens_k[b].item()
        tri_dq = (q.grad[b, :valid_len, :, :] - ref_q.grad[b, :valid_len, :, :]).abs().max().item()
        tri_dk = (k.grad[b, :valid_len, :, :] - ref_k.grad[b, :valid_len, :, :]).abs().max().item()
        tri_dv = (v.grad[b, :valid_len, :, :] - ref_v.grad[b, :valid_len, :, :]).abs().max().item()
        pt_dq = (pt_q.grad[b, :valid_len, :, :] - ref_q.grad[b, :valid_len, :, :]).abs().max().item()
        pt_dk = (pt_k.grad[b, :valid_len, :, :] - ref_k.grad[b, :valid_len, :, :]).abs().max().item()
        pt_dv = (pt_v.grad[b, :valid_len, :, :] - ref_v.grad[b, :valid_len, :, :]).abs().max().item()

        print(
            f"Batch {b} score_mod backward: dQ={tri_dq:.6e}/{pt_dq:.6e}, dK={tri_dk:.6e}/{pt_dk:.6e}, dV={tri_dv:.6e}/{pt_dv:.6e}"
        )

        # Assertions
        assert tri_dq <= 3 * pt_dq + tol
        assert tri_dk <= 3 * pt_dk + tol
        assert tri_dv <= 3 * pt_dv + tol
        assert tri_dq < tol
        assert tri_dk < tol
        assert tri_dv < tol * 2  # Padding edge effects

    # Test that score_mod=None still works
    q.grad = None
    k.grad = None
    v.grad = None
    out_none = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, score_mod=None)
    assert out_none.shape == (B, L, H, HEAD_DIM), "score_mod=None should work"

def test_padded_wrong_tensor_rank_3d():
    """Test that 3D tensors raise an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    q = torch.randn(2, 128, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(2, 128, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(2, 128, 64, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="Expected query, key, and value to be 4D tensors"):
        sigmoid_attention_padded(q, k, v)


def test_padded_wrong_tensor_rank_5d():
    """Test that 5D tensors raise an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    q = torch.randn(2, 128, 8, 64, 1, device=DEVICE, dtype=torch.float16)
    k = torch.randn(2, 128, 8, 64, 1, device=DEVICE, dtype=torch.float16)
    v = torch.randn(2, 128, 8, 64, 1, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="Expected query, key, and value to be 4D tensors"):
        sigmoid_attention_padded(q, k, v)


def test_seq_lens_k_wrong_shape():
    """Test that seq_lens_k with wrong shape raises an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Wrong shape: 2D instead of 1D
    seq_lens_k = torch.tensor([[100, 120]], dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError, match="seq_lens_k must have shape"):
        sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)


def test_seq_lens_k_wrong_batch():
    """Test that seq_lens_k with wrong batch size raises an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Wrong batch size
    seq_lens_k = torch.tensor([100, 120, 110], dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError, match="seq_lens_k must have shape"):
        sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)


def test_seq_lens_q_wrong_shape():
    """Test that seq_lens_q with wrong shape raises an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    seq_lens_k = torch.tensor([100, 120], dtype=torch.int32, device=DEVICE)
    seq_lens_q = torch.tensor([[100], [120]], dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError, match="seq_lens_q must have shape"):
        sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, seq_lens_q=seq_lens_q)


def test_negative_seq_lens():
    """Test behavior with negative sequence lengths."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Negative sequence length
    seq_lens_k = torch.tensor([-1, 120], dtype=torch.int32, device=DEVICE)

    # Should either handle gracefully or raise error
    try:
        out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)
        # If it doesn't error, check output is reasonable
        assert not torch.isnan(out).any(), "Output should not contain NaN"
    except (RuntimeError, ValueError):
        # Acceptable to reject negative lengths
        pass


def test_seq_lens_exceeds_max():
    """Test behavior when seq_lens exceeds actual sequence length."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # seq_lens exceeds T
    seq_lens_k = torch.tensor([100, 200], dtype=torch.int32, device=DEVICE)

    # Should clamp internally or raise error
    try:
        out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)
        assert not torch.isnan(out).any(), "Output should not contain NaN"
    except (RuntimeError, ValueError, IndexError):
        # Acceptable to reject invalid lengths
        pass


def test_padded_score_mod_wrong_shape():
    """Test that score_mod with wrong shape raises an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Wrong shape: should be (B, T_k)
    score_mod = torch.randn(B, T, H, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="score_mod must have shape"):
        sigmoid_attention_padded(q, k, v, score_mod=score_mod)


def test_padded_empty_batch():
    """Test behavior with empty batch (B=0)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    H, T, HEAD_DIM = 4, 128, 64

    q = torch.randn(0, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(0, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(0, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    try:
        out = sigmoid_attention_padded(q, k, v)
        assert out.shape == (0, T, H, HEAD_DIM), "Empty batch should return empty output"
    except (RuntimeError, ValueError):
        pass


def test_padded_zero_sequence_length():
    """Test behavior with zero sequence length (T=0)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, HEAD_DIM = 2, 4, 64

    q = torch.randn(B, 0, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, 0, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, 0, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    try:
        out = sigmoid_attention_padded(q, k, v)
        assert out.shape == (B, 0, H, HEAD_DIM), "Zero length should return empty output"
    except (RuntimeError, ValueError):
        pass


def test_padded_non_contiguous_tensors():
    """Test that non-contiguous tensors are handled correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 2, 4, 128, 64

    # Create non-contiguous tensors
    q = torch.randn(B, H, T, HEAD_DIM, device=DEVICE, dtype=torch.float16).transpose(1, 2)
    k = torch.randn(B, H, T, HEAD_DIM, device=DEVICE, dtype=torch.float16).transpose(1, 2)
    v = torch.randn(B, H, T, HEAD_DIM, device=DEVICE, dtype=torch.float16).transpose(1, 2)

    assert not q.is_contiguous(), "q should be non-contiguous"

    # Should handle by making contiguous internally
    out = sigmoid_attention_padded(q, k, v)
    assert out.shape == (B, T, H, HEAD_DIM), "Should handle non-contiguous tensors"
    assert not torch.isnan(out).any(), "Output should not contain NaN"


def test_seq_lens_auto_conversion_to_int32():
    """Test that seq_lens are automatically converted to int32."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Provide int64 instead of int32
    seq_lens_k = torch.tensor([100, 120], dtype=torch.int64, device=DEVICE)

    # Should auto-convert to int32
    out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)
    assert out.shape == (B, T, H, HEAD_DIM), "Should handle int64 seq_lens"
    assert not torch.isnan(out).any(), "Output should not contain NaN"


@pytest.mark.parametrize("B", [16, 32])
def test_padded_large_batch_sizes(B):
    """Test padded attention with large batch sizes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    H, T, HEAD_DIM = 4, 128, 64
    dtype = torch.float16

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

    # Random sequence lengths
    seq_lens_k = torch.randint(int(0.8 * T), T + 1, (B,), device=DEVICE, dtype=torch.int32)

    out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)
    assert out.shape == (B, T, H, HEAD_DIM)
    assert not torch.isnan(out).any()

    loss = out.sum()
    loss.backward()
    assert not torch.isnan(q.grad).any()


def test_padded_all_same_length():
    """Test padded attention when all sequences have the same length."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 4, 4, 128, 64
    dtype = torch.float16

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

    # All same length (no padding)
    seq_lens_k = torch.full((B,), T, dtype=torch.int32, device=DEVICE)

    # Compare padded vs dense
    out_padded = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)

    from triton_sigmoid import sigmoid_attention

    out_dense = sigmoid_attention(q, k, v)

    # Should produce similar results (within tolerance)
    tol = get_tolerance(dtype) * 2
    error = (out_padded - out_dense).abs().max().item()
    assert error < tol, f"Padded vs dense error {error:.6e} with all same lengths"


def test_padded_extreme_length_variation():
    """Test padded attention with extreme length variation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 4, 4, 256, 64
    dtype = torch.float16

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

    # Extreme variation: very short to full length
    seq_lens_k = torch.tensor([16, 64, 128, T], dtype=torch.int32, device=DEVICE)

    out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)
    assert out.shape == (B, T, H, HEAD_DIM)
    assert not torch.isnan(out).any()

    out.sum().backward()
    assert not torch.isnan(q.grad).any()


def test_cross_attention_extreme_length_ratio():
    """Test cross-attention with very different Q and K lengths."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, HEAD_DIM = 2, 4, 64
    T_q, T_k = 32, 512  # 16x difference
    dtype = torch.float16

    q = torch.randn(B, T_q, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T_k, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T_k, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

    seq_lens_k = torch.tensor([450, 500], dtype=torch.int32, device=DEVICE)
    seq_lens_q = torch.tensor([28, 30], dtype=torch.int32, device=DEVICE)

    out = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, seq_lens_q=seq_lens_q)
    assert out.shape == (B, T_q, H, HEAD_DIM)
    assert not torch.isnan(out).any()

    out.sum().backward()
    assert not torch.isnan(q.grad).any()
