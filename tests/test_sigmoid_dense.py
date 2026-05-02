"""Correctness tests for dense sigmoid attention kernel.

Validates Triton-optimized dense sigmoid attention against high-precision PyTorch reference.

Test Coverage:
- Batch sizes: 1, 4
- Heads: 2, 8
- Sequence lengths: 128-4096
- Head dimensions: 64, 128
- Dtypes: float16, bfloat16, float32
- Causal and non-causal modes
- Forward and backward passes

Tolerances (accounts for accurate sigmoid by default):
- FP32: 1e-5, FP16: 1e-3, BF16: 7e-3
"""

import os

import pytest
import torch

# Enable accurate sigmoid for FP32 tests
os.environ['TRITON_SIGMOID_ACCURATE'] = '1'

from triton_sigmoid import sigmoid_attention, sigmoid_attention_ref  # noqa: E402

from conftest import get_tolerance  # noqa: E402

# Get current CUDA device
device_id = torch.cuda.current_device()
DEVICE = torch.device(f'cuda:{device_id}')


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("H", [2, 8])
@pytest.mark.parametrize("T", [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dense_op(B, H, T, HEAD_DIM, is_causal, mode, dtype):
    """Test dense sigmoid attention correctness against reference implementation.

    Validates forward/backward passes for Triton kernel against FP32 reference.
    Tests both causal and non-causal modes to ensure correct masking.

    Args:
        B: Batch size
        H: Number of attention heads
        T: Sequence length (same for all sequences)
        HEAD_DIM: Dimension per attention head
        is_causal: Whether to apply causal masking
        mode: 'fwd' for forward only, 'bwd' for forward+backward
        dtype: torch.float16, torch.bfloat16, or torch.float32
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tol = get_tolerance(dtype)

    # Create Q, K, V tensors (BTHD format)
    q = torch.empty((B, T, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, T, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, T, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()

    # Create copies for reference comparison
    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    ref_v = v.detach().clone().requires_grad_(True)

    pt_q = q.detach().clone().requires_grad_(True)
    pt_k = k.detach().clone().requires_grad_(True)
    pt_v = v.detach().clone().requires_grad_(True)

    # Forward pass
    ref_out = sigmoid_attention_ref(
        ref_q,
        ref_k,
        ref_v,
        upcast=True,
        reorder_ops=False,
        causal=is_causal,
    )

    pt_out = sigmoid_attention_ref(
        pt_q,
        pt_k,
        pt_v,
        upcast=False,
        reorder_ops=True,
        causal=is_causal,
    )

    tri_out = sigmoid_attention(
        q,
        k,
        v,
        is_causal=is_causal,
    )

    triton_error = (tri_out - ref_out).abs().max().item()
    pytorch_error = (pt_out - ref_out).abs().max().item()

    print(
        f"\nForward (fast sigmoid, {dtype}, causal={is_causal}): Triton={triton_error:.6e}, "
        f"PyTorch={pytorch_error:.6e}, Ratio={triton_error/(pytorch_error+1e-10):.2f}x, Tol={tol:.6e}"
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
    ref_dq = ref_q.grad.clone()
    ref_dk = ref_k.grad.clone()
    ref_dv = ref_v.grad.clone()
    ref_q.grad = ref_k.grad = ref_v.grad = None

    pt_out.backward(dout)
    pt_dq = pt_q.grad.clone()
    pt_dk = pt_k.grad.clone()
    pt_dv = pt_v.grad.clone()
    pt_q.grad = pt_k.grad = pt_v.grad = None

    tri_out.backward(dout)
    tri_dq = q.grad.clone()
    tri_dk = k.grad.clone()
    tri_dv = v.grad.clone()
    q.grad = k.grad = v.grad = None
    dq_triton_error = (tri_dq - ref_dq).abs().max().item()
    dq_pytorch_error = (pt_dq - ref_dq).abs().max().item()
    dk_triton_error = (tri_dk - ref_dk).abs().max().item()
    dk_pytorch_error = (pt_dk - ref_dk).abs().max().item()
    dv_triton_error = (tri_dv - ref_dv).abs().max().item()
    dv_pytorch_error = (pt_dv - ref_dv).abs().max().item()

    print(
        f"Backward (fast sigmoid, {dtype}, causal={is_causal}): "
        f"dQ={dq_triton_error:.6e}/{dq_pytorch_error:.6e}, "
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


def test_dense_causal_correctness():
    """Verify causal mask prevents future from affecting past.

    Tests that modifying key/value at position j > i doesn't change output[i].
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.float16

    # Create inputs
    q = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)
    k = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)

    # Compute causal attention
    out_causal = sigmoid_attention(q, k, v, is_causal=True)

    # Modify future positions (second half of sequence)
    k_modified = k.clone()
    v_modified = v.clone()
    k_modified[:, T // 2 :, :, :] = torch.randn_like(k_modified[:, T // 2 :, :, :])
    v_modified[:, T // 2 :, :, :] = torch.randn_like(v_modified[:, T // 2 :, :, :])

    # Compute causal attention with modified future
    out_causal_modified = sigmoid_attention(q, k_modified, v_modified, is_causal=True)

    # Past outputs (first half) should be unchanged
    past_outputs = out_causal[:, : T // 2, :, :]
    past_outputs_modified = out_causal_modified[:, : T // 2, :, :]

    error = (past_outputs - past_outputs_modified).abs().max().item()
    print(f"Causal correctness: max error in past outputs = {error:.6e}")

    # Past should be identical (within numerical tolerance)
    tol = get_tolerance(dtype)
    assert error < tol, f"Past outputs changed by {error:.6e}, expected < {tol:.6e}"


def test_dense_vs_non_causal():
    """Test that non-causal attention is symmetric while causal is not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 1, 2, 64, 64
    dtype = torch.float16

    # Create inputs
    q = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)
    k = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)

    # Non-causal attention
    out_non_causal = sigmoid_attention(q, k, v, is_causal=False)

    # Causal attention
    out_causal = sigmoid_attention(q, k, v, is_causal=True)

    # They should be different (causal masks future)
    diff = (out_non_causal - out_causal).abs().max().item()
    print(f"Non-causal vs causal difference: {diff:.6e}")

    # Difference should be significant (future information matters)
    assert diff > 1e-3, f"Causal and non-causal outputs too similar: {diff:.6e}"


def test_gradient_numerical_stability():
    """Test that gradients don't explode or vanish with extreme values."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 1, 1, 64, 64
    dtype = torch.float16

    # Test with large values
    q = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE) * 10.0
    k = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE) * 10.0
    v = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE) * 10.0
    q.requires_grad = k.requires_grad = v.requires_grad = True

    # Test both causal and non-causal
    for is_causal in [False, True]:
        out = sigmoid_attention(q, k, v, is_causal=is_causal)
        loss = out.sum()
        loss.backward()

        # Check gradients are finite
        assert torch.isfinite(q.grad).all(), f"dQ contains inf/nan (causal={is_causal})"
        assert torch.isfinite(k.grad).all(), f"dK contains inf/nan (causal={is_causal})"
        assert torch.isfinite(v.grad).all(), f"dV contains inf/nan (causal={is_causal})"

        # Check gradients are not too large
        assert q.grad.abs().max() < 1e6, f"dQ gradients exploded (causal={is_causal})"
        assert k.grad.abs().max() < 1e6, f"dK gradients exploded (causal={is_causal})"
        assert v.grad.abs().max() < 1e6, f"dV gradients exploded (causal={is_causal})"

        # Zero gradients for next iteration
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()


def test_mismatched_sequence_lengths():
    """Test that mismatched sequence lengths raise an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, HEAD_DIM = 2, 4, 64
    dtype = torch.float16

    q = torch.randn(B, 128, H, HEAD_DIM, dtype=dtype, device=DEVICE)
    k = torch.randn(B, 256, H, HEAD_DIM, dtype=dtype, device=DEVICE)  # Different length!
    v = torch.randn(B, 256, H, HEAD_DIM, dtype=dtype, device=DEVICE)

    with pytest.raises(ValueError, match="Dense attention requires same sequence length"):
        sigmoid_attention(q, k, v)


def test_custom_scale_and_bias():
    """Test that custom scale and sigmoid_bias parameters work correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 1, 2, 64, 64
    dtype = torch.float16

    q = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)
    k = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, H, HEAD_DIM, dtype=dtype, device=DEVICE)

    # Default parameters
    out_default = sigmoid_attention(q, k, v)

    # Custom scale (larger scale = sharper attention)
    out_custom_scale = sigmoid_attention(q, k, v, scale=0.5)

    # Custom bias (affects attention distribution)
    out_custom_bias = sigmoid_attention(q, k, v, sigmoid_bias=-1.0)

    # They should all be different
    diff_scale = (out_default - out_custom_scale).abs().max().item()
    diff_bias = (out_default - out_custom_bias).abs().max().item()

    print(f"Scale difference: {diff_scale:.6e}, Bias difference: {diff_bias:.6e}")

    assert diff_scale > 1e-3, f"Custom scale had no effect: {diff_scale:.6e}"
    assert diff_bias > 1e-3, f"Custom bias had no effect: {diff_bias:.6e}"


def test_score_mod():
    """Test that score_mod parameter works correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.float16
    tol = get_tolerance(dtype)

    # Create Q, K, V tensors (BTHD format)
    q = torch.empty((B, T, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, T, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, T, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    score_mod = torch.randn(B, T, dtype=dtype, device=DEVICE)

    # Create copies for reference and PyTorch implementations
    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    ref_v = v.detach().clone().requires_grad_(True)

    pt_q = q.detach().clone().requires_grad_(True)
    pt_k = k.detach().clone().requires_grad_(True)
    pt_v = v.detach().clone().requires_grad_(True)

    # Reference: high-precision
    ref_out = sigmoid_attention_ref(ref_q, ref_k, ref_v, score_mod=score_mod, upcast=True, reorder_ops=False)

    # PyTorch baseline: standard precision
    pt_out = sigmoid_attention_ref(pt_q, pt_k, pt_v, score_mod=score_mod, upcast=False, reorder_ops=True)

    # Triton implementation
    tri_out = sigmoid_attention(q, k, v, score_mod=score_mod)

    # Forward pass errors
    triton_error = (tri_out - ref_out).abs().max().item()
    pytorch_error = (pt_out - ref_out).abs().max().item()
    print(f"score_mod forward: Triton={triton_error:.6e}, PyTorch={pytorch_error:.6e}")

    assert triton_error <= 2 * pytorch_error + tol
    assert triton_error < tol

    # Backward pass
    dout = torch.randn_like(ref_out)
    tri_out.backward(dout)
    ref_out.backward(dout)
    pt_out.backward(dout)

    # Compute gradient errors
    tri_dq = (q.grad - ref_q.grad).abs().max().item()
    tri_dk = (k.grad - ref_k.grad).abs().max().item()
    tri_dv = (v.grad - ref_v.grad).abs().max().item()
    pt_dq = (pt_q.grad - ref_q.grad).abs().max().item()
    pt_dk = (pt_k.grad - ref_k.grad).abs().max().item()
    pt_dv = (pt_v.grad - ref_v.grad).abs().max().item()

    print(f"score_mod backward: dQ={tri_dq:.6e}/{pt_dq:.6e}, dK={tri_dk:.6e}/{pt_dk:.6e}, dV={tri_dv:.6e}/{pt_dv:.6e}")

    # Assertions
    assert tri_dq <= 3 * pt_dq + tol
    assert tri_dk <= 3 * pt_dk + tol
    assert tri_dv <= 3 * pt_dv + tol
    assert tri_dq < tol
    assert tri_dk < tol
    assert tri_dv < tol * 2

    # Test that score_mod=None still works
    q.grad = None
    k.grad = None
    v.grad = None
    out_none = sigmoid_attention(q, k, v, score_mod=None)
    assert out_none.shape == (B, T, H, HEAD_DIM), "score_mod=None should work"

    # Test that score_mod=zeros matches score_mod=None closely
    score_mod_zeros = torch.zeros(B, T, dtype=dtype, device=DEVICE)
    out_zeros = sigmoid_attention(q, k, v, score_mod=score_mod_zeros)
    out_none = sigmoid_attention(q, k, v, score_mod=None)
    diff = (out_zeros - out_none).abs().max().item()
    print(f"score_mod=zeros vs score_mod=None diff: {diff:.6e}")
    assert diff < 1e-5, f"score_mod=zeros should match score_mod=None, got diff {diff:.6e}"


def test_wrong_tensor_rank_3d():
    """Test that 3D tensors raise an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 3D tensor (missing head dimension)
    q = torch.randn(2, 128, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(2, 128, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(2, 128, 64, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="Expected query, key, and value to be 4D tensors"):
        sigmoid_attention(q, k, v)


def test_wrong_tensor_rank_5d():
    """Test that 5D tensors raise an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 5D tensor (extra dimension)
    q = torch.randn(2, 128, 8, 64, 1, device=DEVICE, dtype=torch.float16)
    k = torch.randn(2, 128, 8, 64, 1, device=DEVICE, dtype=torch.float16)
    v = torch.randn(2, 128, 8, 64, 1, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="Expected query, key, and value to be 4D tensors"):
        sigmoid_attention(q, k, v)


def test_mismatched_batch_sizes():
    """Test that mismatched batch sizes raise an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    H, T, HEAD_DIM = 4, 128, 64

    q = torch.randn(2, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(4, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(4, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="Batch size mismatch"):
        sigmoid_attention(q, k, v)


def test_mismatched_num_heads():
    """Test that mismatched number of heads raises an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, T, HEAD_DIM = 2, 128, 64

    q = torch.randn(B, T, 4, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, 8, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, 8, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="Number of heads mismatch"):
        sigmoid_attention(q, k, v)


def test_mismatched_head_dims():
    """Test that mismatched head dimensions raise an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T = 2, 4, 128

    q = torch.randn(B, T, H, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, 128, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, 128, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="Head dimension mismatch"):
        sigmoid_attention(q, k, v)


def test_score_mod_wrong_shape():
    """Test that score_mod with wrong shape raises an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Wrong shape: should be (B, T), not (B, T, H)
    score_mod = torch.randn(B, T, H, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="score_mod must have shape"):
        sigmoid_attention(q, k, v, score_mod=score_mod)


def test_score_mod_wrong_batch():
    """Test that score_mod with wrong batch size raises an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 128, 64

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Wrong batch size
    score_mod = torch.randn(B + 1, T, device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="score_mod must have shape"):
        sigmoid_attention(q, k, v, score_mod=score_mod)


def test_empty_batch():
    """Test behavior with empty batch (B=0)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    H, T, HEAD_DIM = 4, 128, 64

    q = torch.randn(0, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(0, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(0, T, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Should handle gracefully or raise clear error
    try:
        out = sigmoid_attention(q, k, v)
        assert out.shape == (0, T, H, HEAD_DIM), "Empty batch should return empty output"
    except (RuntimeError, ValueError):
        # Acceptable to reject empty batches
        pass


def test_zero_sequence_length():
    """Test behavior with zero sequence length (T=0)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, HEAD_DIM = 2, 4, 64

    q = torch.randn(B, 0, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, 0, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, 0, H, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Should handle gracefully or raise clear error
    try:
        out = sigmoid_attention(q, k, v)
        assert out.shape == (B, 0, H, HEAD_DIM), "Zero length should return empty output"
    except (RuntimeError, ValueError):
        # Acceptable to reject zero length
        pass


def test_non_contiguous_tensors():
    """Test that non-contiguous tensors are handled correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 2, 4, 128, 64

    # Create non-contiguous tensors via transpose
    q = torch.randn(B, H, T, HEAD_DIM, device=DEVICE, dtype=torch.float16).transpose(1, 2)
    k = torch.randn(B, H, T, HEAD_DIM, device=DEVICE, dtype=torch.float16).transpose(1, 2)
    v = torch.randn(B, H, T, HEAD_DIM, device=DEVICE, dtype=torch.float16).transpose(1, 2)

    assert not q.is_contiguous(), "q should be non-contiguous"
    assert not k.is_contiguous(), "k should be non-contiguous"
    assert not v.is_contiguous(), "v should be non-contiguous"

    # Should handle by making contiguous internally (no error)
    out = sigmoid_attention(q, k, v)
    assert out.shape == (B, T, H, HEAD_DIM), "Should handle non-contiguous tensors"
    assert not torch.isnan(out).any(), "Output should not contain NaN"


@pytest.mark.parametrize("B", [16, 32])
def test_large_batch_sizes(B):
    """Test dense attention with large batch sizes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    H, T, HEAD_DIM = 4, 128, 64
    dtype = torch.float16

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

    out = sigmoid_attention(q, k, v)
    assert out.shape == (B, T, H, HEAD_DIM)
    assert not torch.isnan(out).any()

    loss = out.sum()
    loss.backward()
    assert not torch.isnan(q.grad).any()


@pytest.mark.parametrize("HEAD_DIM", [16, 32])
def test_small_head_dimensions(HEAD_DIM):
    """Test dense attention with small head dimensions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T = 2, 4, 128
    dtype = torch.float16

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

    out = sigmoid_attention(q, k, v)
    assert out.shape == (B, T, H, HEAD_DIM)
    assert not torch.isnan(out).any()

    out.sum().backward()
    assert not torch.isnan(q.grad).any()


@pytest.mark.parametrize("H", [1, 16, 32])
def test_various_num_heads(H):
    """Test dense attention with various numbers of heads."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, T, HEAD_DIM = 2, 128, 64
    dtype = torch.float16

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)

    out = sigmoid_attention(q, k, v)
    assert out.shape == (B, T, H, HEAD_DIM)
    assert not torch.isnan(out).any()


def test_all_zeros_input():
    """Test behavior with all-zero inputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 64, 64
    dtype = torch.float16

    q = torch.zeros(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.zeros(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.zeros(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

    out = sigmoid_attention(q, k, v)
    assert not torch.isnan(out).any(), "All-zero inputs should not produce NaN"
    assert torch.isfinite(out).all(), "All-zero inputs should produce finite values"

    out.sum().backward()
    assert not torch.isnan(q.grad).any(), "Gradients should not be NaN for zero inputs"


def test_constant_input():
    """Test behavior with constant (non-zero) inputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, T, HEAD_DIM = 2, 4, 64, 64
    dtype = torch.float16

    q = torch.ones(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.ones(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.ones(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

    out = sigmoid_attention(q, k, v)
    assert not torch.isnan(out).any(), "Constant inputs should not produce NaN"
    assert torch.isfinite(out).all(), "Constant inputs should produce finite values"

    out.sum().backward()
    assert not torch.isnan(q.grad).any(), "Gradients should not be NaN for constant inputs"


def test_requires_grad_false():
    """Test that forward pass works when requires_grad=False."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.float16

    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=False)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=False)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=False)

    out = sigmoid_attention(q, k, v)
    assert out.shape == (B, T, H, HEAD_DIM)
    assert not torch.isnan(out).any()
    assert not out.requires_grad, "Output should not require grad when inputs don't"


def test_mixed_requires_grad():
    """Test with mixed requires_grad settings."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 2, 4, 128, 64
    dtype = torch.float16

    # Only Q requires gradient
    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=False)
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=False)

    out = sigmoid_attention(q, k, v)
    assert out.requires_grad, "Output should require grad if any input does"

    loss = out.sum()
    loss.backward()

    assert q.grad is not None, "Q should have gradient"
    assert k.grad is None, "K should not have gradient"
    assert v.grad is None, "V should not have gradient"


def test_sigmoid_large_positive_values():
    """Test sigmoid with very large positive attention scores."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 1, 1, 32, 64
    dtype = torch.float32

    # Create inputs that will produce large QK^T values
    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype) * 5.0
    k = q.clone()  # High similarity = large dot products
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)

    out = sigmoid_attention(q, k, v)
    assert torch.isfinite(out).all(), "Should handle large positive scores"
    assert not torch.isnan(out).any(), "Should not produce NaN with large scores"


def test_sigmoid_large_negative_values():
    """Test sigmoid with very large negative attention scores."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B, H, T, HEAD_DIM = 1, 1, 32, 64
    dtype = torch.float32

    # Create inputs with high negative similarity
    q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype) * 5.0
    k = -q.clone()  # Opposite vectors = large negative dot products
    v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)

    out = sigmoid_attention(q, k, v)
    assert torch.isfinite(out).all(), "Should handle large negative scores"
    assert not torch.isnan(out).any(), "Should not produce NaN with large negative scores"
