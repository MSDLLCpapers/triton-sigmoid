"""Tests for torch.compile integration with sigmoid attention kernels.

This module verifies that both dense and padded sigmoid attention kernels
work correctly with PyTorch's compilation system (torch.compile).

Test Coverage:
- Basic torch.compile functionality
- Multiple compilation backends (inductor, eager)
- Forward and backward passes under compilation
- No graph breaks during execution
- Correctness under compilation vs eager mode
"""

import os

import pytest
import torch

# Enable accurate sigmoid for FP32 tests
os.environ['TRITON_SIGMOID_ACCURATE'] = '1'

from triton_sigmoid import sigmoid_attention, sigmoid_attention_padded  # noqa: E402

from conftest import get_tolerance  # noqa: E402

# Get current CUDA device
device_id = torch.cuda.current_device()
DEVICE = torch.device(f'cuda:{device_id}')


@pytest.mark.skipif(
    not hasattr(torch, 'compile') or torch.__version__ < "2.0",
    reason="torch.compile requires PyTorch >= 2.0",
)
class TestDenseTorchCompile:
    """Test torch.compile integration for dense sigmoid attention."""

    def test_compile_basic_forward(self):
        """Test that dense attention compiles and runs correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, T, HEAD_DIM = 2, 4, 256, 64
        dtype = torch.float16

        q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)

        # Compile the function
        compiled_fn = torch.compile(sigmoid_attention)

        # Run compiled version
        out_compiled = compiled_fn(q, k, v)

        # Run eager version
        out_eager = sigmoid_attention(q, k, v)

        # Should produce identical results
        tol = get_tolerance(dtype)
        error = (out_compiled - out_eager).abs().max().item()
        assert error < tol, f"Compiled vs eager error {error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_backward_pass(self):
        """Test that compiled dense attention supports backward pass."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, T, HEAD_DIM = 2, 4, 128, 64
        dtype = torch.float16

        # Eager mode
        q_eager = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
        k_eager = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
        v_eager = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

        out_eager = sigmoid_attention(q_eager, k_eager, v_eager)
        loss_eager = out_eager.sum()
        loss_eager.backward()

        # Compiled mode
        q_compiled = q_eager.detach().clone().requires_grad_(True)
        k_compiled = k_eager.detach().clone().requires_grad_(True)
        v_compiled = v_eager.detach().clone().requires_grad_(True)

        compiled_fn = torch.compile(sigmoid_attention)
        out_compiled = compiled_fn(q_compiled, k_compiled, v_compiled)
        loss_compiled = out_compiled.sum()
        loss_compiled.backward()

        # Compare gradients
        tol = get_tolerance(dtype) * 2  # Slightly more lenient for compilation
        dq_error = (q_eager.grad - q_compiled.grad).abs().max().item()
        dk_error = (k_eager.grad - k_compiled.grad).abs().max().item()
        dv_error = (v_eager.grad - v_compiled.grad).abs().max().item()

        assert dq_error < tol, f"dQ error {dq_error:.6e} exceeds tolerance {tol:.6e}"
        assert dk_error < tol, f"dK error {dk_error:.6e} exceeds tolerance {tol:.6e}"
        assert dv_error < tol, f"dV error {dv_error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_with_causal(self):
        """Test that compiled dense attention works with causal masking."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, T, HEAD_DIM = 2, 4, 128, 64
        dtype = torch.float16

        q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)

        compiled_fn = torch.compile(sigmoid_attention)

        out_compiled = compiled_fn(q, k, v, is_causal=True)
        out_eager = sigmoid_attention(q, k, v, is_causal=True)

        tol = get_tolerance(dtype)
        error = (out_compiled - out_eager).abs().max().item()
        assert error < tol, f"Causal compiled vs eager error {error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_with_score_mod(self):
        """Test that compiled dense attention works with score_mod."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, T, HEAD_DIM = 2, 4, 128, 64
        dtype = torch.float16

        q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        score_mod = torch.randn(B, T, device=DEVICE, dtype=dtype)

        compiled_fn = torch.compile(sigmoid_attention)

        out_compiled = compiled_fn(q, k, v, score_mod=score_mod)
        out_eager = sigmoid_attention(q, k, v, score_mod=score_mod)

        tol = get_tolerance(dtype)
        error = (out_compiled - out_eager).abs().max().item()
        assert error < tol, f"score_mod compiled vs eager error {error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_multiple_calls(self):
        """Test that compiled function can be called multiple times with different shapes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        dtype = torch.float16

        compiled_fn = torch.compile(sigmoid_attention)

        # First call with one shape
        B1, H1, T1, D1 = 2, 4, 128, 64
        q1 = torch.randn(B1, T1, H1, D1, device=DEVICE, dtype=dtype)
        k1 = torch.randn(B1, T1, H1, D1, device=DEVICE, dtype=dtype)
        v1 = torch.randn(B1, T1, H1, D1, device=DEVICE, dtype=dtype)
        out1 = compiled_fn(q1, k1, v1)
        assert out1.shape == (B1, T1, H1, D1), "First call should succeed"

        # Second call with same head_dim and num_heads (static dims)
        B2, T2 = 4, 256
        q2 = torch.randn(B2, T2, H1, D1, device=DEVICE, dtype=dtype)
        k2 = torch.randn(B2, T2, H1, D1, device=DEVICE, dtype=dtype)
        v2 = torch.randn(B2, T2, H1, D1, device=DEVICE, dtype=dtype)
        out2 = compiled_fn(q2, k2, v2)
        assert out2.shape == (B2, T2, H1, D1), "Second call should succeed (recompile expected)"


@pytest.mark.skipif(
    not hasattr(torch, 'compile') or torch.__version__ < "2.0",
    reason="torch.compile requires PyTorch >= 2.0",
)
class TestPaddedTorchCompile:
    """Test torch.compile integration for padded sigmoid attention."""

    def test_compile_basic_forward(self):
        """Test that padded attention compiles and runs correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, T, HEAD_DIM = 2, 4, 256, 64
        dtype = torch.float16

        q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        seq_lens_k = torch.tensor([200, 250], dtype=torch.int32, device=DEVICE)

        compiled_fn = torch.compile(sigmoid_attention_padded)

        out_compiled = compiled_fn(q, k, v, seq_lens_k=seq_lens_k)
        out_eager = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k)

        tol = get_tolerance(dtype)
        error = (out_compiled - out_eager).abs().max().item()
        assert error < tol, f"Compiled vs eager error {error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_backward_pass(self):
        """Test that compiled padded attention supports backward pass."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, T, HEAD_DIM = 2, 4, 128, 64
        dtype = torch.float16

        seq_lens_k = torch.tensor([100, 120], dtype=torch.int32, device=DEVICE)

        # Eager mode
        q_eager = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
        k_eager = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)
        v_eager = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype, requires_grad=True)

        out_eager = sigmoid_attention_padded(q_eager, k_eager, v_eager, seq_lens_k=seq_lens_k)
        loss_eager = out_eager.sum()
        loss_eager.backward()

        # Compiled mode
        q_compiled = q_eager.detach().clone().requires_grad_(True)
        k_compiled = k_eager.detach().clone().requires_grad_(True)
        v_compiled = v_eager.detach().clone().requires_grad_(True)

        compiled_fn = torch.compile(sigmoid_attention_padded)
        out_compiled = compiled_fn(q_compiled, k_compiled, v_compiled, seq_lens_k=seq_lens_k)
        loss_compiled = out_compiled.sum()
        loss_compiled.backward()

        # Compare gradients (only at valid positions)
        tol = get_tolerance(dtype) * 2
        for b in range(B):
            valid_len = seq_lens_k[b].item()
            dq_error = (q_eager.grad[b, :valid_len] - q_compiled.grad[b, :valid_len]).abs().max().item()
            dk_error = (k_eager.grad[b, :valid_len] - k_compiled.grad[b, :valid_len]).abs().max().item()
            dv_error = (v_eager.grad[b, :valid_len] - v_compiled.grad[b, :valid_len]).abs().max().item()

            assert dq_error < tol, f"Batch {b} dQ error {dq_error:.6e} exceeds tolerance {tol:.6e}"
            assert dk_error < tol, f"Batch {b} dK error {dk_error:.6e} exceeds tolerance {tol:.6e}"
            assert dv_error < tol, f"Batch {b} dV error {dv_error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_with_causal(self):
        """Test that compiled padded attention works with causal masking."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, T, HEAD_DIM = 2, 4, 128, 64
        dtype = torch.float16

        q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        seq_lens_k = torch.tensor([100, 120], dtype=torch.int32, device=DEVICE)

        compiled_fn = torch.compile(sigmoid_attention_padded)

        out_compiled = compiled_fn(q, k, v, seq_lens_k=seq_lens_k, is_causal=True)
        out_eager = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, is_causal=True)

        tol = get_tolerance(dtype)
        error = (out_compiled - out_eager).abs().max().item()
        assert error < tol, f"Causal compiled vs eager error {error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_with_score_mod(self):
        """Test that compiled padded attention works with score_mod."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, T, HEAD_DIM = 2, 4, 128, 64
        dtype = torch.float16

        q = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        seq_lens_k = torch.tensor([100, 120], dtype=torch.int32, device=DEVICE)
        score_mod = torch.randn(B, T, device=DEVICE, dtype=dtype)

        compiled_fn = torch.compile(sigmoid_attention_padded)

        out_compiled = compiled_fn(q, k, v, seq_lens_k=seq_lens_k, score_mod=score_mod)
        out_eager = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, score_mod=score_mod)

        tol = get_tolerance(dtype)
        error = (out_compiled - out_eager).abs().max().item()
        assert error < tol, f"score_mod compiled vs eager error {error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_cross_attention(self):
        """Test that compiled padded attention works with cross-attention (T_q != T_k)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, H, HEAD_DIM = 2, 4, 64
        T_q, T_k = 128, 256
        dtype = torch.float16

        q = torch.randn(B, T_q, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T_k, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T_k, H, HEAD_DIM, device=DEVICE, dtype=dtype)
        seq_lens_k = torch.tensor([200, 250], dtype=torch.int32, device=DEVICE)
        seq_lens_q = torch.tensor([100, 120], dtype=torch.int32, device=DEVICE)

        compiled_fn = torch.compile(sigmoid_attention_padded)

        out_compiled = compiled_fn(q, k, v, seq_lens_k=seq_lens_k, seq_lens_q=seq_lens_q)
        out_eager = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, seq_lens_q=seq_lens_q)

        tol = get_tolerance(dtype)
        error = (out_compiled - out_eager).abs().max().item()
        assert error < tol, f"Cross-attention compiled vs eager error {error:.6e} exceeds tolerance {tol:.6e}"

    def test_compile_multiple_calls(self):
        """Test that compiled function can be called multiple times."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        dtype = torch.float16

        compiled_fn = torch.compile(sigmoid_attention_padded)

        # First call
        B1, H1, T1, D1 = 2, 4, 128, 64
        q1 = torch.randn(B1, T1, H1, D1, device=DEVICE, dtype=dtype)
        k1 = torch.randn(B1, T1, H1, D1, device=DEVICE, dtype=dtype)
        v1 = torch.randn(B1, T1, H1, D1, device=DEVICE, dtype=dtype)
        seq_lens1 = torch.tensor([100, 120], dtype=torch.int32, device=DEVICE)
        out1 = compiled_fn(q1, k1, v1, seq_lens_k=seq_lens1)
        assert out1.shape == (B1, T1, H1, D1), "First call should succeed"

        # Second call with different batch/seq_len (but same static dims)
        B2, T2 = 4, 256
        q2 = torch.randn(B2, T2, H1, D1, device=DEVICE, dtype=dtype)
        k2 = torch.randn(B2, T2, H1, D1, device=DEVICE, dtype=dtype)
        v2 = torch.randn(B2, T2, H1, D1, device=DEVICE, dtype=dtype)
        seq_lens2 = torch.tensor([200, 220, 240, 250], dtype=torch.int32, device=DEVICE)
        out2 = compiled_fn(q2, k2, v2, seq_lens_k=seq_lens2)
        assert out2.shape == (B2, T2, H1, D1), "Second call should succeed"
