"""Triton Sigmoid Attention: High-Performance GPU Kernels for Sigmoid-Based Attention.

This package provides memory-efficient, fused Triton kernels for computing sigmoid attention,
an alternative attention mechanism to standard softmax attention. The implementation includes
optimized forward and backward passes with automatic differentiation support via PyTorch.

Algorithm
---------
Computes: output = sigmoid(Q @ K^T * scale + bias + score_mod) @ V

Where:
    - Q, K, V: Query, key, value tensors
    - scale: Attention scaling factor (default: 1/sqrt(head_dim))
    - bias: Per-sequence bias term (-log(num_valid_tokens + eps))
    - score_mod: Optional per-token bias [batch, seq_len] (default: None)
    - sigmoid: Element-wise sigmoid activation using fast tanh approximation (default)
              Set TRITON_SIGMOID_ACCURATE=1 for accurate libdevice implementation

Implementations
---------------
sigmoid_attention:
    Dense attention for same-length sequences (the default choice).
    Input shape: [batch, seq_len, n_heads, head_dim]
    Use when: All sequences have the same length.
    Fastest option: no padding masks, no offset arithmetic.
    Supports causal masking for autoregressive generation.

sigmoid_attention_padded:
    Padded attention for variable-length sequences with trailing padding.
    Input shape: [batch, seq_len, n_heads, head_dim] (padded to uniform length)
    Use when: Sequences have variable lengths AND you need torch.compile compatibility.
    Uses sequence length specifications to skip padded tokens (torch.compile friendly, no graph breaks).
    Supports causal masking for autoregressive generation.

Reference Implementation
------------------------
sigmoid_attention_ref:
    PyTorch reference implementation for correctness testing.
    Supports dense format with optional causal masking.

Requirements
------------
- CUDA GPU with compute capability >= 8.0 (Ampere or newer)
- PyTorch >= 2.6.0
- Triton >= 3.2.0
- Python 3.11 or 3.12

Supported dtypes: torch.float16, torch.bfloat16, torch.float32
"""

__version__ = "0.1.0"
__author__ = "Vijay Sadashivaiah"

# Import reference implementation
from .sigmoid_reference import sigmoid_attention_ref

# Import main implementations
from .sigmoid_triton_dense import sigmoid_attention
from .sigmoid_triton_padded import sigmoid_attention_padded

__all__ = [
    # Triton kernels (optimized)
    "sigmoid_attention",  # Same-length sequences (default)
    "sigmoid_attention_padded",  # Variable-length with trailing padding (torch.compile friendly)
    # Reference implementation (for testing)
    "sigmoid_attention_ref",
]
