"""Reference implementations of sigmoid attention for correctness validation.

PyTorch-based reference implementation of sigmoid attention

Precision Modes:
- upcast=True: Full FP32 computation
- upcast=False: Mixed precision matching Triton kernel behavior
"""

import math
from typing import Optional

import torch


def sigmoid_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens_k: Optional[torch.Tensor] = None,
    seq_lens_q: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    sigmoid_bias: Optional[float] = None,
    upcast: bool = True,
    reorder_ops: bool = False,
    causal: bool = False,
    score_mod: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation of sigmoid attention for padded sequences.

    Computes: output = sigmoid(Q @ K^T * scale + bias + score_mod) @ V
    where bias can be explicitly set or auto-computed as -log(num_valid_keys).

    Args:
        q: Query tensor [batch, seq_len_q, n_heads, head_dim]
        k: Key tensor [batch, seq_len_k, n_heads, head_dim]
        v: Value tensor [batch, seq_len_k, n_heads, head_dim]
        seq_lens_k: Sequence lengths for keys [batch], dtype=int32. None means no padding (all valid).
        seq_lens_q: Sequence lengths for queries [batch], dtype=int32. None means no padding (all valid).
        scale: Attention scaling factor (default: 1/sqrt(head_dim))
        sigmoid_bias: Explicit sigmoid bias. If None, auto-computes as -log(num_valid_keys + eps) for normalization.
        upcast: If True, compute in FP32 for max accuracy. If False, match kernel's mixed precision.
        reorder_ops: If True, scale keys instead of queries (affects numerical precision)
        causal: If True, position i attends only to j <= i (autoregressive masking)
        score_mod: Optional per-token bias [batch, seq_len_k]. Applied after bias, broadcasts across heads.

    Returns:
        output: Attention output [batch, seq_len_q, n_heads, head_dim]
    """
    B, Lq, H, D = q.shape
    Lk = k.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    dtype_og = q.dtype

    # Construct padding masks from sequence lengths
    if seq_lens_k is not None:
        seq_lens_k = seq_lens_k.to(q.device)
        key_padding_mask = torch.arange(Lk, device=q.device).unsqueeze(0) >= seq_lens_k.unsqueeze(1)
    else:
        key_padding_mask = None

    if seq_lens_q is not None:
        seq_lens_q = seq_lens_q.to(q.device)
        query_padding_mask = torch.arange(Lq, device=q.device).unsqueeze(0) >= seq_lens_q.unsqueeze(1)
    else:
        query_padding_mask = None

    # Compute or use provided sigmoid bias
    if sigmoid_bias is not None:
        bias = sigmoid_bias
    else:
        # Auto-compute normalization: -log(num_valid_keys + eps)
        eps = 1e-6
        if seq_lens_k is not None:
            bias = -torch.log(seq_lens_k.float() + eps).view(B, 1, 1, 1)
        else:
            bias = -math.log(float(Lk) + eps)

    if upcast:
        # Full FP32 computation for maximum accuracy
        q = q.float()
        k = k.float()
        v = v.float()

        # Compute attention logits: Q @ K^T (BTHD format)
        if reorder_ops:
            logits = torch.einsum("bqhd,bkhd->bhqk", q, k * scale)
        else:
            logits = torch.einsum("bqhd,bkhd->bhqk", q * scale, k)

        logits = logits + bias

        if score_mod is not None:
            logits = logits + score_mod.view(B, 1, 1, Lk)

        # Causal mask: position i attends only to j <= i
        if causal:
            causal_mask = torch.triu(torch.ones(Lq, Lk, dtype=torch.bool, device=q.device), diagonal=1)
            logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.sigmoid(logits)

        if key_padding_mask is not None:
            key_mask = key_padding_mask.view(B, 1, 1, Lk)
            attn = attn.masked_fill(key_mask, 0.0)

        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)

        if query_padding_mask is not None:
            query_mask = query_padding_mask.view(B, Lq, 1, 1)
            out = out.masked_fill(query_mask, 0.0)

    else:
        # Mixed precision mode: matches Triton kernel behavior
        # QK^T in input dtype, promote to FP32 for bias/sigmoid, cast back for output

        # Compute Q @ K^T in input dtype (FP16/BF16)
        if reorder_ops:
            logits = torch.einsum("bqhd,bkhd->bhqk", q, k * scale)
        else:
            logits = torch.einsum("bqhd,bkhd->bhqk", q * scale, k)

        # Promote to FP32 when adding bias (matches kernel)
        logits = logits * 1.0 + bias

        if score_mod is not None:
            logits = logits + score_mod.view(B, 1, 1, Lk)

        # Causal mask: position i attends only to j <= i
        if causal:
            causal_mask = torch.triu(torch.ones(Lq, Lk, dtype=torch.bool, device=q.device), diagonal=1)
            logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.sigmoid(logits)

        if key_padding_mask is not None:
            key_mask = key_padding_mask.view(B, 1, 1, Lk)
            attn = attn.masked_fill(key_mask, 0.0)

        # Cast back to input dtype for final matmul
        attn = attn.to(dtype_og)

        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)

        if query_padding_mask is not None:
            query_mask = query_padding_mask.view(B, Lq, 1, 1)
            out = out.masked_fill(query_mask, 0.0)

    return out.to(dtype_og)
