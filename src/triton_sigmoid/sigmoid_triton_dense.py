"""Optimized Triton kernels for sigmoid attention with dense (same-length) sequences.

Grid Configuration:
- Forward/Backward dQ: (cdiv(T, BLOCK_M), B * H)
- Backward dK/dV: (cdiv(T, BLOCK_N), B * H)
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .kernel_utils import DKDV_CONFIGS, DQ_CONFIGS, FWD_CONFIGS, sigmoid

# ============================================================================
# Dense Forward Kernel
# ============================================================================


@triton.autotune(configs=FWD_CONFIGS, key=["D", "T_bucketed", "IS_FP32", "IS_BF16", "IS_CAUSAL", "HAS_SCORE_MOD"])
@triton.jit
def _fwd_dense_kernel(
    Q,
    K,
    V,
    Out,
    bias_ptr,
    score_mod_ptr,
    T,
    H,
    sm_scale_ptr,
    stride_q_b,
    stride_q_tok,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_tok,
    stride_k_h,
    stride_k_d,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_BF16: tl.constexpr,
    T_bucketed: tl.constexpr,
    HAS_SCORE_MOD: tl.constexpr,
):
    """Forward kernel: Out = sigmoid(Q @ K^T * scale + bias + score_mod) @ V

    Grid: (cdiv(T, BLOCK_M), B * H) - parallelized over queries
    Causal masking: position i attends only to j <= i when IS_CAUSAL=True
    """

    # Determine computation dtype
    if IS_FP32:
        dtype = tl.float32
    elif IS_BF16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16

    # Get program indices
    pid_m = tl.program_id(0)  # Query block index
    pid_bh = tl.program_id(1)  # Batch-head index

    # Compute offsets
    off_b = pid_bh // H
    off_h = pid_bh % H
    offs_d = tl.arange(0, D)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)

    # Compute base pointer offsets (batch only, head applied per-token)
    off_q = (off_b * stride_q_b).to(tl.int64)

    # Mask for query blocks (only for partial blocks)
    mask_m = offs_m < T

    # Load bias for this batch element
    bias = tl.load(bias_ptr + off_b)

    # Load sm_scale from pointer
    sm_scale = tl.load(sm_scale_ptr)

    # Compute base pointer offsets (batch only, head applied per-token)
    off_k = (off_b * stride_k_b).to(tl.int64)

    # Advance pointers to current batch
    Q += off_q
    K += off_k
    V += off_k

    # Load query block: [BLOCK_M, D]
    q_ptrs = Q + offs_m[:, None] * stride_q_tok + off_h * stride_q_h + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Precompute dimension stride offsets
    offs_d_k = offs_d[:, None] * stride_k_d  # For K^T: [D, BLOCK_N]
    offs_d_v = offs_d[None, :] * stride_k_d  # For V: [BLOCK_N, D]

    # Initialize output accumulator in FP32 for numerical stability
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Loop over key/value blocks
    for start_n in tl.range(0, T, BLOCK_N):
        # Compute key/value token offsets for this block
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < T

        # Load K^T block: [D, BLOCK_N]
        kT_ptrs = K + offs_n[None, :] * stride_k_tok + off_h * stride_k_h + offs_d_k
        kT = tl.load(kT_ptrs, mask=mask_n[None, :], other=0.0)

        # Load V block: [BLOCK_N, D]
        v_ptrs = V + offs_n[:, None] * stride_k_tok + off_h * stride_k_h + offs_d_v
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute attention scores: Q @ K^T -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q, kT, allow_tf32=False, out_dtype=tl.float32)

        # Apply scaling and bias in single FMA operation
        s = tl.fma(s, sm_scale, bias)

        # Load and apply per-token score_mod if present
        if HAS_SCORE_MOD:
            # Load score_mod vector for current batch and key block [BLOCK_N]
            score_bias = tl.load(score_mod_ptr + off_b * T + offs_n, mask=mask_n, other=0.0)
            # Broadcast to [BLOCK_M, BLOCK_N] and add to scores
            s = s + score_bias[None, :]

        # Apply causal mask (if enabled) and validity mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            combined_mask = mask_m[:, None] & mask_n[None, :] & causal_mask
        else:
            combined_mask = mask_m[:, None] & mask_n[None, :]

        # Mask invalid positions with -inf BEFORE sigmoid
        s = tl.where(combined_mask, s, float('-inf'))

        # Apply fast sigmoid activation
        p = sigmoid(s)

        # Accumulate weighted values: acc += p @ V
        acc = tl.dot(p.to(dtype), v, acc=acc, allow_tf32=False, out_dtype=tl.float32)

    # Store output block: [BLOCK_M, D]
    Out += off_q
    o_ptrs = Out + offs_m[:, None] * stride_q_tok + off_h * stride_q_h + offs_d[None, :] * stride_q_d
    tl.store(o_ptrs, acc.to(dtype), mask=mask_m[:, None])


# ============================================================================
# Dense Backward dQ Kernel
# ============================================================================


@triton.autotune(configs=DQ_CONFIGS, key=["D", "T_bucketed", "IS_FP32", "IS_BF16", "IS_CAUSAL", "HAS_SCORE_MOD"])
@triton.jit
def _bwd_dense_dq_kernel(
    Q,
    K,
    V,
    dO,
    dQ,
    bias_ptr,
    score_mod_ptr,
    T,
    H,
    sm_scale_ptr,
    stride_q_b,
    stride_q_tok,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_tok,
    stride_k_h,
    stride_k_d,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_BF16: tl.constexpr,
    T_bucketed: tl.constexpr,
    HAS_SCORE_MOD: tl.constexpr,
):
    """Backward dQ kernel: computes query gradients via recomputation.

    Recomputes forward pass (attention weights) and applies sigmoid derivative.
    Grid: (cdiv(T, BLOCK_M), B * H) - parallelized over queries
    """
    # Determine computation dtype
    if IS_FP32:
        dtype = tl.float32
    elif IS_BF16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16

    # Get program indices
    pid_m = tl.program_id(0)  # Query block index
    pid_bh = tl.program_id(1)  # Batch-head index

    # Compute offsets
    off_b = pid_bh // H
    off_h = pid_bh % H
    offs_d = tl.arange(0, D)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)

    # Compute base pointer offsets (batch only, head applied per-token)
    off_q = (off_b * stride_q_b).to(tl.int64)

    # Mask for query blocks
    mask_m = offs_m < T

    # Load bias
    bias = tl.load(bias_ptr + off_b)

    # Load sm_scale from pointer
    sm_scale = tl.load(sm_scale_ptr)

    # Compute base pointer offsets (batch only, head applied per-token)
    off_k = (off_b * stride_k_b).to(tl.int64)

    # Advance pointers to current batch
    Q += off_q
    K += off_k
    V += off_k
    dO += off_q

    # Load Q and dO blocks: [BLOCK_M, D]
    q_ptrs = Q + offs_m[:, None] * stride_q_tok + off_h * stride_q_h + offs_d[None, :] * stride_q_d
    do_ptrs = dO + offs_m[:, None] * stride_q_tok + off_h * stride_q_h + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

    # Precompute dimension stride offsets
    offs_d_kT = offs_d[:, None] * stride_k_d  # For K^T: [D, BLOCK_N]
    offs_d_vT = offs_d[:, None] * stride_k_d  # For V^T: [D, BLOCK_N]

    # Initialize dQ accumulator in fp32 for numerical stability
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Loop over key/value blocks
    for start_n in tl.range(0, T, BLOCK_N):
        # Compute key/value token offsets for this block
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < T

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            combined_mask = mask_m[:, None] & mask_n[None, :] & causal_mask
        else:
            combined_mask = mask_m[:, None] & mask_n[None, :]

        # Load K^T block: [D, BLOCK_N]
        kT_ptrs = K + offs_n[None, :] * stride_k_tok + off_h * stride_k_h + offs_d_kT
        kT = tl.load(kT_ptrs, mask=mask_n[None, :], other=0.0)

        # Load V^T block: [D, BLOCK_N]
        vT_ptrs = V + offs_n[None, :] * stride_k_tok + off_h * stride_k_h + offs_d_vT
        vT = tl.load(vT_ptrs, mask=mask_n[None, :], other=0.0)

        # Recompute forward: attention scores Q @ K^T -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q, kT, allow_tf32=False, out_dtype=tl.float32)
        s = tl.fma(s, sm_scale, bias)

        # Load and apply per-token score_mod if present
        if HAS_SCORE_MOD:
            # Load score_mod vector for current batch and key block [BLOCK_N]
            score_bias = tl.load(score_mod_ptr + off_b * T + offs_n, mask=mask_n, other=0.0)
            # Broadcast to [BLOCK_M, BLOCK_N] and add to scores
            s = s + score_bias[None, :]

        # Mask invalid positions BEFORE sigmoid
        s = tl.where(combined_mask, s, float('-inf'))

        # Recompute forward: attention weights
        p = sigmoid(s)

        # Compute dO @ V^T -> [BLOCK_M, BLOCK_N]
        dOv = tl.dot(do, vT, allow_tf32=False, out_dtype=tl.float32)

        # Apply sigmoid derivative: sigmoid'(s) = sigmoid(s) * (1 - sigmoid(s))
        p_dOv = p * dOv
        dsig = tl.fma(p, -p_dOv, p_dOv)

        # Mask gradients for invalid/causal positions
        dsig = tl.where(combined_mask, dsig, 0.0)

        # Accumulate dQ: dsig @ K -> [BLOCK_M, D]
        dq = tl.dot(dsig.to(dtype), tl.trans(kT), acc=dq, allow_tf32=False, out_dtype=tl.float32)

    # Apply final scaling (chain rule from forward)
    dq *= sm_scale

    # Store dQ block: [BLOCK_M, D]
    dQ += off_q
    dq_ptrs = dQ + offs_m[:, None] * stride_q_tok + off_h * stride_q_h + offs_d[None, :] * stride_q_d
    tl.store(dq_ptrs, dq.to(dtype), mask=mask_m[:, None])


# ============================================================================
# Dense Backward dK/dV Kernel
# ============================================================================


@triton.autotune(configs=DKDV_CONFIGS, key=["D", "T_bucketed", "IS_FP32", "IS_BF16", "IS_CAUSAL", "HAS_SCORE_MOD"])
@triton.jit
def _bwd_dense_dkdv_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    bias_ptr,
    score_mod_ptr,
    T,
    H,
    sm_scale_ptr,
    stride_q_b,
    stride_q_tok,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_tok,
    stride_k_h,
    stride_k_d,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_BF16: tl.constexpr,
    T_bucketed: tl.constexpr,
    HAS_SCORE_MOD: tl.constexpr,
):
    """Backward dK/dV kernel: computes key and value gradients via transposed recomputation.

    Grid: (cdiv(T, BLOCK_N), B * H) - parallelized over keys
    Causal masking: key j receives gradients from queries i where i >= j
    """
    # Determine computation dtype
    if IS_FP32:
        dtype = tl.float32
    elif IS_BF16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16

    # Get program indices
    pid_n = tl.program_id(0)  # Key/value block index
    pid_bh = tl.program_id(1)  # Batch-head index

    # Compute offsets
    off_b = pid_bh // H
    off_h = pid_bh % H
    offs_d = tl.arange(0, D)
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # Compute base pointer offsets (batch only, head applied per-token)
    off_k = (off_b * stride_k_b).to(tl.int64)

    # Key-value masks
    mask_n = offs_n < T

    # Load bias term
    bias = tl.load(bias_ptr + off_b)

    # Load sm_scale from pointer
    sm_scale = tl.load(sm_scale_ptr)

    # Compute base pointer offsets (batch only, head applied per-token)
    off_q = (off_b * stride_q_b).to(tl.int64)

    # Advance pointers to current batch
    Q += off_q
    K += off_k
    V += off_k
    dO += off_q

    # Load K and V blocks: [BLOCK_N, D]
    k_ptrs = K + offs_n[:, None] * stride_k_tok + off_h * stride_k_h + offs_d[None, :] * stride_k_d
    v_ptrs = V + offs_n[:, None] * stride_k_tok + off_h * stride_k_h + offs_d[None, :] * stride_k_d
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    # Precompute dimension stride offsets (Q and dO share same layout)
    offs_d_qdo = offs_d[None, :] * stride_q_d  # For Q and dO: [BLOCK_M, D]

    # Initialize accumulators
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    # Loop over query blocks
    for start_m in tl.range(0, T, BLOCK_M):
        # Compute query token offsets for this block
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < T

        # Causal masking from key perspective
        if IS_CAUSAL:
            causal_mask = offs_m[None, :] >= offs_n[:, None]
            combined_mask = mask_n[:, None] & mask_m[None, :] & causal_mask
        else:
            combined_mask = mask_n[:, None] & mask_m[None, :]

        # Load Q block: [BLOCK_M, D]
        q_ptrs = Q + offs_m[:, None] * stride_q_tok + off_h * stride_q_h + offs_d_qdo
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        # Load dO block: [BLOCK_M, D]
        do_ptrs = dO + offs_m[:, None] * stride_q_tok + off_h * stride_q_h + offs_d_qdo
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

        # Recompute forward: attention scores K @ Q^T -> [BLOCK_N, BLOCK_M]
        sT = tl.dot(k, tl.trans(q), allow_tf32=False, out_dtype=tl.float32)
        sT = tl.fma(sT, sm_scale, bias)

        # Load and apply per-token score_mod if present
        if HAS_SCORE_MOD:
            # Load score_mod vector for current batch and key block [BLOCK_N]
            score_bias = tl.load(score_mod_ptr + off_b * T + offs_n, mask=mask_n, other=0.0)
            # Broadcast to [BLOCK_N, BLOCK_M] and add to scores
            sT = sT + score_bias[:, None]

        # Mask invalid positions BEFORE sigmoid
        sT = tl.where(combined_mask, sT, float('-inf'))

        # Recompute forward: attention weights (transposed)
        pT = sigmoid(sT)

        # Accumulate dV: pT @ dO -> [BLOCK_N, D]
        dv = tl.dot(pT.to(dtype), do, acc=dv, allow_tf32=False, out_dtype=tl.float32)

        # Compute V @ dO^T for sigmoid derivative -> [BLOCK_N, BLOCK_M]
        dOvT = tl.dot(v, tl.trans(do), allow_tf32=False, out_dtype=tl.float32)

        # Apply sigmoid derivative (transposed perspective)
        pT_dOvT = pT * dOvT
        dsigT = tl.fma(pT, -pT_dOvT, pT_dOvT)

        # Mask gradients for invalid/causal positions
        dsigT = tl.where(combined_mask, dsigT, 0.0)

        # Accumulate dK: dsigT @ Q -> [BLOCK_N, D]
        dk = tl.dot(dsigT.to(dtype), q, acc=dk, allow_tf32=False, out_dtype=tl.float32)

    # Apply final scaling to dK
    dk *= sm_scale

    # Store dK and dV blocks
    dK += off_k
    dV += off_k
    dk_ptrs = dK + offs_n[:, None] * stride_k_tok + off_h * stride_k_h + offs_d[None, :] * stride_k_d
    dv_ptrs = dV + offs_n[:, None] * stride_k_tok + off_h * stride_k_h + offs_d[None, :] * stride_k_d
    tl.store(dk_ptrs, dk.to(dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dtype), mask=mask_n[:, None])


# ============================================================================
# Triton Ops Registration
# ============================================================================


@triton_op("myops::sigmoid_attention_fwd", mutates_args={})
def sigmoid_attention_fwd_op(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    bias: torch.Tensor,
    score_mod: Optional[torch.Tensor],
    sm_scale: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """
    Forward pass for sigmoid attention with dense (same-length) sequences.

    Args:
        Q: Query tensor [B, T, H, D]
        K: Key tensor [B, T, H, D]
        V: Value tensor [B, T, H, D]
        bias: Bias tensor per sequence [B]
        score_mod: Optional per-token bias [B, T] or None
        sm_scale: Attention scale factor
        is_causal: Whether to apply causal masking

    Returns:
        Out: Attention output [B, T, H, D]
    """

    # Extract shapes
    B, T, H, D = Q.shape

    # Extract strides (BTHD format)
    stride_q_b, stride_q_tok, stride_q_h, stride_q_d = Q.stride()
    stride_k_b, stride_k_tok, stride_k_h, stride_k_d = K.stride()

    # Allocate output
    Out = torch.empty_like(Q)

    # Handle score_mod
    if score_mod is not None:
        has_score_mod = True
        score_mod_ptr = score_mod
    else:
        has_score_mod = False
        score_mod_ptr = Q  # dummy pointer (won't be accessed when HAS_SCORE_MOD=False)

    # Inline bucketing for T
    if T <= 512:
        T_bucketed = 512
    elif T <= 2048:
        T_bucketed = 2048
    elif T <= 8192:
        T_bucketed = 8192
    else:
        T_bucketed = 16384

    # Dtype mapping
    is_fp32 = Q.dtype == torch.float32
    is_bf16 = Q.dtype == torch.bfloat16

    # Launch kernel
    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)

    wrap_triton(_fwd_dense_kernel)[grid](  # type: ignore[arg-type]
        Q,
        K,
        V,
        Out,
        bias,
        score_mod_ptr,
        T,
        H,
        sm_scale,
        stride_q_b,
        stride_q_tok,
        stride_q_h,
        stride_q_d,
        stride_k_b,
        stride_k_tok,
        stride_k_h,
        stride_k_d,
        D=D,
        IS_CAUSAL=is_causal,
        IS_FP32=is_fp32,
        IS_BF16=is_bf16,
        T_bucketed=T_bucketed,
        HAS_SCORE_MOD=has_score_mod,
    )

    return Out


@triton_op("myops::sigmoid_attention_bwd", mutates_args={})
def sigmoid_attention_bwd_op(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    bias: torch.Tensor,
    score_mod: Optional[torch.Tensor],
    sm_scale: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for sigmoid attention with dense sequences.

    Args:
        Q: Query tensor [B, T, H, D]
        K: Key tensor [B, T, H, D]
        V: Value tensor [B, T, H, D]
        dO: Output gradient [B, T, H, D]
        bias: Bias tensor per sequence [B]
        score_mod: Optional per-token bias [B, T] or None
        sm_scale: Attention scale factor
        is_causal: Whether to apply causal masking

    Returns:
        dQ, dK, dV: Gradients for Q, K, V
    """
    if not dO.is_contiguous():
        dO = dO.contiguous()

    # Extract shapes
    B, T, H, D = Q.shape

    # Allocate output gradients
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    # Handle score_mod
    if score_mod is not None:
        has_score_mod = True
        score_mod_ptr = score_mod
    else:
        has_score_mod = False
        score_mod_ptr = Q  # dummy pointer (won't be accessed when HAS_SCORE_MOD=False)

    # Extract strides (shared by both kernels, BTHD format)
    stride_q_b, stride_q_tok, stride_q_h, stride_q_d = Q.stride()
    stride_k_b, stride_k_tok, stride_k_h, stride_k_d = K.stride()

    # Dtype mapping
    is_fp32 = Q.dtype == torch.float32
    is_bf16 = Q.dtype == torch.bfloat16

    # Inline bucketing for T
    if T <= 512:
        T_bucketed = 512
    elif T <= 2048:
        T_bucketed = 2048
    elif T <= 8192:
        T_bucketed = 8192
    else:
        T_bucketed = 16384

    # Launch dQ kernel: iterate over query positions
    grid_dq = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
    wrap_triton(_bwd_dense_dq_kernel)[grid_dq](  # type: ignore[arg-type]
        Q,
        K,
        V,
        dO,
        dQ,
        bias,
        score_mod_ptr,
        T,
        H,
        sm_scale,
        stride_q_b,
        stride_q_tok,
        stride_q_h,
        stride_q_d,
        stride_k_b,
        stride_k_tok,
        stride_k_h,
        stride_k_d,
        D=D,
        IS_CAUSAL=is_causal,
        IS_FP32=is_fp32,
        IS_BF16=is_bf16,
        T_bucketed=T_bucketed,
        HAS_SCORE_MOD=has_score_mod,
    )

    # Launch dKdV kernel: iterate over key/value positions
    grid_dkdv = lambda META: (triton.cdiv(T, META['BLOCK_N']), B * H)
    wrap_triton(_bwd_dense_dkdv_kernel)[grid_dkdv](  # type: ignore[arg-type]
        Q,
        K,
        V,
        dO,
        dK,
        dV,
        bias,
        score_mod_ptr,
        T,
        H,
        sm_scale,
        stride_q_b,
        stride_q_tok,
        stride_q_h,
        stride_q_d,
        stride_k_b,
        stride_k_tok,
        stride_k_h,
        stride_k_d,
        D=D,
        IS_CAUSAL=is_causal,
        IS_FP32=is_fp32,
        IS_BF16=is_bf16,
        T_bucketed=T_bucketed,
        HAS_SCORE_MOD=has_score_mod,
    )

    return dQ, dK, dV


# ============================================================================
# Autograd Setup
# ============================================================================


def _dense_backward(ctx, dO):
    """Backward pass for dense sigmoid attention.

    Args:
        ctx: Autograd context with saved tensors
        dO: Gradient of loss with respect to output [B, T, H, D]

    Returns:
        (dQ, dK, dV, None, None, None, None)
    """
    if ctx.has_score_mod:
        Q, K, V, bias, score_mod = ctx.saved_tensors
    else:
        Q, K, V, bias = ctx.saved_tensors
        score_mod = None

    dQ, dK, dV = sigmoid_attention_bwd_op(
        Q,
        K,
        V,
        dO,
        bias,
        score_mod,
        ctx.sm_scale,
        ctx.is_causal,
    )

    # Return gradients: dQ, dK, dV, d(bias)=None, d(score_mod)=None, d(sm_scale)=None, d(is_causal)=None
    return dQ, dK, dV, None, None, None, None


def _dense_setup_context(ctx, inputs, output):
    """Setup autograd context for backward pass (saves Q, K, V for recomputation).

    Args:
        ctx: Autograd context to store tensors
        inputs: Tuple of forward pass inputs
        output: Forward pass output (not saved, will be recomputed)
    """
    Q, K, V, bias, score_mod, sm_scale, is_causal = inputs

    # Save tensors needed for backward
    if score_mod is not None:
        ctx.save_for_backward(Q, K, V, bias, score_mod)
        ctx.has_score_mod = True
    else:
        ctx.save_for_backward(Q, K, V, bias)
        ctx.has_score_mod = False

    ctx.sm_scale = sm_scale
    ctx.is_causal = is_causal


# Register autograd for the forward op
sigmoid_attention_fwd_op.register_autograd(
    _dense_backward,
    setup_context=_dense_setup_context,
)


# ============================================================================
# Public API
# ============================================================================


def sigmoid_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    sigmoid_bias: Optional[float] = None,
    score_mod: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sigmoid attention for dense (same-length) sequences with automatic differentiation.

    Computes: output = sigmoid(Q @ K^T * scale + sigmoid_bias + score_mod[k_idx]) @ V

    Args:
        query: Query tensor [B, T, H, D]
        key: Key tensor [B, T, H, D]
        value: Value tensor [B, T, H, D]
        is_causal: If True, apply causal masking (position i attends to j <= i)
        scale: Attention scaling factor (default: 1/sqrt(D))
        sigmoid_bias: Uniform bias added to all attention scores (scalar per batch).
            Default: -log(T) for numerical stability.
        score_mod: Optional per-token bias tensor. Shape [B, T].
            Applied after sigmoid_bias: s = (Q@K^T * scale + sigmoid_bias) + score_mod[k_idx].
            Broadcasts across all queries and heads.
            Not trainable (no gradients computed). Default: None.

    Returns:
        output: Attention output [B, T, H, D]

    Example:
        >>> q = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 1024, 8, 64, device='cuda', dtype=torch.float16)
        >>> # Non-causal attention
        >>> output = sigmoid_attention(q, k, v, is_causal=False)
        >>> # Causal attention for autoregressive generation
        >>> output_causal = sigmoid_attention(q, k, v, is_causal=True)
        >>> # With per-token biases
        >>> score_mod = torch.randn(2, 1024, device='cuda', dtype=torch.float16)
        >>> output_biased = sigmoid_attention(q, k, v, score_mod=score_mod)
    """
    # Validate shapes
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("Expected query, key, and value to be 4D tensors")

    B, T_q, H, D = query.shape
    B_k, T_k, H_k, D_k = key.shape
    B_v, T_v, H_v, D_v = value.shape

    if B != B_k or B != B_v:
        raise ValueError(f"Batch size mismatch: Q={B}, K={B_k}, V={B_v}")
    if H != H_k or H != H_v:
        raise ValueError(f"Number of heads mismatch: Q={H}, K={H_k}, V={H_v}")
    if D != D_k or D != D_v:
        raise ValueError(f"Head dimension mismatch: Q={D}, K={D_k}, V={D_v}")
    if T_q != T_k or T_q != T_v:
        raise ValueError(f"Dense attention requires same sequence length for Q, K, V, got {T_q}, {T_k}, {T_v}")

    # Validate score_mod if provided
    if score_mod is not None:
        if score_mod.shape != (B, T_k):
            raise ValueError(f"score_mod must have shape ({B}, {T_k}), got {score_mod.shape}")
        score_mod = score_mod.contiguous()

    # Ensure Q, K, V are contiguous
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # Mark static dimensions for torch.compile
    if torch.compiler.is_compiling():
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -2)  # num_heads (H)
            torch._dynamo.mark_static(x, -1)  # head_dim (D)

    # Extract sequence length
    T = T_q

    # Compute bias
    if sigmoid_bias is None:
        bias = torch.full((B,), -math.log(float(T)), dtype=torch.float32, device=query.device)
    else:
        bias = torch.full((B,), float(sigmoid_bias), dtype=torch.float32, device=query.device)

    # Ensure bias is contiguous (defensive programming)
    bias = bias.contiguous()

    # Compute scale as 0D tensor
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=query.device)

    return sigmoid_attention_fwd_op(query, key, value, bias, score_mod, scale_tensor, is_causal)
