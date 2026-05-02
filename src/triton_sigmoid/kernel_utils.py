"""Shared utilities for Triton sigmoid attention kernels.

This module contains configuration settings and helper functions used by all sigmoid attention kernels.
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def is_hopper_or_better() -> bool:
    """Check if GPU is Hopper architecture (compute capability >= 9.0) or newer.

    Returns:
        True if H100 or newer, False otherwise.
    """
    return torch.cuda.get_device_capability()[0] >= 9


def is_ampere() -> bool:
    """Check if GPU is Ampere architecture (compute capability 8.x).

    Returns:
        True if A100/A40 generation, False otherwise.
    """
    return torch.cuda.get_device_capability()[0] == 8


# -----------------------------------------------------------------------------
# Autotune configurations
# -----------------------------------------------------------------------------

# Forward kernel configs
if is_hopper_or_better():
    FWD_CONFIGS = [
        # Medium tiles with aggressive pipelining
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        # Small tiles for maximum occupancy
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        # Wide tiles for large N dimensions
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        # Tall tiles for large M dimensions
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        # Large tiles for high instruction-level parallelism
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ]
elif is_ampere():
    FWD_CONFIGS = [
        # Small tiles for maximum occupancy
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        # Medium tiles with moderate pipelining
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        # Larger tiles for better instruction-level parallelism
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ]
else:
    FWD_CONFIGS = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ]

# Backward dQ kernel configs
if is_hopper_or_better():
    DQ_CONFIGS = [
        # Medium tiles with varied pipelining
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        # Small tiles for high occupancy
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        # Wide tiles for gradient accumulation
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        # Tall tiles for large batch dimensions
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ]
elif is_ampere():
    DQ_CONFIGS = [
        # Small tiles for high occupancy
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        # Medium tiles with moderate pipelining
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        # Larger tiles for gradient accumulation
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ]
else:
    DQ_CONFIGS = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ]

# Backward dK/dV kernel configs
if is_hopper_or_better():
    DKDV_CONFIGS = [
        # Medium tiles with varied pipelining
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        # Small tiles for high occupancy
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        # Wide tiles for key/value dimensions
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        # Tall tiles for sequence length dimensions
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ]
elif is_ampere():
    DKDV_CONFIGS = [
        # Small tiles for high occupancy
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        # Medium tiles with moderate pipelining
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        # Larger tiles for key/value gradients
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ]
else:
    DKDV_CONFIGS = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ]

# Override configs for pytest
if "PYTEST_VERSION" in os.environ:
    FWD_CONFIGS = DQ_CONFIGS = DKDV_CONFIGS = [triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2)]


# -----------------------------------------------------------------------------
# Sigmoid implementation
# -----------------------------------------------------------------------------
# Choose sigmoid implementation based on environment variable
# Set TRITON_SIGMOID_ACCURATE to: 1, true, yes, or on (case-insensitive)
# Default: fast approx sigmoid

_accurate_str = os.getenv("TRITON_SIGMOID_ACCURATE", "").lower()
_use_accurate = _accurate_str in ("1", "true", "yes", "on")

USE_APPROX_SIGMOID = tl.constexpr(not _use_accurate)

# Log configuration (info when explicitly enabled, debug for default approx)
if _use_accurate:
    logger.info(
        "Sigmoid attention using accurate sigmoid (TRITON_SIGMOID_ACCURATE=%s)",
        os.getenv("TRITON_SIGMOID_ACCURATE", ""),
    )
else:
    logger.debug("Sigmoid attention using fast tanh approximation (default)")


@triton.jit
def sigmoid(x):
    """Sigmoid implementation for attention kernels.

    Uses fast tanh approximation by default (sigmoid(x) ≈ 0.5 * (1 + tanh(0.5 * x))),
    or accurate sigmoid if TRITON_SIGMOID_ACCURATE=1 env var is set.

    The fast approximation uses NVIDIA's PTX tanh.approx instruction, providing
    ~15-20% speedup with minimal accuracy loss for FP16/BF16.

    Mathematical derivation:
        sigmoid(x) = 1 / (1 + exp(-x))
                   = 0.5 * (1 + tanh(x/2))

    Args:
        x: Input tensor (typically attention logits after scaling + bias)

    Returns:
        sigmoid(x) values
    """
    if USE_APPROX_SIGMOID:
        x_half = x * 0.5
        t = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [x_half],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return tl.fma(t, 0.5, 0.5)
    else:
        return tl.sigmoid(x)
