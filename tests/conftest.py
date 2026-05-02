"""Pytest configuration and shared fixtures for sigmoid attention tests.

This module provides common test fixtures and configuration for the
sigmoid attention test suite. All tests require CUDA availability.

Fixtures:
    cuda_device: Provides a CUDA device handle and skips tests if CUDA unavailable
    reset_test_state: Resets random seeds and CUDA state before each test
    get_tolerance: Returns numerical tolerance based on dtype
"""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def cuda_device():
    """Provide CUDA device and skip tests if CUDA is unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available", allow_module_level=True)
    return torch.device('cuda')


def get_tolerance(dtype):
    """Get numerical tolerance threshold based on dtype."""
    if dtype == torch.float32:
        return 1e-5  # Accurate sigmoid with FP32 precision
    elif dtype == torch.float16:
        return 1e-3  # FP16 rounding errors
    else:  # bfloat16
        return 7e-3  # BF16 reduced mantissa precision


@pytest.fixture(autouse=True)
def reset_test_state():
    """Reset random seeds and CUDA state for reproducible tests."""
    torch.manual_seed(32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(32)
        torch.cuda.manual_seed_all(32)
    np.random.seed(32)
    random.seed(32)

    # Clear CUDA cache to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
