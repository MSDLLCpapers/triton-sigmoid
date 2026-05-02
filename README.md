# Triton Sigmoid Attention

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/pytorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Fused sigmoid attention kernels for NVIDIA GPUs built with Triton.

## Overview

**Two implementations:**
- **Dense** - Same-length sequences (fastest)
- **Padded** - Variable-length with padding masks

Features torch.compile support and causal masking. Requires Ampere architecture or newer.

## Installation

```bash
git clone https://github.com/MSDLLCpapers/triton-sigmoid.git
cd triton-sigmoid

# Install with uv (recommended)
uv sync --extra cu128 --extra dev
source .venv/bin/activate
```

<details>
<summary>Other CUDA versions</summary>

```bash
# For CUDA 12.6 (compatible with CUDA 12.1+)
uv sync --extra cu126 --extra dev

# For CUDA 13.0+ (latest)
uv sync --extra cu130 --extra dev
```
</details>

## Quick Start

### Dense Attention (Same-Length Sequences)

```python
import torch
from triton_sigmoid import sigmoid_attention

batch, seq_len, n_heads, head_dim = 2, 1024, 8, 64
q = torch.randn(batch, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)

output = sigmoid_attention(q, k, v, is_causal=False)
output_causal = sigmoid_attention(q, k, v, is_causal=True)
```

### Padded Attention (Variable-Length Sequences)

```python
import torch
from triton_sigmoid import sigmoid_attention_padded

batch, seq_len, n_heads, head_dim = 2, 1024, 8, 64
q = torch.randn(batch, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seq_len, n_heads, head_dim, device='cuda', dtype=torch.float16)

seq_lens_k = torch.tensor([800, 950], device='cuda', dtype=torch.int32)
seq_lens_q = torch.tensor([800, 950], device='cuda', dtype=torch.int32)

output = sigmoid_attention_padded(q, k, v, seq_lens_k=seq_lens_k, seq_lens_q=seq_lens_q)

compiled_fn = torch.compile(sigmoid_attention_padded)
output = compiled_fn(q, k, v, seq_lens_k=seq_lens_k, seq_lens_q=seq_lens_q, is_causal=True)
```

## Performance

![TFLOPS Comparison](benchmarks/results/H100/tflops_comparison.png)

See [benchmarks/README.md](benchmarks/README.md) for details.

## Documentation

- [API Reference](docs/API.md) - Detailed function signatures and parameters
- [Algorithm Details](docs/ALGORITHM.md) - Implementation details and sigmoid variants
- [Benchmarking Guide](benchmarks/README.md) - Performance benchmarks and how to run them
- [Testing Guide](docs/TESTING.md) - Running tests and test structure

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@misc{sadashivaiah2026bettermodelsfastertraining,
      title={Better Models, Faster Training: Sigmoid Attention for single-cell Foundation Models}, 
      author={Vijay Sadashivaiah and Georgios Dasoulas and Judith Mueller and Soumya Ghosh},
      year={2026},
      eprint={2604.27124},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.27124}, 
}
```

## Acknowledgments

- Built with [Triton](https://github.com/openai/triton)
- Inspired by [Flash Attention](https://github.com/Dao-AILab/flash-attention)
