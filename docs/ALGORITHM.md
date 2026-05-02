# Algorithm Details

## Overview

Sigmoid attention computes: `output = sigmoid(Q @ K^T * scale + bias + score_mod) @ V`

Where:
- `scale = 1 / sqrt(head_dim)` (default)
- `bias = -log(num_valid_tokens)` per sequence
- `score_mod = per-token bias [batch, seq_len]` (optional, default: None)
- `sigmoid(x) = 1 / (1 + exp(-x))` using fast tanh approximation (default)

## Sigmoid Implementation

By default, uses fast sigmoid approximation via `sigmoid(x) ≈ 0.5 * (1 + tanh(0.5 * x))` using NVIDIA's hardware `tanh.approx` instruction.

For maximum accuracy, enable accurate sigmoid:

```bash
export TRITON_SIGMOID_ACCURATE=1
python your_script.py
```

Fast (default) is ~15-20% faster. Accurate is best for FP32 precision validation.

## Kernel Implementations

### Dense Kernel

Assumes all sequences have the same length. No padding mask checks or offset arithmetic.

### Padded Kernel

The padded kernel (`sigmoid_attention_padded`) handles variable-length sequences by:
- Taking per-batch sequence length tensors (`seq_lens_k`, `seq_lens_q`)
- Masking padded positions in attention scores (set to -inf before sigmoid)
- Setting padded output positions to zero
- Computing per-sequence sigmoid bias based on actual (non-padded) token count

The padded kernel maintains torch.compile compatibility by using fixed tensor shapes with dynamic masking, avoiding graph breaks from shape changes.

## Causal Masking

When `is_causal=True`, position i attends only to j <= i. Scores where j > i are masked to -inf before sigmoid.
