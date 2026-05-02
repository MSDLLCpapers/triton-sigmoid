# API Reference

## `sigmoid_attention`

Dense sigmoid attention for same-length sequences.

```python
sigmoid_attention(
    query: torch.Tensor,        # [batch, seq_len, n_heads, head_dim]
    key: torch.Tensor,          # [batch, seq_len, n_heads, head_dim]
    value: torch.Tensor,        # [batch, seq_len, n_heads, head_dim]
    is_causal: bool = False,    # Apply causal masking
    scale: Optional[float] = None,        # Attention scale (default: 1/sqrt(head_dim))
    sigmoid_bias: Optional[float] = None, # Sigmoid bias (default: -log(seq_len))
    score_mod: Optional[torch.Tensor] = None,  # Per-token bias [batch, seq_len]
) -> torch.Tensor  # [batch, seq_len, n_heads, head_dim]
```

**Parameters:**
- `query`, `key`, `value`: Input tensors with shape `[batch, seq_len, n_heads, head_dim]` (BTHD layout)
- `is_causal`: If True, position i attends only to j <= i (autoregressive)
- `scale`: Attention scaling factor (default: 1/sqrt(head_dim))
- `sigmoid_bias`: Bias term for sigmoid (default: -log(seq_len))
- `score_mod`: Optional per-token bias with shape `[batch, seq_len]`. Applied after sigmoid_bias, broadcasts across all queries and heads.

**Returns:**
- Attention output with same shape as query

## `sigmoid_attention_padded`

Padded sigmoid attention for variable-length sequences with padding masks.

```python
sigmoid_attention_padded(
    query: torch.Tensor,        # [batch, seq_len, n_heads, head_dim]
    key: torch.Tensor,          # [batch, seq_len, n_heads, head_dim]
    value: torch.Tensor,        # [batch, seq_len, n_heads, head_dim]
    seq_lens_k: Optional[torch.Tensor] = None,  # [batch] int32 (actual key lengths)
    seq_lens_q: Optional[torch.Tensor] = None,  # [batch] int32 (actual query lengths)
    is_causal: bool = False,    # Apply causal masking
    scale: Optional[float] = None,        # Attention scale (default: 1/sqrt(head_dim))
    sigmoid_bias: Optional[float] = None, # Sigmoid bias (default: -log(num_valid_tokens))
    score_mod: Optional[torch.Tensor] = None,  # Per-token bias [batch, seq_len]
) -> torch.Tensor  # [batch, seq_len, n_heads, head_dim]
```

**Parameters:**
- `query`, `key`, `value`: Input tensors with shape `[batch, seq_len, n_heads, head_dim]` (BTHD layout, padded to uniform length)
- `seq_lens_k`: Actual key sequence lengths per batch element, shape `[batch]` dtype int32. None means no padding (all valid).
- `seq_lens_q`: Actual query sequence lengths per batch element, shape `[batch]` dtype int32. None means no padding (all valid).
- `is_causal`: If True, position i attends only to j <= i (autoregressive)
- `scale`: Attention scaling factor (default: 1/sqrt(head_dim))
- `sigmoid_bias`: Bias term for sigmoid (default: -log(num_valid_tokens) per sequence)
- `score_mod`: Optional per-token bias with shape `[batch, seq_len]`. Applied after sigmoid_bias, broadcasts across all queries and heads.

**Returns:**
- Attention output with same shape as query (padded positions set to zero)
