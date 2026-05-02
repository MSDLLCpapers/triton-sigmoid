"""Performance benchmarking for sigmoid attention.

Benchmarks Triton-optimized sigmoid attention (dense, padded) against PyTorch reference,
Flash Sigmoid, and Flash Attention (if available).

Providers by padding configuration:
- padding=0 (dense): torch, triton-dense, flash_sigmoid, flash_attn
- padding>0 (variable-length): torch, triton-padded, flash_attn


Parameters: Head dims (64, 128), seq lens (128-8192), dtypes (fp16, bf16), padding (0%, 25%, 50%)
Metrics: Latency, throughput (TFLOPS), memory bandwidth
Output: CSV files

Usage:
    python benchmarks/benchmark_sigmoid.py
    python benchmarks/benchmark_sigmoid.py --output-dir ./results
    python benchmarks/benchmark_sigmoid.py --providers triton-dense triton-padded torch
    python benchmarks/benchmark_sigmoid.py --padding-pcts 0 25 50
"""

import argparse
import os
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import triton

from triton_sigmoid import sigmoid_attention, sigmoid_attention_padded, sigmoid_attention_ref

# Device configuration
device_id = torch.cuda.current_device()
DEVICE = torch.device(f'cuda:{device_id}')

# Set seeds for reproducible benchmarks
torch.manual_seed(32)
if torch.cuda.is_available():
    torch.cuda.manual_seed(32)
    torch.cuda.manual_seed_all(32)
np.random.seed(32)
random.seed(32)


# Check flash function availability
HAS_FLASH_SIGMOID = False
HAS_FLASH_ATTN = False

try:
    from flash_sigmoid import flash_attn_func as flash_sigmoid_func

    HAS_FLASH_SIGMOID = True
    print("✓ flash_sigmoid_func imported successfully")
except (ImportError, NameError) as e:
    print(f"✗ flash_sigmoid_func not available: {e}")

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    HAS_FLASH_ATTN = True
    print("✓ flash_attn_func imported successfully")
except (ImportError, NameError) as e:
    print(f"✗ flash_attn_func not available: {e}")


def generate_seq_lens(batch, nc, padding_pct, seed=42):
    """Generate sequence lengths for a benchmark configuration.

    This defines the WORKLOAD - all backends must use same seq_lens for fair comparison.
    Deterministic based on config parameters for reproducibility.

    Args:
        batch: Batch size
        nc: Maximum sequence length
        padding_pct: Percentage of padding (0-100)
        seed: Random seed for deterministic generation

    Returns:
        seq_lens: Tensor of shape [batch] with sequence lengths
    """
    if padding_pct == 0:
        return torch.full((batch,), nc, dtype=torch.int32, device=DEVICE)

    # Deterministic generation based on config
    rng = torch.Generator(device=DEVICE).manual_seed(
        hash((batch, nc, padding_pct, seed)) % (2**31)
    )
    min_len = max(1, int(nc * (1 - padding_pct / 100)))
    return torch.randint(min_len, nc + 1, (batch,),
                        generator=rng, device=DEVICE, dtype=torch.int32)


def create_inputs(batch, nc, n_heads, hd, seq_lens, dtype, format):
    """Create input tensors in specified format using given workload.

    All padded/dense tensors use BTHD layout: [batch, tokens, heads, head_dim]
    Packed (varlen) format uses THD layout: [total_tokens, heads, head_dim]

    Args:
        batch: Batch size
        nc: Maximum sequence length
        n_heads: Number of attention heads
        hd: Head dimension
        seq_lens: Pre-generated sequence lengths (defines workload)
        dtype: torch.float16 or torch.bfloat16
        format: 'dense', 'padded', or 'varlen'

    Returns:
        dict with keys depending on format:
        - dense/padded: {'q', 'k', 'v', 'seq_lens'} - BTHD layout
        - varlen: {'q', 'k', 'v', 'cu_seqlens', 'max_seqlen', 'seq_lens'} - THD layout
    """
    if format == 'varlen':
        # Packed format for flash_attn_varlen_func: THD layout
        total_tokens = seq_lens.sum().item()
        max_seqlen = seq_lens.max().item()
        cu_seqlens = torch.cat([
            torch.zeros(1, device=DEVICE, dtype=torch.int32),
            seq_lens.cumsum(0, dtype=torch.int32)
        ])

        q = torch.randn(total_tokens, n_heads, hd, dtype=dtype, device=DEVICE, requires_grad=True)
        k = torch.randn(total_tokens, n_heads, hd, dtype=dtype, device=DEVICE, requires_grad=True)
        v = torch.randn(total_tokens, n_heads, hd, dtype=dtype, device=DEVICE, requires_grad=True)

        return {
            'q': q, 'k': k, 'v': v,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
            'seq_lens': seq_lens
        }

    else:  # padded or dense - always BTHD layout
        q = torch.randn(batch, nc, n_heads, hd, dtype=dtype, device=DEVICE, requires_grad=True)
        k = torch.randn(batch, nc, n_heads, hd, dtype=dtype, device=DEVICE, requires_grad=True)
        v = torch.randn(batch, nc, n_heads, hd, dtype=dtype, device=DEVICE, requires_grad=True)

        return {
            'q': q, 'k': k, 'v': v,
            'seq_lens': seq_lens if format == 'padded' else None
        }




def _setup_benchmark_fn(forward_call, mode):
    """Set up benchmark function for forward or backward pass.

    Args:
        forward_call: Callable that executes the forward pass
        mode: 'fwd' or 'bwd'

    Returns:
        (fn, o, do): benchmark function and optional output tensors for cleanup
    """
    if mode == "fwd":
        return forward_call, None, None
    else:
        o = forward_call()
        do = torch.randn_like(o)
        return lambda: o.backward(do, retain_graph=True), o, do


def bench(hd, nc, mode, prov, dtype, seq_lens, padding_pct=0, warmup=100, rep=250, warmup_autotune=True, return_mode='median'):
    """Benchmark a single configuration for one provider.

    Args:
        hd: Head dimension (64 or 128)
        nc: Sequence length (128-8192)
        mode: 'fwd' (forward) or 'bwd' (forward+backward)
        prov: 'triton-dense', 'triton-padded', 'torch', 'flash_sigmoid', or 'flash_attn'
        dtype: torch.float16 or torch.bfloat16
        seq_lens: Pre-generated sequence lengths (required for fair comparison)
        padding_pct: Percentage of padding (0-100)
        warmup: Warmup time in ms (default: 100)
        rep: Measurement time in ms (default: 250)
        warmup_autotune: Run autotuning warmup before timing
        return_mode: 'min', 'max', 'mean', 'median', or 'all'

    Returns:
        (latency_ms, tflops) or (None, None) on OOM or invalid combination
    """
    # Skip invalid provider/padding combinations
    if padding_pct == 0 and prov == "triton-padded":
        return None, None  # Use triton-dense for padding=0
    if padding_pct > 0 and prov in ["triton-dense", "flash_sigmoid"]:
        return None, None  # These only support padding=0

    try:
        batch, n_heads = 16384 // nc, 2048 // hd

        if padding_pct == 0:
            # Dense case: all sequences same length
            # Providers: torch, triton-dense, flash_sigmoid, flash_attn
            inputs = create_inputs(batch, nc, n_heads, hd, seq_lens, dtype, format='dense')
            q, k, v = inputs['q'], inputs['k'], inputs['v']

            if prov == "triton-dense":
                forward_call = lambda: sigmoid_attention(q, k, v, sigmoid_bias=0.0)
            elif prov == "torch":
                forward_call = lambda: sigmoid_attention_ref(q, k, v, sigmoid_bias=0.0, upcast=False)
            elif prov == "flash_sigmoid":
                forward_call = lambda: flash_sigmoid_func(q, k, v, sigmoid_bias=0.0)
            elif prov == "flash_attn":
                forward_call = lambda: flash_attn_func(q, k, v)

            fn, o, do = _setup_benchmark_fn(forward_call, mode)

        else:
            # Variable-length case: padding > 0
            # Providers: torch, triton-padded, flash_attn
            if prov == "flash_attn":
                # Flash attention requires varlen/packed format
                inputs = create_inputs(batch, nc, n_heads, hd, seq_lens, dtype, format='varlen')
                forward_call = lambda: flash_attn_varlen_func(
                    inputs['q'], inputs['k'], inputs['v'],
                    inputs['cu_seqlens'], inputs['cu_seqlens'],
                    inputs['max_seqlen'], inputs['max_seqlen']
                )
                q, k, v = inputs['q'], inputs['k'], inputs['v']

            else:
                # Padded format for triton-padded and torch (BTHD layout)
                inputs = create_inputs(batch, nc, n_heads, hd, seq_lens, dtype, format='padded')
                q, k, v = inputs['q'], inputs['k'], inputs['v']

                if prov == "triton-padded":
                    forward_call = lambda: sigmoid_attention_padded(
                        q, k, v, seq_lens_k=seq_lens, seq_lens_q=seq_lens, is_causal=False, sigmoid_bias=0.0
                    )
                elif prov == "torch":
                    forward_call = lambda: sigmoid_attention_ref(
                        q, k, v, seq_lens_k=seq_lens, seq_lens_q=seq_lens, sigmoid_bias=0.0, upcast=False
                    )

            fn, o, do = _setup_benchmark_fn(forward_call, mode)

        # Warmup for Triton kernels to ensure autotuning completes
        if warmup_autotune and prov in ["triton-dense", "triton-padded"]:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()

        # Benchmark
        grad_to_none = [q, k, v] if mode == "bwd" else None
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none, return_mode=return_mode)

        # Compute FLOPs
        if padding_pct > 0:
            # Variable-length: sum(L_i^2) not avg(L_i)^2
            total_flops_per_seq = (seq_lens**2).float()
            flops_fwd = 4 * n_heads * total_flops_per_seq.sum().item() * hd
        else:
            # Dense: batch * L^2
            flops_fwd = 4 * batch * n_heads * nc * nc * hd

        flops = flops_fwd * (2.5 if mode == "bwd" else 1.0)

        if return_mode == 'all':
            tflops = [flops * 1e-12 / (m * 1e-3) for m in ms]
        else:
            tflops = flops * 1e-12 / (ms * 1e-3)

        del q, k, v
        if mode == "bwd":
            del o, do
        torch.cuda.empty_cache()

        return ms, tflops

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"    OOM: {e}")
        torch.cuda.empty_cache()
        return None, None


def run_benchmarks(
    dtype_str='bf16',
    modes=None,
    providers=None,
    head_dims=None,
    seq_lens=None,
    padding_pcts=None,
    warmup=100,
    rep=250,
    return_mode='median',
):
    """
    Run comprehensive benchmark suite.

    Args:
        dtype_str: Data type string ('fp16' or 'bf16', default: 'bf16')
        modes: List of modes to benchmark (['fwd'], ['bwd'], or ['fwd', 'bwd'])
        providers: List of providers to benchmark (None = ['triton-dense', 'triton-padded', 'torch'] + flash if available)
        head_dims: List of head dimensions to test (None = [64, 128])
        seq_lens: List of sequence lengths to test (None = [512, 1024, 2048, 4096, 8192, 16384])
        padding_pcts: List of padding percentages to test (None = [0])
        warmup: Warmup time (in ms)
        rep: Repetition time (in ms)
        return_mode: Statistical measure to return ('min', 'max', 'mean', 'median', or 'all')

    Returns:
        DataFrame with benchmark results (invalid combinations are skipped)
    """
    rows = []

    dtype = torch.float16 if dtype_str == 'fp16' else torch.bfloat16

    if head_dims is None:
        head_dims = [64, 128]
    if seq_lens is None:
        seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    if modes is None:
        modes = ["fwd", "bwd"]
    if padding_pcts is None:
        padding_pcts = [0]

    if providers is None:
        providers = ["triton-dense", "triton-padded", "torch"]
        if HAS_FLASH_SIGMOID:
            providers.append("flash_sigmoid")
        if HAS_FLASH_ATTN:
            providers.append("flash_attn")

    # Build configs - group by (hd, nc, pad_pct) to ensure same seq_lens for all providers
    workload_configs = list(product(head_dims, seq_lens, padding_pcts))
    provider_configs = []
    for hd, nc, pad_pct in workload_configs:
        for mode in modes:
            for prov in providers:
                provider_configs.append((hd, nc, pad_pct, mode, prov))

    print(
        f"\nRunning {len(provider_configs)} benchmark configurations:\n"
        f"  Head dimensions: {head_dims}\n"
        f"  Sequence lengths: {seq_lens}\n"
        f"  Modes: {modes}\n"
        f"  Padding percentages: {padding_pcts}\n"
        f"  Hidden dimension: 2048 (fixed)\n"
        f"  Total tokens: 16,384 (fixed)\n"
        f"  Data type: {dtype_str}\n"
        f"  Warmup: {warmup}ms, Repetitions: {rep}\n"
        f"  Return mode: {return_mode}\n"
        f"  Providers: {', '.join(providers)}\n"
    )

    # Group by workload and generate seq_lens once per workload
    from itertools import groupby
    provider_configs.sort(key=lambda x: (x[0], x[1], x[2]))  # Sort by (hd, nc, pad_pct)

    idx = 0
    for (hd, nc, pad_pct), group in groupby(provider_configs, key=lambda x: (x[0], x[1], x[2])):
        batch, n_heads = 16384 // nc, 2048 // hd

        # Generate seq_lens ONCE for this workload configuration
        workload_seq_lens = generate_seq_lens(batch, nc, pad_pct)

        # Run all (mode, provider) combinations with the SAME seq_lens
        for hd, nc, pad_pct, mode, prov in group:
            idx += 1

            ms, tflops = bench(
                hd, nc, mode, prov, dtype, workload_seq_lens, padding_pct=pad_pct,
                warmup=warmup, rep=rep, return_mode=return_mode
            )

            print(
                f"[{idx:3d}/{len(provider_configs)}] D={hd:3d} L={nc:5d} B={batch:2d} H={n_heads:2d} "
                f"PAD={pad_pct:2d}% {mode:3s} {prov:14s}",
                end=" ",
            )

            if ms is not None:
                # Handle different return modes
                if return_mode == 'all':
                    ms_mean, ms_std = np.mean(ms), np.std(ms)
                    tflops_mean = np.mean(tflops)
                    print(f"✓ {ms_mean:6.3f}±{ms_std:5.3f}ms {tflops_mean:6.2f} TFLOPS")
                else:
                    ms_mean, ms_std = ms, None
                    tflops_mean = tflops
                    print(f"✓ {ms:6.3f}ms {tflops:6.2f} TFLOPS")

                # Build row dictionary
                row = {
                    'head_dim': hd,
                    'n_ctx': nc,
                    'batch': batch,
                    'n_heads': n_heads,
                    'padding_pct': pad_pct,
                    'mode': mode,
                    'backend': prov,
                    'dtype': dtype_str,
                    'latency_ms': ms_mean,
                    'tflops': tflops_mean,
                }

                # Add extra columns for 'all' mode
                if return_mode == 'all':
                    row.update({
                        'latency_ms_std': ms_std,
                        'latency_ms_all': ms,
                        'tflops_all': tflops,
                    })

                rows.append(row)
            else:
                # Invalid combination - skip silently
                print("✗ SKIP (invalid combination)")

    df = pd.DataFrame(rows)

    def calc_speedup(row):
        if row['backend'] == 'torch':
            return 1.0

        torch_baseline = df[
            (df['backend'] == 'torch')
            & (df['head_dim'] == row['head_dim'])
            & (df['n_ctx'] == row['n_ctx'])
            & (df['padding_pct'] == row['padding_pct'])
            & (df['mode'] == row['mode'])
            & (df['dtype'] == row['dtype'])
        ]

        if torch_baseline.empty:
            return None

        return torch_baseline['latency_ms'].values[0] / row['latency_ms']

    df['speedup'] = df.apply(calc_speedup, axis=1)

    if return_mode == 'all':
        column_order = [
            'head_dim',
            'n_ctx',
            'batch',
            'n_heads',
            'padding_pct',
            'mode',
            'backend',
            'dtype',
            'latency_ms',
            'latency_ms_std',
            'tflops',
            'speedup',
            'latency_ms_all',
            'tflops_all',
        ]
    else:
        column_order = [
            'head_dim',
            'n_ctx',
            'batch',
            'n_heads',
            'padding_pct',
            'mode',
            'backend',
            'dtype',
            'latency_ms',
            'tflops',
            'speedup',
        ]

    return df[column_order]


def print_summary(df, return_mode='median'):
    """Print concise benchmark summary."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY - {len(df)} configurations")
    print(f"{'='*80}\n")

    for backend in sorted(df['backend'].unique()):
        backend_data = df[df['backend'] == backend]
        avg_tflops = backend_data['tflops'].mean()
        avg_speedup = backend_data[backend_data['speedup'].notna()]['speedup'].mean()

        if backend == 'torch':
            print(f"{backend:20s}: {avg_tflops:6.1f} TFLOPS avg")
        else:
            print(f"{backend:20s}: {avg_tflops:6.1f} TFLOPS avg, {avg_speedup:5.2f}x speedup")

    print(f"\n{'='*80}\n")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark sigmoid attention performance', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory for results and plots')
    parser.add_argument('--dtype', type=str, choices=['fp16', 'bf16'], default='bf16', help='Data type')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fwd', 'bwd', 'both'],
        default='both',
        help='Benchmark mode: fwd (forward only), bwd (backward only), or both',
    )
    parser.add_argument(
        '--providers',
        type=str,
        nargs='+',
        choices=['triton-dense', 'triton-padded', 'torch', 'flash_sigmoid', 'flash_attn'],
        help='Specific providers to benchmark (default: triton-dense, triton-padded, torch, and any available flash implementations). '
        'Invalid combinations are automatically skipped (e.g., triton-dense with padding>0, triton-padded with padding=0).',
    )
    parser.add_argument('--head-dims', type=int, nargs='+', default=[64, 128], help='Head dimensions to test')
    parser.add_argument(
        '--seq-lens', type=int, nargs='+', default=[512, 1024, 2048, 4096, 8192, 16384], help='Sequence lengths to test'
    )
    parser.add_argument(
        '--padding-pcts',
        type=int,
        nargs='+',
        default=[0],
        help='Padding percentages to test (e.g., 0 25 50)',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=100,
        help='Warmup time in milliseconds',
    )
    parser.add_argument(
        '--rep',
        type=int,
        default=250,
        help='Repetition time in milliseconds',
    )
    parser.add_argument(
        '--return-mode',
        type=str,
        choices=['min', 'max', 'mean', 'median', 'all'],
        default='median',
        help='Statistical measure to return from benchmarks',
    )
    args = parser.parse_args()

    if args.mode == 'both':
        modes = ['fwd', 'bwd']
    else:
        modes = [args.mode]

    print(f"\n{'='*80}")
    print("SYSTEM INFORMATION")
    print(f"{'='*80}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"Data Type: {args.dtype}")
    print(f"Mode: {args.mode}")
    print(f"Warmup: {args.warmup}ms")
    print(f"Repetitions: {args.rep}ms")
    print(f"Return Mode: {args.return_mode}")
    print(f"{'='*80}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = f"{args.output_dir}/results_{args.mode}_{args.return_mode}.csv"

    df = run_benchmarks(
        dtype_str=args.dtype,
        modes=modes,
        providers=args.providers,
        head_dims=args.head_dims,
        seq_lens=args.seq_lens,
        padding_pcts=args.padding_pcts,
        warmup=args.warmup,
        rep=args.rep,
        return_mode=args.return_mode,
    )

    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}\n")

    print_summary(df, return_mode=args.return_mode)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results directory: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
