from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import torch
from torch import nn

from rns_llm.backends import CudaRNSBackend
from rns_llm.layers import RNSLinear, RNSQKVProjection


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def cuda_measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"p50_ms": median(samples), "p95_ms": percentile(samples, 0.95)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 256])
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--quant-bits", type=int, choices=[8, 12], default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    backend = CudaRNSBackend()
    torch.manual_seed(17)
    originals = tuple(
        nn.Linear(args.hidden, args.hidden, bias=True, device=device, dtype=torch.float16).eval()
        for _ in range(3)
    )
    separate = tuple(
        RNSLinear.from_linear(
            layer,
            backend=backend,
            mode="rns",
            quant_bits=args.quant_bits,
            fused=True,
            lut_channels=2,
        ).eval()
        for layer in originals
    )
    fused = RNSQKVProjection.from_linears(
        *originals,
        backend=backend,
        mode="rns",
        quant_bits=args.quant_bits,
        fused=True,
        lut_channels=2,
    ).eval()

    results = []
    for tokens in args.tokens:
        x = torch.randn(1, tokens, args.hidden, device=device, dtype=torch.float16)
        separate_outputs = tuple(layer(x) for layer in separate)
        fused_outputs = fused(x)
        max_error = 0.0
        for left, right in zip(fused_outputs, separate_outputs):
            # QKV fusion must not change RNS arithmetic at all.  This is a
            # stricter check than comparison with floating-point Linear.
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            max_error = max(max_error, float((left - right).abs().max().item()))

        backend.reset_stats()
        separate_timing = cuda_measure(
            lambda: tuple(layer(x) for layer in separate), args.warmup, args.iterations
        )
        separate_stats = backend.stats_snapshot()
        backend.reset_stats()
        fused_timing = cuda_measure(lambda: fused(x), args.warmup, args.iterations)
        fused_stats = backend.stats_snapshot()

        results.append(
            {
                "tokens": tokens,
                "shape": [1, tokens, args.hidden],
                "correctness": {
                    "fused_equals_three_rns_linears": True,
                    "max_absolute_error": max_error,
                },
                "three_separate_rns": separate_timing,
                "one_fused_qkv_rns": fused_timing,
                "speedup": separate_timing["p50_ms"] / fused_timing["p50_ms"],
                "backend_stats_separate": separate_stats,
                "backend_stats_fused": fused_stats,
            }
        )

    payload = {
        "hidden": args.hidden,
        "quant_bits": args.quant_bits,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
