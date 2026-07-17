from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

from rns_llm.backends import CudaRNSBackend
from rns_llm.reference import choose_moduli_for_dot


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {"p50_ms": median(samples), "p95_ms": percentile(samples, 0.95)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--std", type=float, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    moduli = choose_moduli_for_dot(args.k, 127, 127, strategy="dense_coprime")
    backend = CudaRNSBackend()
    # Transformer-like quantized weights are not uniformly at ±127.
    weight = torch.clamp(
        torch.round(torch.randn(args.k, args.n, device=device) * 28), -127, 127
    ).to(torch.int8)
    full = backend.prepare_weight(weight, moduli)
    adaptive = backend.prepare_weight_adaptive(weight, moduli, min_channels=3)
    full_workspace = backend.create_workspace(
        device=device, channels=len(moduli), m=args.m, n=args.n
    )
    adaptive_workspaces = {
        channels: backend.create_workspace(
            device=device, channels=channels, m=args.m, n=args.n
        )
        for channels in adaptive.variants
    }

    results = []
    for std in args.std:
        activation = torch.clamp(
            torch.round(torch.randn(args.m, args.k, device=device) * std), -127, 127
        ).to(torch.int8)
        full_output = backend.matmul_prepared_fused(
            activation, full, workspace=full_workspace, lut_channels=2
        ).clone()
        adaptive_output, metadata = backend.matmul_prepared_adaptive_fused(
            activation,
            adaptive,
            workspace_by_channels=adaptive_workspaces,
            lut_channels=2,
            return_metadata=True,
        )
        adaptive_output = adaptive_output.clone()
        torch.testing.assert_close(adaptive_output, full_output, rtol=0, atol=0)
        expected = (
            activation.cpu().numpy().astype(np.int64)
            @ weight.cpu().numpy().astype(np.int64)
        )
        max_error = int(np.max(np.abs(adaptive_output.cpu().numpy() - expected)))
        if max_error != 0:
            raise AssertionError(f"adaptive result error: {max_error}")

        full_timing = measure(
            lambda: backend.matmul_prepared_fused(
                activation, full, workspace=full_workspace, lut_channels=2
            ),
            args.warmup,
            args.iterations,
        )
        adaptive_timing = measure(
            lambda: backend.matmul_prepared_adaptive_fused(
                activation,
                adaptive,
                workspace_by_channels=adaptive_workspaces,
                lut_channels=2,
            ),
            args.warmup,
            args.iterations,
        )
        results.append(
            {
                "activation_std": std,
                "selected_channels": metadata["channels"],
                "safe_bound": metadata["bound"],
                "capacity": metadata["capacity"],
                "correctness": {"passed": True, "max_absolute_error": max_error},
                "full": full_timing,
                "adaptive_including_selection_sync": adaptive_timing,
                "speedup": full_timing["p50_ms"] / adaptive_timing["p50_ms"],
            }
        )

    payload = {
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "full_moduli": list(moduli),
        "selection_bound": "max_row_L1(A) * max_abs(B)",
        "selection_sync_included": True,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
