from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import torch

from rns_llm.backends import CudaRNSBackend
from rns_llm.reference import choose_moduli_for_dot, moduli_cost_model


def time_ms(fn, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    values.sort()
    return {
        "p50_ms": median(values),
        "p95_ms": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--source-bits", type=int, choices=[8, 12, 16], default=8)
    parser.add_argument("--max-abs", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    max_abs = args.max_abs or ((1 << (args.source_bits - 1)) - 1)
    dtype = torch.int8 if args.source_bits <= 8 else torch.int16
    device = torch.device("cuda")
    torch.manual_seed(7)
    a = torch.randint(-max_abs, max_abs + 1, (args.m, args.k), dtype=dtype, device=device)
    b = torch.randint(-max_abs, max_abs + 1, (args.k, args.n), dtype=dtype, device=device)
    expected = a[:16].cpu().numpy().astype(np.int64) @ b[:, :16].cpu().numpy().astype(np.int64)

    backend = CudaRNSBackend()
    results = []
    for strategy in ("small_primes", "large_primes", "dense_coprime"):
        moduli = choose_moduli_for_dot(
            args.k, max_abs, max_abs, strategy=strategy
        )
        prepared = backend.prepare_weight(b, moduli)
        if prepared.kernel != "cublas":
            results.append({
                "strategy": strategy,
                "moduli": list(moduli),
                "channels": len(moduli),
                "skipped": "not cuBLAS compatible",
            })
            continue
        workspace = backend.create_workspace(
            device=device, channels=len(moduli), m=args.m, n=args.n
        )
        output = backend.matmul_prepared_fused(
            a, prepared, lut_channels=2, workspace=workspace
        )
        torch.cuda.synchronize()
        actual = output[:16, :16].cpu().numpy()
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"correctness failed for {strategy}")

        timing = time_ms(
            lambda: backend.matmul_prepared_fused(
                a, prepared, lut_channels=2, workspace=workspace
            ),
            args.warmup,
            args.iterations,
        )
        cost = moduli_cost_model(
            moduli,
            m=args.m,
            k=args.k,
            n=args.n,
            source_element_bytes=a.element_size(),
            lut_channels=2,
        )
        results.append({
            "strategy": strategy,
            "moduli": list(moduli),
            "channels": len(moduli),
            "range_product": math.prod(moduli),
            "timing": timing,
            "memory": cost,
        })

    payload = {
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "source_bits": args.source_bits,
        "max_abs": max_abs,
        "required_signed_range": 2 * args.k * max_abs * max_abs + 1,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
