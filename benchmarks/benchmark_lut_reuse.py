from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median

import torch

from rns_llm.backends import CudaRNSBackend
from rns_llm.reference import choose_moduli_for_dot, moduli_cost_model


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def benchmark_single(fn, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return {"p50_ms": median(values), "p95_ms": percentile(values, 0.95)}


def benchmark_concurrent(fns, streams, warmup: int, iterations: int) -> dict[str, float]:
    def launch_once():
        for fn, stream in zip(fns, streams):
            with torch.cuda.stream(stream):
                fn()
        for stream in streams:
            stream.synchronize()

    for _ in range(warmup):
        launch_once()
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        launch_once()
        values.append((time.perf_counter() - start) * 1000.0)
    return {
        "batch_p50_ms": median(values),
        "batch_p95_ms": percentile(values, 0.95),
        "requests_per_second_p50": len(fns) / (median(values) / 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--source-bits", type=int, choices=[8, 12], default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    max_abs = (1 << (args.source_bits - 1)) - 1
    dtype = torch.int8 if args.source_bits == 8 else torch.int16
    device = torch.device("cuda")
    moduli = choose_moduli_for_dot(
        args.k, max_abs, max_abs, strategy="dense_coprime"
    )
    backend = CudaRNSBackend()
    torch.manual_seed(11)
    weight = torch.randint(-max_abs, max_abs + 1, (args.k, args.n), dtype=dtype, device=device)
    prepared = backend.prepare_weight(weight, moduli)

    activations = [
        torch.randint(-max_abs, max_abs + 1, (args.m, args.k), dtype=dtype, device=device)
        for _ in range(args.concurrency)
    ]
    workspaces = [
        backend.create_workspace(device=device, channels=len(moduli), m=args.m, n=args.n)
        for _ in range(args.concurrency)
    ]
    streams = [torch.cuda.Stream() for _ in range(args.concurrency)]

    single = {}
    concurrent = {}
    for lut_channels in (0, 1, 2):
        single[str(lut_channels)] = benchmark_single(
            lambda lc=lut_channels: backend.matmul_prepared_fused(
                activations[0], prepared, lut_channels=lc, workspace=workspaces[0]
            ),
            args.warmup,
            args.iterations,
        )
        fns = [
            (lambda index=i, lc=lut_channels: backend.matmul_prepared_fused(
                activations[index], prepared, lut_channels=lc, workspace=workspaces[index]
            ))
            for i in range(args.concurrency)
        ]
        concurrent[str(lut_channels)] = benchmark_concurrent(
            fns, streams, args.warmup, args.iterations
        )

    memory = moduli_cost_model(
        moduli,
        m=args.m,
        k=args.k,
        n=args.n,
        source_element_bytes=torch.empty((), dtype=dtype).element_size(),
        lut_channels=2,
    )
    payload = {
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "source_bits": args.source_bits,
        "moduli": list(moduli),
        "channels": len(moduli),
        "concurrency": args.concurrency,
        "single_request": single,
        "concurrent_requests": concurrent,
        "table_memory": {
            "compact_two_tables_bytes": memory["compact_lut_bytes"],
            "full_two_multiplication_tables_bytes": memory["full_mul_lut_bytes"],
            "saving_fraction": memory["compact_vs_full_lut_saving"],
        },
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
