from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from rns_llm.backends import CudaRNSBackend
from rns_llm.layers import FastRNSLinearV07, RNSLinear
from rns_llm.v07_backend import V07FastPath


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "p50_ms": median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def benchmark_randomized(
    callables: dict[str, Callable[[], object]],
    *,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    names = list(callables)
    rng = random.Random(seed)

    with torch.no_grad():
        for _ in range(warmup):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                callables[name]()
        torch.cuda.synchronize()

        samples: dict[str, list[float]] = {name: [] for name in names}
        for _ in range(iterations):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                callables[name]()
                end.record()
                end.synchronize()
                samples[name].append(float(start.elapsed_time(end)))

    return {name: summarize(values) for name, values in samples.items()}


def try_capture_graph(fn: Callable[[], torch.Tensor]):
    try:
        with torch.no_grad():
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = fn()

        def replay():
            graph.replay()
            return static_output

        return replay, None
    except Exception as exc:
        return None, repr(exc)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--include-v06", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    backend = CudaRNSBackend()
    fast_path = V07FastPath(backend)

    with torch.no_grad():
        source = nn.Linear(
            args.k,
            args.n,
            bias=True,
            dtype=torch.float16,
            device=device,
        ).eval()
        source.weight.normal_(mean=0.0, std=0.025)
        source.bias.normal_(mean=0.0, std=0.01)

        v07 = FastRNSLinearV07.from_linear(
            source,
            backend=backend,
            mode="rns",
            quant_bits=8,
            fused=True,
            lut_channels=2,
            moduli_strategy="dense_coprime",
            use_v07_epilogue=True,
            fuse_quantize_encode=True,
        ).eval()
        v07.prepare_weight()

        v06 = None
        if args.include_v06:
            v06 = RNSLinear.from_linear(
                source,
                backend=backend,
                mode="rns",
                quant_bits=8,
                fused=True,
                lut_channels=2,
                moduli_strategy="dense_coprime",
            ).eval()
            v06.prepare_weight()

        weight_scale = v07._weight_scale
        weight_q = v07._quantize(
            v07.weight.float(),
            weight_scale.unsqueeze(1),
        ).transpose(0, 1).contiguous()
        bias_float = source.bias.detach().float().contiguous()

    results = []
    for m in args.m:
        inputs = torch.randn(
            m,
            args.k,
            dtype=torch.float16,
            device=device,
        )
        native_workspace = fast_path.create_native_workspace(
            device=device,
            m=m,
            k=args.k,
            n=args.n,
        )

        def fp16_projection():
            return F.linear(inputs, source.weight, source.bias)

        def native_projection():
            scale = v07._activation_scale(inputs)
            return fast_path.native_fp16_input_dequant_fp16(
                inputs,
                weight_q,
                activation_scale=scale,
                weight_scale=weight_scale,
                bias=bias_float,
                quant_max=v07.quant_max,
                workspace=native_workspace,
            )

        methods: dict[str, Callable[[], object]] = {
            "fp16_linear": fp16_projection,
            "native_int8_direct_fp16": native_projection,
            "rns_v07_direct_fp16_4ch": lambda: v07(inputs),
        }
        if v06 is not None:
            methods["rns_v06_int64_epilogue"] = lambda: v06(inputs)

        # Allocate every lazy workspace outside timed regions.
        with torch.no_grad():
            for fn in methods.values():
                fn()
            torch.cuda.synchronize()

        timings = benchmark_randomized(
            methods,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed + m,
        )

        graph_timings = {}
        graph_errors = {}
        if args.graphs:
            graph_methods = {}
            for name, fn in methods.items():
                graph_fn, error = try_capture_graph(fn)
                if graph_fn is None:
                    graph_errors[name] = error
                else:
                    graph_methods[f"graph_{name}"] = graph_fn
            if graph_methods:
                graph_timings = benchmark_randomized(
                    graph_methods,
                    warmup=max(5, args.warmup // 2),
                    iterations=args.iterations,
                    seed=args.seed + 1000 + m,
                )

        fp16_t = timings["fp16_linear"]["p50_ms"]
        native_t = timings["native_int8_direct_fp16"]["p50_ms"]
        v07_t = timings["rns_v07_direct_fp16_4ch"]["p50_ms"]
        derived = {
            "v07_gap_to_fp16": v07_t / fp16_t,
            "v07_gap_to_native_int8": v07_t / native_t,
        }
        if v06 is not None:
            derived["v07_speedup_vs_v06"] = (
                timings["rns_v06_int64_epilogue"]["p50_ms"] / v07_t
            )

        channels = len(v07.moduli)
        results.append(
            {
                "shape": {"m": m, "k": args.k, "n": args.n},
                "timings": timings,
                "graph_timings": graph_timings,
                "graph_capture_errors": graph_errors,
                "memory": {
                    "fp16_weight_bytes": tensor_bytes(source.weight),
                    "v07_residue_weight_bytes": tensor_bytes(
                        v07._prepared_weight.residues
                    ),
                    "v07_channels": channels,
                    "v07_runtime_workspace_bytes": (
                        channels * m * args.k
                        + channels * m * args.n * 4
                        + m * args.n * 2
                    ),
                    "native_runtime_workspace_bytes": (
                        m * args.k + m * args.n * 4 + m * args.n * 2
                    ),
                },
                "derived": derived,
            }
        )

    payload = {
        "version": "0.7.1",
        "scope": "performance_and_nsight_baseline_without_correctness_tests",
        "correctness_testing": {
            "enabled": False,
            "note": "Numerical tests, smoke tests and PPL gates were removed by request.",
        },
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "protocol": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "randomized_method_order": True,
            "cuda_events": True,
            "graphs": args.graphs,
            "include_v06": args.include_v06,
        },
        "backend_stats": backend.stats_snapshot(),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
