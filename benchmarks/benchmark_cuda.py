from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
import torch

from rns_llm.backends import CudaRNSBackend
from rns_llm.reference import choose_moduli_for_dot, moduli_cost_model


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_suite(
    callables: dict[str, Callable[[], object]],
    *,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    names = list(callables)
    rng = random.Random(seed)

    for _ in range(warmup):
        order = list(names)
        rng.shuffle(order)
        for name in order:
            callables[name]()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in names}
    wall_samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(iterations):
        order = list(names)
        rng.shuffle(order)
        for name in order:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start_event.record()
            callables[name]()
            end_event.record()
            end_event.synchronize()
            wall_elapsed = (time.perf_counter() - wall_start) * 1000.0
            samples[name].append(float(start_event.elapsed_time(end_event)))
            wall_samples[name].append(wall_elapsed)

    result: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        walls = wall_samples[name]
        result[name] = {
            "min_ms": min(values),
            "p50_ms": median(values),
            "p95_ms": percentile(values, 0.95),
            "mean_ms": sum(values) / len(values),
            "max_ms": max(values),
            "wall_p50_ms": median(walls),
            "wall_p95_ms": percentile(walls, 0.95),
            "samples": len(values),
        }
    return result


def dtype_for_bits(bits: int) -> torch.dtype:
    return torch.int8 if bits <= 8 else torch.int16


def exact_reference(a: torch.Tensor, b: torch.Tensor, rows: int, cols: int) -> np.ndarray:
    return (
        a[:rows].cpu().numpy().astype(np.int64)
        @ b[:, :cols].cpu().numpy().astype(np.int64)
    )


def numerical_baseline_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    expected: np.ndarray,
    rows: int,
    cols: int,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, dtype in (("fp16", torch.float16), ("fp32", torch.float32)):
        actual = torch.mm(a[:rows].to(dtype), b[:, :cols].to(dtype))
        actual_np = actual.float().cpu().numpy().astype(np.float64)
        finite = np.isfinite(actual_np)
        difference = actual_np - expected.astype(np.float64)
        abs_error = np.abs(difference)
        result[name] = {
            "finite_fraction": float(finite.mean()),
            "exact_fraction": float(np.mean(finite & (actual_np == expected))),
            "max_absolute_error_finite": (
                float(np.max(abs_error[finite])) if finite.any() else float("inf")
            ),
            "mean_absolute_error_finite": (
                float(np.mean(abs_error[finite])) if finite.any() else float("inf")
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--source-bits", type=int, choices=[8, 12, 16], default=8)
    parser.add_argument("--max-abs", type=int)
    parser.add_argument("--moduli", type=int, nargs="+")
    parser.add_argument(
        "--moduli-strategy",
        choices=["large_primes", "dense_coprime", "small_primes"],
        default="dense_coprime",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--correctness-rows", type=int, default=32)
    parser.add_argument("--correctness-cols", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    torch.manual_seed(42)
    device = torch.device("cuda")
    max_abs = args.max_abs or ((1 << (args.source_bits - 1)) - 1)
    source_dtype = dtype_for_bits(args.source_bits)
    moduli = tuple(args.moduli) if args.moduli else choose_moduli_for_dot(
        args.k,
        max_abs,
        max_abs,
        strategy=args.moduli_strategy,
    )

    a = torch.randint(
        -max_abs,
        max_abs + 1,
        (args.m, args.k),
        dtype=source_dtype,
        device=device,
    )
    b = torch.randint(
        -max_abs,
        max_abs + 1,
        (args.k, args.n),
        dtype=source_dtype,
        device=device,
    )

    backend = CudaRNSBackend()
    a_centered = backend.encode_centered(a, moduli)
    b_centered = backend.encode_centered(b, moduli)
    prepared = backend.prepare_weight(b, moduli)
    if prepared.kernel != "cublas":
        raise SystemExit("v0.5 fused benchmark currently requires cuBLAS-compatible K/N")

    old_residues = backend.matmul_centered_residues(
        a_centered, b_centered, moduli, kernel="cublas"
    )
    workspace = backend.create_workspace(
        device=device,
        channels=len(moduli),
        m=args.m,
        n=args.n,
    )

    rows = min(args.correctness_rows, args.m)
    cols = min(args.correctness_cols, args.n)
    expected = exact_reference(a, b, rows, cols)
    old_decoded = backend.decode(old_residues, moduli)
    garner_decoded = backend.decode_garner(old_residues, moduli)
    fused_outputs = {
        str(lut_channels): backend.matmul_prepared_fused(
            a,
            prepared,
            lut_channels=lut_channels,
            workspace=workspace,
        ).clone()
        for lut_channels in (0, 1, 2)
    }
    torch.cuda.synchronize()

    correctness = {
        "old_crt": bool(np.array_equal(old_decoded[:rows, :cols].cpu().numpy(), expected)),
        "garner": bool(np.array_equal(garner_decoded[:rows, :cols].cpu().numpy(), expected)),
        "fused_lut0": bool(np.array_equal(fused_outputs["0"][:rows, :cols].cpu().numpy(), expected)),
        "fused_lut1": bool(np.array_equal(fused_outputs["1"][:rows, :cols].cpu().numpy(), expected)),
        "fused_lut2": bool(np.array_equal(fused_outputs["2"][:rows, :cols].cpu().numpy(), expected)),
        "rows": rows,
        "cols": cols,
        "max_absolute_error": int(
            np.max(np.abs(fused_outputs["2"][:rows, :cols].cpu().numpy() - expected))
        ),
    }
    if not all(value for key, value in correctness.items() if key not in {"rows", "cols", "max_absolute_error"}):
        raise RuntimeError(f"correctness check failed: {correctness}")

    callables: dict[str, Callable[[], object]] = {
        "gemm_cublas_residue_output": lambda: backend.matmul_centered_residues(
            a_centered, b_centered, moduli, kernel="cublas"
        ),
        "decode_crt_pytorch": lambda: backend.decode(old_residues, moduli),
        "decode_garner_cuda": lambda: backend.decode_garner(old_residues, moduli),
        "encode_activation_centered": lambda: backend.encode_centered(a, moduli),
        "end_to_end_old_cached_weight": lambda: backend.matmul_prepared(
            a, prepared, kernel="cublas"
        ),
        "fused_encoded_barrett_workspace": lambda: backend.matmul_centered_fused(
            a_centered,
            b_centered,
            moduli,
            lut_channels=0,
            workspace=workspace,
        ),
        "end_to_end_fused_barrett_workspace": lambda: backend.matmul_prepared_fused(
            a, prepared, lut_channels=0, workspace=workspace
        ),
        "end_to_end_fused_lut1_workspace": lambda: backend.matmul_prepared_fused(
            a, prepared, lut_channels=1, workspace=workspace
        ),
        "end_to_end_fused_lut2_workspace": lambda: backend.matmul_prepared_fused(
            a, prepared, lut_channels=2, workspace=workspace
        ),
        "end_to_end_fused_lut2_allocating": lambda: backend.matmul_prepared_fused(
            a, prepared, lut_channels=2, workspace=None
        ),
        "torch_fp16_gemm": lambda: torch.mm(a.to(torch.float16), b.to(torch.float16)),
        "torch_fp32_gemm": lambda: torch.mm(a.to(torch.float32), b.to(torch.float32)),
    }

    timings = benchmark_suite(
        callables,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=42,
    )

    channel_operations = 2 * len(moduli) * args.m * args.k * args.n
    useful_operations = 2 * args.m * args.k * args.n
    for timing in timings.values():
        seconds = timing["p50_ms"] / 1000.0
        timing["rns_channel_tops"] = channel_operations / seconds / 1e12
        timing["useful_wide_tops"] = useful_operations / seconds / 1e12

    numerical = numerical_baseline_metrics(a, b, expected, rows, cols)
    memory = moduli_cost_model(
        moduli,
        m=args.m,
        k=args.k,
        n=args.n,
        source_element_bytes=a.element_size(),
        lut_channels=2,
    )
    memory["workspace_bytes"] = workspace.accumulators.numel() * 4 + workspace.output.numel() * 8

    results = {
        "version": "0.5.0",
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "source": {
            "bits": args.source_bits,
            "dtype": str(source_dtype),
            "max_abs": max_abs,
        },
        "moduli_strategy": args.moduli_strategy,
        "moduli": list(moduli),
        "channels": len(moduli),
        "modulus_product": math.prod(moduli),
        "required_signed_range": 2 * args.k * max_abs * max_abs + 1,
        "correctness": correctness,
        "floating_baseline_accuracy": numerical,
        "memory_model_bytes": memory,
        "benchmark_protocol": {
            "warmup_rounds": args.warmup,
            "iterations": args.iterations,
            "method_order": "randomized_each_round",
            "gpu_timing": "CUDA events",
            "wall_timing": "perf_counter with event synchronization",
        },
        "timings": timings,
    }

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
