from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from rns_llm.architecture_v013 import select_plan
from rns_llm.hybrid_v010 import choose_moduli
from rns_llm.unified_v014 import (
    FullRNSLinearV014,
    HybridLinearV014,
    NativeInt8LinearV014,
)


def parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.lower().split("x"))
    if len(values) != 3 or min(values) <= 0:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    m, k, n = values
    if k % 4 or n % 4:
        raise argparse.ArgumentTypeError("K and N must be multiples of four")
    return m, k, n


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def measure(fn: Callable[[], object], warmup: int, iterations: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": len(samples),
    }


def concurrent_measure(
    module: nn.Module,
    inputs: list[torch.Tensor],
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    streams = [torch.cuda.Stream() for _ in inputs]
    current = torch.cuda.current_stream()

    def once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(current)
        for stream in streams:
            stream.wait_event(start)
        for stream, tensor in zip(streams, inputs):
            with torch.cuda.stream(stream):
                module(tensor)
        for stream in streams:
            current.wait_stream(stream)
        end.record(current)
        end.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(warmup):
        once()
    samples = [once() for _ in range(iterations)]
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "samples": len(samples),
    }


def accuracy(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    out = output.float()
    ref = reference.float()
    difference = out - ref
    return {
        "relative_l2": float(
            (torch.linalg.vector_norm(difference) /
             torch.clamp(torch.linalg.vector_norm(ref), min=1e-30)).item()
        ),
        "cosine": float(
            (torch.sum(out * ref) /
             torch.clamp(
                 torch.linalg.vector_norm(out) * torch.linalg.vector_norm(ref),
                 min=1e-30,
             )).item()
        ),
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
    }


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def resolve_lut(label: str, channels: int) -> int:
    return {
        "none": 0,
        "one": min(1, channels),
        "two": min(2, channels),
        "all": channels,
    }[label]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def module_component_timings(
    module: nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, float | None]:
    preprocess: Callable[[], object] | None = None
    core: Callable[[], object] | None = None

    if isinstance(module, NativeInt8LinearV014):
        flat = sample.reshape(-1, module.in_features).float().contiguous()
        runner = module._runner(int(flat.shape[0]))
        preprocess = lambda: runner.encode(flat)
        runner.encode(flat)
        core = runner.core
    elif isinstance(module, FullRNSLinearV014) and module.v07 is None:
        flat = sample.reshape(-1, module.in_features).float().contiguous()
        runner = module._runner(int(flat.shape[0]))
        preprocess = lambda: runner.encode(flat)
        runner.encode(flat)
        core = runner.core
    elif isinstance(module, HybridLinearV014):
        flat = sample.reshape(-1, module.in_features).float().contiguous()
        runner = module._runner(int(flat.shape[0]))
        if module.correction == "rns":
            preprocess = lambda: runner.preprocess_hybrid_rns(flat)
            runner.preprocess_hybrid_rns(flat)
            core = (
                runner.hybrid_rns_serial_core
                if module.execution == "serial"
                else runner.hybrid_rns_parallel_core
            )
        else:
            preprocess = lambda: runner.preprocess_hybrid_fp16(flat)
            runner.preprocess_hybrid_fp16(flat)
            core = (
                runner.hybrid_fp16_serial_core
                if module.execution == "serial"
                else runner.hybrid_fp16_parallel_core
            )

    result: dict[str, float | None] = {
        "preprocess_p50_ms": None,
        "core_p50_ms": None,
    }
    if preprocess is not None:
        result["preprocess_p50_ms"] = float(
            measure(preprocess, warmup, iterations)["p50_ms"]
        )
    if core is not None:
        result["core_p50_ms"] = float(measure(core, warmup, iterations)["p50_ms"])
    return result


def build_specs(args, k: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"variant": "native_int8", "architecture": "native_int8"},
        {
            "variant": "hybrid_fp16_serial",
            "architecture": "hybrid_fp16",
            "execution": "serial",
        },
    ]
    if args.include_parallel:
        specs.append({
            "variant": "hybrid_fp16_parallel",
            "architecture": "hybrid_fp16",
            "execution": "parallel",
        })

    for moduli_policy in args.moduli_policies:
        for bits in args.full_bits:
            channels = select_plan(k, bits, moduli_policy).channels
            for policy in args.lut_policies:
                specs.append({
                    "variant": f"full_rns_{moduli_policy}_int{bits}_{policy}",
                    "architecture": "full_rns",
                    "bits": bits,
                    "channels": channels,
                    "moduli_policy": moduli_policy,
                    "lut_policy": policy,
                    "lut_channels": resolve_lut(policy, channels),
                    "q8_backend": "v013",
                })
    if 8 in args.full_bits and args.include_v07_q8:
        channels = select_plan(k, 8, "dense_coprime").channels
        specs.append({
            "variant": "full_rns_v07_int8_lut2",
            "architecture": "full_rns",
            "bits": 8,
            "channels": channels,
            "moduli_policy": "legacy_v07_dense_coprime",
            "lut_policy": "two",
            "lut_channels": min(2, channels),
            "q8_backend": "v07",
        })

    p_padded = ((args.protected + 3) // 4) * 4
    for bits in args.hybrid_bits:
        channels = len(choose_moduli(bits, p_padded))
        for policy in args.lut_policies:
            executions = ["serial", "parallel"] if args.include_parallel else ["serial"]
            for execution in executions:
                specs.append({
                    "variant": f"hybrid_rns_q{bits}_{execution}_{policy}",
                    "architecture": "hybrid_rns",
                    "bits": bits,
                    "channels": channels,
                    "moduli_policy": "dense_coprime",
                    "execution": execution,
                    "lut_policy": policy,
                    "lut_channels": resolve_lut(policy, channels),
                })
    return specs


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["shape"]), str(row["variant"])), []).append(row)
    result: list[dict[str, object]] = []
    numeric_keys = [
        "e2e_p50_ms", "e2e_p95_ms", "preprocess_p50_ms", "core_p50_ms",
        "relative_l2", "cosine", "max_abs", "mean_abs", "vs_fp16",
        "weight_vs_fp16", "concurrency_p50_ms", "concurrency_throughput_speedup",
        "peak_memory_allocated_bytes",
    ]
    for (shape, variant), group in grouped.items():
        base = {key: value for key, value in group[0].items() if key not in numeric_keys}
        base["shape"] = shape
        base["variant"] = variant
        base["repeats"] = len(group)
        for key in numeric_keys:
            values = [float(row[key]) for row in group if row.get(key) is not None]
            base[key] = statistics.median(values) if values else None
        result.append(base)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified matrix benchmark for FP16, native INT8, full-RNS and hybrid RNS"
    )
    parser.add_argument("--shape", action="append", type=parse_shape)
    parser.add_argument("--full-bits", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--hybrid-bits", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument(
        "--lut-policies", nargs="+", choices=["none", "one", "two", "all"],
        default=["none", "one", "two", "all"],
    )
    parser.add_argument("--protected", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--moduli-policies", nargs="+", choices=["dense_coprime", "large_primes", "school_small"], default=["dense_coprime", "large_primes", "school_small"])
    parser.add_argument("--include-v07-q8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v0.14.2/matrix"))
    args = parser.parse_args()

    shapes = args.shape or [
        (16, 2560, 2560),
        (128, 2560, 2560),
        (16, 2560, 10240),
        (128, 2560, 10240),
    ]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")

    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for repeat in range(args.repeats):
        torch.manual_seed(args.seed + repeat)
        for m, k, n in shapes:
            weight = (
                torch.randn(n, k, device=device, dtype=torch.float32)
                / math.sqrt(k)
            ).contiguous()
            bias = (
                torch.randn(n, device=device, dtype=torch.float32) * 0.01
            ).contiguous()
            layer = nn.Linear(
                k, n, bias=True, device=device, dtype=torch.float16
            ).eval()
            with torch.no_grad():
                layer.weight.copy_(weight.half())
                layer.bias.copy_(bias.half())
            inputs = [
                torch.randn(m, k, device=device, dtype=torch.float16)
                for _ in range(args.concurrency)
            ]
            reference = F.linear(inputs[0].float(), weight, bias)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            fp16_output = layer(inputs[0])
            fp16_timing = measure(
                lambda: layer(inputs[0]), args.warmup, args.iterations
            )
            fp16_concurrency = concurrent_measure(
                layer, inputs, args.warmup, args.iterations
            )
            fp16_weight_bytes = tensor_bytes(layer.weight) + tensor_bytes(layer.bias)
            rows.append({
                "repeat": repeat,
                "shape": f"{m}x{k}x{n}",
                "variant": "fp16",
                "architecture": "fp16",
                "logical_bits": 16,
                "execution": "native",
                "lut_policy": "n/a",
                "lut_channels": 0,
                "channels": 0,
                "preprocess_p50_ms": 0.0,
                "core_p50_ms": fp16_timing["p50_ms"],
                "e2e_p50_ms": fp16_timing["p50_ms"],
                "e2e_p95_ms": fp16_timing["p95_ms"],
                "vs_fp16": 1.0,
                **accuracy(fp16_output, reference),
                "weight_bytes": fp16_weight_bytes,
                "weight_vs_fp16": 1.0,
                "lut_active_bytes": 0,
                "lut_allocated_bytes": 0,
                "workspace_bytes": 0,
                "static_bytes": fp16_weight_bytes,
                "concurrency": args.concurrency,
                "concurrency_p50_ms": fp16_concurrency["p50_ms"],
                "concurrency_throughput_speedup": (
                    args.concurrency * float(fp16_timing["p50_ms"])
                    / float(fp16_concurrency["p50_ms"])
                ),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            })

            for spec in build_specs(args, k):
                module: nn.Module | None = None
                try:
                    torch.cuda.synchronize()
                    before_module_alloc = int(torch.cuda.memory_allocated())
                    architecture = str(spec["architecture"])
                    if architecture == "native_int8":
                        module = NativeInt8LinearV014(layer).eval()
                    elif architecture == "full_rns":
                        module = FullRNSLinearV014(
                            layer,
                            logical_bits=int(spec["bits"]),
                            lut_channels=int(spec["lut_channels"]),
                            moduli_policy=str(spec.get("moduli_policy", "dense_coprime")),
                            q8_backend=str(spec["q8_backend"]),
                        ).eval()
                    elif architecture == "hybrid_fp16":
                        module = HybridLinearV014(
                            layer,
                            protected_channels=args.protected,
                            correction="fp16",
                            correction_bits=16,
                            lut_channels=0,
                            execution=str(spec["execution"]),
                        ).eval()
                    elif architecture == "hybrid_rns":
                        module = HybridLinearV014(
                            layer,
                            protected_channels=args.protected,
                            correction="rns",
                            correction_bits=int(spec["bits"]),
                            lut_channels=int(spec["lut_channels"]),
                            execution=str(spec["execution"]),
                        ).eval()
                    else:
                        raise ValueError(f"unknown architecture {architecture}")

                    torch.cuda.synchronize()
                    after_construct_alloc = int(torch.cuda.memory_allocated())
                    torch.cuda.reset_peak_memory_stats()
                    output = module(inputs[0])
                    timing = measure(
                        lambda: module(inputs[0]), args.warmup, args.iterations
                    )
                    concurrency = concurrent_measure(
                        module, inputs, args.warmup, args.iterations
                    )
                    components = module_component_timings(
                        module, inputs[0], args.warmup, args.iterations
                    )
                    memory = module.memory_report()  # type: ignore[attr-defined]
                    weight_bytes = int(memory.get("weight_bytes", 0))
                    lut_allocated = int(
                        memory.get("lut_allocated_bytes", memory.get("lut_active_bytes", 0))
                    )
                    rows.append({
                        "repeat": repeat,
                        "shape": f"{m}x{k}x{n}",
                        "variant": spec["variant"],
                        "architecture": architecture,
                        "logical_bits": spec.get("bits", 8),
                        "execution": spec.get("execution", "native"),
                        "moduli_policy": spec.get("moduli_policy", "n/a"),
                        "lut_policy": spec.get("lut_policy", "n/a"),
                        "lut_channels": spec.get("lut_channels", 0),
                        "channels": spec.get("channels", 0),
                        **components,
                        "e2e_p50_ms": timing["p50_ms"],
                        "e2e_p95_ms": timing["p95_ms"],
                        "vs_fp16": float(timing["p50_ms"]) / float(fp16_timing["p50_ms"]),
                        **accuracy(output, reference),
                        "weight_bytes": weight_bytes,
                        "weight_vs_fp16": weight_bytes / max(fp16_weight_bytes, 1),
                        "lut_active_bytes": int(memory.get("lut_active_bytes", 0)),
                        "lut_allocated_bytes": lut_allocated,
                        "workspace_bytes": int(memory.get("runner_workspace_bytes", 0)),
                        "static_bytes": weight_bytes + lut_allocated,
                        "concurrency": args.concurrency,
                        "concurrency_p50_ms": concurrency["p50_ms"],
                        "concurrency_throughput_speedup": (
                            args.concurrency * float(timing["p50_ms"])
                            / float(concurrency["p50_ms"])
                        ),
                        "module_static_allocation_delta_bytes": max(0, after_construct_alloc - before_module_alloc),
                        "module_peak_allocation_delta_bytes": max(0, int(torch.cuda.max_memory_allocated()) - before_module_alloc),
                        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    })
                except Exception as exc:
                    errors.append({
                        "repeat": repeat,
                        "shape": f"{m}x{k}x{n}",
                        "variant": spec.get("variant"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    print("ERROR", spec.get("variant"), f"{m}x{k}x{n}", repr(exc))
                finally:
                    if module is not None:
                        del module
                    gc.collect()
                    torch.cuda.empty_cache()

            del layer, weight, bias, inputs, reference, fp16_output
            gc.collect()
            torch.cuda.empty_cache()

    aggregate = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "0.14.2",
        "experiment": "unified full-RNS/hybrid matrix comparison",
        "gpu": torch.cuda.get_device_name(0),
        "shapes": [list(shape) for shape in shapes],
        "repeats": args.repeats,
        "protected_channels": args.protected,
        "rows": rows,
        "aggregate": aggregate,
        "errors": errors,
    }
    (args.output_dir / "matrix_benchmark_v014.json").write_text(
        json.dumps(payload, indent=2)
    )
    write_csv(args.output_dir / "matrix_benchmark_v014.csv", rows)
    write_csv(args.output_dir / "matrix_aggregate_v014.csv", aggregate)

    tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Shape & Variant & E2E (ms) & /FP16 & Rel. L2 & Weight/FP16 " + r"\\",
        r"\midrule",
    ]
    for row in aggregate:
        tex.append(
            f"{row['shape']} & {row['variant']} & "
            f"{float(row['e2e_p50_ms']):.3f} & {float(row['vs_fp16']):.2f} & "
            f"{float(row['relative_l2']):.2e} & {float(row['weight_vs_fp16']):.2f} "
            + r"\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Unified matrix-level comparison.}", r"\end{table}"]
    (args.output_dir / "matrix_table_v014.tex").write_text("\n".join(tex) + "\n")
    print(json.dumps({
        "rows": len(rows), "aggregate": len(aggregate),
        "errors": len(errors), "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
