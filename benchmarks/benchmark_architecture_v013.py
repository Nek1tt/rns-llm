from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable

import torch

from rns_llm.architecture_v013 import (
    MODULI_POLICIES,
    NativeInt8Runner,
    RNSArchitectureRunner,
    build_compact_lut,
    logical_dense_weight_bytes,
    lut_bytes,
    prepare_int8_weight,
    prepare_rns_weight,
    quant_max,
    select_plan,
    tensor_bytes,
)


def parse_shape(value: str) -> tuple[int, int, int]:
    text = value.lower().replace(" ", "")
    parts = text.split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must use MxKxN, e.g. 128x2560x2560")
    try:
        m, k, n = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc
    if min(m, k, n) <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    if k % 4 != 0 or n % 4 != 0:
        raise argparse.ArgumentTypeError("K and N must be multiples of 4 for INT8 GEMM")
    return m, k, n


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_times(values: list[float]) -> dict[str, float | int]:
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def benchmark_methods(
    methods: dict[str, Callable[[], object]],
    *,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    names = list(methods)
    rng = random.Random(seed)
    with torch.no_grad():
        for _ in range(warmup):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                methods[name]()
        torch.cuda.synchronize()
        samples: dict[str, list[float]] = {name: [] for name in names}
        for _ in range(iterations):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                methods[name]()
                end.record()
                end.synchronize()
                samples[name].append(float(start.elapsed_time(end)))
    return {name: summarize_times(values) for name, values in samples.items()}


def benchmark_concurrent(
    launches: list[tuple[torch.cuda.Stream, Callable[[], object]]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    device = torch.device("cuda")
    current = torch.cuda.current_stream(device)

    def launch_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(current)
        for stream, _ in launches:
            stream.wait_event(start)
        for stream, fn in launches:
            with torch.cuda.stream(stream):
                fn()
        for stream, _ in launches:
            current.wait_stream(stream)
        end.record(current)
        end.synchronize()
        return float(start.elapsed_time(end))

    with torch.no_grad():
        for _ in range(warmup):
            launch_once()
        values = [launch_once() for _ in range(iterations)]
    return summarize_times(values)


def accuracy_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    out = output.float()
    ref = reference.float()
    diff = out - ref
    ref_norm = torch.linalg.vector_norm(ref)
    diff_norm = torch.linalg.vector_norm(diff)
    rel_l2 = float((diff_norm / torch.clamp(ref_norm, min=1e-30)).item())
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    denom = torch.linalg.vector_norm(out) * torch.linalg.vector_norm(ref)
    cosine = float((torch.sum(out * ref) / torch.clamp(denom, min=1e-30)).item())
    return {
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
    }


def resolve_lut_variants(channels: int, variants: Iterable[str]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    seen: set[int] = set()
    for variant in variants:
        if variant == "none":
            count = 0
        elif variant == "one":
            count = min(1, channels)
        elif variant == "two":
            count = min(2, channels)
        elif variant == "all":
            count = channels
        else:
            raise ValueError(f"unknown LUT variant: {variant}")
        if count not in seen:
            seen.add(count)
            result.append((variant, count))
    return result


def exact_sample_check(
    *,
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    output: torch.Tensor,
    bits: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    m, k = map(int, a_cpu.shape)
    n = int(b_cpu.shape[1])
    qmax = quant_max(bits)
    sample_pairs = sorted({(rng.randrange(m), rng.randrange(n)) for _ in range(max(samples * 2, 4))})[:samples]
    a_scales_cpu = activation_scales.detach().cpu()
    b_scales_cpu = weight_scales.detach().cpu()
    output_cpu = output.detach().cpu()
    records = []
    maximum_output_error = 0.0
    for row, col in sample_pairs:
        sa = float(a_scales_cpu[row].item())
        sb = float(b_scales_cpu[col].item())
        dot = 0
        a_row = a_cpu[row]
        b_col = b_cpu[:, col]
        for idx in range(k):
            qa = round(float(a_row[idx].item()) / sa)
            qb = round(float(b_col[idx].item()) / sb)
            qa = max(-qmax, min(qmax, qa))
            qb = max(-qmax, min(qmax, qb))
            dot += int(qa) * int(qb)
        expected = float(dot * sa * sb)
        actual = float(output_cpu[row, col].item())
        error = abs(actual - expected)
        maximum_output_error = max(maximum_output_error, error)
        records.append(
            {
                "row": row,
                "col": col,
                "exact_integer_dot": str(dot),
                "expected_dequantized": expected,
                "gpu_output": actual,
                "absolute_error": error,
            }
        )
    return {
        "samples": records,
        "maximum_dequantized_error": maximum_output_error,
        "note": (
            "The integer dot products are evaluated with Python arbitrary-precision integers. "
            "The GPU reconstructs the same signed value in a two-limb 128-bit Garner path, "
            "then rounds the dequantized result to FP32."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(text: object) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%")


def write_tex_assets(output_dir: Path, payload: dict[str, object]) -> None:
    rows = payload["flat_results"]
    table_rows = []
    for row in rows:
        if row["method"] in {"fp32", "fp16", "native_int8"} or (
            row["method"] == "rns" and row["policy"] == "large_primes" and row["lut_variant"] in {"none", "two"}
        ):
            table_rows.append(row)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Shape & Method & Ch. & Core, ms & E2E, ms & Rel. $L_2$ \\\\",
        "\\midrule",
    ]
    current_shape = None
    for row in table_rows:
        shape = row["shape"]
        shape_cell = tex_escape(shape) if shape != current_shape else ""
        current_shape = shape
        if row["method"] == "rns":
            method = f"RNS-q{row['bits']} ({row['lut_variant']} LUT)"
        else:
            method = row["method"].upper()
        lines.append(
            f"{shape_cell} & {tex_escape(method)} & {row.get('channels', 1)} & "
            f"{row['core_p50_ms']:.3f} & {row['e2e_p50_ms']:.3f} & "
            f"{row['relative_l2']:.3e} \\\\" 
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Architecture and logical-precision comparison. RNS core time includes all residue-channel GEMMs, modular reduction, Garner reconstruction, and dequantization.}",
            "\\label{tab:v013-architecture}",
            "\\end{table}",
        ]
    )
    (output_dir / "architecture_results_table.tex").write_text("\n".join(lines) + "\n")

    lut_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Logical format & Channels & LUT channels & LUT bytes & Saving vs. all \\\\",
        "\\midrule",
    ]
    for row in payload["lut_memory_rows"]:
        lut_lines.append(
            f"q{row['bits']} / {tex_escape(row['policy'])} & {row['channels']} & "
            f"{row['lut_channels']} & {row['lut_bytes']} & {100.0 * row['saving_vs_all']:.1f}\\% \\\\"
        )
    lut_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Compact reduction-table footprint. Each table contains four byte-position slices of 256 signed 16-bit entries (2048 bytes).}",
            "\\label{tab:v013-lut-memory}",
            "\\end{table}",
        ]
    )
    (output_dir / "lut_memory_table.tex").write_text("\n".join(lut_lines) + "\n")

    macros = ["% Auto-generated by benchmark_architecture_v013.py"]
    large_rows = [
        row for row in rows
        if row["method"] == "rns" and row["policy"] == "large_primes" and row["lut_variant"] == "two"
    ]
    if large_rows:
        median_gap = statistics.median(row["core_vs_fp16"] for row in large_rows)
        macros.append(f"\\newcommand{{\\RNSArchitectureMedianCoreGap}}{{{median_gap:.2f}}}")
    (output_dir / "architecture_result_macros.tex").write_text("\n".join(macros) + "\n")

    paragraph = (
        "The v0.13 architecture matrix evaluates FP32, FP16, native INT8, and full-RNS "
        "q8/q16/q32 multiplication under a common matrix protocol. The RNS dynamic range "
        "is selected from the complete dot-product bound rather than from the scalar operand "
        "range. Compact byte-decomposition lookup tables are compared against arithmetic "
        "Barrett reduction with zero, one, two, and all RNS channels using tables. The generated "
        "tables and macros must be inserted only after the GPU run; this template deliberately "
        "contains no unmeasured speed claim."
    )
    (output_dir / "architecture_results_paragraph.tex").write_text(paragraph + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        default=None,
        help="repeatable MxKxN shape",
    )
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--policy",
        choices=sorted(MODULI_POLICIES),
        nargs="+",
        default=["large_primes"],
    )
    parser.add_argument(
        "--lut-variant",
        choices=["none", "one", "two", "all"],
        nargs="+",
        default=["none", "one", "two", "all"],
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--exact-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if any(bits not in (8, 16, 32) for bits in args.bits):
        raise SystemExit("--bits supports only 8, 16, 32")
    if min(args.warmup, args.iterations) < 1:
        raise SystemExit("warmup and iterations must be positive")
    if any(value not in (1, 4) for value in args.concurrency):
        raise SystemExit("concurrency comparison supports 1 and 4")

    shapes = args.shape or [(16, 768, 768), (128, 768, 768)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    results: list[dict[str, object]] = []
    flat_results: list[dict[str, object]] = []
    concurrency_rows: list[dict[str, object]] = []
    lut_memory_rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []

    for shape_index, (m, k, n) in enumerate(shapes):
        print(f"\n=== Shape M={m}, K={k}, N={n} ===", flush=True)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + shape_index * 97)
        a = torch.randn((m, k), generator=generator, device=device, dtype=torch.float32)
        b = torch.randn((k, n), generator=generator, device=device, dtype=torch.float32) * 0.025
        reference = torch.empty((m, n), dtype=torch.float32, device=device)
        fp16_output = torch.empty((m, n), dtype=torch.float16, device=device)
        a_cpu_for_exactness = a.detach().cpu() if args.exact_samples > 0 else None
        b_cpu_for_exactness = b.detach().cpu() if args.exact_samples > 0 else None
        a_half = a.half()
        b_half = b.half()

        baseline_methods = {
            "fp32_core": lambda: torch.mm(a, b, out=reference),
            "fp16_core": lambda: torch.mm(a_half, b_half, out=fp16_output),
            "fp32_e2e": lambda: torch.mm(a, b, out=reference),
            "fp16_e2e": lambda: (a_half.copy_(a), torch.mm(a_half, b_half, out=fp16_output))[-1],
        }
        baseline_timings = benchmark_methods(
            baseline_methods,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed + shape_index,
        )
        torch.mm(a, b, out=reference)
        torch.mm(a_half, b_half, out=fp16_output)
        fp16_metrics = accuracy_metrics(fp16_output, reference)
        fp32_metrics = accuracy_metrics(reference, reference)

        fp32_row = {
            "shape": f"{m}x{k}x{n}",
            "method": "fp32",
            "bits": 32,
            "policy": "native",
            "lut_variant": "none",
            "channels": 1,
            "core_p50_ms": baseline_timings["fp32_core"]["p50_ms"],
            "e2e_p50_ms": baseline_timings["fp32_e2e"]["p50_ms"],
            **fp32_metrics,
            "weight_bytes": tensor_bytes(b),
            "runtime_workspace_bytes": tensor_bytes(reference),
            "core_vs_fp16": baseline_timings["fp32_core"]["p50_ms"] / baseline_timings["fp16_core"]["p50_ms"],
            "e2e_vs_fp16": baseline_timings["fp32_e2e"]["p50_ms"] / baseline_timings["fp16_e2e"]["p50_ms"],
            "lut_bytes": 0,
        }
        fp16_row = {
            "shape": f"{m}x{k}x{n}",
            "method": "fp16",
            "bits": 16,
            "policy": "native",
            "lut_variant": "none",
            "channels": 1,
            "core_p50_ms": baseline_timings["fp16_core"]["p50_ms"],
            "e2e_p50_ms": baseline_timings["fp16_e2e"]["p50_ms"],
            **fp16_metrics,
            "weight_bytes": tensor_bytes(b_half),
            "runtime_workspace_bytes": tensor_bytes(fp16_output) + tensor_bytes(a_half),
            "core_vs_fp16": 1.0,
            "e2e_vs_fp16": 1.0,
            "lut_bytes": 0,
        }
        flat_results.extend([fp32_row, fp16_row])

        start_wall = time.perf_counter()
        native_weight = prepare_int8_weight(b)
        torch.cuda.synchronize()
        native_prepare_ms = 1000.0 * (time.perf_counter() - start_wall)
        native_runner = NativeInt8Runner(native_weight, m=m)
        native_runner.encode(a)
        native_methods = {
            "native_int8_core": native_runner.core,
            "native_int8_e2e": lambda: native_runner.e2e(a),
        }
        native_timings = benchmark_methods(
            native_methods,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed + 100 + shape_index,
        )
        native_output = native_runner.e2e(a).clone()
        torch.cuda.synchronize()
        native_metrics = accuracy_metrics(native_output, reference)
        native_row = {
            "shape": f"{m}x{k}x{n}",
            "method": "native_int8",
            "bits": 8,
            "policy": "native",
            "lut_variant": "none",
            "channels": 1,
            "core_p50_ms": native_timings["native_int8_core"]["p50_ms"],
            "e2e_p50_ms": native_timings["native_int8_e2e"]["p50_ms"],
            **native_metrics,
            "weight_bytes": native_weight.storage_bytes,
            "runtime_workspace_bytes": native_runner.runtime_workspace_bytes,
            "weight_prepare_ms": native_prepare_ms,
            "core_vs_fp16": native_timings["native_int8_core"]["p50_ms"] / baseline_timings["fp16_core"]["p50_ms"],
            "e2e_vs_fp16": native_timings["native_int8_e2e"]["p50_ms"] / baseline_timings["fp16_e2e"]["p50_ms"],
            "lut_bytes": 0,
            "dense_logical_weight_bytes": logical_dense_weight_bytes(k, n, 8),
            "weight_ratio_to_dense_logical": native_weight.storage_bytes / logical_dense_weight_bytes(k, n, 8),
        }
        flat_results.append(native_row)
        del native_runner, native_output

        shape_result: dict[str, object] = {
            "shape": {"m": m, "k": k, "n": n},
            "baseline_timings": baseline_timings,
            "baseline_accuracy": {"fp32": fp32_metrics, "fp16": fp16_metrics, "native_int8": native_metrics},
            "rns": [],
        }

        for policy_index, policy in enumerate(args.policy):
            for bits_index, bits in enumerate(args.bits):
                plan = select_plan(k, bits, policy)
                print(
                    f"q{bits} {policy}: {plan.channels} channels, "
                    f"product={plan.product_bits} bits, required={plan.required_bits} bits",
                    flush=True,
                )
                plan_rows.append({"shape": f"{m}x{k}x{n}", **plan.to_dict()})
                torch.cuda.synchronize()
                allocated_before_weight = torch.cuda.memory_allocated(device)
                start_wall = time.perf_counter()
                prepared = prepare_rns_weight(b, plan)
                torch.cuda.synchronize()
                prepare_ms = 1000.0 * (time.perf_counter() - start_wall)
                allocated_after_weight = torch.cuda.memory_allocated(device)
                weight_allocation_delta = allocated_after_weight - allocated_before_weight
                variant_results = []
                baseline_variant_output: torch.Tensor | None = None

                for variant_index, (variant, lut_count) in enumerate(
                    resolve_lut_variants(plan.channels, args.lut_variant)
                ):
                    torch.cuda.synchronize()
                    allocated_before_lut = torch.cuda.memory_allocated(device)
                    shared_lut = build_compact_lut(plan.moduli, lut_count, device=device)
                    torch.cuda.synchronize()
                    allocated_after_lut = torch.cuda.memory_allocated(device)
                    lut_allocation_delta = allocated_after_lut - allocated_before_lut
                    allocated_before_runner = allocated_after_lut
                    runner = RNSArchitectureRunner(
                        prepared,
                        m=m,
                        lut_channels=lut_count,
                        compact_lut=shared_lut,
                    )
                    torch.cuda.synchronize()
                    allocated_after_runner = torch.cuda.memory_allocated(device)
                    runner_allocation_delta = allocated_after_runner - allocated_before_runner
                    runner.encode(a)
                    timings = benchmark_methods(
                        {"core": runner.core, "e2e": lambda: runner.e2e(a)},
                        warmup=args.warmup,
                        iterations=args.iterations,
                        seed=(
                            args.seed
                            + shape_index * 1000
                            + policy_index * 200
                            + bits_index * 20
                            + variant_index
                        ),
                    )
                    output = runner.e2e(a).clone()
                    torch.cuda.synchronize()
                    metrics = accuracy_metrics(output, reference)
                    is_reference_variant = baseline_variant_output is None
                    if is_reference_variant:
                        baseline_variant_output = output.clone()
                    lut_equivalence_max_abs = float(
                        (output - baseline_variant_output).abs().max().item()
                    )
                    if is_reference_variant and args.exact_samples > 0:
                        exactness = exact_sample_check(
                            a_cpu=a_cpu_for_exactness,
                            b_cpu=b_cpu_for_exactness,
                            activation_scales=runner.activation_scales,
                            weight_scales=prepared.scales_n,
                            output=output,
                            bits=bits,
                            samples=args.exact_samples,
                            seed=args.seed + bits * 17 + shape_index,
                        )
                    else:
                        exactness = {
                            "reused_reference_variant": True,
                            "note": "Exact Python integer verification is run once per shape/bit/policy; LUT variants are checked by direct GPU output equality.",
                        }

                    full_lut_bytes = lut_bytes(plan.channels)
                    current_lut_bytes = lut_bytes(lut_count)
                    saving = 1.0 - current_lut_bytes / full_lut_bytes
                    dense_logical_bytes = logical_dense_weight_bytes(k, n, bits)
                    memory = {
                        "fp32_weight_bytes": tensor_bytes(b),
                        "fp16_weight_bytes": tensor_bytes(b_half),
                        "dense_logical_weight_bytes": dense_logical_bytes,
                        "rns_weight_bytes": prepared.storage_bytes,
                        "rns_weight_ratio_to_fp16": prepared.storage_bytes / tensor_bytes(b_half),
                        "rns_weight_ratio_to_fp32": prepared.storage_bytes / tensor_bytes(b),
                        "rns_weight_ratio_to_dense_logical": prepared.storage_bytes / dense_logical_bytes,
                        "lut_bytes": current_lut_bytes,
                        "full_lut_bytes": full_lut_bytes,
                        "lut_saving_vs_all": saving,
                        "lut_saving_gt_50_percent": saving > 0.5,
                        "constant_bytes": runner.constant_bytes,
                        "runtime_workspace_bytes": runner.runtime_workspace_bytes,
                        "cuda_weight_allocation_delta_bytes": weight_allocation_delta,
                        "cuda_lut_allocation_delta_bytes": lut_allocation_delta,
                        "cuda_runner_allocation_delta_bytes": runner_allocation_delta,
                    }
                    lut_memory_rows.append(
                        {
                            "shape": f"{m}x{k}x{n}",
                            "bits": bits,
                            "policy": policy,
                            "channels": plan.channels,
                            "lut_variant": variant,
                            "lut_channels": lut_count,
                            "lut_bytes": current_lut_bytes,
                            "full_lut_bytes": full_lut_bytes,
                            "saving_vs_all": saving,
                            "passes_gt_50_target": saving > 0.5,
                        }
                    )

                    concurrency_result: dict[str, object] = {}
                    if 4 in args.concurrency:
                        runners = [runner]
                        streams = [torch.cuda.Stream(device=device)]
                        torch.cuda.synchronize()
                        allocated_before_extra_runners = torch.cuda.memory_allocated(device)
                        for _ in range(3):
                            runners.append(
                                RNSArchitectureRunner(
                                    prepared,
                                    m=m,
                                    lut_channels=lut_count,
                                    compact_lut=shared_lut,
                                )
                            )
                            streams.append(torch.cuda.Stream(device=device))
                        torch.cuda.synchronize()
                        extra_runner_allocation_delta = (
                            torch.cuda.memory_allocated(device) - allocated_before_extra_runners
                        )
                        inputs = [a]
                        for idx in range(1, 4):
                            inputs.append(a.roll(shifts=idx, dims=0).contiguous())
                        for item_runner, item_input in zip(runners, inputs):
                            item_runner.encode(item_input)
                        single_core_ms = float(timings["core"]["p50_ms"])
                        concurrent_core = benchmark_concurrent(
                            [(stream, item_runner.core) for stream, item_runner in zip(streams, runners)],
                            warmup=max(2, args.warmup // 2),
                            iterations=max(5, args.iterations // 2),
                        )
                        concurrent_e2e = benchmark_concurrent(
                            [
                                (stream, lambda r=item_runner, x=item_input: r.e2e(x))
                                for stream, item_runner, item_input in zip(streams, runners, inputs)
                            ],
                            warmup=max(2, args.warmup // 2),
                            iterations=max(5, args.iterations // 2),
                        )
                        core_wall = float(concurrent_core["p50_ms"])
                        e2e_wall = float(concurrent_e2e["p50_ms"])
                        concurrency_result = {
                            "requests": 4,
                            "core_wall_p50_ms": core_wall,
                            "e2e_wall_p50_ms": e2e_wall,
                            "core_throughput_speedup": 4.0 * single_core_ms / core_wall,
                            "core_contention_ratio": core_wall / single_core_ms,
                            "shared_lut": True,
                            "shared_weight": True,
                            "aggregate_workspace_bytes": sum(r.runtime_workspace_bytes for r in runners),
                            "cuda_extra_three_runners_allocation_delta_bytes": extra_runner_allocation_delta,
                        }
                        concurrency_rows.append(
                            {
                                "shape": f"{m}x{k}x{n}",
                                "bits": bits,
                                "policy": policy,
                                "lut_variant": variant,
                                "lut_channels": lut_count,
                                **concurrency_result,
                            }
                        )
                        del runners, streams, inputs

                    result = {
                        "bits": bits,
                        "policy": policy,
                        "plan": plan.to_dict(),
                        "lut_variant": variant,
                        "lut_channels": lut_count,
                        "weight_prepare_ms": prepare_ms,
                        "timings": timings,
                        "accuracy": metrics,
                        "lut_equivalence_max_abs": lut_equivalence_max_abs,
                        "exactness_sample": exactness,
                        "memory": memory,
                        "concurrency": concurrency_result,
                    }
                    variant_results.append(result)
                    flat_results.append(
                        {
                            "shape": f"{m}x{k}x{n}",
                            "method": "rns",
                            "bits": bits,
                            "policy": policy,
                            "lut_variant": variant,
                            "lut_channels": lut_count,
                            "channels": plan.channels,
                            "core_p50_ms": timings["core"]["p50_ms"],
                            "e2e_p50_ms": timings["e2e"]["p50_ms"],
                            **metrics,
                            "weight_bytes": prepared.storage_bytes,
                            "dense_logical_weight_bytes": dense_logical_bytes,
                            "weight_ratio_to_dense_logical": prepared.storage_bytes / dense_logical_bytes,
                            "lut_bytes": current_lut_bytes,
                            "runtime_workspace_bytes": runner.runtime_workspace_bytes,
                            "cuda_weight_allocation_delta_bytes": weight_allocation_delta,
                            "cuda_lut_allocation_delta_bytes": lut_allocation_delta,
                            "cuda_runner_allocation_delta_bytes": runner_allocation_delta,
                            "weight_prepare_ms": prepare_ms,
                            "core_vs_fp16": timings["core"]["p50_ms"] / baseline_timings["fp16_core"]["p50_ms"],
                            "e2e_vs_fp16": timings["e2e"]["p50_ms"] / baseline_timings["fp16_e2e"]["p50_ms"],
                            "lut_equivalence_max_abs": lut_equivalence_max_abs,
                            "lut_saving_vs_all": saving,
                        }
                    )
                    print(
                        f"  {variant:>4} LUT ({lut_count:2d}): "
                        f"core={timings['core']['p50_ms']:.3f} ms, "
                        f"e2e={timings['e2e']['p50_ms']:.3f} ms, "
                        f"relL2={metrics['relative_l2']:.3e}, "
                        f"LUT={current_lut_bytes} B",
                        flush=True,
                    )
                    del runner, output, shared_lut
                shape_result["rns"].append(
                    {
                        "bits": bits,
                        "policy": policy,
                        "plan": plan.to_dict(),
                        "weight_prepare_ms": prepare_ms,
                        "variants": variant_results,
                    }
                )
                del prepared, baseline_variant_output
                torch.cuda.empty_cache()

        results.append(shape_result)
        del a, b, reference, fp16_output, a_half, b_half, native_weight
        del a_cpu_for_exactness, b_cpu_for_exactness
        torch.cuda.empty_cache()

    payload: dict[str, object] = {
        "version": "0.13.0",
        "experiment": "architecture_precision_and_lut_ablation",
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "protocol": {
            "shapes": [list(shape) for shape in shapes],
            "logical_bits": args.bits,
            "policies": args.policy,
            "lut_variants": args.lut_variant,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "concurrency": args.concurrency,
            "cuda_events": True,
            "randomized_method_order": True,
            "weight_preparation_excluded_from_hot_path": True,
            "rns_core_definition": (
                "all int8 residue-channel GEMMs + modular reduction + 128-bit Garner + FP32 dequantization"
            ),
            "rns_e2e_definition": "activation max reduction + q8/q16/q32 quantization + residue encoding + RNS core",
        },
        "plan_rows": plan_rows,
        "results": results,
        "flat_results": flat_results,
        "lut_memory_rows": lut_memory_rows,
        "concurrency_rows": concurrency_rows,
        "interpretation_guardrails": [
            "LUT saving is measured relative to the LUT subsystem, not total model memory.",
            "RNS q8/q16/q32 denotes the logical integer quantizer before residue encoding; every residue channel is int8.",
            "The q32 dot-product range is reconstructed in a custom two-limb 128-bit Garner path.",
            "No speed or memory claim is valid until this notebook is run on the target GPU.",
        ],
    }

    json_path = args.output_dir / "architecture_results_v013.json"
    json_path.write_text(json.dumps(payload, indent=2))
    write_csv(args.output_dir / "architecture_matrix_v013.csv", flat_results)
    write_csv(args.output_dir / "lut_memory_v013.csv", lut_memory_rows)
    write_csv(args.output_dir / "concurrency_v013.csv", concurrency_rows)
    write_csv(args.output_dir / "moduli_plans_v013.csv", plan_rows)
    write_tex_assets(args.output_dir, payload)
    print("\nSaved:", json_path)


if __name__ == "__main__":
    main()
