from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from rns_llm.prefill_v011 import PrefillLayerV011


def parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in text.lower().split("x"))
    if len(values) != 3 or min(values) <= 0:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    m, k, n = values
    if k % 4 or n % 4:
        raise argparse.ArgumentTypeError("K and N must be multiples of 4")
    return m, k, n


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def measure(fn: Callable[[], object], warmup: int, iterations: int) -> dict[str, float | int]:
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
        samples.append(float(start.elapsed_time(end)))
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": len(samples),
    }


def concurrent_measure(
    runners: list[object],
    inputs: list[torch.Tensor],
    method: str,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    streams = [torch.cuda.Stream() for _ in runners]
    current = torch.cuda.current_stream()

    def launch() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(current)
        for stream in streams:
            stream.wait_event(start)
        for stream, runner, x in zip(streams, runners, inputs):
            with torch.cuda.stream(stream):
                getattr(runner, method)(x)
        for stream in streams:
            current.wait_stream(stream)
        end.record(current)
        end.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(warmup):
        launch()
    values = [launch() for _ in range(iterations)]
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    out = output.float()
    ref = reference.float()
    diff = out - ref
    rel = torch.linalg.vector_norm(diff) / torch.clamp(
        torch.linalg.vector_norm(ref), min=1e-30
    )
    cosine = torch.sum(out * ref) / torch.clamp(
        torch.linalg.vector_norm(out) * torch.linalg.vector_norm(ref), min=1e-30
    )
    return {
        "relative_l2": float(rel.item()),
        "cosine": float(cosine.item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", type=parse_shape)
    ap.add_argument("--bits", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--protected", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1401)
    ap.add_argument("--output-dir", type=Path, default=Path("results/v0.14.2/hybrid"))
    args = ap.parse_args()
    shapes = args.shape or [(16, 2560, 2560), (128, 2560, 2560), (16, 2560, 10240), (128, 2560, 10240)]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    concurrency_rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for m, k, n in shapes:
        weight = (torch.randn(n, k, device=device, dtype=torch.float32) / math.sqrt(k)).contiguous()
        bias = (torch.randn(n, device=device, dtype=torch.float32) * 0.01).contiguous()
        protected = torch.topk(weight.abs().sum(dim=0), min(args.protected, k - 1)).indices.sort().values.int().cpu()
        pack = {
            "layer_name": f"synthetic_{m}x{k}x{n}",
            "weight": weight.cpu(),
            "bias": bias.cpu(),
            "protected_indices": protected,
            "selected_plan": {"source": "weight_l1", "protected": protected.tolist()},
            "statistics": {},
        }
        layer = PrefillLayerV011.from_pack(pack, device=device, optimized_rns_bits=tuple(sorted(set(args.bits))))
        inputs = [torch.randn(m, k, device=device, dtype=torch.float32) for _ in range(args.concurrency)]
        reference = F.linear(inputs[0], weight, bias)

        for bits in args.bits:
            channels = len(layer.protected_rns[bits].moduli)
            lut_variants = [("none", 0), ("one", min(1, channels)), ("two", min(2, channels)), ("all", channels)]
            dedup = []
            seen = set()
            for name, count in lut_variants:
                if count not in seen:
                    seen.add(count)
                    dedup.append((name, count))
            for lut_name, lut_count in dedup:
                runner = layer.runner(m, logical_bits=bits, lut_channels=lut_count)
                runner.cast_fp16(inputs[0]); runner.preprocess_native(inputs[0]); runner.preprocess_hybrid(inputs[0])
                method_map = {
                    "fp16": (runner.fp16_core, lambda: runner.fp16_e2e(inputs[0])),
                    "native_int8": (runner.native_core, lambda: runner.native_e2e(inputs[0])),
                    "hybrid_fp16_serial": (runner.hybrid_fp16_serial_core, lambda: runner.hybrid_fp16_serial_e2e(inputs[0])),
                    "hybrid_rns_serial": (runner.hybrid_rns_serial_core, lambda: runner.hybrid_rns_serial_e2e(inputs[0])),
                    "hybrid_rns_parallel": (runner.hybrid_rns_parallel_core, lambda: runner.hybrid_rns_parallel_e2e(inputs[0])),
                }
                for method, (core_fn, e2e_fn) in method_map.items():
                    core = measure(core_fn, args.warmup, args.iterations)
                    e2e = measure(e2e_fn, args.warmup, args.iterations)
                    output = e2e_fn().detach().clone()
                    torch.cuda.synchronize()
                    row = {
                        "shape": f"{m}x{k}x{n}",
                        "architecture": "hybrid" if method.startswith("hybrid") else method,
                        "method": method,
                        "bits": bits if "rns" in method else "",
                        "protected": int(layer.p),
                        "protected_padded": int(layer.p_padded),
                        "rns_channels": channels if "rns" in method else "",
                        "lut_variant": lut_name if "rns" in method else "n/a",
                        "lut_channels": lut_count if "rns" in method else 0,
                        "core_p50_ms": core["p50_ms"],
                        "core_p95_ms": core["p95_ms"],
                        "e2e_p50_ms": e2e["p50_ms"],
                        "e2e_p95_ms": e2e["p95_ms"],
                        **metrics(output, reference),
                        **runner.storage_bytes(),
                    }
                    rows.append(row)

                concurrent_runners = [layer.runner(m, logical_bits=bits, lut_channels=lut_count) for _ in range(args.concurrency)]
                serial = measure(
                    lambda: concurrent_runners[0].hybrid_rns_serial_e2e(inputs[0]),
                    args.warmup,
                    args.iterations,
                )
                concurrent = concurrent_measure(
                    concurrent_runners, inputs, "hybrid_rns_serial_e2e", args.warmup, args.iterations
                )
                concurrency_rows.append({
                    "shape": f"{m}x{k}x{n}",
                    "architecture": "hybrid_rns",
                    "bits": bits,
                    "lut_variant": lut_name,
                    "lut_channels": lut_count,
                    "concurrency": args.concurrency,
                    "single_p50_ms": serial["p50_ms"],
                    "concurrent_p50_ms": concurrent["p50_ms"],
                    "throughput_speedup": args.concurrency * float(serial["p50_ms"]) / float(concurrent["p50_ms"]),
                    "contention_ratio": float(concurrent["p50_ms"]) / float(serial["p50_ms"]),
                })
                details.append({
                    "shape": [m, k, n], "bits": bits, "lut": lut_name,
                    "moduli": list(layer.protected_rns[bits].moduli),
                    "storage": runner.storage_bytes(),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "0.14.2",
        "experiment": "hybrid LUT/latency/accuracy/memory/concurrency",
        "gpu": torch.cuda.get_device_name(0),
        "rows": rows,
        "concurrency": concurrency_rows,
        "details": details,
    }
    (args.output_dir / "hybrid_benchmark_v014.json").write_text(json.dumps(payload, indent=2))
    write_csv(args.output_dir / "hybrid_benchmark_v014.csv", rows)
    write_csv(args.output_dir / "hybrid_concurrency_v014.csv", concurrency_rows)
    print(json.dumps({"rows": len(rows), "concurrency_rows": len(concurrency_rows), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
