from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch

from rns_llm.architecture_v013 import select_plan
from rns_llm.hybrid_v010 import choose_moduli
from rns_llm.unified_v014 import (
    collect_attention_memory,
    install_full_rns_opt_attention,
    install_hybrid_opt_attention,
    install_native_int8_opt_attention,
)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def measure(fn: Callable[[], object], warmup: int, iterations: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def concurrent_measure(
    modules: list[torch.nn.Module],
    inputs: list[torch.Tensor],
    masks: list[torch.Tensor],
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
        for stream, module, x, mask in zip(streams, modules, inputs, masks):
            with torch.cuda.stream(stream):
                module(x, attention_mask=mask, output_attentions=False)
        for stream in streams:
            current.wait_stream(stream)
        end.record(current)
        end.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(warmup):
        once()
    values = [once() for _ in range(iterations)]
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "samples": len(values),
    }


def shared_fp16_request_modules(
    template: torch.nn.Module,
    source: torch.nn.Module,
    count: int,
) -> list[torch.nn.Module]:
    """Create request-local attention state while sharing immutable FP16 weights."""
    modules = []
    for _ in range(count):
        clone = copy.deepcopy(template).eval()
        clone.q_proj = source.q_proj
        clone.k_proj = source.k_proj
        clone.v_proj = source.v_proj
        clone.out_proj = source.out_proj
        modules.append(clone)
    return modules


def shared_installed_request_modules(
    template: torch.nn.Module,
    installed,
    count: int,
) -> list[torch.nn.Module]:
    """Separate QKV coordinator caches per request, shared prepared weights/LUTs.

    A single CachedRNSQKV object is stateful and cannot safely be called from
    multiple streams. Each request therefore gets its own coordinator while
    the projection and output-projection modules (and their immutable prepared
    weights/LUT tensors) remain shared. Runner workspaces are already cached by
    CUDA stream in the projection modules.
    """
    from rns_llm.layers.rns_qkv import CachedRNSQKV

    modules = []
    for _ in range(count):
        clone = copy.deepcopy(template).eval()
        coordinator = CachedRNSQKV(installed.coordinator.projection).eval()
        clone.rns_qkv_v014 = coordinator
        clone.q_proj, clone.k_proj, clone.v_proj = coordinator.slices()
        clone.out_proj = installed.out_projection
        modules.append(clone)
    return modules


def accuracy(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    out = output.float()
    ref = reference.float()
    diff = out - ref
    return {
        "relative_l2": float(
            (torch.linalg.vector_norm(diff) /
             torch.clamp(torch.linalg.vector_norm(ref), min=1e-30)).item()
        ),
        "cosine": float(
            (torch.sum(out * ref) /
             torch.clamp(
                 torch.linalg.vector_norm(out) * torch.linalg.vector_norm(ref),
                 min=1e-30,
             )).item()
        ),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
    }


def causal_mask(batch: int, seq: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    value = torch.finfo(dtype).min
    mask = torch.zeros((seq, seq), dtype=dtype, device=device)
    upper = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(upper, value)
    return mask.view(1, 1, seq, seq).expand(batch, 1, seq, seq).contiguous()


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


def resolve_lut(label: str, channels: int) -> int:
    return {
        "none": 0,
        "one": min(1, channels),
        "two": min(2, channels),
        "all": channels,
    }[label]


def build_specs(args, hidden: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"name": "native_int8", "architecture": "native_int8"},
        {"name": "hybrid_fp16_serial", "architecture": "hybrid_fp16", "execution": "serial"},
    ]
    if args.include_parallel:
        specs.append({
            "name": "hybrid_fp16_parallel",
            "architecture": "hybrid_fp16",
            "execution": "parallel",
        })

    for bits in args.full_bits:
        channels = select_plan(hidden, bits, args.moduli_policy).channels
        for policy in args.lut_policies:
            count = resolve_lut(policy, channels)
            specs.append({
                "name": f"full_rns_int{bits}_{policy}",
                "architecture": "full_rns",
                "bits": bits,
                "lut_policy": policy,
                "lut_channels": count,
                "q8_backend": "v013",
            })
        if bits == 8 and args.include_v07_q8:
            specs.append({
                "name": "full_rns_int8_v07_lut2",
                "architecture": "full_rns",
                "bits": 8,
                "lut_policy": "two",
                "lut_channels": 2,
                "q8_backend": "v07",
            })

    p_pad = ((args.protected + 3) // 4) * 4
    for bits in args.hybrid_bits:
        channels = len(choose_moduli(bits, p_pad))
        for policy in args.lut_policies:
            count = resolve_lut(policy, channels)
            executions = ["serial", "parallel"] if args.include_parallel else ["serial"]
            for execution in executions:
                specs.append({
                    "name": f"hybrid_rns_q{bits}_{execution}_{policy}",
                    "architecture": "hybrid_rns",
                    "bits": bits,
                    "execution": execution,
                    "lut_policy": policy,
                    "lut_channels": count,
                })
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete OPT self-attention benchmark for FP16, INT8, full-RNS and hybrid RNS"
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--protected", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--full-bits", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--hybrid-bits", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument(
        "--lut-policies", nargs="+", choices=["none", "one", "two", "all"],
        default=["none", "one", "two", "all"],
    )
    parser.add_argument("--moduli-policy", choices=["dense_coprime", "large_primes", "school_small"], default="dense_coprime")
    parser.add_argument("--include-v07-q8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v0.14.2/attention"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    if any(bits not in (8, 16, 32) for bits in args.full_bits + args.hybrid_bits):
        raise SystemExit("only 8/16/32 logical bits are supported")

    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device).eval()
    original_attention = copy.deepcopy(model.model.decoder.layers[0].self_attn).eval()
    hidden = int(model.config.hidden_size)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    baseline = copy.deepcopy(original_attention).eval()
    torch.manual_seed(1402)
    inputs = [
        torch.randn(args.batch, args.seq, hidden, device=device, dtype=torch.float16)
        for _ in range(args.concurrency)
    ]
    masks = [
        causal_mask(args.batch, args.seq, torch.float16, device)
        for _ in range(args.concurrency)
    ]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        reference = baseline(
            inputs[0], attention_mask=masks[0], output_attentions=False
        )[0]
    baseline_timing = measure(
        lambda: baseline(inputs[0], attention_mask=masks[0], output_attentions=False),
        args.warmup,
        args.iterations,
    )
    baseline_request_modules = shared_fp16_request_modules(
        original_attention, baseline, args.concurrency
    )
    baseline_concurrency = concurrent_measure(
        baseline_request_modules, inputs, masks, args.warmup, args.iterations
    )
    baseline_weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in baseline.parameters()
    )

    rows: list[dict[str, object]] = [{
        "variant": "fp16",
        "architecture": "fp16",
        "logical_bits": 16,
        "execution": "native",
        "lut_policy": "n/a",
        "lut_channels": 0,
        "p50_ms": baseline_timing["p50_ms"],
        "p95_ms": baseline_timing["p95_ms"],
        "vs_fp16": 1.0,
        "relative_l2": 0.0,
        "cosine": 1.0,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "weight_bytes": baseline_weight_bytes,
        "weight_vs_fp16": 1.0,
        "lut_active_bytes": 0,
        "lut_allocated_bytes": 0,
        "workspace_bytes": 0,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "concurrency": args.concurrency,
        "concurrency_p50_ms": baseline_concurrency["p50_ms"],
        "concurrency_throughput_speedup": (
            args.concurrency * float(baseline_timing["p50_ms"])
            / float(baseline_concurrency["p50_ms"])
        ),
    }]

    errors: list[dict[str, str]] = []
    for spec in build_specs(args, hidden):
        module = None
        installed = None
        request_modules = None
        try:
            torch.cuda.synchronize()
            before_module_alloc = int(torch.cuda.memory_allocated())
            module = copy.deepcopy(original_attention).eval()
            architecture = str(spec["architecture"])
            if architecture == "native_int8":
                installed = install_native_int8_opt_attention(
                    module, include_out_proj=True
                )
            elif architecture == "full_rns":
                installed = install_full_rns_opt_attention(
                    module,
                    logical_bits=int(spec["bits"]),
                    lut_channels=int(spec["lut_channels"]),
                    moduli_policy=args.moduli_policy,
                    q8_backend=str(spec["q8_backend"]),
                    include_out_proj=True,
                )
            elif architecture == "hybrid_fp16":
                installed = install_hybrid_opt_attention(
                    module,
                    protected_channels=args.protected,
                    correction_bits=16,
                    lut_channels=0,
                    correction="fp16",
                    execution=str(spec["execution"]),
                    include_out_proj=True,
                )
            elif architecture == "hybrid_rns":
                installed = install_hybrid_opt_attention(
                    module,
                    protected_channels=args.protected,
                    correction_bits=int(spec["bits"]),
                    lut_channels=int(spec["lut_channels"]),
                    correction="rns",
                    execution=str(spec["execution"]),
                    include_out_proj=True,
                )
            else:
                raise ValueError(f"unknown architecture {architecture}")

            torch.cuda.synchronize()
            after_construct_alloc = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                output = module(
                    inputs[0], attention_mask=masks[0], output_attentions=False
                )[0]
            timing = measure(
                lambda: module(
                    inputs[0], attention_mask=masks[0], output_attentions=False
                ),
                args.warmup,
                args.iterations,
            )
            request_modules = shared_installed_request_modules(
                original_attention, installed, args.concurrency
            )
            concurrency = concurrent_measure(
                request_modules, inputs, masks, args.warmup, args.iterations
            )
            memory = collect_attention_memory(installed)
            weight_bytes = int(memory["weight_bytes"])
            rows.append({
                "variant": spec["name"],
                "architecture": architecture,
                "logical_bits": spec.get("bits", 8),
                "execution": spec.get("execution", "native"),
                "lut_policy": spec.get("lut_policy", "n/a"),
                "lut_channels": spec.get("lut_channels", 0),
                "p50_ms": timing["p50_ms"],
                "p95_ms": timing["p95_ms"],
                "vs_fp16": float(timing["p50_ms"]) / float(baseline_timing["p50_ms"]),
                **accuracy(output, reference),
                "weight_bytes": weight_bytes,
                "weight_vs_fp16": weight_bytes / max(baseline_weight_bytes, 1),
                "lut_active_bytes": memory["lut_active_bytes"],
                "lut_allocated_bytes": memory["lut_allocated_bytes"],
                "workspace_bytes": memory["workspace_bytes"],
                "module_static_allocation_delta_bytes": max(0, after_construct_alloc - before_module_alloc),
                "module_peak_allocation_delta_bytes": max(0, int(torch.cuda.max_memory_allocated()) - before_module_alloc),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "concurrency": args.concurrency,
                "concurrency_p50_ms": concurrency["p50_ms"],
                "concurrency_throughput_speedup": (
                    args.concurrency * float(timing["p50_ms"])
                    / float(concurrency["p50_ms"])
                ),
                "qkv_compute_count": installed.coordinator.projection.compute_count,
            })
        except Exception as exc:
            errors.append({
                "variant": str(spec.get("name")),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print("ERROR", spec.get("name"), repr(exc))
        finally:
            if module is not None:
                del module
            if request_modules is not None:
                del request_modules
            if installed is not None:
                del installed
            gc.collect()
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "0.14.2",
        "model": args.model,
        "batch": args.batch,
        "seq": args.seq,
        "hidden": hidden,
        "gpu": torch.cuda.get_device_name(0),
        "rows": rows,
        "errors": errors,
        "scope": (
            "complete OPT self-attention module: fused QKV projections, "
            "native QK^T/softmax/AV, and replaced output projection"
        ),
        "non_modular_operations": {
            "softmax": "native PyTorch/Transformers",
            "masking": "native PyTorch/Transformers",
            "QK_and_AV": "native PyTorch/Transformers",
        },
    }
    (args.output_dir / "attention_benchmark_v014.json").write_text(
        json.dumps(payload, indent=2)
    )
    write_csv(args.output_dir / "attention_benchmark_v014.csv", rows)

    tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Variant & Latency (ms) & /FP16 & Rel. L2 & Weight/FP16 " + r"\\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"{row['variant']} & {float(row['p50_ms']):.3f} & "
            f"{float(row.get('vs_fp16', 1.0)):.2f} & "
            f"{float(row['relative_l2']):.2e} & "
            f"{float(row.get('weight_vs_fp16', 1.0)):.2f} " + r"\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Complete OPT self-attention comparison.}",
            r"\end{table}"]
    (args.output_dir / "attention_table_v014.tex").write_text("\n".join(tex) + "\n")
    print(json.dumps({
        "rows": len(rows), "errors": len(errors), "output": str(args.output_dir)
    }, indent=2))


if __name__ == "__main__":
    main()
