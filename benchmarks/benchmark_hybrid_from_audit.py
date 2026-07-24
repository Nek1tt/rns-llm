from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Callable

import torch

from rns_llm.hybrid_v010 import (
    HybridCudaOps,
    accuracy_metrics,
    benchmark_cuda_callable,
    choose_moduli,
    pad_inner_dimension,
    tensor_bytes,
)


def load_rows(samples: torch.Tensor, m: int, device: torch.device) -> torch.Tensor:
    samples = samples.float()
    if samples.shape[0] < m:
        repeats = math.ceil(m / samples.shape[0])
        samples = samples.repeat((repeats, 1))
    return samples[:m].to(device=device, dtype=torch.float32).contiguous()


def complement_indices(k: int, protected: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = torch.ones(k, dtype=torch.bool, device=device)
    mask[protected] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


def bias_or_zero(bias: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor:
    if bias is None:
        return torch.zeros(n, dtype=torch.float32, device=device)
    return bias.to(device=device, dtype=torch.float32).contiguous()


def timed_prepare(fn: Callable[[], object]) -> tuple[object, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def method_record(
    name: str,
    reference: torch.Tensor,
    output_fn: Callable[[], torch.Tensor],
    core_fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    prepared_weight_bytes: int,
    preparation_ms: float,
    metadata: dict,
) -> dict:
    torch.cuda.reset_peak_memory_stats()
    output = output_fn()
    torch.cuda.synchronize()
    record = {
        "name": name,
        "accuracy": accuracy_metrics(reference, output),
        "end_to_end": benchmark_cuda_callable(output_fn, warmup=warmup, iterations=iterations),
        "prepared_core": benchmark_cuda_callable(core_fn, warmup=warmup, iterations=iterations),
        "prepared_weight_bytes": prepared_weight_bytes,
        "weight_preparation_ms": preparation_ms,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "metadata": metadata,
    }
    return record


def benchmark_layer(
    pack_path: Path,
    m_values: list[int],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict:
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    weight = pack["weight"].float().to(device).contiguous()
    bias = bias_or_zero(pack.get("bias"), int(weight.shape[0]), device)
    samples = pack["activation_samples"]
    protected_cpu = pack["protected_indices"].long()
    n, k = map(int, weight.shape)
    if n % 4:
        raise RuntimeError(f"N={n} is not a multiple of four; this first Tensor Core prototype cannot run it")

    protected = protected_cpu.to(device)
    safe = complement_indices(k, protected, device)
    ops = HybridCudaOps(device)
    layer_result = {
        "pack": str(pack_path),
        "model": pack.get("model"),
        "layer_name": pack.get("layer_name"),
        "shape": {"n": n, "k": k},
        "protected_k": int(protected.numel()),
        "protected_ratio": float(protected.numel() / k),
        "statistics": pack.get("statistics"),
        "selected_plan": pack.get("selected_plan"),
        "shapes": [],
    }

    for m in m_values:
        x = load_rows(samples, m, device)
        x_full, w_full, full_pad = pad_inner_dimension(x, weight)
        reference = x @ weight.transpose(0, 1) + bias
        shape_result = {"m": m, "methods": [], "full_k_padded": int(x_full.shape[1])}

        # FP32 reference speed.
        fp32_fn = lambda: x @ weight.transpose(0, 1) + bias
        shape_result["methods"].append(method_record(
            "fp32", reference, fp32_fn, fp32_fn,
            warmup=warmup, iterations=iterations,
            prepared_weight_bytes=tensor_bytes(weight), preparation_ms=0.0,
            metadata={"input_dtype": "float32", "weight_dtype": "float32"},
        ))

        # FP16: e2e includes activation cast, core uses prepared activation.
        w16 = weight.half().contiguous()
        b16 = bias.half().contiguous()
        x16 = x.half().contiguous()
        fp16_e2e = lambda: (x.half() @ w16.transpose(0, 1) + b16).float()
        fp16_core = lambda: (x16 @ w16.transpose(0, 1) + b16).float()
        shape_result["methods"].append(method_record(
            "fp16", reference, fp16_e2e, fp16_core,
            warmup=warmup, iterations=iterations,
            prepared_weight_bytes=tensor_bytes(w16), preparation_ms=0.0,
            metadata={"input_cast_in_e2e": True, "accumulation": "PyTorch/cuBLAS default"},
        ))
        del w16, b16, x16

        # Native INT8 baseline.
        native_w, prep_ms = timed_prepare(lambda: ops.prepare_native_weight(w_full))
        native_a = ops.prepare_native_activation(x_full)
        native_out = torch.empty((m, n), dtype=torch.float32, device=device)
        native_acc = torch.empty((m, n), dtype=torch.int32, device=device)
        native_e2e = lambda: ops.native_e2e(x_full, native_w, bias)
        native_core = lambda: ops.native_core(native_a, native_w, bias, output=native_out, accumulators=native_acc)
        native_record = method_record(
            "native_int8", reference, native_e2e, native_core,
            warmup=warmup, iterations=iterations,
            prepared_weight_bytes=tensor_bytes(native_w.quantized_t) + tensor_bytes(native_w.scales),
            preparation_ms=prep_ms,
            metadata={"logical_bits": 8, "k_padding": full_pad, "channels": 1},
        )
        shape_result["methods"].append(native_record)
        del native_w, native_a, native_out, native_acc
        torch.cuda.empty_cache()

        # Full RNS q8/q16/q32.
        for bits in (8, 16, 32):
            try:
                rns_w, prep_ms = timed_prepare(lambda bits=bits: ops.prepare_rns_weight(w_full, bits))
                rns_a = ops.prepare_rns_activation(x_full, rns_w)
                rns_out = torch.empty((m, n), dtype=torch.float32, device=device)
                rns_acc = torch.empty((len(rns_w.moduli), m, n), dtype=torch.int32, device=device)
                e2e = lambda rns_w=rns_w: ops.rns_e2e(x_full, rns_w, bias)
                core = lambda rns_a=rns_a, rns_w=rns_w, rns_out=rns_out, rns_acc=rns_acc: ops.rns_core(
                    rns_a, rns_w, bias, output=rns_out, accumulators=rns_acc
                )
                shape_result["methods"].append(method_record(
                    f"full_rns_q{bits}", reference, e2e, core,
                    warmup=warmup, iterations=iterations,
                    prepared_weight_bytes=tensor_bytes(rns_w.residues) + tensor_bytes(rns_w.scales),
                    preparation_ms=prep_ms,
                    metadata={
                        "logical_bits": bits,
                        "channels": len(rns_w.moduli),
                        "moduli": list(rns_w.moduli),
                        "k_padding": full_pad,
                    },
                ))
                del rns_w, rns_a, rns_out, rns_acc
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                shape_result["methods"].append({
                    "name": f"full_rns_q{bits}",
                    "error": f"CUDA OOM: {exc}",
                    "metadata": {"logical_bits": bits},
                })
            gc.collect()
            torch.cuda.empty_cache()

        # Channel split. Both branches are padded independently to preserve Tensor Core alignment.
        x_safe = x.index_select(1, safe)
        w_safe = weight.index_select(1, safe)
        x_protected = x.index_select(1, protected)
        w_protected = weight.index_select(1, protected)
        x_safe_pad, w_safe_pad, safe_pad = pad_inner_dimension(x_safe, w_safe)
        x_prot_pad, w_prot_pad, prot_pad = pad_inner_dimension(x_protected, w_protected)

        main_w, main_prep_ms = timed_prepare(lambda: ops.prepare_native_weight(w_safe_pad))
        main_a = ops.prepare_native_activation(x_safe_pad)
        main_out = torch.empty((m, n), dtype=torch.float32, device=device)
        main_acc = torch.empty((m, n), dtype=torch.int32, device=device)

        # INT8 + FP16 protected baseline.
        w_prot16 = w_protected.half().contiguous()
        x_prot16 = x_protected.half().contiguous()
        def hybrid_fp16_e2e():
            xs = x.index_select(1, safe)
            if safe_pad:
                xs = torch.nn.functional.pad(xs, (0, safe_pad))
            return (
                ops.native_e2e(xs.contiguous(), main_w, None)
                + (x.index_select(1, protected).half() @ w_prot16.transpose(0, 1)).float()
                + bias
            )
        hybrid_fp16_core = lambda: (
            ops.native_core(main_a, main_w, None, output=main_out, accumulators=main_acc)
            + (x_prot16 @ w_prot16.transpose(0, 1)).float()
            + bias
        )
        shape_result["methods"].append(method_record(
            "hybrid_int8_plus_fp16", reference, hybrid_fp16_e2e, hybrid_fp16_core,
            warmup=warmup, iterations=iterations,
            prepared_weight_bytes=(
                tensor_bytes(main_w.quantized_t) + tensor_bytes(main_w.scales) + tensor_bytes(w_prot16)
            ),
            preparation_ms=main_prep_ms,
            metadata={
                "protected_k": int(protected.numel()),
                "safe_k_padding": safe_pad,
                "protected_k_padding": 0,
            },
        ))
        del w_prot16, x_prot16

        # INT8 + RNS protected variants.
        for bits in (8, 16, 32):
            try:
                prot_w, prot_prep_ms = timed_prepare(
                    lambda bits=bits: ops.prepare_rns_weight(w_prot_pad, bits)
                )
                prot_a = ops.prepare_rns_activation(x_prot_pad, prot_w)
                prot_out = torch.empty((m, n), dtype=torch.float32, device=device)
                prot_acc = torch.empty((len(prot_w.moduli), m, n), dtype=torch.int32, device=device)

                def e2e(prot_w=prot_w):
                    xs = x.index_select(1, safe)
                    xp = x.index_select(1, protected)
                    if safe_pad:
                        xs = torch.nn.functional.pad(xs, (0, safe_pad))
                    if prot_pad:
                        xp = torch.nn.functional.pad(xp, (0, prot_pad))
                    return ops.native_e2e(xs.contiguous(), main_w, None) + ops.rns_e2e(
                        xp.contiguous(), prot_w, None
                    ) + bias

                def core(prot_a=prot_a, prot_w=prot_w, prot_out=prot_out, prot_acc=prot_acc):
                    return ops.native_core(
                        main_a, main_w, None, output=main_out, accumulators=main_acc
                    ) + ops.rns_core(
                        prot_a, prot_w, None, output=prot_out, accumulators=prot_acc
                    ) + bias

                shape_result["methods"].append(method_record(
                    f"hybrid_int8_plus_rns_q{bits}", reference, e2e, core,
                    warmup=warmup, iterations=iterations,
                    prepared_weight_bytes=(
                        tensor_bytes(main_w.quantized_t) + tensor_bytes(main_w.scales)
                        + tensor_bytes(prot_w.residues) + tensor_bytes(prot_w.scales)
                    ),
                    preparation_ms=main_prep_ms + prot_prep_ms,
                    metadata={
                        "protected_k": int(protected.numel()),
                        "protected_k_padded": int(x_prot_pad.shape[1]),
                        "safe_k_padded": int(x_safe_pad.shape[1]),
                        "logical_bits": bits,
                        "channels": len(prot_w.moduli),
                        "moduli": list(prot_w.moduli),
                        "safe_k_padding": safe_pad,
                        "protected_k_padding": prot_pad,
                    },
                ))
                del prot_w, prot_a, prot_out, prot_acc
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                shape_result["methods"].append({
                    "name": f"hybrid_int8_plus_rns_q{bits}",
                    "error": f"CUDA OOM: {exc}",
                    "metadata": {"logical_bits": bits},
                })
            gc.collect()
            torch.cuda.empty_cache()

        del main_w, main_a, main_out, main_acc
        del x_safe, w_safe, x_protected, w_protected, x_safe_pad, w_safe_pad, x_prot_pad, w_prot_pad
        del x, x_full, w_full, reference
        gc.collect()
        torch.cuda.empty_cache()
        layer_result["shapes"].append(shape_result)

    del weight, bias
    gc.collect()
    torch.cuda.empty_cache()
    return layer_result


def determine_final_decision(layers: list[dict]) -> dict:
    faster_any = False
    all_accuracy_ok = True
    comparisons = []
    for layer in layers:
        for shape in layer["shapes"]:
            methods = {m["name"]: m for m in shape["methods"] if "error" not in m}
            if not {"fp16", "native_int8", "hybrid_int8_plus_rns_q16"}.issubset(methods):
                all_accuracy_ok = False
                continue
            fp16 = methods["fp16"]["end_to_end"]["p50_ms"]
            native_err = methods["native_int8"]["accuracy"]["relative_l2"]
            hybrid = methods["hybrid_int8_plus_rns_q16"]
            hybrid_time = hybrid["end_to_end"]["p50_ms"]
            hybrid_err = hybrid["accuracy"]["relative_l2"]
            faster = hybrid_time < fp16
            accuracy_ok = hybrid_err <= native_err + 1e-7
            faster_any |= faster
            all_accuracy_ok &= accuracy_ok
            comparisons.append({
                "layer_name": layer["layer_name"],
                "m": shape["m"],
                "hybrid_q16_over_fp16": hybrid_time / fp16,
                "hybrid_q16_relative_l2": hybrid_err,
                "native_int8_relative_l2": native_err,
                "faster_than_fp16": faster,
                "accuracy_not_worse_than_native_int8": accuracy_ok,
            })
    decision = "CONTINUE_KERNEL_OPTIMIZATION" if faster_any and all_accuracy_ok else "STOP_OR_REDESIGN"
    return {
        "decision": decision,
        "faster_than_fp16_on_at_least_one_shape": faster_any,
        "accuracy_not_worse_than_native_int8_everywhere": all_accuracy_ok,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    pack_paths = [
        Path(item["pack_file"])
        for item in audit["layer_decisions"]
        if item.get("pack_file")
    ][: args.max_layers]
    if not pack_paths:
        raise RuntimeError("Audit produced no layer packs")

    device = torch.device("cuda")
    layers = []
    for index, pack_path in enumerate(pack_paths, start=1):
        print(f"Benchmark layer {index}/{len(pack_paths)}: {pack_path}")
        layers.append(benchmark_layer(pack_path, args.m, args.warmup, args.iterations, device))

    payload = {
        "version": "0.10.1",
        "audit_decision": audit["decision"],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "benchmark": {
            "m": args.m,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "layers": layers,
    }
    payload["prototype_gate"] = determine_final_decision(layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["prototype_gate"], indent=2))


if __name__ == "__main__":
    main()
