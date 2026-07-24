from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Callable

import torch

from rns_llm.hybrid_v010 import accuracy_metrics, benchmark_cuda_callable, tensor_bytes
from rns_llm.prefill_v011 import PrefillLayerV011


def repeat_rows(samples: torch.Tensor, m: int, device: torch.device) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError("activation_samples must be [rows,K]")
    repeats = (m + int(samples.size(0)) - 1) // int(samples.size(0))
    return samples.repeat((repeats, 1))[:m].to(device=device, dtype=torch.float32).contiguous()


def timed(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> dict:
    return benchmark_cuda_callable(fn, warmup=warmup, iterations=iterations)


def output_and_metrics(reference: torch.Tensor, fn: Callable[[], torch.Tensor]) -> dict:
    candidate = fn()
    torch.cuda.synchronize()
    return accuracy_metrics(reference, candidate)


def load_pack_paths(audit: dict, max_layers: int) -> list[Path]:
    candidates: list[tuple[float, Path]] = []
    for layer in audit["layer_decisions"]:
        path = layer.get("pack_file")
        if not layer.get("passed") or not path:
            continue
        reduction = float(
            layer.get("evaluation", {})
            .get("selected_plan", {})
            .get("native_int8_error_reduction", 0.0)
        )
        candidates.append((reduction, Path(path)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates[:max_layers]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--m", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--workspace-mib", type=int, default=32)
    parser.add_argument("--logical-bits", type=int, default=16, choices=[8, 16])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    torch.manual_seed(11011)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    pack_paths = load_pack_paths(audit, args.max_layers)
    if not pack_paths:
        raise SystemExit("No PASS packs were found in the audit")

    payload: dict = {
        "version": "0.11.0",
        "goal": "prefill-first optimized native INT8 plus rank-k correction",
        "device": torch.cuda.get_device_name(0),
        "logical_bits": args.logical_bits,
        "m_values": args.m,
        "layers": [],
    }

    for layer_index, pack_path in enumerate(pack_paths):
        print(f"Layer {layer_index + 1}/{len(pack_paths)}: {pack_path}")
        pack = torch.load(pack_path, map_location="cpu", weights_only=False)
        prepare_start = time.perf_counter()
        layer = PrefillLayerV011.from_pack(pack, device=device, optimized_rns_bits=(8, 16))
        torch.cuda.synchronize()
        preparation_ms = (time.perf_counter() - prepare_start) * 1000.0
        weight_fp32 = pack["weight"].to(device=device, dtype=torch.float32).contiguous()
        bias = (
            torch.zeros(layer.n, dtype=torch.float32, device=device)
            if pack.get("bias") is None
            else pack["bias"].to(device=device, dtype=torch.float32).contiguous()
        )
        samples = pack["activation_samples"]
        layer_result = {
            "layer_name": layer.layer_name,
            "shape": {"n": layer.n, "k": layer.k},
            "protected": {
                "count": layer.p,
                "padded": layer.p_padded,
                "ratio": layer.p / layer.k,
                "indices": [int(v) for v in layer.protected_indices.cpu().tolist()],
            },
            "preparation_ms": preparation_ms,
            "statistics": layer.statistics,
            "shapes": [],
        }

        for m in args.m:
            print(f"  M={m}")
            x = repeat_rows(samples, m, device)
            reference = x @ weight_fp32.transpose(0, 1)
            if bias.numel():
                reference = reference + bias
            runner = layer.runner(
                m,
                logical_bits=args.logical_bits,
                workspace_bytes=args.workspace_mib * 1024 * 1024,
            )

            # Prepare all inputs exactly once for core measurements.
            runner.cast_fp16(x)
            runner.preprocess_native(x)
            runner.preprocess_hybrid(x)
            runner.main_int8_only()
            runner.rns_correction_only()
            torch.cuda.synchronize()

            fp32_out = torch.empty((m, layer.n), dtype=torch.float32, device=device)
            weight_t = weight_fp32.transpose(0, 1).contiguous()

            def fp32_core() -> torch.Tensor:
                torch.mm(x, weight_t, out=fp32_out)
                if bias.numel():
                    fp32_out.add_(bias)
                return fp32_out

            core_fns: dict[str, Callable[[], torch.Tensor]] = {
                "fp32": fp32_core,
                "fp16": runner.fp16_core,
                "native_int8": runner.native_core,
                "main_int8_only": runner.main_int8_only,
                "rns_correction_only": runner.rns_correction_only,
                "fp16_correction_only": runner.fp16_correction_only,
                "rns_fused_epilogue_only": runner.rns_fused_epilogue_only,
                "fp16_fused_epilogue_only": runner.fp16_fused_epilogue_only,
                "merge_only": runner.merge_only,
                "hybrid_rns_serial": runner.hybrid_rns_serial_core,
                "hybrid_rns_parallel": runner.hybrid_rns_parallel_core,
                "hybrid_fp16_serial": runner.hybrid_fp16_serial_core,
                "hybrid_fp16_parallel": runner.hybrid_fp16_parallel_core,
            }
            e2e_fns: dict[str, Callable[[], torch.Tensor]] = {
                "fp32": fp32_core,
                "fp16": lambda: runner.fp16_e2e(x),
                "native_int8": lambda: runner.native_e2e(x),
                "hybrid_rns_serial": lambda: runner.hybrid_rns_serial_e2e(x),
                "hybrid_rns_parallel": lambda: runner.hybrid_rns_parallel_e2e(x),
                "hybrid_fp16_serial": lambda: runner.hybrid_fp16_serial_e2e(x),
                "hybrid_fp16_parallel": lambda: runner.hybrid_fp16_parallel_e2e(x),
                "preprocess_hybrid": lambda: (runner.preprocess_hybrid(x), runner.main_q)[1],
                "preprocess_native": lambda: (runner.preprocess_native(x), runner.native_q)[1],
            }

            core = {
                name: timed(fn, args.warmup, args.iterations)
                for name, fn in core_fns.items()
            }
            e2e = {
                name: timed(fn, args.warmup, args.iterations)
                for name, fn in e2e_fns.items()
            }
            accuracy_names = (
                "fp32",
                "fp16",
                "native_int8",
                "hybrid_rns_serial",
                "hybrid_rns_parallel",
                "hybrid_fp16_serial",
                "hybrid_fp16_parallel",
            )
            accuracy = {
                name: output_and_metrics(reference, core_fns[name])
                for name in accuracy_names
            }

            fp16_core_ms = float(core["fp16"]["p50_ms"])
            fp16_e2e_ms = float(e2e["fp16"]["p50_ms"])
            native_core_ms = float(core["native_int8"]["p50_ms"])
            rns_epilogue_ms = float(core["rns_fused_epilogue_only"]["p50_ms"])
            fp16_epilogue_ms = float(core["fp16_fused_epilogue_only"]["p50_ms"])
            hybrid_core_best = min(
                float(core["hybrid_rns_serial"]["p50_ms"]),
                float(core["hybrid_rns_parallel"]["p50_ms"]),
            )
            hybrid_e2e_best = min(
                float(e2e["hybrid_rns_serial"]["p50_ms"]),
                float(e2e["hybrid_rns_parallel"]["p50_ms"]),
            )
            shape_result = {
                "m": m,
                "core": core,
                "end_to_end": e2e,
                "accuracy": accuracy,
                "storage_bytes": runner.storage_bytes(),
                "ratios": {
                    "native_int8_core_over_fp16": native_core_ms / fp16_core_ms,
                    "rns_fused_epilogue_over_fp16": rns_epilogue_ms / fp16_core_ms,
                    "fp16_fused_epilogue_over_fp16": fp16_epilogue_ms / fp16_core_ms,
                    "best_hybrid_rns_core_over_fp16": hybrid_core_best / fp16_core_ms,
                    "best_hybrid_rns_e2e_over_fp16": hybrid_e2e_best / fp16_e2e_ms,
                    "parallel_over_serial_core": float(core["hybrid_rns_parallel"]["p50_ms"])
                    / float(core["hybrid_rns_serial"]["p50_ms"]),
                    "parallel_over_serial_e2e": float(e2e["hybrid_rns_parallel"]["p50_ms"])
                    / float(e2e["hybrid_rns_serial"]["p50_ms"]),
                },
                "gates": {
                    "A_native_int8_has_30pct_headroom": native_core_ms <= 0.70 * fp16_core_ms,
                    "B_rns_correction_epilogue_within_15pct": rns_epilogue_ms <= 0.15 * fp16_core_ms,
                    "C_hybrid_e2e_beats_fp16_by_10pct": hybrid_e2e_best <= 0.90 * fp16_e2e_ms,
                    "hybrid_accuracy_better_than_native": accuracy["hybrid_rns_serial"]["relative_l2"]
                    <= accuracy["native_int8"]["relative_l2"] + 1e-7,
                    "rns_not_slower_than_fp16_correction": rns_epilogue_ms <= fp16_epilogue_ms,
                },
            }
            layer_result["shapes"].append(shape_result)

            del runner, x, reference, fp32_out, weight_t
            gc.collect()
            torch.cuda.empty_cache()

        payload["layers"].append(layer_result)
        del layer, weight_fp32, bias, samples, pack
        gc.collect()
        torch.cuda.empty_cache()

    all_shapes = [shape for layer in payload["layers"] for shape in layer["shapes"]]
    gate_a = [s for s in all_shapes if s["gates"]["A_native_int8_has_30pct_headroom"]]
    gate_b = [s for s in all_shapes if s["gates"]["B_rns_correction_epilogue_within_15pct"]]
    gate_c = [s for s in all_shapes if s["gates"]["C_hybrid_e2e_beats_fp16_by_10pct"]]
    payload["decision"] = {
        "gate_A_pass_count": len(gate_a),
        "gate_B_pass_count": len(gate_b),
        "gate_C_pass_count": len(gate_c),
        "total_shape_count": len(all_shapes),
        "verdict": (
            "CONTINUE_PREFILL_INTEGRATION"
            if gate_c
            else "CONTINUE_CORRECTION_ENGINEERING"
            if gate_a and gate_b
            else "STOP_PREFILL_ON_T4"
            if not gate_a
            else "REDESIGN_CORRECTION"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
