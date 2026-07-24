from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from rns_llm.architecture_v013 import select_plan
from rns_llm.hybrid_v010 import choose_moduli
from rns_llm.ppl_v012 import (
    CalibrationCollector,
    build_calibration_plan,
)
from rns_llm.unified_v014 import (
    collect_attention_memory,
    install_full_rns_opt_attention,
    install_hybrid_opt_attention,
    install_native_int8_opt_attention,
)


def load_dataset_text(dataset: str, config: str, split: str, samples: int) -> str:
    from datasets import load_dataset
    try:
        data = load_dataset(dataset, config, split=split)
    except Exception as first_error:
        if dataset == "Salesforce/wikitext" and config == "wikitext-2-raw-v1":
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=dataset, repo_type="dataset",
                filename=f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
            )
            data = load_dataset("parquet", data_files={split: path}, split=split)
            print("dataset fallback:", repr(first_error))
        else:
            raise
    return "\n\n".join(data["text"][:samples])


@torch.no_grad()
def perplexity(model, input_ids: torch.Tensor, *, stride: int, max_length: int) -> tuple[float, int]:
    """Sliding-window causal-LM perplexity with exact predicted-token counting."""
    model.eval()
    sequence_length = int(input_ids.size(1))
    total_nll = torch.zeros((), dtype=torch.float64, device=model.device)
    total_predicted_tokens = 0
    previous_end = 0
    for begin in range(0, sequence_length, stride):
        end = min(begin + max_length, sequence_length)
        new_tokens = end - previous_end
        window = input_ids[:, begin:end].to(model.device)
        labels = window.clone()
        labels[:, :-new_tokens] = -100
        output = model(window, labels=labels, use_cache=False)
        valid = int((labels[:, 1:] != -100).sum().item())
        if valid > 0:
            total_nll += output.loss.double() * valid
            total_predicted_tokens += valid
        previous_end = end
        if end == sequence_length:
            break
    if total_predicted_tokens <= 0:
        raise RuntimeError("no predicted tokens were evaluated")
    return float(torch.exp(total_nll / total_predicted_tokens).item()), total_predicted_tokens


def attention_targets(model, blocks: int):
    result = []
    names = dict(model.named_modules())
    for i in range(min(blocks, len(model.model.decoder.layers))):
        prefix = f"model.decoder.layers.{i}.self_attn"
        for suffix in ("q_proj", "k_proj", "v_proj", "out_proj"):
            name = f"{prefix}.{suffix}"
            module = names.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise RuntimeError(f"target not found: {name}")
            result.append((name, module))
    return result


@torch.no_grad()
def calibrate(model, input_ids: torch.Tensor, blocks: int, args) -> dict[str, Any]:
    targets = attention_targets(model, blocks)
    collector = CalibrationCollector(
        targets, threshold=args.outlier_threshold,
        max_sample_rows=args.max_sample_rows,
    )
    length = min(int(input_ids.size(1)), args.calibration_tokens)
    if length < 16:
        raise RuntimeError("not enough calibration tokens")
    midpoint = length // 2
    fit = input_ids[:, :midpoint].to(model.device)
    held = input_ids[:, midpoint:length].to(model.device)
    collector.phase = "fit"
    model(fit, use_cache=False)
    collector.phase = "heldout"
    model(held, use_cache=False)
    collector.close()
    plan = build_calibration_plan(
        model,
        collector,
        model_id=args.model,
        target_patterns=[name for name, _ in targets],
        max_protected_ratio=args.max_protected_ratio,
        min_error_reduction=args.min_error_reduction,
        output_sample=args.calibration_output_sample,
        dataset_name=args.dataset,
        calibration_config={
            "tokens": length,
            "fit_tokens": midpoint,
            "heldout_tokens": length - midpoint,
            "outlier_threshold": args.outlier_threshold,
        },
    )
    return plan


def layer_indices(plan: dict[str, Any], block: int) -> tuple[list[int], list[int]]:
    prefix = f"model.decoder.layers.{block}.self_attn"
    qkv = set()
    for suffix in ("q_proj", "k_proj", "v_proj"):
        entry = plan["layer_plans"][f"{prefix}.{suffix}"]
        qkv.update(entry["selected"]["protected_indices"])
    out = plan["layer_plans"][f"{prefix}.out_proj"]["selected"]["protected_indices"]
    return sorted(int(v) for v in qkv), sorted(int(v) for v in out)


def resolve_lut(label: str, channels: int) -> int:
    return {"none": 0, "one": min(1, channels), "two": min(2, channels), "all": channels}[label]


def install_variant(model, variant: str, lut_label: str, blocks: int, plan: dict[str, Any], args):
    installed = []
    for i in range(min(blocks, len(model.model.decoder.layers))):
        attention = model.model.decoder.layers[i].self_attn
        if variant == "native_int8":
            installed.append(install_native_int8_opt_attention(
                attention, include_out_proj=True
            ))
            continue
        if variant.startswith("full_rns_int"):
            is_v07 = variant == "full_rns_int8_v07"
            bits = 8 if is_v07 else int(variant.removeprefix("full_rns_int"))
            channels = select_plan(int(model.config.hidden_size), bits, args.moduli_policy).channels
            lut = resolve_lut(lut_label, channels)
            if is_v07 and lut > 2:
                raise ValueError("v0.7 optimized epilogue supports at most two active LUT tables")
            q8_backend = "v07" if is_v07 or (bits == 8 and lut <= 2 and args.prefer_v07_q8) else "v013"
            installed.append(install_full_rns_opt_attention(
                attention, logical_bits=bits, lut_channels=lut,
                q8_backend=q8_backend, moduli_policy=args.moduli_policy, include_out_proj=True,
            ))
            continue

        qkv_idx, out_idx = layer_indices(plan, i)
        if variant == "hybrid_fp16":
            correction = "fp16"
            bits = 16
            lut = 0
        elif variant.startswith("hybrid_rns_q"):
            correction = "rns"
            bits = int(variant.removeprefix("hybrid_rns_q"))
            p_pad = ((max(len(qkv_idx), 1) + 3) // 4) * 4
            channels = len(choose_moduli(bits, p_pad))
            lut = resolve_lut(lut_label, channels)
        else:
            raise ValueError(f"unknown variant {variant}")
        installed.append(install_hybrid_opt_attention(
            attention,
            protected_indices=qkv_idx,
            out_protected_indices=out_idx,
            correction_bits=bits,
            lut_channels=lut,
            correction=correction,
            execution=args.hybrid_execution,
            include_out_proj=True,
        ))
    return installed


def memory_summary(installed) -> dict[str, Any]:
    reports = [collect_attention_memory(item) for item in installed]
    return {
        "blocks": reports,
        "weight_bytes": sum(int(r["weight_bytes"]) for r in reports),
        "lut_active_bytes": sum(int(r["lut_active_bytes"]) for r in reports),
        "workspace_bytes": sum(int(r["workspace_bytes"]) for r in reports),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Actual-kernel PPL comparison for full-RNS and hybrid attention")
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--dataset-samples", type=int, default=256)
    ap.add_argument("--calibration-split", default="validation")
    ap.add_argument("--calibration-samples", type=int, default=128)
    ap.add_argument("--max-eval-tokens", type=int, default=8192)
    ap.add_argument("--attention-blocks", type=int, default=2)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--calibration-tokens", type=int, default=1024)
    ap.add_argument("--max-sample-rows", type=int, default=2048)
    ap.add_argument("--calibration-output-sample", type=int, default=256)
    ap.add_argument("--outlier-threshold", type=float, default=6.0)
    ap.add_argument("--max-protected-ratio", type=float, default=0.01)
    ap.add_argument("--min-error-reduction", type=float, default=0.8)
    ap.add_argument("--variants", nargs="+", default=["native_int8", "full_rns_int8", "full_rns_int16", "hybrid_fp16", "hybrid_rns_q8", "hybrid_rns_q16"])
    ap.add_argument("--lut-policies", nargs="+", choices=["none", "one", "two", "all"], default=["none", "two"])
    ap.add_argument("--hybrid-execution", choices=["serial", "parallel"], default="serial")
    ap.add_argument("--prefer-v07-q8", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--moduli-policy", choices=["dense_coprime", "large_primes", "school_small"], default="dense_coprime")
    ap.add_argument("--output-dir", type=Path, default=Path("results/v0.14.2/ppl"))
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = load_dataset_text(
        args.dataset, args.dataset_config, args.dataset_split, args.dataset_samples
    )
    calibration_text = load_dataset_text(
        args.dataset, args.dataset_config, args.calibration_split,
        args.calibration_samples,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(text, return_tensors="pt").input_ids
    calibration_ids = tokenizer(calibration_text, return_tensors="pt").input_ids
    if args.max_eval_tokens > 0:
        ids = ids[:, :args.max_eval_tokens]
    device = torch.device("cuda")

    baseline_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    plan = calibrate(
        baseline_model, calibration_ids, args.attention_blocks, args
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration_plan_v014.json").write_text(json.dumps(plan, indent=2))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    baseline_ppl, eval_tokens = perplexity(baseline_model, ids, stride=args.stride, max_length=args.max_length)
    baseline_seconds = time.perf_counter() - start
    baseline_peak = int(torch.cuda.max_memory_allocated())
    baseline_allocated = int(torch.cuda.memory_allocated())
    del baseline_model
    gc.collect(); torch.cuda.empty_cache()

    results = [{
        "variant": "fp16", "lut_policy": "n/a", "status": "PASS",
        "ppl": baseline_ppl, "relative_ppl_increase_percent": 0.0,
        "ppl_gate_pass": True, "seconds": baseline_seconds,
        "peak_memory_allocated_bytes": baseline_peak,
        "memory_allocated_bytes": baseline_allocated,
        "peak_memory_vs_fp16": 1.0,
        "memory_allocated_vs_fp16": 1.0,
        "eval_tokens": eval_tokens,
    }]

    for variant in args.variants:
        if variant == "full_rns_int8_v07":
            policies = [policy for policy in args.lut_policies if policy != "all"]
        elif variant.startswith("full_rns_") or variant.startswith("hybrid_rns_"):
            policies = args.lut_policies
        else:
            policies = ["n/a"]
        for lut_label in policies:
            print(f"PPL variant={variant} LUT={lut_label}")
            model = None
            try:
                model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
                installed = install_variant(model, variant, "none" if lut_label == "n/a" else lut_label, args.attention_blocks, plan, args)
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                ppl, tokens = perplexity(model, ids, stride=args.stride, max_length=args.max_length)
                seconds = time.perf_counter() - start
                increase = 100.0 * (ppl / baseline_ppl - 1.0)
                calls = sum(
                    int(item.coordinator.projection.compute_count)
                    + int(getattr(item.out_projection, "calls", 0))
                    for item in installed
                )
                if calls <= 0:
                    raise RuntimeError("installed RNS/hybrid attention path was not executed")
                results.append({
                    "variant": variant, "lut_policy": lut_label, "status": "PASS" if increase < 5.0 else "FAIL",
                    "ppl": ppl, "relative_ppl_increase_percent": increase,
                    "ppl_gate_pass": bool(increase < 5.0), "seconds": seconds,
                    "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "peak_memory_vs_fp16": int(torch.cuda.max_memory_allocated()) / max(baseline_peak, 1),
                    "memory_allocated_vs_fp16": int(torch.cuda.memory_allocated()) / max(baseline_allocated, 1),
                    "eval_tokens": tokens, "kernel_path_calls": calls,
                    "memory": memory_summary(installed),
                })
            except Exception as exc:
                results.append({
                    "variant": variant, "lut_policy": lut_label, "status": "ERROR",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "ppl": None, "relative_ppl_increase_percent": None,
                    "ppl_gate_pass": False,
                })
                print("ERROR", variant, lut_label, repr(exc))
            finally:
                if model is not None: del model
                gc.collect(); torch.cuda.empty_cache()

    payload = {
        "version": "0.14.2", "model": args.model, "dataset": args.dataset,
        "evaluation_split": args.dataset_split,
        "calibration_split": args.calibration_split,
        "attention_blocks": args.attention_blocks, "eval_tokens": eval_tokens,
        "baseline_ppl": baseline_ppl, "gate": "relative PPL increase < 5%",
        "actual_cuda_kernels": True,
        "scope": "QKV and output projections replaced; QK^T, softmax, AV, LayerNorm and MLP remain native",
        "results": results,
    }
    (args.output_dir / "ppl_unified_v014.json").write_text(json.dumps(payload, indent=2))
    flat_rows = []
    for row in results:
        flat_rows.append({
            key: row.get(key) for key in (
                "variant", "lut_policy", "status", "ppl",
                "relative_ppl_increase_percent", "ppl_gate_pass", "seconds",
                "eval_tokens", "kernel_path_calls",
                "memory_allocated_bytes", "peak_memory_allocated_bytes",
                "memory_allocated_vs_fp16", "peak_memory_vs_fp16",
            )
        })
    with (args.output_dir / "ppl_unified_v014.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader(); writer.writerows(flat_rows)
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{llrrc}", r"\toprule",
        r"Architecture & LUT & PPL & $\Delta$PPL (\%) & Gate " + r"\\",
        r"\midrule",
    ]
    for row in results:
        ppl_text = "--" if row.get("ppl") is None else f"{row['ppl']:.3f}"
        inc = row.get("relative_ppl_increase_percent")
        inc_text = "--" if inc is None else f"{inc:.2f}"
        gate = "PASS" if row.get("ppl_gate_pass") else row.get("status", "FAIL")
        tex.append(
            f"{row['variant']} & {row['lut_policy']} & {ppl_text} & "
            f"{inc_text} & {gate} " + r"\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Perplexity of actual CUDA attention-projection variants.}",
            r"\end{table}"]
    (args.output_dir / "ppl_table_v014.tex").write_text("\n".join(tex) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
