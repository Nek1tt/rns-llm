from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rns_llm.hybrid_v010 import choose_moduli


BUILTIN_PROMPTS = [
    "The history of numerical computing shows that representation and hardware must be designed together.",
    "Explain why activation outliers make post training quantization difficult for transformer models.",
    "A residue number system represents an integer by its residues modulo pairwise coprime moduli.",
    "Large language model inference contains a prefill phase and an autoregressive decode phase.",
    "In a matrix multiplication the inner dimension determines the length of every dot product.",
    "Scientific experiments need falsifiable criteria, reproducible measurements, and strong baselines.",
    "The quick brown fox jumps over the lazy dog while a GPU kernel processes many independent tiles.",
    "Quantization reduces memory traffic but introduces rounding error and can be sensitive to rare channels.",
] * 64


@dataclass
class BatchObservation:
    rows: int
    row_outliers: int
    threshold_count: torch.Tensor
    absmax: torch.Tensor
    sum2: torch.Tensor
    sum4: torch.Tensor
    top1_set: set[int]
    sample: torch.Tensor


@dataclass
class LayerStats:
    name: str
    module: nn.Linear
    k: int
    n: int
    threshold: float
    observations: list[BatchObservation] = field(default_factory=list)

    def update(self, x: torch.Tensor, saved_rows_per_batch: int) -> None:
        flat = x.detach().reshape(-1, x.shape[-1]).float()
        if flat.shape[1] != self.k:
            return
        abs_x = flat.abs()
        rows = int(flat.shape[0])
        threshold_count = (abs_x > self.threshold).sum(dim=0).to(torch.int64).cpu()
        absmax = abs_x.amax(dim=0).cpu()
        sum2 = (flat * flat).sum(dim=0).double().cpu()
        sum4 = ((flat * flat) ** 2).sum(dim=0).double().cpu()
        top_count = max(1, math.ceil(self.k * 0.01))
        batch_energy = (flat * flat).sum(dim=0)
        top = torch.topk(batch_energy, k=top_count, largest=True).indices.cpu().tolist()

        take = max(1, min(saved_rows_per_batch, rows))
        if take == rows:
            sample = flat
        else:
            ids = torch.linspace(0, rows - 1, steps=take, device=flat.device).round().long()
            sample = flat.index_select(0, ids)

        self.observations.append(BatchObservation(
            rows=rows,
            row_outliers=int((abs_x.amax(dim=1) > self.threshold).sum().item()),
            threshold_count=threshold_count,
            absmax=absmax,
            sum2=sum2,
            sum4=sum4,
            top1_set=set(int(v) for v in top),
            sample=sample.to(dtype=torch.float16, device="cpu"),
        ))


def select_representative_layers(model: nn.Module, max_layers: int) -> list[tuple[str, nn.Linear]]:
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or module.weight.ndim != 2:
            continue
        n, k = map(int, module.weight.shape)
        if min(n, k) < 256 or name.endswith("lm_head"):
            continue
        candidates.append((name, module))
    if not candidates:
        raise RuntimeError("No representative nn.Linear layers were found")

    # Sample projection families at multiple depths instead of taking only one middle layer.
    families = ("q_proj", "out_proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj")
    selected: list[tuple[str, nn.Linear]] = []
    used_ids: set[int] = set()
    per_family = max(1, math.ceil(max_layers / max(1, min(4, len(families)))))
    depth_fracs = torch.linspace(0.15, 0.85, steps=per_family).tolist()

    for token in families:
        matches = [item for item in candidates if token in item[0]]
        if not matches:
            continue
        for frac in depth_fracs:
            index = int(round(frac * (len(matches) - 1)))
            item = matches[index]
            if id(item[1]) in used_ids:
                continue
            selected.append(item)
            used_ids.add(id(item[1]))
            if len(selected) >= max_layers:
                return selected

    for item in candidates:
        if id(item[1]) in used_ids:
            continue
        selected.append(item)
        used_ids.add(id(item[1]))
        if len(selected) >= max_layers:
            break
    return selected[:max_layers]


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def jaccard_mean(sets: list[set[int]]) -> float:
    if len(sets) < 2:
        return 1.0
    return float(sum(jaccard(a, b) for a, b in zip(sets[:-1], sets[1:])) / (len(sets) - 1))


def aggregate(observations: list[BatchObservation]) -> dict[str, Any]:
    if not observations:
        raise ValueError("Cannot aggregate an empty observation list")
    rows = sum(item.rows for item in observations)
    return {
        "rows": rows,
        "row_outliers": sum(item.row_outliers for item in observations),
        "threshold_count": torch.stack([item.threshold_count for item in observations]).sum(dim=0),
        "absmax": torch.stack([item.absmax for item in observations]).amax(dim=0),
        "sum2": torch.stack([item.sum2 for item in observations]).sum(dim=0),
        "sum4": torch.stack([item.sum4 for item in observations]).sum(dim=0),
        "samples": torch.cat([item.sample for item in observations], dim=0),
        "top1_sets": [item.top1_set for item in observations],
    }


def risk_from_aggregate(agg: dict[str, Any], weight_l1_by_input: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    absmax = agg["absmax"].to(device)
    frequency = agg["threshold_count"].double() / max(int(agg["rows"]), 1)
    energy = agg["sum2"] / torch.clamp(agg["sum2"].sum(), min=1e-30)
    risk = absmax * weight_l1_by_input
    return risk * (1.0 + frequency.to(device).float()) * (1.0 + energy.to(device).float() * k)


def quantized_contribution(x: torch.Tensor, weight: torch.Tensor, bits: int) -> torch.Tensor:
    if x.numel() == 0 or weight.numel() == 0:
        return torch.zeros((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    qmax = float((1 << (bits - 1)) - 1)
    x_scale = torch.clamp(x.abs().amax(dim=1, keepdim=True) / qmax, min=torch.finfo(torch.float32).tiny)
    w_scale = torch.clamp(weight.abs().amax(dim=1, keepdim=True) / qmax, min=torch.finfo(torch.float32).tiny)
    xq = torch.round(x / x_scale).clamp(-qmax, qmax)
    wq = torch.round(weight / w_scale).clamp(-qmax, qmax)
    return (xq @ wq.transpose(0, 1)) * (x_scale * w_scale.transpose(0, 1))


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denom = torch.clamp(torch.linalg.vector_norm(reference), min=torch.finfo(torch.float32).eps)
    return float((torch.linalg.vector_norm(candidate - reference) / denom).item())


def evaluate_protected_plan(
    x: torch.Tensor,
    weight: torch.Tensor,
    risk: torch.Tensor,
    ratio: float,
    output_sample: int,
) -> dict[str, Any]:
    k = int(x.shape[1])
    protected_k = max(1, min(k - 1, math.ceil(k * ratio)))
    protected = torch.topk(risk, k=protected_k, largest=True).indices.sort().values
    mask = torch.ones(k, dtype=torch.bool, device=x.device)
    mask[protected] = False
    safe = torch.nonzero(mask, as_tuple=False).flatten()

    if weight.shape[0] > output_sample:
        out_ids = torch.linspace(0, weight.shape[0] - 1, steps=output_sample, device=weight.device).round().long()
        w_eval = weight.index_select(0, out_ids)
    else:
        w_eval = weight

    reference = x @ w_eval.transpose(0, 1)
    native = quantized_contribution(x, w_eval, 8)
    main = quantized_contribution(x.index_select(1, safe), w_eval.index_select(1, safe), 8)
    protected_q16 = quantized_contribution(x.index_select(1, protected), w_eval.index_select(1, protected), 16)
    hybrid = main + protected_q16
    native_error = relative_l2(reference, native)
    hybrid_error = relative_l2(reference, hybrid)
    reduction = 0.0 if native_error == 0 else (native_error - hybrid_error) / native_error

    safe_padded = ((int(safe.numel()) + 3) // 4) * 4
    protected_padded = ((protected_k + 3) // 4) * 4
    ratios: dict[str, float] = {}
    channels: dict[str, int] = {}
    for bits in (8, 16, 32):
        channel_count = len(choose_moduli(bits, protected_padded))
        channels[str(bits)] = channel_count
        ratios[str(bits)] = (safe_padded + channel_count * protected_padded) / (2.0 * k)

    return {
        "protected_indices": [int(v) for v in protected.cpu().tolist()],
        "protected_ratio": protected_k / k,
        "protected_ratio_after_padding": protected_padded / k,
        "safe_k": int(safe.numel()),
        "safe_k_padded": safe_padded,
        "protected_k": protected_k,
        "protected_k_padded": protected_padded,
        "rns_channels": channels,
        "ideal_compute_ratio_vs_fp16": ratios,
        "native_int8_relative_l2": native_error,
        "hybrid_q16_relative_l2": hybrid_error,
        "native_int8_error_reduction": reduction,
    }


def heldout_map_metrics(
    protected_indices: list[int],
    observations: list[BatchObservation],
    weight_l1_by_input: torch.Tensor,
    k: int,
    device: torch.device,
) -> dict[str, Any]:
    protected = torch.tensor(protected_indices, dtype=torch.long)
    protected_set = set(int(v) for v in protected_indices)
    total_events = 0
    protected_events = 0
    total_energy = 0.0
    protected_energy = 0.0
    local_overlaps: list[float] = []
    local_jaccards: list[float] = []

    for item in observations:
        total_events += int(item.threshold_count.sum().item())
        protected_events += int(item.threshold_count.index_select(0, protected).sum().item())
        total_energy += float(item.sum2.sum().item())
        protected_energy += float(item.sum2.index_select(0, protected).sum().item())
        agg = {
            "rows": item.rows,
            "threshold_count": item.threshold_count,
            "absmax": item.absmax,
            "sum2": item.sum2,
        }
        local_risk = risk_from_aggregate(agg, weight_l1_by_input, k, device)
        local = set(int(v) for v in torch.topk(local_risk, k=len(protected_indices)).indices.cpu().tolist())
        local_overlaps.append(len(local & protected_set) / max(1, len(protected_set)))
        local_jaccards.append(jaccard(local, protected_set))

    return {
        "heldout_outlier_events": total_events,
        "heldout_outlier_event_recall": None if total_events == 0 else protected_events / total_events,
        "heldout_protected_energy_ratio": 0.0 if total_energy == 0 else protected_energy / total_energy,
        "heldout_selected_k_overlap_mean": float(sum(local_overlaps) / max(1, len(local_overlaps))),
        "heldout_selected_k_jaccard_mean": float(sum(local_jaccards) / max(1, len(local_jaccards))),
    }


def load_calibration_texts(limit: int) -> tuple[list[str], str]:
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [str(row["text"]).strip() for row in dataset if str(row["text"]).strip()]
        if texts:
            return texts[:limit], "wikitext-2-raw-v1/validation"
    except Exception as exc:
        print("Dataset fallback:", repr(exc))
    return BUILTIN_PROMPTS[:limit], "builtin_prompts"


def build_full_token_blocks(tokenizer: Any, texts: list[str], block_count: int, sequence_length: int) -> torch.Tensor:
    token_ids: list[int] = []
    eos = tokenizer.eos_token_id
    cursor = 0
    while len(token_ids) < block_count * sequence_length:
        text = texts[cursor % len(texts)]
        cursor += 1
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            token_ids.extend(int(v) for v in ids)
        if eos is not None:
            token_ids.append(int(eos))
        if cursor > max(10_000, block_count * 1000):
            raise RuntimeError("Could not build enough non-padding calibration tokens")
    tensor = torch.tensor(token_ids[: block_count * sequence_length], dtype=torch.long)
    return tensor.reshape(block_count, sequence_length)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-1.3b")
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--absolute-threshold", type=float, default=6.0)
    parser.add_argument("--max-protected-ratio", type=float, default=0.03)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--max-pack-layers", type=int, default=4)
    parser.add_argument("--max-saved-rows", type=int, default=256)
    parser.add_argument("--output-sample", type=int, default=256)
    parser.add_argument("--seed", type=int, default=10010)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for the model audit")
    if args.calibration_batches < 4:
        raise SystemExit("At least four calibration batches are required for train/held-out evaluation")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    selected = select_representative_layers(model, args.max_layers)
    print("Representative layers:")
    states: list[LayerStats] = []
    hooks = []
    saved_rows_per_batch = max(1, math.ceil(args.max_saved_rows / args.calibration_batches))
    for name, module in selected:
        n, k = map(int, module.weight.shape)
        print(f"  {name}: N={n}, K={k}")
        state = LayerStats(name=name, module=module, k=k, n=n, threshold=args.absolute_threshold)
        states.append(state)

        def make_hook(target: LayerStats):
            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor):
                if inputs and torch.is_tensor(inputs[0]):
                    target.update(inputs[0], saved_rows_per_batch)
            return hook

        hooks.append(module.register_forward_hook(make_hook(state)))

    block_count = args.calibration_batches * args.batch_size
    texts, dataset_name = load_calibration_texts(max(64, block_count * 16))
    token_blocks = build_full_token_blocks(tokenizer, texts, block_count, args.sequence_length)
    device = torch.device("cuda")
    with torch.inference_mode():
        for batch_index in range(args.calibration_batches):
            start = batch_index * args.batch_size
            input_ids = token_blocks[start:start + args.batch_size].to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            print(f"calibration batch {batch_index + 1}/{args.calibration_batches}")

    for handle in hooks:
        handle.remove()

    ratio_candidates = sorted(set(
        min(args.max_protected_ratio, value)
        for value in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03)
        if min(args.max_protected_ratio, value) > 0
    ))
    args.pack_dir.mkdir(parents=True, exist_ok=True)
    layer_decisions = []
    passed_count = 0
    pack_count = 0

    for state in states:
        if len(state.observations) < 2:
            layer_decisions.append({
                "layer_name": state.name,
                "passed": False,
                "reasons": ["not enough activation observations"],
                "evaluation": {},
            })
            continue

        split = max(1, len(state.observations) // 2)
        train_obs = state.observations[:split]
        heldout_obs = state.observations[split:]
        train = aggregate(train_obs)
        heldout = aggregate(heldout_obs)

        module = state.module
        x_heldout = heldout["samples"].float().to(device)
        weight = module.weight.detach().float()
        weight_l1_by_input = weight.abs().sum(dim=0)
        risk = risk_from_aggregate(train, weight_l1_by_input, state.k, device)

        plans = [evaluate_protected_plan(x_heldout, weight, risk, ratio, args.output_sample) for ratio in ratio_candidates]
        acceptable = [p for p in plans if p["native_int8_error_reduction"] >= 0.20]
        selected_plan = min(acceptable, key=lambda p: p["protected_ratio"]) if acceptable else max(
            plans, key=lambda p: p["native_int8_error_reduction"]
        )

        all_agg = aggregate(state.observations)
        top_count = max(1, math.ceil(state.k * 0.01))
        top_energy = float(torch.topk(all_agg["sum2"], k=top_count).values.sum().item() / all_agg["sum2"].sum().item())
        mean2 = all_agg["sum2"] / all_agg["rows"]
        mean4 = all_agg["sum4"] / all_agg["rows"]
        kurtosis = mean4 / torch.clamp(mean2 * mean2, min=1e-30)
        map_metrics = heldout_map_metrics(
            selected_plan["protected_indices"], heldout_obs, weight_l1_by_input, state.k, device
        )
        heldout_risk = risk_from_aggregate(heldout, weight_l1_by_input, state.k, device)
        heldout_top = set(int(v) for v in torch.topk(heldout_risk, k=selected_plan["protected_k"]).indices.cpu().tolist())
        selected_set = set(selected_plan["protected_indices"])

        summary = {
            "rows": int(all_agg["rows"]),
            "train_rows": int(train["rows"]),
            "heldout_rows": int(heldout["rows"]),
            "row_outlier_rate": all_agg["row_outliers"] / all_agg["rows"],
            "channels_with_any_outlier_ratio": float((all_agg["threshold_count"] > 0).float().mean().item()),
            "top1_energy_ratio": top_energy,
            "top1_jaccard_mean": jaccard_mean(all_agg["top1_sets"]),
            "selected_map_cross_split_jaccard": jaccard(selected_set, heldout_top),
            "kurtosis_median": float(kurtosis.median().item()),
            "kurtosis_max": float(kurtosis.max().item()),
            "activation_absmax": float(all_agg["absmax"].max().item()),
            "threshold": args.absolute_threshold,
            **map_metrics,
        }

        reasons = []
        warnings = []
        if selected_plan["protected_ratio"] > args.max_protected_ratio + 1e-12:
            reasons.append("protected channel ratio exceeds gate")
        if selected_plan["native_int8_error_reduction"] < 0.20:
            reasons.append("held-out q16 protected branch reduces native INT8 error by less than 20%")
        if selected_plan["ideal_compute_ratio_vs_fp16"]["16"] > 0.90:
            reasons.append("ideal q16 hybrid compute estimate is slower than 0.9x FP16")
        event_recall = summary["heldout_outlier_event_recall"]
        if event_recall is not None and event_recall < 0.50:
            warnings.append("protected map captures less than 50% of held-out threshold events")
        if summary["heldout_selected_k_overlap_mean"] < 0.25:
            warnings.append("selected-k map overlap is low; dynamic or layer-specific protection may be needed")
        # Functional held-out accuracy is the gate. Rank/Jaccard metrics are diagnostics only;
        # a fixed top-1% set contains many non-critical filler channels when the true protected set is tiny.
        passed = not reasons
        passed_count += int(passed)

        pack_file = None
        if pack_count < args.max_pack_layers:
            protected = torch.tensor(selected_plan["protected_indices"], dtype=torch.int64)
            pack_file = args.pack_dir / f"{pack_count:02d}_{safe_name(state.name)}.pt"
            torch.save({
                "version": "0.10.1",
                "model": args.model,
                "layer_name": state.name,
                "weight": module.weight.detach().to(dtype=torch.float16, device="cpu").contiguous(),
                "bias": None if module.bias is None else module.bias.detach().to(dtype=torch.float32, device="cpu").contiguous(),
                "activation_samples": heldout["samples"].contiguous(),
                "protected_indices": protected,
                "selected_plan": selected_plan,
                "statistics": summary,
            }, pack_file)
            pack_count += 1

        layer_decisions.append({
            "layer_name": state.name,
            "shape": {"n": state.n, "k": state.k},
            "passed": passed,
            "reasons": reasons,
            "warnings": warnings,
            "statistics": summary,
            "evaluation": {"plans": plans, "selected_plan": selected_plan},
            "pack_file": None if pack_file is None else str(pack_file),
        })

    minimum_pass = math.ceil(len(states) / 2)
    decision = "PROCEED" if passed_count >= minimum_pass else "STOP"
    payload = {
        "version": "0.10.1",
        "decision": decision,
        "model": args.model,
        "dataset": dataset_name,
        "methodology": {
            "non_padding_fixed_token_blocks": True,
            "map_fit_split": "first_half_calibration_batches",
            "evaluation_split": "second_half_heldout_batches",
            "jaccard_is_diagnostic_not_gate": True,
            "saved_rows_per_batch": saved_rows_per_batch,
        },
        "calibration": {
            "batches": args.calibration_batches,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "absolute_threshold": args.absolute_threshold,
            "max_protected_ratio": args.max_protected_ratio,
        },
        "gate": {
            "minimum_passing_layers": minimum_pass,
            "passing_layers": passed_count,
            "total_layers": len(states),
        },
        "layer_decisions": layer_decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "gate": payload["gate"]}, indent=2))

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
