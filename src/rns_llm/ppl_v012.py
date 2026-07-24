from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from rns_llm.hybrid_v010 import choose_moduli, modulus_product


DEFAULT_TARGET_PATTERNS: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "fc1",
    "fc2",
)
RATIO_CANDIDATES: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03)
SUPPORTED_VARIANTS: tuple[str, ...] = (
    "fp16",
    "native_int8",
    "hybrid_fp16",
    "hybrid_rns_q16",
)


def quant_max(bits: int) -> int:
    if bits not in {8, 16, 32}:
        raise ValueError(f"unsupported logical precision q{bits}")
    return (1 << (bits - 1)) - 1


def _tiny(dtype: torch.dtype = torch.float32) -> float:
    return float(torch.finfo(dtype).tiny)


def qdq_rows(x: torch.Tensor, qmax: int) -> torch.Tensor:
    """Symmetric per-row fake quantization, returned in float32."""
    if x.ndim != 2:
        raise ValueError(f"qdq_rows expects rank-2 input, got {tuple(x.shape)}")
    xf = x.float()
    scales = torch.clamp(xf.abs().amax(dim=1, keepdim=True) / float(qmax), min=_tiny())
    q = torch.round(xf / scales).clamp(-qmax, qmax)
    return q * scales


def qdq_weight_per_output(weight: torch.Tensor, qmax: int) -> torch.Tensor:
    """Symmetric per-output-channel fake quantization, returned in float32."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank-2, got {tuple(weight.shape)}")
    wf = weight.float()
    scales = torch.clamp(wf.abs().amax(dim=1, keepdim=True) / float(qmax), min=_tiny())
    q = torch.round(wf / scales).clamp(-qmax, qmax)
    return q * scales


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = torch.clamp(
        torch.linalg.vector_norm(reference.float()),
        min=torch.finfo(torch.float32).eps,
    )
    return float(
        (torch.linalg.vector_norm(candidate.float() - reference.float()) / denominator).item()
    )


def _safe_qdq_rows(x: torch.Tensor, protected: torch.Tensor, qmax: int = 127) -> torch.Tensor:
    """Match v0.11 preprocessing: scale over safe channels and zero protected ones."""
    xf = x.float()
    safe_abs = xf.abs().clone()
    safe_abs.index_fill_(1, protected, 0.0)
    scales = torch.clamp(safe_abs.amax(dim=1, keepdim=True) / float(qmax), min=_tiny())
    q = torch.round(xf / scales).clamp(-qmax, qmax)
    q.index_fill_(1, protected, 0.0)
    return q * scales


def native_linear_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(qdq_rows(x, 127), qdq_weight_per_output(weight, 127))


def hybrid_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    protected: torch.Tensor,
    *,
    correction: str,
) -> torch.Tensor:
    if protected.numel() == 0:
        return native_linear_reference(x, weight)
    safe_weight = weight.float().clone()
    safe_weight.index_fill_(1, protected, 0.0)
    main = F.linear(
        _safe_qdq_rows(x, protected, 127),
        qdq_weight_per_output(safe_weight, 127),
    )
    x_protected = x.float().index_select(1, protected)
    w_protected = weight.float().index_select(1, protected)
    if correction == "fp16":
        corr = F.linear(x_protected, w_protected)
    elif correction == "q16":
        corr = F.linear(
            qdq_rows(x_protected, quant_max(16)),
            qdq_weight_per_output(w_protected, quant_max(16)),
        )
    else:
        raise ValueError(f"unknown correction mode {correction!r}")
    return main + corr


@dataclass
class PhaseStats:
    rows: int = 0
    absmax: torch.Tensor | None = None
    threshold_count: torch.Tensor | None = None
    sum2: torch.Tensor | None = None
    samples: list[torch.Tensor] = field(default_factory=list)
    sampled_rows: int = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, *, threshold: float, max_sample_rows: int) -> None:
        if x.ndim < 2:
            return
        k = int(x.shape[-1])
        flat = x.detach().reshape(-1, k).float()
        if flat.numel() == 0:
            return
        abs_flat = flat.abs()
        batch_absmax = abs_flat.amax(dim=0).cpu()
        batch_threshold = (abs_flat > threshold).sum(dim=0).cpu().to(torch.int64)
        batch_sum2 = (flat * flat).sum(dim=0).cpu().to(torch.float64)
        self.rows += int(flat.shape[0])
        self.absmax = batch_absmax if self.absmax is None else torch.maximum(self.absmax, batch_absmax)
        self.threshold_count = (
            batch_threshold
            if self.threshold_count is None
            else self.threshold_count + batch_threshold
        )
        self.sum2 = batch_sum2 if self.sum2 is None else self.sum2 + batch_sum2

        remaining = max(0, int(max_sample_rows) - self.sampled_rows)
        if remaining:
            take = min(remaining, int(flat.shape[0]))
            if take == int(flat.shape[0]):
                sample = flat
            else:
                ids = torch.linspace(
                    0,
                    int(flat.shape[0]) - 1,
                    steps=take,
                    device=flat.device,
                ).round().long()
                sample = flat.index_select(0, ids)
            self.samples.append(sample.to(device="cpu", dtype=torch.float16).contiguous())
            self.sampled_rows += take

    def sample_tensor(self) -> torch.Tensor:
        if not self.samples:
            if self.absmax is None:
                return torch.empty((0, 0), dtype=torch.float16)
            return torch.empty((0, int(self.absmax.numel())), dtype=torch.float16)
        return torch.cat(self.samples, dim=0)

    def serializable_summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "sampled_rows": self.sampled_rows,
            "activation_absmax": None if self.absmax is None else float(self.absmax.max().item()),
            "channels_with_threshold_events": None
            if self.threshold_count is None
            else int((self.threshold_count > 0).sum().item()),
        }


@dataclass
class LayerCalibration:
    name: str
    in_features: int
    out_features: int
    fit: PhaseStats = field(default_factory=PhaseStats)
    heldout: PhaseStats = field(default_factory=PhaseStats)


class CalibrationCollector:
    def __init__(
        self,
        modules: Sequence[tuple[str, nn.Linear]],
        *,
        threshold: float,
        max_sample_rows: int,
    ) -> None:
        self.threshold = float(threshold)
        self.max_sample_rows = int(max_sample_rows)
        self.phase = "fit"
        self.layers: dict[str, LayerCalibration] = {
            name: LayerCalibration(name, module.in_features, module.out_features)
            for name, module in modules
        }
        self._handles: list[Any] = []
        for name, module in modules:
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: Any) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            target = self.layers[name].fit if self.phase == "fit" else self.layers[name].heldout
            target.update(
                inputs[0],
                threshold=self.threshold,
                max_sample_rows=self.max_sample_rows,
            )

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def matches_target(name: str, patterns: Sequence[str]) -> bool:
    return any(name == pattern or name.endswith("." + pattern) or name.endswith(pattern) for pattern in patterns)


def select_target_linears(
    model: nn.Module, patterns: Sequence[str]
) -> list[tuple[str, nn.Linear]]:
    result: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and matches_target(name, patterns):
            result.append((name, module))
    return result


def _risk_from_fit(stats: PhaseStats, weight: torch.Tensor) -> torch.Tensor:
    if stats.absmax is None or stats.threshold_count is None or stats.sum2 is None:
        raise ValueError("calibration statistics are incomplete")
    device = weight.device
    absmax = stats.absmax.to(device=device, dtype=torch.float32)
    frequency = stats.threshold_count.to(device=device, dtype=torch.float32) / max(stats.rows, 1)
    energy = stats.sum2.to(device=device, dtype=torch.float32)
    energy = energy / torch.clamp(energy.sum(), min=1e-30)
    weight_l1 = weight.float().abs().sum(dim=0)
    k = int(weight.shape[1])
    return absmax * weight_l1 * (1.0 + frequency) * (1.0 + energy * k)


def _evaluate_ratio(
    x: torch.Tensor,
    weight: torch.Tensor,
    risk: torch.Tensor,
    ratio: float,
    *,
    output_sample: int,
) -> dict[str, Any]:
    k = int(weight.shape[1])
    protected_k = max(1, min(k - 1, math.ceil(k * float(ratio))))
    protected = torch.topk(risk, k=protected_k, largest=True).indices.sort().values
    if int(weight.shape[0]) > output_sample:
        output_ids = torch.linspace(
            0,
            int(weight.shape[0]) - 1,
            steps=output_sample,
            device=weight.device,
        ).round().long()
        w_eval = weight.index_select(0, output_ids)
    else:
        w_eval = weight
    reference = F.linear(x.float(), w_eval.float())
    native = native_linear_reference(x, w_eval)
    hybrid = hybrid_linear_reference(x, w_eval, protected, correction="q16")
    native_error = relative_l2(reference, native)
    hybrid_error = relative_l2(reference, hybrid)
    reduction = 0.0 if native_error == 0.0 else (native_error - hybrid_error) / native_error
    p_padded = ((protected_k + 3) // 4) * 4
    moduli = choose_moduli(16, p_padded)
    return {
        "ratio": protected_k / k,
        "protected_k": protected_k,
        "protected_k_padded": p_padded,
        "protected_indices": [int(v) for v in protected.detach().cpu().tolist()],
        "native_int8_relative_l2": native_error,
        "hybrid_q16_relative_l2": hybrid_error,
        "native_int8_error_reduction": reduction,
        "q16_moduli": [int(v) for v in moduli],
        "q16_channels": len(moduli),
        "q16_modulus_product": str(modulus_product(moduli)),
    }


@torch.no_grad()
def build_calibration_plan(
    model: nn.Module,
    collector: CalibrationCollector,
    *,
    model_id: str,
    target_patterns: Sequence[str],
    max_protected_ratio: float,
    min_error_reduction: float,
    output_sample: int,
    dataset_name: str,
    calibration_config: dict[str, Any],
) -> dict[str, Any]:
    modules = dict(select_target_linears(model, target_patterns))
    candidates = sorted(
        set(
            min(float(max_protected_ratio), value)
            for value in RATIO_CANDIDATES
            if min(float(max_protected_ratio), value) > 0
        )
    )
    if not candidates:
        raise ValueError("max_protected_ratio is too small")
    layer_plans: dict[str, Any] = {}
    passed = 0
    for index, (name, layer_stats) in enumerate(collector.layers.items(), start=1):
        module = modules[name]
        samples = layer_stats.heldout.sample_tensor()
        if samples.numel() == 0:
            raise RuntimeError(f"no held-out samples collected for {name}")
        device = module.weight.device
        x = samples.to(device=device, dtype=torch.float32)
        weight = module.weight.detach().float()
        risk = _risk_from_fit(layer_stats.fit, weight)
        evaluations = [
            _evaluate_ratio(
                x,
                weight,
                risk,
                ratio,
                output_sample=output_sample,
            )
            for ratio in candidates
        ]
        acceptable = [
            item
            for item in evaluations
            if item["native_int8_error_reduction"] >= float(min_error_reduction)
        ]
        selected = (
            min(acceptable, key=lambda item: item["ratio"])
            if acceptable
            else max(evaluations, key=lambda item: item["native_int8_error_reduction"])
        )
        local_pass = bool(
            selected["ratio"] <= float(max_protected_ratio) + 1e-12
            and selected["native_int8_error_reduction"] >= float(min_error_reduction)
        )
        passed += int(local_pass)
        layer_plans[name] = {
            "in_features": layer_stats.in_features,
            "out_features": layer_stats.out_features,
            "passed_local_gate": local_pass,
            "selected": selected,
            "all_candidates": evaluations,
            "fit_stats": layer_stats.fit.serializable_summary(),
            "heldout_stats": layer_stats.heldout.serializable_summary(),
        }
        print(
            f"plan {index}/{len(collector.layers)} {name}: "
            f"P={selected['protected_k']} reduction={selected['native_int8_error_reduction']:.3f} "
            f"{'PASS' if local_pass else 'BEST_EFFORT'}"
        )
        del x, weight, risk
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "version": "0.12.0",
        "model": model_id,
        "dataset": dataset_name,
        "target_patterns": list(target_patterns),
        "methodology": {
            "calibration_split": "first half fit / second half held-out",
            "risk_score": "activation_absmax * weight_L1 * (1+frequency) * (1+normalized_energy*K)",
            "selection": "smallest candidate reaching local error-reduction gate, otherwise best effort",
            "ppl_quantization": "pure-PyTorch fake quantization matching v0.11 scales; q16 correction is ideal exact-RNS reconstruction",
        },
        "calibration": calibration_config,
        "local_gate": {
            "minimum_error_reduction": float(min_error_reduction),
            "max_protected_ratio": float(max_protected_ratio),
            "passing_layers": passed,
            "total_layers": len(layer_plans),
        },
        "layer_plans": layer_plans,
    }


def _get_parent_and_child(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    pieces = name.split(".")
    parent = model
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    return parent, pieces[-1]


class SimulatedQuantLinear(nn.Module):
    """Quality-only simulation of v0.11 linear arithmetic.

    The module stores integer-valued quantized weights in floating tensors so
    standard PyTorch GEMM can be used without the custom fixed-shape CUDA
    extension. Scales are applied after the dot product, as in the v0.11
    epilogue. This is a model-quality simulation, not a latency implementation.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        variant: str,
        protected_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if variant not in {"native_int8", "hybrid_fp16", "hybrid_rns_q16"}:
            raise ValueError(f"unsupported simulated variant {variant!r}")
        self.variant = variant
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        source_dtype = linear.weight.dtype
        source_device = linear.weight.device
        weight = linear.weight.detach().float()
        bias = None if linear.bias is None else linear.bias.detach().float()
        self.register_buffer("bias_value", bias)

        def quantize_weight(values: torch.Tensor, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
            scales = torch.clamp(
                values.abs().amax(dim=1, keepdim=True) / float(qmax), min=_tiny()
            )
            quantized = torch.round(values / scales).clamp(-qmax, qmax)
            return quantized, scales.squeeze(1)

        if variant == "native_int8":
            weight_q, weight_scale = quantize_weight(weight, 127)
            self.register_buffer(
                "main_weight_q",
                weight_q.to(device=source_device, dtype=source_dtype),
            )
            self.register_buffer(
                "main_weight_scale", weight_scale.to(device=source_device, dtype=torch.float32)
            )
            self.register_buffer(
                "protected_indices", torch.empty(0, dtype=torch.long, device=source_device)
            )
            self.register_buffer("correction_weight", torch.empty(0, device=source_device))
            self.register_buffer("correction_weight_scale", torch.empty(0, device=source_device))
            return

        if not protected_indices:
            raise ValueError("hybrid variants require non-empty protected_indices")
        protected = torch.tensor(
            [int(v) for v in protected_indices],
            dtype=torch.long,
            device=source_device,
        )
        if int(protected.min().item()) < 0 or int(protected.max().item()) >= self.in_features:
            raise ValueError("protected index is out of bounds")
        self.register_buffer("protected_indices", protected)
        safe_weight = weight.clone()
        safe_weight.index_fill_(1, protected, 0.0)
        main_q, main_scale = quantize_weight(safe_weight, 127)
        self.register_buffer(
            "main_weight_q", main_q.to(device=source_device, dtype=source_dtype)
        )
        self.register_buffer(
            "main_weight_scale", main_scale.to(device=source_device, dtype=torch.float32)
        )
        protected_weight = weight.index_select(1, protected)
        if variant == "hybrid_fp16":
            self.register_buffer(
                "correction_weight",
                protected_weight.to(device=source_device, dtype=torch.float32),
            )
            self.register_buffer(
                "correction_weight_scale", torch.empty(0, device=source_device)
            )
        else:
            correction_q, correction_scale = quantize_weight(
                protected_weight, quant_max(16)
            )
            self.register_buffer(
                "correction_weight",
                correction_q.to(device=source_device, dtype=torch.float32),
            )
            self.register_buffer(
                "correction_weight_scale",
                correction_scale.to(device=source_device, dtype=torch.float32),
            )

    @staticmethod
    def _quantize_rows_for_gemm(
        x: torch.Tensor,
        qmax: int,
        protected: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        abs_values = xf.abs()
        if protected is not None and protected.numel():
            abs_values = abs_values.clone()
            abs_values.index_fill_(1, protected, 0.0)
        scales = torch.clamp(
            abs_values.amax(dim=1, keepdim=True) / float(qmax), min=_tiny()
        )
        quantized = torch.round(xf / scales).clamp(-qmax, qmax)
        if protected is not None and protected.numel():
            quantized.index_fill_(1, protected, 0.0)
        return quantized, scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if self.variant == "native_int8":
            x_q, x_scale = self._quantize_rows_for_gemm(x2, 127)
            accumulator = F.linear(
                x_q.to(dtype=self.main_weight_q.dtype), self.main_weight_q, None
            ).float()
            output = accumulator * x_scale * self.main_weight_scale.unsqueeze(0)
        else:
            x_q, x_scale = self._quantize_rows_for_gemm(
                x2, 127, self.protected_indices
            )
            accumulator = F.linear(
                x_q.to(dtype=self.main_weight_q.dtype), self.main_weight_q, None
            ).float()
            output = accumulator * x_scale * self.main_weight_scale.unsqueeze(0)
            x_protected = x2.float().index_select(1, self.protected_indices)
            if self.variant == "hybrid_rns_q16":
                protected_q, protected_scale = self._quantize_rows_for_gemm(
                    x_protected, quant_max(16)
                )
                correction_acc = F.linear(
                    protected_q, self.correction_weight, None
                )
                correction = (
                    correction_acc
                    * protected_scale
                    * self.correction_weight_scale.unsqueeze(0)
                )
            else:
                correction = F.linear(x_protected, self.correction_weight, None)
            output = output + correction
        if self.bias_value is not None:
            output = output + self.bias_value
        return output.to(dtype=x.dtype).reshape(*original_shape, self.out_features)


@torch.no_grad()
def apply_simulated_variant(
    model: nn.Module,
    plan: dict[str, Any],
    *,
    variant: str,
    fallback: str = "best_effort",
) -> dict[str, Any]:
    if variant == "fp16":
        return {"replaced_layers": 0, "fallback_layers": 0}
    if fallback not in {"best_effort", "fp16", "native_int8"}:
        raise ValueError(f"unknown fallback {fallback!r}")
    modules = dict(model.named_modules())
    replaced = 0
    fallback_layers = 0
    skipped: list[str] = []
    for name, layer_plan in plan["layer_plans"].items():
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            skipped.append(name)
            continue
        actual_variant = variant
        protected = layer_plan["selected"]["protected_indices"]
        if variant.startswith("hybrid") and not layer_plan["passed_local_gate"]:
            fallback_layers += 1
            if fallback == "fp16":
                continue
            if fallback == "native_int8":
                actual_variant = "native_int8"
        replacement = SimulatedQuantLinear(
            module,
            variant=actual_variant,
            protected_indices=protected if actual_variant.startswith("hybrid") else None,
        )
        parent, child = _get_parent_and_child(model, name)
        setattr(parent, child, replacement)
        replaced += 1
    return {
        "replaced_layers": replaced,
        "fallback_layers": fallback_layers,
        "skipped_layers": skipped,
        "fallback_policy": fallback,
    }


def _garner_signed(residues: Sequence[int], moduli: Sequence[int]) -> int:
    value = 0
    prefix = 1
    for residue, modulus in zip(residues, moduli):
        correction = ((int(residue) - value) % modulus) * pow(prefix % modulus, -1, modulus)
        correction %= modulus
        value += prefix * correction
        prefix *= modulus
    if value > prefix // 2:
        value -= prefix
    return int(value)


def _quantize_vector(values: torch.Tensor, qmax: int) -> tuple[torch.Tensor, float]:
    vf = values.float()
    scale = max(float(vf.abs().max().item()) / float(qmax), _tiny())
    q = torch.round(vf / scale).clamp(-qmax, qmax).to(torch.int64)
    return q, scale


@torch.no_grad()
def verify_ideal_rns_equivalence(
    model: nn.Module,
    collector: CalibrationCollector,
    plan: dict[str, Any],
    *,
    max_layers: int = 4,
    rows_per_layer: int = 2,
    outputs_per_layer: int = 4,
) -> dict[str, Any]:
    modules = dict(model.named_modules())
    checks = 0
    failures: list[dict[str, Any]] = []
    checked_layers: list[str] = []
    for name, layer_plan in plan["layer_plans"].items():
        if len(checked_layers) >= max_layers:
            break
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            continue
        samples = collector.layers[name].heldout.sample_tensor()
        if samples.numel() == 0:
            continue
        protected = torch.tensor(layer_plan["selected"]["protected_indices"], dtype=torch.long)
        p_padded = int(layer_plan["selected"]["protected_k_padded"])
        moduli = tuple(int(v) for v in choose_moduli(16, p_padded))
        x = samples[:rows_per_layer].float().index_select(1, protected)
        w = module.weight.detach().cpu().float()[:outputs_per_layer].index_select(1, protected)
        x_quantized = [_quantize_vector(row, quant_max(16))[0] for row in x]
        w_quantized = [_quantize_vector(row, quant_max(16))[0] for row in w]
        for row_index, aq in enumerate(x_quantized):
            for output_index, wq in enumerate(w_quantized):
                exact = int(torch.dot(aq, wq).item())
                residues = [exact % modulus for modulus in moduli]
                reconstructed = _garner_signed(residues, moduli)
                checks += 1
                if reconstructed != exact:
                    failures.append(
                        {
                            "layer": name,
                            "row": row_index,
                            "output": output_index,
                            "exact": exact,
                            "reconstructed": reconstructed,
                            "moduli": list(moduli),
                        }
                    )
        checked_layers.append(name)
    return {
        "checked_layers": checked_layers,
        "checks": checks,
        "failures": failures,
        "passed": not failures and checks > 0,
        "interpretation": "A passing check confirms that ideal q16 fake-quant correction and exact RNS+CRT produce the same integer dot products for sampled cases.",
    }


def tokenize_dataset(
    tokenizer: Any,
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
) -> torch.Tensor:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = [str(value) for value in dataset["text"]]
    encoded = tokenizer("\n\n".join(texts), return_tensors="pt")
    return encoded.input_ids.contiguous()


def build_calibration_blocks(
    token_ids: torch.Tensor,
    *,
    block_count: int,
    sequence_length: int,
) -> torch.Tensor:
    required = int(block_count) * int(sequence_length)
    flat = token_ids.reshape(-1)
    if int(flat.numel()) < required:
        repeats = math.ceil(required / max(int(flat.numel()), 1))
        flat = flat.repeat(repeats)
    return flat[:required].reshape(block_count, sequence_length).contiguous()


@torch.no_grad()
def evaluate_sliding_window_ppl(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    context_length: int,
    stride: int,
    max_eval_tokens: int,
) -> dict[str, Any]:
    if max_eval_tokens > 0:
        token_ids = token_ids[:, :max_eval_tokens]
    sequence_length = int(token_ids.shape[1])
    if sequence_length < 2:
        raise ValueError("at least two evaluation tokens are required")
    context_length = min(int(context_length), sequence_length)
    if stride <= 0 or stride > context_length:
        raise ValueError("stride must be in [1, context_length]")
    nll_sum = torch.zeros((), dtype=torch.float64, device=device)
    n_tokens = 0
    previous_end = 0
    windows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    begin = 0
    window_index = 0
    while begin < sequence_length:
        end = min(begin + context_length, sequence_length)
        target_length = end - previous_end
        input_ids = token_ids[:, begin:end].to(device=device, non_blocking=True)
        target_ids = input_ids.clone()
        target_ids[:, :-target_length] = -100
        output = model(input_ids=input_ids, labels=target_ids, use_cache=False)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError(
                f"non-finite language-model loss in window {window_index}: {output.loss}"
            )
        valid_tokens = int((target_ids != -100).sum().item())
        loss_tokens = valid_tokens - int(target_ids.shape[0])
        if loss_tokens > 0:
            nll = output.loss.detach().double()
            nll_sum += nll * loss_tokens
            n_tokens += loss_tokens
            windows.append(
                {
                    "window": window_index,
                    "begin": begin,
                    "end": end,
                    "loss_tokens": loss_tokens,
                    "mean_nll": float(nll.item()),
                }
            )
        previous_end = end
        window_index += 1
        print(
            f"PPL window {window_index}: tokens {begin}:{end}, "
            f"running evaluated tokens={n_tokens}"
        )
        if end == sequence_length:
            break
        begin += stride
    elapsed = time.perf_counter() - start_time
    average_nll = nll_sum / max(n_tokens, 1)
    ppl = torch.exp(average_nll)
    return {
        "ppl": float(ppl.item()),
        "average_nll": float(average_nll.item()),
        "evaluated_loss_tokens": n_tokens,
        "input_tokens": sequence_length,
        "context_length": context_length,
        "stride": stride,
        "windows": windows,
        "elapsed_seconds": elapsed,
    }


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    try:
        import transformers

        snapshot["transformers"] = transformers.__version__
    except Exception:
        snapshot["transformers"] = None
    try:
        import datasets

        snapshot["datasets"] = datasets.__version__
    except Exception:
        snapshot["datasets"] = None
    return snapshot


def write_paper_artifacts(summary: dict[str, Any], paper_dir: Path) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    results = summary.get("results", {})
    baseline = results.get("fp16", {}).get("ppl")
    rows: list[str] = []
    for variant in SUPPORTED_VARIANTS:
        item = results.get(variant)
        if not item or "ppl" not in item:
            continue
        increase = item.get("relative_ppl_increase_percent")
        gate = item.get("ppl_gate_pass")
        rows.append(
            f"{variant.replace('_', r'\_')} & {item['ppl']:.4f} & "
            f"{('--' if increase is None else f'{increase:+.2f}\\%')} & "
            f"{('--' if gate is None else ('PASS' if gate else 'FAIL'))} \\\\"
        )
    table = "\n".join(
        [
            r"\begin{table}[!tbh]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrc}",
            r"\toprule",
            r"Variant & PPL & Relative increase & 5\% gate \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{WikiText-2 perplexity under the v0.12 full-model fake-quantization protocol.}",
            r"\label{tab:ppl-v012}",
            r"\end{table}",
            "",
        ]
    )
    (paper_dir / "ppl_results_table.tex").write_text(table, encoding="utf-8")

    rns = results.get("hybrid_rns_q16")
    if baseline is not None and rns is not None and "ppl" in rns:
        increase = rns.get("relative_ppl_increase_percent")
        verdict = "passed" if rns.get("ppl_gate_pass") else "failed"
        paragraph_text = (
            f"The FP16 baseline achieved PPL {baseline:.4f}. The hybrid RNS q16 "
            f"simulation achieved PPL {rns['ppl']:.4f}, corresponding to a relative "
            f"increase of {increase:.2f}%. Therefore, the predefined PPL increase "
            f"below 5% gate {verdict}. This experiment evaluates model quality only; "
            f"it does not imply an end-to-end latency speedup.\n"
        )
        paragraph_tex = paragraph_text.replace("%", r"\%")
    else:
        paragraph_text = "PPL results are incomplete; no paper claim should be made.\n"
        paragraph_tex = paragraph_text
    (paper_dir / "ppl_results_paragraph.txt").write_text(paragraph_text, encoding="utf-8")
    (paper_dir / "ppl_results_paragraph.tex").write_text(paragraph_tex, encoding="utf-8")

    def value_or_na(value: Any, fmt: str = "{}") -> str:
        return "N/A" if value is None else fmt.format(value)

    macros = [
        r"% Auto-generated by evaluate_ppl_v012.py. Do not edit manually.",
        rf"\newcommand{{\PPLBaseline}}{{{value_or_na(baseline, '{:.4f}')}}}",
    ]
    if rns is not None and "ppl" in rns:
        macros.extend(
            [
                rf"\newcommand{{\PPLRNS}}{{{rns['ppl']:.4f}}}",
                rf"\newcommand{{\PPLRNSIncrease}}{{{rns['relative_ppl_increase_percent']:.2f}\%}}",
                rf"\newcommand{{\PPLRNSGate}}{{{'PASS' if rns['ppl_gate_pass'] else 'FAIL'}}}",
            ]
        )
    (paper_dir / "ppl_result_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def finalize_summary(summary: dict[str, Any], *, gate_threshold_percent: float = 5.0) -> dict[str, Any]:
    results = summary.setdefault("results", {})
    baseline = results.get("fp16", {}).get("ppl")
    if baseline is None:
        return summary
    for variant, item in results.items():
        if "ppl" not in item:
            continue
        if variant == "fp16":
            item["relative_ppl_increase_percent"] = 0.0
            item["ppl_gate_pass"] = True
            continue
        increase = 100.0 * (float(item["ppl"]) / float(baseline) - 1.0)
        item["relative_ppl_increase_percent"] = increase
        item["ppl_gate_pass"] = bool(increase < float(gate_threshold_percent))
    rns = results.get("hybrid_rns_q16")
    summary["ppl_requirement"] = {
        "definition": "100 * (PPL_optimized / PPL_FP16 - 1)",
        "threshold_percent": float(gate_threshold_percent),
        "variant": "hybrid_rns_q16",
        "status": "NOT_RUN"
        if rns is None
        else ("ERROR" if "ppl" not in rns else ("PASS" if rns["ppl_gate_pass"] else "FAIL")),
    }
    return summary


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cleanup_model(model: nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
