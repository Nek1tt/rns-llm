from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import median
from typing import Callable, Iterable

import torch

# Pairwise-coprime values <= 255. Keeping residues centered makes every value
# representable by signed int8 and therefore consumable by INT8 Tensor Cores.
MODULI_CANDIDATES: tuple[int, ...] = (
    255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 211, 199,
)


def _extension():
    try:
        from rns_llm import _HYBRID  # type: ignore
    except Exception as exc:  # pragma: no cover - requires CUDA build
        raise RuntimeError(
            "rns_llm._HYBRID is unavailable. Build with RNS_LLM_BUILD_CUDA=1."
        ) from exc
    return _HYBRID


def quant_max(logical_bits: int) -> int:
    if logical_bits not in {8, 16, 32}:
        raise ValueError(f"logical_bits must be 8, 16 or 32, got {logical_bits}")
    return (1 << (logical_bits - 1)) - 1


def required_signed_range(logical_bits: int, k: int) -> int:
    """Worst-case modulus product needed for a signed dot product."""
    if k <= 0:
        raise ValueError("k must be positive")
    qmax = quant_max(logical_bits)
    return 2 * k * qmax * qmax + 1


def choose_moduli(logical_bits: int, k: int) -> tuple[int, ...]:
    required = required_signed_range(logical_bits, k)
    product = 1
    selected: list[int] = []
    for modulus in MODULI_CANDIDATES:
        selected.append(modulus)
        product *= modulus
        if product >= required:
            return tuple(selected)
    raise ValueError(
        f"The built-in modulus pool cannot cover q{logical_bits}, K={k}; "
        f"required product={required}, available product={product}."
    )


def modulus_product(moduli: Iterable[int]) -> int:
    product = 1
    for modulus in moduli:
        product *= int(modulus)
    return product


def _mod_inverse(value: int, modulus: int) -> int:
    return pow(value % modulus, -1, modulus)


def rns_constants(moduli: Iterable[int], device: torch.device) -> dict[str, torch.Tensor]:
    moduli_tuple = tuple(int(v) for v in moduli)
    prefix = 1
    prefix_inverses: list[int] = []
    two64_mod: list[int] = []
    for index, modulus in enumerate(moduli_tuple):
        prefix_inverses.append(1 if index == 0 else _mod_inverse(prefix, modulus))
        two64_mod.append(pow(2, 64, modulus))
        prefix *= modulus
    return {
        "moduli": torch.tensor(moduli_tuple, dtype=torch.int32, device=device),
        "prefix_inverses": torch.tensor(prefix_inverses, dtype=torch.int32, device=device),
        "two64_mod": torch.tensor(two64_mod, dtype=torch.int32, device=device),
    }


def per_row_scales(x: torch.Tensor, logical_bits: int) -> torch.Tensor:
    if x.dtype != torch.float32 or x.ndim != 2:
        raise ValueError("x must be contiguous FP32 [M,K]")
    maximum = x.abs().amax(dim=1).double()
    qmax = float(quant_max(logical_bits))
    return torch.clamp(maximum / qmax, min=torch.finfo(torch.float64).tiny).contiguous()


def per_output_scales(weight: torch.Tensor, logical_bits: int) -> torch.Tensor:
    if weight.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError("weight must be contiguous FP32 [N,K]")
    maximum = weight.abs().amax(dim=1).double()
    qmax = float(quant_max(logical_bits))
    return torch.clamp(maximum / qmax, min=torch.finfo(torch.float64).tiny).contiguous()


def empty_bias(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32, device=device)


def pad_inner_dimension(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    multiple: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if x.ndim != 2 or weight.ndim != 2 or x.size(1) != weight.size(1):
        raise ValueError("expected x[M,K] and weight[N,K]")
    k = int(x.size(1))
    padded_k = ((k + multiple - 1) // multiple) * multiple
    if padded_k == k:
        return x.contiguous(), weight.contiguous(), 0
    pad = padded_k - k
    x_pad = torch.nn.functional.pad(x, (0, pad)).contiguous()
    w_pad = torch.nn.functional.pad(weight, (0, pad)).contiguous()
    return x_pad, w_pad, pad


@dataclass
class NativeWeight:
    quantized_t: torch.Tensor  # [K,N] int8
    scales: torch.Tensor       # [N] float64
    k: int
    n: int


@dataclass
class RNSWeight:
    residues: torch.Tensor     # [C,K,N] int8
    scales: torch.Tensor       # [N] float64
    constants: dict[str, torch.Tensor]
    moduli: tuple[int, ...]
    logical_bits: int
    k: int
    n: int


@dataclass
class NativeActivation:
    quantized: torch.Tensor
    scales: torch.Tensor


@dataclass
class RNSActivation:
    residues: torch.Tensor
    scales: torch.Tensor


class HybridCudaOps:
    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("HybridCudaOps requires a CUDA GPU")
        self.ext = _extension()

    @torch.no_grad()
    def prepare_native_weight(self, weight: torch.Tensor) -> NativeWeight:
        weight = weight.to(self.device, dtype=torch.float32).contiguous()
        n, k = map(int, weight.shape)
        if k % 4 or n % 4:
            raise ValueError(f"native INT8 requires K,N multiples of 4; got K={k}, N={n}")
        scales = per_output_scales(weight, 8)
        output = torch.empty((k, n), dtype=torch.int8, device=self.device)
        self.ext.quantize_weight_int8_out(weight, scales, output)
        return NativeWeight(output, scales, k, n)

    @torch.no_grad()
    def prepare_rns_weight(self, weight: torch.Tensor, logical_bits: int) -> RNSWeight:
        weight = weight.to(self.device, dtype=torch.float32).contiguous()
        n, k = map(int, weight.shape)
        if k % 4 or n % 4:
            raise ValueError(f"RNS INT8 GEMM requires K,N multiples of 4; got K={k}, N={n}")
        moduli = choose_moduli(logical_bits, k)
        constants = rns_constants(moduli, self.device)
        scales = per_output_scales(weight, logical_bits)
        output = torch.empty((len(moduli), k, n), dtype=torch.int8, device=self.device)
        self.ext.encode_weight_fp32_out(
            weight, scales, quant_max(logical_bits), constants["moduli"], output
        )
        return RNSWeight(output, scales, constants, moduli, logical_bits, k, n)

    @torch.no_grad()
    def prepare_native_activation(self, x: torch.Tensor) -> NativeActivation:
        x = x.to(self.device, dtype=torch.float32).contiguous()
        scales = per_row_scales(x, 8)
        output = torch.empty_like(x, dtype=torch.int8)
        self.ext.quantize_activation_int8_out(x, scales, output)
        return NativeActivation(output, scales)

    @torch.no_grad()
    def prepare_rns_activation(self, x: torch.Tensor, weight: RNSWeight) -> RNSActivation:
        x = x.to(self.device, dtype=torch.float32).contiguous()
        if int(x.size(1)) != weight.k:
            raise ValueError("activation/weight K mismatch")
        scales = per_row_scales(x, weight.logical_bits)
        output = torch.empty(
            (len(weight.moduli), int(x.size(0)), weight.k),
            dtype=torch.int8,
            device=self.device,
        )
        self.ext.encode_activation_fp32_out(
            x, scales, quant_max(weight.logical_bits), weight.constants["moduli"], output
        )
        return RNSActivation(output, scales)

    @torch.no_grad()
    def native_core(
        self,
        activation: NativeActivation,
        weight: NativeWeight,
        bias: torch.Tensor | None = None,
        *,
        output: torch.Tensor | None = None,
        accumulators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = int(activation.quantized.size(0))
        bias_t = empty_bias(self.device) if bias is None else bias.to(
            self.device, dtype=torch.float32
        ).contiguous()
        if output is None:
            output = torch.empty((m, weight.n), dtype=torch.float32, device=self.device)
        if accumulators is None:
            accumulators = torch.empty((m, weight.n), dtype=torch.int32, device=self.device)
        return self.ext.native_mm_dequant_fp32_out(
            activation.quantized,
            weight.quantized_t,
            activation.scales,
            weight.scales,
            bias_t,
            accumulators,
            output,
        )

    @torch.no_grad()
    def rns_core(
        self,
        activation: RNSActivation,
        weight: RNSWeight,
        bias: torch.Tensor | None = None,
        *,
        output: torch.Tensor | None = None,
        accumulators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        channels = len(weight.moduli)
        m = int(activation.residues.size(1))
        bias_t = empty_bias(self.device) if bias is None else bias.to(
            self.device, dtype=torch.float32
        ).contiguous()
        if output is None:
            output = torch.empty((m, weight.n), dtype=torch.float32, device=self.device)
        if accumulators is None:
            accumulators = torch.empty(
                (channels, m, weight.n), dtype=torch.int32, device=self.device
            )
        return self.ext.rns_mm_dequant_fp32_out(
            activation.residues,
            weight.residues,
            weight.constants["moduli"],
            weight.constants["prefix_inverses"],
            weight.constants["two64_mod"],
            activation.scales,
            weight.scales,
            bias_t,
            accumulators,
            output,
        )

    @torch.no_grad()
    def native_e2e(self, x: torch.Tensor, weight: NativeWeight, bias: torch.Tensor | None) -> torch.Tensor:
        return self.native_core(self.prepare_native_activation(x), weight, bias)

    @torch.no_grad()
    def rns_e2e(self, x: torch.Tensor, weight: RNSWeight, bias: torch.Tensor | None) -> torch.Tensor:
        return self.rns_core(self.prepare_rns_activation(x, weight), weight, bias)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    alpha = position - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
    return {
        "p50_ms": median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


@torch.no_grad()
def benchmark_cuda_callable(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
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
    return summarize_latencies(samples)


@torch.no_grad()
def benchmark_randomized(
    callables: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    names = list(callables)
    rng = random.Random(seed)
    for _ in range(warmup):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            callables[name]()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(iterations):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            callables[name]()
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)))
    return {name: summarize_latencies(vals) for name, vals in samples.items()}


@torch.no_grad()
def accuracy_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().float().reshape(-1)
    cand = candidate.detach().float().reshape(-1)
    diff = cand - ref
    ref_norm = torch.linalg.vector_norm(ref)
    denom = torch.clamp(ref_norm, min=torch.finfo(torch.float32).eps)
    cosine = torch.nn.functional.cosine_similarity(ref.unsqueeze(0), cand.unsqueeze(0)).item()
    return {
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "mae": float(torch.mean(diff.abs()).item()),
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
        "cosine_similarity": float(cosine),
    }


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())
