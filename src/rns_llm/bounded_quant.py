from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from rns_llm.adaptive import signed_capacity
from rns_llm.reference import validate_moduli


@dataclass(frozen=True)
class StaticChannelPlan:
    """A host-computed, synchronization-free RNS channel plan.

    The plan constrains the L1 norm of every quantized activation row so that
    every dot product is guaranteed to fit the centered range of a fixed RNS
    prefix.  No GPU scalar is copied to the host on the inference hot path.
    """

    moduli: tuple[int, ...]
    k: int
    max_abs_weight: int
    capacity: int
    row_l1_budget: int
    rounding_guard: int

    @property
    def channels(self) -> int:
        return len(self.moduli)

    @property
    def maximum_dot_bound(self) -> int:
        return self.row_l1_budget * self.max_abs_weight

    def to_dict(self) -> dict[str, int | list[int]]:
        payload = asdict(self)
        payload["moduli"] = list(self.moduli)
        payload["channels"] = self.channels
        payload["maximum_dot_bound"] = self.maximum_dot_bound
        return payload


def build_static_channel_plan(
    full_moduli: Iterable[int],
    *,
    channels: int,
    k: int,
    max_abs_weight: int,
    rounding_guard: int | None = None,
) -> StaticChannelPlan:
    """Build a fixed-prefix plan with a strict centered-RNS overflow guard.

    For quantized activations ``A`` and weights ``B``:

    ``|A_i @ B_j| <= ||A_i||_1 * max_abs_weight``.

    The quantizer below enforces ``||A_i||_1 <= row_l1_budget`` entirely on the
    device.  Therefore all dot products fit the selected RNS prefix.
    """

    mods = validate_moduli(full_moduli)
    if not 2 <= channels <= len(mods):
        raise ValueError("channels must be between 2 and len(full_moduli)")
    if k <= 0:
        raise ValueError("k must be positive")
    if max_abs_weight <= 0:
        raise ValueError("max_abs_weight must be positive")

    prefix = mods[:channels]
    capacity = signed_capacity(prefix)
    row_l1_budget = capacity // int(max_abs_weight)
    guard = int(k if rounding_guard is None else rounding_guard)
    if guard < (k + 1) // 2:
        raise ValueError("rounding_guard must be at least ceil(k/2)")
    if row_l1_budget <= guard:
        raise ValueError(
            "selected RNS prefix is too small for bounded-L1 quantization: "
            f"budget={row_l1_budget}, guard={guard}"
        )

    return StaticChannelPlan(
        moduli=prefix,
        k=int(k),
        max_abs_weight=int(max_abs_weight),
        capacity=capacity,
        row_l1_budget=row_l1_budget,
        rounding_guard=guard,
    )



def l1_bounded_symmetric_scale(
    values: torch.Tensor,
    *,
    quant_max: int,
    plan: StaticChannelPlan,
) -> torch.Tensor:
    """Return the per-row scale required by the static L1 plan.

    This is the scale-only counterpart of ``l1_bounded_symmetric_quantize``.
    v0.7 uses it before a fused CUDA quantize-and-RNS-encode kernel, avoiding
    materialization of an intermediate quantized activation matrix.
    """

    if values.ndim != 2:
        raise ValueError("values must have shape [rows, k]")
    if int(values.shape[1]) != plan.k:
        raise ValueError("values K dimension does not match the plan")
    if not values.is_floating_point():
        raise ValueError("values must be floating point")
    if quant_max <= 0:
        raise ValueError("quant_max must be positive")

    detached = values.detach().float()
    max_abs = detached.abs().amax(dim=1, keepdim=True)
    row_l1 = detached.abs().sum(dim=1, keepdim=True)
    remaining_budget = plan.row_l1_budget - plan.rounding_guard

    eps = torch.finfo(torch.float32).eps
    max_abs_scale = max_abs / float(quant_max)
    l1_scale = row_l1 / float(remaining_budget)
    return torch.clamp(torch.maximum(max_abs_scale, l1_scale), min=eps)

def l1_bounded_symmetric_quantize(
    values: torch.Tensor,
    *,
    quant_max: int,
    plan: StaticChannelPlan,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize rows while guaranteeing the plan's L1 budget.

    For round-to-nearest, ``sum(abs(round(x/s))) <= sum(abs(x))/s + K/2``.
    We reserve ``rounding_guard >= K/2`` and choose a scale large enough to
    satisfy the remaining budget.  Clamping can only decrease the L1 norm.

    Returns ``(quantized, scale)`` where scale has shape ``[rows, 1]``.
    """

    if values.ndim != 2:
        raise ValueError("values must have shape [rows, k]")
    if int(values.shape[1]) != plan.k:
        raise ValueError("values K dimension does not match the plan")
    if not values.is_floating_point():
        raise ValueError("values must be floating point")
    if quant_max <= 0:
        raise ValueError("quant_max must be positive")

    if dtype is None:
        dtype = torch.int8 if quant_max <= 127 else torch.int16
    if dtype not in (torch.int8, torch.int16, torch.int32):
        raise ValueError("dtype must be an integer tensor dtype")

    scale = l1_bounded_symmetric_scale(
        values, quant_max=quant_max, plan=plan
    )

    quantized = torch.clamp(
        torch.round(values / scale),
        -quant_max,
        quant_max,
    ).to(dtype)
    return quantized, scale


def l1_budget_tensor(quantized: torch.Tensor) -> torch.Tensor:
    """Return per-row L1 norms without forcing a device-to-host sync."""

    if quantized.ndim != 2:
        raise ValueError("quantized must have shape [rows, k]")
    if quantized.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("quantized must use an integer dtype")
    return quantized.to(torch.int64).abs().sum(dim=1)
