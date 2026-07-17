from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .rns_linear import RNSLinear


class RNSQKVProjection(nn.Module):
    """One combined QKV projection, split back into Q, K and V.

    Combining weights is mathematically identical to three Linear operations
    because all three consume the same activation tensor.  Per-output-channel
    weight scales are preserved by RNSLinear.
    """

    def __init__(self, combined: RNSLinear, split_sizes: tuple[int, int, int]) -> None:
        super().__init__()
        if sum(split_sizes) != combined.out_features:
            raise ValueError("split sizes do not match combined output size")
        self.combined = combined
        self.split_sizes = tuple(int(x) for x in split_sizes)
        self.compute_count = 0

    @classmethod
    def from_linears(
        cls,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        *,
        backend=None,
        moduli: Iterable[int] | None = None,
        moduli_strategy: str = "dense_coprime",
        mode: str = "rns",
        quant_bits: int = 8,
        fused: bool = True,
        lut_channels: int = 2,
        adaptive_channels: bool = False,
        adaptive_min_channels: int = 3,
    ) -> "RNSQKVProjection":
        linears = (q_proj, k_proj, v_proj)
        if any(not isinstance(layer, nn.Linear) for layer in linears):
            raise TypeError("q_proj, k_proj and v_proj must be nn.Linear")
        if len({layer.in_features for layer in linears}) != 1:
            raise ValueError("Q/K/V input sizes must match")
        if len({layer.weight.device for layer in linears}) != 1:
            raise ValueError("Q/K/V weights must share a device")
        if len({layer.weight.dtype for layer in linears}) != 1:
            raise ValueError("Q/K/V weights must share a dtype")
        if len({layer.bias is None for layer in linears}) != 1:
            raise ValueError("Q/K/V must consistently use or omit bias")

        in_features = q_proj.in_features
        split_sizes = tuple(layer.out_features for layer in linears)
        combined = RNSLinear(
            in_features,
            sum(split_sizes),
            bias=q_proj.bias is not None,
            backend=backend,
            moduli=moduli,
            moduli_strategy=moduli_strategy,
            mode=mode,
            quant_bits=quant_bits,
            fused=fused,
            lut_channels=lut_channels,
            adaptive_channels=adaptive_channels,
            adaptive_min_channels=adaptive_min_channels,
        ).to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)

        with torch.no_grad():
            combined.weight.copy_(torch.cat([layer.weight for layer in linears], dim=0))
            if combined.bias is not None:
                combined.bias.copy_(torch.cat([layer.bias for layer in linears], dim=0))
        return cls(combined, split_sizes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.compute_count += 1
        combined = self.combined(inputs)
        q, k, v = torch.split(combined, self.split_sizes, dim=-1)
        return q, k, v


class _QKVSlice(nn.Module):
    """Model-compatible proxy used as q_proj/k_proj/v_proj.

    The coordinator itself is registered once on the attention module; this
    proxy stores only a weak reference, avoiding duplicate parameter
    registration in the state dict.
    """

    def __init__(self, coordinator: "CachedRNSQKV", index: int) -> None:
        super().__init__()
        object.__setattr__(self, "_coordinator_ref", weakref.ref(coordinator))
        self.index = int(index)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        coordinator = self._coordinator_ref()
        if coordinator is None:
            raise RuntimeError("QKV coordinator has been destroyed")
        return coordinator.forward_slice(inputs, self.index)


class CachedRNSQKV(nn.Module):
    """Compute a fused QKV projection once while preserving the OPT API."""

    def __init__(self, projection: RNSQKVProjection) -> None:
        super().__init__()
        self.projection = projection
        self._cache_key = None
        self._cache_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._served: set[int] = set()
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _key(inputs: torch.Tensor):
        return (
            id(inputs),
            int(inputs.data_ptr()),
            int(inputs._version),
            tuple(inputs.shape),
            inputs.dtype,
            inputs.device,
        )

    def clear_cache(self) -> None:
        self._cache_key = None
        self._cache_outputs = None
        self._served.clear()

    def forward_slice(self, inputs: torch.Tensor, index: int) -> torch.Tensor:
        if index not in (0, 1, 2):
            raise ValueError("QKV index must be 0, 1 or 2")
        key = self._key(inputs)
        # Repeated use of the same slice indicates a new logical call; recompute
        # instead of returning potentially stale data.
        if self._cache_outputs is None or self._cache_key != key or index in self._served:
            self._cache_outputs = self.projection(inputs)
            self._cache_key = key
            self._served = set()
            self.cache_misses += 1
        else:
            self.cache_hits += 1

        assert self._cache_outputs is not None
        result = self._cache_outputs[index]
        self._served.add(index)
        if len(self._served) == 3:
            # `result` owns a reference to its tensor; clearing here is safe and
            # prevents retaining hidden states between Transformer forwards.
            self.clear_cache()
        return result

    def slices(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return (_QKVSlice(self, 0), _QKVSlice(self, 1), _QKVSlice(self, 2))


@dataclass(frozen=True)
class InstalledQKVFusion:
    module_name: str
    coordinator: CachedRNSQKV


def install_opt_qkv_fusion(
    attention: nn.Module,
    *,
    backend,
    quant_bits: int = 8,
    moduli_strategy: str = "dense_coprime",
    lut_channels: int = 2,
    adaptive_channels: bool = False,
    adaptive_min_channels: int = 3,
) -> CachedRNSQKV:
    """Replace OPT-style q/k/v projections with one cached fused projection."""
    for name in ("q_proj", "k_proj", "v_proj"):
        if not isinstance(getattr(attention, name, None), nn.Linear):
            raise TypeError(f"attention.{name} must be nn.Linear before fusion")

    projection = RNSQKVProjection.from_linears(
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        backend=backend,
        mode="rns",
        quant_bits=quant_bits,
        moduli_strategy=moduli_strategy,
        lut_channels=lut_channels,
        adaptive_channels=adaptive_channels,
        adaptive_min_channels=adaptive_min_channels,
    ).eval()
    coordinator = CachedRNSQKV(projection).eval()
    # Register exactly one owner of the combined parameters.
    attention.rns_qkv = coordinator
    q_proxy, k_proxy, v_proxy = coordinator.slices()
    attention.q_proj = q_proxy
    attention.k_proj = k_proxy
    attention.v_proj = v_proxy
    return coordinator
