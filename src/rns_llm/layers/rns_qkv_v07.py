from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from rns_llm.layers.rns_linear_v07 import FastRNSLinearV07
from rns_llm.layers.rns_qkv import CachedRNSQKV, RNSQKVProjection


class FastRNSQKVProjectionV07(RNSQKVProjection):
    """Fused QKV projection backed by the v0.7 direct FP16 epilogue."""

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
        static_channels: int | None = None,
        use_v07_epilogue: bool = True,
    ) -> "FastRNSQKVProjectionV07":
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

        split_sizes = tuple(layer.out_features for layer in linears)
        combined = FastRNSLinearV07(
            q_proj.in_features,
            sum(split_sizes),
            bias=q_proj.bias is not None,
            backend=backend,
            moduli=moduli,
            moduli_strategy=moduli_strategy,
            mode=mode,
            quant_bits=quant_bits,
            fused=fused,
            lut_channels=lut_channels,
            static_channels=static_channels,
            use_v07_epilogue=use_v07_epilogue,
        ).to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)

        with torch.no_grad():
            combined.weight.copy_(torch.cat([layer.weight for layer in linears], dim=0))
            if combined.bias is not None:
                combined.bias.copy_(torch.cat([layer.bias for layer in linears], dim=0))
        return cls(combined, split_sizes)


def install_opt_qkv_fusion_v07(
    attention: nn.Module,
    *,
    backend,
    quant_bits: int = 8,
    moduli_strategy: str = "dense_coprime",
    lut_channels: int = 2,
    static_channels: int | None = None,
    use_v07_epilogue: bool = True,
) -> CachedRNSQKV:
    """Replace OPT q/k/v projections with the v0.7 fused projection."""

    for name in ("q_proj", "k_proj", "v_proj"):
        if not isinstance(getattr(attention, name, None), nn.Linear):
            raise TypeError(f"attention.{name} must be nn.Linear before fusion")

    projection = FastRNSQKVProjectionV07.from_linears(
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        backend=backend,
        mode="rns",
        quant_bits=quant_bits,
        moduli_strategy=moduli_strategy,
        lut_channels=lut_channels,
        static_channels=static_channels,
        use_v07_epilogue=use_v07_epilogue,
    ).eval()
    coordinator = CachedRNSQKV(projection).eval()
    attention.rns_qkv = coordinator
    q_proxy, k_proxy, v_proxy = coordinator.slices()
    attention.q_proj = q_proxy
    attention.k_proj = k_proxy
    attention.v_proj = v_proxy
    return coordinator
