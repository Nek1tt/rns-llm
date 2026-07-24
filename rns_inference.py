"""Runtime-only RNS integration for nanoGPT inference.

Training remains ordinary PyTorch. This module is imported by ``sample.py``
only after a checkpoint has been loaded and moved to its inference device.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rns_llm
from torch import nn

from rns_llm.backends import CudaRNSBackend
from rns_llm.integration import replace_linear_modules
from rns_llm.layers import RNSLinear, RNSMatmul


@dataclass
class RNSInstallation:
    mode: str
    package_path: str
    backend: CudaRNSBackend | None
    replaced_linears: list[str]
    attention_blocks: int
    tied_lm_head: bool
    attention_matmuls: list[tuple[RNSMatmul, RNSMatmul]]

    def stats_snapshot(self) -> dict[str, int]:
        if self.backend is None:
            return {}
        return self.backend.stats_snapshot()

    def reset_stats(self) -> None:
        if self.backend is not None:
            self.backend.reset_stats()

    def metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "package_path": self.package_path,
            "replaced_linears": self.replaced_linears,
            "linear_count": len(self.replaced_linears),
            "attention_blocks": self.attention_blocks,
            "tied_lm_head_preserved": self.tied_lm_head,
            "qk_compute_count": sum(pair[0].compute_count for pair in self.attention_matmuls),
            "av_compute_count": sum(pair[1].compute_count for pair in self.attention_matmuls),
        }


def install_rns_inference(
    model: nn.Module,
    *,
    mode: str,
    quant_bits: int = 8,
    include_attention_matmul: bool = True,
    include_lm_head: bool = True,
    fused: bool = True,
    lut_channels: int = 2,
    moduli_strategy: str = "dense_coprime",
) -> RNSInstallation:
    """Replace nanoGPT inference GEMMs while preserving trained parameters."""

    if mode not in ("rns", "software-rns"):
        raise ValueError("RNS mode must be 'rns' or 'software-rns'")
    if model.training:
        raise RuntimeError("call model.eval() before installing RNS inference")

    backend = CudaRNSBackend() if mode == "rns" else None
    lm_head = getattr(model, "lm_head", None)
    embedding = getattr(getattr(model, "transformer", None), "wte", None)
    was_tied = (
        isinstance(lm_head, nn.Linear)
        and embedding is not None
        and lm_head.weight is embedding.weight
    )

    replaced = replace_linear_modules(
        model,
        mode=mode,
        backend=backend,
        moduli=None,
        quant_bits=quant_bits,
        moduli_strategy=moduli_strategy,
        fused=fused,
        lut_channels=lut_channels,
        predicate=lambda name, _: include_lm_head or name != "lm_head",
    )

    tied_preserved = False
    if include_lm_head and was_tied and isinstance(model.lm_head, RNSLinear):
        model.lm_head.weight = model.transformer.wte.weight
        model.lm_head.clear_weight_cache()
        tied_preserved = model.lm_head.weight is model.transformer.wte.weight

    attention_blocks = 0
    attention_matmuls: list[tuple[RNSMatmul, RNSMatmul]] = []
    if include_attention_matmul:
        for block in model.transformer.h:
            qk = RNSMatmul(
                mode=mode,
                backend=backend,
                quant_bits=quant_bits,
                moduli=None,
                fused=fused,
                lut_channels=lut_channels,
                moduli_strategy=moduli_strategy,
            ).eval()
            av = RNSMatmul(
                mode=mode,
                backend=backend,
                quant_bits=quant_bits,
                moduli=None,
                fused=fused,
                lut_channels=lut_channels,
                moduli_strategy=moduli_strategy,
            ).eval()
            block.attn.set_rns_matmul(qk, av)
            attention_matmuls.append((qk, av))
            attention_blocks += 1

    return RNSInstallation(
        mode=mode,
        package_path=str(Path(rns_llm.__file__).resolve()),
        backend=backend,
        replaced_linears=replaced,
        attention_blocks=attention_blocks,
        tied_lm_head=tied_preserved,
        attention_matmuls=attention_matmuls,
    )
