"""Transformer-facing Linear wrapper. OWNER: Transformer integration."""
from __future__ import annotations
from collections.abc import Iterable
import torch
from torch import nn
from torch.nn import functional as F

class RNSLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, moduli: Iterable[int] = (3,5,7,11), mode: str = "torch", backend=None) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.moduli = tuple(int(m) for m in moduli)
        self.mode = mode
        self.backend = backend
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / self.in_features**0.5 if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, layer: nn.Linear, *, moduli=(3,5,7,11), mode="torch", backend=None):
        replacement = cls(layer.in_features, layer.out_features, bias=layer.bias is not None, moduli=moduli, mode=mode, backend=backend)
        replacement.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            replacement.bias.data.copy_(layer.bias.data)
        return replacement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "torch":
            return F.linear(x, self.weight, self.bias)
        if self.mode != "rns":
            raise ValueError(f"unknown mode: {self.mode}")
        # TODO(Integration owner): quantize -> backend.matmul -> decode/dequantize -> reshape -> bias.
        raise NotImplementedError("RNS mode is not implemented yet")
