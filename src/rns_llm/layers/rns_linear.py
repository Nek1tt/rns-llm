from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from rns_llm.backends import (
    PreparedAdaptiveRNSWeight,
    PreparedRNSWeight,
    RNSWorkspace,
)
from rns_llm.reference import choose_moduli_for_dot, validate_moduli


class RNSLinear(nn.Module):
    """Inference-only Linear layer using the v0.5 fused RNS backend.

    The weight is quantized and encoded once.  At runtime only activations are
    encoded; cuBLAS computes all residue channels and a single CUDA kernel
    performs modulo + Garner reconstruction.  This covers the curator's
    matrix-multiplication and non-modular-operation tasks in one model-facing
    interface.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        backend=None,
        moduli: Iterable[int] | None = None,
        moduli_strategy: str = "dense_coprime",
        mode: str = "torch",
        kernel: str = "auto",
        quant_bits: int = 8,
        fused: bool = True,
        lut_channels: int = 2,
        adaptive_channels: bool = False,
        adaptive_min_channels: int = 3,
    ) -> None:
        super().__init__()
        if quant_bits not in (8, 12, 16):
            raise ValueError("quant_bits must be 8, 12 or 16")
        if lut_channels not in (0, 1, 2):
            raise ValueError("lut_channels must be 0, 1 or 2")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.backend = backend
        self.mode = mode
        self.kernel = kernel
        self.quant_bits = int(quant_bits)
        self.fused = bool(fused)
        self.lut_channels = int(lut_channels)
        self.moduli_strategy = moduli_strategy
        self.adaptive_channels = bool(adaptive_channels)
        self.adaptive_min_channels = int(adaptive_min_channels)

        quant_max = (1 << (self.quant_bits - 1)) - 1
        chosen = (
            tuple(moduli)
            if moduli is not None
            else choose_moduli_for_dot(
                self.in_features,
                quant_max,
                quant_max,
                strategy=moduli_strategy,
            )
        )
        self.moduli = validate_moduli(chosen)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("_weight_scale", torch.empty(0), persistent=False)
        self.register_buffer(
            "_weight_residues", torch.empty(0, dtype=torch.int8), persistent=False
        )
        self._prepared_weight_version: int | None = None
        self._prepared_weight: PreparedRNSWeight | None = None
        self._adaptive_prepared_weight: PreparedAdaptiveRNSWeight | None = None
        # A workspace cannot be shared by concurrent CUDA streams.
        self._workspace_by_rows: dict[tuple[int, int, int], RNSWorkspace] = {}
        self.last_adaptive_metadata: dict[str, int] | None = None
        self.reset_parameters()

    @property
    def quant_max(self) -> int:
        return (1 << (self.quant_bits - 1)) - 1

    @property
    def quant_dtype(self) -> torch.dtype:
        return torch.int8 if self.quant_bits == 8 else torch.int16

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / self.in_features**0.5
            nn.init.uniform_(self.bias, -bound, bound)
        self.clear_weight_cache()

    def clear_weight_cache(self) -> None:
        self._weight_scale = torch.empty(0, device=self.weight.device)
        self._weight_residues = torch.empty(
            0, dtype=torch.int8, device=self.weight.device
        )
        self._prepared_weight_version = None
        self._prepared_weight = None
        self._adaptive_prepared_weight = None
        self._workspace_by_rows = {}
        self.last_adaptive_metadata = None

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        backend=None,
        moduli: Iterable[int] | None = None,
        moduli_strategy: str = "dense_coprime",
        mode: str = "torch",
        kernel: str = "auto",
        quant_bits: int = 8,
        fused: bool = True,
        lut_channels: int = 2,
        adaptive_channels: bool = False,
        adaptive_min_channels: int = 3,
    ) -> "RNSLinear":
        result = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            backend=backend,
            moduli=moduli,
            moduli_strategy=moduli_strategy,
            mode=mode,
            kernel=kernel,
            quant_bits=quant_bits,
            fused=fused,
            lut_channels=lut_channels,
            adaptive_channels=adaptive_channels,
            adaptive_min_channels=adaptive_min_channels,
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)
        result.weight.data.copy_(layer.weight.data)
        if layer.bias is not None and result.bias is not None:
            result.bias.data.copy_(layer.bias.data)
        return result

    def _symmetric_scale(self, values: torch.Tensor, dim=None) -> torch.Tensor:
        max_abs = values.detach().abs().amax(dim=dim, keepdim=True)
        return torch.clamp(
            max_abs / float(self.quant_max),
            min=torch.finfo(torch.float32).eps,
        )

    def _quantize(self, values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.round(values / scale),
            -self.quant_max,
            self.quant_max,
        ).to(self.quant_dtype)

    @torch.no_grad()
    def prepare_weight(self) -> None:
        if self.backend is None:
            raise RuntimeError("RNSLinear requires a backend in mode='rns'")
        if not self.weight.is_cuda:
            raise RuntimeError("RNSLinear currently expects CUDA weights")

        current_version = self.weight._version
        if (
            self._prepared_weight_version == current_version
            and self._prepared_weight is not None
        ):
            return

        # Per-output-channel quantization. B layout for backend is [K,N].
        scale = self._symmetric_scale(self.weight.float(), dim=1)
        quantized = self._quantize(self.weight.float(), scale)
        quantized_b = quantized.transpose(0, 1).contiguous()
        adaptive_prepared = None
        if self.adaptive_channels:
            # Encode once. The full-channel PreparedRNSWeight is one of the
            # prefix variants and shares the same underlying residue storage.
            adaptive_prepared = self.backend.prepare_weight_adaptive(
                quantized_b,
                self.moduli,
                min_channels=self.adaptive_min_channels,
            )
            prepared = adaptive_prepared.variants[len(self.moduli)]
        else:
            prepared = self.backend.prepare_weight(quantized_b, self.moduli)

        self._weight_scale = scale.squeeze(1).to(self.weight.device)
        self._weight_residues = prepared.residues
        self._prepared_weight = prepared
        self._adaptive_prepared_weight = adaptive_prepared
        self._prepared_weight_version = current_version
        self._workspace_by_rows = {}

    def _workspace(self, rows: int, channels: int | None = None) -> RNSWorkspace:
        assert self._prepared_weight is not None
        channels = len(self.moduli) if channels is None else int(channels)
        stream = torch.cuda.current_stream(self.weight.device)
        device_index = self.weight.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (int(rows), int(channels), int(stream.cuda_stream))
        cached = self._workspace_by_rows.get(key)
        if cached is None:
            cached = self.backend.create_workspace(
                device=self.weight.device,
                channels=channels,
                m=rows,
                n=self.out_features,
            )
            self._workspace_by_rows[key] = cached
        return cached

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "torch":
            return F.linear(inputs, self.weight, self.bias)
        if self.mode != "rns":
            raise ValueError(f"unknown mode: {self.mode}")
        if self.training:
            raise RuntimeError("RNSLinear is inference-only; call eval()")
        if self.backend is None:
            raise RuntimeError("RNSLinear mode='rns' requires a backend")

        self.prepare_weight()
        assert self._prepared_weight is not None

        original_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).float().contiguous()
        activation_scale = self._symmetric_scale(flat)
        quantized_a = self._quantize(flat, activation_scale).contiguous()

        if self.adaptive_channels:
            assert self._adaptive_prepared_weight is not None
            workspaces = {
                channels: self._workspace(flat.shape[0], channels)
                for channels in self._adaptive_prepared_weight.variants
            }
            integer_result, metadata = self.backend.matmul_prepared_adaptive_fused(
                quantized_a,
                self._adaptive_prepared_weight,
                lut_channels=self.lut_channels,
                workspace_by_channels=workspaces,
                return_metadata=True,
            )
            self.last_adaptive_metadata = metadata
            integer_result = integer_result.float()
        elif self.fused and self._prepared_weight.kernel == "cublas":
            integer_result = self.backend.matmul_prepared_fused(
                quantized_a,
                self._prepared_weight,
                lut_channels=self.lut_channels,
                workspace=self._workspace(flat.shape[0]),
            ).float()
        else:
            integer_result = self.backend.matmul_prepared(
                quantized_a,
                self._prepared_weight,
                kernel=self.kernel,
            ).float()

        output = integer_result * activation_scale * self._weight_scale.unsqueeze(0)
        if self.bias is not None:
            output = output + self.bias.float()
        return output.reshape(*original_shape, self.out_features).to(inputs.dtype)
