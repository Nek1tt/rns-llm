from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from rns_llm.bounded_quant import (
    StaticChannelPlan,
    build_static_channel_plan,
    l1_bounded_symmetric_quantize,
    l1_bounded_symmetric_scale,
)
from rns_llm.layers.rns_linear import RNSLinear
from rns_llm.reference import choose_moduli_for_dot, validate_moduli
from rns_llm.v07_backend import V07FastPath, V07RNSWorkspace


class FastRNSLinearV07(RNSLinear):
    """RNS Linear with a direct Garner/dequant/bias-to-FP16 CUDA epilogue.

    v0.5/v0.6 materialize an ordinary quantized activation, launch a separate
    RNS encoder, write an INT64 reconstructed matrix, and then launch several
    PyTorch epilogue kernels. v0.7 fuses FP16 quantization with RNS encoding and
    reconstructs/dequantizes directly into FP16.

    ``static_channels=3`` enables synchronization-free bounded-L1 activation
    quantization. It is opt-in because it changes the quantizer and must pass a
    model-level PPL/accuracy gate before production use.
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
        static_channels: int | None = None,
        use_v07_epilogue: bool = True,
        fuse_quantize_encode: bool = True,
    ) -> None:
        if adaptive_channels and static_channels is not None:
            raise ValueError("adaptive_channels and static_channels are mutually exclusive")
        if static_channels is not None and static_channels < 2:
            raise ValueError("static_channels must be at least 2")

        quant_max = (1 << (int(quant_bits) - 1)) - 1
        full_moduli = (
            validate_moduli(moduli)
            if moduli is not None
            else choose_moduli_for_dot(
                int(in_features),
                quant_max,
                quant_max,
                strategy=moduli_strategy,
            )
        )
        selected_moduli = full_moduli
        if static_channels is not None:
            if static_channels > len(full_moduli):
                raise ValueError("static_channels exceeds the selected full moduli set")
            selected_moduli = full_moduli[: int(static_channels)]

        super().__init__(
            in_features,
            out_features,
            bias=bias,
            backend=backend,
            moduli=selected_moduli,
            moduli_strategy=moduli_strategy,
            mode=mode,
            kernel=kernel,
            quant_bits=quant_bits,
            fused=fused,
            lut_channels=lut_channels,
            adaptive_channels=adaptive_channels,
            adaptive_min_channels=adaptive_min_channels,
        )
        self.full_moduli = tuple(full_moduli)
        self.static_channels = None if static_channels is None else int(static_channels)
        self.use_v07_epilogue = bool(use_v07_epilogue)
        self.fuse_quantize_encode = bool(fuse_quantize_encode)
        self._static_plan: StaticChannelPlan | None = None
        self._v07_fast_path: V07FastPath | None = None
        self._v07_workspace_by_rows: dict[tuple[int, int, int], V07RNSWorkspace] = {}
        self._v07_prepared_version: int | None = None
        self.register_buffer(
            "_bias_float_cache", torch.empty(0, dtype=torch.float32), persistent=False
        )

    def clear_weight_cache(self) -> None:
        super().clear_weight_cache()
        if hasattr(self, "_static_plan"):
            self._static_plan = None
        if hasattr(self, "_v07_workspace_by_rows"):
            self._v07_workspace_by_rows = {}
        if hasattr(self, "_v07_prepared_version"):
            self._v07_prepared_version = None
        if hasattr(self, "_bias_float_cache"):
            self._bias_float_cache = torch.empty(
                0, dtype=torch.float32, device=self.weight.device
            )

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
        static_channels: int | None = None,
        use_v07_epilogue: bool = True,
        fuse_quantize_encode: bool = True,
    ) -> "FastRNSLinearV07":
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
            static_channels=static_channels,
            use_v07_epilogue=use_v07_epilogue,
            fuse_quantize_encode=fuse_quantize_encode,
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)
        result.weight.data.copy_(layer.weight.data)
        if layer.bias is not None and result.bias is not None:
            result.bias.data.copy_(layer.bias.data)
        return result

    @torch.no_grad()
    def prepare_weight(self) -> None:
        super().prepare_weight()
        assert self._prepared_weight is not None
        if self._v07_prepared_version == self._prepared_weight_version:
            return

        if self.static_channels is not None:
            # Symmetric weight quantization always clamps to quant_max. Using
            # this host constant is conservative and has no hot-path sync.
            self._static_plan = build_static_channel_plan(
                self.moduli,
                channels=len(self.moduli),
                k=self.in_features,
                max_abs_weight=self.quant_max,
            )
        self._bias_float_cache = (
            torch.empty(0, dtype=torch.float32, device=self.weight.device)
            if self.bias is None
            else self.bias.detach().to(dtype=torch.float32).contiguous()
        )
        if self._v07_fast_path is None:
            self._v07_fast_path = V07FastPath(self.backend)
        self._v07_prepared_version = self._prepared_weight_version

    def _v07_workspace(self, rows: int, channels: int) -> V07RNSWorkspace:
        assert self._prepared_weight is not None
        assert self._v07_fast_path is not None
        stream = torch.cuda.current_stream(self.weight.device)
        device_index = self.weight.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (int(rows), int(channels), int(stream.cuda_stream))
        cached = self._v07_workspace_by_rows.get(key)
        if cached is None:
            cached = self._v07_fast_path.create_rns_workspace(
                device=self.weight.device,
                channels=channels,
                m=rows,
                k=self.in_features,
                n=self.out_features,
            )
            self._v07_workspace_by_rows[key] = cached
        return cached

    def _activation_scale(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        if self.static_channels is None:
            max_abs = flat_inputs.detach().abs().amax().float().reshape(1)
            return torch.clamp(
                max_abs / float(self.quant_max),
                min=torch.finfo(torch.float32).eps,
            )
        if self._static_plan is None:
            raise RuntimeError("static channel plan was not prepared")
        return l1_bounded_symmetric_scale(
            flat_inputs,
            quant_max=self.quant_max,
            plan=self._static_plan,
        ).contiguous()

    def _quantize_activation(
        self, flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.static_channels is None:
            scale = self._symmetric_scale(flat)
            return self._quantize(flat, scale).contiguous(), scale
        if self._static_plan is None:
            raise RuntimeError("static channel plan was not prepared")
        quantized, scale = l1_bounded_symmetric_quantize(
            flat,
            quant_max=self.quant_max,
            plan=self._static_plan,
            dtype=self.quant_dtype,
        )
        return quantized.contiguous(), scale.contiguous()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "torch":
            return F.linear(inputs, self.weight, self.bias)
        if self.mode != "rns":
            raise ValueError(f"unknown mode: {self.mode}")
        if self.training:
            raise RuntimeError("FastRNSLinearV07 is inference-only; call eval()")
        if self.backend is None:
            raise RuntimeError("mode='rns' requires a CUDA backend")

        self.prepare_weight()
        assert self._prepared_weight is not None
        original_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, self.in_features).contiguous()

        can_use_v07 = (
            self.use_v07_epilogue
            and self.fused
            and self._prepared_weight.kernel == "cublas"
            and self.quant_bits == 8
            and inputs.dtype == torch.float16
            and self._v07_fast_path is not None
        )
        if can_use_v07:
            activation_scale = self._activation_scale(flat_inputs)
            workspace = self._v07_workspace(
                flat_inputs.shape[0], len(self._prepared_weight.moduli)
            )
            if self.fuse_quantize_encode:
                output = self._v07_fast_path.rns_fp16_input_dequant_fp16(
                    flat_inputs,
                    self._prepared_weight,
                    activation_scale=activation_scale,
                    weight_scale=self._weight_scale,
                    bias=None if self.bias is None else self._bias_float_cache,
                    quant_max=self.quant_max,
                    lut_channels=self.lut_channels,
                    workspace=workspace,
                )
            else:
                quantized_a, _ = self._quantize_activation(flat_inputs.float())
                output = self._v07_fast_path.rns_prepared_dequant_fp16(
                    quantized_a,
                    self._prepared_weight,
                    activation_scale=activation_scale,
                    weight_scale=self._weight_scale,
                    bias=None if self.bias is None else self._bias_float_cache,
                    lut_channels=self.lut_channels,
                    workspace=workspace,
                )
            return output.reshape(*original_shape, self.out_features)

        # Preserve the established correctness path for unsupported shapes and
        # dtypes. This also makes A/B testing against the v0.6 epilogue simple.
        return super().forward(inputs)
