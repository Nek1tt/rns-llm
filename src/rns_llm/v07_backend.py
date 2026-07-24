from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from rns_llm.backends import CudaRNSBackend, PreparedRNSWeight


@lru_cache(maxsize=1)
def _load_v07_extension():
    try:
        from rns_llm import _V07
    except ImportError as exc:
        raise RuntimeError(
            "v0.7 CUDA extension is not built. Run: "
            "RNS_LLM_BUILD_CUDA=1 pip install -e . --no-build-isolation"
        ) from exc
    return _V07


def v07_extension_available() -> bool:
    try:
        from rns_llm import _V07  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class V07RNSWorkspace:
    """Reusable activation residues, channel accumulators and FP16 output."""

    activation_residues: torch.Tensor  # [R,M,K], int8
    accumulators: torch.Tensor  # [R,M,N], int32
    output: torch.Tensor  # [M,N], fp16
    channels: int
    m: int
    k: int
    n: int


@dataclass
class V07NativeWorkspace:
    """Reusable quantized activation, INT32 accumulator and FP16 output."""

    activation_quantized: torch.Tensor  # [M,K], int8
    accumulators: torch.Tensor  # [M,N], int32
    output: torch.Tensor  # [M,N], fp16
    m: int
    k: int
    n: int


class V07FastPath:
    """v0.7 CUDA fast paths shared by RNS and fair native INT8 baselines.

    The RNS path retains exact INT32 channel accumulators but removes the large
    INT64 reconstructed tensor. Garner reconstruction, dequantization, bias and
    the final FP16 store are performed in one CUDA kernel.
    """

    def __init__(self, backend: CudaRNSBackend) -> None:
        self.backend = backend
        self._C = _load_v07_extension()

    @staticmethod
    def _validate_cuda_matrix(tensor: torch.Tensor, name: str, ndim: int) -> None:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def create_rns_workspace(
        self,
        *,
        device: torch.device,
        channels: int,
        m: int,
        k: int,
        n: int,
    ) -> V07RNSWorkspace:
        return V07RNSWorkspace(
            activation_residues=torch.empty(
                (channels, m, k), dtype=torch.int8, device=device
            ),
            accumulators=torch.empty(
                (channels, m, n), dtype=torch.int32, device=device
            ),
            output=torch.empty((m, n), dtype=torch.float16, device=device),
            channels=int(channels),
            m=int(m),
            k=int(k),
            n=int(n),
        )

    def create_native_workspace(
        self,
        *,
        device: torch.device,
        m: int,
        k: int,
        n: int,
    ) -> V07NativeWorkspace:
        return V07NativeWorkspace(
            activation_quantized=torch.empty((m, k), dtype=torch.int8, device=device),
            accumulators=torch.empty((m, n), dtype=torch.int32, device=device),
            output=torch.empty((m, n), dtype=torch.float16, device=device),
            m=int(m),
            k=int(k),
            n=int(n),
        )

    @staticmethod
    def _float_vector(
        tensor: torch.Tensor,
        *,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        result = tensor.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if not result.is_contiguous():
            result = result.contiguous()
        if not result.is_cuda:
            raise ValueError(f"{name} must be CUDA")
        return result

    def quantize_fp16(
        self,
        inputs: torch.Tensor,
        *,
        scales: torch.Tensor,
        quant_max: int = 127,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_cuda_matrix(inputs, "inputs", 2)
        if inputs.dtype != torch.float16:
            raise ValueError("fused quantizer expects FP16 inputs")
        scale_vector = self._float_vector(
            scales, device=inputs.device, name="scales"
        )
        if output is None:
            output = torch.empty_like(inputs, dtype=torch.int8)
        return self._C.quantize_fp16_out(
            inputs, scale_vector, int(quant_max), output
        )

    def quantize_encode_fp16(
        self,
        inputs: torch.Tensor,
        moduli: tuple[int, ...],
        *,
        scales: torch.Tensor,
        quant_max: int = 127,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_cuda_matrix(inputs, "inputs", 2)
        if inputs.dtype != torch.float16:
            raise ValueError("fused quantize/encode expects FP16 inputs")
        constants = self.backend._constants(inputs.device, moduli)
        scale_vector = self._float_vector(
            scales, device=inputs.device, name="scales"
        )
        if output is None:
            output = torch.empty(
                (len(moduli), int(inputs.shape[0]), int(inputs.shape[1])),
                dtype=torch.int8,
                device=inputs.device,
            )
        return self._C.quantize_encode_fp16_out(
            inputs, scale_vector, constants.moduli, int(quant_max), output
        )

    def native_int8_mm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        workspace: V07NativeWorkspace | None = None,
    ) -> torch.Tensor:
        self._validate_cuda_matrix(a, "a", 2)
        self._validate_cuda_matrix(b, "b", 2)
        if a.dtype != torch.int8 or b.dtype != torch.int8:
            raise ValueError("native INT8 GEMM expects torch.int8 operands")
        if a.device != b.device or int(a.shape[1]) != int(b.shape[0]):
            raise ValueError("native INT8 operands have a device/K mismatch")
        m, n = int(a.shape[0]), int(b.shape[1])
        if workspace is None:
            workspace = self.create_native_workspace(
                device=a.device, m=m, k=int(a.shape[1]), n=n
            )
        if (workspace.m, workspace.k, workspace.n) != (m, int(a.shape[1]), n):
            raise ValueError("native workspace shape mismatch")
        return self._C.native_int8_mm_out(a, b, workspace.accumulators)

    def native_int8_dequant_fp16(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        workspace: V07NativeWorkspace | None = None,
    ) -> torch.Tensor:
        self._validate_cuda_matrix(a, "a", 2)
        self._validate_cuda_matrix(b, "b", 2)
        if a.dtype != torch.int8 or b.dtype != torch.int8:
            raise ValueError("native INT8 projection expects torch.int8 operands")
        if a.device != b.device or int(a.shape[1]) != int(b.shape[0]):
            raise ValueError("native INT8 operands have a device/K mismatch")
        m, n = int(a.shape[0]), int(b.shape[1])
        if workspace is None:
            workspace = self.create_native_workspace(
                device=a.device, m=m, k=int(a.shape[1]), n=n
            )
        if (workspace.m, workspace.k, workspace.n) != (m, int(a.shape[1]), n):
            raise ValueError("native workspace shape mismatch")

        a_scale = self._float_vector(
            activation_scale, device=a.device, name="activation_scale"
        )
        w_scale = self._float_vector(
            weight_scale, device=a.device, name="weight_scale"
        )
        bias_vector = (
            torch.empty(0, dtype=torch.float32, device=a.device)
            if bias is None
            else self._float_vector(bias, device=a.device, name="bias")
        )
        return self._C.native_int8_dequant_fp16_out(
            a,
            b,
            a_scale,
            w_scale,
            bias_vector,
            workspace.accumulators,
            workspace.output,
        )

    def native_fp16_input_dequant_fp16(
        self,
        inputs: torch.Tensor,
        weight_int8: torch.Tensor,
        *,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        quant_max: int = 127,
        workspace: V07NativeWorkspace | None = None,
    ) -> torch.Tensor:
        self._validate_cuda_matrix(inputs, "inputs", 2)
        self._validate_cuda_matrix(weight_int8, "weight_int8", 2)
        if inputs.dtype != torch.float16 or weight_int8.dtype != torch.int8:
            raise ValueError("native fused projection expects FP16 input and INT8 weight")
        m, k = map(int, inputs.shape)
        n = int(weight_int8.shape[1])
        if int(weight_int8.shape[0]) != k:
            raise ValueError("native fused projection K mismatch")
        if workspace is None:
            workspace = self.create_native_workspace(
                device=inputs.device, m=m, k=k, n=n
            )
        if (workspace.m, workspace.k, workspace.n) != (m, k, n):
            raise ValueError("native workspace shape mismatch")
        quantized = self.quantize_fp16(
            inputs,
            scales=activation_scale,
            quant_max=quant_max,
            output=workspace.activation_quantized,
        )
        return self.native_int8_dequant_fp16(
            quantized,
            weight_int8,
            activation_scale=activation_scale,
            weight_scale=weight_scale,
            bias=bias,
            workspace=workspace,
        )

    def rns_fp16_input_dequant_fp16(
        self,
        inputs: torch.Tensor,
        prepared_weight: PreparedRNSWeight,
        *,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        quant_max: int = 127,
        lut_channels: int = 2,
        workspace: V07RNSWorkspace | None = None,
    ) -> torch.Tensor:
        self._validate_cuda_matrix(inputs, "inputs", 2)
        if inputs.dtype != torch.float16:
            raise ValueError("v0.7 fused input path expects FP16 inputs")
        if inputs.device != prepared_weight.residues.device:
            raise ValueError("input and prepared weight must share a device")
        m, k = map(int, inputs.shape)
        if k != prepared_weight.k:
            raise ValueError("K dimension does not match prepared weight")
        channels = len(prepared_weight.moduli)
        n = prepared_weight.n
        if workspace is None:
            workspace = self.create_rns_workspace(
                device=inputs.device, channels=channels, m=m, k=k, n=n
            )
        if (workspace.channels, workspace.m, workspace.k, workspace.n) != (
            channels, m, k, n
        ):
            raise ValueError("RNS v0.7 workspace shape mismatch")
        residues = self.quantize_encode_fp16(
            inputs,
            prepared_weight.moduli,
            scales=activation_scale,
            quant_max=quant_max,
            output=workspace.activation_residues,
        )
        self.backend._stats["v07_fused_quantize_encode_calls"] += 1
        return self.rns_encoded_dequant_fp16(
            residues,
            prepared_weight,
            activation_scale=activation_scale,
            weight_scale=weight_scale,
            bias=bias,
            lut_channels=lut_channels,
            workspace=workspace,
        )

    def rns_prepared_dequant_fp16(
        self,
        a: torch.Tensor,
        prepared_weight: PreparedRNSWeight,
        *,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        lut_channels: int = 2,
        workspace: V07RNSWorkspace | None = None,
    ) -> torch.Tensor:
        self._validate_cuda_matrix(a, "a", 2)
        if a.dtype not in (torch.int8, torch.int16, torch.int32):
            raise ValueError("a must be int8/int16/int32")
        if a.device != prepared_weight.residues.device:
            raise ValueError("activation and prepared weight must share a device")
        if int(a.shape[1]) != prepared_weight.k:
            raise ValueError("K dimension does not match prepared weight")
        if prepared_weight.kernel != "cublas":
            raise ValueError("v0.7 epilogue requires a cuBLAS-compatible weight")

        residues = self.backend.encode_centered(
            a.contiguous(), prepared_weight.moduli
        )
        return self.rns_encoded_dequant_fp16(
            residues,
            prepared_weight,
            activation_scale=activation_scale,
            weight_scale=weight_scale,
            bias=bias,
            lut_channels=lut_channels,
            workspace=workspace,
        )

    def rns_encoded_dequant_fp16(
        self,
        a_residues: torch.Tensor,
        prepared_weight: PreparedRNSWeight,
        *,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        lut_channels: int = 2,
        workspace: V07RNSWorkspace | None = None,
    ) -> torch.Tensor:
        self._validate_cuda_matrix(a_residues, "a_residues", 3)
        if a_residues.dtype != torch.int8:
            raise ValueError("a_residues must use centered int8 storage")
        channels, m, k = map(int, a_residues.shape)
        if channels != len(prepared_weight.moduli) or k != prepared_weight.k:
            raise ValueError("encoded activation shape does not match prepared weight")
        if a_residues.device != prepared_weight.residues.device:
            raise ValueError("encoded activation and weight must share a device")
        n = prepared_weight.n
        if workspace is None:
            workspace = self.create_rns_workspace(
                device=a_residues.device,
                channels=channels,
                m=m,
                k=k,
                n=n,
            )
        if (workspace.channels, workspace.m, workspace.k, workspace.n) != (
            channels, m, k, n
        ):
            raise ValueError("RNS v0.7 workspace shape mismatch")

        constants = self.backend._constants(  # intentional shared constant cache
            a_residues.device, prepared_weight.moduli
        )
        a_scale = self._float_vector(
            activation_scale, device=a_residues.device, name="activation_scale"
        )
        w_scale = self._float_vector(
            weight_scale, device=a_residues.device, name="weight_scale"
        )
        bias_vector = (
            torch.empty(0, dtype=torch.float32, device=a_residues.device)
            if bias is None
            else self._float_vector(
                bias, device=a_residues.device, name="bias"
            )
        )
        self.backend._stats["v07_fused_epilogue_calls"] += 1
        self.backend._stats[f"v07_fused_epilogue_channels_{channels}"] += 1
        return self._C.rns_matmul_dequant_fp16_out(
            a_residues,
            prepared_weight.residues,
            constants.moduli,
            constants.reciprocals,
            constants.pairwise_inverses,
            constants.compact_lut,
            constants.modulus_product,
            int(lut_channels),
            a_scale,
            w_scale,
            bias_vector,
            workspace.accumulators,
            workspace.output,
        )
