from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from typing import Iterable, Literal, Sequence

import torch

from rns_llm.adaptive import minimal_prefix_channels, safe_l1_dot_bound
from rns_llm.reference import (
    compact_byte_mod_lut,
    crt_constants,
    garner_pairwise_inverses,
    validate_moduli,
)

CanonicalKernelName = Literal["auto", "naive", "tiled", "tiled_safe"]
CenteredKernelName = Literal[
    "auto",
    "scalar",
    "dp4a",
    "dp4a_safe",
    "cublas",
]


def _load_extension():
    try:
        from rns_llm import _C
    except ImportError as exc:
        raise RuntimeError(
            "CUDA extension is not built. Run: "
            "RNS_LLM_BUILD_CUDA=1 pip install -e . --no-build-isolation"
        ) from exc
    return _C


def cuda_extension_available() -> bool:
    try:
        from rns_llm import _C  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=128)
def _host_constants(moduli: tuple[int, ...]):
    mods = validate_moduli(moduli)
    reciprocals = tuple((1 << 32) // modulus for modulus in mods)
    modulus_product, crt_coefficients = crt_constants(mods)
    pairwise = garner_pairwise_inverses(mods)
    compact_lut = compact_byte_mod_lut(mods)
    return reciprocals, modulus_product, crt_coefficients, pairwise, compact_lut


@dataclass
class _DeviceConstants:
    moduli: torch.Tensor
    reciprocals: torch.Tensor
    crt_coefficients: torch.Tensor
    pairwise_inverses: torch.Tensor
    compact_lut: torch.Tensor
    modulus_product: int


@dataclass(frozen=True)
class PreparedRNSWeight:
    """Cached Transformer weight already encoded into centered RNS planes."""

    residues: torch.Tensor
    moduli: tuple[int, ...]
    k: int
    n: int
    kernel: str


@dataclass
class RNSWorkspace:
    """Reusable temporary memory for one shape/stream/request."""

    accumulators: torch.Tensor  # [R,M,N], int32
    output: torch.Tensor  # [M,N], int64
    channels: int
    m: int
    n: int


@dataclass(frozen=True)
class PreparedAdaptiveRNSWeight:
    """One encoded weight with safe prefix variants for adaptive channels."""

    variants: dict[int, PreparedRNSWeight]
    full_moduli: tuple[int, ...]
    max_abs_weight: int
    min_channels: int


@dataclass
class RNSRequestBatchWorkspace:
    """Preallocated activation buffer plus fused GEMM workspace."""

    activations: torch.Tensor
    fused: RNSWorkspace
    rows_per_request: tuple[int, ...]
    k: int


class CudaRNSBackend:
    """CUDA backend for RNS experiments aligned with the curator scope.

    Fast path:
      centered int8 residues -> cuBLAS strided-batched INT8 GEMM -> one fused
      modulo + mixed-radix reconstruction kernel.

    The compact modulo LUT is optional.  `lut_channels=1/2` reuses only the
    largest one or two [4,256] byte-decomposition tables, matching the project
    requirement to limit table count and measure memory/bus contention.
    """

    name = "cuda-rns-v05"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this PyTorch installation")
        self._C = _load_extension()
        self._device_cache: dict[tuple[int, tuple[int, ...]], _DeviceConstants] = {}
        self._stats: Counter[str] = Counter()

    def reset_stats(self) -> None:
        self._stats.clear()

    def stats_snapshot(self) -> dict[str, int]:
        return dict(self._stats)

    def _constants(self, device: torch.device, moduli: Iterable[int]) -> _DeviceConstants:
        mods = validate_moduli(moduli)
        index = device.index if device.index is not None else torch.cuda.current_device()
        key = (index, mods)
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached

        reciprocals, modulus_product, crt_coefficients, pairwise, compact_lut = (
            _host_constants(mods)
        )
        if modulus_product > 2**63 - 1:
            raise OverflowError("moduli product must fit signed int64")

        cached = _DeviceConstants(
            moduli=torch.tensor(mods, dtype=torch.int32, device=device),
            reciprocals=torch.tensor(reciprocals, dtype=torch.int64, device=device),
            crt_coefficients=torch.tensor(
                crt_coefficients, dtype=torch.int64, device=device
            ),
            pairwise_inverses=torch.tensor(
                pairwise, dtype=torch.int32, device=device
            ),
            compact_lut=torch.as_tensor(
                compact_lut, dtype=torch.int16, device=device
            ).contiguous(),
            modulus_product=modulus_product,
        )
        self._device_cache[key] = cached
        return cached

    @staticmethod
    def _validate_matrix(matrix: torch.Tensor, name: str, ndim: int) -> None:
        if not matrix.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if matrix.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}")
        if not matrix.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, values: torch.Tensor, moduli: Iterable[int]) -> torch.Tensor:
        """Legacy canonical [0,m) encoder used by old uint8 kernels."""
        self._validate_matrix(values, "values", values.ndim)
        if values.dtype != torch.int8:
            raise ValueError("canonical CUDA encoder currently expects torch.int8")
        constants = self._constants(values.device, moduli)
        return self._C.encode_int8(values, constants.moduli)

    def encode_centered(
        self,
        values: torch.Tensor,
        moduli: Iterable[int],
    ) -> torch.Tensor:
        """Encode int8/int16/int32 values as signed-int8 centered residues."""
        self._validate_matrix(values, "values", values.ndim)
        if values.dtype not in (torch.int8, torch.int16, torch.int32):
            raise ValueError("centered encoder supports int8, int16 and int32")
        constants = self._constants(values.device, moduli)
        self._stats["encode_centered_calls"] += 1
        self._stats["encode_centered_values"] += int(values.numel())
        return self._C.encode_centered(values, constants.moduli)

    # ------------------------------------------------------------------
    # Bounds and selection
    # ------------------------------------------------------------------
    @staticmethod
    def accumulator_bound(k: int, moduli: Iterable[int]) -> int:
        mods = validate_moduli(moduli)
        return int(k) * (max(mods) - 1) ** 2

    @staticmethod
    def centered_accumulator_bound(k: int, moduli: Iterable[int]) -> int:
        mods = validate_moduli(moduli)
        max_centered = max(modulus // 2 for modulus in mods)
        return int(k) * max_centered**2

    def select_kernel(
        self,
        k: int,
        moduli: Iterable[int],
        requested: CanonicalKernelName,
    ) -> str:
        if requested != "auto":
            return requested
        return "tiled" if self.accumulator_bound(k, moduli) <= 0xFFFFFFFF else "tiled_safe"

    def cublas_compatible(
        self,
        *,
        k: int,
        n: int,
        moduli: Iterable[int],
        device: torch.device,
    ) -> bool:
        capability = torch.cuda.get_device_capability(device)
        return (
            capability >= (6, 1)
            and k % 4 == 0
            and n % 4 == 0
            and 2 <= len(validate_moduli(moduli)) <= 10
            and self.centered_accumulator_bound(k, moduli) <= 0x7FFFFFFF
        )

    def select_centered_kernel(
        self,
        *,
        k: int,
        n: int,
        moduli: Iterable[int],
        requested: CenteredKernelName,
        device: torch.device,
    ) -> str:
        if requested != "auto":
            return requested
        if self.cublas_compatible(k=k, n=n, moduli=moduli, device=device):
            return "cublas"
        capability = torch.cuda.get_device_capability(device)
        if capability >= (6, 1):
            if self.centered_accumulator_bound(k, moduli) <= 0x7FFFFFFF:
                return "dp4a"
            return "dp4a_safe"
        return "scalar"

    # ------------------------------------------------------------------
    # Legacy/correctness GEMMs
    # ------------------------------------------------------------------
    def matmul_residues(
        self,
        a_residues: torch.Tensor,
        b_residues: torch.Tensor,
        moduli: Iterable[int],
        *,
        kernel: CanonicalKernelName = "auto",
    ) -> torch.Tensor:
        self._validate_matrix(a_residues, "a_residues", 3)
        self._validate_matrix(b_residues, "b_residues", 3)
        if a_residues.dtype != torch.uint8 or b_residues.dtype != torch.uint8:
            raise ValueError("canonical residue matrices must use torch.uint8")
        if a_residues.device != b_residues.device:
            raise ValueError("a_residues and b_residues must be on the same device")
        if a_residues.shape[0] != b_residues.shape[0]:
            raise ValueError("residue channel counts do not match")
        if a_residues.shape[2] != b_residues.shape[1]:
            raise ValueError("K dimensions do not match")

        mods = validate_moduli(moduli)
        if a_residues.shape[0] != len(mods):
            raise ValueError("tensor channel count does not match moduli")

        constants = self._constants(a_residues.device, mods)
        selected = self.select_kernel(a_residues.shape[2], mods, kernel)
        kernel_id = {"naive": 0, "tiled": 1, "tiled_safe": 2}[selected]
        return self._C.matmul_residues(
            a_residues,
            b_residues,
            constants.moduli,
            constants.reciprocals,
            kernel_id,
        )

    def matmul_centered_residues(
        self,
        a_residues: torch.Tensor,
        b_residues: torch.Tensor,
        moduli: Iterable[int],
        *,
        kernel: CenteredKernelName = "auto",
    ) -> torch.Tensor:
        self._validate_matrix(a_residues, "a_residues", 3)
        self._validate_matrix(b_residues, "b_residues", 3)
        if a_residues.dtype != torch.int8 or b_residues.dtype != torch.int8:
            raise ValueError("centered residue matrices must use torch.int8")
        if a_residues.device != b_residues.device:
            raise ValueError("a_residues and b_residues must share a device")
        if a_residues.shape[0] != b_residues.shape[0]:
            raise ValueError("channel counts do not match")
        if a_residues.shape[2] != b_residues.shape[1]:
            raise ValueError("K dimensions do not match")

        mods = validate_moduli(moduli)
        if len(mods) != a_residues.shape[0]:
            raise ValueError("moduli count does not match residue channels")

        k = int(a_residues.shape[2])
        n = int(b_residues.shape[2])
        selected = self.select_centered_kernel(
            k=k,
            n=n,
            moduli=mods,
            requested=kernel,
            device=a_residues.device,
        )
        if selected == "cublas" and not self.cublas_compatible(
            k=k, n=n, moduli=mods, device=a_residues.device
        ):
            raise ValueError("cuBLAS path constraints are not satisfied")

        constants = self._constants(a_residues.device, mods)
        kernel_id = {"scalar": 0, "dp4a": 1, "dp4a_safe": 2, "cublas": 3}[selected]
        return self._C.matmul_centered_residues(
            a_residues,
            b_residues,
            constants.moduli,
            kernel_id,
        )

    # ------------------------------------------------------------------
    # Fused non-modular operations: modulo + Garner reconstruction
    # ------------------------------------------------------------------
    def decode_garner(
        self,
        residues: torch.Tensor,
        moduli: Iterable[int],
    ) -> torch.Tensor:
        self._validate_matrix(residues, "residues", residues.ndim)
        if residues.dtype != torch.int8:
            raise ValueError("Garner decoder expects centered int8 residues")
        constants = self._constants(residues.device, moduli)
        return self._C.decode_garner(
            residues,
            constants.moduli,
            constants.reciprocals,
            constants.pairwise_inverses,
            constants.modulus_product,
        )

    def create_workspace(
        self,
        *,
        device: torch.device,
        channels: int,
        m: int,
        n: int,
    ) -> RNSWorkspace:
        return RNSWorkspace(
            accumulators=torch.empty(
                (channels, m, n), dtype=torch.int32, device=device
            ),
            output=torch.empty((m, n), dtype=torch.int64, device=device),
            channels=channels,
            m=m,
            n=n,
        )

    @staticmethod
    def _check_workspace(
        workspace: RNSWorkspace,
        *,
        channels: int,
        m: int,
        n: int,
        device: torch.device,
    ) -> None:
        if (workspace.channels, workspace.m, workspace.n) != (channels, m, n):
            raise ValueError("workspace shape does not match operation")
        if workspace.accumulators.device != device or workspace.output.device != device:
            raise ValueError("workspace device does not match operation")

    def matmul_centered_fused(
        self,
        a_residues: torch.Tensor,
        b_residues: torch.Tensor,
        moduli: Iterable[int],
        *,
        lut_channels: int = 0,
        workspace: RNSWorkspace | None = None,
    ) -> torch.Tensor:
        """cuBLAS INT8 GEMM + one fused modulo/Garner kernel.

        No intermediate int8 residue output is written.  `lut_channels` may be
        0, 1 or 2; tables are shared/cached and only used for the first (largest)
        moduli in the chosen set.
        """
        self._validate_matrix(a_residues, "a_residues", 3)
        self._validate_matrix(b_residues, "b_residues", 3)
        if a_residues.dtype != torch.int8 or b_residues.dtype != torch.int8:
            raise ValueError("fused path expects centered int8 residue planes")
        if a_residues.device != b_residues.device:
            raise ValueError("inputs must share device")
        if a_residues.shape[0] != b_residues.shape[0] or a_residues.shape[2] != b_residues.shape[1]:
            raise ValueError("R/K dimensions do not match")
        if lut_channels not in (0, 1, 2):
            raise ValueError("lut_channels must be 0, 1 or 2")

        mods = validate_moduli(moduli)
        channels, m, k = map(int, a_residues.shape)
        n = int(b_residues.shape[2])
        if channels != len(mods):
            raise ValueError("moduli count does not match channels")
        if not self.cublas_compatible(
            k=k, n=n, moduli=mods, device=a_residues.device
        ):
            raise ValueError("fused path requires a cuBLAS-compatible shape/device")

        if workspace is None:
            workspace = self.create_workspace(
                device=a_residues.device, channels=channels, m=m, n=n
            )
        self._check_workspace(
            workspace,
            channels=channels,
            m=m,
            n=n,
            device=a_residues.device,
        )
        constants = self._constants(a_residues.device, mods)
        self._stats["fused_gemm_calls"] += 1
        self._stats[f"fused_gemm_channels_{channels}"] += 1
        return self._C.matmul_centered_fused_out(
            a_residues,
            b_residues,
            constants.moduli,
            constants.reciprocals,
            constants.pairwise_inverses,
            constants.compact_lut,
            constants.modulus_product,
            lut_channels,
            workspace.accumulators,
            workspace.output,
        )

    # ------------------------------------------------------------------
    # Prepared weights and end-to-end helpers
    # ------------------------------------------------------------------
    def prepare_weight(
        self,
        b: torch.Tensor,
        moduli: Iterable[int],
    ) -> PreparedRNSWeight:
        self._validate_matrix(b, "b", 2)
        if b.dtype not in (torch.int8, torch.int16, torch.int32):
            raise ValueError("prepared weights support int8/int16/int32")
        mods = validate_moduli(moduli)
        residues = self.encode_centered(b.contiguous(), mods)
        kernel = self.select_centered_kernel(
            k=int(b.shape[0]),
            n=int(b.shape[1]),
            moduli=mods,
            requested="auto",
            device=b.device,
        )
        return PreparedRNSWeight(
            residues=residues,
            moduli=mods,
            k=int(b.shape[0]),
            n=int(b.shape[1]),
            kernel=kernel,
        )

    def matmul_prepared(
        self,
        a: torch.Tensor,
        prepared_weight: PreparedRNSWeight,
        *,
        decode: bool = True,
        kernel: CenteredKernelName = "auto",
    ) -> torch.Tensor:
        self._validate_matrix(a, "a", 2)
        if a.dtype not in (torch.int8, torch.int16, torch.int32):
            raise ValueError("a must be int8/int16/int32")
        if int(a.shape[1]) != prepared_weight.k:
            raise ValueError("K dimension does not match prepared weight")
        if a.device != prepared_weight.residues.device:
            raise ValueError("activation and prepared weight must share device")

        a_residues = self.encode_centered(a.contiguous(), prepared_weight.moduli)
        result_residues = self.matmul_centered_residues(
            a_residues,
            prepared_weight.residues,
            prepared_weight.moduli,
            kernel=kernel,
        )
        return self.decode(result_residues, prepared_weight.moduli) if decode else result_residues

    def matmul_prepared_fused(
        self,
        a: torch.Tensor,
        prepared_weight: PreparedRNSWeight,
        *,
        lut_channels: int = 0,
        workspace: RNSWorkspace | None = None,
    ) -> torch.Tensor:
        self._validate_matrix(a, "a", 2)
        if a.dtype not in (torch.int8, torch.int16, torch.int32):
            raise ValueError("a must be int8/int16/int32")
        if int(a.shape[1]) != prepared_weight.k:
            raise ValueError("K dimension does not match prepared weight")
        if a.device != prepared_weight.residues.device:
            raise ValueError("activation and prepared weight must share device")
        if prepared_weight.kernel != "cublas":
            raise ValueError("fused prepared path currently requires cuBLAS-compatible weight")

        a_residues = self.encode_centered(a.contiguous(), prepared_weight.moduli)
        return self.matmul_centered_fused(
            a_residues,
            prepared_weight.residues,
            prepared_weight.moduli,
            lut_channels=lut_channels,
            workspace=workspace,
        )

    def prepare_weight_adaptive(
        self,
        b: torch.Tensor,
        moduli: Iterable[int],
        *,
        min_channels: int = 3,
    ) -> PreparedAdaptiveRNSWeight:
        """Encode once and expose safe prefix variants without duplicating memory."""
        full = self.prepare_weight(b, moduli)
        if not 2 <= min_channels <= len(full.moduli):
            raise ValueError("min_channels must be between 2 and full channel count")
        variants: dict[int, PreparedRNSWeight] = {}
        for channels in range(min_channels, len(full.moduli) + 1):
            prefix = full.moduli[:channels]
            selected = self.select_centered_kernel(
                k=full.k,
                n=full.n,
                moduli=prefix,
                requested="auto",
                device=full.residues.device,
            )
            variants[channels] = PreparedRNSWeight(
                residues=full.residues[:channels],
                moduli=prefix,
                k=full.k,
                n=full.n,
                kernel=selected,
            )
        max_abs_weight = int(b.to(torch.int32).abs().max().item())
        return PreparedAdaptiveRNSWeight(
            variants=variants,
            full_moduli=full.moduli,
            max_abs_weight=max_abs_weight,
            min_channels=min_channels,
        )

    def select_adaptive_variant(
        self,
        a: torch.Tensor,
        prepared: PreparedAdaptiveRNSWeight,
    ) -> tuple[PreparedRNSWeight, int]:
        """Choose a mathematically safe prefix using an L1 dot-product bound.

        This performs a device-to-host scalar synchronization.  It is intended
        for correctness-first experiments and calibration; benchmark the sync
        cost before enabling it in latency-critical serving.
        """
        self._validate_matrix(a, "a", 2)
        if a.dtype not in (torch.int8, torch.int16, torch.int32):
            raise ValueError("a must be int8/int16/int32")
        max_row_l1 = int(a.to(torch.int32).abs().sum(dim=1).max().item())
        bound = safe_l1_dot_bound(max_row_l1, prepared.max_abs_weight)
        channels = minimal_prefix_channels(
            prepared.full_moduli,
            bound,
            min_channels=prepared.min_channels,
        )
        self._stats["adaptive_selections"] += 1
        self._stats[f"adaptive_channels_{channels}"] += 1
        return prepared.variants[channels], bound

    def matmul_prepared_adaptive_fused(
        self,
        a: torch.Tensor,
        prepared: PreparedAdaptiveRNSWeight,
        *,
        lut_channels: int = 2,
        workspace_by_channels: dict[int, RNSWorkspace] | None = None,
        return_metadata: bool = False,
    ):
        variant, bound = self.select_adaptive_variant(a, prepared)
        workspace = None if workspace_by_channels is None else workspace_by_channels.get(len(variant.moduli))
        output = self.matmul_prepared_fused(
            a, variant, lut_channels=lut_channels, workspace=workspace
        )
        if return_metadata:
            return output, {
                "channels": len(variant.moduli),
                "bound": bound,
                "capacity": (prod(variant.moduli) - 1) // 2,
            }
        return output

    def create_request_batch_workspace(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rows_per_request: Sequence[int],
        k: int,
        channels: int,
        n: int,
    ) -> RNSRequestBatchWorkspace:
        rows = tuple(int(x) for x in rows_per_request)
        if not rows or any(x <= 0 for x in rows):
            raise ValueError("rows_per_request must contain positive sizes")
        total_rows = sum(rows)
        return RNSRequestBatchWorkspace(
            activations=torch.empty((total_rows, k), dtype=dtype, device=device),
            fused=self.create_workspace(
                device=device, channels=channels, m=total_rows, n=n
            ),
            rows_per_request=rows,
            k=int(k),
        )

    def matmul_prepared_fused_requests(
        self,
        requests: Sequence[torch.Tensor],
        prepared_weight: PreparedRNSWeight,
        *,
        lut_channels: int = 2,
        workspace: RNSRequestBatchWorkspace | None = None,
        clone_outputs: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Continuous-batching prototype: concatenate M and run one RNS GEMM.

        All requests must share K, dtype, device and weights. Queueing delay is
        deliberately outside this primitive and must be reported separately in
        a serving benchmark.
        """
        if not requests:
            raise ValueError("requests must not be empty")
        first = requests[0]
        rows = tuple(int(x.shape[0]) for x in requests)
        for index, tensor in enumerate(requests):
            self._validate_matrix(tensor, f"requests[{index}]", 2)
            if tensor.dtype != first.dtype or tensor.device != first.device:
                raise ValueError("all requests must share dtype/device")
            if int(tensor.shape[1]) != prepared_weight.k:
                raise ValueError("request K does not match prepared weight")
        if workspace is None:
            workspace = self.create_request_batch_workspace(
                device=first.device,
                dtype=first.dtype,
                rows_per_request=rows,
                k=prepared_weight.k,
                channels=len(prepared_weight.moduli),
                n=prepared_weight.n,
            )
        if workspace.rows_per_request != rows or workspace.k != prepared_weight.k:
            raise ValueError("request batch workspace does not match request shapes")

        offset = 0
        for tensor in requests:
            next_offset = offset + int(tensor.shape[0])
            workspace.activations[offset:next_offset].copy_(tensor)
            offset = next_offset
        merged = self.matmul_prepared_fused(
            workspace.activations,
            prepared_weight,
            lut_channels=lut_channels,
            workspace=workspace.fused,
        )
        outputs = []
        offset = 0
        for row_count in rows:
            view = merged[offset : offset + row_count]
            outputs.append(view.clone() if clone_outputs else view)
            offset += row_count
        self._stats["continuous_batch_calls"] += 1
        self._stats["continuous_batch_requests"] += len(requests)
        return tuple(outputs)

    # Original CRT path retained as a benchmark/reference.
    def decode(
        self,
        residues: torch.Tensor,
        moduli: Iterable[int],
        *,
        signed: bool = True,
    ) -> torch.Tensor:
        self._validate_matrix(residues, "residues", residues.ndim)
        if residues.dtype not in (torch.uint8, torch.int8):
            raise ValueError("decoder expects uint8 canonical or int8 centered residues")
        constants = self._constants(residues.device, moduli)
        result = torch.zeros(residues.shape[1:], dtype=torch.int64, device=residues.device)
        for channel in range(residues.shape[0]):
            result = torch.remainder(
                result
                + residues[channel].to(torch.int64)
                * constants.crt_coefficients[channel],
                constants.modulus_product,
            )
        if signed:
            result = torch.where(
                result > constants.modulus_product // 2,
                result - constants.modulus_product,
                result,
            )
        return result

    def matmul_wide(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        moduli: Iterable[int],
        *,
        decode: bool = True,
        kernel: CenteredKernelName = "auto",
    ) -> torch.Tensor:
        self._validate_matrix(a, "a", 2)
        self._validate_matrix(b, "b", 2)
        if a.dtype not in (torch.int8, torch.int16, torch.int32):
            raise ValueError("a must be int8/int16/int32")
        if b.dtype != a.dtype:
            raise ValueError("a and b must use the same integer dtype")
        if a.shape[1] != b.shape[0]:
            raise ValueError("K dimensions do not match")

        mods = validate_moduli(moduli)
        a_residues = self.encode_centered(a.contiguous(), mods)
        b_residues = self.encode_centered(b.contiguous(), mods)
        output = self.matmul_centered_residues(
            a_residues,
            b_residues,
            mods,
            kernel=kernel,
        )
        return self.decode(output, mods) if decode else output

    def matmul_wide_fused(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        moduli: Iterable[int],
        *,
        lut_channels: int = 0,
        workspace: RNSWorkspace | None = None,
    ) -> torch.Tensor:
        self._validate_matrix(a, "a", 2)
        self._validate_matrix(b, "b", 2)
        if a.dtype not in (torch.int8, torch.int16, torch.int32):
            raise ValueError("a must be int8/int16/int32")
        if b.dtype != a.dtype or a.shape[1] != b.shape[0]:
            raise ValueError("dtype/K mismatch")
        mods = validate_moduli(moduli)
        a_residues = self.encode_centered(a.contiguous(), mods)
        b_residues = self.encode_centered(b.contiguous(), mods)
        return self.matmul_centered_fused(
            a_residues,
            b_residues,
            mods,
            lut_channels=lut_channels,
            workspace=workspace,
        )

    def matmul_int8(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        moduli: Iterable[int],
        *,
        decode: bool = True,
        kernel: CenteredKernelName = "auto",
    ) -> torch.Tensor:
        if a.dtype != torch.int8 or b.dtype != torch.int8:
            raise ValueError("matmul_int8 expects int8 inputs")
        return self.matmul_wide(a, b, moduli, decode=decode, kernel=kernel)
