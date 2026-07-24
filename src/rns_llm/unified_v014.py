from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from rns_llm.backends import CudaRNSBackend
from rns_llm.architecture_v013 import (
    NativeInt8Runner,
    RNSArchitectureRunner,
    build_compact_lut,
    prepare_int8_weight,
    prepare_rns_weight,
    select_plan,
    tensor_bytes,
)
from rns_llm.layers.rns_linear_v07 import FastRNSLinearV07
from rns_llm.layers.rns_qkv import CachedRNSQKV, RNSQKVProjection
from rns_llm.prefill_v011 import PrefillLayerV011


@contextlib.contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def _empty_or_cat_bias(linears: Sequence[nn.Linear]) -> torch.Tensor | None:
    if all(layer.bias is None for layer in linears):
        return None
    if any(layer.bias is None for layer in linears):
        raise ValueError("Q/K/V must consistently use or omit bias")
    return torch.cat([layer.bias.detach() for layer in linears], dim=0)


def select_protected_channels(
    weight_nk: torch.Tensor,
    *,
    protected_channels: int | None = None,
    protected_ratio: float = 0.0012,
    provided: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Select deterministic high-risk input channels.

    A calibration plan can pass explicit indices.  The fallback ranks input
    dimensions by output-aggregated absolute weight energy, which is stable and
    requires no model-specific data.  The PPL script uses calibrated indices
    when available.
    """
    k = int(weight_nk.shape[1])
    if provided is not None:
        result = torch.as_tensor(provided, dtype=torch.long).unique(sorted=True)
    else:
        count = protected_channels
        if count is None:
            count = max(1, math.ceil(k * float(protected_ratio)))
        count = max(1, min(k - 1, int(count)))
        score = weight_nk.detach().float().abs().sum(dim=0)
        result = torch.topk(score, count, largest=True).indices.sort().values.cpu()
    if result.numel() == 0:
        raise ValueError("protected channel set cannot be empty")
    if int(result.min()) < 0 or int(result.max()) >= k:
        raise ValueError("protected index is out of bounds")
    return result.to(dtype=torch.int32, device="cpu").contiguous()


class FullRNSLinearV014(nn.Module):
    """Actual full-RNS Linear for q8/q16/q32 with reusable prepared weights.

    q8 can optionally use the more optimized v0.7 fused FP16 epilogue.  q16 and
    q32 use the v0.13 two-limb Garner implementation.  Runners are cached per M
    and CUDA stream so attention/PPL calls avoid reallocating workspaces.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        logical_bits: int = 8,
        lut_channels: int = 2,
        moduli_policy: str = "dense_coprime",
        q8_backend: str = "v07",
    ) -> None:
        super().__init__()
        if logical_bits not in (8, 16, 32):
            raise ValueError("logical_bits must be 8, 16 or 32")
        if q8_backend not in {"v07", "v013"}:
            raise ValueError("q8_backend must be v07 or v013")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.logical_bits = int(logical_bits)
        self.lut_channels = int(lut_channels)
        self.moduli_policy = str(moduli_policy)
        self.q8_backend = q8_backend
        self.calls = 0
        self._runner_cache: dict[tuple[int, int], RNSArchitectureRunner] = {}
        self._lut_cache: dict[int, torch.Tensor] = {}

        if logical_bits == 8 and q8_backend == "v07":
            self.v07 = FastRNSLinearV07.from_linear(
                linear,
                backend=CudaRNSBackend(),
                mode="rns",
                quant_bits=8,
                fused=True,
                lut_channels=lut_channels,
                moduli_strategy="dense_coprime",
                use_v07_epilogue=True,
                fuse_quantize_encode=True,
            ).eval()
            self.register_buffer("bias_value", torch.empty(0), persistent=False)
            self.prepared = None
            self.plan = None
        else:
            self.v07 = None
            weight_kn = linear.weight.detach().float().transpose(0, 1).contiguous()
            self.plan = select_plan(self.in_features, logical_bits, moduli_policy)
            self.prepared = prepare_rns_weight(weight_kn, self.plan)
            bias = (
                torch.empty(0, dtype=torch.float32, device=linear.weight.device)
                if linear.bias is None
                else linear.bias.detach().float().contiguous()
            )
            self.register_buffer("bias_value", bias, persistent=False)

    @classmethod
    def from_linears(
        cls,
        linears: Sequence[nn.Linear],
        **kwargs,
    ) -> tuple["FullRNSLinearV014", tuple[int, ...]]:
        if not linears:
            raise ValueError("linears cannot be empty")
        if len({layer.in_features for layer in linears}) != 1:
            raise ValueError("all linears must share in_features")
        combined = nn.Linear(
            linears[0].in_features,
            sum(layer.out_features for layer in linears),
            bias=linears[0].bias is not None,
            device=linears[0].weight.device,
            dtype=linears[0].weight.dtype,
        )
        with torch.no_grad():
            combined.weight.copy_(torch.cat([layer.weight for layer in linears], dim=0))
            bias = _empty_or_cat_bias(linears)
            if bias is not None:
                combined.bias.copy_(bias)
        return cls(combined, **kwargs), tuple(layer.out_features for layer in linears)

    def _runner(self, m: int) -> RNSArchitectureRunner:
        assert self.prepared is not None and self.plan is not None
        stream_id = int(torch.cuda.current_stream(self.prepared.device).cuda_stream)
        key = (int(m), stream_id)
        runner = self._runner_cache.get(key)
        if runner is None:
            active = min(self.lut_channels, self.plan.channels)
            lut = self._lut_cache.get(active)
            if lut is None:
                lut = build_compact_lut(
                    self.plan.moduli, active, device=self.prepared.device
                )
                self._lut_cache[active] = lut
            runner = RNSArchitectureRunner(
                self.prepared, m=m, lut_channels=active, compact_lut=lut
            )
            self._runner_cache[key] = runner
        return runner

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.v07 is not None:
            with nvtx_range("full_rns_q8_v07_linear"):
                return self.v07(inputs)
        original_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).float().contiguous()
        runner = self._runner(int(flat.shape[0]))
        with nvtx_range(f"full_rns_q{self.logical_bits}_linear"):
            output = runner.e2e(flat)
            if self.bias_value.numel():
                output = output + self.bias_value
        return output.to(dtype=inputs.dtype).reshape(*original_shape, self.out_features)

    def memory_report(self) -> dict[str, int | str]:
        if self.v07 is not None:
            self.v07.prepare_weight()
            prepared = self.v07._prepared_weight
            residue = 0 if prepared is None else tensor_bytes(prepared.residues)
            channels = 0 if prepared is None else len(prepared.moduli)
            registered = sum(
                tensor_bytes(tensor)
                for tensor in list(self.v07.parameters()) + list(self.v07.buffers())
            )
            return {
                "architecture": "full_rns",
                "logical_bits": 8,
                "channels": channels,
                "weight_bytes": registered + residue,
                "selected_rns_representation_bytes": residue,
                "legacy_retains_fp16_master_weight": True,
                "lut_active_bytes": min(self.lut_channels, channels) * 4 * 256 * 2,
                "lut_allocated_bytes": channels * 4 * 256 * 2,
                "runner_workspace_bytes": sum(
                    sum(tensor_bytes(v) for v in vars(ws).values() if torch.is_tensor(v))
                    for ws in self.v07._v07_workspace_by_rows.values()
                ),
            }
        assert self.prepared is not None and self.plan is not None
        bias_bytes = tensor_bytes(self.bias_value)
        return {
            "architecture": "full_rns",
            "logical_bits": self.logical_bits,
            "channels": self.plan.channels,
            "weight_bytes": self.prepared.storage_bytes + bias_bytes,
            "bias_bytes": bias_bytes,
            "lut_active_bytes": min(self.lut_channels, self.plan.channels) * 4 * 256 * 2,
            "lut_allocated_bytes": sum(t.numel() * t.element_size() for t in self._lut_cache.values()),
            "runner_workspace_bytes": sum(r.runtime_workspace_bytes for r in self._runner_cache.values()),
        }


class NativeInt8LinearV014(nn.Module):
    """Per-row/per-output symmetric INT8 baseline using the v0.13 CUDA path."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        weight_kn = linear.weight.detach().float().transpose(0, 1).contiguous()
        self.prepared = prepare_int8_weight(weight_kn)
        bias = (
            torch.empty(0, dtype=torch.float32, device=linear.weight.device)
            if linear.bias is None
            else linear.bias.detach().float().contiguous()
        )
        self.register_buffer("bias_value", bias, persistent=False)
        self._runner_cache: dict[tuple[int, int], NativeInt8Runner] = {}
        self.calls = 0

    @classmethod
    def from_linears(
        cls, linears: Sequence[nn.Linear]
    ) -> tuple["NativeInt8LinearV014", tuple[int, ...]]:
        if not linears:
            raise ValueError("linears cannot be empty")
        if len({layer.in_features for layer in linears}) != 1:
            raise ValueError("all linears must share in_features")
        combined = nn.Linear(
            linears[0].in_features,
            sum(layer.out_features for layer in linears),
            bias=linears[0].bias is not None,
            device=linears[0].weight.device,
            dtype=linears[0].weight.dtype,
        )
        with torch.no_grad():
            combined.weight.copy_(torch.cat([layer.weight for layer in linears], dim=0))
            bias = _empty_or_cat_bias(linears)
            if bias is not None:
                combined.bias.copy_(bias)
        return cls(combined), tuple(layer.out_features for layer in linears)

    def _runner(self, m: int) -> NativeInt8Runner:
        stream_id = int(torch.cuda.current_stream(self.prepared.quantized_kn.device).cuda_stream)
        key = (int(m), stream_id)
        runner = self._runner_cache.get(key)
        if runner is None:
            runner = NativeInt8Runner(self.prepared, m=int(m))
            self._runner_cache[key] = runner
        return runner

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        original_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).float().contiguous()
        with nvtx_range("native_int8_linear"):
            output = self._runner(int(flat.shape[0])).e2e(flat)
            if self.bias_value.numel():
                output = output + self.bias_value
        return output.to(dtype=inputs.dtype).reshape(*original_shape, self.out_features)

    def memory_report(self) -> dict[str, int | str]:
        bias_bytes = tensor_bytes(self.bias_value)
        return {
            "architecture": "native_int8",
            "weight_bytes": self.prepared.storage_bytes + bias_bytes,
            "bias_bytes": bias_bytes,
            "lut_active_bytes": 0,
            "lut_allocated_bytes": 0,
            "runner_workspace_bytes": sum(
                runner.runtime_workspace_bytes for runner in self._runner_cache.values()
            ),
        }


class HybridLinearV014(nn.Module):
    """v0.11 fused INT8-main + protected correction as a model Linear."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        protected_indices: Sequence[int] | torch.Tensor | None = None,
        protected_channels: int | None = None,
        protected_ratio: float = 0.0012,
        correction_bits: int = 16,
        lut_channels: int = 2,
        correction: str = "rns",
        execution: str = "serial",
        workspace_bytes: int = 32 * 1024 * 1024,
        layer_name: str = "hybrid_linear",
    ) -> None:
        super().__init__()
        if correction not in {"rns", "fp16"}:
            raise ValueError("correction must be rns or fp16; use NativeInt8LinearV014 for native INT8")
        if execution not in {"serial", "parallel"}:
            raise ValueError("execution must be serial or parallel")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.correction_bits = int(correction_bits)
        self.lut_channels = int(lut_channels)
        self.correction = correction
        self.execution = execution
        self.workspace_bytes = int(workspace_bytes)
        self.calls = 0
        protected = select_protected_channels(
            linear.weight,
            protected_channels=protected_channels,
            protected_ratio=protected_ratio,
            provided=protected_indices,
        )
        pack = {
            "layer_name": layer_name,
            "weight": linear.weight.detach().float().cpu(),
            "bias": None if linear.bias is None else linear.bias.detach().float().cpu(),
            "protected_indices": protected,
            "selected_plan": {"source": "v014", "protected_indices": protected.tolist()},
            "statistics": {},
        }
        optimized_bits = (self.correction_bits,) if self.correction == "rns" else ()
        self.layer = PrefillLayerV011.from_pack(
            pack,
            device=linear.weight.device,
            optimized_rns_bits=optimized_bits,
            storage_mode="rns" if self.correction == "rns" else "fp16",
        )
        self._runner_cache: dict[tuple[int, int], object] = {}

    @classmethod
    def from_linears(
        cls,
        linears: Sequence[nn.Linear],
        *,
        protected_indices: Sequence[int] | torch.Tensor | None = None,
        **kwargs,
    ) -> tuple["HybridLinearV014", tuple[int, ...]]:
        if not linears:
            raise ValueError("linears cannot be empty")
        combined = nn.Linear(
            linears[0].in_features,
            sum(layer.out_features for layer in linears),
            bias=linears[0].bias is not None,
            device=linears[0].weight.device,
            dtype=linears[0].weight.dtype,
        )
        with torch.no_grad():
            combined.weight.copy_(torch.cat([layer.weight for layer in linears], dim=0))
            bias = _empty_or_cat_bias(linears)
            if bias is not None:
                combined.bias.copy_(bias)
        return cls(combined, protected_indices=protected_indices, **kwargs), tuple(
            layer.out_features for layer in linears
        )

    def _runner(self, m: int):
        stream_id = int(torch.cuda.current_stream(self.layer.device).cuda_stream)
        key = (int(m), stream_id)
        runner = self._runner_cache.get(key)
        if runner is None:
            if self.correction == "rns":
                rns_weight = self.layer.protected_rns[self.correction_bits]
                logical_bits: int | None = self.correction_bits
                active_lut = min(self.lut_channels, len(rns_weight.moduli))
                mode = "rns"
            else:
                logical_bits = None
                active_lut = 0
                mode = "fp16"
            runner = self.layer.runner(
                m,
                logical_bits=logical_bits,
                workspace_bytes=self.workspace_bytes,
                lut_channels=active_lut,
                mode=mode,
            )
            self._runner_cache[key] = runner
        return runner

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        original_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).float().contiguous()
        runner = self._runner(int(flat.shape[0]))
        with nvtx_range(f"hybrid_{self.correction}_{self.execution}_linear"):
            if self.correction == "fp16":
                output = (
                    runner.hybrid_fp16_serial_e2e(flat)
                    if self.execution == "serial"
                    else runner.hybrid_fp16_parallel_e2e(flat)
                )
            else:
                output = (
                    runner.hybrid_rns_serial_e2e(flat)
                    if self.execution == "serial"
                    else runner.hybrid_rns_parallel_e2e(flat)
                )
        return output.to(dtype=inputs.dtype).reshape(*original_shape, self.out_features)

    def memory_report(self) -> dict[str, int | str]:
        base = next(iter(self._runner_cache.values()), None)
        if base is None:
            if self.correction == "rns":
                bits: int | None = self.correction_bits
                active_lut = min(
                    self.lut_channels,
                    len(self.layer.protected_rns[self.correction_bits].moduli),
                )
                mode = "rns"
            else:
                bits = None
                active_lut = 0
                mode = "fp16"
            base = self.layer.runner(
                1,
                logical_bits=bits,
                workspace_bytes=self.workspace_bytes,
                lut_channels=active_lut,
                mode=mode,
            )
        storage = base.storage_bytes()
        weight_key = (
            f"hybrid_rns_q{self.correction_bits}_weight"
            if self.correction == "rns"
            else "hybrid_fp16_weight"
        )
        lut_key = f"hybrid_rns_q{self.correction_bits}_lut_active"
        bias_bytes = tensor_bytes(self.layer.bias)
        return {
            "architecture": "hybrid",
            "correction": self.correction,
            "correction_bits": self.correction_bits,
            "protected_channels": self.layer.p,
            "protected_padded": self.layer.p_padded,
            "weight_bytes": storage.get(weight_key, 0) + bias_bytes,
            "bias_bytes": bias_bytes,
            "lut_active_bytes": storage.get(lut_key, 0),
            "lut_allocated_bytes": storage.get(lut_key, 0),
            "runner_workspace_bytes": sum(
                int(getattr(r, "runtime_workspace_bytes", 0))
                for r in self._runner_cache.values()
            ),
        }



class CombinedProjectionV014(nn.Module):
    def __init__(self, combined: nn.Module, split_sizes: tuple[int, int, int]) -> None:
        super().__init__()
        self.combined = combined
        self.split_sizes = split_sizes
        self.compute_count = 0

    def forward(self, inputs: torch.Tensor):
        self.compute_count += 1
        with nvtx_range("attention_qkv_fused"):
            combined = self.combined(inputs)
        return torch.split(combined, self.split_sizes, dim=-1)


@dataclass
class InstalledAttentionV014:
    architecture: str
    coordinator: CachedRNSQKV
    out_projection: nn.Module
    replaced: list[str]



def install_native_int8_opt_attention(
    attention: nn.Module,
    *,
    include_out_proj: bool = True,
) -> InstalledAttentionV014:
    linears = [attention.q_proj, attention.k_proj, attention.v_proj]
    if not all(isinstance(layer, nn.Linear) for layer in linears):
        raise TypeError("OPT q/k/v projections must be nn.Linear")
    combined, splits = NativeInt8LinearV014.from_linears(linears)
    projection = CombinedProjectionV014(combined, splits).eval()
    coordinator = CachedRNSQKV(projection).eval()
    attention.rns_qkv_v014 = coordinator
    attention.q_proj, attention.k_proj, attention.v_proj = coordinator.slices()
    replaced = ["qkv_fused"]
    if include_out_proj:
        if not isinstance(attention.out_proj, nn.Linear):
            raise TypeError("attention.out_proj must be nn.Linear")
        attention.out_proj = NativeInt8LinearV014(attention.out_proj).eval()
        replaced.append("out_proj")
    return InstalledAttentionV014(
        architecture="native_int8",
        coordinator=coordinator,
        out_projection=attention.out_proj,
        replaced=replaced,
    )

def install_full_rns_opt_attention(
    attention: nn.Module,
    *,
    logical_bits: int = 8,
    lut_channels: int = 2,
    moduli_policy: str = "dense_coprime",
    q8_backend: str = "v07",
    include_out_proj: bool = True,
) -> InstalledAttentionV014:
    linears = [attention.q_proj, attention.k_proj, attention.v_proj]
    if not all(isinstance(layer, nn.Linear) for layer in linears):
        raise TypeError("OPT q/k/v projections must be nn.Linear")
    combined, splits = FullRNSLinearV014.from_linears(
        linears,
        logical_bits=logical_bits,
        lut_channels=lut_channels,
        moduli_policy=moduli_policy,
        q8_backend=q8_backend,
    )
    projection = CombinedProjectionV014(combined, splits).eval()
    coordinator = CachedRNSQKV(projection).eval()
    attention.rns_qkv_v014 = coordinator
    attention.q_proj, attention.k_proj, attention.v_proj = coordinator.slices()
    replaced = ["qkv_fused"]
    if include_out_proj:
        if not isinstance(attention.out_proj, nn.Linear):
            raise TypeError("attention.out_proj must be nn.Linear")
        attention.out_proj = FullRNSLinearV014(
            attention.out_proj,
            logical_bits=logical_bits,
            lut_channels=lut_channels,
            moduli_policy=moduli_policy,
            q8_backend=q8_backend,
        ).eval()
        replaced.append("out_proj")
    return InstalledAttentionV014(
        architecture=f"full_rns_int{logical_bits}",
        coordinator=coordinator,
        out_projection=attention.out_proj,
        replaced=replaced,
    )


def install_hybrid_opt_attention(
    attention: nn.Module,
    *,
    protected_indices: Sequence[int] | torch.Tensor | None = None,
    out_protected_indices: Sequence[int] | torch.Tensor | None = None,
    protected_channels: int | None = None,
    protected_ratio: float = 0.0012,
    correction_bits: int = 16,
    lut_channels: int = 2,
    correction: str = "rns",
    execution: str = "serial",
    include_out_proj: bool = True,
) -> InstalledAttentionV014:
    linears = [attention.q_proj, attention.k_proj, attention.v_proj]
    if not all(isinstance(layer, nn.Linear) for layer in linears):
        raise TypeError("OPT q/k/v projections must be nn.Linear")
    combined, splits = HybridLinearV014.from_linears(
        linears,
        protected_indices=protected_indices,
        protected_channels=protected_channels,
        protected_ratio=protected_ratio,
        correction_bits=correction_bits,
        lut_channels=lut_channels,
        correction=correction,
        execution=execution,
        layer_name="attention.qkv",
    )
    projection = CombinedProjectionV014(combined, splits).eval()
    coordinator = CachedRNSQKV(projection).eval()
    attention.rns_qkv_v014 = coordinator
    attention.q_proj, attention.k_proj, attention.v_proj = coordinator.slices()
    replaced = ["qkv_fused"]
    if include_out_proj:
        if not isinstance(attention.out_proj, nn.Linear):
            raise TypeError("attention.out_proj must be nn.Linear")
        attention.out_proj = HybridLinearV014(
            attention.out_proj,
            protected_indices=(
                protected_indices if out_protected_indices is None else out_protected_indices
            ),
            protected_channels=protected_channels,
            protected_ratio=protected_ratio,
            correction_bits=correction_bits,
            lut_channels=lut_channels,
            correction=correction,
            execution=execution,
            layer_name="attention.out_proj",
        ).eval()
        replaced.append("out_proj")
    return InstalledAttentionV014(
        architecture=f"hybrid_{correction}_q{correction_bits}",
        coordinator=coordinator,
        out_projection=attention.out_proj,
        replaced=replaced,
    )


def collect_attention_memory(installed: InstalledAttentionV014) -> dict[str, object]:
    combined = installed.coordinator.projection.combined
    reports = []
    if hasattr(combined, "memory_report"):
        reports.append(combined.memory_report())
    if hasattr(installed.out_projection, "memory_report"):
        reports.append(installed.out_projection.memory_report())
    return {
        "architecture": installed.architecture,
        "components": reports,
        "weight_bytes": sum(int(r.get("weight_bytes", 0)) for r in reports),
        "lut_active_bytes": sum(int(r.get("lut_active_bytes", 0)) for r in reports),
        "lut_allocated_bytes": sum(int(r.get("lut_allocated_bytes", r.get("lut_active_bytes", 0))) for r in reports),
        "workspace_bytes": sum(int(r.get("runner_workspace_bytes", 0)) for r in reports),
    }
