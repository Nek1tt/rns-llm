from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from rns_llm.hybrid_v010 import (
    MODULI_CANDIDATES,
    accuracy_metrics,
    benchmark_cuda_callable,
    choose_moduli,
    quant_max,
)


def _ext():
    try:
        from rns_llm import _PREFILL  # type: ignore
    except Exception as exc:  # pragma: no cover - CUDA build required
        raise RuntimeError(
            "rns_llm._PREFILL is unavailable. Build the project with RNS_LLM_BUILD_CUDA=1."
        ) from exc
    return _PREFILL


def _prefix_inverses(moduli: tuple[int, ...], device: torch.device) -> torch.Tensor:
    values: list[int] = []
    prefix = 1
    for index, modulus in enumerate(moduli):
        values.append(1 if index == 0 else pow(prefix % modulus, -1, modulus))
        prefix *= modulus
    return torch.tensor(values, dtype=torch.int32, device=device)


def _build_compact_lut(
    moduli: tuple[int, ...],
    lut_channels: int,
    device: torch.device,
) -> torch.Tensor:
    """Build only the active compact reduction tables.

    A table is [4,256] int16 (2048 bytes) per residue channel.  Runners using
    the same layer/policy reuse the same cached tensor across CUDA streams.
    """
    if not 0 <= int(lut_channels) <= len(moduli):
        raise ValueError("lut_channels must be in [0, number of RNS channels]")
    if lut_channels == 0:
        return torch.empty((0, 4, 256), dtype=torch.int16, device=device)
    table = torch.empty((lut_channels, 4, 256), dtype=torch.int16)
    values = torch.arange(256, dtype=torch.int64)
    for channel, modulus in enumerate(moduli[:lut_channels]):
        factor = 1
        for byte_position in range(4):
            table[channel, byte_position] = ((values * factor) % modulus).to(torch.int16)
            factor = (factor * 256) % modulus
    return table.to(device=device).contiguous()


def _float_scales_per_output(weight: torch.Tensor, qmax: int) -> torch.Tensor:
    maximum = weight.abs().amax(dim=1)
    return torch.clamp(maximum / float(qmax), min=torch.finfo(torch.float32).tiny).float().contiguous()


def _pad_columns(x: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, int]:
    p = int(x.shape[1])
    padded = ((p + multiple - 1) // multiple) * multiple
    if padded == p:
        return x.contiguous(), 0
    return torch.nn.functional.pad(x, (0, padded - p)).contiguous(), padded - p


@dataclass
class ProtectedWeight:
    logical_bits: int
    moduli: tuple[int, ...]
    moduli_t: torch.Tensor
    prefix_inverses_t: torch.Tensor
    scales: torch.Tensor
    residues: torch.Tensor  # [C,N,Ppad]


@dataclass
class PrefillLayerV011:
    layer_name: str
    n: int
    k: int
    p: int
    p_padded: int
    protected_indices: torch.Tensor  # int32 [P]
    protected_mask: torch.Tensor  # uint8 [K]
    bias: torch.Tensor
    fp16_weight_kn: torch.Tensor
    native_weight_kn: torch.Tensor
    native_weight_scales: torch.Tensor
    main_weight_kn: torch.Tensor
    main_weight_scales: torch.Tensor
    protected_fp16_np: torch.Tensor
    protected_rns: dict[int, ProtectedWeight]
    selected_plan: dict
    statistics: dict
    lut_cache: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict, repr=False)

    @property
    def device(self) -> torch.device:
        return self.main_weight_kn.device

    @classmethod
    @torch.no_grad()
    def from_pack(
        cls,
        pack: dict,
        *,
        device: torch.device | str = "cuda",
        optimized_rns_bits: tuple[int, ...] = (8, 16),
        storage_mode: str = "all",
    ) -> "PrefillLayerV011":
        ext = _ext()
        device = torch.device(device)
        if storage_mode not in {"all", "rns", "fp16"}:
            raise ValueError("storage_mode must be all, rns, or fp16")
        weight = pack["weight"].to(device=device, dtype=torch.float32).contiguous()
        bias_obj = pack.get("bias")
        bias = (
            torch.empty(0, dtype=torch.float32, device=device)
            if bias_obj is None
            else bias_obj.to(device=device, dtype=torch.float32).contiguous()
        )
        n, k = map(int, weight.shape)
        protected = pack["protected_indices"].to(device=device, dtype=torch.int32).contiguous()
        if protected.numel() == 0:
            raise ValueError("protected channel set cannot be empty")
        p = int(protected.numel())
        p_padded = ((p + 3) // 4) * 4
        mask = torch.zeros(k, dtype=torch.uint8, device=device)
        mask[protected.long()] = 1

        fp16_weight_kn = (
            weight.transpose(0, 1).half().contiguous()
            if storage_mode == "all"
            else torch.empty((0, 0), dtype=torch.float16, device=device)
        )

        if storage_mode == "all":
            native_scales = _float_scales_per_output(weight, 127)
            native_weight_kn = torch.empty((k, n), dtype=torch.int8, device=device)
            ext.quantize_weight_all_out(weight, native_scales, native_weight_kn)
        else:
            native_scales = torch.empty(0, dtype=torch.float32, device=device)
            native_weight_kn = torch.empty((0, 0), dtype=torch.int8, device=device)

        safe_weight = weight.clone()
        safe_weight.index_fill_(1, protected.long(), 0.0)
        main_scales = _float_scales_per_output(safe_weight, 127)
        main_weight_kn = torch.empty((k, n), dtype=torch.int8, device=device)
        ext.quantize_weight_masked_out(weight, main_scales, mask, main_weight_kn)

        protected_weight = weight.index_select(1, protected.long()).contiguous()
        protected_fp16_np = (
            _pad_columns(protected_weight.half())[0]
            if storage_mode in {"all", "fp16"}
            else torch.empty((0, 0), dtype=torch.float16, device=device)
        )
        rns_weights: dict[int, ProtectedWeight] = {}
        rns_bits = optimized_rns_bits if storage_mode in {"all", "rns"} else ()
        for bits in rns_bits:
            moduli = choose_moduli(bits, p_padded)
            if len(moduli) > 10:
                raise ValueError(
                    f"optimized v0.11 correction supports at most 10 channels; q{bits} requires {len(moduli)}"
                )
            moduli_t = torch.tensor(moduli, dtype=torch.int32, device=device)
            prefix = _prefix_inverses(moduli, device)
            scales = _float_scales_per_output(protected_weight, quant_max(bits))
            residues = torch.empty(
                (len(moduli), n, p_padded), dtype=torch.int8, device=device
            )
            ext.encode_protected_weight_out(
                protected_weight, scales, moduli_t, quant_max(bits), residues
            )
            rns_weights[bits] = ProtectedWeight(
                logical_bits=bits,
                moduli=moduli,
                moduli_t=moduli_t,
                prefix_inverses_t=prefix,
                scales=scales,
                residues=residues,
            )

        return cls(
            layer_name=str(pack["layer_name"]),
            n=n,
            k=k,
            p=p,
            p_padded=p_padded,
            protected_indices=protected,
            protected_mask=mask,
            bias=bias,
            fp16_weight_kn=fp16_weight_kn,
            native_weight_kn=native_weight_kn,
            native_weight_scales=native_scales,
            main_weight_kn=main_weight_kn,
            main_weight_scales=main_scales,
            protected_fp16_np=protected_fp16_np,
            protected_rns=rns_weights,
            selected_plan=dict(pack.get("selected_plan", {})),
            statistics=dict(pack.get("statistics", {})),
        )

    def compact_lut(self, logical_bits: int, lut_channels: int) -> torch.Tensor:
        if logical_bits not in self.protected_rns:
            raise ValueError(f"q{logical_bits} protected weight was not prepared")
        channels = len(self.protected_rns[logical_bits].moduli)
        if not 0 <= int(lut_channels) <= channels:
            raise ValueError("lut_channels must be in [0, number of RNS channels]")
        key = (int(logical_bits), int(lut_channels))
        cached = self.lut_cache.get(key)
        if cached is None:
            cached = _build_compact_lut(
                self.protected_rns[logical_bits].moduli, int(lut_channels), self.device
            )
            self.lut_cache[key] = cached
        return cached

    def runner(
        self,
        m: int,
        *,
        logical_bits: int | None = 16,
        workspace_bytes: int = 32 * 1024 * 1024,
        lut_channels: int = 2,
        mode: str = "all",
    ) -> "PrefillRunnerV011":
        return PrefillRunnerV011(
            self, m=m, logical_bits=logical_bits, workspace_bytes=workspace_bytes,
            lut_channels=lut_channels, mode=mode,
        )


class PrefillRunnerV011:
    def __init__(
        self,
        layer: PrefillLayerV011,
        *,
        m: int,
        logical_bits: int | None,
        workspace_bytes: int,
        lut_channels: int = 2,
        mode: str = "all",
    ) -> None:
        if logical_bits is not None and logical_bits not in layer.protected_rns:
            raise ValueError(f"q{logical_bits} protected weight was not prepared")
        if m <= 0:
            raise ValueError("M must be positive")
        if mode not in {"all", "rns", "fp16"}:
            raise ValueError("mode must be all, rns, or fp16")
        if mode == "rns" and logical_bits is None:
            raise ValueError("RNS runner mode requires logical_bits")
        if mode == "fp16" and logical_bits is not None:
            raise ValueError("FP16-correction runner mode requires logical_bits=None")
        self.layer = layer
        self.mode = mode
        self.m = int(m)
        self.bits = logical_bits
        self.ext = _ext()
        self.rns_weight = None if logical_bits is None else layer.protected_rns[logical_bits]
        self.lut_channels = int(lut_channels)
        channel_count = 0 if self.rns_weight is None else len(self.rns_weight.moduli)
        if not 0 <= self.lut_channels <= channel_count:
            raise ValueError("lut_channels must be in [0, number of RNS channels]")
        self.compact_lut = (
            torch.empty((0, 4, 256), dtype=torch.int16, device=layer.device)
            if self.rns_weight is None
            else layer.compact_lut(int(logical_bits), self.lut_channels)
        )
        self.device = layer.device
        self.workspace_bytes = int(workspace_bytes)

        need_reference = mode == "all"
        need_main = mode in {"all", "rns", "fp16"}
        self.fp16_plan = (
            self.ext.LtFp16Plan(m, layer.k, layer.n, workspace_bytes)
            if need_reference else None
        )
        self.native_plan = (
            self.ext.LtInt8Plan(m, layer.k, layer.n, workspace_bytes)
            if need_reference else None
        )
        self.main_plan = (
            self.ext.LtInt8Plan(m, layer.k, layer.n, workspace_bytes)
            if need_main else None
        )
        empty_u8 = lambda: torch.empty(0, dtype=torch.uint8, device=self.device)
        self.fp16_workspace = (
            torch.empty(workspace_bytes, dtype=torch.uint8, device=self.device)
            if need_reference else empty_u8()
        )
        self.native_workspace = (
            torch.empty(workspace_bytes, dtype=torch.uint8, device=self.device)
            if need_reference else empty_u8()
        )
        self.main_workspace = (
            torch.empty(workspace_bytes, dtype=torch.uint8, device=self.device)
            if need_main else empty_u8()
        )

        self.x_half = (
            torch.empty((m, layer.k), dtype=torch.float16, device=self.device)
            if need_reference else torch.empty(0, dtype=torch.float16, device=self.device)
        )
        self.native_q = (
            torch.empty((m, layer.k), dtype=torch.int8, device=self.device)
            if need_reference else torch.empty(0, dtype=torch.int8, device=self.device)
        )
        self.native_scales = (
            torch.empty(m, dtype=torch.float32, device=self.device)
            if need_reference else torch.empty(0, dtype=torch.float32, device=self.device)
        )
        self.native_acc = (
            torch.empty((m, layer.n), dtype=torch.int32, device=self.device)
            if need_reference else torch.empty(0, dtype=torch.int32, device=self.device)
        )
        self.native_out = (
            torch.empty((m, layer.n), dtype=torch.float32, device=self.device)
            if need_reference else torch.empty(0, dtype=torch.float32, device=self.device)
        )
        self.fp16_out = (
            torch.empty((m, layer.n), dtype=torch.float32, device=self.device)
            if need_reference else torch.empty(0, dtype=torch.float32, device=self.device)
        )

        self.main_q = torch.empty((m, layer.k), dtype=torch.int8, device=self.device)
        self.main_scales = torch.empty(m, dtype=torch.float32, device=self.device)
        self.main_acc = torch.empty((m, layer.n), dtype=torch.int32, device=self.device)
        self.protected_half = torch.empty(
            (m, layer.p_padded), dtype=torch.float16, device=self.device
        )
        self.protected_residues = torch.empty(
            (0 if self.rns_weight is None else len(self.rns_weight.moduli), m, layer.p_padded),
            dtype=torch.int8,
            device=self.device,
        )
        self.protected_scales = torch.empty(m, dtype=torch.float32, device=self.device)
        self.correction = torch.empty((m, layer.n), dtype=torch.float32, device=self.device)
        self.hybrid_rns_out = torch.empty((m, layer.n), dtype=torch.float32, device=self.device)
        self.hybrid_fp16_out = torch.empty((m, layer.n), dtype=torch.float32, device=self.device)

        self.main_stream = torch.cuda.Stream(device=self.device)
        self.correction_stream = torch.cuda.Stream(device=self.device)

    @property
    def runtime_workspace_bytes(self) -> int:
        tensors = (
            self.fp16_workspace, self.native_workspace, self.main_workspace,
            self.x_half, self.native_q, self.native_scales, self.native_acc,
            self.native_out, self.fp16_out, self.main_q, self.main_scales,
            self.main_acc, self.protected_half, self.protected_residues,
            self.protected_scales, self.correction, self.hybrid_rns_out,
            self.hybrid_fp16_out,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @torch.no_grad()
    def cast_fp16(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp16_plan is None:
            raise RuntimeError("FP16 reference path was not allocated for this runner")
        return self.ext.cast_fp32_to_fp16_out(x, self.x_half)

    @torch.no_grad()
    def preprocess_native(self, x: torch.Tensor) -> None:
        if self.native_plan is None:
            raise RuntimeError("native INT8 path was not allocated for this runner")
        self.ext.quantize_rows_out(x, self.native_q, self.native_scales)

    def _require_rns(self) -> ProtectedWeight:
        if self.rns_weight is None or self.bits is None:
            raise RuntimeError("this runner was created without an RNS correction weight")
        return self.rns_weight

    @torch.no_grad()
    def preprocess_hybrid_rns(self, x: torch.Tensor) -> None:
        rns_weight = self._require_rns()
        self.ext.fused_hybrid_preprocess_out(
            x,
            self.layer.protected_mask,
            self.layer.protected_indices,
            rns_weight.moduli_t,
            quant_max(self.bits),
            self.main_q,
            self.main_scales,
            self.protected_half,
            self.protected_residues,
            self.protected_scales,
        )

    @torch.no_grad()
    def preprocess_hybrid_fp16(self, x: torch.Tensor) -> None:
        self.ext.fused_hybrid_preprocess_fp16_out(
            x,
            self.layer.protected_mask,
            self.layer.protected_indices,
            self.main_q,
            self.main_scales,
            self.protected_half,
        )

    @torch.no_grad()
    def preprocess_hybrid(self, x: torch.Tensor) -> None:
        # Backward-compatible alias for v0.11 scripts.
        if self.rns_weight is None:
            self.preprocess_hybrid_fp16(x)
        else:
            self.preprocess_hybrid_rns(x)

    @torch.no_grad()
    def fp16_core(self) -> torch.Tensor:
        if self.fp16_plan is None:
            raise RuntimeError("FP16 reference path was not allocated for this runner")
        self.fp16_plan.run(
            self.x_half,
            self.layer.fp16_weight_kn,
            self.fp16_out,
            self.fp16_workspace,
        )
        if self.layer.bias.numel():
            self.ext.add_bias_out(self.fp16_out, self.layer.bias)
        return self.fp16_out

    @torch.no_grad()
    def native_core(self) -> torch.Tensor:
        if self.native_plan is None:
            raise RuntimeError("native INT8 path was not allocated for this runner")
        self.native_plan.run(
            self.native_q,
            self.layer.native_weight_kn,
            self.native_acc,
            self.native_workspace,
        )
        self.ext.dequant_epilogue_out(
            self.native_acc,
            self.native_scales,
            self.layer.native_weight_scales,
            self.layer.bias,
            self.native_out,
        )
        return self.native_out

    @torch.no_grad()
    def main_int8_only(self) -> torch.Tensor:
        if self.main_plan is None:
            raise RuntimeError("hybrid main path was not allocated for this runner")
        return self.main_plan.run(
            self.main_q,
            self.layer.main_weight_kn,
            self.main_acc,
            self.main_workspace,
        )

    @torch.no_grad()
    def rns_correction_only(self) -> torch.Tensor:
        rns_weight = self._require_rns()
        return self.ext.rns_rankk_correction_out(
            self.protected_residues,
            rns_weight.residues,
            rns_weight.moduli_t,
            rns_weight.prefix_inverses_t,
            self.compact_lut,
            self.lut_channels,
            self.protected_scales,
            rns_weight.scales,
            self.correction,
        )

    @torch.no_grad()
    def fp16_correction_only(self) -> torch.Tensor:
        return self.ext.fp16_rankk_correction_out(
            self.protected_half, self.layer.protected_fp16_np, self.correction
        )

    @torch.no_grad()
    def rns_fused_epilogue_only(self) -> torch.Tensor:
        rns_weight = self._require_rns()
        return self.ext.rns_fused_epilogue_out(
            self.main_acc,
            self.main_scales,
            self.layer.main_weight_scales,
            self.protected_residues,
            rns_weight.residues,
            rns_weight.moduli_t,
            rns_weight.prefix_inverses_t,
            self.compact_lut,
            self.lut_channels,
            self.protected_scales,
            rns_weight.scales,
            self.layer.bias,
            self.hybrid_rns_out,
        )

    @torch.no_grad()
    def fp16_fused_epilogue_only(self) -> torch.Tensor:
        return self.ext.fp16_fused_epilogue_out(
            self.main_acc,
            self.main_scales,
            self.layer.main_weight_scales,
            self.protected_half,
            self.layer.protected_fp16_np,
            self.layer.bias,
            self.hybrid_fp16_out,
        )

    @torch.no_grad()
    def merge_only(self) -> torch.Tensor:
        return self.ext.merge_epilogue_out(
            self.main_acc,
            self.main_scales,
            self.layer.main_weight_scales,
            self.correction,
            self.layer.bias,
            self.hybrid_rns_out,
        )

    @torch.no_grad()
    def hybrid_rns_serial_core(self) -> torch.Tensor:
        self.main_int8_only()
        return self.rns_fused_epilogue_only()

    @torch.no_grad()
    def hybrid_fp16_serial_core(self) -> torch.Tensor:
        self.main_int8_only()
        return self.fp16_fused_epilogue_only()

    def _launch_parallel(
        self,
        correction_fn: Callable[[], torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        current = torch.cuda.current_stream(self.device)
        self.main_stream.wait_stream(current)
        self.correction_stream.wait_stream(current)
        with torch.cuda.stream(self.main_stream):
            self.main_int8_only()
        with torch.cuda.stream(self.correction_stream):
            correction_fn()
        current.wait_stream(self.main_stream)
        current.wait_stream(self.correction_stream)
        self.ext.merge_epilogue_out(
            self.main_acc,
            self.main_scales,
            self.layer.main_weight_scales,
            self.correction,
            self.layer.bias,
            output,
        )
        return output

    @torch.no_grad()
    def hybrid_rns_parallel_core(self) -> torch.Tensor:
        return self._launch_parallel(self.rns_correction_only, self.hybrid_rns_out)

    @torch.no_grad()
    def hybrid_fp16_parallel_core(self) -> torch.Tensor:
        return self._launch_parallel(self.fp16_correction_only, self.hybrid_fp16_out)

    @torch.no_grad()
    def fp16_e2e(self, x: torch.Tensor) -> torch.Tensor:
        self.cast_fp16(x)
        return self.fp16_core()

    @torch.no_grad()
    def native_e2e(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_native(x)
        return self.native_core()

    @torch.no_grad()
    def hybrid_rns_serial_e2e(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_hybrid_rns(x)
        return self.hybrid_rns_serial_core()

    @torch.no_grad()
    def hybrid_rns_parallel_e2e(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_hybrid_rns(x)
        return self.hybrid_rns_parallel_core()

    @torch.no_grad()
    def hybrid_fp16_serial_e2e(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_hybrid_fp16(x)
        return self.hybrid_fp16_serial_core()

    @torch.no_grad()
    def hybrid_fp16_parallel_e2e(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_hybrid_fp16(x)
        return self.hybrid_fp16_parallel_core()

    def storage_bytes(self) -> dict[str, int]:
        def size(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        result = {
            "fp16_weight": size(self.layer.fp16_weight_kn),
            "native_int8_weight": size(self.layer.native_weight_kn)
            + size(self.layer.native_weight_scales),
            "hybrid_fp16_weight": size(self.layer.main_weight_kn)
            + size(self.layer.main_weight_scales)
            + size(self.layer.protected_fp16_np),
        }
        if self.rns_weight is not None and self.bits is not None:
            result.update({
                f"hybrid_rns_q{self.bits}_weight": size(self.layer.main_weight_kn)
                + size(self.layer.main_weight_scales)
                + size(self.rns_weight.residues)
                + size(self.rns_weight.scales),
                f"hybrid_rns_q{self.bits}_lut_all": len(self.rns_weight.moduli) * 4 * 256 * 2,
                f"hybrid_rns_q{self.bits}_lut_active": size(self.compact_lut),
            })
        return result



def benchmark_runner_method(
    runner: PrefillRunnerV011,
    x: torch.Tensor,
    method: str,
    *,
    warmup: int,
    iterations: int,
    prepared: bool,
) -> tuple[dict, torch.Tensor]:
    if prepared:
        runner.cast_fp16(x)
        runner.preprocess_native(x)
        runner.preprocess_hybrid(x)
    methods: dict[str, Callable[[], torch.Tensor]] = {
        "fp16": runner.fp16_core if prepared else lambda: runner.fp16_e2e(x),
        "native_int8": runner.native_core if prepared else lambda: runner.native_e2e(x),
        "main_int8_only": runner.main_int8_only,
        "rns_correction_only": runner.rns_correction_only,
        "fp16_correction_only": runner.fp16_correction_only,
        "rns_fused_epilogue_only": runner.rns_fused_epilogue_only,
        "fp16_fused_epilogue_only": runner.fp16_fused_epilogue_only,
        "merge_only": runner.merge_only,
        "hybrid_rns_serial": runner.hybrid_rns_serial_core
        if prepared
        else lambda: runner.hybrid_rns_serial_e2e(x),
        "hybrid_rns_parallel": runner.hybrid_rns_parallel_core
        if prepared
        else lambda: runner.hybrid_rns_parallel_e2e(x),
        "hybrid_fp16_serial": runner.hybrid_fp16_serial_core
        if prepared
        else lambda: runner.hybrid_fp16_serial_e2e(x),
        "hybrid_fp16_parallel": runner.hybrid_fp16_parallel_core
        if prepared
        else lambda: runner.hybrid_fp16_parallel_e2e(x),
        "preprocess_hybrid": lambda: (runner.preprocess_hybrid(x), runner.main_q)[1],
        "preprocess_native": lambda: (runner.preprocess_native(x), runner.native_q)[1],
    }
    if method not in methods:
        raise KeyError(f"unknown method {method!r}")
    fn = methods[method]
    timing = benchmark_cuda_callable(fn, warmup=warmup, iterations=iterations)
    output = fn()
    torch.cuda.synchronize()
    return timing, output.detach().clone()
