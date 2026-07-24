from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gcd
from typing import Iterable

import torch


DENSE_COPRIME_MODULI: tuple[int, ...] = (
    255, 253, 251, 247, 241, 239, 233, 229, 227, 223,
    211, 199, 197, 193, 191, 181, 179, 173, 167, 163,
)

LARGE_PRIME_MODULI: tuple[int, ...] = (
    251, 241, 239, 233, 229, 227, 223, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149,
)

SCHOOL_SMALL_MODULI: tuple[int, ...] = (
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
    37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
)

MODULI_POLICIES: dict[str, tuple[int, ...]] = {
    "dense_coprime": DENSE_COPRIME_MODULI,
    "large_primes": LARGE_PRIME_MODULI,
    "school_small": SCHOOL_SMALL_MODULI,
}


@lru_cache(maxsize=1)
def _extension():
    try:
        from rns_llm import _ARCH
    except ImportError as exc:
        raise RuntimeError(
            "v0.13 CUDA extension is not built. Run "
            "RNS_LLM_BUILD_CUDA=1 python -m pip install . "
            "--no-build-isolation --no-deps --force-reinstall"
        ) from exc
    return _ARCH


def extension_available() -> bool:
    try:
        from rns_llm import _ARCH  # noqa: F401
    except ImportError:
        return False
    return True


def quant_max(bits: int) -> int:
    if bits not in (8, 16, 32):
        raise ValueError("logical bits must be one of 8, 16, 32")
    return (1 << (bits - 1)) - 1


def required_signed_range(k: int, bits: int) -> int:
    if k <= 0:
        raise ValueError("K must be positive")
    qmax = quant_max(bits)
    return 2 * int(k) * qmax * qmax + 1


def _validate_moduli(moduli: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in moduli)
    if len(result) < 2:
        raise ValueError("at least two moduli are required")
    for i, modulus in enumerate(result):
        if not 3 <= modulus <= 255:
            raise ValueError(f"modulus {modulus} does not fit the required 8-bit range")
        if modulus % 2 == 0:
            raise ValueError(f"modulus {modulus} must be odd for centered int8 residues")
        for previous in result[:i]:
            if gcd(modulus, previous) != 1:
                raise ValueError(f"moduli {previous} and {modulus} are not coprime")
    return result


@dataclass(frozen=True)
class RNSArchitecturePlan:
    logical_bits: int
    k: int
    policy: str
    moduli: tuple[int, ...]
    modulus_product: int
    required_range: int
    qmax: int

    @property
    def channels(self) -> int:
        return len(self.moduli)

    @property
    def product_bits(self) -> int:
        return self.modulus_product.bit_length()

    @property
    def required_bits(self) -> int:
        return self.required_range.bit_length()

    @property
    def range_headroom(self) -> float:
        return self.modulus_product / self.required_range

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["moduli"] = list(self.moduli)
        payload["channels"] = self.channels
        payload["product_bits"] = self.product_bits
        payload["required_bits"] = self.required_bits
        payload["range_headroom"] = self.range_headroom
        return payload


def select_plan(k: int, bits: int, policy: str = "dense_coprime") -> RNSArchitecturePlan:
    if policy not in MODULI_POLICIES:
        raise ValueError(f"unknown moduli policy: {policy}")
    candidates = _validate_moduli(MODULI_POLICIES[policy])
    required = required_signed_range(k, bits)
    product = 1
    selected: list[int] = []
    for modulus in candidates:
        selected.append(modulus)
        product *= modulus
        if product > required:
            break
    if product <= required:
        raise ValueError(
            f"policy {policy} does not provide enough moduli for q{bits}, K={k}"
        )
    if product.bit_length() > 127:
        raise ValueError(
            "selected RNS product exceeds the 128-bit reconstruction implementation"
        )
    if len(selected) > 20:
        raise ValueError("CUDA implementation supports at most 20 channels")
    return RNSArchitecturePlan(
        logical_bits=int(bits),
        k=int(k),
        policy=policy,
        moduli=tuple(selected),
        modulus_product=product,
        required_range=required,
        qmax=quant_max(bits),
    )


def modular_inverse(value: int, modulus: int) -> int:
    return pow(int(value), -1, int(modulus))


def prefix_inverses(moduli: Iterable[int]) -> tuple[int, ...]:
    mods = _validate_moduli(moduli)
    inverses = [1]
    prefix = mods[0]
    for modulus in mods[1:]:
        inverses.append(modular_inverse(prefix % modulus, modulus))
        prefix *= modulus
    return tuple(inverses)


def reciprocal_u32(modulus: int) -> int:
    return (1 << 32) // int(modulus)


def build_compact_lut(
    moduli: Iterable[int],
    lut_channels: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    mods = _validate_moduli(moduli)
    if not 0 <= lut_channels <= len(mods):
        raise ValueError("lut_channels must be between 0 and channel count")
    if lut_channels == 0:
        return torch.empty((0, 4, 256), dtype=torch.int16, device=device)
    table = torch.empty((lut_channels, 4, 256), dtype=torch.int16)
    values = torch.arange(256, dtype=torch.int64)
    for channel, modulus in enumerate(mods[:lut_channels]):
        factor = 1
        for byte_position in range(4):
            table[channel, byte_position] = ((values * factor) % modulus).to(torch.int16)
            factor = (factor * 256) % modulus
    return table.to(device=device, non_blocking=False).contiguous()


def lut_bytes(lut_channels: int) -> int:
    if lut_channels < 0:
        raise ValueError("lut_channels must be non-negative")
    return int(lut_channels) * 4 * 256 * 2


def logical_dense_weight_bytes(k: int, n: int, bits: int, *, include_scales: bool = True) -> int:
    if k <= 0 or n <= 0:
        raise ValueError("K and N must be positive")
    if bits not in (8, 16, 32):
        raise ValueError("logical bits must be one of 8, 16, 32")
    total = int(k) * int(n) * (bits // 8)
    if include_scales:
        total += int(n) * 4  # one FP32 scale per output column
    return total


@dataclass
class PreparedRNSWeight:
    plan: RNSArchitecturePlan
    residues_ckn: torch.Tensor
    scales_n: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.residues_ckn.device

    @property
    def k(self) -> int:
        return int(self.residues_ckn.shape[1])

    @property
    def n(self) -> int:
        return int(self.residues_ckn.shape[2])

    @property
    def storage_bytes(self) -> int:
        return (
            self.residues_ckn.numel() * self.residues_ckn.element_size()
            + self.scales_n.numel() * self.scales_n.element_size()
        )


@dataclass
class PreparedInt8Weight:
    quantized_kn: torch.Tensor
    scales_n: torch.Tensor

    @property
    def storage_bytes(self) -> int:
        return (
            self.quantized_kn.numel() * self.quantized_kn.element_size()
            + self.scales_n.numel() * self.scales_n.element_size()
        )


@torch.no_grad()
def prepare_rns_weight(
    weight_kn: torch.Tensor,
    plan: RNSArchitecturePlan,
) -> PreparedRNSWeight:
    if not weight_kn.is_cuda or weight_kn.dtype != torch.float32 or weight_kn.ndim != 2:
        raise ValueError("weight_kn must be contiguous CUDA float32 [K,N]")
    if not weight_kn.is_contiguous():
        weight_kn = weight_kn.contiguous()
    k, n = map(int, weight_kn.shape)
    if k != plan.k:
        raise ValueError("weight K does not match plan")
    ext = _extension()
    moduli_t = torch.tensor(plan.moduli, dtype=torch.int32, device=weight_kn.device)
    weight_nk = weight_kn.transpose(0, 1).contiguous()
    residues_cnk = torch.empty(
        (plan.channels, n, k), dtype=torch.int8, device=weight_kn.device
    )
    scales_n = torch.empty(n, dtype=torch.float32, device=weight_kn.device)
    ext.quantize_encode_rows_out(
        weight_nk,
        moduli_t,
        plan.qmax,
        residues_cnk,
        scales_n,
    )
    residues_ckn = residues_cnk.transpose(1, 2).contiguous()
    return PreparedRNSWeight(plan=plan, residues_ckn=residues_ckn, scales_n=scales_n)


@torch.no_grad()
def prepare_int8_weight(weight_kn: torch.Tensor) -> PreparedInt8Weight:
    if not weight_kn.is_cuda or weight_kn.dtype != torch.float32 or weight_kn.ndim != 2:
        raise ValueError("weight_kn must be contiguous CUDA float32 [K,N]")
    if not weight_kn.is_contiguous():
        weight_kn = weight_kn.contiguous()
    ext = _extension()
    k, n = map(int, weight_kn.shape)
    weight_nk = weight_kn.transpose(0, 1).contiguous()
    quantized_nk = torch.empty((n, k), dtype=torch.int8, device=weight_kn.device)
    scales_n = torch.empty(n, dtype=torch.float32, device=weight_kn.device)
    ext.quantize_rows_int8_out(weight_nk, quantized_nk, scales_n)
    return PreparedInt8Weight(
        quantized_kn=quantized_nk.transpose(0, 1).contiguous(),
        scales_n=scales_n,
    )


class RNSArchitectureRunner:
    def __init__(
        self,
        prepared_weight: PreparedRNSWeight,
        *,
        m: int,
        lut_channels: int,
        compact_lut: torch.Tensor | None = None,
    ) -> None:
        if m <= 0:
            raise ValueError("M must be positive")
        self.weight = prepared_weight
        self.plan = prepared_weight.plan
        self.m = int(m)
        self.k = prepared_weight.k
        self.n = prepared_weight.n
        self.device = prepared_weight.device
        self.ext = _extension()
        self.lut_channels = int(lut_channels)
        if compact_lut is None:
            compact_lut = build_compact_lut(
                self.plan.moduli, self.lut_channels, device=self.device
            )
        if tuple(compact_lut.shape) != (self.lut_channels, 4, 256):
            raise ValueError("compact_lut shape mismatch")
        self.compact_lut = compact_lut
        self.moduli_t = torch.tensor(
            self.plan.moduli, dtype=torch.int32, device=self.device
        )
        self.reciprocals_t = torch.tensor(
            [reciprocal_u32(m) for m in self.plan.moduli],
            dtype=torch.int64,
            device=self.device,
        )
        self.prefix_inverses_t = torch.tensor(
            prefix_inverses(self.plan.moduli),
            dtype=torch.int32,
            device=self.device,
        )
        self.activation_residues = torch.empty(
            (self.plan.channels, self.m, self.k),
            dtype=torch.int8,
            device=self.device,
        )
        self.activation_scales = torch.empty(
            self.m, dtype=torch.float32, device=self.device
        )
        self.accumulators = torch.empty(
            (self.plan.channels, self.m, self.n),
            dtype=torch.int32,
            device=self.device,
        )
        self.output = torch.empty(
            (self.m, self.n), dtype=torch.float32, device=self.device
        )

    @property
    def runtime_workspace_bytes(self) -> int:
        tensors = (
            self.activation_residues,
            self.activation_scales,
            self.accumulators,
            self.output,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @property
    def constant_bytes(self) -> int:
        tensors = (
            self.moduli_t,
            self.reciprocals_t,
            self.prefix_inverses_t,
            self.compact_lut,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @torch.no_grad()
    def encode(self, input_mk: torch.Tensor) -> torch.Tensor:
        if (
            not input_mk.is_cuda
            or input_mk.dtype != torch.float32
            or tuple(input_mk.shape) != (self.m, self.k)
        ):
            raise ValueError("input must be CUDA float32 [M,K]")
        if not input_mk.is_contiguous():
            input_mk = input_mk.contiguous()
        self.ext.quantize_encode_rows_out(
            input_mk,
            self.moduli_t,
            self.plan.qmax,
            self.activation_residues,
            self.activation_scales,
        )
        return self.activation_residues

    @torch.no_grad()
    def core(self) -> torch.Tensor:
        return self.ext.rns_mm_dequant_out(
            self.activation_residues,
            self.weight.residues_ckn,
            self.moduli_t,
            self.reciprocals_t,
            self.prefix_inverses_t,
            self.compact_lut,
            self.lut_channels,
            self.activation_scales,
            self.weight.scales_n,
            self.accumulators,
            self.output,
        )

    @torch.no_grad()
    def e2e(self, input_mk: torch.Tensor) -> torch.Tensor:
        self.encode(input_mk)
        return self.core()


class NativeInt8Runner:
    def __init__(self, prepared_weight: PreparedInt8Weight, *, m: int) -> None:
        if m <= 0:
            raise ValueError("M must be positive")
        self.weight = prepared_weight
        self.m = int(m)
        self.k = int(prepared_weight.quantized_kn.shape[0])
        self.n = int(prepared_weight.quantized_kn.shape[1])
        self.device = prepared_weight.quantized_kn.device
        self.ext = _extension()
        self.activation_quantized = torch.empty(
            (self.m, self.k), dtype=torch.int8, device=self.device
        )
        self.activation_scales = torch.empty(
            self.m, dtype=torch.float32, device=self.device
        )
        self.accumulators = torch.empty(
            (self.m, self.n), dtype=torch.int32, device=self.device
        )
        self.output = torch.empty(
            (self.m, self.n), dtype=torch.float32, device=self.device
        )

    @property
    def runtime_workspace_bytes(self) -> int:
        tensors = (
            self.activation_quantized,
            self.activation_scales,
            self.accumulators,
            self.output,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @torch.no_grad()
    def encode(self, input_mk: torch.Tensor) -> torch.Tensor:
        if (
            not input_mk.is_cuda
            or input_mk.dtype != torch.float32
            or tuple(input_mk.shape) != (self.m, self.k)
        ):
            raise ValueError("input must be CUDA float32 [M,K]")
        if not input_mk.is_contiguous():
            input_mk = input_mk.contiguous()
        self.ext.quantize_rows_int8_out(
            input_mk, self.activation_quantized, self.activation_scales
        )
        return self.activation_quantized

    @torch.no_grad()
    def core(self) -> torch.Tensor:
        return self.ext.native_int8_mm_dequant_out(
            self.activation_quantized,
            self.weight.quantized_kn,
            self.activation_scales,
            self.weight.scales_n,
            self.accumulators,
            self.output,
        )

    @torch.no_grad()
    def e2e(self, input_mk: torch.Tensor) -> torch.Tensor:
        self.encode(input_mk)
        return self.core()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
