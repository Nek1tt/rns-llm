from __future__ import annotations

from math import gcd, prod
from typing import Iterable, Literal

import numpy as np

ModuliStrategy = Literal["large_primes", "dense_coprime", "small_primes"]


def validate_moduli(moduli: Iterable[int]) -> tuple[int, ...]:
    mods = tuple(int(m) for m in moduli)
    if not mods:
        raise ValueError("moduli must not be empty")
    # The curator's requirement is an at-most-8-bit residue channel.
    # We keep m <= 255 so centered residues always lie in signed int8.
    if any(m < 2 or m > 255 for m in mods):
        raise ValueError("every modulus must be in [2, 255]")
    for i, left in enumerate(mods):
        for right in mods[i + 1 :]:
            if gcd(left, right) != 1:
                raise ValueError(f"moduli are not pairwise coprime: {left}, {right}")
    return mods


def encode_numpy(values: np.ndarray, moduli: Iterable[int]) -> np.ndarray:
    mods = validate_moduli(moduli)
    values = np.asarray(values, dtype=np.int64)
    return np.stack([np.remainder(values, m) for m in mods], axis=0).astype(np.uint8)


def encode_centered_numpy(values: np.ndarray, moduli: Iterable[int]) -> np.ndarray:
    mods = validate_moduli(moduli)
    values = np.asarray(values, dtype=np.int64)
    planes = []
    for modulus in mods:
        residue = np.remainder(values, modulus)
        residue = np.where(residue > modulus // 2, residue - modulus, residue)
        planes.append(residue.astype(np.int8))
    return np.stack(planes, axis=0)


def crt_constants(moduli: Iterable[int]) -> tuple[int, tuple[int, ...]]:
    mods = validate_moduli(moduli)
    modulus_product = prod(mods)
    if modulus_product > np.iinfo(np.int64).max:
        raise OverflowError("product(moduli) must fit signed int64 for this decoder")

    coefficients = []
    for modulus in mods:
        partial = modulus_product // modulus
        inverse = pow(partial, -1, modulus)
        coefficients.append((partial * inverse) % modulus_product)
    return modulus_product, tuple(coefficients)


def garner_pairwise_inverses(moduli: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Pairwise inverse table used by mixed-radix/Garner reconstruction.

    Entry [j][i], j < i, is inverse(moduli[j]) modulo moduli[i].
    """
    mods = validate_moduli(moduli)
    result = [[0 for _ in mods] for _ in mods]
    for i in range(len(mods)):
        for j in range(i):
            result[j][i] = pow(mods[j], -1, mods[i])
    return tuple(tuple(row) for row in result)


def compact_byte_mod_lut(moduli: Iterable[int]) -> np.ndarray:
    """Build [R,4,256] int16 tables for 32-bit modulo by byte decomposition.

    A full multiplication table for an 8-bit modulus is about 64 KiB.  This
    compact table is only 2 KiB per modulus and can be shared by every CUDA
    block/thread.  The runtime benchmark decides whether its extra memory loads
    are actually faster than Barrett reduction.
    """
    mods = validate_moduli(moduli)
    table = np.empty((len(mods), 4, 256), dtype=np.int16)
    for channel, modulus in enumerate(mods):
        factor = 1
        for byte_position in range(4):
            for byte_value in range(256):
                table[channel, byte_position, byte_value] = (
                    byte_value * factor
                ) % modulus
            factor = (factor * 256) % modulus
    return table


def decode_numpy(
    residues: np.ndarray,
    moduli: Iterable[int],
    *,
    signed: bool = True,
) -> np.ndarray:
    mods = validate_moduli(moduli)
    residues = np.asarray(residues, dtype=np.int64)
    if residues.shape[0] != len(mods):
        raise ValueError("residue channel count does not match moduli")

    modulus_product, coefficients = crt_constants(mods)
    result = np.zeros(residues.shape[1:], dtype=np.int64)
    for channel, coefficient in enumerate(coefficients):
        result = np.remainder(
            result + residues[channel] * coefficient,
            modulus_product,
        )

    if signed:
        result = np.where(
            result > modulus_product // 2,
            result - modulus_product,
            result,
        )
    return result


def decode_garner_numpy(
    residues: np.ndarray,
    moduli: Iterable[int],
    *,
    signed: bool = True,
) -> np.ndarray:
    """Reference mixed-radix reconstruction matching the fused CUDA kernel."""
    mods = validate_moduli(moduli)
    source = np.asarray(residues, dtype=np.int64)
    if source.shape[0] != len(mods):
        raise ValueError("residue channel count does not match moduli")

    inv = garner_pairwise_inverses(mods)
    digits: list[np.ndarray] = []
    for i, modulus in enumerate(mods):
        value = np.remainder(source[i], modulus)
        for j in range(i):
            value = np.remainder((value - digits[j]) * inv[j][i], modulus)
        digits.append(value)

    result = np.zeros(source.shape[1:], dtype=np.int64)
    prefix = 1
    for modulus, digit in zip(mods, digits):
        result += prefix * digit
        prefix *= modulus

    if signed:
        result = np.where(result > prefix // 2, result - prefix, result)
    return result


def matmul_residues_numpy(
    a_residues: np.ndarray,
    b_residues: np.ndarray,
    moduli: Iterable[int],
) -> np.ndarray:
    mods = validate_moduli(moduli)
    a = np.asarray(a_residues, dtype=np.uint8)
    b = np.asarray(b_residues, dtype=np.uint8)

    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("expected [R, M, K] and [R, K, N]")
    if a.shape[0] != len(mods) or b.shape[0] != len(mods):
        raise ValueError("residue channel count does not match moduli")
    if a.shape[2] != b.shape[1]:
        raise ValueError("K dimensions do not match")

    outputs = []
    for channel, modulus in enumerate(mods):
        value = a[channel].astype(np.int64) @ b[channel].astype(np.int64)
        outputs.append(np.remainder(value, modulus).astype(np.uint8))
    return np.stack(outputs, axis=0)


def rns_matmul_numpy(
    a: np.ndarray,
    b: np.ndarray,
    moduli: Iterable[int],
    *,
    decode: bool = True,
) -> np.ndarray:
    a_residues = encode_numpy(a, moduli)
    b_residues = encode_numpy(b, moduli)
    result_residues = matmul_residues_numpy(a_residues, b_residues, moduli)
    if not decode:
        return result_residues
    return decode_numpy(result_residues, moduli, signed=True)


# Large prime moduli were used in v0.3.  They minimize channel count and are a
# useful baseline for cuBLAS, but their modulo operation has no special form.
LARGE_PRIME_MODULI: tuple[int, ...] = (
    251, 241, 239, 233, 229, 227, 223, 211, 199, 197,
    193, 191, 181, 179, 173, 167, 163, 157, 151, 149,
    139, 137, 131, 127,
)

# This candidate order packs more dynamic range into early channels while every
# modulus remains <=255 and pairwise-coprime selection is still enforced.  It
# includes composite moduli such as 255, 253 and 247 because primality is not an
# RNS requirement; pairwise coprimality is.
DENSE_COPRIME_MODULI: tuple[int, ...] = (
    255, 253, 251, 247, 241, 239, 233, 229, 227, 223,
    211, 199, 197, 193, 191, 181, 179, 173, 167, 163,
    157, 151, 149, 139, 137, 131, 127, 125, 121,
)

# Curator-style small moduli.  This usually needs more channels but is useful
# for explicitly measuring the speed/parallelism/memory tradeoff and for LUTs.
SMALL_PRIME_MODULI: tuple[int, ...] = (
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
    107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
    163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
    223, 227, 229, 233, 239, 241, 251,
)

# Backwards-compatible alias.
HARDWARE_FRIENDLY_MODULI = LARGE_PRIME_MODULI


def candidates_for_strategy(strategy: ModuliStrategy) -> tuple[int, ...]:
    if strategy == "large_primes":
        return LARGE_PRIME_MODULI
    if strategy == "dense_coprime":
        return DENSE_COPRIME_MODULI
    if strategy == "small_primes":
        return SMALL_PRIME_MODULI
    raise ValueError(f"unknown moduli strategy: {strategy}")


def dot_product_bound(k: int, max_abs_a: int, max_abs_b: int) -> int:
    if k <= 0:
        raise ValueError("k must be positive")
    if max_abs_a < 0 or max_abs_b < 0:
        raise ValueError("absolute bounds must be non-negative")
    return int(k) * int(max_abs_a) * int(max_abs_b)


def choose_moduli_for_dot(
    k: int,
    max_abs_a: int,
    max_abs_b: int,
    *,
    candidates: Iterable[int] | None = None,
    strategy: ModuliStrategy = "large_primes",
) -> tuple[int, ...]:
    """Greedy centered-RNS range selector.

    Exact signed reconstruction requires product(moduli) > 2*bound.
    All candidates are <=255, so centered residues fit signed int8.
    """
    target = 2 * dot_product_bound(k, max_abs_a, max_abs_b) + 1
    source = tuple(candidates) if candidates is not None else candidates_for_strategy(strategy)
    selected: list[int] = []
    current_product = 1
    for modulus in source:
        candidate = tuple(selected + [int(modulus)])
        try:
            validate_moduli(candidate)
        except ValueError:
            continue
        selected.append(int(modulus))
        current_product *= int(modulus)
        if current_product >= target:
            return tuple(selected)
    raise ValueError(f"candidate moduli cannot cover required range {target}")


def moduli_cost_model(
    moduli: Iterable[int],
    *,
    m: int,
    k: int,
    n: int,
    source_element_bytes: int,
    lut_channels: int = 0,
) -> dict[str, int | float]:
    mods = validate_moduli(moduli)
    channels = len(mods)
    source_inputs = (m * k + k * n) * source_element_bytes
    encoded_inputs = channels * (m * k + k * n)
    residue_output = channels * m * n
    int32_accumulator = channels * m * n * 4
    compact_lut = min(lut_channels, channels) * 4 * 256 * 2
    full_mul_lut = sum(modulus * modulus for modulus in mods[:lut_channels])
    return {
        "channels": channels,
        "modulus_product": prod(mods),
        "source_inputs_bytes": source_inputs,
        "encoded_inputs_bytes": encoded_inputs,
        "residue_output_bytes": residue_output,
        "int32_accumulator_bytes": int32_accumulator,
        "compact_lut_bytes": compact_lut,
        "full_mul_lut_bytes": full_mul_lut,
        "compact_vs_full_lut_saving": (
            0.0 if full_mul_lut == 0 else 1.0 - compact_lut / full_mul_lut
        ),
    }
