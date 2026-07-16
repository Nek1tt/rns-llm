"""Moduli selection helpers. OWNER: RNS mathematics/reference."""
from __future__ import annotations
from math import prod
from typing import Iterable
from .arithmetic import validate_moduli

DEFAULT_CANDIDATES = (
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
    191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
)


def modulus_product(moduli: Iterable[int]) -> int:
    return prod(validate_moduli(moduli))


def dot_product_bound(k: int, max_abs_a: int, max_abs_b: int) -> int:
    if k <= 0 or max_abs_a < 0 or max_abs_b < 0:
        raise ValueError("invalid bounds")
    return int(k) * int(max_abs_a) * int(max_abs_b)


def required_modulus_product(k: int, max_abs_a: int, max_abs_b: int) -> int:
    return 2 * dot_product_bound(k, max_abs_a, max_abs_b) + 1


def choose_moduli(k: int, max_abs_a: int, max_abs_b: int, *, candidates=DEFAULT_CANDIDATES, max_modulus: int = 255) -> tuple[int, ...]:
    """Simple greedy baseline. TODO: replace with documented cost-model experiments."""
    target = required_modulus_product(k, max_abs_a, max_abs_b)
    selected, current = [], 1
    for modulus in candidates:
        if modulus > max_modulus:
            continue
        selected.append(int(modulus))
        current *= int(modulus)
        if current >= target:
            return validate_moduli(selected)
    raise ValueError(f"candidate list cannot reach required product {target}")
