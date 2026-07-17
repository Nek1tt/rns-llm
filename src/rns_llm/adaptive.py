from __future__ import annotations

from math import prod
from typing import Iterable

from rns_llm.reference import validate_moduli


def signed_capacity(moduli: Iterable[int]) -> int:
    """Largest guaranteed absolute integer representable by centered RNS."""
    mods = validate_moduli(moduli)
    return (prod(mods) - 1) // 2


def minimal_prefix_channels(
    moduli: Iterable[int],
    absolute_bound: int,
    *,
    min_channels: int = 2,
) -> int:
    """Return the smallest prefix whose centered range contains ±absolute_bound.

    The strict correctness condition is ``2 * bound < product(prefix)``.
    Raises when even the full set is insufficient; callers must never silently
    wrap around.
    """
    mods = validate_moduli(moduli)
    if absolute_bound < 0:
        raise ValueError("absolute_bound must be non-negative")
    if not 2 <= min_channels <= len(mods):
        raise ValueError("min_channels must be between 2 and len(moduli)")

    running = 1
    for channels, modulus in enumerate(mods, start=1):
        running *= modulus
        if channels >= min_channels and 2 * absolute_bound < running:
            return channels
    raise OverflowError(
        f"full moduli set cannot represent bound {absolute_bound}; "
        f"capacity={signed_capacity(mods)}"
    )


def safe_l1_dot_bound(max_row_l1: int, max_abs_weight: int) -> int:
    """Safe bound for every output entry of A@B.

    For every row i and output column j:
      |sum_k A[i,k] B[k,j]| <= sum_k |A[i,k]| * max_{k,j}|B[k,j]|.
    """
    if max_row_l1 < 0 or max_abs_weight < 0:
        raise ValueError("bounds must be non-negative")
    return int(max_row_l1) * int(max_abs_weight)
