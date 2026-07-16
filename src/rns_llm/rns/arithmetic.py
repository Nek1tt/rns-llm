"""Reference RNS arithmetic. OWNER: RNS mathematics/reference."""
from __future__ import annotations
from math import gcd, prod
from typing import Iterable
import numpy as np


def validate_moduli(moduli: Iterable[int]) -> tuple[int, ...]:
    mods = tuple(int(m) for m in moduli)
    if not mods:
        raise ValueError("moduli must not be empty")
    if any(m <= 1 for m in mods):
        raise ValueError("every modulus must be greater than 1")
    for i, left in enumerate(mods):
        for right in mods[i + 1:]:
            if gcd(left, right) != 1:
                raise ValueError(f"moduli must be pairwise coprime: {left}, {right}")
    return mods


def encode(values: np.ndarray, moduli: Iterable[int]) -> np.ndarray:
    """Return residue planes with shape [R, *values.shape]."""
    mods = validate_moduli(moduli)
    values = np.asarray(values, dtype=np.int64)
    return np.stack([np.mod(values, m) for m in mods], axis=0)


def decode(residues: np.ndarray, moduli: Iterable[int], *, signed: bool = True) -> np.ndarray:
    """Chinese Remainder Theorem decoder with centered signed reconstruction."""
    mods = validate_moduli(moduli)
    residues = np.asarray(residues, dtype=np.int64)
    if residues.ndim < 1 or residues.shape[0] != len(mods):
        raise ValueError("residues must have shape [R, ...]")

    modulus_range = prod(mods)
    if modulus_range > np.iinfo(np.int64).max:
        raise OverflowError("reference decoder requires product(moduli) <= int64 max")

    result = np.zeros(residues.shape[1:], dtype=np.int64)
    for channel, modulus in enumerate(mods):
        partial = modulus_range // modulus
        inverse = pow(partial, -1, modulus)
        result = (result + residues[channel] * partial * inverse) % modulus_range

    if signed:
        threshold = modulus_range // 2
        result = np.where(result > threshold, result - modulus_range, result)
    return result.astype(np.int64, copy=False)


def add_residues(left: np.ndarray, right: np.ndarray, moduli: Iterable[int]) -> np.ndarray:
    mods = validate_moduli(moduli)
    left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    if left.shape != right.shape or left.shape[0] != len(mods):
        raise ValueError("left/right must have equal shape [R, ...]")
    return np.stack([(left[i] + right[i]) % m for i, m in enumerate(mods)], axis=0)


def multiply_residues(left: np.ndarray, right: np.ndarray, moduli: Iterable[int]) -> np.ndarray:
    mods = validate_moduli(moduli)
    left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    if left.shape != right.shape or left.shape[0] != len(mods):
        raise ValueError("left/right must have equal shape [R, ...]")
    return np.stack([(left[i] * right[i]) % m for i, m in enumerate(mods)], axis=0)
