"""Slow correctness-first RNS matmul. OWNER: RNS mathematics/reference."""
from __future__ import annotations
from typing import Iterable
import numpy as np
from .arithmetic import decode, encode, validate_moduli


def matmul_residues(left_residues: np.ndarray, right_residues: np.ndarray, moduli: Iterable[int]) -> np.ndarray:
    mods = validate_moduli(moduli)
    left = np.asarray(left_residues, dtype=np.int64)
    right = np.asarray(right_residues, dtype=np.int64)
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("expected [R,M,K] and [R,K,N]")
    if left.shape[0] != len(mods) or right.shape[0] != len(mods):
        raise ValueError("R must equal len(moduli)")
    if left.shape[2] != right.shape[1]:
        raise ValueError("K dimensions do not match")

    outputs = []
    for channel, modulus in enumerate(mods):
        # TODO(RNS owner): document safe accumulation bounds for target K/moduli.
        outputs.append((left[channel] @ right[channel]) % modulus)
    return np.stack(outputs, axis=0).astype(np.int64, copy=False)


def rns_matmul(left: np.ndarray, right: np.ndarray, moduli: Iterable[int], *, decode_result: bool = True) -> np.ndarray:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("expected compatible rank-2 matrices")
    result_rns = matmul_residues(encode(left, moduli), encode(right, moduli), moduli)
    return decode(result_rns, moduli, signed=True) if decode_result else result_rns
