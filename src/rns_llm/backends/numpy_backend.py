"""NumPy correctness backend."""
from typing import Iterable
import numpy as np
from rns_llm.rns.matmul import rns_matmul

class NumPyReferenceBackend:
    name = "numpy-reference"
    def matmul(self, a: np.ndarray, b: np.ndarray, moduli: Iterable[int], *, decode: bool = True) -> np.ndarray:
        return rns_matmul(a, b, moduli, decode_result=decode)
