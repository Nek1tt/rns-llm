"""CUDA adapter. OWNER: CUDA/performance."""
from __future__ import annotations
from typing import Iterable

class CudaBackend:
    name = "cuda"
    def __init__(self) -> None:
        self._extension = None

    def _load_extension(self):
        # TODO(CUDA owner): import compiled extension from cuda/ here.
        raise NotImplementedError("CUDA extension is not implemented yet")

    def matmul(self, a, b, moduli: Iterable[int], *, decode: bool = True):
        if self._extension is None:
            self._load_extension()
        return self._extension(a, b, tuple(moduli), decode)
