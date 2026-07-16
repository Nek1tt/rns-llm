"""Shared backend contract."""
from __future__ import annotations
from typing import Iterable, Protocol, runtime_checkable

@runtime_checkable
class RNSMatmulBackend(Protocol):
    name: str
    def matmul(self, a, b, moduli: Iterable[int], *, decode: bool = True):
        """Return [M,N] if decoded, otherwise [R,M,N]."""
        ...
