from __future__ import annotations

from math import prod
from typing import Iterable

import torch

from rns_llm.reference import crt_constants, validate_moduli


class TorchReferenceBackend:
    """Readable PyTorch backend used for integration before CUDA is built."""

    name = "torch-reference"

    def encode(self, values: torch.Tensor, moduli: Iterable[int]) -> torch.Tensor:
        mods = validate_moduli(moduli)
        mods_tensor = torch.tensor(mods, dtype=torch.int64, device=values.device)
        shape = (len(mods),) + (1,) * values.ndim
        return torch.remainder(values.to(torch.int64).unsqueeze(0), mods_tensor.view(shape)).to(
            torch.uint8
        )

    def matmul_residues(
        self,
        a_residues: torch.Tensor,
        b_residues: torch.Tensor,
        moduli: Iterable[int],
        *,
        kernel: str = "reference",
    ) -> torch.Tensor:
        del kernel
        mods = validate_moduli(moduli)
        if a_residues.ndim != 3 or b_residues.ndim != 3:
            raise ValueError("expected [R, M, K] and [R, K, N]")

        outputs = []
        for channel, modulus in enumerate(mods):
            # Intended mainly for CPU tests. CUDA integer matmul support varies.
            value = a_residues[channel].to(torch.int64) @ b_residues[channel].to(torch.int64)
            outputs.append(torch.remainder(value, modulus).to(torch.uint8))
        return torch.stack(outputs, dim=0)

    def decode(
        self,
        residues: torch.Tensor,
        moduli: Iterable[int],
        *,
        signed: bool = True,
    ) -> torch.Tensor:
        mods = validate_moduli(moduli)
        modulus_product, coefficients = crt_constants(mods)
        # Keep a safety margin because residue * coefficient is evaluated in int64.
        if modulus_product > (2**63 - 1) // max(mods):
            raise OverflowError("CRT product is too large for the torch int64 decoder")

        result = torch.zeros(residues.shape[1:], dtype=torch.int64, device=residues.device)
        for channel, coefficient in enumerate(coefficients):
            result = torch.remainder(
                result + residues[channel].to(torch.int64) * coefficient,
                modulus_product,
            )

        if signed:
            result = torch.where(
                result > modulus_product // 2,
                result - modulus_product,
                result,
            )
        return result

    def matmul_int8(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        moduli: Iterable[int],
        *,
        decode: bool = True,
        kernel: str = "reference",
    ) -> torch.Tensor:
        a_residues = self.encode(a, moduli)
        b_residues = self.encode(b, moduli)
        output = self.matmul_residues(a_residues, b_residues, moduli, kernel=kernel)
        return self.decode(output, moduli) if decode else output
