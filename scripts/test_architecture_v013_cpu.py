from __future__ import annotations

import random

from rns_llm.architecture_v013 import MODULI_POLICIES, prefix_inverses, select_plan


def canonical_lut_i32(value: int, modulus: int) -> int:
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError("compact LUT reduces a signed int32 GEMM accumulator")
    magnitude = abs(value)
    residue = 0
    factor = 1
    for byte_position in range(4):
        byte_value = (magnitude >> (8 * byte_position)) & 0xFF
        residue += (byte_value * factor) % modulus
        factor = (factor * 256) % modulus
    while residue >= modulus:
        residue -= modulus
    if value < 0 and residue:
        residue = modulus - residue
    return residue


def garner_signed(residues: list[int], moduli: tuple[int, ...]) -> int:
    inverses = prefix_inverses(moduli)
    x = residues[0] % moduli[0]
    prefix = moduli[0]
    for channel in range(1, len(moduli)):
        modulus = moduli[channel]
        delta = (residues[channel] - (x % modulus)) % modulus
        digit = (delta * inverses[channel]) % modulus
        x += prefix * digit
        prefix *= modulus
    return x - prefix if x > prefix // 2 else x


def main() -> None:
    rng = random.Random(130013)
    lut_cases = 0
    integer_cases = 0
    for modulus in sorted({m for values in MODULI_POLICIES.values() for m in values}):
        for value in (-(1 << 31), -(1 << 31) + 1, -1, 0, 1, (1 << 31) - 1):
            assert canonical_lut_i32(value, modulus) == value % modulus
            lut_cases += 1
        for _ in range(200):
            value = rng.randint(-(1 << 31), (1 << 31) - 1)
            assert canonical_lut_i32(value, modulus) == value % modulus
            lut_cases += 1

    for k in (256, 768, 2560, 10240):
        for policy in MODULI_POLICIES:
            for bits in (8, 16, 32):
                plan = select_plan(k, bits, policy)
                assert plan.modulus_product > plan.required_range
                bound = k * plan.qmax * plan.qmax
                for value in (-bound, -bound + 1, -1, 0, 1, bound - 1, bound):
                    residues = [value % modulus for modulus in plan.moduli]
                    assert garner_signed(residues, plan.moduli) == value
                    integer_cases += 1
                for _ in range(100):
                    value = rng.randint(-bound, bound)
                    residues = [value % modulus for modulus in plan.moduli]
                    assert garner_signed(residues, plan.moduli) == value
                    integer_cases += 1
    print({"status": "PASS", "lut_i32_cases": lut_cases, "integer_cases": integer_cases})


if __name__ == "__main__":
    main()
