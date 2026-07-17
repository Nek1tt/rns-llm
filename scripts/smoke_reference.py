from __future__ import annotations

import numpy as np

from rns_llm.reference import rns_matmul_numpy


def main() -> None:
    rng = np.random.default_rng(42)
    a = rng.integers(-127, 128, size=(32, 64), dtype=np.int16).astype(np.int8)
    b = rng.integers(-127, 128, size=(64, 24), dtype=np.int16).astype(np.int8)
    moduli = (251, 241, 239, 233)

    expected = a.astype(np.int64) @ b.astype(np.int64)
    actual = rns_matmul_numpy(a, b, moduli)
    np.testing.assert_array_equal(actual, expected)
    print("PASS reference end-to-end RNS matmul")


if __name__ == "__main__":
    main()
