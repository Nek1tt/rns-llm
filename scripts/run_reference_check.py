import json
from pathlib import Path
import numpy as np
from rns_llm.rns.matmul import rns_matmul


def main():
    path = Path(__file__).resolve().parents[1] / "configs" / "reference.json"
    c = json.loads(path.read_text())
    rng = np.random.default_rng(c["seed"])
    m,k,n = c["matrix"]["m"], c["matrix"]["k"], c["matrix"]["n"]
    lo,hi = c["value_range"]["min"], c["value_range"]["max"]
    a = rng.integers(lo, hi+1, size=(m,k), dtype=np.int64)
    b = rng.integers(lo, hi+1, size=(k,n), dtype=np.int64)
    actual = rns_matmul(a, b, tuple(c["moduli"]))
    np.testing.assert_array_equal(actual, a @ b)
    print("OK: decoded RNS matmul exactly matches integer matmul")
    print(f"shape: ({m}, {k}) @ ({k}, {n})")
    print(f"moduli: {tuple(c['moduli'])}")

if __name__ == "__main__": main()
