"""CPU workflow benchmark; not a GPU performance claim."""
import argparse, time
import numpy as np
from rns_llm.rns.matmul import rns_matmul


def timed(fn, repeats):
    values = []
    for _ in range(repeats):
        start = time.perf_counter(); fn(); values.append(time.perf_counter() - start)
    return min(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64); p.add_argument("--k", type=int, default=64); p.add_argument("--n", type=int, default=64); p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    rng = np.random.default_rng(42)
    left = rng.integers(-8, 9, size=(a.m, a.k), dtype=np.int64)
    right = rng.integers(-8, 9, size=(a.k, a.n), dtype=np.int64)
    mods = (3,5,7,11,13,17,19)
    print(f"baseline_ms={timed(lambda: left @ right, a.repeats)*1e3:.3f}")
    print(f"rns_reference_ms={timed(lambda: rns_matmul(left, right, mods), a.repeats)*1e3:.3f}")
    print("NOTE: NumPy reference latency is not the CUDA target.")

if __name__ == "__main__": main()
