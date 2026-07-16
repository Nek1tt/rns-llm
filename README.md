# RNS LLM Inference

Research prototype: map Residue Number System (RNS) matrix multiplication to GPU/Transformer inference.

## One shared pipeline

```text
integer matrices
      -> RNS encode
      -> independent residue GEMMs
      -> RNS decode
      -> compare with INT32
      -> CUDA backend
      -> one Transformer Linear layer
      -> latency / memory / PPL
```

We do **not** assume RNS is faster. First: correctness. Then: integration. Then: benchmark. Then: optimization.

## Team ownership

| Area | Main folders |
|---|---|
| RNS mathematics/reference | `src/rns_llm/rns/`, `tests/test_rns_*` |
| CUDA/performance | `src/rns_llm/backends/cuda_backend.py`, `cuda/`, `benchmarks/` |
| Transformer integration | `src/rns_llm/layers/`, `src/rns_llm/integration/`, `scripts/` |

Read `docs/TEAM_TASKS.md` first.

## First milestone

```python
C_ref = A.astype(np.int64) @ B.astype(np.int64)
C_rns = rns_matmul(A, B, moduli)
assert np.array_equal(C_ref, C_rns)
```

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_reference_check.py
```

Optional Transformer dependencies:

```bash
pip install -e ".[transformer]"
```

## Repository rule

1. Define interface.
2. Add correctness test.
3. Implement simple version.
4. Benchmark.
5. Optimize measured bottleneck only.
