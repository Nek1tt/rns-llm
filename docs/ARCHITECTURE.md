# Architecture

## Stage 1 — correct reference

```text
A [M,K] int             B [K,N] int
      |                       |
      +---- RNS encode -------+
                  |
                  v
A_rns [R,M,K]       B_rns [R,K,N]
                  |
                  v
for each residue channel r:
C_rns[r] = A_rns[r] @ B_rns[r] mod m[r]
                  |
                  v
             RNS decode
                  |
                  v
             C [M,N] int
```

## Stage 2 — backend interface

All higher-level code calls:

```python
backend.matmul(a, b, moduli, decode=True)
```

Backends:

```text
NumPyReferenceBackend  -> correctness oracle
CudaBackend            -> GPU implementation
future CUTLASS backend  -> optimized Tensor Core path
```

Transformer code must not know CUDA kernel details.

## Stage 3 — Transformer

```text
float tensor
  -> quantize
  -> backend.matmul(...)
  -> decode/dequantize
  -> rest of model
```

First integration milestone: replace exactly one Linear layer.

## Future GPU hypothesis

```text
RNS residue planes
  -> batched/grouped INT8 GEMM
  -> modular reduction in output/epilogue
  -> residue output
```

This is a hypothesis to benchmark, not an assumed speedup.
