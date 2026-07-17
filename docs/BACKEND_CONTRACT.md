# Backend Contract

The CUDA owner can work without waiting for final signed-RNS or moduli-selection
work because the CUDA boundary uses already encoded residue planes.

## Residue GEMM

```python
backend.matmul_residues(
    a_residues,   # uint8 CUDA [R, M, K]
    b_residues,   # uint8 CUDA [R, K, N]
    moduli,       # Python sequence, each 2..255
    kernel="auto",
) -> uint8 CUDA [R, M, N]
```

For every residue channel `r`:

```text
C[r] = (A[r] @ B[r]) mod moduli[r]
```

## Temporary end-to-end int8 contract

```python
backend.matmul_int8(a_int8, b_int8, moduli, decode=True)
```

This helper:

```text
int8 A/B
  -> CUDA RNS encode
  -> residue GEMM
  -> PyTorch CRT decode
  -> int64 result
```

It exists to validate the pipeline before the mathematician finalizes the
representation. The residue GEMM interface should remain stable even if encode,
decode, or moduli selection changes later.
