# CUDA backend

OWNER: CUDA/performance.

## First target

```text
A residues [R,M,K]
B residues [R,K,N]
moduli     [R]
      -> GPU
C residues [R,M,N]
```

Correctness requirement:

```text
CUDA output == NumPy matmul_residues output
```

Milestones:

1. One modulus.
2. Multiple independent moduli.
3. Separate launches benchmark.
4. Batched/grouped execution benchmark.
5. Investigate INT8 Tensor Core backend.
6. Investigate modular reduction in output/epilogue.
7. Connect native code to `CudaBackend`.

Do not put Transformer logic here.
