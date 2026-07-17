# Optimizations added in v0.3

## 1. Centered signed-INT8 residues

Residues are stored in `torch.int8` instead of canonical `torch.uint8` for the
fast path. This makes moduli up to 255 compatible with signed INT8 hardware.

## 2. DP4A kernel

The custom kernel transposes each B tile in shared memory and packs four signed
bytes from A and B into one `__dp4a` instruction.

Available kernels:

```text
scalar
DP4A
DP4A with periodic modular reduction
```

## 3. cuBLAS strided-batched INT8 backend

All residue channels are sent to one `cublasGemmStridedBatchedEx` call:

```text
INT8 x INT8 -> INT32
```

A separate CUDA kernel immediately converts the INT32 outputs back to centered
residues. The row-major product is mapped to cuBLAS through the transpose
identity `C^T = B^T A^T` without copying the matrices.

Requirements for this path:

```text
SM >= 6.1
K % 4 == 0
N % 4 == 0
full dot product fits signed INT32
```

## 4. Cached encoded weights

`prepare_weight()` encodes B once. The benchmark now compares:

```text
uncached: encode A + encode B + GEMM + decode
cached:   encode A + GEMM + decode
```

## 5. Better benchmark protocol

- correctness is checked before timing;
- methods are measured in randomized order every round;
- default warmup and iteration counts are larger;
- encode, GEMM, decode, cached and uncached paths are separated;
- 8-, 12- and 16-bit source experiments are supported;
- throughput and approximate memory values are saved in JSON.
