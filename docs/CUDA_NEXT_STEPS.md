# CUDA Development Path

## Implemented now

- One launch handles all residue channels through `grid.z`.
- Shared-memory tiled GEMM.
- Fast Barrett reduction.
- Safe periodic reduction.
- CUDA encode from signed int8 to residue planes.
- PyTorch extension and Transformer-facing Python API.

## Next experiment 1 — Profile first

Measure separately:

```text
encode
residue GEMM
CRT decode
total pipeline
```

Use:

```bash
nsys profile -o results/rns_nsys \
  python benchmarks/benchmark_cuda.py --m 256 --k 768 --n 768

ncu --set full -o results/rns_ncu \
  python benchmarks/benchmark_cuda.py --m 256 --k 768 --n 768 --iterations 5
```

Questions:

- Is the tiled kernel compute-bound or memory-bound?
- How much time is spent in modular reduction?
- Does fusing residue channels reduce launch overhead?
- How much does CRT decode dominate end-to-end latency?

## Next experiment 2 — NVIDIA GEMM as the engine

The strongest future direction is not to replace NVIDIA GEMM. It is:

```text
RNS residue planes
    -> strided-batched/grouped INT8 GEMM
    -> INT32 accumulators
    -> fused modulo epilogue
    -> uint8 residue output
```

Candidate implementations:

1. CUTLASS custom epilogue.
2. cuBLASLt INT8/INT32 GEMM followed by a fused/reduction kernel.
3. A grouped kernel when residue channels or matrix shapes differ.

Do not start this before the current backend is validated and profiled.

## Next experiment 3 — Weight prepacking

Transformer weights are static during inference. Cache:

```text
quantized weight
transposed contiguous weight
RNS-encoded weight planes
```

The included `RNSLinear.prepare_weight()` already provides the integration point.

## Next experiment 4 — Reduce conversion overhead

Potential fusion points:

```text
activation quantization + RNS encoding
GEMM + modulo output
CRT decode + dequantization + bias
```

Without fusion, conversion overhead may hide any GEMM benefit.
