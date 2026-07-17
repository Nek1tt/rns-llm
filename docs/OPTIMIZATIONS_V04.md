# v0.4 Optimizations

## Fused modulo and Garner reconstruction

The main v0.3 bottleneck was CRT decode.  v0.4 reconstructs each output element
inside one CUDA thread using a mixed-radix/Garner algorithm.  The kernel reads
all channel accumulators, reduces them modulo their bases, reconstructs the
integer in registers and writes one output value.

This avoids:

- an intermediate int8 residue-output tensor;
- one int8-to-int64 conversion per channel;
- repeated large-modulus PyTorch `remainder` kernels;
- repeated temporary tensors.

## Preallocated workspace

`RNSWorkspace` holds:

```text
int32 accumulators [R,M,N]
int64 output       [M,N]
```

A workspace must not be shared by simultaneously executing requests.  One
workspace per CUDA stream/request is used in the concurrency benchmark.

## Compact modulo tables

For up to two largest moduli, int32 modulo can be computed by decomposing the
absolute value into four bytes and reading four precomputed residues.  Each
modulus needs only 2 KiB of int16 table data.

Barrett reduction remains the default control.  LUTs are an experiment, because
extra table loads can be slower even when memory use is small.

## Dense coprime moduli

RNS bases do not have to be prime.  Candidate values `255,253,251,247,...` pack
more representable range into early channels while preserving pairwise
coprimality and signed-int8 centered residues.

## Remaining high-value work

1. Fuse integer reconstruction directly with dequantization, bias and FP16/BF16
   output for `RNSLinear`.
2. Use CUTLASS epilogue to avoid writing the int32 accumulator workspace.
3. Implement grouped multi-head QK/AV GEMM.
4. Capture fixed-shape encode/GEMM/reconstruct in CUDA Graphs.
5. Calibrate each layer's actual ranges so some layers need fewer channels.
