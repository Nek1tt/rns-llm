# v0.5 Architecture Optimizations

## 1. Fused QKV projection

For a shared input `X`:

```text
Q = X Wq
K = X Wk
V = X Wv
```

is exactly equivalent to:

```text
[Q | K | V] = X [Wq | Wk | Wv].
```

The implementation concatenates output channels, preserves per-output weight
scales, runs one RNS path, then returns three views. A cache proxy preserves the
existing OPT `q_proj/k_proj/v_proj` API and clears after all three slices are
served.

## 2. Continuous batching

Requests sharing the same prepared weight are copied into one preallocated
activation matrix and concatenated along `M`. One encode, one batched residue
GEMM and one Garner reconstruction replace several independent launches.

This is exact. It does not approximate or mix requests. Output row ranges are
split back to the original request boundaries.

## 3. Safe adaptive channel prefix

The full moduli set remains the correctness fallback. For each invocation:

```text
bound = max_i sum_k |A[i,k]| * max_{k,j}|B[k,j]|.
```

The smallest prefix satisfying

```text
2 * bound < product(prefix_moduli)
```

is selected. This proof is conservative; it may use more channels than needed,
but it cannot select too few. The current prototype includes scalar
GPU-to-host synchronization for the bound, so selection overhead is measured.

## 4. Stream-safe scratch memory

A workspace is keyed by `(rows, channels, CUDA stream)`. Sharing the same
accumulator/output buffers between concurrent streams would be a data race and
could produce plausible but incorrect results; v0.5 prevents that.
