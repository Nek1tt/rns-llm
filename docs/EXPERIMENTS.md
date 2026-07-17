# Experiments

## H1 — Fused channels reduce overhead

Compare:

```text
one launch with grid.z = R
versus
R independent launches
```

Use `--compare-separate`.

## H2 — Barrett reduction is faster than `%`

Compare `naive` against `tiled` on shapes large enough that launch overhead is
not dominant. Note that these kernels also differ in tiling, so a clean ablation
would require adding a tiled `%` kernel later.

## H3 — Periodic reduction is only needed for unsafe bounds

The backend computes:

```text
K * (max_modulus - 1)^2
```

If it fits uint32, `auto` selects `tiled`. Otherwise it selects `tiled_safe`.
Test both correctness and speed.

## H4 — End-to-end conversion dominates

Compare:

```text
GEMM-only latency
versus
encode + GEMM + decode
```

If total time is much larger, focus on fusion/caching rather than GEMM tiling.

## H5 — Cached weights matter for Transformer inference

Compare first and subsequent calls to `RNSLinear`. The first call prepares and
encodes the weight; later calls reuse it.
