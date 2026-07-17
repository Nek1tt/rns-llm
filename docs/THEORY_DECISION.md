# Theory decision for the next prototype

## Direct answer to the delayed theory question

RNS should not be used to re-encode values that are already ordinary INT8.
Native NVIDIA INT8 GEMM already multiplies INT8 operands and accumulates into
INT32 exactly for the matrix sizes we are testing.

The meaningful target is:

```text
12-bit or 16-bit fixed-point operands
        -> centered signed-INT8 RNS channels
        -> NVIDIA INT8 GEMM / DP4A
        -> exact wide integer result
```

RNS preserves the chosen integer/fixed-point representation. It does not by
itself preserve FP16/FP32 model accuracy.

## Why centered residues

A canonical residue modulo 251 lies in `[0, 250]`, which does not fit signed
INT8. The same residue class can be represented in centered form:

```text
[-125, 125]
```

Therefore every modulus up to 255 can still be used with signed INT8 hardware:

```text
canonical 250 mod 251  <=> centered -1 mod 251
```

This is the key mapping used by the new DP4A and cuBLAS paths.

## Correct range condition

For a dot product of length `K` with bounds:

```text
|a_i| <= Amax
|b_i| <= Bmax
```

the worst-case result is:

```text
B = K * Amax * Bmax
```

For exact signed reconstruction, choose pairwise-coprime moduli such that:

```text
product(moduli) > 2 * B
```

## Concrete project choices

### Existing INT8 experiment

For `K=768`, `Amax=Bmax=127`:

```text
B = 12,387,072
required product > 24,774,144
```

Four large moduli are sufficient, but this experiment gives no accuracy
advantage over native INT8 GEMM.

### Recommended first meaningful experiment: 12-bit fixed point

For `K=768`, `Amax=Bmax=2047`:

```text
B = 3,218,080,512
required product > 6,436,161,024
```

Five large 8-bit moduli are sufficient. This is the recommended next research
case because it represents values that one INT8 operand cannot hold.

### 16-bit fixed point

For full-range signed INT16 and `K=768`, six large moduli are sufficient with
the current candidate set, but the extra channels increase memory and compute.
Use this after the 12-bit experiment.

## Recommendation for Transformer integration

1. Quantize activations to a 12-bit signed fixed-point integer.
2. Quantize weights per output channel to 12-bit signed fixed point.
3. Encode weights once and cache their centered RNS planes.
4. Encode activations per request.
5. Run the cuBLAS INT8 batched backend.
6. Decode only when the following operation cannot stay in RNS.
7. Compare against FP16 and ordinary INT8 baselines.

The result may still be slower. The scientific question is whether the extra
precision of 12-bit fixed point can justify the channel overhead.
