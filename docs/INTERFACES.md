# Shared interfaces

Do not change these signatures without team discussion.

## Encode

```python
encode(values, moduli) -> residues
```

Shape: `[D0, D1, ...] -> [R, D0, D1, ...]`.

## Decode

```python
decode(residues, moduli, signed=True) -> values
```

Signed reconstruction is centered around zero. `M = product(moduli)`.

## RNS matmul

```python
rns_matmul(a, b, moduli, decode_result=True)
```

```text
a: [M,K]
b: [K,N]
result: [M,N] when decoded
result: [R,M,N] when residues are returned
```

## Backend

```python
backend.matmul(a, b, moduli, decode=True)
```

All CUDA paths must satisfy this contract.

## Transformer layer

`RNSLinear` owns model-facing logic:

```text
float -> quantize -> RNS backend -> decode -> dequantize -> float
```

Do not put Transformer code into `cuda/`.
