# Team tasks

## RNS owner — mathematics/reference

Goal: make the NumPy reference mathematically correct and prove the supported range.

Files:

```text
src/rns_llm/rns/arithmetic.py
src/rns_llm/rns/moduli.py
src/rns_llm/rns/matmul.py
tests/test_rns_arithmetic.py
tests/test_rns_matmul.py
tests/test_moduli.py
```

Tasks:

1. Validate signed encode/decode.
2. Add randomized tests.
3. Define exact supported range.
4. Improve `choose_moduli`.
5. Compare candidate moduli sets.
6. Document overflow assumptions.
7. Decide whether full-K accumulation before modulo is safe for target shapes.

Definition of done:

```text
decoded RNS matmul == integer matmul
```

Do not work on CUDA optimization.

---

## CUDA owner — GPU/performance

Goal: run residue-channel matmul on GPU and measure the real bottleneck.

Files:

```text
src/rns_llm/backends/cuda_backend.py
cuda/
benchmarks/
```

Tasks:

1. Freeze input/output layout.
2. Implement one-modulus GPU path.
3. Compare exactly with NumPy reference.
4. Extend to multiple residue channels.
5. Benchmark separate launches vs batched/grouped execution.
6. Investigate INT8 Tensor Core backend.
7. Investigate modular reduction in output/epilogue.
8. Profile memory traffic and kernel time.

Definition of done:

```text
GPU output == NumPy RNS reference
```

---

## Integration owner — Transformer/model

Goal: prepare one Transformer Linear layer to call the shared backend interface.

Files:

```text
src/rns_llm/layers/rns_linear.py
src/rns_llm/integration/linear_tools.py
scripts/list_linear_layers.py
```

Tasks:

1. Select a small pretrained Transformer.
2. List Linear layers.
3. Choose one projection layer.
4. Replace one `nn.Linear` with `RNSLinear`.
5. Keep PyTorch fallback working.
6. Add quantization/dequantization hooks.
7. Add latency measurement.
8. Add perplexity evaluation.

Do not wait for CUDA. Use the fallback/backend interface.

Definition of done:

```text
model runs with one Linear layer replaced
```

---

## Team lead daily check

Ask only:

```text
What was completed yesterday?
What exact output will exist today?
Are you blocked?
```
