# Alignment with the Curator's Five Work Packages

This document is the checklist for the final report.  It separates implemented
work from open research claims so the team does not accidentally overstate the
results.

## 1. Moduli Set Selection and Optimization

### Implemented

Three reproducible selection strategies, all with moduli `<=255`:

- `small_primes`: curator-style `3,5,7,11,13,...`; many channels.
- `large_primes`: previous baseline `251,241,239,233,...`.
- `dense_coprime`: `255,253,251,247,241,...`; primes are not required, only
  pairwise coprimality.  This strategy tries to maximize range per channel.

Exact signed reconstruction requires:

```text
product(moduli) > 2 * K * max_abs(A) * max_abs(B)
```

The benchmark reports channel count, encoded-input memory, accumulator memory,
LUT memory and measured latency:

```bash
python benchmarks/benchmark_moduli_sets.py \
  --m 256 --k 768 --n 768 --source-bits 8 \
  --output results/moduli_int8.json

python benchmarks/benchmark_moduli_sets.py \
  --m 256 --k 768 --n 768 --source-bits 12 \
  --output results/moduli_int12.json
```

### Interpretation

More moduli provide more independent work, but on a fixed GPU they also create
more GEMMs and more memory traffic.  Parallelism does not automatically imply
lower latency.  The measured Pareto frontier is the result.

---

## 2. RNS Matrix Multiplication for Transformer / Self-Attention

### Implemented

- Centered signed-int8 residue planes.
- cuBLAS strided-batched INT8 GEMM across residue channels.
- DP4A and scalar kernels as correctness/fallback baselines.
- Cached encoded Transformer weights.
- Reusable accumulator/output workspace.
- Shape benchmark for Q/K/V projections, fused QKV, output projection, MLP,
  one-head `QK^T` and `AV` arithmetic.

```bash
python benchmarks/benchmark_attention.py \
  --tokens 1 16 128 256 --hidden 768 --heads 12 \
  --source-bits 8 \
  --output results/attention_shapes_int8.json
```

### Scope limitation

The current optimized backend directly covers the large Linear/GEMM operations
inside attention.  A fully batched multi-head `QK^T` kernel is not yet fused
with Softmax.  The single-head shapes are measured to expose the small-matrix
problem before adding a grouped-head implementation.

---

## 3. Non-Modular Operations

### Implemented

The old path wrote residue output and ran multiple PyTorch int64 CRT kernels.
Version 0.4 adds:

```text
cuBLAS INT8 accumulators
        -> one CUDA kernel
        -> fast modulo
        -> mixed-radix/Garner reconstruction
        -> signed int64 output
```

This removes the intermediate residue-output tensor and Python CRT loop.

Two decode methods remain benchmarked:

- old PyTorch CRT: correctness/reference;
- single-kernel Garner: optimized path.

### Softmax decision

Softmax is not naturally modular.  The realistic first architecture is:

```text
RNS QK^T -> fused reconstruction -> floating Softmax -> next operation
```

The project therefore optimizes the conversion boundary rather than claiming a
full RNS Softmax.  PPL and model integration determine whether this boundary is
acceptable.

---

## 4. Memory Table Reuse

### Implemented experiment

A full 8-bit multiplication table costs approximately `m*m` bytes, close to
64 KiB for a large modulus.  Version 0.4 instead builds a compact byte-reduction
LUT:

```text
[R, 4, 256] int16
```

Cost per modulus: `4 * 256 * 2 = 2048 bytes`.

Only the first zero, one or two largest-modulus tables are enabled.  The same
device tensor is cached and reused by all blocks and requests.

```bash
python benchmarks/benchmark_lut_reuse.py \
  --source-bits 12 --concurrency 4 \
  --output results/lut_reuse_int12.json
```

The report compares:

- Barrett only (`lut_channels=0`);
- one shared table;
- two shared tables;
- single-request latency;
- four-stream latency and throughput.

### Claim boundary

The compact table is over 50% smaller than full multiplication tables (normally
over 95% smaller).  This is a table-memory result, not a claim that total model
VRAM falls by 50%.  The measured latency decides whether the table loads are
worth using.

---

## 5. Concurrency, Latency and PPL

### Four concurrent requests

```bash
python benchmarks/benchmark_concurrency.py \
  --m 16 --k 768 --n 768 --source-bits 8 \
  --concurrency 1 2 4 \
  --output results/concurrency_int8.json
```

Each request has its own CUDA stream and workspace while sharing cached encoded
weights and constant/LUT tables.  The output reports p50/p95 batch latency and
requests per second.

Run the same test for generation-like `M=1` and prompt-like `M=128`.

### PPL under 5%

The included evaluator replaces a limited number of attention Linear layers and
compares WikiText-2 perplexity:

```bash
pip install -e ".[transformer]" --no-build-isolation
python scripts/evaluate_ppl.py \
  --model facebook/opt-125m \
  --max-layers 1 --quant-bits 8
```

Then increase `--max-layers` gradually.  The required metric is:

```text
relative_ppl_increase = RNS_PPL / baseline_PPL - 1
relative_ppl_increase < 0.05
```

PPL degradation comes from quantization/scaling, not from exact RNS arithmetic.

---

# Recommended Final Evidence Table

| Curator task | Required evidence |
|---|---|
| Moduli selection | latency/memory table for 3 strategies and channel counts |
| RNS GEMM | exactness + attention-shape latency + cuBLAS utilization |
| Non-modular operations | old CRT vs Garner/fused end-to-end latency |
| Table reuse | 0/1/2 LUT latency, memory saving, 4-stream contention |
| Concurrency/PPL | 1/2/4 request p50/p95/throughput and relative PPL increase |
