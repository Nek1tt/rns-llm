# Team tasks for the v0.5 baseline

## Why we freeze v0.5 first

Version 0.5 is the current shared baseline. It already contains:

- CUDA RNS backend using batched INT8 cuBLAS GEMM;
- centered signed residues;
- fused modulo + Garner reconstruction;
- cached encoded weights and reusable workspaces;
- compact lookup tables;
- fused QKV for OPT attention;
- continuous batching experiments;
- adaptive-channel experiments;
- model-level PPL checks with backend call counters.

The primary project objective is now practical:

> Improve end-to-end LLM inference speed and/or reduce total GPU memory relative to FP16 and native INT8, while keeping the PPL increase below 5%.

The tasks below must use the current v0.5 interfaces. Do not rewrite the CUDA backend unless the task explicitly requires it.

---

# Task A — RNS mathematics and correctness

Owner: mathematics / physics team member.

## Goal in simple words

Independently check that the RNS mathematics used by v0.5 is safe at the borders, that the chosen moduli are valid, and that the assumptions are clearly documented.

The final result must answer:

1. For which integer range does the current RNS decode return the correct signed value?
2. Are the current moduli pairwise coprime?
3. Is the modulus product large enough for the tested matrix shapes and value ranges?
4. What happens when the valid range is exceeded?
5. Why do CRT and Garner reconstruct the same integer?

## A1. Add boundary tests

### Existing files to read first

- `src/rns_llm/reference.py`
- `src/rns_llm/adaptive.py`
- `tests/test_reference.py`
- `tests/test_cuda_backend.py`
- `docs/CORRECTNESS_GUARDS.md`

### New file

- `tests/test_rns_boundaries.py`

### What to test

For every selected modulus set, calculate:

```text
M = product(moduli)
```

Test signed encode/decode for values near the centered range:

```text
0
1
-1
M // 2 - 1
-(M // 2) + 1
```

Also explicitly test the exact boundary values and values outside the valid range:

```text
M // 2
-(M // 2)
M // 2 + 1
-(M // 2) - 1
```

The test must not silently call an out-of-range wrapped value “correct”. It should either:

- expect the documented wrapped result; or
- expect the API to reject the value.

Add matrix-multiplication boundary cases:

- all values are positive maximums;
- all values are negative maximums;
- alternating signs;
- one non-zero row or column;
- random values very close to the safe bound.

For every valid case:

```python
expected = a.astype(np.int64) @ b.astype(np.int64)
actual = rns_matmul(...)
assert np.array_equal(actual, expected)
```

### Definition of done

- CPU tests pass with `pytest -q`.
- Every supported range is written explicitly in the test comments.
- At least one intentionally unsafe case demonstrates wrap-around or rejection.
- No CUDA changes are required.

---

## A2. Verify candidate modulus sets

### Existing files to read first

- `src/rns_llm/reference.py`
- `src/rns_llm/adaptive.py`
- `scripts/theory_ranges.py`
- `benchmarks/benchmark_moduli_sets.py`

### New script

- `scripts/analyze_moduli.py`

### Modulus families to check

```text
small primes
large primes
dense coprime
```

At minimum, include the current v0.5 prefixes:

```text
[255, 253, 251]
[255, 253, 251, 247]
[255, 253, 251, 247, 241]
```

For each set, print and save:

- whether all pairs are coprime;
- the product `M`;
- the centered signed capacity;
- the number of channels;
- the bytes per encoded value with the current byte-per-residue layout;
- the safe worst-case dot-product range for user-supplied `K`, `a_max`, and `b_max`;
- whether the set is safe for that requested case.

Use the conservative condition:

```text
2 * K * a_max * b_max < M
```

Also document that layer-specific bounds may be tighter than this global worst case.

### Output file

- `results/moduli_analysis.csv`

Suggested columns:

```text
strategy,moduli,channels,product,centered_capacity,
bytes_per_value,k,a_max,b_max,required_bound,safe
```

### Definition of done

- The current dense-coprime set is independently validated.
- Unsafe prefixes are clearly marked as unsafe.
- The script works without a GPU.
- The exact command is added to the documentation.

---

## A3. Write the RNS mathematics documentation

### New file

- `docs/RNS_MATH_SPEC.md`

### Required sections

1. **What RNS stores** — an integer represented by residues.
2. **Why moduli must be pairwise coprime.**
3. **Encoding** of positive and negative integers.
4. **Residue addition and multiplication.**
5. **Matrix multiplication in independent residue channels.**
6. **CRT reconstruction.**
7. **Garner reconstruction** with a small example such as moduli `3, 5, 7` and result `17`.
8. **Why CRT and Garner solve the same reconstruction problem.**
9. **Centered signed decoding** and the exact supported interval.
10. **Dot-product range requirement** for matrix multiplication.
11. **What overflow/wrap-around means.**
12. **Which statements are mathematical guarantees and which are only experimental observations.**

Keep the explanation readable, but every condition must be precise enough to cite in a report or article.

### Definition of done

A reader should be able to answer:

> Why is a selected modulus set safe for a given matrix multiplication, and under which assumptions can Garner recover the exact signed integer result?

---

# Task B — Transformer integration and model evaluation

Owner: Transformer / second-year team member.

## Goal in simple words

Create one reproducible script that runs the same model in several modes and reports quality, speed, and GPU memory in one machine-readable result file.

Do not modify the CUDA kernels. Use the existing v0.5 layers and public functions.

## B1. Create one model benchmark entry point

### Existing files to read first

- `scripts/evaluate_ppl.py`
- `scripts/verify_opt_qkv.py`
- `benchmarks/benchmark_attention.py`
- `src/rns_llm/layers/rns_linear.py`
- `src/rns_llm/layers/rns_qkv.py`

### New script

- `scripts/benchmark_model.py`

### Required modes

Initially support:

```text
fp16
rns-int8-1-block
rns-int8-4-blocks
rns-int8-all-attention-blocks
```

Reserve a clean mode name for the future baseline:

```text
native-int8
```

The script must fail with a clear message if a requested backend is not implemented. It must not silently fall back to FP16.

### Required command-line arguments

At minimum:

```text
--model
--mode
--prompt-tokens
--generate-tokens
--dataset-samples
--attention-blocks
--seed
--output
```

### Required output

Write JSON containing:

- model and mode;
- exact replaced module names;
- backend call counters;
- prompt length;
- generated token count;
- warmup count and measured iteration count;
- prefill latency;
- decode latency per token;
- total generation latency;
- tokens per second;
- PPL fields when requested;
- GPU and software environment;
- all memory fields from Task B2.

### Anti-cheating check

For every RNS mode:

```text
fused_gemm_calls > 0
```

If no RNS call was observed, the script must raise an error instead of reporting a result.

---

## B2. Add GPU memory instrumentation

Before each measured mode:

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

After the run, report:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Also calculate separate estimates for:

- model parameter bytes;
- original floating weights still stored by RNS layers;
- encoded RNS weight bytes;
- lookup-table bytes;
- reusable workspace bytes;
- peak allocated VRAM;
- peak reserved VRAM.

The output must make clear that LUT savings are not the same as whole-model savings.

### Suggested JSON section

```json
{
  "memory": {
    "model_parameters_bytes": 0,
    "floating_backup_weights_bytes": 0,
    "encoded_rns_weights_bytes": 0,
    "lookup_tables_bytes": 0,
    "workspace_bytes": 0,
    "peak_allocated_bytes": 0,
    "peak_reserved_bytes": 0
  }
}
```

### Definition of done

The same script must produce comparable memory values for FP16 and every RNS mode.

---

## B3. Run the PPL scaling experiment

Use the same model, dataset, split, seed, tokenizer, sequence length, and number of samples for every row.

Required RNS attention-block counts:

```text
1
4
8
all attention blocks
```

Use the current OPT replacement mode:

```text
--replacement-mode opt-qkv
--include-out-proj
```

Record:

- baseline PPL;
- RNS PPL;
- absolute difference;
- relative increase;
- whether the increase is below 5%;
- replaced module names;
- backend call counts.

### Output files

- `results/ppl_scaling.json`
- `results/ppl_scaling.md`

### Definition of done

- Every run uses the same evaluation configuration.
- Every RNS run proves that the backend was actually called.
- Results can be reproduced from exact commands written in the Markdown summary.

---

# Pull-request rules for both owners

1. Create a separate branch.
2. Do not edit `csrc/` or CUDA backend files unless agreed first.
3. Add or update tests for every behavior change.
4. Include the exact command used for testing.
5. Attach generated CSV/JSON summaries when the task requires them.
6. Keep one responsibility per pull request.

Suggested branches:

```text
math/rns-boundaries-and-moduli
integration/model-benchmark-and-memory
```
