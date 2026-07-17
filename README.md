# RNS LLM CUDA Lab v0.5

Research prototype for mapping Residue Number System (RNS) arithmetic onto NVIDIA INT8 GPU operations inside Transformer inference.

## Primary objective

The main project goal is practical:

> Improve end-to-end LLM inference speed and/or reduce total GPU memory relative to FP16 and native INT8, while keeping the relative PPL increase below 5%.

Version 0.5 is the frozen correctness-first baseline before the final FP16/native-INT8/RNS comparison.

## What v0.5 contains

1. **Real QKV fusion** for OPT-style attention: three `q_proj/k_proj/v_proj` operations share one combined RNS GEMM.
2. **Continuous batching**: requests using the same weight can be concatenated along `M` and executed once.
3. **Safe adaptive channel prefixes**: a strict bound selects fewer channels only when reconstruction is guaranteed safe.
4. **Stream-safe workspaces**: concurrent CUDA streams do not reuse the same writable scratch buffers.
5. **Fused reconstruction**: modular reduction, Garner reconstruction, and signed correction run in one CUDA kernel.
6. **Prepared static weights**: encoded model weights are cached and reused.
7. **Compact lookup tables**: zero, one, or two small reconstruction tables can be benchmarked.
8. **Anti-cheating checks**: benchmarks compare against an independent NumPy `int64` oracle, and PPL reporting fails if the RNS backend was not actually executed.

Softmax is intentionally unchanged.

## Important current limitation

The cached byte-per-residue layout does **not** reduce model-weight memory:

```text
FP16:        2 bytes/value
native INT8: 1 byte/value
RNS 4 ch:    4 bytes/value
```

Lookup-table memory is much smaller in v0.5, but table savings must not be reported as whole-model savings. A real memory advantage requires packed residues or on-the-fly residue generation inside a fused kernel.

## Recorded pre-results

The supplied v0.5 run contains:

- exact fused QKV outputs with a conservative speedup of about `2.2x` over three separate RNS projections;
- `2.13x` continuous-batching speedup for four `M=1` requests after all requests are ready;
- exact adaptive-channel outputs, but slower runtime due to GPU-to-CPU synchronization;
- `0.429%` PPL increase for four real OPT attention blocks.

See:

- [`docs/RELEASE_V05.md`](docs/RELEASE_V05.md)
- [`results/v0.5/`](results/v0.5/)

## Team tasks

The next work is split into two independent areas:

- RNS mathematics and boundary validation;
- Transformer model benchmarking, PPL, and memory instrumentation.

Read [`docs/TEAM_TASKS_V05.md`](docs/TEAM_TASKS_V05.md).

## Build

```bash
python -m pip install --upgrade pip setuptools wheel ninja
RNS_LLM_BUILD_CUDA=1 pip install -e ".[dev]" --no-build-isolation
python scripts/smoke_cuda.py
pytest -q -m cuda
```

CPU-only reference checks:

```bash
RNS_LLM_BUILD_CUDA=0 pip install -e ".[dev]"
pytest -q
python scripts/smoke_reference.py
```

## Main experiments

### QKV fusion

```bash
python benchmarks/benchmark_qkv_fusion.py \
  --tokens 1 16 128 256 \
  --hidden 768 \
  --output results/qkv_fusion.json
```

The fused output must be bit-for-bit equal to three separate `RNSLinear` operations.

### Continuous batching: 1/2/4 requests

```bash
python benchmarks/benchmark_concurrency.py \
  --m 1 --k 768 --n 768 \
  --concurrency 1 2 4 \
  --output results/concurrency_v05_m1.json
```

Queueing delay is not included and is explicitly marked in JSON.

### Adaptive channel count

```bash
python benchmarks/benchmark_adaptive_channels.py \
  --m 128 --k 768 --n 768 \
  --output results/adaptive_channels.json
```

A prefix is selected only when its centered RNS capacity is strictly larger than the computed safe bound. Otherwise the full set is used; silent wrap-around is forbidden.

### Verify OPT QKV arithmetic

```bash
pip install -e ".[transformer]" --no-build-isolation
python scripts/verify_opt_qkv.py --model facebook/opt-125m --tokens 32
```

### PPL with actual attention blocks

One fused QKV block plus its output projection:

```bash
python scripts/evaluate_ppl.py \
  --model facebook/opt-125m \
  --replacement-mode opt-qkv \
  --attention-blocks 1 \
  --include-out-proj \
  --dataset-samples 64
```

Four actual Transformer attention blocks:

```bash
python scripts/evaluate_ppl.py \
  --model facebook/opt-125m \
  --replacement-mode opt-qkv \
  --attention-blocks 4 \
  --include-out-proj \
  --dataset-samples 256
```

The script prints backend call counts and stops with an error if the model did not execute the fused RNS backend.

## Correctness policy

See [`docs/CORRECTNESS_GUARDS.md`](docs/CORRECTNESS_GUARDS.md).
