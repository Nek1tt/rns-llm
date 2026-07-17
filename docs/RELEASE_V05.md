# v0.5 baseline — frozen pre-results

## Primary project objective

The primary target is practical end-to-end LLM inference acceleration and/or total GPU-memory reduction relative to FP16 and native INT8, with a relative PPL increase below 5%.

INT12 remains an auxiliary wider-than-INT8 arithmetic experiment. It is not the default model path because the current INT8 RNS attention experiment already satisfies the quality threshold.

## Implemented in v0.5

- centered signed 8-bit RNS residues;
- batched cuBLAS INT8 residue GEMM;
- fused modulo + Garner signed reconstruction;
- prepared encoded-weight caches;
- reusable stream-safe workspaces;
- dense-coprime modulus strategies;
- compact 0/1/2 lookup-table modes;
- fused OPT QKV projection;
- continuous batching benchmark for 1/2/4 requests;
- mathematically safe adaptive-channel experiment;
- OPT attention-block replacement and PPL evaluation;
- backend counters that reject silent fallback results.

Softmax is unchanged.

## Recorded pre-results

### QKV fusion

Full fused outputs matched three separate RNS linear projections exactly (`max_absolute_error = 0`).

Representative p50 speedups:

| Tokens | Three separate RNS projections | One fused QKV | Speedup |
|---:|---:|---:|---:|
| 1 | 1.968 ms | 0.859 ms | 2.29x |
| 16 | 2.184 ms | 0.592 ms | 3.69x |
| 128 | 1.390 ms | 0.635 ms | 2.19x |
| 256 | 1.544 ms | 0.702 ms | 2.20x |

The 16-token result should be treated as a strong run rather than the guaranteed universal speedup. The stable conservative message is approximately 2.2x versus three separate RNS projections.

### Continuous batching

For four requests with `M=1`:

```text
independent CUDA streams: 0.869 ms p50
continuous batch:         0.408 ms p50
speedup:                  2.13x
```

For `M=128`, continuous batching was slightly slower than independent streams. The correct future design is a workload-dependent dispatcher, not unconditional batching.

Queue formation delay is not included in these numbers.

### Adaptive channels

The safe bound selected three channels and produced exact outputs in the recorded tests. However, the current runtime GPU-to-CPU synchronization made adaptive execution about 1.8x slower than the fixed four-channel path. This mode is a correctness experiment, not a production default.

### PPL

One real OPT attention block (`fused QKV + out_proj`):

```text
baseline_ppl:       31.571423
rns_ppl:            31.570559
relative_increase: -0.003%
```

Four real OPT attention blocks:

```text
baseline_ppl:      32.923428
rns_ppl:           33.064659
relative_increase: 0.429%
```

The four-block result is below the 5% requirement. The two PPL runs used different sample counts, so baselines must only be compared within the same run.

## Important limitation

The current cached residue layout does not reduce model-weight memory:

```text
FP16:        2 bytes/value
native INT8: 1 byte/value
RNS 4 ch:    4 bytes/value
```

Lookup-table memory was greatly reduced, but lookup-table savings are not whole-model savings. A real memory advantage will require packed residues or on-the-fly residue generation inside a fused kernel.

## Next mandatory baseline

A fair native INT8 model path must use the same quantized tensors, scales, QKV fusion, and batching policy as RNS. The final practical comparison must report:

- PPL;
- prefill latency;
- decode latency;
- TTFT;
- tokens/s for 1 and 4 requests;
- model-weight VRAM;
- workspace and temporary memory;
- peak allocated and reserved VRAM.
