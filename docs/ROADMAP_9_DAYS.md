# 9-day roadmap

| Day | Main result |
|---|---|
| 1 | Interfaces frozen; encode/decode checked; model selected |
| 2 | Correct reference RNS matmul; `RNSLinear` fallback runs |
| 3 | Moduli baseline; one-modulus GPU prototype; one Linear replaced |
| 4 | Multi-channel GPU path; exact GPU/reference comparison |
| 5 | Batched/grouped experiment; first end-to-end model run |
| 6 | Optimize measured bottleneck; first PPL result |
| 7 | Target-shape benchmarks; memory profile; correctness fixes |
| 8 | Final latency/PPL/memory comparison; optional concurrency |
| 9 | Freeze code; export results; limitations and conclusion |

Priority:

```text
correctness > integration > reproducible benchmark > optimization > extras
```
