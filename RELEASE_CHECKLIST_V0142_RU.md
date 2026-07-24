# RNS LLM v0.14.2 — проверка покрытия ТЗ

## Архитектуры

- **Full-RNS:** legacy optimized q8 path v0.7 и generalized q8/q16/q32 path v0.13 с двухсловным 128-bit Garner.
- **Hybrid:** v0.11.3 INT8 main branch + FP16 correction control или RNS q8/q16/q32 correction; serial и parallel execution.
- **Baselines:** FP16, FP32 reference на matrix-level и native INT8.

## Соответствие требованиям

| Требование | Что запускает notebook | Артефакт результата |
|---|---|---|
| Moduli не более 8 бит | dense-coprime, large-prime и school-small планы; проверка signed dot-product bound для INT8/16/32 | `matrix_benchmark_v014.json`, preflight |
| Speed/parallelism vs memory tradeoff | число channels, latency, weight/LUT/workspace/peak bytes | matrix JSON/CSV/LaTeX |
| Full-RNS GEMM | v0.7 q8 и v0.13 q8/q16/q32 actual CUDA kernels | matrix + Attention + Nsight |
| Hybrid GEMM | INT8 main + FP16/RNS protected correction | matrix + Attention + Nsight |
| Self-Attention | fused QKV и replaced out-projection; QK^T, mask, Softmax и AV включены в total latency | `attention_benchmark_v014.json` |
| Non-modular overhead | измеряется внутри полного OPT Attention; projections отмечены NVTX, остаток виден в NSYS/NCU | Attention JSON, `.nsys-rep`, `.sqlite`, `.ncu-rep` |
| LUT reuse | `none/one/two/all` в latency/accuracy benchmarks; policy `two` используется в PPL/Nsight; один immutable LUT tensor разделяется между stream-specific runners | matrix/Attention/PPL + Nsight |
| Ограничение 1–2 LUT и >50% LUT saving | actual allocated LUT bytes сравниваются с all-LUT | unified summary |
| Shared-table contention | 4 CUDA streams, shared prepared weights/LUT, separate writable workspaces/coordinator state | matrix/Attention concurrency + Nsight |
| Strict latency | core, preprocess, E2E p50/p95 и ratio к FP16 | JSON/CSV |
| 4 concurrent connections | layer-level и complete-Attention four-stream throughput/contension | matrix/Attention JSON |
| PPL <5% | WikiText-2; calibration=validation, evaluation=test; actual CUDA QKV/out paths; automatic gate | `ppl_unified_v014.json` |
| Model memory | whole-model allocated/peak CUDA memory during PPL и component-level storage | PPL JSON + summary |
| Nsight Systems | `.nsys-rep`, full `.sqlite`, schema SQL, executed queries SQL, stats JSON, derived JSON, optional JSONL | `reports/v0.14.2/nsys/` |
| Nsight Compute | `.ncu-rep`, article-essential section set по умолчанию, raw/details CSV и complete parsed JSON rows; exhaustive `--set full` опционален | `reports/v0.14.2/ncu/` |

## Обязательный preflight

До длинных экспериментов notebook проверяет:

- native INT8;
- full-RNS v0.13 q8/q16/q32 и LUT equivalence;
- legacy full-RNS v0.7 с 0/1/2 LUT;
- hybrid RNS q8/q16/q32, serial и parallel, LUT equivalence;
- hybrid FP16 serial и parallel;
- finite outputs и равенство execution paths с указанным допуском.

## Что является результатом запуска, а не свойством релиза

Проект содержит все требуемые реализации и измерения, но заранее не утверждает, что целевые speedup, memory saving или PPL gate будут достигнуты. Их фактический PASS/FAIL формируется только из результатов пользовательского GPU запуска.
