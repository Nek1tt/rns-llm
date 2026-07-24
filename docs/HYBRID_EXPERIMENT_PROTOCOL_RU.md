# Протокол эксперимента v0.10

## Фаза A — аудит реальной модели

По умолчанию используется `facebook/opt-2.7b`, FP16 weights и WikiText-2
validation. При недоступности датасета используются встроенные calibration
prompts.

Результат:

- `results/v0.10/model_audit.json`;
- временные packs representative layers;
- решение `PROCEED` или `STOP`.

Важно различать:

- `row_outlier_rate` — сколько токеновых строк содержат хотя бы один выброс;
- `protected_ratio` — сколько input channels реально направлено в correction;
- `protected_ratio_after_padding` — сколько MAC фактически приходится на
  correction после округления размеров до кратности четырём.

Производительность определяется последней величиной.

## Фаза B — benchmark из FP32 matrices

Для каждого прошедшего слоя и `M = 1, 16, 128` исходными данными являются FP32
матрицы. Измеряются CUDA Events p50/p95/p99 и accuracy относительно FP32.

Обязательные методы из ТЗ:

| Метод | Содержание |
|---|---|
| `fp32` | FP32 GEMM, reference |
| `fp16` | cast FP32→FP16, FP16 GEMM |
| `full_rns_q32` | logical INT32 quantization, RNS GEMM |
| `full_rns_q16` | logical INT16 quantization, RNS GEMM |
| `full_rns_q8` | logical INT8 quantization, RNS GEMM |

Дополнительные необходимые baselines:

| Метод | Зачем нужен |
|---|---|
| `native_int8` | показывает, оправдана ли RNS вообще |
| `hybrid_int8_plus_fp16` | наиболее простой mixed-precision конкурент |
| `hybrid_int8_plus_rns_q8/q16/q32` | исследуемая архитектура |

Accuracy metrics:

- RMSE;
- MAE;
- maximum absolute error;
- relative L2;
- cosine similarity.

Для RNS и native INT8 отдельно измеряются:

- end-to-end: scale, quantize/encode, GEMM, reconstruction/dequant;
- prepared core: подготовленная activation representation → GEMM → epilogue.

## Фаза C — окончательный prototype gate

Для первого prototype основным кандидатом считается
`hybrid_int8_plus_rns_q16`.

Решение `CONTINUE_KERNEL_OPTIMIZATION` выдаётся только если:

1. хотя бы на одной реальной layer/shape hybrid q16 быстрее FP16;
2. на всех измеренных layer/shapes его relative-L2 не хуже native INT8.

Иначе решение — `STOP_OR_REDESIGN`. Это намеренно жёсткий критерий: наличие
интересной статистики outliers само по себе не доказывает practical speedup.

## Nsight

Nsight Systems собирает:

- CUDA/NVTX/cuBLAS/OS runtime;
- launch gaps;
- chronological GPU trace;
- optional Turing GPU Metrics;
- fresh SQLite и CSV summaries.

NVTX stages hybrid:

- `HYBRID_GATHER`;
- `HYBRID_NATIVE_INT8_MAIN`;
- `HYBRID_RNS_Q*_PROTECTED`;
- `HYBRID_MERGE`.

Nsight Compute профилирует отдельно:

- encode/quantize;
- INT8 GEMM;
- U128 Garner epilogue;
- native dequant epilogue.

## Интерпретация

- `STOP` на фазе A означает, что модель не даёт достаточно малого и стабильного
  channel subspace; CUDA-архитектура не проектируется дальше.
- `PROCEED`, но `STOP_OR_REDESIGN` на фазе C означает, что статистическая идея
  правдоподобна, однако unfused implementation overhead съел выигрыш.
- Только `CONTINUE_KERNEL_OPTIMIZATION` оправдывает fused/CUTLASS prototype и
  end-to-end model integration.

## Исправление v0.10.1

Первоначальный gate по Jaccard между top-1% каналов признан некорректным для очень малого protected set. При K=2560 top-1% содержит 26 каналов, хотя реально критическими могут быть только 3–10; нестабильные каналы-заполнители искусственно снижают Jaccard.

В v0.10.1 карта строится на первой половине calibration batches, а уменьшение ошибки проверяется на второй, ранее не использованной половине. Входные блоки не содержат padding-токенов. Jaccard и overlap сохраняются как диагностические показатели, но решение PROCEED/STOP определяется held-out accuracy, размером protected subspace и аналитическим performance bound.
