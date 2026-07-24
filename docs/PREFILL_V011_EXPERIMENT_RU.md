# Протокол эксперимента v0.11

## 1. Статистический аудит

Модель по умолчанию: `facebook/opt-2.7b`.

Аудит:

- 16 calibration batches;
- sequence length 128;
- train/held-out split по batches;
- 8 representative layers на разных глубинах;
- сохраняются только лучшие PASS-слои;
- приоритет `q_proj`, `k_proj`, `v_proj`, `fc1`, `up_proj`, `gate_proj`;
- pack ranking — по held-out снижению ошибки native INT8.

## 2. Основной prefill benchmark

Для каждого из четырёх лучших PASS-слоёв:

```text
M = 16 32 64 128 256 512 1024 2048
```

Измерения:

- p50, p95, p99, min, max;
- prepared core;
- FP32-input end-to-end;
- relative L2, RMSE, MAE, max error, cosine;
- storage bytes;
- serial/parallel ratio;
- Gate A/B/C.

## 3. Reference format matrix

На двух лучших слоях и нескольких M запускается старый reference benchmark:

```text
FP32
FP16
native INT8
full RNS q8/q16/q32
generic hybrid INT8 + FP16
generic hybrid INT8 + RNS q8/q16/q32
```

Он нужен не для выбора новой архитектуры, а для сохранения сопоставимости с исходным ТЗ.

## 4. Nsight Systems

Профилируются на первом лучшем PASS-слое при M=128 и M=512:

- FP16 prepared;
- native INT8 prepared;
- hybrid RNS serial prepared;
- hybrid RNS parallel prepared;
- hybrid FP16 serial/parallel prepared;
- hybrid RNS serial/parallel end-to-end.

Собираются:

- `.nsys-rep`;
- свежий `.sqlite`;
- CUDA kernel/API summaries;
- NVTX GPU projection;
- chronological GPU trace;
- memory operations;
- OS runtime;
- optional Turing GPU Metrics.

NVTX stages:

```text
V011_FUSED_PREPROCESS
V011_FP16_GEMM
V011_NATIVE_INT8_MAIN
V011_MAIN_INT8
V011_RNS_CORRECTION
V011_FP16_CORRECTION
V011_RNS_FUSED_EPILOGUE
V011_HYBRID_RNS_PARALLEL
```

## 5. Nsight Compute

Detailed reports собираются отдельно для:

- FP16 cuBLASLt kernel;
- native/main INT8 cuBLASLt kernel;
- fused preprocessing;
- DP4A RNS correction;
- half2 FP16 correction;
- fused RNS Garner/dequant/bias epilogue.

Sections:

```text
LaunchStats
Occupancy
SpeedOfLight
SchedulerStats
WarpStateStats
MemoryWorkloadAnalysis
InstructionStats
SourceCounters
```

NCU может быть запрещён в hosted Colab (`ERR_NVGPUCTRPERM`). Скрипт сохраняет остальные результаты и не уничтожает Nsys bundle.

## 6. Что отправить на анализ

Последняя ячейка создаёт:

```text
rns_hybrid_v011_prefill_results.zip
```

В архив не включаются тяжёлые weight packs, но включаются:

- audit JSON;
- optimized benchmark JSON;
- full format reference JSON;
- human-readable summary;
- environment probe;
- все Nsys/NCU reports, SQLite и CSV;
- CUDA/Python source snapshot.
