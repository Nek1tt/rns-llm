# Changelog v0.13.0

## Добавлено

- full-RNS logical q8/q16/q32 benchmark;
- автоматический выбор модулей из полного signed dot-product bound;
- `large_primes` и исходная `school_small` policy;
- compact reduction LUT с вариантами 0/1/2/all channels;
- arithmetic Barrett baseline;
- двухсловный 128-bit Garner для q32;
- FP32, FP16 и native INT8 baselines;
- раздельный учёт weight, dense logical-q, LUT, constants и workspace memory;
- four-stream shared-weight/shared-LUT benchmark;
- Python arbitrary-precision sample oracle;
- CSV, JSON, LaTeX и PNG assets для статьи;
- отдельный Colab notebook.

## Исправлено перед release

В раннем рабочем варианте v0.13 block-wide max reduction передавал правильный максимум только первому warp. Это могло приводить к неверным scales и saturated quantization в остальных warp. Финальный release использует двухступенчатую warp reduction с записью результата в shared memory и broadcast после `__syncthreads()`, аналогично проверенной v0.11 реализации.

Preflight дополнен сравнением LUT/Barrett outputs и sample-level Python arbitrary-precision oracle для q8/q16/q32.

## Ограничения проверки в среде сборки

CPU mathematical tests, Python/shell/notebook syntax and CPU wheel packaging проверены локально. CUDA toolkit и NVIDIA GPU в среде упаковки отсутствовали, поэтому фактическая компиляция `_ARCH` и GPU preflight выполняются первой частью Colab notebook и должны пройти до основного benchmark.
