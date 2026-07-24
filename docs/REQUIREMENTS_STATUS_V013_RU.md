# Соответствие исходному ТЗ после v0.13

| Требование | Статус | Проверка |
|---|---|---|
| Модули не более 8 бит | Реализовано | `select_plan`, все модули 3..255 |
| Оптимизация набора модулей | Реализовано как сравнение | `large_primes` против `school_small` |
| Учитывать overflow dot product | Реализовано | `M_RNS > 2*K*qmax^2` |
| RNS matrix multiplication | Реализовано | batched INT8 cuBLAS GEMM |
| FP32/FP16/q32/q16/q8 comparison | Реализовано | FP32, FP16, native INT8, full-RNS logical q8/q16/q32 |
| Non-modular reconstruction | Реализовано | Barrett/LUT + 128-bit Garner |
| Ограничить LUT 1-2 таблицами | Реализовано | `one` и `two` variants |
| Показать >50% LUT memory saving | Измеряется | относительно `all` LUT; q8 требует 1 table для строгого >50% |
| Проверить shared-table contention | Реализовано | 4 streams с общей LUT и весами |
| Проверить 4 concurrent connections | Частично | layer-level 4-stream benchmark, не full serving |
| PPL increase <5% | Код v0.12 сохранён | требуется отдельный Colab run |
| Ускорение относительно FP16 | Не предполагается заранее | подтверждается или опровергается результатами v0.13 |
| Full RNS Softmax/LayerNorm | Не реализовано | остаётся вне текущего layer-level scope |

## Важная оговорка

В исходном ТЗ память LUT и память всей модели смешаны. v0.13 выводит их отдельно. Сокращение двух таблиц относительно десяти на 80% не означает 80% экономии model storage.
