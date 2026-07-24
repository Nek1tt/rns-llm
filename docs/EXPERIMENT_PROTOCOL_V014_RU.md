# Экспериментальный протокол v0.14.2

## 1. Единая матрица методов

На одинаковых weights и activations сравниваются:

- FP32 reference;
- FP16;
- native symmetric INT8;
- full-RNS INT8/16/32;
- hybrid INT8 + FP16 correction;
- hybrid INT8 + RNS q8/q16/q32 correction.

Для каждого RNS method проверяются LUT policies `none/one/two/all`. Для hybrid также сравниваются serial и parallel execution.

## 2. Диапазон RNS

Для signed dot product используется bound

```text
M_RNS > 2 * K * qmax^2 + 1.
```

В full-RNS `K` — полный reduction dimension. В hybrid `K` заменяется на padded protected rank `Ppad`, поскольку RNS считает только correction branch.

## 3. Память

Отдельно фиксируются:

- FP16/native/RNS weight bytes;
- active и actually allocated LUT bytes;
- reusable runner workspace;
- `torch.cuda.max_memory_allocated()`;
- memory при четырёх stream.

Экономия LUT никогда не называется экономией всей модели.

## 4. Attention

В OPT attention объединяются q/k/v weights и выполняется одна fused projection. Выход разбивается обратно на Q, K, V. `out_proj` заменяется тем же architecture mode. Native `QK^T`, mask, Softmax и `AV` сохраняются и измеряются вместе с projections.

## 5. PPL

Calibration выбирает protected input dimensions на fit/held-out tokens. Затем одинаковый plan применяется к hybrid variants. PPL считается sliding-window методом. В JSON сохраняются actual kernel call counts, elapsed seconds, peak memory и PASS/FAIL для порога `<5%`.

## 6. Concurrency

Один module и его prepared weights используются четырьмя CUDA streams. Writable workspace создаётся отдельно на stream, а LUT tensor и weights разделяются. Выводятся wall latency, throughput speedup и contention ratio.

## 7. Nsight

Для каждого workload:

- NCU: `.ncu-rep`, all raw metric rows from the selected article-essential sections, details CSV/JSON/text; exhaustive `NCU_MODE=full` is optional;
- NSYS: `.nsys-rep`, full SQLite, JSON stats, JSONL при поддержке, schema SQL, executed queries SQL и derived JSON.

Profiler workload ограничен NVTX/CUDA Profiler API range `PROFILE`, поэтому warmup не смешивается с измеряемым участком.
