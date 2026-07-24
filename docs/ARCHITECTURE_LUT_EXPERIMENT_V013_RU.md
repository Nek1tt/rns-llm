# v0.13: сравнение FP32/FP16/INT8 и full-RNS q8/q16/q32

## Цель

Эксперимент закрывает исходный план куратора:

1. FP32 GEMM как эталон скорости и точности;
2. FP16 GEMM;
3. RNS после логической q32-квантизации;
4. RNS после логической q16-квантизации;
5. RNS после логической q8-квантизации;
6. отдельная native INT8 baseline;
7. ablation compact LUT против арифметического Barrett reduction;
8. проверка четырёх одновременных запросов с общими весами и LUT.

`q8/q16/q32` означает разрядность исходного знакового целого квантователя. Все RNS-каналы всё равно хранятся и перемножаются как `int8`, поскольку каждый модуль не превышает 255.

## Корректный диапазон

Для матриц с размером редукции `K` требуется покрыть не диапазон одного операнда, а signed dot-product bound:

```
M_RNS > 2 * K * qmax^2,
qmax = 2^(bits-1) - 1.
```

Поэтому число каналов автоматически зависит от `K`, логической разрядности и набора модулей.

Для типичного `K=2560` политика `large_primes` обычно выбирает:

- q8: 4 канала;
- q16: 6 каналов;
- q32: 10 каналов.

Исходный школьный набор малых модулей требует больше каналов. Он доступен как политика `school_small` и нужен для прямой проверки тезиса «больше каналов даёт больше параллелизма».

## CUDA dataflow

Для каждого логического формата:

1. входная строка квантуется с per-row scale;
2. вес квантуется с per-output-column scale;
3. каждое целое кодируется центрированными остатками;
4. выполняется `C` INT8 GEMM через `cublasGemmStridedBatchedEx`;
5. каждый INT32 accumulator приводится к каноническому остатку;
6. Garner реконструирует signed dot product;
7. результат деквантуется в FP32.

Для q32 произведение модулей и аккумулятор превышают 64 бита. В v0.13 используется собственный двухсловный 128-битный unsigned representation `(hi, lo)`. Garner остаётся точным до перевода реконструированного целого в `double`, после чего итог сохраняется в FP32.

## Compact LUT

LUT из v0.7 возвращена как отдельная оптимизация modular reduction. Это не таблица `256 x 256` для умножения. Для одного модуля хранится:

```
[4 byte positions, 256 byte values], int16
```

Размер одной таблицы:

```
4 * 256 * 2 = 2048 bytes.
```

Сравниваются варианты:

- `none`: все каналы используют Barrett;
- `one`: LUT только для крупнейшего модуля;
- `two`: LUT для двух крупнейших модулей;
- `all`: LUT для всех каналов.

Экономия считается относительно памяти полного LUT subsystem. Она не является экономией всей памяти модели.

## Память

Отчёт разделяет:

- model-static weight storage;
- LUT storage;
- constant metadata;
- runtime activation and accumulator workspace;
- aggregate workspace четырёх запросов.

Full-RNS хранит один байт на вес на каждый канал. Поэтому q8/q16/q32 могут использовать соответственно примерно 4/6/10 байт на вес при `large_primes`. Отчёт дополнительно вычисляет плотный logical-q storage (1/2/4 байта на вес плюс FP32 scales) и отношение RNS storage к нему. Эксперимент должен честно показать, даёт ли LUT-экономия практический эффект на фоне residue-weight storage.

## Concurrency

Четыре CUDA stream используют:

- общие residue weights;
- одну общую compact LUT;
- раздельные activation residues, accumulators и outputs.

Измеряются:

- wall-clock latency четырёх запросов;
- throughput speedup относительно четырёх последовательных запусков;
- contention ratio;
- суммарный workspace.

Это layer-level concurrency benchmark, а не полноценный LLM serving benchmark.

## Выходные файлы

`benchmark_architecture_v013.py` создаёт:

- `architecture_results_v013.json`;
- `architecture_matrix_v013.csv`;
- `moduli_plans_v013.csv`;
- `lut_memory_v013.csv`;
- `concurrency_v013.csv`;
- `architecture_results_table.tex`;
- `lut_memory_table.tex`;
- `architecture_result_macros.tex`;
- `architecture_results_paragraph.tex`;
- `latency_vs_fp16_v013.png`;
- `weight_storage_v013.png`;
- `lut_tradeoff_v013.png`;
- `concurrency_v013.png`.

Числа разрешается переносить в статью только после реального запуска на целевой GPU.
