# RNS LLM v0.11 — prefill-first hybrid architecture

## Цель

Проверить, может ли channel-structured hybrid обогнать FP16 в prefill-режиме на Tesla T4 после устранения главных дефектов v0.10:

- generic PyTorch `index_select` и padding;
- два полноценных GEMM вместо tiny-rank correction;
- последовательные preprocessing, correction и merge;
- случайный выбор первых pack-файлов вместо лучших PASS-слоёв;
- generic cuBLAS для `Kprotected=3`.

Архитектура рассчитана на слои, где аудит нашёл несколько устойчивых protected input channels. Для OPT-2.7B сильными кандидатами были `q_proj` и `fc1`, где 3 из 2560 входных каналов устраняли большую часть ошибки native INT8.

## Математическая декомпозиция

Исходное произведение:

\[
Y=XW^T+b.
\]

Пусть \(\mathcal P\) — статический набор protected input channels. Основная матрица весов хранит нули в protected columns:

\[
W_{main}[:,p]=0,\qquad p\in\mathcal P.
\]

Тогда:

\[
Y=Q_8(X_{safe})Q_8(W_{main})^T+
Q_q(X_{\mathcal P})Q_q(W_{\mathcal P})^T+b.
\]

Вторая часть является не вторым большим GEMM, а rank-\(|\mathcal P|\) correction. При трёх protected channels её арифметическая стоимость пропорциональна \(3MN\), а не \(KMN\).

## Шаг 1 — cuBLASLt INT8 против FP16

Вместо generic PyTorch executor используются постоянные `cublasLt` планы:

- FP16 inputs, FP32 accumulation/output;
- INT8 inputs, INT32 accumulation;
- row-major layouts;
- heuristic выбирается один раз при создании плана;
- workspace выделяется заранее;
- повторные вызовы не выполняют algorithm search и allocations.

Sweep:

```text
M = 16, 32, 64, 128, 256, 512, 1024, 2048
```

Реальные формы берутся из PASS-слоёв `q_proj` и `fc1`.

Gate A:

\[
T_{INT8,core}\le 0.70\,T_{FP16,core}.
\]

Если native INT8 не создаёт не менее 30% временного запаса, RNS correction не сможет сделать prefill быстрее FP16.

## Шаг 2 — main path без gather

Вход не делится на два больших tensor. Один fused preprocessing kernel:

1. вычисляет per-row absmax только по safe channels;
2. квантует весь `[M,K]` tensor в INT8;
3. записывает нули на protected positions;
4. извлекает только малый protected vector;
5. создаёт FP16 protected buffer;
6. одновременно кодирует protected values в q8/q16 residues.

Основной INT8 GEMM сохраняет исходную форму `[M,K] × [K,N]`, alignment и Tensor Core-friendly layout.

## Шаг 3 — специализированный rank-k correction

### RNS

Protected weights заранее упакованы как:

```text
[C, N, PaddedP]
```

Protected activations:

```text
[C, M, PaddedP]
```

Внутри custom CUDA kernel четыре signed INT8 пары обрабатываются одной `__dp4a`. Для q16 при `P=3`, `PaddedP=4`, используется пять residue channels и всего пять DP4A на output element. Garner выполняется в том же kernel.

### FP16 baseline

FP16 correction реализован с тем же dataflow и `__half2`, а не через generic tiny GEMM. Это обязательный конкурент RNS.

## Шаг 4 — fused merge/epilogue

Serial-вариант:

```text
fused preprocess
→ cuBLASLt INT8 main GEMM
→ correction + Garner + main dequant + bias + output
```

Correction output и отдельный merge tensor не создаются.

Parallel-вариант:

```text
stream 1: cuBLASLt INT8 main GEMM
stream 2: rank-k correction
          ↓
    fused merge/dequant/bias
```

Оба режима измеряются, поскольку на prefill основной INT8 GEMM может насыщать Tensor Cores и параллельная correction-ветка может либо скрываться на CUDA cores, либо конкурировать за ресурсы.

## Шаг 5 — три независимых gate

### Gate A: основной INT8

```text
INT8 core <= 0.70 × FP16 core
```

### Gate B: correction budget

```text
RNS correction + fused epilogue <= 0.15 × FP16 core
```

или практически эквивалентная экономия за счёт overlap.

### Gate C: полный слой

```text
best hybrid RNS E2E <= 0.90 × FP16 E2E
```

при снижении ошибки относительно native INT8.

Вердикты:

- `CONTINUE_PREFILL_INTEGRATION` — Gate C пройден хотя бы на одной форме;
- `CONTINUE_CORRECTION_ENGINEERING` — main INT8 и correction budgets уже проходят, но E2E ещё нет;
- `REDESIGN_CORRECTION` — main INT8 имеет запас, но correction слишком дорогая;
- `STOP_PREFILL_ON_T4` — main INT8 не создаёт необходимого запаса.

## Форматы benchmark

Основной v0.11 benchmark измеряет:

- FP32;
- FP16;
- native INT8;
- optimized INT8 + FP16 correction;
- optimized INT8 + RNS q16 correction;
- serial и parallel variants;
- prepared core и end-to-end.

Дополнительный compatibility benchmark сохраняет матрицу форматов из предыдущего ТЗ:

- full RNS q8;
- full RNS q16;
- full RNS q32;
- generic hybrid RNS q8/q16/q32.

Это позволяет отделить новый optimized prefill executor от полной RNS reference-линии.

## Ограничения

- CUDA-сборка должна быть проверена на реальном GPU; пакет подготовлен в среде без `nvcc` и NVIDIA GPU.
- Optimized RNS correction поддерживает до восьми 8-bit residue channels; q32 остаётся reference path старого extension.
- Перестановка каналов физически не выполняется: main weights сохраняют исходный K, а protected columns зануляются.
- На уровне полной модели ещё не измеряются perplexity и tokens/s. Сначала должен пройти layer-level Gate C.
