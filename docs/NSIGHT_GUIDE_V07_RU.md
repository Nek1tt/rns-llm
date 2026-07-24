# Интеграция NVIDIA Nsight для замороженной базовой версии RNS v0.7

## Назначение

Этот проект сохраняет вычислительную архитектуру CUDA v0.7 без изменений и перестраивает только окружение измерений. Из проекта удалены численные проверки, pytest, smoke-тесты, PPL/accuracy gates и сценарии Compute Sanitizer.

Профилирование разделено на три независимых уровня:

1. **End-to-end**: FP16-вход → вычисление масштаба → quantize/encode → четыре residue GEMM → Garner/dequant/bias → FP16-выход.
2. **Prepared RNS core**: residues входа создаются до измеряемого диапазона; внутри остаются четыре cuBLAS INT8 GEMM и fused Garner-epilogue.
3. **Отдельные kernels**: Nsight Compute отдельно анализирует quantize/encode, один representative cuBLAS GEMM и fused Garner/dequant kernel.

## NVTX-метки и границы захвата

Каждый запуск содержит вложенные push/pop ranges:

```text
RNS_V07_PROFILE
└── V07_<BACKEND>_<STAGE>_M<M>_K<K>_N<N>
```

Основной режим Nsight Systems начинает сбор по `torch.cuda.profiler.start()` и завершает по `torch.cuda.profiler.stop()`:

```text
--capture-range=cudaProfilerApi
--capture-range-end=stop
```

Это исключает сборку модулей, подготовку весов, выделение workspace и warmup. Если конкретная Colab-сессия не создаёт отчёт при таком режиме, скрипт автоматически повторяет запуск с full-process capture.

## Что собирает Nsight Systems

Основной набор трасс:

```text
cuda,nvtx,osrt,cublas
```

Также включены CPU sampling и context switches для process tree. Это позволяет отделить:

- реальное время GPU kernels;
- задержку между CPU launch и началом kernel;
- Python/PyTorch dispatcher overhead;
- синхронизации;
- gaps между короткими kernels;
- вызовы cuBLAS.

### GPU Metrics

GPU Metrics отключены по умолчанию, поскольку они требуют доступа к performance counters и увеличивают overhead сбора. Включение:

```bash
GPU_METRICS=1 bash scripts/profile_nsys_v07.sh ...
```

Скрипт определяет, какая форма параметра поддерживается установленной версией Nsight Systems:

```text
--gpu-metrics-device
--gpu-metrics-devices
```

Частота по умолчанию — 10 кГц. Для Tesla T4 это позволяет увидеть SM activity, tensor activity, memory activity, частоты и возможный throttling во времени.

## Защита от устаревшего SQLite

Nsight Systems может повторно использовать соседний `.sqlite`, даже если `.nsys-rep` был пересоздан с тем же именем. Это уже приводило к неполным или несогласованным summary.

В новой интеграции старый SQLite всегда удаляется, после чего выполняется один явный экспорт:

```text
nsys export --type=sqlite --force-overwrite=true
```

Все текстовые и CSV-отчёты строятся именно из этого свежего SQLite.

## Файлы, создаваемые одним запуском Nsight Systems

- `.nsys-rep` — основной отчёт для GUI;
- `.sqlite` — полный машинно-читаемый экспорт;
- `_summary.txt` — лог workload, expert analysis и все текстовые таблицы;
- `_manifest.json` — backend, stage, shape, repeats и имена файлов;
- CSV для:
  - общего GPU workload;
  - kernel summary;
  - grid/block summary;
  - хронологического CUDA GPU trace;
  - CUDA API summary;
  - launch-to-execution latency;
  - GPU memory operations;
  - NVTX CPU duration и GPU projection;
  - OS runtime и syscalls.

## Рекомендуемая матрица Nsight Systems

```bash
python scripts/run_nsight_matrix_v07.py \
  --baselines \
  --prepared \
  --large
```

Она создаёт:

- RNS v0.7 end-to-end для `M=1,16,128`, `K=N=768`;
- prepared RNS core для `M=1` и `M=128`;
- FP16, native INT8 и v0.6 baselines;
- RNS/FP16/native INT8 для `M=1`, `K=N=4096`.

## Nsight Compute

### Quantize + centered RNS encode

```bash
bash scripts/profile_ncu_v07.sh quantize e2e 1 768 768 detailed
```

### Один representative cuBLAS residue GEMM

```bash
bash scripts/profile_ncu_v07.sh gemm prepared 1 768 768 detailed
```

### Fused Garner + dequant + bias + FP16 epilogue

```bash
bash scripts/profile_ncu_v07.sh epilogue prepared 1 768 768 detailed
```

Для анализа prefill следует повторить GEMM и epilogue при `M=128`; для крупного decode — GEMM при `4096×4096`.

Каждый запуск создаёт:

- `.ncu-rep`;
- `_summary.txt` с Details page и rule output;
- `_raw.csv` с исходными метриками.

Nsight Compute использует kernel replay. Его Duration предназначен для диагностики отдельного kernel и не должен заменять end-to-end latency из CUDA Events или Nsight Systems.

## Что анализировать в Nsight Systems

1. **NVTX GPU Projection** — суммарная работа GPU на одну логическую projection.
2. **CUDA GPU Trace** — порядок и количество quantize, cuBLAS и epilogue kernels.
3. **Kernel Grid/Block Summary** — grid, block, registers, shared memory и средняя длительность.
4. **Kernel Launch & Exec Summary** — задержка CPU launch → GPU start и время выполнения.
5. **CUDA API Summary** — цена запусков, синхронизаций и cuBLAS API.
6. **Timeline gaps** — простаивает ли GPU между короткими kernels.
7. **GPU Metrics** — загрузка SM/Tensor/memory, частоты, power и throttling в пределах NVTX range.

## Что анализировать в Nsight Compute

- размеры grid/block;
- registers/thread и shared memory;
- theoretical и achieved occupancy;
- active/eligible warps per scheduler;
- Long Scoreboard, Math Pipe Throttle, LG Throttle;
- SM, L1/TEX, L2 и DRAM throughput;
- INT8/Tensor instruction mix;
- local loads/stores и register spills;
- Source/SASS correlation благодаря `-lineinfo` и `--import-source yes`.

## Правила воспроизводимости

- Компилировать только под текущую архитектуру через `TORCH_CUDA_ARCH_LIST`.
- Выполнять подготовку весов и workspace до начала capture.
- Полностью завершать warmup до `cudaProfilerStart`.
- Использовать одинаковые shape, repeats и CUDA Graph policy для сравниваемых методов.
- Сохранять `environment.txt` с версиями, clocks, power и температурой.
- Сравнивать методы внутри одной Colab-сессии: application clocks на hosted GPU могут быть не зафиксированы.
