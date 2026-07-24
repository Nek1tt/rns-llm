# RNS LLM v0.14.2 — full-RNS and hybrid Attention study

Это единый release для итоговой статьи SMILES 2026. Он объединяет full-RNS ветку v0.7/v0.13 и hybrid ветку v0.11.3, возвращает compact LUT в обе архитектуры и запускает одинаковые проверки на одном GPU.

## Что проверяется

1. **Full-RNS**
   - RNS-INT8, RNS-INT16 и RNS-INT32;
   - каждый modulus не превышает 8 бит;
   - large-prime и school-small moduli policies;
   - v0.7 optimized q8 epilogue как отдельный вариант;
   - v0.13 128-bit Garner для INT16/INT32.

2. **Hybrid**
   - INT8 main path;
   - FP16 correction как контроль;
   - RNS correction q8/q16/q32;
   - serial и parallel execution;
   - protected channels выбираются детерминированно либо из calibration plan.

3. **LUT ablation для обеих архитектур**
   - `none`, `one`, `two`, `all`;
   - реально выделяется только активное число таблиц;
   - одна compact LUT имеет форму `[4,256] int16`, то есть 2048 bytes;
   - один LUT tensor переиспользуется runners разных CUDA streams.

4. **Сравнение**
   - FP32 reference, FP16, native INT8, full-RNS, hybrid FP16, hybrid RNS;
   - core/preprocess/end-to-end latency;
   - relative L2, cosine, max/mean absolute error;
   - static weight memory, allocated LUT memory, workspace и CUDA peak memory;
   - четыре одновременных CUDA streams;
   - полная OPT self-attention latency.

5. **PPL**
   - WikiText-2 sliding-window PPL;
   - реальные CUDA kernels заменяют fused QKV и output projection в выбранных OPT blocks;
   - автоматический gate: relative PPL increase `<5%`;
   - LUT policies проверяются отдельно.

6. **Nsight**
   - Nsight Systems `.nsys-rep`, полный `.sqlite`, schema/queries SQL, stats JSON и derived JSON;
   - Nsight Compute `.ncu-rep`, `essential` sections по умолчанию (SpeedOfLight, occupancy, memory/compute, scheduler/instruction metrics), raw/details CSV и полные JSON; `NCU_MODE=full` оставлен как опциональный exhaustive режим;
   - matrix и complete-attention workloads для обеих архитектур.

## Attention scope

Оптимизированные projection kernels заменяют fused `Q/K/V` и `out_proj`. Операции `QK^T`, causal masking, Softmax и `AV` остаются native PyTorch/Transformers и входят в измеряемую полную latency Attention. Это позволяет измерить реальный overhead non-modular operations без выдачи их за RNS-реализацию.

## Главный notebook

Загрузите в Google Colab:

- `RNS_LLM_v0142_Full_Hybrid_Attention_Colab.ipynb`;
- `rns_llm_v014_2_full_hybrid_attention_project.zip`.

Notebook:

1. клонирует `Nek1tt/rns-llm` на фиксированном commit;
2. применяет overlay;
3. собирает все CUDA extensions;
4. выполняет preflight;
5. запускает unified matrix, Attention, PPL и Nsight;
6. создаёт `rns_llm_v0142_results.zip`.

Все длительные этапы включаются отдельными переменными в первой ячейке. Стандартный Nsight-набор ограничен 55 минутами: три Systems timeline и два Compute kernel-профиля. Существующие корректные manifest-файлы пропускаются при повторном запуске.

## Основные команды после применения overlay

```bash
make install-cuda
make v014-preflight
make v014-matrix-paper
make v014-attention-paper
make v014-ppl-paper
make v014-nsight
make v014-summary
make v014-collect
```

## Результаты

```text
results/v0.14.2/preflight_v014.json
results/v0.14.2/matrix/
results/v0.14.2/attention/
results/v0.14.2/ppl/
results/v0.14.2/summary/
reports/v0.14.2/ncu/
reports/v0.14.2/nsys/
```

## Важное ограничение упаковки

CUDA build и GPU benchmark не выполнялись в среде упаковки, где нет NVIDIA Toolkit/GPU. Notebook обязательно компилирует extensions и прекращает работу при провале preflight. Никакие latency, PPL или Nsight результаты заранее не подставлены.
