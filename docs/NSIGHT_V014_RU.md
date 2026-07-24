# Nsight v0.14.2

## Рекомендуемый запуск для статьи (до 55 минут)

```bash
python scripts/run_minimal_nsight_v014.py \
  --output-root reports/v0.14.2 \
  --model facebook/opt-2.7b \
  --matrix-shape 16x2560x2560 \
  --attention-seq 64 \
  --total-minutes 55 \
  --ncu-mode essential
```

Набор намеренно репрезентативный, а не декартов продукт всех вариантов:

- Nsight Systems, complete Attention: FP16, full-RNS INT8 и hybrid-RNS q16;
- Nsight Compute, matrix kernels: full-RNS INT8 и hybrid-RNS q16;
- LUT policy `two`; все `none/one/two/all` уже сравниваются обычными benchmark-скриптами.

## Nsight Compute

```bash
bash scripts/profile_ncu_v014.sh hybrid_rns_q16 matrix two reports/v0.14.2/ncu
```

По умолчанию `NCU_MODE=essential`: собираются доступные sections SpeedOfLight, LaunchStats, Occupancy, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, InstructionStats, SchedulerStats и WarpStateStats. Число профилируемых запусков ограничено `NCU_MAX_LAUNCHES=4`.

Выход:

- `.ncu-rep`;
- `_raw.csv` и `_raw_full.json` со всеми полученными metric rows;
- `_details.txt`;
- `_details.csv` и `_details_full.json`, когда CSV details поддерживается установленной версией;
- `_manifest.json` только после проверки обязательных файлов.

Для exhaustive режима одного выбранного варианта:

```bash
NCU_MODE=full NCU_MAX_LAUNCHES=2 \
  bash scripts/profile_ncu_v014.sh full_rns_int8 matrix two reports/v0.14.2/ncu_full
```

## Nsight Systems

```bash
bash scripts/profile_nsys_v014.sh full_rns_int8 attention two reports/v0.14.2/nsys
```

Скрипт автоматически выбирает совместимый параметр окончания capture range:

- текущий CLI: `--capture-range-end=stop`;
- legacy CLI: `--stop-on-range-end=true`.

Выход:

- `.nsys-rep` (или legacy `.qdrep`);
- полный `.sqlite`;
- `_schema.sql`, `_tables.txt`, `_queries.sql`;
- `_sql_summary.json`;
- report-specific stats JSON/status;
- `.jsonl`, когда exporter поддерживает `jsonlines`;
- `_manifest.json` только после проверки обязательных файлов.

## Возобновление

`run_minimal_nsight_v014.py` пропускает только профили с валидным manifest и существующими непустыми артефактами. Незавершённые профили запускаются заново.
