# Интеграция результатов v0.13 в статью

После Colab run распакуйте `rns_architecture_v013_results.zip` и выполните:

```bash
python scripts/integrate_architecture_into_paper.py \
  --results-dir path/to/results/v0.13 \
  --paper-dir path/to/overleaf-project
```

Скрипт копирует LaTeX assets в `generated/v013` и создаёт файл
`generated/v013_architecture_snippet.tex` с командами `\\input`.

Не вставляйте автоматически сгенерированный вывод в Abstract до проверки:

- одинаковости LUT и Barrett outputs;
- exact sample verification;
- отсутствия OOM/thermal throttling;
- одинакового GPU для всех вариантов;
- разделения core и end-to-end latency;
- корректной трактовки LUT memory savings.
