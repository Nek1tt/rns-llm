# v0.12.0: проверка требования PPL < 5%

## Что именно измеряется

Требование проекта интерпретируется как относительный рост perplexity относительно FP16:

\[
\Delta_{PPL}=100\left(\frac{PPL_{opt}}{PPL_{FP16}}-1\right)\%.
\]

Gate считается пройденным, если для `hybrid_rns_q16` выполняется `Delta_PPL < 5%`.

## Варианты

1. `fp16` — исходная модель без замены линейных слоев.
2. `native_int8` — симметричная W8A8 fake quantization: активации масштабируются по строкам, веса — по выходным каналам.
3. `hybrid_fp16` — safe-каналы считаются W8A8, защищенные каналы остаются FP16.
4. `hybrid_rns_q16` — safe-каналы считаются W8A8, защищенная ветвь квантуется в q16. Для PPL используется идеальная реконструкция: это численно тот же квантованный dot product, который должен вернуть точная RNS+CRT реализация при отсутствии overflow.

Отдельный sampled check сравнивает q16 integer dot product и CRT-реконструкцию. Он подтверждает эквивалентность PPL-симуляции и RNS-математики, но не измеряет latency.

## Почему PPL код не использует CUDA RNS kernel

Текущий CUDA kernel является layer-level prefill executor с заранее фиксированной формой `M x K x N`. Полная Hugging Face модель вызывает десятки Linear-модулей с динамическими формами. Встраивание benchmark kernel в каждый модуль изменило бы одновременно арифметику, планирование памяти и serving runtime.

Поэтому v0.12 разделяет два вопроса:

- v0.11.3 CUDA benchmark отвечает на вопрос о скорости;
- v0.12 fake-quant full-model run отвечает на вопрос о качестве/PPL.

Такой PPL эксперимент нельзя использовать как доказательство ускорения.

## Данные и разбиение

- calibration: WikiText-2 raw validation;
- evaluation: WikiText-2 raw test;
- первая половина calibration batches строит risk map;
- вторая половина выбирает минимальный protected ratio, уменьшающий локальную ошибку не менее чем на 20%, либо лучший доступный вариант;
- evaluation split не используется для выбора protected channels.

## Режимы запуска

### Preview

Рекомендуется сначала выполнить 16–32k токенов, чтобы проверить память, отсутствие NaN и направление изменения PPL.

### Paper run

Для итоговой таблицы увеличить `MAX_EVAL_TOKENS` до 32768 или 65536 и сохранить одинаковые `context_length`, `stride`, model revision, tokenizer и target scope для всех вариантов. Полный WikiText-2 test (`MAX_EVAL_TOKENS=0`) допустим, но на Colab T4 может быть слишком долгим из-за quality-only fake quantization.

## Target scope

По умолчанию заменяются:

- `q_proj`, `k_proj`, `v_proj`, `out_proj`;
- `fc1`, `fc2`.

Это строже, чем layer-level benchmark v0.11.3, где для timing выбирались репрезентативные `q_proj` и `fc1`. Итоговая статья должна явно указать scope из JSON.

## Выходные файлы

- `ppl_calibration_plan_v012.json`;
- `ppl_summary_v012.json`;
- `paper/ppl_results_table.tex`;
- `paper/ppl_result_macros.tex`;
- `paper/ppl_results_paragraph.tex` и human-readable `.txt`.

Числа в статью разрешено переносить только после полного завершения всех заявленных вариантов.
