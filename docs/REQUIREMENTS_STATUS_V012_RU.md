# Проверка исходных требований проекта после v0.12.0

Статусы означают: **DONE** — реализовано и измерено; **PARTIAL** — реализована только часть; **REJECTED** — целевой показатель проверен и не достигнут; **PENDING RUN** — код есть, фактический результат должен быть получен на GPU.

| Требование | Статус | Фактическое состояние |
|---|---|---|
| 1. Выбор взаимно простых модулей, каждый не более 8 бит | DONE | Используется пул попарно взаимно простых модулей 199–255. Число каналов выбирается по диапазону аккумулятора `2*K*qmax^2+1`, а не только по разрядности отдельного операнда. |
| 1. Оптимальный баланс числа модулей и памяти | PARTIAL / NEGATIVE RESULT | Реализован автоматический выбор минимального числа модулей, покрывающего диапазон. На T4 рост числа каналов увеличивает работу и не дает бесплатного параллелизма. |
| 2. RNS matrix multiplication для Transformer | PARTIAL | Реализованы full-RNS и hybrid CUDA kernels, DP4A, Garner reconstruction, prefill формы `q_proj/fc1`. Это не полноценная замена всех Linear в serving runtime. |
| 2. Распределение модулей по GPU cores/streams | DONE, target rejected | Реализованы serial и dual-stream варианты. По v0.11.3 parallel execution не ускорило end-to-end path. |
| 3. Non-modular operations | PARTIAL | Реализованы encoding, dequantization, sign/CRT reconstruction, bias/merge epilogue. Softmax, LayerNorm, GELU/SwiGLU и division не перенесены в RNS. |
| 3. Overhead не должен уничтожать speed gain | REJECTED | Garner/modular correction и дополнительная работа уничтожают выигрыш; speed gates v0.11.3 не пройдены. |
| 4. Reuse lookup tables, только 1–2 крупнейшие таблицы | NOT IMPLEMENTED AS STATED | Финальная архитектура использует константы Garner, а не большие lookup tables. Поэтому требование reuse tables неприменимо к выбранной реализации и должно быть описано как design change. |
| 4. Более 50% memory efficiency | ALMOST, BUT FORMALLY REJECTED | Hybrid q16 использует около 50.55% от FP16 footprint, то есть экономия около 49.45%, чуть меньше строгого >50% target. |
| 4. Проверка data-bus contention shared tables | NOT APPLICABLE / NOT MEASURED | Большие shared lookup tables отсутствуют. Конкуренция main/correction streams измерена и оказалась неблагоприятной, но это не отдельный shared-table test. |
| 5. Strict latency / >30% speedup | REJECTED | На 32 формах ни один RNS вариант не обогнал FP16; median best hybrid RNS e2e около 2.19x FP16 latency. |
| 5. Четыре concurrent connections на одной GPU | NOT IMPLEMENTED | Layer-level single-request gate не пройден; полноценный serving benchmark отсутствует. |
| 5. Рост PPL менее 5% | PENDING RUN | v0.12.0 добавляет full-model WikiText-2 protocol и автоматический PASS/FAIL. Статус станет DONE/REJECTED после Colab run. |

## Честная формулировка для статьи

Проект не выполнил исходную цель ускорения. Он реализовал и проверил архитектурную гипотезу, сохранил точность на layer level и выявил аппаратную границу применимости RNS на NVIDIA T4. PPL и concurrency нельзя объявлять пройденными до появления соответствующих результатов.
