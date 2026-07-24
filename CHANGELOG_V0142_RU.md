# v0.14.2

Исправления относительно v0.14.1:

- исправлена совместимость Nsight Systems: current `--capture-range-end=stop` и legacy fallback;
- удалено молчаливое подавление ошибок профайлера;
- добавлены live logs и проверка обязательных артефактов до создания manifest;
- стандартный Nsight protocol сокращён до пяти репрезентативных профилей с общим лимитом 55 минут;
- Nsight Compute по умолчанию использует article-essential sections и максимум четыре kernel launches; exhaustive `--set full` оставлен опциональным;
- исправлена установка Nsight в Colab и добавлен реальный CUDA probe;
- PPL запускается с основной policy `two`, поскольку LUT policies математически эквивалентны и отдельно измеряются во всех latency/accuracy benchmarks;
- notebook проверяет SHA-256 release manifest до применения overlay.
- исправлен сборщик итогового результата: абсолютный output path из Colab теперь поддерживается, архив не получает двойную вложенность `v0.14.2/v0.14.2`;
- повторно проверены modern и legacy Nsight Systems CLI branches, Nsight Compute raw/details export и manifest validation с mock profiler runs;
- удалены build caches и `egg-info` из release overlay.
