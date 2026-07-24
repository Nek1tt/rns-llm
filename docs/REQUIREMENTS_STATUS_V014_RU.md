# Покрытие исходного ТЗ в v0.14.2

| Требование | Код/эксперимент |
|---|---|
| Moduli <=8 bit | full-RNS и hybrid planners валидируют moduli <=255 |
| Выбор moduli | large-primes и school-small; полный accumulator bound |
| Speed/memory tradeoff | unified matrix JSON: channels, latency, weights, LUT, workspace |
| RNS matrix multiplication | full-RNS CUDA residue GEMM и hybrid protected correction |
| Self-Attention | fused QKV + replaced out_proj, полная module latency |
| Non-modular operations | native QK, mask, Softmax, AV включены в Attention timing |
| LUT reuse | actual 0/1/2/all allocation и shared tensor across streams |
| >50% LUT saving | вычисляется автоматически относительно all-LUT |
| Shared-table contention | четыре streams для matrix и Attention |
| Strict latency | сравнение с FP16/native INT8; результат определяется запуском |
| 4 concurrent requests | four-stream proxy, throughput и contention metrics |
| PPL <5% | actual-kernel WikiText-2 gate |
| Nsight Systems | `.nsys-rep`, `.sqlite`, SQL, JSON |
| Nsight Compute | `.ncu-rep`, article-essential raw/details JSON; exhaustive full mode optional |

После запуска `scripts/summarize_v014.py` формирует фактический статус без предположений о недостающих файлах.
