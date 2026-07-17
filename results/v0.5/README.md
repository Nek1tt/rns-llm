# Recorded v0.5 pre-results

This directory contains the measured outputs supplied from the successful Tesla T4 / Colab run.

Files:

- `qkv_fusion.json` — fused QKV versus three separate RNS projections;
- `concurrency_v05_m1.json` — 1/2/4 requests with one row per request;
- `concurrency_v05_m128.json` — 1/2/4 requests with 128 rows per request;
- `adaptive_channels.json` — safe channel-prefix selection including synchronization cost;
- `ppl_attention.json` — one-block and four-block OPT PPL runs.

All arithmetic correctness fields in the supplied CUDA benchmark JSON files passed with zero maximum absolute error.

These are pre-results, not final FP16/native-INT8/RNS end-to-end comparisons.
