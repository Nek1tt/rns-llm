# Benchmarks

Every benchmark must record:

```text
hardware
software version
matrix/model shape
dtype
warmups
measured runs
synchronization method
p50 latency
p95 latency
peak memory if available
```

GPU timing must synchronize before stopping the timer.

Store outputs in `results/`.
