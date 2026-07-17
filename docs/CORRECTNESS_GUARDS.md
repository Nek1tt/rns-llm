# Correctness and Anti-Cheating Guards

## Independent oracle

CUDA benchmarks compare the entire output matrix with:

```python
A.cpu().numpy().astype(np.int64) @ B.cpu().numpy().astype(np.int64)
```

This oracle does not call the CUDA extension or reuse its CRT/Garner code.

## QKV fusion

The fused QKV result must be bit-for-bit equal to three independent RNSLinear
calls (`rtol=0`, `atol=0`). Therefore performance differences cannot come from
changing quantization, scales or arithmetic.

## Adaptive channels

A reduced channel prefix is legal only after a proved upper bound fits the
centered RNS capacity. If no prefix fits, the code raises/falls back to the full
set; modular wraparound is never accepted as an approximation.

## Continuous batching

Merged outputs are compared both with independent per-request RNS executions
and with the NumPy int64 oracle. Queue delay is not included and is labelled as
such; throughput is not presented as end-user latency.

## PPL audit

The PPL script prints:

- exact module names replaced;
- fused RNS GEMM call count;
- QKV fused compute count;
- baseline and RNS PPL.

It refuses to report if the backend was installed but not executed.

## Known limits

- GPU code must still be compiled and executed on the target NVIDIA device.
- PPL on a small WikiText subset is a smoke test, not a final quality claim.
- Adaptive runtime selection currently introduces a synchronization point.
- Softmax remains the model's original floating-point implementation.

## Floating-point QKV reference note

The `mode="torch"` QKV unit test uses a very small floating-point tolerance instead of bit-for-bit equality. Concatenating Q/K/V weights changes the GEMM output shape and may change floating-point accumulation order by about one unit in the last place. This is not an RNS arithmetic error.

The CUDA RNS benchmark keeps the stronger requirement for quantized integer arithmetic:

```text
fused RNS QKV integer output == three separate RNS outputs
max_absolute_error = 0
```
