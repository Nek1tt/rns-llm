---
name: CUDA backend task
about: GPU implementation or benchmark
title: "[CUDA] "
labels: cuda
---

## Goal

Implement or benchmark one GPU-side RNS component.

## Files

`src/rns_llm/backends/cuda_backend.py`, `cuda/`, `benchmarks/`

## Definition of done

- [ ] Input/output shapes are explicit.
- [ ] Output is checked against NumPy reference.
- [ ] Benchmark command is reproducible.
- [ ] GPU timing uses synchronization.
