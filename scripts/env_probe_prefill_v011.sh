#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/v0.11/environment.txt}"
mkdir -p "$(dirname "$OUT")"
{
  echo "generated=$(date -Is)"
  echo "pwd=$(pwd)"
  echo "uname=$(uname -a)"
  echo; echo "=== NVIDIA-SMI ==="; nvidia-smi || true
  echo; echo "=== NVIDIA-SMI QUERY ==="; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.memory --format=csv || true
  echo; echo "=== NVCC ==="; nvcc --version || true
  echo; echo "=== PYTHON ==="; python --version
  echo; echo "=== PYTORCH ==="; python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print('device', p.name)
    print('cc', p.major, p.minor)
    print('sms', p.multi_processor_count)
    print('total_memory', p.total_memory)
PY
  echo; echo "=== NSIGHT SYSTEMS ==="; nsys --version || true
  echo; echo "=== NSIGHT COMPUTE ==="; ncu --version || true
  echo; echo "=== PIP ==="; python -m pip freeze | grep -E 'torch|transformers|datasets|accelerate|ninja|setuptools' || true
  echo; echo "=== GIT ==="; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
} > "$OUT" 2>&1
cat "$OUT"
