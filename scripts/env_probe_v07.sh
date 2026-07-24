#!/usr/bin/env bash
set -uo pipefail

OUTDIR="${OUTDIR:-reports/v0.7/nsight}"
mkdir -p "$OUTDIR"
OUT="${1:-$OUTDIR/environment.txt}"

{
  echo "=== timestamp ==="
  date -Is

  echo "=== uname ==="
  uname -a || true

  echo "=== os-release ==="
  cat /etc/os-release || true

  echo "=== nvidia-smi ==="
  nvidia-smi || true

  echo "=== GPU telemetry ==="
  nvidia-smi \
    --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used \
    --format=csv || true

  echo "=== nvcc ==="
  command -v nvcc || true
  nvcc --version || true

  echo "=== nsys ==="
  command -v nsys || true
  readlink -f "$(command -v nsys)" || true
  nsys --version || true

  echo "=== Nsight Systems GPU metrics support ==="
  nsys profile --gpu-metrics-devices=help 2>&1 || \
    nsys profile --gpu-metrics-device=help 2>&1 || true

  echo "=== ncu ==="
  command -v ncu || true
  readlink -f "$(command -v ncu)" || true
  ncu --version || true

  echo "=== git ==="
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true

  echo "=== Python / PyTorch ==="
  python - <<'PY'
import json
import platform
import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    payload.update(
        {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "device_count": torch.cuda.device_count(),
        }
    )
print(json.dumps(payload, indent=2))
PY
} 2>&1 | tee "$OUT"
