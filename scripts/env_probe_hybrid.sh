#!/usr/bin/env bash
set -uo pipefail
OUTDIR="${OUTDIR:-reports/v0.10/nsight}"
mkdir -p "$OUTDIR"
OUT="${1:-$OUTDIR/environment.txt}"
{
  echo "=== timestamp ==="; date -Is
  echo "=== uname ==="; uname -a || true
  echo "=== os-release ==="; cat /etc/os-release || true
  echo "=== nvidia-smi ==="; nvidia-smi || true
  echo "=== telemetry ==="
  nvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used --format=csv || true
  echo "=== nvcc ==="; command -v nvcc || true; nvcc --version || true
  echo "=== nsys ==="; command -v nsys || true; nsys --version || true
  echo "=== ncu ==="; command -v ncu || true; ncu --version || true
  echo "=== git ==="; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true
  echo "=== python/torch/extensions ==="
  python - <<'PY'
import json, platform, torch
payload={"python":platform.python_version(),"torch":torch.__version__,"torch_cuda":torch.version.cuda,"cuda":torch.cuda.is_available()}
if torch.cuda.is_available():
    payload.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
for name in ("rns_llm._C","rns_llm._V07","rns_llm._HYBRID"):
    try:
        __import__(name)
        payload[name]="ok"
    except Exception as exc:
        payload[name]=repr(exc)
print(json.dumps(payload,indent=2))
PY
} 2>&1 | tee "$OUT"
