#!/usr/bin/env bash
set -u

echo "=== OS ==="
uname -a || true

echo
echo "=== NVIDIA ==="
nvidia-smi || true

echo
echo "=== CUDA compiler ==="
nvcc --version || true

echo
echo "=== C/C++ tools ==="
gcc --version | head -1 || true
g++ --version | head -1 || true
cmake --version | head -1 || true
ninja --version || true

echo
echo "=== Python / PyTorch ==="
python - <<'PY'
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name())
        print("capability:", torch.cuda.get_device_capability())
except Exception as exc:
    print("PyTorch check failed:", exc)
PY
