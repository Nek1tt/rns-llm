#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-/content/rns_v07_1_nsight_results.zip}"
mkdir -p results/v0.7 reports/v0.7/nsight

python - <<'PYMETA'
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import torch


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return repr(exc)

payload = {
    "timestamp": command(["date", "-Is"]),
    "version": Path("VERSION").read_text().strip(),
    "scope": "frozen_v07_performance_and_nsight_without_correctness_tests",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    "nvidia_smi": command([
        "nvidia-smi",
        "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv",
    ]),
    "git_commit": command(["git", "rev-parse", "HEAD"]),
    "git_status": command(["git", "status", "--short"]),
    "nsys": command(["nsys", "--version"]),
    "ncu": command(["ncu", "--version"]),
}
Path("reports/v0.7/nsight/collection_metadata.json").write_text(json.dumps(payload, indent=2))
PYMETA

rm -f "$OUT"
zip -r "$OUT" \
  results/v0.7 \
  reports/v0.7 \
  VERSION \
  docs/NSIGHT_GUIDE_V07_RU.md \
  docs/V08_V09_RETROSPECTIVE_RU.md \
  benchmarks/benchmark_v07_epilogue.py \
  scripts/profile_v07.py \
  scripts/profile_nsys_v07.sh \
  scripts/profile_ncu_v07.sh \
  scripts/run_nsight_matrix_v07.py \
  csrc/v07_extension.cu \
  src/rns_llm/v07_backend.py \
  src/rns_llm/layers/rns_linear_v07.py \
  >/dev/null

echo "$OUT"
