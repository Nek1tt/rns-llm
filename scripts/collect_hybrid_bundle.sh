#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-/content/rns_hybrid_v010_1_results.zip}"
mkdir -p results/v0.10 reports/v0.10/nsight
python - <<'PY'
from __future__ import annotations
import json, platform, subprocess
from pathlib import Path
import torch

def cmd(args):
    try: return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc: return repr(exc)

payload={
  "timestamp":cmd(["date","-Is"]),
  "version":Path("VERSION").read_text().strip() if Path("VERSION").exists() else None,
  "python":platform.python_version(),
  "torch":torch.__version__,
  "torch_cuda":torch.version.cuda,
  "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
  "capability":list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
  "nvidia_smi":cmd(["nvidia-smi","--query-gpu=name,driver_version,pstate,power.draw,clocks.sm,clocks.mem,memory.total,memory.used","--format=csv"]),
  "git_commit":cmd(["git","rev-parse","HEAD"]),
  "git_status":cmd(["git","status","--short"]),
  "nsys":cmd(["nsys","--version"]),
  "ncu":cmd(["ncu","--version"]),
}
Path("reports/v0.10/nsight/collection_metadata.json").write_text(json.dumps(payload,indent=2))
PY
rm -f "$OUT"
FILES=(
  reports/v0.10
  VERSION
  docs/HYBRID_ARCHITECTURE_RU.md
  docs/HYBRID_EXPERIMENT_PROTOCOL_RU.md
  docs/V08_V09_RETROSPECTIVE_RU.md
  benchmarks/benchmark_hybrid_from_audit.py
  scripts/audit_model_hybrid.py
  scripts/summarize_hybrid.py
  scripts/profile_hybrid.py
  scripts/profile_nsys_hybrid.sh
  scripts/profile_ncu_hybrid.sh
  scripts/run_nsight_hybrid_matrix.py
  csrc/hybrid_rns_extension.cu
  src/rns_llm/hybrid_v010.py
)
for CANDIDATE in results/v0.10/model_audit.json results/v0.10/hybrid_benchmark.json results/v0.10/summary.txt; do
  [[ -e "$CANDIDATE" ]] && FILES+=("$CANDIDATE")
done
zip -r "$OUT" "${FILES[@]}" >/dev/null
echo "$OUT"
