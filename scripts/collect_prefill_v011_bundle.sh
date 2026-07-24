#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
OUT="${2:-rns_hybrid_v011_prefill_results.zip}"
cd "$ROOT"
rm -f "$OUT"
python - <<'PY' "$OUT"
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import sys
out=Path(sys.argv[1])
patterns=[
    'results/v0.11/*.json',
    'results/v0.11/*.txt',
    'results/v0.11/*.log',
    'reports/v0.11/nsight/*',
    'src/rns_llm/prefill_v011.py',
    'csrc/v011_prefill_extension.cu',
    'benchmarks/benchmark_prefill_v011.py',
    'scripts/audit_model_prefill_v011.py',
    'scripts/profile_prefill_v011.py',
    'scripts/profile_nsys_prefill_v011.sh',
    'scripts/profile_ncu_prefill_v011.sh',
    'docs/PREFILL_V011_ARCHITECTURE_RU.md',
    'docs/PREFILL_V011_EXPERIMENT_RU.md',
    'VERSION',
]
files=[]
for pattern in patterns:
    files.extend(Path('.').glob(pattern))
files=sorted({p for p in files if p.is_file() and 'packs' not in p.parts})
with ZipFile(out, 'w', ZIP_DEFLATED, compresslevel=6) as z:
    for path in files:
        z.write(path, path.as_posix())
print(out, out.stat().st_size, 'bytes', len(files), 'files')
PY
