#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-results/v0.13}"
ARCHIVE="${2:-rns_architecture_v013_results.zip}"
if [[ ! -d "$OUT_DIR" ]]; then
  echo "Missing output directory: $OUT_DIR" >&2
  exit 2
fi
python - <<'PY' "$OUT_DIR" "$ARCHIVE"
from pathlib import Path
import sys, zipfile
root = Path(sys.argv[1]).resolve()
archive = Path(sys.argv[2]).resolve()
files = [p for p in root.rglob('*') if p.is_file()]
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(files):
        zf.write(path, Path(root.name) / path.relative_to(root))
print(archive)
print('files:', len(files))
PY
