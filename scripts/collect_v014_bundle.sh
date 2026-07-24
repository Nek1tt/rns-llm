#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
OUT="${2:-rns_llm_v0142_results.zip}"
cd "$ROOT"
if [[ "$OUT" = /* ]]; then
  OUT_ABS="$OUT"
else
  OUT_ABS="$(pwd)/$OUT"
fi
mkdir -p "$(dirname "$OUT_ABS")"
rm -f "$OUT_ABS"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/v0.14.2/results" "$TMP/v0.14.2/reports"
if [[ -d results/v0.14.2 ]]; then
  cp -a results/v0.14.2/. "$TMP/v0.14.2/results/"
fi
if [[ -d reports/v0.14.2 ]]; then
  cp -a reports/v0.14.2/. "$TMP/v0.14.2/reports/"
fi
python - <<'PY' > "$TMP/v0.14.2/environment.json"
import json, platform, subprocess, torch

def cmd(x):
    try: return subprocess.check_output(x, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return repr(e)
print(json.dumps({
  'python':platform.python_version(), 'platform':platform.platform(),
  'torch':torch.__version__, 'cuda':torch.version.cuda,
  'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'ncu':cmd(['ncu','--version']), 'nsys':cmd(['nsys','--version']),
},indent=2))
PY
python - "$TMP/v0.14.2" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = []
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.name == 'SHA256SUMS.txt':
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
(root / 'SHA256SUMS.txt').write_text("\n".join(lines) + "\n")
PY
(cd "$TMP" && zip -qr "$OUT_ABS" v0.14.2)
[[ -s "$OUT_ABS" ]] || { echo "Result archive was not created: $OUT_ABS" >&2; exit 4; }
echo "$OUT_ABS"
