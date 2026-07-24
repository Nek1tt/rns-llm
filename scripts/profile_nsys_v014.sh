#!/usr/bin/env bash
set -euo pipefail

ARCH="${1:?architecture required}"
SCOPE="${2:-matrix}"
LUT="${3:-two}"
OUTDIR="${4:-reports/v0.14.2/nsys}"
mkdir -p "$OUTDIR"
TAG="${SCOPE}_${ARCH}_lut-${LUT}"
BASE="$OUTDIR/$TAG"
META="${BASE}_metadata.json"
LOG="${BASE}.log"
MODEL_ID="${MODEL_ID:-facebook/opt-2.7b}"
MATRIX_SHAPE="${MATRIX_SHAPE:-16x2560x2560}"
ATTENTION_SEQ="${ATTENTION_SEQ:-64}"
NSYS_WARMUP="${NSYS_WARMUP:-2}"
NSYS_ITERATIONS="${NSYS_ITERATIONS:-3}"

command -v nsys >/dev/null 2>&1 || { echo "nsys not found" >&2; exit 3; }
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKLOAD_SCRIPT="${PROFILE_WORKLOAD_SCRIPT:-scripts/profile_workload_v014.py}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "$PYTHON_BIN not found" >&2; exit 3; }
[[ -f "$WORKLOAD_SCRIPT" ]] || { echo "$WORKLOAD_SCRIPT not found; run from repository root" >&2; exit 3; }

NSYS_VERSION="$(nsys --version 2>&1 | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
PROFILE_HELP="$(nsys profile --help 2>&1 || true)"
EXPORT_HELP="$(nsys export --help 2>&1 || true)"
STATS_HELP="$(nsys stats --help 2>&1 || true)"

# Use only trace domains available across recent Nsight Systems releases.
TRACE_DOMAINS="cuda,nvtx,osrt"
if grep -qiE '(^|[^[:alpha:]])cublas([^[:alpha:]]|$)' <<<"$PROFILE_HELP"; then
  TRACE_DOMAINS="${TRACE_DOMAINS},cublas"
fi

PROFILE_ARGS=(
  profile
  "--trace=${TRACE_DOMAINS}"
  "--sample=none"
  "--cpuctxsw=none"
  "--capture-range=cudaProfilerApi"
  "--force-overwrite=true"
  "--output=${BASE}"
)

# New releases use --capture-range-end, old releases use --stop-on-range-end.
if grep -q -- '--capture-range-end' <<<"$PROFILE_HELP"; then
  PROFILE_ARGS+=("--capture-range-end=stop")
elif grep -q -- '--stop-on-range-end' <<<"$PROFILE_HELP"; then
  PROFILE_ARGS+=("--stop-on-range-end=true")
fi

rm -f \
  "${BASE}.nsys-rep" "${BASE}.qdrep" "${BASE}.sqlite" "${BASE}.jsonl" \
  "${BASE}_manifest.json" "$META" "$LOG"

echo "Nsight Systems: $NSYS_VERSION" | tee "$LOG"
echo "Trace domains: $TRACE_DOMAINS" | tee -a "$LOG"
printf 'Command:' | tee -a "$LOG"
printf ' %q' nsys "${PROFILE_ARGS[@]}" "$PYTHON_BIN" -u "$WORKLOAD_SCRIPT" \
  --scope "$SCOPE" --architecture "$ARCH" --lut "$LUT" \
  --warmup "$NSYS_WARMUP" --iterations "$NSYS_ITERATIONS" \
  --model "$MODEL_ID" --shape "$MATRIX_SHAPE" --seq "$ATTENTION_SEQ" \
  --metadata "$META" | tee -a "$LOG"
printf '\n' | tee -a "$LOG"

set +e
nsys "${PROFILE_ARGS[@]}" \
  "$PYTHON_BIN" -u "$WORKLOAD_SCRIPT" \
    --scope "$SCOPE" --architecture "$ARCH" --lut "$LUT" \
    --warmup "$NSYS_WARMUP" --iterations "$NSYS_ITERATIONS" \
    --model "$MODEL_ID" --shape "$MATRIX_SHAPE" --seq "$ATTENTION_SEQ" \
    --metadata "$META" \
  2>&1 | tee -a "$LOG"
PROFILE_STATUS=${PIPESTATUS[0]}
set -e
if [[ $PROFILE_STATUS -ne 0 ]]; then
  echo "nsys profile failed with code $PROFILE_STATUS; see $LOG" >&2
  exit "$PROFILE_STATUS"
fi

REP="${BASE}.nsys-rep"
if [[ ! -s "$REP" && -s "${BASE}.qdrep" ]]; then
  REP="${BASE}.qdrep"
fi
[[ -s "$REP" ]] || { echo "Missing NSYS report: ${BASE}.nsys-rep" >&2; exit 4; }
[[ -s "$META" ]] || { echo "Missing workload metadata: $META" >&2; exit 4; }

SQLITE="${BASE}.sqlite"
nsys export --type=sqlite --force-overwrite=true --output="$SQLITE" "$REP" \
  2>&1 | tee -a "$LOG"
[[ -s "$SQLITE" ]] || { echo "Missing SQLite export: $SQLITE" >&2; exit 5; }

# Current releases call this JSON-family raw export jsonlines.
JSONL_STATUS="unsupported"
if grep -qi 'jsonlines' <<<"$EXPORT_HELP"; then
  if nsys export --type=jsonlines --force-overwrite=true \
      --output="${BASE}.jsonl" "$REP" >>"$LOG" 2>&1; then
    JSONL_STATUS="ok"
  else
    JSONL_STATUS="failed"
  fi
fi

# Produce machine-readable summary reports when supported by this release.
REPORTS=(cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum nvtx_sum osrt_sum)
STATS_STATUS_JSON="${BASE}_stats_status.json"
python - "$STATS_STATUS_JSON" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({}, indent=2))
PY
for report in "${REPORTS[@]}"; do
  output="${BASE}_${report}.json"
  stderr="${BASE}_${report}.stderr"
  set +e
  nsys stats --report="$report" --format=json "$REP" >"$output" 2>"$stderr"
  status=$?
  set -e
  python - "$STATS_STATUS_JSON" "$report" "$status" "$output" "$stderr" <<'PY'
import json, sys
from pathlib import Path
status_path = Path(sys.argv[1])
report = sys.argv[2]
status = int(sys.argv[3])
output = Path(sys.argv[4])
stderr = Path(sys.argv[5])
payload=json.loads(status_path.read_text())
payload[report]={
    "status": status,
    "output": str(output),
    "stderr": str(stderr),
    "output_bytes": output.stat().st_size if output.exists() else 0,
}
status_path.write_text(json.dumps(payload, indent=2))
PY
done

python scripts/parse_nsys_sqlite_v014.py \
  "$SQLITE" --metadata "$META" \
  --output "${BASE}_sql_summary.json" \
  --queries-output "${BASE}_queries.sql"

# Save full SQLite schema without depending on the sqlite3 command-line package.
python - "$SQLITE" "${BASE}_schema.sql" "${BASE}_tables.txt" <<'PY'
import sqlite3, sys
from pathlib import Path
source, schema_out, tables_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
connection=sqlite3.connect(source)
rows=connection.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name").fetchall()
schema_out.write_text("\n\n".join((sql or "").rstrip(";") + ";" for _,_,sql in rows) + "\n")
tables=connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
tables_out.write_text("\n".join(row[0] for row in tables) + "\n")
connection.close()
PY

python - "$BASE" "$ARCH" "$SCOPE" "$LUT" "$REP" "$NSYS_VERSION" "$TRACE_DOMAINS" "$JSONL_STATUS" <<'PY'
import json, sys
from pathlib import Path
base=Path(sys.argv[1])
rep=Path(sys.argv[5])
payload={
  "version":"0.14.2",
  "architecture":sys.argv[2],
  "scope":sys.argv[3],
  "lut_policy":sys.argv[4],
  "nsys_version":sys.argv[6],
  "trace_domains":sys.argv[7],
  "jsonlines_status":sys.argv[8],
  "nsys_report":str(rep),
  "sqlite":str(base.with_suffix('.sqlite')),
  "metadata":str(base.parent/(base.name+'_metadata.json')),
  "sql_summary_json":str(base.parent/(base.name+'_sql_summary.json')),
  "queries_sql":str(base.parent/(base.name+'_queries.sql')),
  "schema_sql":str(base.parent/(base.name+'_schema.sql')),
  "tables_txt":str(base.parent/(base.name+'_tables.txt')),
  "stats_status_json":str(base.parent/(base.name+'_stats_status.json')),
  "jsonlines":str(base.with_suffix('.jsonl')),
  "log":str(base.with_suffix('.log')),
}
required=[Path(payload[k]) for k in ("nsys_report","sqlite","metadata","sql_summary_json","queries_sql","schema_sql","tables_txt")]
missing=[str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit("Missing required NSYS artifacts: " + ", ".join(missing))
(base.parent/(base.name+'_manifest.json')).write_text(json.dumps(payload,indent=2))
PY

echo "$REP"
