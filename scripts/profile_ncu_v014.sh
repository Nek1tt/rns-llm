#!/usr/bin/env bash
set -euo pipefail

ARCH="${1:?architecture required}"
SCOPE="${2:-matrix}"
LUT="${3:-two}"
OUTDIR="${4:-reports/v0.14.2/ncu}"
mkdir -p "$OUTDIR"
TAG="${SCOPE}_${ARCH}_lut-${LUT}"
BASE="$OUTDIR/$TAG"
META="${BASE}_metadata.json"
LOG="${BASE}.log"
MODEL_ID="${MODEL_ID:-facebook/opt-2.7b}"
MATRIX_SHAPE="${MATRIX_SHAPE:-16x2560x2560}"
ATTENTION_SEQ="${ATTENTION_SEQ:-64}"
NCU_MODE="${NCU_MODE:-essential}"          # essential | full
NCU_MAX_LAUNCHES="${NCU_MAX_LAUNCHES:-4}"
NCU_WARMUP="${NCU_WARMUP:-2}"
NCU_ITERATIONS="${NCU_ITERATIONS:-1}"

command -v ncu >/dev/null 2>&1 || { echo "ncu not found" >&2; exit 3; }
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKLOAD_SCRIPT="${PROFILE_WORKLOAD_SCRIPT:-scripts/profile_workload_v014.py}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "$PYTHON_BIN not found" >&2; exit 3; }
[[ -f "$WORKLOAD_SCRIPT" ]] || { echo "$WORKLOAD_SCRIPT not found; run from repository root" >&2; exit 3; }

NCU_VERSION="$(ncu --version 2>&1 | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
NCU_HELP="$(ncu --help 2>&1 || true)"

case "$ARCH" in
  full_rns_int8|full_rns_int16|full_rns_int32)
    KERNEL_REGEX='quantize_encode_rows_kernel|rns_reduce_garner_dequant_kernel'
    ;;
  full_rns_int8_v07)
    KERNEL_REGEX='quantize|encode|rns|garner|epilogue'
    ;;
  hybrid_rns_q8*|hybrid_rns_q16*|hybrid_rns_q32*)
    KERNEL_REGEX='fused_hybrid_preprocess_kernel|rns_fused_epilogue_kernel|rns_rankk_correction_kernel'
    ;;
  hybrid_fp16*)
    KERNEL_REGEX='fused_hybrid_preprocess_fp16_kernel|fp16_fused_epilogue_kernel|fp16_rankk_correction_kernel'
    ;;
  native_int8)
    KERNEL_REGEX='quantize_rows_int8_kernel|dequant_int32_fp32_kernel'
    ;;
  fp16)
    # Vendor GEMM names vary by CUDA/cuBLAS release. No restrictive filter.
    KERNEL_REGEX=''
    ;;
  *)
    KERNEL_REGEX='rns|hybrid|quantize|epilogue|correction|garner'
    ;;
esac

# A metadata-only warmup catches model/download/build errors before NCU starts.
"$PYTHON_BIN" -u "$WORKLOAD_SCRIPT" \
  --scope "$SCOPE" --architecture "$ARCH" --lut "$LUT" \
  --warmup 1 --iterations 1 --model "$MODEL_ID" \
  --shape "$MATRIX_SHAPE" --seq "$ATTENTION_SEQ" --metadata "$META"
[[ -s "$META" ]] || { echo "Missing workload metadata after preflight: $META" >&2; exit 4; }

PROFILE_ARGS=(
  "--replay-mode=kernel"
  "--target-processes=all"
  "--profile-from-start=off"
  "--kernel-name-base=demangled"
  "--force-overwrite"
  "--export=${BASE}"
  "--launch-count=${NCU_MAX_LAUNCHES}"
)

if [[ -n "$KERNEL_REGEX" ]]; then
  if grep -q -- '--kernel-name' <<<"$NCU_HELP"; then
    PROFILE_ARGS+=("--kernel-name=regex:${KERNEL_REGEX}")
  elif grep -q -- '--kernel-regex' <<<"$NCU_HELP"; then
    PROFILE_ARGS+=("--kernel-regex=${KERNEL_REGEX}")
  fi
fi

SECTION_IDS=(
  SpeedOfLight
  LaunchStats
  Occupancy
  MemoryWorkloadAnalysis
  ComputeWorkloadAnalysis
  InstructionStats
  SchedulerStats
  WarpStateStats
)
COLLECTED_SECTIONS=()
if [[ "$NCU_MODE" == "full" ]]; then
  PROFILE_ARGS+=("--set=full")
elif [[ "$NCU_MODE" == "essential" ]]; then
  LIST_SECTIONS="$(ncu --list-sections 2>&1 || true)"
  for section in "${SECTION_IDS[@]}"; do
    if grep -qE "(^|[[:space:]])${section}([[:space:]]|$)" <<<"$LIST_SECTIONS"; then
      PROFILE_ARGS+=("--section=${section}")
      COLLECTED_SECTIONS+=("$section")
    fi
  done
  if [[ ${#COLLECTED_SECTIONS[@]} -eq 0 ]]; then
    PROFILE_ARGS+=("--set=basic")
    COLLECTED_SECTIONS=(basic)
  fi
else
  echo "NCU_MODE must be essential or full" >&2
  exit 4
fi

rm -f \
  "${BASE}.ncu-rep" "${BASE}_raw.csv" "${BASE}_raw_full.json" \
  "${BASE}_details.txt" "${BASE}_details.csv" "${BASE}_details_full.json" \
  "${BASE}_manifest.json" "$LOG"

echo "Nsight Compute: $NCU_VERSION" | tee "$LOG"
echo "Mode: $NCU_MODE" | tee -a "$LOG"
echo "Kernel regex: ${KERNEL_REGEX:-<none>}" | tee -a "$LOG"
echo "Sections: ${COLLECTED_SECTIONS[*]}" | tee -a "$LOG"
printf 'Command:' | tee -a "$LOG"
printf ' %q' ncu "${PROFILE_ARGS[@]}" "$PYTHON_BIN" -u "$WORKLOAD_SCRIPT" \
  --scope "$SCOPE" --architecture "$ARCH" --lut "$LUT" \
  --warmup "$NCU_WARMUP" --iterations "$NCU_ITERATIONS" \
  --model "$MODEL_ID" --shape "$MATRIX_SHAPE" --seq "$ATTENTION_SEQ" \
  --metadata "$META" | tee -a "$LOG"
printf '\n' | tee -a "$LOG"

set +e
ncu "${PROFILE_ARGS[@]}" \
  "$PYTHON_BIN" -u "$WORKLOAD_SCRIPT" \
    --scope "$SCOPE" --architecture "$ARCH" --lut "$LUT" \
    --warmup "$NCU_WARMUP" --iterations "$NCU_ITERATIONS" \
    --model "$MODEL_ID" --shape "$MATRIX_SHAPE" --seq "$ATTENTION_SEQ" \
    --metadata "$META" \
  2>&1 | tee -a "$LOG"
PROFILE_STATUS=${PIPESTATUS[0]}
set -e
if [[ $PROFILE_STATUS -ne 0 ]]; then
  echo "ncu failed with code $PROFILE_STATUS; see $LOG" >&2
  if grep -qiE 'ERR_NVGPUCTRPERM|permission.*performance counters' "$LOG"; then
    echo "GPU performance counters are not available in this runtime." >&2
  fi
  exit "$PROFILE_STATUS"
fi

REP="${BASE}.ncu-rep"
[[ -s "$REP" ]] || { echo "Missing NCU report: $REP" >&2; exit 5; }

ncu --import "$REP" --page raw --csv > "${BASE}_raw.csv"
python scripts/parse_ncu_csv_v014.py \
  "${BASE}_raw.csv" --page raw --metadata "$META" \
  --output "${BASE}_raw_full.json"

# Details text is always useful; CSV availability varies by NCU release.
ncu --import "$REP" --page details > "${BASE}_details.txt"
DETAILS_JSON_STATUS="unsupported"
if ncu --import "$REP" --page details --csv > "${BASE}_details.csv" 2>"${BASE}_details_csv.stderr"; then
  python scripts/parse_ncu_csv_v014.py \
    "${BASE}_details.csv" --page details --metadata "$META" \
    --output "${BASE}_details_full.json"
  DETAILS_JSON_STATUS="ok"
fi

python - "$BASE" "$ARCH" "$SCOPE" "$LUT" "$NCU_VERSION" "$NCU_MODE" "$KERNEL_REGEX" "$DETAILS_JSON_STATUS" "${COLLECTED_SECTIONS[*]}" <<'PY'
import json, sys
from pathlib import Path
base=Path(sys.argv[1])
raw_json=base.parent/(base.name+'_raw_full.json')
payload_raw=json.loads(raw_json.read_text())
if int(payload_raw.get('metric_row_count', 0)) <= 0:
    raise SystemExit('NCU report contains no metric rows; kernel filter likely matched nothing')
payload={
  "version":"0.14.2",
  "architecture":sys.argv[2],
  "scope":sys.argv[3],
  "lut_policy":sys.argv[4],
  "ncu_version":sys.argv[5],
  "collection_mode":sys.argv[6],
  "kernel_regex":sys.argv[7],
  "details_json_status":sys.argv[8],
  "sections":sys.argv[9].split(),
  "ncu_report":str(base.with_suffix('.ncu-rep')),
  "metadata":str(base.parent/(base.name+'_metadata.json')),
  "raw_csv":str(base.parent/(base.name+'_raw.csv')),
  "raw_json":str(raw_json),
  "details_text":str(base.parent/(base.name+'_details.txt')),
  "details_csv":str(base.parent/(base.name+'_details.csv')),
  "details_json":str(base.parent/(base.name+'_details_full.json')),
  "log":str(base.with_suffix('.log')),
  "metric_row_count":int(payload_raw.get('metric_row_count', 0)),
}
required=[Path(payload[k]) for k in ("ncu_report","metadata","raw_csv","raw_json","details_text")]
missing=[str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit('Missing required NCU artifacts: ' + ', '.join(missing))
(base.parent/(base.name+'_manifest.json')).write_text(json.dumps(payload,indent=2))
PY

echo "$REP"
