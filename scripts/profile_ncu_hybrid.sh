#!/usr/bin/env bash
set -euo pipefail
PACK="${1:?pack path required}"
METHOD="${2:-hybrid_int8_plus_rns_q16}"
STAGE="${3:-prepared}"
M="${4:-1}"
TARGET="${5:-all}"
OUTDIR="${OUTDIR:-reports/v0.10/nsight}"
mkdir -p "$OUTDIR"
command -v ncu >/dev/null || { echo "ncu unavailable" >&2; exit 2; }
PACK="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PACK")"
case "$TARGET" in
  all) REGEX='regex:.*' ;;
  gemm) REGEX='regex:.*gemm.*|.*gemv.*' ;;
  encode) REGEX='regex:.*encode_.*|.*quantize_.*' ;;
  epilogue) REGEX='regex:.*dequant.*|.*garner.*' ;;
  *) echo "target: all|gemm|encode|epilogue" >&2; exit 3 ;;
esac
LABEL="HYBRID_PROFILE_${METHOD^^}_${STAGE^^}_M${M}/"
TAG="${METHOD}_${STAGE}_M${M}_${TARGET}"
BASE="$OUTDIR/$TAG"
rm -f "${BASE}.ncu-rep" "${BASE}_summary.txt" "${BASE}_raw.csv" || true
AVAILABLE="$(ncu --list-sections 2>/dev/null || true)"
SECTIONS=()
for S in LaunchStats Occupancy SpeedOfLight SchedulerStats WarpStateStats MemoryWorkloadAnalysis InstructionStats SourceCounters; do
  grep -q "$S" <<<"$AVAILABLE" && SECTIONS+=(--section "$S")
done
[[ ${#SECTIONS[@]} -gt 0 ]] || SECTIONS=(--set detailed)
set +e
ncu --target-processes all --clock-control none --kernel-name-base demangled \
  --kernel-name "$REGEX" --nvtx --nvtx-include "$LABEL" --launch-count 1 \
  --import-source yes --force-overwrite --export "$BASE" "${SECTIONS[@]}" \
  python -u scripts/profile_hybrid.py --pack "$PACK" --method "$METHOD" --stage "$STAGE" \
    --m "$M" --warmup 5 --repeats 1 --disable-profiler-api
STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
  echo "NCU failed; hosted runtimes may block counters (ERR_NVGPUCTRPERM)." >&2
  exit "$STATUS"
fi
REPORT="${BASE}.ncu-rep"
[[ -s "$REPORT" ]] || { echo "NCU report missing" >&2; exit 4; }
ncu --import "$REPORT" --page details --print-summary per-kernel --print-rule-details true > "${BASE}_summary.txt" 2>&1 || true
ncu --import "$REPORT" --page raw --csv > "${BASE}_raw.csv" 2>&1 || true
echo "$REPORT"
