#!/usr/bin/env bash
set -euo pipefail
AUDIT="${1:?audit json required}"
LAYER_INDEX="${2:-0}"
TARGET="${3:-main_int8}"
M="${4:-128}"
OUTDIR="${OUTDIR:-reports/v0.11/nsight}"
mkdir -p "$OUTDIR"
AUDIT="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$AUDIT")"
command -v ncu >/dev/null || { echo "ncu unavailable" >&2; exit 2; }
case "$TARGET" in
  fp16) METHOD=fp16; RANGE=V011_FP16_GEMM ;;
  native_int8) METHOD=native_int8; RANGE=V011_NATIVE_INT8_MAIN ;;
  preprocess) METHOD=preprocess; RANGE=V011_FUSED_PREPROCESS ;;
  main_int8) METHOD=main_int8; RANGE=V011_MAIN_INT8 ;;
  rns_correction) METHOD=rns_correction; RANGE=V011_RNS_CORRECTION ;;
  fp16_correction) METHOD=fp16_correction; RANGE=V011_FP16_CORRECTION ;;
  rns_fused_epilogue) METHOD=rns_fused_epilogue; RANGE=V011_RNS_FUSED_EPILOGUE ;;
  *) echo "target: fp16|native_int8|preprocess|main_int8|rns_correction|fp16_correction|rns_fused_epilogue" >&2; exit 3 ;;
esac
TAG="L${LAYER_INDEX}_${TARGET}_M${M}_detailed"
BASE="$OUTDIR/$TAG"
rm -f "${BASE}.ncu-rep" "${BASE}_summary.txt" "${BASE}_raw.csv" "${BASE}_metrics.csv" || true
AVAILABLE="$(ncu --list-sections 2>/dev/null || true)"
SECTIONS=()
for S in LaunchStats Occupancy SpeedOfLight SchedulerStats WarpStateStats MemoryWorkloadAnalysis InstructionStats SourceCounters; do
  grep -q "$S" <<<"$AVAILABLE" && SECTIONS+=(--section "$S")
done
[[ ${#SECTIONS[@]} -gt 0 ]] || SECTIONS=(--set full)
set +e
ncu \
  --target-processes all \
  --clock-control none \
  --kernel-name-base demangled \
  --nvtx --nvtx-include "${RANGE}/" \
  --launch-count 1 \
  --import-source yes \
  --force-overwrite \
  --export "$BASE" \
  "${SECTIONS[@]}" \
  python -u scripts/profile_prefill_v011.py \
    --audit "$AUDIT" --layer-index "$LAYER_INDEX" --m "$M" \
    --method "$METHOD" --prepared --warmup 5 --repeats 1
STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
  echo "Nsight Compute failed. Hosted Colab can deny hardware counters (ERR_NVGPUCTRPERM)." >&2
  exit "$STATUS"
fi
REPORT="${BASE}.ncu-rep"
[[ -s "$REPORT" ]] || { echo "NCU report missing" >&2; exit 4; }
ncu --import "$REPORT" --page details --print-summary per-kernel --print-rule-details true \
  > "${BASE}_summary.txt" 2>&1 || true
ncu --import "$REPORT" --page raw --csv > "${BASE}_raw.csv" 2>&1 || true
ncu --import "$REPORT" --page details --csv > "${BASE}_metrics.csv" 2>&1 || true
echo "$REPORT"
