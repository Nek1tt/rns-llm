#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-gemm}"
STAGE="${2:-prepared}"
M="${3:-1}"
K="${4:-768}"
N="${5:-768}"
PROFILE_SET="${6:-detailed}"

OUTDIR="${OUTDIR:-reports/v0.7/nsight}"
mkdir -p "$OUTDIR"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found. Run scripts/install_nsight_colab_full.sh first." >&2
  exit 2
fi

case "$TARGET" in
  gemm)
    KERNEL_REGEX='regex:.*gemm.*|.*gemv.*'
    ;;
  epilogue)
    KERNEL_REGEX='regex:.*fused_reduce_garner_dequant_fp16_kernel.*'
    ;;
  quantize)
    KERNEL_REGEX='regex:.*quantize_encode_fp16_kernel.*'
    STAGE='e2e'
    ;;
  *)
    echo "TARGET must be gemm, epilogue or quantize" >&2
    exit 3
    ;;
esac

LABEL="V07_RNS_V07_${STAGE^^}_M${M}_K${K}_N${N}/"
TAG="rns_v07_${STAGE}_${TARGET}_M${M}_K${K}_N${N}_${PROFILE_SET}"
BASE="$OUTDIR/$TAG"
rm -f "${BASE}.ncu-rep" "${BASE}_summary.txt" "${BASE}_raw.csv" || true

COMMON=(
  --target-processes all
  --clock-control none
  --kernel-name-base demangled
  --kernel-name "$KERNEL_REGEX"
  --nvtx
  --nvtx-include "$LABEL"
  --launch-count 1
  --import-source yes
  --force-overwrite
  --export "$BASE"
)

if [[ "$PROFILE_SET" == "sol" ]]; then
  METRICS=(--set speed-of-light)
else
  AVAILABLE="$(ncu --list-sections 2>/dev/null || true)"
  WANTED=(
    LaunchStats
    Occupancy
    SpeedOfLight
    SchedulerStats
    WarpStateStats
    MemoryWorkloadAnalysis
    InstructionStats
    SourceCounters
  )
  METRICS=()
  for section in "${WANTED[@]}"; do
    if grep -q "$section" <<<"$AVAILABLE"; then
      METRICS+=(--section "$section")
    fi
  done
  if [[ ${#METRICS[@]} -eq 0 ]]; then
    METRICS=(--set detailed)
  fi
fi

set +e
ncu \
  "${COMMON[@]}" \
  "${METRICS[@]}" \
  python -u scripts/profile_v07.py \
    --backend rns_v07 \
    --stage "$STAGE" \
    --m "$M" --k "$K" --n "$N" \
    --warmup 10 \
    --repeats 1 \
    --disable-profiler-api
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "Nsight Compute failed. Hosted runtimes may block GPU performance counters (ERR_NVGPUCTRPERM)." >&2
  exit "$status"
fi

REPORT="${BASE}.ncu-rep"
if [[ ! -s "$REPORT" ]]; then
  echo "Nsight Compute completed but report is missing: $REPORT" >&2
  exit 4
fi

ncu \
  --import "$REPORT" \
  --page details \
  --print-summary per-kernel \
  --print-rule-details true \
  > "${BASE}_summary.txt" 2>&1 || true

ncu \
  --import "$REPORT" \
  --page raw \
  --csv \
  > "${BASE}_raw.csv" 2>&1 || true

echo "Nsight Compute report: $REPORT"
echo "Text summary: ${BASE}_summary.txt"
echo "Raw CSV: ${BASE}_raw.csv"
