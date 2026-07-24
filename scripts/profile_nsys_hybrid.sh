#!/usr/bin/env bash
set -euo pipefail
PACK="${1:?pack path required}"
METHOD="${2:-hybrid_int8_plus_rns_q16}"
STAGE="${3:-e2e}"
M="${4:-1}"
REPEATS="${5:-100}"
WARMUP="${WARMUP:-10}"
OUTDIR="${OUTDIR:-reports/v0.10/nsight}"
GPU_METRICS="${GPU_METRICS:-0}"
mkdir -p "$OUTDIR"
OUTDIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTDIR")"
PACK="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PACK")"
command -v nsys >/dev/null || { echo "nsys unavailable" >&2; exit 2; }
TAG="${METHOD}_${STAGE}_M${M}_R${REPEATS}"
BASE="$OUTDIR/$TAG"
LOG="${BASE}_profile.log"
rm -f "${BASE}.nsys-rep" "${BASE}.qdrep" "${BASE}.sqlite" "${BASE}_summary.txt" "$LOG" || true
TRACE="cuda,nvtx,osrt,cublas"
METRIC_ARGS=()
if [[ "$GPU_METRICS" == "1" ]]; then
  METRIC_ARGS+=(--gpu-metrics-device=0 --gpu-metrics-frequency=10000)
fi
set +e
nsys profile \
  --trace="$TRACE" \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --show-output=true \
  --output="$BASE" \
  "${METRIC_ARGS[@]}" \
  python -u scripts/profile_hybrid.py \
    --pack "$PACK" --method "$METHOD" --stage "$STAGE" --m "$M" \
    --warmup "$WARMUP" --repeats "$REPEATS" \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
if [[ $STATUS -ne 0 ]]; then exit "$STATUS"; fi
REPORT="${BASE}.nsys-rep"
[[ -s "$REPORT" ]] || REPORT="${BASE}.qdrep"
[[ -s "$REPORT" ]] || { echo "report missing" >&2; exit 3; }
SQLITE="${BASE}.sqlite"
rm -f "$SQLITE"
nsys export --type=sqlite --force-overwrite=true --output="$SQLITE" "$REPORT"
SUMMARY="${BASE}_summary.txt"
{
  echo "Report: $REPORT"; echo "SQLite: $SQLITE"; echo "Generated: $(date -Is)"
  echo "Pack: $PACK"; echo "Method: $METHOD"; echo "Stage: $STAGE"; echo "M: $M"; echo "Repeats: $REPEATS"
  echo; echo "=== LOG ==="; cat "$LOG"
  echo; echo "=== ANALYZE ==="; nsys analyze "$REPORT" 2>&1 || true
} > "$SUMMARY"
REPORTS=(
  'cuda_gpu_sum:nvtx-name' 'cuda_gpu_kern_sum:nvtx-name' 'cuda_gpu_kern_gb_sum:nvtx-name'
  'cuda_gpu_trace:nvtx-name' 'cuda_api_sum' 'cuda_kern_exec_sum:nvtx-name'
  'cuda_gpu_mem_time_sum' 'cuda_gpu_mem_size_sum' 'nvtx_pushpop_sum' 'nvtx_gpu_proj_sum' 'osrt_sum'
)
for ITEM in "${REPORTS[@]}"; do
  SAFE="${ITEM//:/_}"
  { echo; echo "=== $ITEM ==="; nsys stats --timeunit usec --report "$ITEM" "$SQLITE" 2>&1 || true; } >> "$SUMMARY"
  nsys stats --timeunit usec --report "$ITEM" --format csv --output - "$SQLITE" > "${BASE}_${SAFE}.csv" 2>&1 || true
done
cat > "${BASE}_manifest.json" <<JSON
{"pack":"$PACK","method":"$METHOD","stage":"$STAGE","m":$M,"repeats":$REPEATS,"report":"$(basename "$REPORT")","sqlite":"$(basename "$SQLITE")","summary":"$(basename "$SUMMARY")"}
JSON
echo "$REPORT"
