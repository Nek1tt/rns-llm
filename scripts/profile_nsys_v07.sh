#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-rns_v07}"
STAGE="${2:-e2e}"
M="${3:-1}"
K="${4:-768}"
N="${5:-768}"
REPEATS="${6:-300}"

WARMUP="${WARMUP:-20}"
OUTDIR="${OUTDIR:-reports/v0.7/nsight}"
GPU_METRICS="${GPU_METRICS:-0}"
GPU_METRICS_FREQUENCY="${GPU_METRICS_FREQUENCY:-10000}"
TRACE_PYTHON_GIL="${TRACE_PYTHON_GIL:-0}"

mkdir -p "$OUTDIR"
OUTDIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTDIR")"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found. Run scripts/install_nsight_colab_full.sh first." >&2
  exit 2
fi

NSYS_REAL="$(readlink -f "$(command -v nsys)")"
if [[ "$NSYS_REAL" == *"/nsight-compute/"* ]]; then
  echo "Invalid nsys executable: $NSYS_REAL" >&2
  exit 20
fi

TAG="${BACKEND}_${STAGE}_M${M}_K${K}_N${N}_R${REPEATS}"
BASE="$OUTDIR/$TAG"
LOG="${BASE}_profile.log"

TRACE="cuda,nvtx,osrt,cublas"
if [[ "$TRACE_PYTHON_GIL" == "1" ]]; then
  TRACE="${TRACE},python-gil"
fi

GPU_METRIC_ARGS=()
if [[ "$GPU_METRICS" == "1" ]]; then
  PROFILE_HELP="$(nsys profile --help 2>&1 || true)"
  if grep -q -- '--gpu-metrics-devices' <<<"$PROFILE_HELP"; then
    GPU_METRIC_ARGS+=(--gpu-metrics-devices=all)
  elif grep -q -- '--gpu-metrics-device' <<<"$PROFILE_HELP"; then
    GPU_METRIC_ARGS+=(--gpu-metrics-device=all)
  else
    echo "GPU metrics requested but this nsys version exposes no GPU metrics option." >&2
  fi
  if [[ ${#GPU_METRIC_ARGS[@]} -gt 0 ]]; then
    GPU_METRIC_ARGS+=(--gpu-metrics-frequency="$GPU_METRICS_FREQUENCY")
  fi
fi

clean_outputs() {
  rm -f \
    "${BASE}.nsys-rep" \
    "${BASE}.qdrep" \
    "${BASE}.sqlite" \
    "${BASE}.qstrm" \
    "${BASE}"-*.nsys-rep \
    "${BASE}"-*.qdrep \
    "${BASE}"_*.csv \
    "${BASE}"_summary.txt \
    "$LOG" || true
}

find_report() {
  local candidate
  for candidate in "${BASE}.nsys-rep" "${BASE}.qdrep"; do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  find "$OUTDIR" -maxdepth 1 -type f \
    \( -name "${TAG}*.nsys-rep" -o -name "${TAG}*.qdrep" \) \
    -size +0c -print 2>/dev/null | sort | tail -n 1
}

run_profile() {
  local capture_mode="$1"
  shift
  clean_outputs

  echo "Profiling: $TAG"
  echo "Capture mode: $capture_mode"
  echo "nsys: $NSYS_REAL"
  echo "trace: $TRACE"
  echo "GPU metric args: ${GPU_METRIC_ARGS[*]:-none}"

  local profiler_arg=()
  if [[ "$capture_mode" == "full-process" ]]; then
    profiler_arg=(--disable-profiler-api)
  fi

  set +e
  nsys profile \
    --trace="$TRACE" \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    --cuda-event-trace=false \
    --force-overwrite=true \
    --show-output=true \
    --output="$BASE" \
    "${GPU_METRIC_ARGS[@]}" \
    "$@" \
    python -u scripts/profile_v07.py \
      --backend "$BACKEND" \
      --stage "$STAGE" \
      --m "$M" --k "$K" --n "$N" \
      --warmup "$WARMUP" \
      --repeats "$REPEATS" \
      "${profiler_arg[@]}" \
    2>&1 | tee "$LOG"
  local status=${PIPESTATUS[0]}
  set -e
  echo "nsys exit code: $status"
  return "$status"
}

run_profile \
  "cudaProfilerApi" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop || true

REPORT="$(find_report || true)"
if [[ -z "$REPORT" ]]; then
  echo "No report from cudaProfilerApi capture; retrying full-process trace."
  run_profile "full-process" || true
  REPORT="$(find_report || true)"
fi

if [[ -z "$REPORT" || ! -s "$REPORT" ]]; then
  echo "Nsight Systems report was not created." >&2
  tail -n 200 "$LOG" >&2 || true
  exit 3
fi

# Export once and always overwrite. This avoids stale SQLite summaries from an
# earlier report with the same base name.
SQLITE="${BASE}.sqlite"
rm -f "$SQLITE"
nsys export \
  --type=sqlite \
  --force-overwrite=true \
  --output="$SQLITE" \
  "$REPORT"

SUMMARY="${BASE}_summary.txt"
{
  echo "Report: $REPORT"
  echo "SQLite: $SQLITE"
  echo "Generated: $(date -Is)"
  echo "nsys: $NSYS_REAL"
  echo "Backend: $BACKEND"
  echo "Stage: $STAGE"
  echo "Shape: M=$M K=$K N=$N"
  echo "Repeats: $REPEATS"
  echo
  echo "=== PROFILE LOG ==="
  cat "$LOG"
  echo
  echo "================================================================================"
  echo "=== NSIGHT EXPERT ANALYSIS ==="
  echo "================================================================================"
  nsys analyze "$REPORT" 2>&1 || true
} > "$SUMMARY"

REPORTS=(
  'cuda_gpu_sum:nvtx-name'
  'cuda_gpu_kern_sum:nvtx-name'
  'cuda_gpu_kern_gb_sum:nvtx-name'
  'cuda_gpu_trace:nvtx-name'
  'cuda_api_sum'
  'cuda_kern_exec_sum:nvtx-name'
  'cuda_gpu_mem_time_sum'
  'cuda_gpu_mem_size_sum'
  'nvtx_pushpop_sum'
  'nvtx_gpu_proj_sum'
  'osrt_sum'
  'syscall_sum'
)

for item in "${REPORTS[@]}"; do
  safe_item="${item//:/_}"
  {
    echo
    echo "================================================================================"
    echo "=== ${item} ==="
    echo "================================================================================"
    nsys stats \
      --timeunit usec \
      --report "$item" \
      "$SQLITE" 2>&1
  } >> "$SUMMARY" || true

  nsys stats \
    --timeunit usec \
    --report "$item" \
    --format csv \
    --output - \
    "$SQLITE" \
    > "${BASE}_${safe_item}.csv" 2>&1 || true
done

cat > "${BASE}_manifest.json" <<JSON
{
  "backend": "$BACKEND",
  "stage": "$STAGE",
  "shape": {"m": $M, "k": $K, "n": $N},
  "repeats": $REPEATS,
  "report": "$(basename "$REPORT")",
  "sqlite": "$(basename "$SQLITE")",
  "summary": "$(basename "$SUMMARY")",
  "gpu_metrics_requested": $([[ "$GPU_METRICS" == "1" ]] && echo true || echo false)
}
JSON

echo "Nsight Systems report: $REPORT"
echo "Fresh SQLite export: $SQLITE"
echo "Summary: $SUMMARY"
