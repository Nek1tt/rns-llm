#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${1:-reports/v0.14.2}"
SCOPES="${SCOPES:-matrix attention}"
ARCHS="${ARCHS:-fp16 native_int8 full_rns_int8_v07 full_rns_int8 full_rns_int16 full_rns_int32 hybrid_fp16 hybrid_fp16_parallel hybrid_rns_q8 hybrid_rns_q8_parallel hybrid_rns_q16 hybrid_rns_q16_parallel hybrid_rns_q32 hybrid_rns_q32_parallel}"
LUTS="${LUTS:-none one two all}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
RUN_NSYS="${RUN_NSYS:-1}"
RUN_NCU="${RUN_NCU:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
mkdir -p "$OUTROOT/ncu" "$OUTROOT/nsys"

run_checked() {
  local label="$1"; shift
  if "$@"; then
    return 0
  fi
  echo "${label}_FAILED: $*" >&2
  [[ "$STOP_ON_ERROR" == "1" ]] && return 1
  return 0
}

for scope in $SCOPES; do
  for arch in $ARCHS; do
    case "$arch" in
      fp16|native_int8|hybrid_fp16|hybrid_fp16_parallel)
        policies="none"
        ;;
      full_rns_int8_v07)
        policies="none one two"
        ;;
      *)
        policies="$LUTS"
        ;;
    esac
    for lut in $policies; do
      if [[ "$RUN_NSYS" == "1" ]]; then
        manifest="$OUTROOT/nsys/${scope}_${arch}_lut-${lut}_manifest.json"
        if [[ "$SKIP_EXISTING" == "1" && -s "$manifest" ]]; then
          echo "=== NSYS SKIP existing $scope $arch $lut ==="
        else
          echo "=== NSYS $scope $arch $lut ==="
          run_checked NSYS bash scripts/profile_nsys_v014.sh \
            "$arch" "$scope" "$lut" "$OUTROOT/nsys"
        fi
      fi
      if [[ "$RUN_NCU" == "1" ]]; then
        manifest="$OUTROOT/ncu/${scope}_${arch}_lut-${lut}_manifest.json"
        if [[ "$SKIP_EXISTING" == "1" && -s "$manifest" ]]; then
          echo "=== NCU SKIP existing $scope $arch $lut ==="
        else
          echo "=== NCU $scope $arch $lut ==="
          run_checked NCU bash scripts/profile_ncu_v014.sh \
            "$arch" "$scope" "$lut" "$OUTROOT/ncu"
        fi
      fi
    done
  done
done
