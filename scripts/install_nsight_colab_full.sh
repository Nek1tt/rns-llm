#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
INSTALL_NCU="${INSTALL_NCU:-1}"
NSYS_SERIES="${NSYS_SERIES:-}"
NCU_SERIES="${NCU_SERIES:-}"

realpath_safe() { readlink -f "$1" 2>/dev/null || printf '%s\n' "$1"; }

valid_command() {
  local name="$1"
  local candidate
  candidate="$(command -v "$name" 2>/dev/null || true)"
  [[ -n "$candidate" && -x "$candidate" ]] || return 1
  if [[ "$name" == "nsys" && "$(realpath_safe "$candidate")" == *"/nsight-compute/"* ]]; then
    return 1
  fi
  "$candidate" --version >/dev/null 2>&1
}

choose_versioned_package() {
  local prefix="$1"
  local requested_series="$2"
  local candidate=""
  if [[ -n "$requested_series" ]]; then
    local escaped="${requested_series//./\\.}"
    candidate="$(apt-cache pkgnames 2>/dev/null | grep -E "^${prefix}-${escaped}(\\.[0-9]+)+$" | sort -V | tail -n 1 || true)"
  fi
  if [[ -z "$candidate" ]]; then
    candidate="$(apt-cache pkgnames 2>/dev/null | grep -E "^${prefix}-[0-9]{4}\\.[0-9]+(\\.[0-9]+)+$" | sort -V | tail -n 1 || true)"
  fi
  [[ -n "$candidate" ]] && printf '%s\n' "$candidate"
}

find_binary_in_package() {
  local package="$1"
  local binary="$2"
  local candidate
  while IFS= read -r candidate; do
    [[ -x "$candidate" ]] || continue
    if [[ "$binary" == "nsys" && "$(realpath_safe "$candidate")" == *"/nsight-compute/"* ]]; then
      continue
    fi
    printf '%s\n' "$(realpath_safe "$candidate")"
    return 0
  done < <(dpkg -L "$package" 2>/dev/null | grep -E "/${binary}$" || true)
  return 1
}

install_cuda_repo() {
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" ]] || { echo "Expected Ubuntu, found ${ID:-unknown}" >&2; exit 2; }
  local distro="ubuntu${VERSION_ID//./}"
  local arch="$(dpkg --print-architecture)"
  [[ "$arch" == "amd64" ]] && arch="x86_64"
  apt-get update -qq
  apt-get install -y --no-install-recommends ca-certificates wget gnupg
  local keyring="/tmp/cuda-keyring_1.1-1_all.deb"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-keyring_1.1-1_all.deb" -O "$keyring"
  dpkg -i "$keyring"
  apt-get update -qq
}

need_install=0
valid_command nsys || need_install=1
if [[ "$INSTALL_NCU" != "0" ]]; then
  valid_command ncu || need_install=1
fi

if [[ $need_install -eq 1 ]]; then
  install_cuda_repo
fi

if ! valid_command nsys; then
  if [[ -L /usr/local/bin/nsys && "$(realpath_safe /usr/local/bin/nsys)" == *"/nsight-compute/"* ]]; then
    rm -f /usr/local/bin/nsys
  fi
  NSYS_PACKAGE="$(choose_versioned_package nsight-systems "$NSYS_SERIES" || true)"
  [[ -n "$NSYS_PACKAGE" ]] || { echo "No concrete Nsight Systems package found" >&2; exit 3; }
  echo "Installing $NSYS_PACKAGE"
  apt-get install -y --no-install-recommends "$NSYS_PACKAGE"
  NSYS_BIN="$(find_binary_in_package "$NSYS_PACKAGE" nsys || true)"
  [[ -n "$NSYS_BIN" ]] || { echo "Could not locate nsys in $NSYS_PACKAGE" >&2; exit 4; }
  ln -sfn "$NSYS_BIN" /usr/local/bin/nsys
  hash -r
fi

if [[ "$INSTALL_NCU" != "0" ]] && ! valid_command ncu; then
  NCU_PACKAGE="$(choose_versioned_package nsight-compute "$NCU_SERIES" || true)"
  if [[ -n "$NCU_PACKAGE" ]]; then
    echo "Installing $NCU_PACKAGE"
    apt-get install -y --no-install-recommends "$NCU_PACKAGE"
    NCU_BIN="$(find_binary_in_package "$NCU_PACKAGE" ncu || true)"
    [[ -n "$NCU_BIN" ]] && ln -sfn "$NCU_BIN" /usr/local/bin/ncu
    hash -r
  fi
fi

valid_command nsys || { echo "nsys installation failed" >&2; exit 5; }
echo "nsys: $(command -v nsys) -> $(realpath_safe "$(command -v nsys)")"
nsys --version

if [[ "$INSTALL_NCU" != "0" ]]; then
  valid_command ncu || { echo "ncu installation failed" >&2; exit 6; }
  echo "ncu: $(command -v ncu) -> $(realpath_safe "$(command -v ncu)")"
  ncu --version
fi

# A real NSYS probe catches invalid plugin installations before long runs.
PROBE_BASE="/tmp/nsys_v0142_probe"
rm -f "${PROBE_BASE}.nsys-rep" "${PROBE_BASE}.qdrep" "${PROBE_BASE}.log"
set +e
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output="$PROBE_BASE" \
  python -u -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); print(float(x.item()))' \
  >"${PROBE_BASE}.log" 2>&1
status=$?
set -e
cat "${PROBE_BASE}.log"
[[ $status -eq 0 ]] || { echo "Nsight Systems probe failed with code $status" >&2; exit 7; }
[[ -s "${PROBE_BASE}.nsys-rep" || -s "${PROBE_BASE}.qdrep" ]] || { echo "Nsight Systems probe created no report" >&2; exit 8; }
echo "Nsight installation probe passed."
