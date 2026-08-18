#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEDGRAPH_DIR="${FEDGRAPH_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-fedgraph312}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/${ENV_NAME}/bin/python}"
RAY_BIN="${RAY_BIN:-${CONDA_ROOT}/envs/${ENV_NAME}/bin/ray}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'Warning: %s\n' "$*" >&2
}

require_file() {
    [[ -f "$1" ]] || die "Required file does not exist: $1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command is unavailable: $1"
}

activate_fedgraph_env() {
    require_file "${CONDA_ROOT}/etc/profile.d/conda.sh"
    # shellcheck disable=SC1090
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    require_file "${PYTHON_BIN}"
    require_file "${RAY_BIN}"
}

private_ipv4() {
    hostname -I | awk '{print $1}'
}

require_ipv4() {
    [[ "$1" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die "Invalid IPv4 address: $1"
}

gib_to_bytes() {
    local gib="$1"
    [[ "$gib" =~ ^[0-9]+$ ]] || die "Expected an integer GiB value, got: $gib"
    printf '%s\n' "$((gib * 1024 * 1024 * 1024))"
}

check_hf_cache() {
    local hf_home="$1"
    local min_free_gib="$2"
    local available_gib

    mkdir -p "${hf_home}"
    available_gib="$(df -BG --output=avail "${hf_home}" | tail -n 1 | tr -dc '0-9')"
    [[ -n "$available_gib" ]] || die "Could not determine free space for ${hf_home}"
    if (( available_gib < min_free_gib )); then
        die "${hf_home} has ${available_gib} GiB free; need at least ${min_free_gib} GiB"
    fi
    printf 'HF_HOME=%s (%s GiB free)\n' "${hf_home}" "${available_gib}"
}

check_worker_runtime() {
    require_command nvidia-smi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    "${PYTHON_BIN}" - <<'PY'
import ray
import torch
import torch_geometric

print(f"torch={torch.__version__}")
print(f"torch_geometric={torch_geometric.__version__}")
print(f"ray={ray.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot use CUDA on this worker")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
}
