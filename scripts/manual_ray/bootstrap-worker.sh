#!/usr/bin/env bash

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-fedgraph312}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: bootstrap-worker.sh [options]

Prepare a GPU worker's persistent root-disk software state. Run as the normal
Ubuntu user, not through sudo. This script does not mount or format local NVMe;
use prepare-worker-nvme.sh for that per-boot task.

Options:
  --system-packages       Install git, tmux, rsync, curl, htop, wget, and
                          ubuntu-drivers-common with apt.
  --install-nvidia-driver Install the recommended Ubuntu NVIDIA driver, then
                          stop. Reboot the instance and rerun this script.
                          Implies --system-packages.
  --system-only          Stop after the system package and optional driver step.
                          Requires --system-packages.
  --conda-root PATH       Miniconda root. Default: ~/miniconda3.
  --env-name NAME         Conda environment. Default: fedgraph312.
  --help                  Show this message.

Environment variables CONDA_ROOT, ENV_NAME, and MINICONDA_URL provide the same
customization for non-interactive fleet setup.
EOF
}

install_system_packages=false
install_nvidia_driver=false
system_only=false

while (( $# > 0 )); do
    case "$1" in
        --system-packages) install_system_packages=true; shift ;;
        --install-nvidia-driver) install_nvidia_driver=true; shift ;;
        --system-only) system_only=true; shift ;;
        --conda-root) CONDA_ROOT="$2"; shift 2 ;;
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

(( EUID != 0 )) || die "Run this as the normal Ubuntu user; the script invokes sudo only where needed"

[[ "$system_only" != "true" || "$install_system_packages" == "true" ]] \
    || die "--system-only requires --system-packages"

if [[ "$install_nvidia_driver" == "true" ]]; then
    install_system_packages=true
fi

if [[ "$install_system_packages" == "true" ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo is required for --system-packages"
    sudo apt-get update
    sudo apt-get install -y git tmux rsync curl htop wget ubuntu-drivers-common
fi

if [[ "$install_nvidia_driver" == "true" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        printf 'NVIDIA driver is already active; no driver installation was needed.\n'
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required for --install-nvidia-driver"
        sudo ubuntu-drivers install
        printf '\nNVIDIA driver packages were installed. Reboot this instance, confirm nvidia-smi works, then rerun bootstrap-worker.sh without --install-nvidia-driver.\n'
        exit 0
    fi
fi

if [[ "$system_only" == "true" ]]; then
    printf 'System package stage is ready.\n'
    exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable; use --install-nvidia-driver on a fresh Ubuntu GPU worker"
nvidia-smi >/dev/null 2>&1 || die "NVIDIA driver is installed but not active; reboot the worker before continuing"

download_miniconda() {
    local installer

    installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$installer" "$MINICONDA_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl --fail --location --output "$installer" "$MINICONDA_URL"
    else
        die "Install wget or curl first, for example with --system-packages"
    fi
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
}

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    printf 'Installing Miniconda under %s\n' "$CONDA_ROOT"
    download_miniconda
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda tos accept

if [[ ! -x "${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" ]]; then
    conda create -n "$ENV_NAME" python=3.12 -y
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

gpu_stack_matches() {
    python - <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

expected = {
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "torchaudio": "2.5.1+cu121",
    "pyg-lib": "0.4.0+pt25cu121",
    "torch-scatter": "2.1.2+pt25cu121",
    "torch-sparse": "0.6.18+pt25cu121",
    "torch-cluster": "1.6.3+pt25cu121",
    "torch-spline-conv": "1.2.2+pt25cu121",
}

try:
    actual = {name: version(name) for name in expected}
except PackageNotFoundError:
    raise SystemExit(1)

if actual != expected:
    print(f"GPU stack differs: {actual}")
    raise SystemExit(1)
PY
}

if ! gpu_stack_matches; then
    python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121
    python -m pip install \
        pyg-lib==0.4.0+pt25cu121 \
        torch-scatter==2.1.2+pt25cu121 \
        torch-sparse==0.6.18+pt25cu121 \
        torch-cluster==1.6.3+pt25cu121 \
        torch-spline-conv==1.2.2+pt25cu121 \
        -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
fi

runtime_stack_matches() {
    python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

expected = {
    "ray": "2.56.0",
    "torch-geometric": "2.8.0",
    "omegaconf": "2.3.1",
    "PyYAML": "6.0.1",
    "attridict": "0.0.9",
    "torchmetrics": "1.9.0",
    "matplotlib": "3.11.0",
    "tensorboard": "2.21.0",
    "dtaidistance": "2.4.0",
    "gdown": "6.1.0",
    "pandas": "3.0.3",
    "scikit-learn": "1.9.0",
    "tenseal": "0.3.16",
    "huggingface_hub": "1.22.0",
    "ogb": "1.3.6",
}

try:
    actual = {name: version(name) for name in expected}
except PackageNotFoundError:
    raise SystemExit(1)

if actual != expected:
    print(f"Runtime stack differs: {actual}")
    raise SystemExit(1)
PY
}

if ! runtime_stack_matches; then
    python -m pip install \
        'ray[default]==2.56.0' \
        torch-geometric==2.8.0 \
        omegaconf==2.3.1 \
        PyYAML==6.0.1 \
        attridict==0.0.9 \
        torchmetrics==1.9.0 \
        matplotlib==3.11.0 \
        tensorboard==2.21.0 \
        dtaidistance==2.4.0 \
        gdown==6.1.0 \
        pandas==3.0.3 \
        scikit-learn==1.9.0 \
        tenseal==0.3.16 \
        huggingface_hub==1.22.0 \
        ogb==1.3.6
fi

python - <<'PY'
import ray
import tenseal
import torch
import torch_geometric

print(f"torch={torch.__version__}")
print(f"torch_geometric={torch_geometric.__version__}")
print(f"ray={ray.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot use the worker GPU")
print(f"gpu={torch.cuda.get_device_name(0)}")
print("Persistent worker environment is ready. Run sync-workers.sh --install-editable after source synchronization.")
PY
