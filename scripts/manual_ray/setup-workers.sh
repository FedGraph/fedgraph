#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: setup-workers.sh --stage STAGE --workers-file PATH [options]

Copy one self-contained setup helper to each worker and run it over SSH. The
system, environment, and NVMe stages are intentionally separate because an
NVIDIA driver installation needs a reboot and EC2 instance-store NVMe has a
different lifecycle from the root disk.

Stages:
  system       Install system packages. Add --install-nvidia-driver on a fresh
               Ubuntu GPU image; reboot workers before the environment stage.
  environment  Install Miniconda, fedgraph312, the pinned CUDA/PyG stack, and
               FedGraph's required runtime packages.
  nvme         Mount local instance-store NVMe and create HF/Ray directories.

Options:
  --stage STAGE         Required: system, environment, or nvme.
  --workers-file PATH   Required: one private IP or SSH host per non-comment line.
  --ssh-user USER       Remote SSH user. Default: ubuntu.
  --jobs N              Concurrent workers. Default: 2.
  --install-nvidia-driver  Valid only in the system stage.
  --nvme-device PATH|auto
                        Required in the nvme stage, e.g. /dev/nvme1n1. auto
                        detects exactly one safe AWS instance-store NVMe per worker.
  --format-nvme         Pass --format to the NVMe helper. Required for a blank
                        fresh or stop/started instance-store device.
  --log-dir PATH        Head-side setup logs. Default: ~/fedgraph-logs/worker-setup.
  --help                Show this message.

Set SSH_OPTS if SSH needs an identity file or jump host. Driver installation
does not reboot workers automatically.
EOF
}

stage=""
workers_file=""
ssh_user="ubuntu"
jobs=2
install_nvidia_driver=false
nvme_device=""
format_nvme=false
log_dir="${HOME}/fedgraph-logs/worker-setup"

while (( $# > 0 )); do
    case "$1" in
        --stage) stage="$2"; shift 2 ;;
        --workers-file) workers_file="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --install-nvidia-driver) install_nvidia_driver=true; shift ;;
        --nvme-device) nvme_device="$2"; shift 2 ;;
        --format-nvme) format_nvme=true; shift ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ "$stage" == "system" || "$stage" == "environment" || "$stage" == "nvme" ]] \
    || die "--stage must be system, environment, or nvme"
[[ -n "$workers_file" ]] || die "--workers-file is required"
require_file "$workers_file"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
require_command scp
require_command ssh

if [[ "$install_nvidia_driver" == "true" && "$stage" != "system" ]]; then
    die "--install-nvidia-driver is valid only with --stage system"
fi
if [[ "$stage" == "nvme" && -z "$nvme_device" ]]; then
    die "--nvme-device is required with --stage nvme"
fi
if [[ "$stage" != "nvme" && "$format_nvme" == "true" ]]; then
    die "--format-nvme is valid only with --stage nvme"
fi

case "$stage" in
    system)
        helper="${SCRIPT_DIR}/bootstrap-worker.sh"
        remote_helper="/tmp/fedgraph-bootstrap-worker.sh"
        helper_args=(--system-packages --system-only)
        [[ "$install_nvidia_driver" == "true" ]] && helper_args+=(--install-nvidia-driver)
        ;;
    environment)
        helper="${SCRIPT_DIR}/bootstrap-worker.sh"
        remote_helper="/tmp/fedgraph-bootstrap-worker.sh"
        helper_args=()
        ;;
    nvme)
        helper="${SCRIPT_DIR}/prepare-worker-nvme.sh"
        remote_helper="/tmp/fedgraph-prepare-worker-nvme.sh"
        helper_args=(--device "$nvme_device")
        [[ "$format_nvme" == "true" ]] && helper_args+=(--format)
        ;;
esac
require_file "$helper"

read -r -a ssh_opts <<< "${SSH_OPTS:-}"
mkdir -p "$log_dir"

remote_command=(bash "$remote_helper" "${helper_args[@]}")
quoted_remote_command=""
for argument in "${remote_command[@]}"; do
    printf -v escaped_argument '%q' "$argument"
    quoted_remote_command+="${escaped_argument} "
done

run_worker() {
    local worker="$1"

    scp "${ssh_opts[@]}" "$helper" "${ssh_user}@${worker}:${remote_helper}" </dev/null
    ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" "$quoted_remote_command" </dev/null
}

wait_for_oldest() {
    local pid="${pids[0]}"
    local worker="${pid_workers[0]}"

    if wait "$pid"; then
        printf 'Completed %s on %s\n' "$stage" "$worker"
    else
        printf 'Failed %s on %s; inspect %s\n' "$stage" "$worker" "$log_dir/${stage}_${worker//./-}.log" >&2
        failures=$((failures + 1))
    fi
    pids=("${pids[@]:1}")
    pid_workers=("${pid_workers[@]:1}")
}

declare -a pids=()
declare -a pid_workers=()
failures=0
worker_count=0

while IFS= read -r worker || [[ -n "$worker" ]]; do
    worker="${worker%%#*}"
    worker="${worker//[[:space:]]/}"
    [[ -z "$worker" ]] && continue

    while (( ${#pids[@]} >= jobs )); do
        wait_for_oldest
    done

    log_file="$log_dir/${stage}_${worker//./-}.log"
    printf 'Starting %s on %s; log: %s\n' "$stage" "$worker" "$log_file"
    run_worker "$worker" >"$log_file" 2>&1 &
    pids+=("$!")
    pid_workers+=("$worker")
    worker_count=$((worker_count + 1))
done < "$workers_file"

(( worker_count > 0 )) || die "No workers were found in $workers_file"
while (( ${#pids[@]} > 0 )); do
    wait_for_oldest
done

(( failures == 0 )) || die "$failures worker setup task(s) failed"
printf 'Completed %s stage on %s worker(s).\n' "$stage" "$worker_count"
if [[ "$stage" == "system" && "$install_nvidia_driver" == "true" ]]; then
    printf 'Reboot every worker now. After nvidia-smi succeeds, run the environment stage.\n'
fi
