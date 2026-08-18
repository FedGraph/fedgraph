#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: sync-workers.sh --workers-file PATH [options]

Copy the current source tree, including intentional uncommitted code changes,
from the head to every worker. This script deliberately excludes Git metadata,
large datasets, and generated results.

Options:
  --workers-file PATH  One private IPv4 or SSH host per non-comment line.
  --ssh-user USER      Remote SSH user. Default: ubuntu.
  --remote-dir PATH    Remote checkout path. Default: /home/ubuntu/fedgraph.
  --install-editable   Run the one-time editable FedGraph install after syncing.
  --conda-root PATH    Remote Conda root. Default: /home/ubuntu/miniconda3.
  --env-name NAME      Remote Conda environment. Default: fedgraph312.
  --help               Show this message.

Set SSH_OPTS for an SSH identity or jump-host option. Example:
  export SSH_OPTS='-i /home/ubuntu/.ssh/AWS_FedGraph.pem'
EOF
}

workers_file=""
ssh_user="ubuntu"
remote_dir="/home/ubuntu/fedgraph"
install_editable=false
conda_root="/home/ubuntu/miniconda3"
env_name="fedgraph312"

while (( $# > 0 )); do
    case "$1" in
        --workers-file) workers_file="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --install-editable) install_editable=true; shift ;;
        --conda-root) conda_root="$2"; shift 2 ;;
        --env-name) env_name="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$workers_file" ]] || die "--workers-file is required"
require_file "$workers_file"
require_command rsync
require_command ssh
require_file "${FEDGRAPH_DIR}/fedgraph/trainer_class.py"

read -r -a ssh_opts <<< "${SSH_OPTS:-}"
worker_count=0
while IFS= read -r worker || [[ -n "$worker" ]]; do
    worker="${worker%%#*}"
    worker="${worker//[[:space:]]/}"
    [[ -z "$worker" ]] && continue

    printf '\nSyncing %s\n' "$worker"
    ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" "mkdir -p '${remote_dir}'" </dev/null
    rsync -az \
        -e "ssh ${SSH_OPTS:-}" \
        --exclude '.git/' \
        --exclude '.codex/' \
        --exclude 'dataset/' \
        --exclude 'benchmark/results/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        "${FEDGRAPH_DIR}/" "${ssh_user}@${worker}:${remote_dir}/" </dev/null

    if [[ "$install_editable" == "true" ]]; then
        printf 'Registering editable FedGraph install on %s\n' "$worker"
        ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" \
            "source ${conda_root}/etc/profile.d/conda.sh && conda activate ${env_name} && cd ${remote_dir} && python -m pip install -e . --no-deps" \
            </dev/null
    fi
    ((worker_count += 1))
done < "$workers_file"

(( worker_count > 0 )) || die "No workers were found in ${workers_file}"
printf '\nSynced source tree to %s worker(s).\n' "$worker_count"
