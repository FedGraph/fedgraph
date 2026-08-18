#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: launch-workers.sh --head-ip IP --workers-file PATH [options]

Create one detached tmux session per listed worker. start-worker.sh performs
the actual environment, GPU, disk, and Ray checks on each remote host.

Options:
  --head-ip IP          Ray-head private IPv4.
  --workers-file PATH   One private IPv4 or SSH host per non-comment line.
  --ssh-user USER       Remote SSH user. Default: ubuntu.
  --remote-dir PATH     Remote checkout path. Default: /home/ubuntu/fedgraph.
  --num-cpus N          Ray CPUs per worker. Default: 12.
  --hf-home PATH        Worker-local HF cache. Default: /mnt/fedgraph-nvme/hf-cache.
  --preflight           Run worker prerequisite and head-connectivity checks only.
  --help                Show this message.

Set SSH_OPTS if SSH requires an identity file. Each worker may keep HF_TOKEN in
~/.config/fedgraph/hf.env; it is sourced only by start-worker.sh on that host.
EOF
}

head_ip=""
workers_file=""
ssh_user="ubuntu"
remote_dir="/home/ubuntu/fedgraph"
num_cpus=12
hf_home="/mnt/fedgraph-nvme/hf-cache"
preflight_only=false

while (( $# > 0 )); do
    case "$1" in
        --head-ip) head_ip="$2"; shift 2 ;;
        --workers-file) workers_file="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --num-cpus) num_cpus="$2"; shift 2 ;;
        --hf-home) hf_home="$2"; shift 2 ;;
        --preflight) preflight_only=true; shift ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$head_ip" ]] || die "--head-ip is required"
[[ -n "$workers_file" ]] || die "--workers-file is required"
require_ipv4 "$head_ip"
require_file "$workers_file"
[[ "$num_cpus" =~ ^[0-9]+$ ]] || die "--num-cpus must be an integer"
require_command ssh

read -r -a ssh_opts <<< "${SSH_OPTS:-}"
worker_count=0
preflight_failures=0
while IFS= read -r worker || [[ -n "$worker" ]]; do
    worker="${worker%%#*}"
    worker="${worker//[[:space:]]/}"
    [[ -z "$worker" ]] && continue

    session="ray-worker-${worker//./-}"
    remote_command="cd ${remote_dir} && ${remote_dir}/scripts/manual_ray/start-worker.sh --head-ip ${head_ip} --num-cpus ${num_cpus} --hf-home ${hf_home}"
    [[ "$preflight_only" == "true" ]] && remote_command+=" --preflight-only"
    if [[ "$preflight_only" == "true" ]]; then
        printf 'Preflighting %s\n' "$worker"
        if ! ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" "$remote_command" </dev/null; then
            warn "Preflight failed on ${worker}"
            preflight_failures=$((preflight_failures + 1))
        fi
    else
        printf 'Launching %s in tmux session %s\n' "$worker" "$session"
        ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" \
            "tmux has-session -t '${session}' 2>/dev/null || tmux new-session -d -s '${session}' \"${remote_command}\"" \
            </dev/null
    fi
    ((worker_count += 1))
done < "$workers_file"

(( worker_count > 0 )) || die "No workers were found in ${workers_file}"
if [[ "$preflight_only" == "true" ]]; then
    (( preflight_failures == 0 )) || die "${preflight_failures} worker preflight(s) failed"
    printf 'Completed worker preflight on %s worker(s).\n' "$worker_count"
else
    printf 'Requested Ray startup on %s worker(s).\n' "$worker_count"
fi
