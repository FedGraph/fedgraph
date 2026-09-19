#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: sync-local-artifact.sh --workers-file PATH --source-dir PATH --remote-dir PATH [options]

Copy one completed local NC artifact root to the same path on every worker.
The source must contain manifest.json. Existing remote files are retained; this
script never passes rsync --delete.

Options:
  --workers-file PATH  One private IPv4 or SSH host per non-comment line.
  --source-dir PATH    Completed local artifact root on the head.
  --remote-dir PATH    Destination artifact root on every worker.
  --ssh-user USER      Remote SSH user. Default: ubuntu.
  --help               Show this message.

Set SSH_OPTS for an SSH identity or jump-host option.
EOF
}

workers_file=""
source_dir=""
remote_dir=""
ssh_user="ubuntu"

while (( $# > 0 )); do
    case "$1" in
        --workers-file) workers_file="$2"; shift 2 ;;
        --source-dir) source_dir="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$workers_file" ]] || die "--workers-file is required"
[[ -n "$source_dir" ]] || die "--source-dir is required"
[[ -n "$remote_dir" ]] || die "--remote-dir is required"
require_file "$workers_file"
require_file "${source_dir}/manifest.json"
require_command rsync
require_command ssh

source_dir="$(cd "$source_dir" && pwd)"
read -r -a ssh_opts <<< "${SSH_OPTS:-}"

worker_count=0
while IFS= read -r worker || [[ -n "$worker" ]]; do
    worker="${worker%%#*}"
    worker="${worker//[[:space:]]/}"
    [[ -z "$worker" ]] && continue

    printf '\nSyncing local artifact to %s\n' "$worker"
    ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" \
        "mkdir -p '${remote_dir}'" </dev/null
    rsync -az \
        -e "ssh ${SSH_OPTS:-}" \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        "${source_dir}/" "${ssh_user}@${worker}:${remote_dir}/" </dev/null
    ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" \
        "test -f '${remote_dir}/manifest.json'" </dev/null
    ((worker_count += 1))
done < "$workers_file"

(( worker_count > 0 )) || die "No workers were found in ${workers_file}"
printf '\nSynced local artifact to %s worker(s).\n' "$worker_count"
