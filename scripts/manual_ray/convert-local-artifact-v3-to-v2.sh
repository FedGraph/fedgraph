#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: convert-local-artifact-v3-to-v2.sh --rank-hosts-file PATH --source-dir PATH --output-dir PATH [options]

Convert the compact v3 shards already staged on each worker into a separate v2
artifact. Each host converts only the trainer ranks mapped to it.

Options:
  --rank-hosts-file PATH  Complete '<rank> <private-ipv4>' placement map.
  --source-dir PATH       Worker-local compact v3 artifact root.
  --output-dir PATH       New worker-local v2 artifact root.
  --remote-dir PATH       Worker checkout. Default: /home/ubuntu/fedgraph.
  --python-bin PATH       Worker Python. Default:
                          /home/ubuntu/miniconda3/envs/fedgraph312/bin/python.
  --chunk-edges N         Conversion edge batch. Default: 1000000.
  --jobs N                Concurrent worker hosts. Default: 2.
  --ssh-user USER         Remote SSH user. Default: ubuntu.
  --no-checksums          Skip output checksum generation.
  --copy-files            Copy unchanged files instead of hard-linking them.
  --log-dir PATH          Head-side logs. Default: ~/fedgraph-logs/artifact-convert.
  --help                  Show this message.

The source artifact is never modified. Set SSH_OPTS when SSH needs an identity
file or jump host.
EOF
}

rank_hosts_file=""
source_dir=""
output_dir=""
remote_dir="/home/ubuntu/fedgraph"
python_bin="/home/ubuntu/miniconda3/envs/fedgraph312/bin/python"
chunk_edges=1000000
jobs=2
ssh_user=ubuntu
checksums=true
copy_files=false
log_dir="${HOME}/fedgraph-logs/artifact-convert"

while (( $# > 0 )); do
    case "$1" in
        --rank-hosts-file) rank_hosts_file="$2"; shift 2 ;;
        --source-dir) source_dir="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --python-bin) python_bin="$2"; shift 2 ;;
        --chunk-edges) chunk_edges="$2"; shift 2 ;;
        --jobs) jobs="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --no-checksums) checksums=false; shift ;;
        --copy-files) copy_files=true; shift ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$rank_hosts_file" ]] || die "--rank-hosts-file is required"
[[ -n "$source_dir" ]] || die "--source-dir is required"
[[ -n "$output_dir" ]] || die "--output-dir is required"
[[ "$source_dir" == /* && "$output_dir" == /* ]] || \
    die "--source-dir and --output-dir must be absolute worker-local paths"
[[ "$source_dir" != "$output_dir" ]] || die "source and output directories must differ"
[[ "$chunk_edges" =~ ^[1-9][0-9]*$ ]] || die "--chunk-edges must be positive"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be positive"
require_file "$rank_hosts_file"
require_file "${source_dir}/manifest.json"
require_command python3
require_command ssh

n_trainer="$(python3 - "${source_dir}/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if (manifest.get("artifact_version"), manifest.get("hop_semantics")) != (3, 2):
    raise SystemExit("source artifact must use the compact v3 2-hop contract")
print(manifest["n_trainer"])
PY
)"

declare -A host_ranks=()
declare -A seen_ranks=()
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    read -r raw_rank host extra <<< "$line"
    [[ -n "${raw_rank:-}" && -n "${host:-}" && -z "${extra:-}" ]] || \
        die "Invalid placement entry: '${raw_line}'"
    [[ "$raw_rank" =~ ^[0-9]+$ ]] || die "Invalid trainer rank: $raw_rank"
    rank="$((10#$raw_rank))"
    (( rank < n_trainer )) || die "Rank ${rank} is outside [0, $((n_trainer - 1))]"
    [[ -z "${seen_ranks[$rank]+x}" ]] || die "Rank ${rank} appears more than once"
    seen_ranks[$rank]="$host"
    if [[ -n "${host_ranks[$host]:-}" ]]; then
        host_ranks[$host]+=",${rank}"
    else
        host_ranks[$host]="$rank"
    fi
done < "$rank_hosts_file"

for ((rank = 0; rank < n_trainer; rank += 1)); do
    [[ -n "${seen_ranks[$rank]+x}" ]] || die "Rank-host map is missing rank ${rank}"
done

read -r -a ssh_opts <<< "${SSH_OPTS:-}"
mkdir -p "$log_dir"

run_host() {
    local host="$1"
    local ranks="${host_ranks[$host]}"
    local -a remote_command=(
        "$python_bin"
        "${remote_dir}/scripts/data/convert_nc_artifact_v3_to_v2.py"
        --source-dir "$source_dir"
        --output-dir "$output_dir"
        --trainer-ids "$ranks"
        --chunk-edges "$chunk_edges"
    )
    [[ "$checksums" == "false" ]] && remote_command+=(--no-checksums)
    [[ "$copy_files" == "true" ]] && remote_command+=(--copy-files)

    local quoted=""
    local argument escaped
    for argument in "${remote_command[@]}"; do
        printf -v escaped '%q' "$argument"
        quoted+="${escaped} "
    done
    ssh "${ssh_opts[@]}" "${ssh_user}@${host}" "$quoted" </dev/null
}

wait_for_oldest() {
    local pid="${pids[0]}"
    local host="${pid_hosts[0]}"
    if wait "$pid"; then
        printf 'Completed v3-to-v2 conversion on %s (ranks %s)\n' \
            "$host" "${host_ranks[$host]}"
    else
        printf 'Failed conversion on %s; inspect %s\n' \
            "$host" "$log_dir/${host//./-}.log" >&2
        failures=$((failures + 1))
    fi
    pids=("${pids[@]:1}")
    pid_hosts=("${pid_hosts[@]:1}")
}

declare -a pids=()
declare -a pid_hosts=()
failures=0
for host in "${!host_ranks[@]}"; do
    while (( ${#pids[@]} >= jobs )); do
        wait_for_oldest
    done
    printf 'Starting conversion on %s for ranks %s\n' "$host" "${host_ranks[$host]}"
    run_host "$host" >"$log_dir/${host//./-}.log" 2>&1 &
    pids+=("$!")
    pid_hosts+=("$host")
done
while (( ${#pids[@]} > 0 )); do
    wait_for_oldest
done

(( failures == 0 )) || die "$failures worker conversion task(s) failed"
require_file "$PYTHON_BIN"
all_ranks=""
for ((rank = 0; rank < n_trainer; rank += 1)); do
    all_ranks+="${all_ranks:+,}${rank}"
done
"$PYTHON_BIN" "${FEDGRAPH_DIR}/scripts/data/convert_nc_artifact_v3_to_v2.py" \
    --source-dir "$source_dir" \
    --output-dir "$output_dir" \
    --trainer-ids "$all_ranks" \
    --chunk-edges "$chunk_edges" \
    --manifest-only
printf 'Converted compact shards on %s worker host(s).\n' "${#host_ranks[@]}"
printf 'Created head-side v2 control manifest at %s/manifest.json.\n' "$output_dir"
