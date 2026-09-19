#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: generate-rank-hosts.sh --workers-file PATH --n-trainer N --output-file PATH

Generate the '<trainer-rank> <worker-ipv4>' placement map used by local-artifact
staging and submission. Ranks are assigned to workers in round-robin order, so
multiple trainer ranks can share one physical worker while remaining balanced.

Options:
  --workers-file PATH  Worker IPv4 addresses, one per line. Comments are allowed.
  --n-trainer N        Number of trainer ranks to map.
  --output-file PATH   Destination rank-host map.
  --help               Show this message.
EOF
}

workers_file=""
n_trainer=""
output_file=""

while (( $# > 0 )); do
    case "$1" in
        --workers-file) workers_file="$2"; shift 2 ;;
        --n-trainer) n_trainer="$2"; shift 2 ;;
        --output-file) output_file="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$workers_file" ]] || die "--workers-file is required"
[[ -n "$n_trainer" ]] || die "--n-trainer is required"
[[ -n "$output_file" ]] || die "--output-file is required"
require_file "$workers_file"
[[ "$n_trainer" =~ ^[1-9][0-9]*$ ]] || die "--n-trainer must be a positive integer"
[[ "$workers_file" != "$output_file" ]] || die "Input and output files must differ"

declare -a workers=()
declare -A seen_workers=()
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    read -r worker extra <<< "$line"
    [[ -n "${worker:-}" && -z "${extra:-}" ]] || \
        die "Invalid worker entry: '${raw_line}'"
    require_ipv4 "$worker"
    [[ -z "${seen_workers[$worker]+x}" ]] || \
        die "Worker appears more than once: ${worker}"
    seen_workers[$worker]=1
    workers+=("$worker")
done < "$workers_file"

(( ${#workers[@]} > 0 )) || die "No workers found in ${workers_file}"

output_dir="$(dirname "$output_file")"
[[ -d "$output_dir" ]] || die "Output directory does not exist: ${output_dir}"
temp_file="$(mktemp "${output_file}.tmp.XXXXXX")"
trap 'rm -f "$temp_file"' EXIT

for ((rank = 0; rank < n_trainer; rank += 1)); do
    worker_index=$((rank % ${#workers[@]}))
    printf '%s %s\n' "$rank" "${workers[$worker_index]}" >> "$temp_file"
done
mv "$temp_file" "$output_file"
trap - EXIT

printf 'Generated %s rank mapping(s) across %s worker(s): %s\n' \
    "$n_trainer" "${#workers[@]}" "$output_file"
for ((worker_index = 0; worker_index < ${#workers[@]}; worker_index += 1)); do
    rank_count=$((n_trainer / ${#workers[@]}))
    if (( worker_index < n_trainer % ${#workers[@]} )); then
        ((rank_count += 1))
    fi
    printf '  %s: %s rank(s)\n' "${workers[$worker_index]}" "$rank_count"
done
