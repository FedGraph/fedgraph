#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: gpu-monitor.sh start|stop --run-id NAME [options]

Run exactly one nvidia-smi sampler per physical worker for a single benchmark.
The sampler is independent of the Ray worker lifetime and writes a compact
summary when it stops.

Options:
  --run-id NAME                Required experiment-safe identifier.
  --log-dir PATH               Default: /home/ubuntu/fedgraph-logs/papers100m-0hop.
  --sample-interval-seconds N  Default: 1.
  --help                       Show this message.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ $# -gt 0 ]] || { usage; exit 2; }
command="$1"
shift

run_id=""
log_dir="${FEDGRAPH_LOG_DIR:-/home/ubuntu/fedgraph-logs/papers100m-0hop}"
sample_interval_seconds=1

while (( $# > 0 )); do
    case "$1" in
        --run-id) run_id="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --sample-interval-seconds) sample_interval_seconds="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ "$command" == "start" || "$command" == "stop" ]] || die "Expected start or stop"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-id must contain only letters, numbers, dots, underscores, or hyphens"
[[ "$sample_interval_seconds" =~ ^[1-9][0-9]*$ ]] || die "--sample-interval-seconds must be a positive integer"
require_command nvidia-smi

host="$(hostname -s)"
run_dir="${log_dir}/gpu/${run_id}"
csv_path="${run_dir}/${host}_nvidia_smi.csv"
pid_path="${run_dir}/${host}_nvidia_smi.pid"
summary_path="${run_dir}/${host}_gpu_summary.json"

monitor_is_running() {
    [[ -f "$pid_path" ]] || return 1
    local pid
    pid="$(<"$pid_path")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    ps -p "$pid" -o comm= 2>/dev/null | tr -d '[:space:]' | grep -qx "nvidia-smi"
}

if [[ "$command" == "start" ]]; then
    if monitor_is_running; then
        die "GPU monitor is already running for ${run_id} on ${host}"
    fi
    if [[ -e "$csv_path" || -e "${csv_path}.gz" || -e "$summary_path" ]]; then
        die "GPU monitor artifacts already exist for ${run_id} on ${host}; use a new experiment name"
    fi
    mkdir -p "$run_dir"
    nohup nvidia-smi \
        --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits -l "$sample_interval_seconds" \
        > "$csv_path" 2>&1 &
    monitor_pid=$!
    printf '%s\n' "$monitor_pid" > "$pid_path"
    printf 'Started GPU monitor for %s on %s: pid=%s, log=%s\n' \
        "$run_id" "$host" "$monitor_pid" "$csv_path"
    exit 0
fi

if monitor_is_running; then
    monitor_pid="$(<"$pid_path")"
    kill "$monitor_pid"
    for _ in {1..50}; do
        kill -0 "$monitor_pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$monitor_pid" 2>/dev/null; then
        warn "GPU monitor ${monitor_pid} did not exit promptly"
    fi
fi

if [[ -f "$csv_path" ]]; then
    require_command python3
    python3 "${SCRIPT_DIR}/summarize-gpu-log.py" \
        --input "$csv_path" \
        --output "$summary_path"
    if command -v gzip >/dev/null 2>&1; then
        gzip -f "$csv_path"
    fi
fi
printf 'Stopped GPU monitor for %s on %s. Summary: %s\n' \
    "$run_id" "$host" "$summary_path"
