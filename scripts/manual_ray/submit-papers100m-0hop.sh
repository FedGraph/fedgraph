#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: submit-papers100m-0hop.sh --head-ip IP --rounds N --experiment-name NAME [options]

Submit the existing ten-trainer, legacy-Hugging-Face Papers100M artifact as a
0-hop FedAvg run. The script deliberately requires --rounds and a fresh output
name, preventing accidental long or overwritten experiments.

Options:
  --head-ip IP                 Ray-head private IPv4.
  --rounds N                   Required fixed global-round budget.
  --experiment-name NAME       Required unique result subdirectory name.
  --batch-size N               Seed nodes per local update. Default: 16.
  --local-step N               Updates per trainer per global round. Default: 3.
  --num-cpus-per-trainer N     Ray CPU request for each trainer. Default: 8.
  --hf-home PATH               Head-local cache path. Default: ~/fedgraph-hf-cache.
  --log-dir PATH               Driver log directory. Default: ~/fedgraph-logs/papers100m-0hop.
  --resource-monitor-mode MODE Application resource mode: off, manual, prometheus, or hybrid.
                               Default: manual.
  --resource-snapshot-interval-rounds N
                               Snapshot round 1 and every Nth round. Default: 10.
  --gpu-monitor-detail MODE    off or raw. Raw starts one experiment-scoped
                               nvidia-smi monitor per worker. Default: off.
  --workers-file PATH          Required with --gpu-monitor-detail raw.
  --ssh-user USER              Remote SSH user. Default: ubuntu.
  --remote-dir PATH            Remote FedGraph checkout. Default: /home/ubuntu/fedgraph.
  --worker-log-dir PATH        Worker monitor root. Default: /home/ubuntu/fedgraph-logs/papers100m-0hop.
  --gpu-sample-interval-seconds N
                               nvidia-smi sampling interval. Default: 1.
  --output-root PATH           Benchmark result root. Default: benchmark/results/nc_batch_size_convergence.
  --help                       Show this message.
EOF
}

head_ip="${HEAD_PRIVATE_IP:-}"
rounds=""
experiment_name=""
batch_size=16
local_step=3
num_cpus_per_trainer=8
hf_home="${HF_HOME:-${HOME}/fedgraph-hf-cache}"
log_dir="${FEDGRAPH_LOG_DIR:-${HOME}/fedgraph-logs/papers100m-0hop}"
output_root="benchmark/results/nc_batch_size_convergence"
resource_monitor_mode=manual
resource_snapshot_interval_rounds=10
gpu_monitor_detail=off
workers_file=""
ssh_user=ubuntu
remote_dir="/home/ubuntu/fedgraph"
worker_log_dir="/home/ubuntu/fedgraph-logs/papers100m-0hop"
gpu_sample_interval_seconds=1

while (( $# > 0 )); do
    case "$1" in
        --head-ip) head_ip="$2"; shift 2 ;;
        --rounds) rounds="$2"; shift 2 ;;
        --experiment-name) experiment_name="$2"; shift 2 ;;
        --batch-size) batch_size="$2"; shift 2 ;;
        --local-step) local_step="$2"; shift 2 ;;
        --num-cpus-per-trainer) num_cpus_per_trainer="$2"; shift 2 ;;
        --hf-home) hf_home="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --resource-monitor-mode) resource_monitor_mode="$2"; shift 2 ;;
        --resource-snapshot-interval-rounds) resource_snapshot_interval_rounds="$2"; shift 2 ;;
        --gpu-monitor-detail) gpu_monitor_detail="$2"; shift 2 ;;
        --workers-file) workers_file="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --worker-log-dir) worker_log_dir="$2"; shift 2 ;;
        --gpu-sample-interval-seconds) gpu_sample_interval_seconds="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$head_ip" ]] || die "--head-ip is required"
[[ -n "$rounds" ]] || die "--rounds is required"
[[ -n "$experiment_name" ]] || die "--experiment-name is required"
require_ipv4 "$head_ip"
for value in "$rounds" "$batch_size" "$local_step" "$num_cpus_per_trainer" \
    "$resource_snapshot_interval_rounds" "$gpu_sample_interval_seconds"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "Expected an integer, got: $value"
done
[[ "$resource_monitor_mode" =~ ^(off|manual|prometheus|hybrid)$ ]] || die "Invalid --resource-monitor-mode: $resource_monitor_mode"
[[ "$gpu_monitor_detail" =~ ^(off|raw)$ ]] || die "--gpu-monitor-detail must be off or raw"
if [[ "$gpu_monitor_detail" == "raw" ]]; then
    [[ -n "$workers_file" ]] || die "--workers-file is required for raw GPU monitoring"
    require_file "$workers_file"
    [[ "$experiment_name" =~ ^[A-Za-z0-9._-]+$ ]] || die "Raw GPU monitoring requires an experiment name without spaces or shell characters"
    require_command ssh
    require_command rsync
fi

read -r -a ssh_opts <<< "${SSH_OPTS:-}"
monitor_workers=()

for_each_worker() {
    local callback="$1"
    [[ -n "$workers_file" ]] || return 0
    local worker
    while IFS= read -r worker || [[ -n "$worker" ]]; do
        worker="${worker%%#*}"
        worker="${worker//[[:space:]]/}"
        [[ -z "$worker" ]] && continue
        "$callback" "$worker"
    done < "$workers_file"
}

start_gpu_monitor_on_worker() {
    local worker="$1"
    local remote_command
    remote_command="cd ${remote_dir} && ${remote_dir}/scripts/manual_ray/gpu-monitor.sh start --run-id ${experiment_name} --log-dir ${worker_log_dir} --sample-interval-seconds ${gpu_sample_interval_seconds}"
    ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" "$remote_command" </dev/null
    monitor_workers+=("$worker")
}

stop_gpu_monitors() {
    local worker
    for worker in "${monitor_workers[@]}"; do
        local remote_command
        remote_command="cd ${remote_dir} && ${remote_dir}/scripts/manual_ray/gpu-monitor.sh stop --run-id ${experiment_name} --log-dir ${worker_log_dir} --sample-interval-seconds ${gpu_sample_interval_seconds}"
        ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" "$remote_command" || \
            warn "Could not stop GPU monitor on ${worker}"
    done
}

collect_gpu_metrics() {
    local worker
    local result_gpu_dir="${output_root}/${experiment_name}/gpu_metrics"
    mkdir -p "$result_gpu_dir"
    for worker in "${monitor_workers[@]}"; do
        local worker_dir="${worker//[^A-Za-z0-9._-]/-}"
        rsync -az -e "ssh ${SSH_OPTS:-}" \
            "${ssh_user}@${worker}:${worker_log_dir}/gpu/${experiment_name}/" \
            "${result_gpu_dir}/${worker_dir}/" || \
            warn "Could not collect GPU metrics from ${worker}"
    done
}

cleanup_gpu_monitors() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ "$gpu_monitor_detail" == "raw" ]]; then
        set +e
        stop_gpu_monitors
        collect_gpu_metrics
    fi
    exit "$exit_code"
}

trap cleanup_gpu_monitors EXIT
trap 'exit 130' INT TERM

activate_fedgraph_env
mkdir -p "$hf_home" "$log_dir"
export RAY_ADDRESS="${head_ip}:6379"
export HF_HOME="$hf_home"
export PYTHONPATH="${FEDGRAPH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"$RAY_BIN" status --address="$RAY_ADDRESS"

driver_log="${log_dir}/${experiment_name}_driver.log"
command=(
    "$PYTHON_BIN" benchmark/benchmark_NC_batch_size_convergence.py
    --dataset ogbn-papers100M
    --n-trainer 10
    --num-hops 0
    --use-huggingface
    --hf-artifact-num-hops 1
    --evaluation-split test
    --batch-sizes "$batch_size"
    --rounds "$rounds"
    --local-step "$local_step"
    --iid-betas 10000
    --seeds 42
    --num-layers 2
    --gpu
    --server-device cpu
    --num-gpus-per-trainer 1
    --num-cpus-per-trainer "$num_cpus_per_trainer"
    --experiment-name "$experiment_name"
    --resource-monitor-mode "$resource_monitor_mode"
    --resource-snapshot-interval-rounds "$resource_snapshot_interval_rounds"
    --output-root "$output_root"
)

printf 'Submitting Papers100M 0-hop run to %s\n' "$RAY_ADDRESS"
printf 'Driver log: %s\n' "$driver_log"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$gpu_monitor_detail" == "raw" ]]; then
    printf 'Starting experiment-scoped GPU monitors on workers from %s\n' "$workers_file"
    for_each_worker start_gpu_monitor_on_worker
fi

"${command[@]}" 2>&1 | tee "$driver_log"
