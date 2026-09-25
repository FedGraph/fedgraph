#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: submit-local-artifact-0hop.sh --head-ip IP --artifact-dir PATH --rank-hosts-file PATH --rounds N --experiment-name NAME [options]

Submit a validated 0-hop local-artifact FedAvg benchmark. The optional rank-host
map hard-pins each trainer rank to the worker containing its staged shard.

Options:
  --head-ip IP                 Ray-head private IPv4.
  --artifact-dir PATH          Local artifact root on the head and workers.
  --rank-hosts-file PATH       Complete '<rank> <private-ipv4>' placement map.
  --rounds N                   Required fixed global-round budget.
  --experiment-name NAME       Required fresh result subdirectory name.
  --dataset NAME               Dataset label. Default: ogbn-papers100M.
  --n-trainer N                Expected trainer/shard count. Default: 10.
  --batch-size N               Seed nodes per local update. Default: 16.
  --local-step N               Updates per trainer per global round. Default: 1.
  --num-cpus-per-trainer N     Ray CPU request per trainer. Default: 1.
  --num-gpus-per-trainer N     Ray GPU request per trainer. Default: 1.
  --graph-storage-mode MODE    device, cpu, or mmap. Default: device.
  --graph-relabel-cache-dir PATH
                               Optional absolute worker-local relabel cache path.
  --graph-relabel-chunk-edges N
                               Edges per bounded relabel batch. Default: 1000000.
  --evaluation-split NAME      validation or test. Default: validation.
  --resource-monitor-mode MODE off, manual, prometheus, or hybrid. Default: manual.
  --resource-snapshot-interval-rounds N
                               Snapshot round 1 and every Nth round. Default: 10.
  --gpu-monitor-detail MODE    off or raw. Raw starts one experiment-scoped
                               nvidia-smi monitor per physical worker. Default: off.
  --gpu-sample-interval-seconds N
                               nvidia-smi sampling interval. Default: 1.
  --worker-log-dir PATH        Worker monitor root. Default:
                               /home/ubuntu/fedgraph-logs/local-artifact-0hop.
  --ssh-user USER              Worker SSH user. Default: ubuntu.
  --remote-dir PATH            Worker FedGraph checkout. Default:
                               /home/ubuntu/fedgraph.
  --output-root PATH           Benchmark result root. Default: benchmark/results/nc_batch_size_convergence.
  --help                       Show this message.
EOF
}

head_ip="${HEAD_PRIVATE_IP:-}"
artifact_dir=""
rank_hosts_file=""
rounds=""
experiment_name=""
dataset="ogbn-papers100M"
n_trainer=10
batch_size=16
local_step=1
graph_storage_mode=device
graph_relabel_cache_dir=""
graph_relabel_chunk_edges=1000000
num_cpus_per_trainer=1
num_gpus_per_trainer=1
evaluation_split=validation
resource_monitor_mode=manual
resource_snapshot_interval_rounds=10
gpu_monitor_detail=off
gpu_sample_interval_seconds=1
worker_log_dir="/home/ubuntu/fedgraph-logs/local-artifact-0hop"
ssh_user=ubuntu
remote_dir="/home/ubuntu/fedgraph"
output_root="benchmark/results/nc_batch_size_convergence"

while (( $# > 0 )); do
    case "$1" in
        --head-ip) head_ip="$2"; shift 2 ;;
        --artifact-dir) artifact_dir="$2"; shift 2 ;;
        --rank-hosts-file) rank_hosts_file="$2"; shift 2 ;;
        --rounds) rounds="$2"; shift 2 ;;
        --experiment-name) experiment_name="$2"; shift 2 ;;
        --dataset) dataset="$2"; shift 2 ;;
        --n-trainer) n_trainer="$2"; shift 2 ;;
        --batch-size) batch_size="$2"; shift 2 ;;
        --local-step) local_step="$2"; shift 2 ;;
        --num-cpus-per-trainer) num_cpus_per_trainer="$2"; shift 2 ;;
        --num-gpus-per-trainer) num_gpus_per_trainer="$2"; shift 2 ;;
        --evaluation-split) evaluation_split="$2"; shift 2 ;;
        --graph-storage-mode) graph_storage_mode="$2"; shift 2 ;;
        --graph-relabel-cache-dir) graph_relabel_cache_dir="$2"; shift 2 ;;
        --graph-relabel-chunk-edges) graph_relabel_chunk_edges="$2"; shift 2 ;;
        --resource-monitor-mode) resource_monitor_mode="$2"; shift 2 ;;
        --resource-snapshot-interval-rounds) resource_snapshot_interval_rounds="$2"; shift 2 ;;
        --gpu-monitor-detail) gpu_monitor_detail="$2"; shift 2 ;;
        --gpu-sample-interval-seconds) gpu_sample_interval_seconds="$2"; shift 2 ;;
        --worker-log-dir) worker_log_dir="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$head_ip" ]] || die "--head-ip is required"
[[ -n "$artifact_dir" ]] || die "--artifact-dir is required"
[[ -n "$rank_hosts_file" ]] || die "--rank-hosts-file is required"
[[ -n "$rounds" ]] || die "--rounds is required"
[[ -n "$experiment_name" ]] || die "--experiment-name is required"
require_ipv4 "$head_ip"
require_file "${artifact_dir}/manifest.json"
require_file "$rank_hosts_file"
for value in "$n_trainer" "$rounds" "$batch_size" "$local_step" \
    "$num_cpus_per_trainer" "$resource_snapshot_interval_rounds" \
    "$graph_relabel_chunk_edges"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "Expected an integer, got: $value"
done
(( graph_relabel_chunk_edges > 0 )) || die "--graph-relabel-chunk-edges must be positive"
[[ "$graph_storage_mode" =~ ^(device|cpu|mmap)$ ]] || die "--graph-storage-mode must be device, cpu, or mmap"
[[ "$gpu_sample_interval_seconds" =~ ^[1-9][0-9]*$ ]] || \
    die "--gpu-sample-interval-seconds must be a positive integer"
[[ "$num_gpus_per_trainer" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    die "Expected a positive decimal GPU request, got: ${num_gpus_per_trainer}"
[[ "$evaluation_split" =~ ^(validation|test)$ ]] || \
    die "--evaluation-split must be validation or test"
[[ "$resource_monitor_mode" =~ ^(off|manual|prometheus|hybrid)$ ]] || \
    die "Invalid --resource-monitor-mode: ${resource_monitor_mode}"
[[ "$gpu_monitor_detail" =~ ^(off|raw)$ ]] || \
    die "--gpu-monitor-detail must be off or raw"
[[ "$experiment_name" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "--experiment-name must contain only letters, numbers, dots, underscores, or hyphens"

read -r -a ssh_opts <<< "${SSH_OPTS:-}"
monitor_workers=()

for_each_artifact_worker() {
    local callback="$1"
    local worker
    while IFS= read -r worker; do
        [[ -n "$worker" ]] || continue
        require_ipv4 "$worker"
        "$callback" "$worker"
    done < <(
        awk '
            {
                sub(/#.*/, "")
                if (NF >= 2 && !seen[$2]++) {
                    print $2
                }
            }
        ' "$rank_hosts_file"
    )
}

remote_gpu_monitor_command() {
    local action="$1"
    local remote_command
    printf -v remote_command \
        'cd %q && %q %q --run-id %q --log-dir %q --sample-interval-seconds %q' \
        "$remote_dir" "${remote_dir}/scripts/manual_ray/gpu-monitor.sh" \
        "$action" "$experiment_name" "$worker_log_dir" \
        "$gpu_sample_interval_seconds"
    printf '%s\n' "$remote_command"
}

start_gpu_monitor_on_worker() {
    local worker="$1"
    local remote_command
    remote_command="$(remote_gpu_monitor_command start)"
    ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" "$remote_command" </dev/null
    monitor_workers+=("$worker")
}

stop_gpu_monitors() {
    local worker
    local remote_command
    for worker in "${monitor_workers[@]}"; do
        remote_command="$(remote_gpu_monitor_command stop)"
        ssh "${ssh_opts[@]}" "${ssh_user}@${worker}" "$remote_command" </dev/null || \
            warn "Could not stop GPU monitor on ${worker}"
    done
}

collect_gpu_metrics() {
    local worker
    local worker_dir
    local result_gpu_dir="${output_root}/${experiment_name}/gpu_metrics"
    mkdir -p "$result_gpu_dir"
    for worker in "${monitor_workers[@]}"; do
        worker_dir="${worker//[^A-Za-z0-9._-]/-}"
        rsync -az -e "ssh ${SSH_OPTS:-}" \
            "${ssh_user}@${worker}:${worker_log_dir}/gpu/${experiment_name}/" \
            "${result_gpu_dir}/${worker_dir}/" || \
            warn "Could not collect GPU metrics from ${worker}"
    done
}

cleanup_gpu_monitors() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ "$gpu_monitor_detail" == "raw" ]] && \
        (( ${#monitor_workers[@]} > 0 )); then
        set +e
        stop_gpu_monitors
        collect_gpu_metrics
    fi
    exit "$exit_code"
}

if [[ "$gpu_monitor_detail" == "raw" ]]; then
    require_command awk
    require_command ssh
    require_command rsync
fi

trap cleanup_gpu_monitors EXIT
trap 'exit 130' INT TERM

activate_fedgraph_env
export RAY_ADDRESS="${head_ip}:6379"
export PYTHONPATH="${FEDGRAPH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"$RAY_BIN" status --address="$RAY_ADDRESS"

command=(
    "$PYTHON_BIN" benchmark/benchmark_NC_batch_size_convergence.py
    --dataset "$dataset"
    --n-trainer "$n_trainer"
    --num-hops 0
    --local-artifact-dir "$artifact_dir"
    --local-artifact-rank-hosts "$rank_hosts_file"
    --batch-sizes "$batch_size"
    --rounds "$rounds"
    --local-step "$local_step"
    --iid-betas 10000
    --seeds 42
    --num-layers 2
    --gpu
    --server-device cpu
    --num-gpus-per-trainer "$num_gpus_per_trainer"
    --num-cpus-per-trainer "$num_cpus_per_trainer"
    --graph-storage-mode "$graph_storage_mode"
    --graph-relabel-chunk-edges "$graph_relabel_chunk_edges"
    --evaluation-split "$evaluation_split"
    --resource-monitor-mode "$resource_monitor_mode"
    --resource-snapshot-interval-rounds "$resource_snapshot_interval_rounds"
    --experiment-name "$experiment_name"
    --output-root "$output_root"
)

if [[ -n "$graph_relabel_cache_dir" ]]; then
    command+=(--graph-relabel-cache-dir "$graph_relabel_cache_dir")
fi
printf 'Submitting local-artifact 0-hop run to %s\n' "$RAY_ADDRESS"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$gpu_monitor_detail" == "raw" ]]; then
    printf 'Starting experiment-scoped GPU monitors for unique hosts in %s\n' \
        "$rank_hosts_file"
    for_each_artifact_worker start_gpu_monitor_on_worker
    (( ${#monitor_workers[@]} > 0 )) || die \
        "No worker addresses found in ${rank_hosts_file}"
fi
"${command[@]}"
