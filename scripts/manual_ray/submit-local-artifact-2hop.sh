#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: submit-local-artifact-2hop.sh --head-ip IP --artifact-dir PATH --rank-hosts-file PATH --rounds N --experiment-name NAME [options]

Submit a plaintext 2-hop manifest-artifact benchmark to an existing manual Ray
cluster. The rank-host map hard-pins each trainer to the worker containing its
staged v2/v3 shard.

Options:
  --head-ip IP                 Ray-head private IPv4.
  --artifact-dir PATH          V2/v3 artifact root on the head and workers.
  --rank-hosts-file PATH       Complete '<rank> <private-ipv4>' placement map.
  --rounds N                   Required fixed global-round budget.
  --experiment-name NAME       Required fresh case name (starts/ends alphanumeric).
  --dataset NAME               Dataset label. Default: ogbn-papers100M.
  --n-trainer N                Expected trainer/shard count. Default: 10.
  --batch-size N               Seed nodes per local update. Default: 16.
  --evaluation-batch-size N    Seed nodes per validation/test batch. Default: 512.
  --local-step N               Updates per trainer per global round. Default: 1.
  --optimizer MODE             sgd or adamw. Default: sgd.
  --learning-rate RATE         Initial LR; required for AdamW or cosine decay.
  --lr-scheduler MODE          constant or cosine. Default: constant.
  --min-learning-rate RATE     Cosine LR floor. Default: 0.
  --weight-decay RATE          Local optimizer weight decay. Default: 5e-4.
  --dropout RATE               Papers100M hidden dropout. Default: 0.5.
  --hidden-size N              Papers100M FedGCN hidden width. Default: 256.
  --model-head MODE            gcn or gcn-skip for Papers100M. Default: gcn.
  --first-layer-norm           Enable LayerNorm after the first Papers100M linear layer.
  --no-first-layer-norm        Disable the optional LayerNorm. This is the default.
  --seed N                     Training seed. Default: 42.
  --num-cpus-per-trainer N     Ray CPU request per trainer. Default: 1.
  --num-gpus-per-trainer N     Ray GPU request per trainer. Default: 1.
  --graph-storage-mode MODE    device, cpu, or mmap. Default: mmap.
  --graph-relabel-cache-dir PATH
                               Optional absolute worker-local relabel cache.
  --graph-relabel-chunk-edges N
                               Edges per bounded relabel batch. Default: 1000000.
  --pretrain-aggregation-mode MODE
                               full or chunked. Default: chunked.
  --pretrain-chunk-rows N      Global rows per server chunk. Default: 65536.
  --pretrain-chunk-edges N     Trainer edges per sub-batch. Default: 1000000.
  --pretrain-max-inflight-trainers N
                               Concurrent trainer RPCs. Default: 4.
  --pretrain-output-dir PATH   Worker-local chunked output root. Default:
                               /mnt/fedgraph-nvme/fedgraph-runtime/pretrain.
  --pretrain-feature-cache-dir PATH
                               Reuse a promoted cache on workers; the head needs
                               its manifest. Requires a separate relabel cache dir.
  --feature-compression-mode MODE
                               none or shared-pca. Default: none.
  --feature-compression-rank N Shared-PCA output width. Default: 16.
  --feature-compression-chunk-rows N
                               Projection/statistics rows. Default: 65536.
  --feature-compression-max-inflight-trainers N
                               Concurrent compression RPCs. Default: 4.
  --feature-compression-output-dir PATH
                               Worker-local mmap projection root. Default:
                               /mnt/fedgraph-nvme/fedgraph-runtime/compression.
  --evaluation-split NAME      validation or test. Default: validation.
  --resource-monitor-mode MODE off, manual, prometheus, or hybrid. Default: manual.
  --resource-snapshot-interval-rounds N
                               Snapshot round 1 and every Nth round. Default: 1.
  --gpu-monitor-detail MODE    off or raw. Default: raw.
  --gpu-sample-interval-seconds N
                               nvidia-smi interval. Default: 1.
  --worker-log-dir PATH        Worker GPU-monitor root. Default:
                               /home/ubuntu/fedgraph-logs/local-artifact-2hop.
  --ssh-user USER              Worker SSH user. Default: ubuntu.
  --remote-dir PATH            Worker checkout. Default: /home/ubuntu/fedgraph.
  --output-root PATH           Benchmark result root. Default:
                               benchmark/results/nc_2hop_optimization.
  --dry-run                    Write the benchmark plan without contacting Ray
                               or starting worker monitors. Use a scratch root.
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
evaluation_batch_size=512
local_step=1
optimizer=sgd
learning_rate=""
lr_scheduler=constant
min_learning_rate=0
weight_decay=5e-4
dropout=0.5
hidden_size=""
model_head=gcn
first_layer_norm=false
seed=42
num_cpus_per_trainer=1
num_gpus_per_trainer=1
graph_storage_mode=mmap
graph_relabel_cache_dir=""
graph_relabel_chunk_edges=1000000
pretrain_aggregation_mode=chunked
pretrain_chunk_rows=65536
pretrain_chunk_edges=1000000
pretrain_max_inflight_trainers=4
pretrain_output_dir="/mnt/fedgraph-nvme/fedgraph-runtime/pretrain"
pretrain_feature_cache_dir=""
feature_compression_mode=none
feature_compression_rank=16
feature_compression_chunk_rows=65536
feature_compression_max_inflight_trainers=4
feature_compression_output_dir="/mnt/fedgraph-nvme/fedgraph-runtime/compression"
evaluation_split=validation
resource_monitor_mode=manual
resource_snapshot_interval_rounds=1
gpu_monitor_detail=raw
gpu_sample_interval_seconds=1
worker_log_dir="/home/ubuntu/fedgraph-logs/local-artifact-2hop"
ssh_user=ubuntu
remote_dir="/home/ubuntu/fedgraph"
output_root="benchmark/results/nc_2hop_optimization"
dry_run=false

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
        --evaluation-batch-size) evaluation_batch_size="$2"; shift 2 ;;
        --local-step) local_step="$2"; shift 2 ;;
        --optimizer) optimizer="$2"; shift 2 ;;
        --learning-rate)
            [[ -n "${2:-}" ]] || die "--learning-rate requires a value"
            learning_rate="$2"; shift 2 ;;
        --lr-scheduler) lr_scheduler="$2"; shift 2 ;;
        --min-learning-rate) min_learning_rate="$2"; shift 2 ;;
        --weight-decay) weight_decay="$2"; shift 2 ;;
        --dropout) dropout="$2"; shift 2 ;;
        --hidden-size)
            [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "--hidden-size must be a positive integer"
            hidden_size="$2"; shift 2 ;;
        --model-head)
            [[ -n "${2:-}" ]] || die "--model-head requires a value"
            model_head="$2"; shift 2 ;;
        --first-layer-norm) first_layer_norm=true; shift ;;
        --no-first-layer-norm) first_layer_norm=false; shift ;;
        --seed) seed="$2"; shift 2 ;;
        --num-cpus-per-trainer) num_cpus_per_trainer="$2"; shift 2 ;;
        --num-gpus-per-trainer) num_gpus_per_trainer="$2"; shift 2 ;;
        --graph-storage-mode) graph_storage_mode="$2"; shift 2 ;;
        --graph-relabel-cache-dir) graph_relabel_cache_dir="$2"; shift 2 ;;
        --graph-relabel-chunk-edges) graph_relabel_chunk_edges="$2"; shift 2 ;;
        --pretrain-aggregation-mode) pretrain_aggregation_mode="$2"; shift 2 ;;
        --pretrain-chunk-rows) pretrain_chunk_rows="$2"; shift 2 ;;
        --pretrain-chunk-edges) pretrain_chunk_edges="$2"; shift 2 ;;
        --pretrain-max-inflight-trainers) pretrain_max_inflight_trainers="$2"; shift 2 ;;
        --pretrain-output-dir) pretrain_output_dir="$2"; shift 2 ;;
        --pretrain-feature-cache-dir) pretrain_feature_cache_dir="$2"; shift 2 ;;
        --feature-compression-mode) feature_compression_mode="$2"; shift 2 ;;
        --feature-compression-rank) feature_compression_rank="$2"; shift 2 ;;
        --feature-compression-chunk-rows) feature_compression_chunk_rows="$2"; shift 2 ;;
        --feature-compression-max-inflight-trainers) feature_compression_max_inflight_trainers="$2"; shift 2 ;;
        --feature-compression-output-dir) feature_compression_output_dir="$2"; shift 2 ;;
        --evaluation-split) evaluation_split="$2"; shift 2 ;;
        --resource-monitor-mode) resource_monitor_mode="$2"; shift 2 ;;
        --resource-snapshot-interval-rounds) resource_snapshot_interval_rounds="$2"; shift 2 ;;
        --gpu-monitor-detail) gpu_monitor_detail="$2"; shift 2 ;;
        --gpu-sample-interval-seconds) gpu_sample_interval_seconds="$2"; shift 2 ;;
        --worker-log-dir) worker_log_dir="$2"; shift 2 ;;
        --ssh-user) ssh_user="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
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
require_command python3
[[ "$optimizer" =~ ^(sgd|adamw)$ ]] || die "--optimizer must be sgd or adamw"
[[ "$model_head" =~ ^(gcn|gcn-skip)$ ]] || die "--model-head must be gcn or gcn-skip"
if [[ "$model_head" == "gcn-skip" ]]; then
    [[ "$dataset" == "ogbn-papers100M" ]] || die "--model-head gcn-skip requires ogbn-papers100M"
fi
[[ "$lr_scheduler" =~ ^(constant|cosine)$ ]] || die "--lr-scheduler must be constant or cosine"
if [[ "$optimizer" == "adamw" ]]; then
    [[ "$dataset" == "ogbn-papers100M" ]] || die "AdamW requires ogbn-papers100M"
    [[ -n "$learning_rate" ]] || die "AdamW requires --learning-rate"
fi
if [[ "$lr_scheduler" == "cosine" ]]; then
    [[ "$dataset" == "ogbn-papers100M" ]] || die "Cosine scheduling requires ogbn-papers100M"
    [[ -n "$learning_rate" ]] || die "Cosine scheduling requires --learning-rate"
fi
if [[ -n "$hidden_size" || "$first_layer_norm" == "true" ]]; then
    [[ "$dataset" == "ogbn-papers100M" ]] || die "--hidden-size and --first-layer-norm require ogbn-papers100M"
fi

for value in "$n_trainer" "$rounds" "$batch_size" "$evaluation_batch_size" "$local_step" "$seed" \
    "$num_cpus_per_trainer" "$graph_relabel_chunk_edges" \
    "$pretrain_chunk_rows" "$pretrain_chunk_edges" \
    "$pretrain_max_inflight_trainers" "$feature_compression_rank" \
    "$feature_compression_chunk_rows" \
    "$feature_compression_max_inflight_trainers" \
    "$resource_snapshot_interval_rounds"; do
    [[ "$value" =~ ^[0-9]+$ ]] || die "Expected a non-negative integer, got: $value"
done
(( n_trainer > 0 && rounds > 0 && batch_size > 0 && \
    evaluation_batch_size > 0 && local_step > 0 )) || \
    die "trainer count, rounds, batch sizes, and local step must be positive"
(( graph_relabel_chunk_edges > 0 && pretrain_chunk_rows > 0 )) || \
    die "graph/pretraining chunk limits must be positive"
(( pretrain_chunk_edges > 0 && pretrain_max_inflight_trainers > 0 )) || \
    die "pretraining edge and concurrency limits must be positive"
(( feature_compression_rank > 0 && feature_compression_chunk_rows > 0 )) || \
    die "feature-compression rank and chunk rows must be positive"
(( feature_compression_max_inflight_trainers > 0 )) || \
    die "feature-compression concurrency must be positive"

[[ "$num_gpus_per_trainer" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    die "Expected a non-negative decimal GPU request, got: ${num_gpus_per_trainer}"
[[ "$graph_storage_mode" =~ ^(device|cpu|mmap)$ ]] || \
    die "--graph-storage-mode must be device, cpu, or mmap"
[[ "$pretrain_aggregation_mode" =~ ^(full|chunked)$ ]] || \
    die "--pretrain-aggregation-mode must be full or chunked"
[[ "$feature_compression_mode" =~ ^(none|shared-pca)$ ]] || \
    die "--feature-compression-mode must be none or shared-pca"
[[ "$evaluation_split" =~ ^(validation|test)$ ]] || \
    die "--evaluation-split must be validation or test"
[[ "$resource_monitor_mode" =~ ^(off|manual|prometheus|hybrid)$ ]] || \
    die "Invalid --resource-monitor-mode: ${resource_monitor_mode}"
[[ "$gpu_monitor_detail" =~ ^(off|raw)$ ]] || \
    die "--gpu-monitor-detail must be off or raw"
[[ "$gpu_sample_interval_seconds" =~ ^[1-9][0-9]*$ ]] || \
    die "--gpu-sample-interval-seconds must be a positive integer"
[[ "$experiment_name" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$ ]] || \
    die "--experiment-name must start and end with a letter or number and contain only letters, numbers, dots, underscores, or hyphens"
[[ ! -e "${output_root}/${experiment_name}" ]] || \
    die "Case directory already exists: ${output_root}/${experiment_name}"

for path in "$pretrain_output_dir" "$feature_compression_output_dir"; do
    [[ "$path" == /* ]] || die "Worker-local output paths must be absolute: $path"
done
if [[ -n "$graph_relabel_cache_dir" && "$graph_relabel_cache_dir" != /* ]]; then
    die "--graph-relabel-cache-dir must be absolute"
fi
if [[ -n "$pretrain_feature_cache_dir" ]]; then
    [[ "$pretrain_feature_cache_dir" == /* ]] || die "--pretrain-feature-cache-dir must be absolute"
    [[ -n "$graph_relabel_cache_dir" ]] || die "Feature cache reuse requires --graph-relabel-cache-dir"
    [[ "$graph_storage_mode" == "mmap" && "$pretrain_aggregation_mode" == "chunked" ]] || \
        die "Feature cache reuse requires mmap graph storage and chunked mode"
    require_file "$pretrain_feature_cache_dir/feature-cache.json"
fi
if [[ "$pretrain_aggregation_mode" == "chunked" && "$graph_storage_mode" == "device" ]]; then
    die "chunked pretraining requires graph storage cpu or mmap"
fi
if [[ "$graph_storage_mode" == "mmap" && "$pretrain_aggregation_mode" != "chunked" ]]; then
    die "mmap 2-hop execution requires chunked pretraining"
fi
if [[ "$feature_compression_mode" == "shared-pca" && "$graph_storage_mode" == "device" ]]; then
    die "shared-pca requires graph storage cpu or mmap"
fi

python3 - "$artifact_dir" "$n_trainer" "$pretrain_aggregation_mode" "$learning_rate" <<'PY'
import json
import math
import sys
from pathlib import Path

if sys.argv[4]:
    try:
        rate = float(sys.argv[4])
    except ValueError:
        raise SystemExit("--learning-rate must be a positive finite number") from None
    if not math.isfinite(rate) or rate <= 0:
        raise SystemExit("--learning-rate must be a positive finite number")

manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
expected_trainers = int(sys.argv[2])
aggregation_mode = sys.argv[3]
contract = (manifest.get("artifact_version"), manifest.get("hop_semantics"))
if contract not in {(2, 2), (3, 2)}:
    raise SystemExit(f"expected a version-2/3 2-hop artifact, got {contract}")
if manifest.get("n_trainer") != expected_trainers:
    raise SystemExit(
        f"artifact n_trainer={manifest.get('n_trainer')} != {expected_trainers}"
    )
if contract == (3, 2) and aggregation_mode != "chunked":
    raise SystemExit(
        "compact v3 artifacts require chunked pretraining; use a separately "
        "converted v2 artifact for the full baseline"
    )
PY

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

activate_fedgraph_env
export RAY_ADDRESS="${head_ip}:6379"
export PYTHONPATH="${FEDGRAPH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

command=(
    "$PYTHON_BIN" benchmark/benchmark_NC_batch_size_convergence.py
    --dataset "$dataset"
    --n-trainer "$n_trainer"
    --num-hops 2
    --local-artifact-dir "$artifact_dir"
    --local-artifact-rank-hosts "$rank_hosts_file"
    --batch-sizes "$batch_size"
    --evaluation-batch-size "$evaluation_batch_size"
    --rounds "$rounds"
    --local-step "$local_step"
    --optimizer "$optimizer"
    --lr-scheduler "$lr_scheduler"
    --min-learning-rate "$min_learning_rate"
    --weight-decay "$weight_decay"
    --dropout "$dropout"
    --iid-betas 10000
    --seeds "$seed"
    --num-layers 2
    --model-head "$model_head"
    --gpu
    --server-device cpu
    --num-gpus-per-trainer "$num_gpus_per_trainer"
    --num-cpus-per-trainer "$num_cpus_per_trainer"
    --graph-storage-mode "$graph_storage_mode"
    --graph-relabel-chunk-edges "$graph_relabel_chunk_edges"
    --pretrain-feature-upload-mode indexed
    --pretrain-feature-aggregation-mode "$pretrain_aggregation_mode"
    --pretrain-feature-chunk-rows "$pretrain_chunk_rows"
    --pretrain-feature-chunk-edges "$pretrain_chunk_edges"
    --pretrain-feature-max-inflight-trainers "$pretrain_max_inflight_trainers"
    --feature-compression-mode "$feature_compression_mode"
    --feature-compression-rank "$feature_compression_rank"
    --feature-compression-chunk-rows "$feature_compression_chunk_rows"
    --feature-compression-max-inflight-trainers "$feature_compression_max_inflight_trainers"
    --evaluation-split "$evaluation_split"
    --resource-monitor-mode "$resource_monitor_mode"
    --resource-snapshot-interval-rounds "$resource_snapshot_interval_rounds"
    --experiment-name "$experiment_name"
    --output-root "$output_root"
)

if [[ -n "$learning_rate" ]]; then
    command+=(--learning-rate "$learning_rate")
fi
if [[ -n "$hidden_size" ]]; then
    command+=(--hidden-size "$hidden_size")
fi
if [[ "$first_layer_norm" == "true" ]]; then
    command+=(--first-layer-norm)
fi
if [[ -n "$pretrain_feature_cache_dir" ]]; then
    command+=(--pretrain-feature-cache-dir "$pretrain_feature_cache_dir")
elif [[ "$pretrain_aggregation_mode" == "chunked" ]]; then
    command+=(--pretrain-feature-output-dir "$pretrain_output_dir")
fi
if [[ -z "$pretrain_feature_cache_dir" && "$feature_compression_mode" == "shared-pca" && "$graph_storage_mode" == "mmap" ]]; then
    command+=(--feature-compression-output-dir "$feature_compression_output_dir")
fi
if [[ -n "$graph_relabel_cache_dir" ]]; then
    command+=(--graph-relabel-cache-dir "$graph_relabel_cache_dir")
fi

if [[ "$dry_run" == "true" ]]; then
    exec "${command[@]}" --dry-run
fi

if [[ "$gpu_monitor_detail" == "raw" ]]; then
    require_command awk
    require_command ssh
    require_command rsync
fi

trap cleanup_gpu_monitors EXIT
trap 'exit 130' INT TERM

"$RAY_BIN" status --address="$RAY_ADDRESS"

printf 'Submitting local-artifact 2-hop run to %s\n' "$RAY_ADDRESS"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$gpu_monitor_detail" == "raw" ]]; then
    printf 'Starting GPU monitors for unique hosts in %s\n' "$rank_hosts_file"
    for_each_artifact_worker start_gpu_monitor_on_worker
    (( ${#monitor_workers[@]} > 0 )) || \
        die "No worker addresses found in ${rank_hosts_file}"
fi
"${command[@]}"
