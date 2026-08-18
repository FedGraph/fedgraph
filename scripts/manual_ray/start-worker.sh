#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: start-worker.sh --head-ip IP [options]

Verify one GPU worker, then join the Ray cluster
in the foreground. Run this script inside tmux, either directly on the worker
or through launch-workers.sh from the head.

Options:
  --head-ip IP                 Required Ray-head private IPv4.
  --worker-ip IP               Worker private IPv4. Default: first hostname -I IP.
  --num-cpus N                 CPUs advertised to Ray. Default: 12.
  --num-gpus N                 GPUs advertised to Ray. Default: 1.
  --object-store-memory-gib N  Ray object-store capacity. Default: 8.
  --hf-home PATH               Local Hugging Face cache. Default: /mnt/fedgraph-nvme/hf-cache.
  --min-cache-free-gib N       Required free cache space. Default: 60.
  --ray-tmp-dir PATH           Ray spill/log directory. Default: /mnt/fedgraph-nvme/ray.
  --preflight-only              Verify prerequisites and head connectivity, then exit.
  --help                       Show this message.
EOF
}

head_ip="${HEAD_PRIVATE_IP:-}"
worker_ip="${WORKER_PRIVATE_IP:-$(private_ipv4)}"
num_cpus=12
num_gpus=1
object_store_memory_gib=8
hf_home="${HF_HOME:-/mnt/fedgraph-nvme/hf-cache}"
min_cache_free_gib=60
ray_tmp_dir="${RAY_TMPDIR:-/mnt/fedgraph-nvme/ray}"
preflight_only=false

while (( $# > 0 )); do
    case "$1" in
        --head-ip) head_ip="$2"; shift 2 ;;
        --worker-ip) worker_ip="$2"; shift 2 ;;
        --num-cpus) num_cpus="$2"; shift 2 ;;
        --num-gpus) num_gpus="$2"; shift 2 ;;
        --object-store-memory-gib) object_store_memory_gib="$2"; shift 2 ;;
        --hf-home) hf_home="$2"; shift 2 ;;
        --min-cache-free-gib) min_cache_free_gib="$2"; shift 2 ;;
        --ray-tmp-dir) ray_tmp_dir="$2"; shift 2 ;;
        --preflight-only) preflight_only=true; shift ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$head_ip" ]] || die "--head-ip is required"
require_ipv4 "$head_ip"
require_ipv4 "$worker_ip"
[[ "$num_cpus" =~ ^[0-9]+$ ]] || die "--num-cpus must be an integer"
[[ "$num_gpus" =~ ^[0-9]+$ ]] || die "--num-gpus must be an integer"
[[ "$min_cache_free_gib" =~ ^[0-9]+$ ]] || die "--min-cache-free-gib must be an integer"
object_store_bytes="$(gib_to_bytes "$object_store_memory_gib")"

# An optional local file avoids placing HF_TOKEN in SSH command histories.
hf_env_file="${HF_ENV_FILE:-${HOME}/.config/fedgraph/hf.env}"
if [[ -z "${HF_TOKEN:-}" && -r "$hf_env_file" ]]; then
    # shellcheck disable=SC1090
    source "$hf_env_file"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN is unset; ten concurrent shard downloads may be rate limited"
fi

activate_fedgraph_env
check_worker_runtime
check_hf_cache "$hf_home" "$min_cache_free_gib"
mkdir -p "$ray_tmp_dir"

if [[ "$preflight_only" == "true" ]]; then
    require_command timeout
    timeout 5 bash -c ">/dev/tcp/${head_ip}/6379" 2>/dev/null \
        || die "Cannot reach Ray head ${head_ip}:6379 from this worker"
    printf 'Worker preflight passed: worker=%s head=%s:6379\n' "$worker_ip" "$head_ip"
    exit 0
fi

export HF_HOME="$hf_home"
export PYTHONPATH="${FEDGRAPH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_TMPDIR="$ray_tmp_dir"

printf 'Joining Ray head %s:6379 as worker %s.\n' "$head_ip" "$worker_ip"
"$RAY_BIN" start --address="${head_ip}:6379" --block \
    --node-ip-address="$worker_ip" \
    --num-cpus="$num_cpus" \
    --num-gpus="$num_gpus" \
    --object-store-memory="$object_store_bytes" \
    --temp-dir="$ray_tmp_dir" \
    --disable-usage-stats
