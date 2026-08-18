#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: start-head.sh [options]

Start a CPU-only Ray head in the foreground. Run this script inside tmux.

Options:
  --head-ip IP                 Head private IPv4. Default: first hostname -I IP.
  --num-cpus N                 CPUs advertised by the head. Default: 8.
  --object-store-memory-gib N  Ray object-store capacity. Default: 8.
  --ray-tmp-dir PATH           Spill/log directory. Default: $HOME/fedgraph-ray.
  --help                       Show this message.
EOF
}

head_ip="${HEAD_PRIVATE_IP:-$(private_ipv4)}"
num_cpus=8
object_store_memory_gib=8
ray_tmp_dir="${RAY_TMPDIR:-${HOME}/fedgraph-ray}"

while (( $# > 0 )); do
    case "$1" in
        --head-ip) head_ip="$2"; shift 2 ;;
        --num-cpus) num_cpus="$2"; shift 2 ;;
        --object-store-memory-gib) object_store_memory_gib="$2"; shift 2 ;;
        --ray-tmp-dir) ray_tmp_dir="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

require_ipv4 "$head_ip"
[[ "$num_cpus" =~ ^[0-9]+$ ]] || die "--num-cpus must be an integer"
object_store_bytes="$(gib_to_bytes "$object_store_memory_gib")"
activate_fedgraph_env
mkdir -p "$ray_tmp_dir"

printf 'Starting Ray head at %s:6379 with %s CPUs and %s GiB object store.\n' \
    "$head_ip" "$num_cpus" "$object_store_memory_gib"
exec "$RAY_BIN" start --head --block \
    --node-ip-address="$head_ip" \
    --port=6379 \
    --num-cpus="$num_cpus" \
    --num-gpus=0 \
    --dashboard-host=127.0.0.1 \
    --object-store-memory="$object_store_bytes" \
    --temp-dir="$ray_tmp_dir" \
    --disable-usage-stats
