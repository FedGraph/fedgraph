#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage: status.sh --head-ip IP

Print Ray's cluster resource view. Run from the head or any node with the same
fedgraph312 environment and private connectivity to the head.
EOF
}

head_ip="${HEAD_PRIVATE_IP:-}"
while (( $# > 0 )); do
    case "$1" in
        --head-ip) head_ip="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -n "$head_ip" ]] || die "--head-ip is required"
require_ipv4 "$head_ip"
activate_fedgraph_env

"$RAY_BIN" status --address="${head_ip}:6379"
