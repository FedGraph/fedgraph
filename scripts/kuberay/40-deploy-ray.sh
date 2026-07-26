#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

mode=plan
case "${1:-}" in
    "") ;;
    --check) mode=check ;;
    --apply) mode=apply ;;
    *) die "Usage: $0 [--check|--apply]" ;;
esac

if [[ "${mode}" == plan ]]; then
    printf 'RayCluster deployment plan (no command executed):\n'
    printf '  template:  %s\n' "${RAY_CLUSTER_TEMPLATE}"
    printf '  image:     %s\n' "${FEDGRAPH_IMAGE}"
    printf '  namespace: %s\n\n' "${RAY_NAMESPACE}"
    print_command "$0" --check
    print_command "$0" --apply
    exit 0
fi

require_command kubectl
require_cluster_context

rendered_ray="$(mktemp)"
trap 'rm -f "${rendered_ray}"' EXIT
render_ray_cluster "${rendered_ray}"

if [[ "${mode}" == check ]]; then
    kubectl apply --dry-run=server -f "${rendered_ray}"
    exit 0
fi

kubectl apply -f "${rendered_ray}"
kubectl wait \
    --namespace "${RAY_NAMESPACE}" \
    --for=condition=Ready pod \
    --selector "ray.io/cluster=fedgraph-gpu,ray.io/node-type=head" \
    --timeout 15m

kubectl get raycluster,pods,services,pvc -n "${RAY_NAMESPACE}" -o wide
