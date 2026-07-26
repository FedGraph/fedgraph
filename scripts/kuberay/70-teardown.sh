#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

mode=plan
case "${1:-}" in
    "") ;;
    --ray) mode=ray ;;
    --cluster) mode=cluster ;;
    *) die "Usage: $0 [--ray|--cluster]" ;;
esac

if [[ "${mode}" == plan ]]; then
    printf 'Teardown plan (no command executed):\n'
    printf '  --ray: delete the RayCluster; Cluster Autoscaler can then scale GPU nodes to zero.\n'
    printf '  --cluster: delete Ray, Helm releases, PVCs, retained results volume, and EKS.\n'
    printf '\nRequired confirmations:\n'
    printf '  FEDGRAPH_ALLOW_RAY_DELETE=yes %q --ray\n' "$0"
    printf '  FEDGRAPH_ALLOW_AWS_DELETE=yes FEDGRAPH_RESULTS_SAVED=yes %q --cluster\n' "$0"
    exit 0
fi

require_command kubectl
require_cluster_context

delete_ray_cluster() {
    kubectl delete raycluster fedgraph-gpu \
        --namespace "${RAY_NAMESPACE}" \
        --ignore-not-found
    if [[ -n "$(kubectl get pods \
        --namespace "${RAY_NAMESPACE}" \
        --selector ray.io/cluster=fedgraph-gpu --output name)" ]]; then
        kubectl wait --for=delete pod \
            --namespace "${RAY_NAMESPACE}" \
            --selector ray.io/cluster=fedgraph-gpu \
            --timeout 15m
    fi
}

if [[ "${mode}" == ray ]]; then
    [[ "${FEDGRAPH_ALLOW_RAY_DELETE:-}" == yes ]] || die \
        "Set FEDGRAPH_ALLOW_RAY_DELETE=yes before deleting the RayCluster."
    delete_ray_cluster
    printf 'RayCluster deleted. Cluster Autoscaler can now return idle GPU nodes to zero.\n'
    exit 0
fi

[[ "${FEDGRAPH_ALLOW_AWS_DELETE:-}" == yes ]] || die \
    "Set FEDGRAPH_ALLOW_AWS_DELETE=yes before deleting AWS resources."
[[ "${FEDGRAPH_RESULTS_SAVED:-}" == yes ]] || die \
    "Set FEDGRAPH_RESULTS_SAVED=yes only after copying all required results out of EKS."
require_command aws
require_command eksctl
require_command helm
require_eks_region_alignment

result_pv="$(kubectl get pvc fedgraph-results \
    --namespace "${RAY_NAMESPACE}" \
    --output jsonpath='{.spec.volumeName}' 2>/dev/null || true)"
if [[ -n "${result_pv}" ]]; then
    kubectl patch pv "${result_pv}" --type merge \
        --patch '{"spec":{"persistentVolumeReclaimPolicy":"Delete"}}'
fi

delete_ray_cluster

for release_namespace in \
    gpu-monitoring:dcgm-exporter \
    "${KUBERAY_NAMESPACE}":kuberay-operator \
    kube-system:cluster-autoscaler \
    "${MONITORING_NAMESPACE}":prometheus; do
    namespace="${release_namespace%%:*}"
    release="${release_namespace#*:}"
    if helm status "${release}" --namespace "${namespace}" >/dev/null 2>&1; then
        helm uninstall "${release}" --namespace "${namespace}" --wait
    fi
done

kubectl delete pvc fedgraph-results \
    --namespace "${RAY_NAMESPACE}" \
    --ignore-not-found
kubectl delete pvc --all \
    --namespace "${MONITORING_NAMESPACE}" \
    --ignore-not-found
if [[ -n "${result_pv}" ]]; then
    kubectl wait --for=delete "pv/${result_pv}" --timeout 10m
fi

eksctl delete cluster --config-file "${EKS_CONFIG}" --wait
printf 'EKS cluster deleted. Public ECR images are not removed by this script.\n'
