#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

mode=plan
case "${1:-}" in
    "") ;;
    --apply) mode=apply ;;
    *) die "Usage: $0 [--apply]" ;;
esac

DCGM_CHART_VERSION="${DCGM_CHART_VERSION:-4.8.2}"
GPU_MONITORING_NAMESPACE="${GPU_MONITORING_NAMESPACE:-gpu-monitoring}"
DCGM_VALUES="${REPO_ROOT}/deploy/kuberay/dcgm-values.yaml"

if [[ "${mode}" == plan ]]; then
    printf 'GPU monitoring installation plan (no command executed):\n'
    print_command helm upgrade --install dcgm-exporter gpu-helm-charts/dcgm-exporter \
        --version "${DCGM_CHART_VERSION}" \
        --namespace "${GPU_MONITORING_NAMESPACE}" \
        --create-namespace \
        --values "${DCGM_VALUES}" \
        --wait \
        --timeout 10m
    exit 0
fi

require_command kubectl
require_command helm
require_cluster_context

kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null || die \
    "ServiceMonitor CRD is missing. Install the Prometheus platform first."

helm repo add gpu-helm-charts \
    https://nvidia.github.io/dcgm-exporter/helm-charts \
    --force-update
helm repo update
helm upgrade --install dcgm-exporter gpu-helm-charts/dcgm-exporter \
    --version "${DCGM_CHART_VERSION}" \
    --namespace "${GPU_MONITORING_NAMESPACE}" \
    --create-namespace \
    --values "${DCGM_VALUES}" \
    --wait \
    --timeout 10m

kubectl get daemonset,pods,services,servicemonitors \
    --namespace "${GPU_MONITORING_NAMESPACE}" -o wide
