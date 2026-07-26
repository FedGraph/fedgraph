#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

mode=plan
case "${1:-}" in
    "") ;;
    --apply) mode=apply ;;
    *) die "Usage: $0 [--apply]" ;;
esac

DASHBOARD_DIR="${REPO_ROOT}/kuberay/config/grafana"

if [[ "${mode}" == plan ]]; then
    printf 'Platform installation plan (no command executed):\n'
    print_command kubectl apply -f "${STORAGE_MANIFEST}"
    print_command helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --version "${PROMETHEUS_CHART_VERSION}" --namespace "${MONITORING_NAMESPACE}" \
        --create-namespace --values "${PROMETHEUS_VALUES}" --wait --timeout 15m
    print_command helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
        --version "${CLUSTER_AUTOSCALER_CHART_VERSION}" --namespace kube-system \
        --values "${CLUSTER_AUTOSCALER_VALUES}" \
        --set autoDiscovery.clusterName="${CLUSTER_NAME}" \
        --set awsRegion="${AWS_REGION}" \
        --wait --timeout 10m
    printf '  kubectl create/apply ConfigMap ray-grafana-dashboards from %q\n' "${DASHBOARD_DIR}"
    print_command helm upgrade --install kuberay-operator kuberay/kuberay-operator \
        --version "${KUBERAY_CHART_VERSION}" --namespace "${KUBERAY_NAMESPACE}" \
        --create-namespace --set nodeSelector.workload=system \
        --set metrics.serviceMonitor.enabled=true \
        --set metrics.serviceMonitor.selector.release=prometheus \
        --wait --timeout 10m
    print_command kubectl apply -f "${RAY_MONITORS}"
    exit 0
fi

require_command kubectl
require_command helm
require_cluster_context

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
helm repo add autoscaler https://kubernetes.github.io/autoscaler --force-update
helm repo update

kubectl apply -f "${STORAGE_MANIFEST}"

helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
    --version "${PROMETHEUS_CHART_VERSION}" \
    --namespace "${MONITORING_NAMESPACE}" \
    --create-namespace \
    --values "${PROMETHEUS_VALUES}" \
    --wait \
    --timeout 15m
kubectl get serviceaccount cluster-autoscaler --namespace kube-system \
    -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' | grep -q .

helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
    --version "${CLUSTER_AUTOSCALER_CHART_VERSION}" \
    --namespace kube-system \
    --values "${CLUSTER_AUTOSCALER_VALUES}" \
    --set autoDiscovery.clusterName="${CLUSTER_NAME}" \
    --set awsRegion="${AWS_REGION}" \
    --wait \
    --timeout 10m

kubectl create configmap ray-grafana-dashboards \
    --namespace "${MONITORING_NAMESPACE}" \
    --from-file "${DASHBOARD_DIR}/default_grafana_dashboard.json" \
    --from-file "${DASHBOARD_DIR}/data_grafana_dashboard.json" \
    --from-file "${DASHBOARD_DIR}/serve_grafana_dashboard.json" \
    --from-file "${DASHBOARD_DIR}/serve_deployment_grafana_dashboard.json" \
    --dry-run=client \
    --output yaml \
    | kubectl apply -f -
kubectl label configmap ray-grafana-dashboards \
    --namespace "${MONITORING_NAMESPACE}" \
    grafana_dashboard=1 \
    --overwrite

helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --version "${KUBERAY_CHART_VERSION}" \
    --namespace "${KUBERAY_NAMESPACE}" \
    --create-namespace \
    --set nodeSelector.workload=system \
    --set metrics.serviceMonitor.enabled=true \
    --set metrics.serviceMonitor.selector.release=prometheus \
    --wait \
    --timeout 10m

kubectl apply -f "${RAY_MONITORS}"

printf '\nPlatform pods:\n'
kubectl get deployment,pods -n kube-system \
    -l app.kubernetes.io/instance=cluster-autoscaler
kubectl get pods -n "${KUBERAY_NAMESPACE}"
kubectl get pods -n "${MONITORING_NAMESPACE}"
kubectl get storageclass,pvc -A
