#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AWS_REGION="${AWS_REGION:-eu-central-1}"
CLUSTER_NAME="${CLUSTER_NAME:-fedgraph-gpu}"
RAY_NAMESPACE="${RAY_NAMESPACE:-default}"
KUBERAY_NAMESPACE="${KUBERAY_NAMESPACE:-kuberay-system}"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-prometheus-system}"

RAY_VERSION="${RAY_VERSION:-2.56.0}"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.34}"
KUBERAY_CHART_VERSION="${KUBERAY_CHART_VERSION:-1.6.0}"
PROMETHEUS_CHART_VERSION="${PROMETHEUS_CHART_VERSION:-86.0.0}"
CLUSTER_AUTOSCALER_CHART_VERSION="${CLUSTER_AUTOSCALER_CHART_VERSION:-9.53.0}"

FEDGRAPH_IMAGE_REPOSITORY="${FEDGRAPH_IMAGE_REPOSITORY:-public.ecr.aws/i7t1s5i1/fedgraph}"
FEDGRAPH_IMAGE_TAG="${FEDGRAPH_IMAGE_TAG:-gpu-ray2.56.0-py312-cu121}"
FEDGRAPH_IMAGE="${FEDGRAPH_IMAGE:-${FEDGRAPH_IMAGE_REPOSITORY}:${FEDGRAPH_IMAGE_TAG}}"

EKS_CONFIG="${REPO_ROOT}/deploy/kuberay/eks-cluster.yaml"
STORAGE_MANIFEST="${REPO_ROOT}/deploy/kuberay/storage.yaml"
PROMETHEUS_VALUES="${REPO_ROOT}/deploy/kuberay/prometheus-values.yaml"
CLUSTER_AUTOSCALER_VALUES="${REPO_ROOT}/deploy/kuberay/cluster-autoscaler-values.yaml"
RAY_CLUSTER_TEMPLATE="${REPO_ROOT}/deploy/kuberay/ray-cluster.yaml"
RAY_MONITORS="${REPO_ROOT}/deploy/kuberay/monitoring/ray-monitors.yaml"

RAY_HEAD_SERVICE="fedgraph-gpu-head-svc"
PROMETHEUS_SERVICE="prometheus-kube-prometheus-prometheus"
GRAFANA_SERVICE="prometheus-grafana"
RAY_JOBS_ADDRESS="${RAY_JOBS_ADDRESS:-http://127.0.0.1:8265}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

require_aws_create_confirmation() {
    [[ "${FEDGRAPH_ALLOW_AWS_CREATE:-}" == "yes" ]] || die \
        "Set FEDGRAPH_ALLOW_AWS_CREATE=yes before creating billable AWS resources."
}

require_eks_region_alignment() {
    local configured_region
    configured_region="$(awk '$1 == "region:" {gsub(/"/, "", $2); print $2; exit}' "${EKS_CONFIG}")"
    [[ -n "${configured_region}" ]] || die \
        "Unable to read metadata.region from ${EKS_CONFIG}."
    [[ "${configured_region}" == "${AWS_REGION}" ]] || die \
        "AWS_REGION=${AWS_REGION} does not match EKS config region ${configured_region}."
}

require_cluster_context() {
    local context
    context="$(kubectl config current-context 2>/dev/null)" || die \
        "kubectl has no current context. Run aws eks update-kubeconfig first."
    [[ "${context}" == *"${CLUSTER_NAME}"* ]] || die \
        "kubectl context '${context}' does not contain expected cluster '${CLUSTER_NAME}'."
}

render_ray_cluster() {
    local destination="$1"
    [[ "${FEDGRAPH_IMAGE}" =~ ^[A-Za-z0-9./:@_-]+$ ]] || die \
        "FEDGRAPH_IMAGE contains unsupported characters: ${FEDGRAPH_IMAGE}"
    sed "s|__FEDGRAPH_IMAGE__|${FEDGRAPH_IMAGE}|g" \
        "${RAY_CLUSTER_TEMPLATE}" >"${destination}"
    ! grep -q '__FEDGRAPH_IMAGE__' "${destination}" || die \
        "The RayCluster image placeholder was not replaced."
}
