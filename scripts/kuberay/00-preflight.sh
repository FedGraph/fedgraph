#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

missing=()
for command_name in aws docker eksctl kubectl helm; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        missing+=("${command_name}")
    fi
done

printf 'FedGraph KubeRay preflight\n'
printf '  repository:       %s\n' "${REPO_ROOT}"
printf '  AWS region:       %s\n' "${AWS_REGION}"
printf '  EKS cluster:      %s\n' "${CLUSTER_NAME}"
printf '  Kubernetes:       %s\n' "${KUBERNETES_VERSION}"
printf '  Ray:              %s\n' "${RAY_VERSION}"
printf '  KubeRay chart:    %s\n' "${KUBERAY_CHART_VERSION}"
printf '  Prometheus chart: %s\n' "${PROMETHEUS_CHART_VERSION}"
printf '  Autoscaler chart: %s\n' "${CLUSTER_AUTOSCALER_CHART_VERSION}"
printf '  FedGraph image:   %s\n' "${FEDGRAPH_IMAGE}"

if ((${#missing[@]})); then
    printf '\nMissing CLI tools: %s\n' "${missing[*]}" >&2
    printf 'Install them before the first apply; see docs/kuberay_gpu_first_run.md.\n' >&2
    exit 1
fi

require_eks_region_alignment

printf '\nChecking Docker...\n'
docker version >/dev/null
docker buildx version
docker manifest inspect "rayproject/ray:${RAY_VERSION}-py312-cu121" >/dev/null

printf '\nChecking AWS identity and basic read permissions...\n'
aws configure list
aws sts get-caller-identity --query '{Account:Account,Arn:Arn}' --output table
aws eks describe-cluster-versions \
    --region "${AWS_REGION}" \
    --cluster-versions "${KUBERNETES_VERSION}" \
    --max-results 1 >/dev/null
aws ec2 describe-instance-type-offerings \
    --region "${AWS_REGION}" \
    --location-type region \
    --filters Name=instance-type,Values=g4dn.xlarge \
    --max-results 5 >/dev/null

if [[ "${FEDGRAPH_IMAGE_REPOSITORY}" == public.ecr.aws/* ]]; then
    public_repository="${FEDGRAPH_IMAGE_REPOSITORY#public.ecr.aws/*/}"
    aws ecr-public describe-repositories \
        --region us-east-1 \
        --repository-names "${public_repository}" >/dev/null
elif [[ "${FEDGRAPH_IMAGE_REPOSITORY}" =~ ^[0-9]+\.dkr\.ecr\.([^.]+)\.amazonaws\.com/(.+)$ ]]; then
    private_region="${BASH_REMATCH[1]}"
    private_repository="${BASH_REMATCH[2]}"
    aws ecr describe-repositories \
        --region "${private_region}" \
        --repository-names "${private_repository}" >/dev/null
fi

printf '\nChecking local manifests...\n'
eksctl create cluster --config-file "${EKS_CONFIG}" --dry-run >/dev/null
kubectl apply --dry-run=client --validate=false -f "${STORAGE_MANIFEST}" >/dev/null

rendered_ray="$(mktemp)"
trap 'rm -f "${rendered_ray}"' EXIT
render_ray_cluster "${rendered_ray}"
kubectl apply --dry-run=client --validate=false -f "${rendered_ray}" >/dev/null

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update >/dev/null
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update >/dev/null
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts --force-update >/dev/null
helm repo add autoscaler https://kubernetes.github.io/autoscaler --force-update >/dev/null
helm repo update >/dev/null
helm template kuberay-operator kuberay/kuberay-operator \
    --version "${KUBERAY_CHART_VERSION}" \
    --namespace "${KUBERAY_NAMESPACE}" \
    --set nodeSelector.workload=system \
    --set metrics.serviceMonitor.enabled=true \
    --set metrics.serviceMonitor.selector.release=prometheus >/dev/null
helm template prometheus prometheus-community/kube-prometheus-stack \
    --version "${PROMETHEUS_CHART_VERSION}" \
    --namespace "${MONITORING_NAMESPACE}" \
    --values "${PROMETHEUS_VALUES}" >/dev/null
helm template cluster-autoscaler autoscaler/cluster-autoscaler \
    --version "${CLUSTER_AUTOSCALER_CHART_VERSION}" \
    --namespace kube-system \
    --values "${CLUSTER_AUTOSCALER_VALUES}" \
    --set autoDiscovery.clusterName="${CLUSTER_NAME}" \
    --set awsRegion="${AWS_REGION}" >/dev/null
helm template dcgm-exporter gpu-helm-charts/dcgm-exporter \
    --version 4.8.2 \
    --namespace gpu-monitoring \
    --values "${REPO_ROOT}/deploy/kuberay/dcgm-values.yaml" >/dev/null

printf '\nPreflight passed. No AWS or Kubernetes resources were changed.\n'
