#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

require_command kubectl
require_cluster_context

printf 'Nodes and allocatable GPUs:\n'
kubectl get nodes \
    -o custom-columns='NAME:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,ROLE:.metadata.labels.ray-node-type,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory,GPUS:.status.allocatable.nvidia\.com/gpu'

printf '\nRay resources:\n'
kubectl get raycluster,pods,services,pvc -n "${RAY_NAMESPACE}" -o wide

printf '\nKubeRay operator:\n'
kubectl get pods,services,servicemonitors -n "${KUBERAY_NAMESPACE}" -o wide

printf '\nCluster Autoscaler:\n'
kubectl get deployment,pods -n kube-system \
    -l app.kubernetes.io/instance=cluster-autoscaler -o wide

printf '\nMonitoring:\n'
kubectl get pods,services,pvc,podmonitors,servicemonitors \
    -n "${MONITORING_NAMESPACE}" -o wide

printf '\nGPU monitoring:\n'
kubectl get daemonset,pods,services,servicemonitors -n gpu-monitoring -o wide

printf '\nPort-forward commands and local addresses:\n'
"${SCRIPT_DIR}/50-port-forward.sh"
