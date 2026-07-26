#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

mode=plan
case "${1:-}" in
    "") ;;
    --apply) mode=apply ;;
    *) die "Usage: $0 [--apply]" ;;
esac

require_eks_region_alignment

if [[ "${mode}" == plan ]]; then
    printf 'EKS creation plan (no command executed):\n'
    print_command eksctl create cluster --config-file "${EKS_CONFIG}" --dry-run
    printf '\nActual creation requires:\n'
    printf '  export FEDGRAPH_ALLOW_AWS_CREATE=yes\n'
    print_command "$0" --apply
    exit 0
fi

require_aws_create_confirmation
require_command aws
require_command eksctl
require_command kubectl

aws sts get-caller-identity >/dev/null

describe_error="$(mktemp)"
trap 'rm -f "${describe_error}"' EXIT
if aws eks describe-cluster \
    --region "${AWS_REGION}" \
    --name "${CLUSTER_NAME}" >/dev/null 2>"${describe_error}"; then
    printf 'EKS cluster already exists; refreshing kubeconfig.\n'
elif grep -q 'ResourceNotFoundException' "${describe_error}"; then
    eksctl create cluster --config-file "${EKS_CONFIG}"
else
    printf 'Unable to determine whether the EKS cluster exists:\n' >&2
    sed 's/^/  /' "${describe_error}" >&2
    exit 1
fi

aws eks update-kubeconfig --region "${AWS_REGION}" --name "${CLUSTER_NAME}"
require_cluster_context
kubectl get nodes -o wide
