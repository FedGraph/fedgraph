#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

mode=plan
case "${1:-}" in
    "") ;;
    --build) mode=build ;;
    --push) mode=push ;;
    *) die "Usage: $0 [--build|--push]" ;;
esac

build_command=(
    docker buildx build
    --platform linux/amd64
    --file "${REPO_ROOT}/Dockerfile.gpu"
    --tag "${FEDGRAPH_IMAGE}"
)

registry_login() {
    local registry
    if [[ "${FEDGRAPH_IMAGE_REPOSITORY}" == public.ecr.aws/* ]]; then
        aws ecr-public get-login-password --region us-east-1 \
            | docker login --username AWS --password-stdin public.ecr.aws
    elif [[ "${FEDGRAPH_IMAGE_REPOSITORY}" =~ ^[0-9]+\.dkr\.ecr\.([^.]+)\.amazonaws\.com/ ]]; then
        registry="${FEDGRAPH_IMAGE_REPOSITORY%%/*}"
        aws ecr get-login-password --region "${BASH_REMATCH[1]}" \
            | docker login --username AWS --password-stdin "${registry}"
    else
        printf 'No automatic registry login is defined for %s; using Docker credentials.\n' \
            "${FEDGRAPH_IMAGE_REPOSITORY}"
    fi
}

if [[ "${mode}" == plan ]]; then
    printf 'Image build plan (no command executed):\n'
    print_command "${build_command[@]}" --load "${REPO_ROOT}"
    printf '\nPush plan for repository %s:\n' "${FEDGRAPH_IMAGE_REPOSITORY}"
    if [[ "${FEDGRAPH_IMAGE_REPOSITORY}" == public.ecr.aws/* ]]; then
        print_command aws ecr-public get-login-password --region us-east-1
        print_command docker login --username AWS --password-stdin public.ecr.aws
    elif [[ "${FEDGRAPH_IMAGE_REPOSITORY}" =~ ^[0-9]+\.dkr\.ecr\.([^.]+)\.amazonaws\.com/ ]]; then
        print_command aws ecr get-login-password --region "${BASH_REMATCH[1]}"
        print_command docker login --username AWS --password-stdin \
            "${FEDGRAPH_IMAGE_REPOSITORY%%/*}"
    else
        printf '  use existing Docker registry credentials\n'
    fi
    print_command "${build_command[@]}" --push "${REPO_ROOT}"
    exit 0
fi

require_command docker
docker buildx version >/dev/null

if [[ "${mode}" == push ]]; then
    require_command aws
    registry_login
    "${build_command[@]}" --push "${REPO_ROOT}"
else
    "${build_command[@]}" --load "${REPO_ROOT}"
fi

printf 'Image ready: %s\n' "${FEDGRAPH_IMAGE}"
