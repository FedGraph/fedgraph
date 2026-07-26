#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

case "${1:-}" in
    "")
        printf 'Run each command in its own terminal:\n'
        print_command kubectl -n "${RAY_NAMESPACE}" port-forward \
            "service/${RAY_HEAD_SERVICE}" 8265:8265
        print_command kubectl -n "${MONITORING_NAMESPACE}" port-forward \
            "service/${PROMETHEUS_SERVICE}" 9090:9090
        print_command kubectl -n "${MONITORING_NAMESPACE}" port-forward \
            "service/${GRAFANA_SERVICE}" 3000:80
        printf '\nLocal addresses:\n'
        printf '  Ray dashboard and Jobs API: %s\n' "${RAY_JOBS_ADDRESS}"
        printf '  Prometheus:                 http://127.0.0.1:9090\n'
        printf '  Grafana:                    http://127.0.0.1:3000\n'
        ;;
    ray)
        require_command kubectl
        require_cluster_context
        exec kubectl -n "${RAY_NAMESPACE}" port-forward \
            "service/${RAY_HEAD_SERVICE}" 8265:8265
        ;;
    prometheus)
        require_command kubectl
        require_cluster_context
        exec kubectl -n "${MONITORING_NAMESPACE}" port-forward \
            "service/${PROMETHEUS_SERVICE}" 9090:9090
        ;;
    grafana)
        require_command kubectl
        require_cluster_context
        exec kubectl -n "${MONITORING_NAMESPACE}" port-forward \
            "service/${GRAFANA_SERVICE}" 3000:80
        ;;
    *) die "Usage: $0 [ray|prometheus|grafana]" ;;
esac
