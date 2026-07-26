#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

mode=plan
case "${1:-}" in
    "") ;;
    --submit) mode=submit ;;
    *) die "Usage: $0 [--submit]" ;;
esac

RAY_CLI="${RAY_CLI:-${HOME}/miniconda3/envs/fedgraph312/bin/ray}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-kuberay_cora_smoke_bs32_full_20r_seed42}"
SUBMISSION_ID="${SUBMISSION_ID:-${EXPERIMENT_NAME}}"

job_command=(
    "${RAY_CLI}" job submit
    --address "${RAY_JOBS_ADDRESS}"
    --submission-id "${SUBMISSION_ID}"
    --
    python /app/benchmark/benchmark_NC_batch_size_convergence.py
    --experiment-name "${EXPERIMENT_NAME}"
    --output-root /results/nc_batch_size_convergence
    --dataset cora
    --num-hops 2
    --batch-sizes 32,-1
    --rounds 20
    --local-step 3
    --n-trainer 5
    --iid-betas 10000
    --seeds 42
    --num-layers 2
    --gpu
    --server-device cpu
    --num-gpus-per-trainer 1
    --num-cpus-per-trainer 1
)

if [[ "${mode}" == plan ]]; then
    printf 'Cora smoke submission plan (no job submitted):\n'
    printf '  Requires scripts/kuberay/50-port-forward.sh ray in another terminal.\n'
    print_command "${job_command[@]}"
    exit 0
fi

[[ -x "${RAY_CLI}" ]] || die \
    "Ray CLI not executable: ${RAY_CLI}. Override it with RAY_CLI=/path/to/ray."
command -v curl >/dev/null 2>&1 || die "Required command not found: curl"
curl --fail --silent --show-error "${RAY_JOBS_ADDRESS}/api/version" >/dev/null || die \
    "Ray Jobs API is not reachable at ${RAY_JOBS_ADDRESS}. Start the Ray port-forward."

"${job_command[@]}"
