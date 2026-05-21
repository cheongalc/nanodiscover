#!/usr/bin/env bash
set -euo pipefail

export NANODISCOVER_ALLOC_TASK_NAME=ac1
export NANODISCOVER_ALLOC_COMMON_ENV="tasks/ac1/launchers/common_ac1_env.sh"
export NANODISCOVER_ALLOC_INNER_MODE=single_node
export NANODISCOVER_ALLOC_INNER_ENTRYPOINT=run_one_epoch.sh
export NANODISCOVER_ALLOC_JOB_NAME_SUFFIX=alloc-single-node-1ep

if [[ -n "${NANODISCOVER_ROOT:-}" ]] && [[ -f "${NANODISCOVER_ROOT}/tasks/ac1/launchers/slurm_alloc/common.sh" ]]; then
  # shellcheck disable=SC1090
  source "${NANODISCOVER_ROOT}/tasks/ac1/launchers/slurm_alloc/common.sh"
else
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../common.sh"
fi
