#!/usr/bin/env bash
set -euo pipefail

export NANODISCOVER_ALLOC_TASK_NAME=circle_packing
export NANODISCOVER_ALLOC_COMMON_ENV="tasks/circle_packing/launchers/common_circle_packing_env.sh"
export NANODISCOVER_ALLOC_INNER_MODE=slurm_attached
export NANODISCOVER_ALLOC_INNER_ENTRYPOINT=run_all.sh
export NANODISCOVER_ALLOC_JOB_NAME_SUFFIX=alloc-attached

if [[ -n "${NANODISCOVER_ROOT:-}" ]] && [[ -f "${NANODISCOVER_ROOT}/tasks/circle_packing/launchers/slurm_alloc/common.sh" ]]; then
  # shellcheck disable=SC1090
  source "${NANODISCOVER_ROOT}/tasks/circle_packing/launchers/slurm_alloc/common.sh"
else
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../common.sh"
fi
