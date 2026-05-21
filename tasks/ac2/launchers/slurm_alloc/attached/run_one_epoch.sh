#!/usr/bin/env bash
set -euo pipefail

export NANODISCOVER_ALLOC_TASK_NAME=ac2
export NANODISCOVER_ALLOC_COMMON_ENV="tasks/ac2/launchers/common_ac2_env.sh"
export NANODISCOVER_ALLOC_INNER_MODE=slurm_attached
export NANODISCOVER_ALLOC_INNER_ENTRYPOINT=run_one_epoch.sh
export NANODISCOVER_ALLOC_JOB_NAME_SUFFIX=alloc-attached-1ep

if [[ -n "${NANODISCOVER_ROOT:-}" ]] && [[ -f "${NANODISCOVER_ROOT}/tasks/ac2/launchers/slurm_alloc/common.sh" ]]; then
  # shellcheck disable=SC1090
  source "${NANODISCOVER_ROOT}/tasks/ac2/launchers/slurm_alloc/common.sh"
else
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../common.sh"
fi
