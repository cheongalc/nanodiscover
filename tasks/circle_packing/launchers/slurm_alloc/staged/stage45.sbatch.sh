#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source tasks/circle_packing/launchers/common_circle_packing_env.sh
source tasks/circle_packing/launchers/slurm_alloc/staged/profile_env.sh

export NANODISCOVER_STAGE_START=archive_update
export NANODISCOVER_STAGE_STOP="${NANODISCOVER_STAGE_STOP:-train}"
export NANODISCOVER_STAGE_MAX_EPOCHS=1

python main.py
