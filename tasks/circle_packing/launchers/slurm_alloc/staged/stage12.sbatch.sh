#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source tasks/circle_packing/launchers/common_circle_packing_env.sh
source tasks/circle_packing/launchers/slurm_alloc/staged/profile_env.sh

export NANODISCOVER_STAGE_START=sample
export NANODISCOVER_STAGE_STOP=generate
export NANODISCOVER_STAGE_MAX_EPOCHS=1

python main.py
