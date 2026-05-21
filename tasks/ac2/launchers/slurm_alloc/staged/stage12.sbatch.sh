#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source tasks/ac2/launchers/common_ac2_env.sh
source tasks/ac2/launchers/slurm_alloc/staged/profile_env.sh

export NANODISCOVER_STAGE_START=sample
export NANODISCOVER_STAGE_STOP=generate
export NANODISCOVER_STAGE_MAX_EPOCHS=1

python main.py
