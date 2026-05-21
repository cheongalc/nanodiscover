#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

if [[ -z "${NANODISCOVER_RESUME_DIR:-}" ]]; then
  echo "NANODISCOVER_RESUME_DIR must point to an existing run directory" >&2
  exit 2
fi
mkdir -p "$NANODISCOVER_RESUME_DIR"

source tasks/erdos/launchers/common_erdos_env.sh
source tasks/erdos/launchers/slurm_attached/profile_env.sh

export NANODISCOVER_STAGE_START=sample
export NANODISCOVER_STAGE_STOP=generate
export NANODISCOVER_STAGE_MAX_EPOCHS=1

python main.py
