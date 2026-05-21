#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

: "${NANODISCOVER_LOG_ROOT:=./logs}"
export NANODISCOVER_LOG_ROOT
export NANODISCOVER_RESUME_DIR="${NANODISCOVER_RESUME_DIR:-}"

source tasks/ac2/launchers/common_ac2_env.sh

# Local full pipeline, no stage splitting.
export NANODISCOVER_STAGE_START=sample
export NANODISCOVER_STAGE_STOP="${NANODISCOVER_STAGE_STOP:-train}"
export NANODISCOVER_STAGE_MAX_EPOCHS=0

python main.py
