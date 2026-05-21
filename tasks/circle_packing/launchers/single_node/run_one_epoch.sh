#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

: "${NANODISCOVER_LOG_ROOT:=./logs}"
export NANODISCOVER_LOG_ROOT
export NANODISCOVER_RESUME_DIR="${NANODISCOVER_RESUME_DIR:-}"

source tasks/circle_packing/launchers/common_circle_packing_env.sh

# Local full pipeline, but stop after exactly one epoch.
export NANODISCOVER_STAGE_START=sample
export NANODISCOVER_STAGE_STOP="${NANODISCOVER_STAGE_STOP:-train}"
export NANODISCOVER_STAGE_MAX_EPOCHS=1

python main.py
