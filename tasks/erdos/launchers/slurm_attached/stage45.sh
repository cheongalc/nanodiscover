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

export NANODISCOVER_STAGE_START=archive_update
export NANODISCOVER_STAGE_STOP="${NANODISCOVER_STAGE_STOP:-train}"
export NANODISCOVER_STAGE_MAX_EPOCHS=1

echo "stage45 start stage_start=${NANODISCOVER_STAGE_START} stage_stop=${NANODISCOVER_STAGE_STOP} trainer_workers=${NANODISCOVER_TRAINER_NUM_WORKERS:-unset} sp_size=${NANODISCOVER_SEQUENCE_PARALLEL_SIZE:-unset} trainer_max_tokens=${NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK:-unset} reference_max_tokens=${NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK:-unset}"

python main.py
