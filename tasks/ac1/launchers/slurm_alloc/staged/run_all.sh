#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# Set NANODISCOVER_RESUME_DIR in the environment to resume an existing run.
# Leave it unset to auto-create a new run directory under NANODISCOVER_LOG_ROOT.
: "${NANODISCOVER_RESUME_DIR:=}"

: "${NANODISCOVER_LOG_ROOT:=./logs}"
if [[ -z "${NANODISCOVER_RESUME_DIR}" ]]; then
  ts=$(date +%Y%m%d-%H%M%S)
  export NANODISCOVER_RESUME_DIR="${NANODISCOVER_LOG_ROOT}/ac1-${ts}"
fi
mkdir -p "$NANODISCOVER_RESUME_DIR"

source tasks/ac1/launchers/common_ac1_env.sh

start_epoch=$(python - <<'PY'
from pathlib import Path
import utils
import os
run_dir = Path(os.environ["NANODISCOVER_RESUME_DIR"]).resolve()
stage_stop = os.environ.get("NANODISCOVER_STAGE_STOP", "train")
print(utils.resume_epoch(run_dir, stage_stop=stage_stop))
PY
)

if (( start_epoch >= NANODISCOVER_NUM_EPOCHS )); then
  echo "nothing to submit: start_epoch=$start_epoch num_epochs=$NANODISCOVER_NUM_EPOCHS"
  exit 0
fi

prev_dep=""
for (( epoch = start_epoch; epoch < NANODISCOVER_NUM_EPOCHS; epoch++ )); do
  echo "submitting epoch=$epoch"
  out=$(NANODISCOVER_CHAIN_EPOCH="$epoch" NANODISCOVER_CHAIN_DEPENDENCY="$prev_dep" tasks/ac1/launchers/slurm_alloc/staged/run_one_epoch.sh)
  echo "$out"
  prev_dep=$(printf '%s\n' "$out" | awk -F= '/^stage45_job=/{print $2}' | tail -n 1)
  if [[ -z "$prev_dep" ]]; then
    echo "failed to parse stage45_job for epoch=$epoch" >&2
    exit 2
  fi
done

echo "submitted full chain run_dir=$NANODISCOVER_RESUME_DIR final_job=$prev_dep"
