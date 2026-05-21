#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# Set NANODISCOVER_RESUME_DIR in the environment to resume an existing run.
# Leave it unset to auto-create a new run directory under NANODISCOVER_LOG_ROOT.
: "${NANODISCOVER_RESUME_DIR:=}"

: "${NANODISCOVER_LOG_ROOT:=./logs}"
if [[ -z "${NANODISCOVER_RESUME_DIR}" ]]; then
  ts=$(date +%Y%m%d-%H%M%S)
  export NANODISCOVER_RESUME_DIR="${NANODISCOVER_LOG_ROOT}/circle_packing-${ts}"
fi
mkdir -p "$NANODISCOVER_RESUME_DIR"

source tasks/circle_packing/launchers/common_circle_packing_env.sh

current_epoch() {
  python - <<'PY'
from pathlib import Path
import utils
import os
run_dir = Path(os.environ["NANODISCOVER_RESUME_DIR"]).resolve()
stage_stop = os.environ.get("NANODISCOVER_STAGE_STOP", "train")
print(utils.resume_epoch(run_dir, stage_stop=stage_stop))
PY
}

while true; do
  epoch=$(current_epoch)
  if (( epoch >= NANODISCOVER_NUM_EPOCHS )); then
    echo "done run_dir=$NANODISCOVER_RESUME_DIR epoch=$epoch/$NANODISCOVER_NUM_EPOCHS"
    break
  fi
  echo "launching epoch=$epoch run_dir=$NANODISCOVER_RESUME_DIR"
  tasks/circle_packing/launchers/slurm_attached/run_one_epoch.sh
done
