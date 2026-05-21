#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# Set NANODISCOVER_RESUME_DIR in the environment to resume an existing run.
# Leave it unset to auto-create a new run directory under NANODISCOVER_LOG_ROOT.
: "${NANODISCOVER_RESUME_DIR:=}"

: "${NANODISCOVER_LOG_ROOT:=./logs}"
if [[ -z "${NANODISCOVER_RESUME_DIR}" ]]; then
  ts=$(date +%Y%m%d-%H%M%S)
  export NANODISCOVER_RESUME_DIR="${NANODISCOVER_LOG_ROOT}/erdos-${ts}"
fi
mkdir -p "$NANODISCOVER_RESUME_DIR"

source tasks/erdos/launchers/common_erdos_env.sh

echo "run_dir=$NANODISCOVER_RESUME_DIR"

epoch_status=$(python - <<'PY'
from pathlib import Path
import os
import utils

run_dir = Path(os.environ["NANODISCOVER_RESUME_DIR"]).resolve()
stage_stop = os.environ.get("NANODISCOVER_STAGE_STOP", "train")
epoch = utils.resume_epoch(run_dir, stage_stop=stage_stop)
sub = utils.epoch_subdir(run_dir, epoch)

print(f"epoch={epoch}")
print(f"stage_stop={stage_stop}")
print(f"is_complete={int(utils.stage_stop_reached(sub, stage_stop))}")
print(f"has_sample={int(sub.has_sample())}")
print(f"has_generation={int(sub.has_generation())}")
print(f"has_evaluation={int(sub.has_evaluation())}")
print(f"has_training={int(sub.has_training_result())}")
PY
)

eval "$(printf '%s\n' "$epoch_status" | sed 's/^/export /')"
echo "target_epoch=$epoch stage_stop=$stage_stop is_complete=$is_complete has_sample=$has_sample has_generation=$has_generation has_evaluation=$has_evaluation has_training=$has_training"

if [[ "$is_complete" == "1" ]]; then
  echo "epoch $epoch already complete for stage_stop=$stage_stop; nothing to do"
  exit 0
fi

if [[ "$has_evaluation" == "1" ]]; then
  # Evaluation is already done for this epoch; only archive update + train remain.
  tasks/erdos/launchers/slurm_attached/stage45.sh
  exit 0
fi

if [[ "$has_generation" == "1" ]]; then
  # Generation exists but evaluation is missing.
  tasks/erdos/launchers/slurm_attached/stage3.sh
  tasks/erdos/launchers/slurm_attached/stage45.sh
  exit 0
fi

# Fresh epoch: run all three stage bundles.
tasks/erdos/launchers/slurm_attached/stage12.sh
tasks/erdos/launchers/slurm_attached/stage3.sh
tasks/erdos/launchers/slurm_attached/stage45.sh
