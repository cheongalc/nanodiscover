#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${NANODISCOVER_CHAIN_EPOCH:-}" ]]; then
  echo "NANODISCOVER_CHAIN_EPOCH is required" >&2
  exit 2
fi

idx="$SLURM_ARRAY_TASK_ID"
"${NANODISCOVER_EVAL_PYTHON:?NANODISCOVER_EVAL_PYTHON must point to the task evaluator python}" -m core.evaluator evaluate-shard \
  --task circle_packing \
  --run-dir "$NANODISCOVER_RESUME_DIR" \
  --epoch "$NANODISCOVER_CHAIN_EPOCH" \
  --start "$idx" \
  --stop "$((idx + 1))" \
  --workers 1 \
  --cpu-pack-slot 0 \
  --output "$NANODISCOVER_RESUME_DIR/epoch$(printf "%03d" "$NANODISCOVER_CHAIN_EPOCH")/evaluation_shards/shard_${idx}.json"
