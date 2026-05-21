#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [[ -z "${NANODISCOVER_CHAIN_EPOCH:-}" ]]; then
  echo "NANODISCOVER_CHAIN_EPOCH is required" >&2
  exit 2
fi

TOTAL=$(( NANODISCOVER_SEEDS_PER_EPOCH * NANODISCOVER_ROLLOUTS_PER_SEED ))
python -m core.evaluator merge-shards \
  --run-dir "$NANODISCOVER_RESUME_DIR" \
  --epoch "$NANODISCOVER_CHAIN_EPOCH" \
  --shard-dir "$NANODISCOVER_RESUME_DIR/epoch$(printf "%03d" "$NANODISCOVER_CHAIN_EPOCH")/evaluation_shards" \
  --expected-total "$TOTAL"
