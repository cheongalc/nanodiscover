#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$1"
RUN_DIR="$2"
EPOCH="$3"
INDEX_FILE="$4"
SHARD_DIR="$5"
WORKERS="$6"
SHARDS_PER_TASK="$7"
TASK_PARALLELISM="$8"

if [[ -z "${SHARDS_PER_TASK:-}" ]]; then
  SHARDS_PER_TASK=1
fi
if (( SHARDS_PER_TASK < 1 )); then
  SHARDS_PER_TASK=1
fi
if [[ -z "${TASK_PARALLELISM:-}" ]]; then
  TASK_PARALLELISM=1
fi
if (( TASK_PARALLELISM < 1 )); then
  TASK_PARALLELISM=1
fi

START_LINE=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK + 1))
END_LINE=$((START_LINE + SHARDS_PER_TASK - 1))

mapfile -t INDICES < <(sed -n "${START_LINE},${END_LINE}p" "$INDEX_FILE")
if [[ "${#INDICES[@]}" -eq 0 ]]; then
  echo "No shard indices for array task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
declare -a CPU_PACK_FREE_SLOTS=()
declare -A CPU_PACK_PID_TO_SLOT
for (( _cpu_pack_i=0; _cpu_pack_i<TASK_PARALLELISM; _cpu_pack_i++ )); do
  CPU_PACK_FREE_SLOTS+=("$_cpu_pack_i")
done
active=0
pids=()
for IDX in "${INDICES[@]}"; do
  if [[ -z "$IDX" ]]; then
    continue
  fi
  if (( active >= TASK_PARALLELISM )); then
    wait "${pids[0]}"
    _cpu_pack_done_pid="${pids[0]}"
    CPU_PACK_FREE_SLOTS+=("${CPU_PACK_PID_TO_SLOT[$_cpu_pack_done_pid]}")
    unset 'CPU_PACK_PID_TO_SLOT[$_cpu_pack_done_pid]'
    pids=("${pids[@]:1}")
    active=$((active - 1))
  fi
  _cpu_pack_slot="${CPU_PACK_FREE_SLOTS[0]}"
  CPU_PACK_FREE_SLOTS=("${CPU_PACK_FREE_SLOTS[@]:1}")
  "${NANODISCOVER_EVAL_PYTHON:?NANODISCOVER_EVAL_PYTHON must point to the task evaluator python}" -m core.evaluator evaluate-shard \
    --task ac2 \
    --run-dir "$RUN_DIR" \
    --epoch "$EPOCH" \
    --start "$IDX" \
    --stop "$((IDX + 1))" \
    --workers "$WORKERS" \
    --cpu-pack-slot "$_cpu_pack_slot" \
    --output "$SHARD_DIR/shard_${IDX}.json" &
  _cpu_pack_pid=$!
  CPU_PACK_PID_TO_SLOT[$_cpu_pack_pid]="$_cpu_pack_slot"
  pids+=("$_cpu_pack_pid")
  active=$((active + 1))
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
