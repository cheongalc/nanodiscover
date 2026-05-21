#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${NANODISCOVER_RESUME_DIR:-}" ]]; then
  echo "NANODISCOVER_RESUME_DIR must point to an existing run directory" >&2
  exit 2
fi

TASK_NAME="circle_packing"
source tasks/circle_packing/launchers/common_circle_packing_env.sh
source tasks/circle_packing/launchers/slurm_attached/profile_env.sh

RUN_DIR="$NANODISCOVER_RESUME_DIR"
STAGE3_ABORTED=0
RUN_LOG="$RUN_DIR/log.txt"
declare -a ACTIVE_REMOTE_JOB_IDS=()
ACTIVE_REMOTE_JOB_IDS_FILE=""
mkdir -p "$RUN_DIR"
touch "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

on_interrupt() {
  STAGE3_ABORTED=1
  echo "stage3 interrupt received; stopping outstanding work"
  exit 130
}

cleanup_stage3() {
  local exit_code=$?
  local job_id
  local -A cleanup_job_ids=()
  trap - EXIT INT TERM HUP

  if (( STAGE3_ABORTED )) || (( exit_code != 0 )); then
    mapfile -t local_jobs < <(jobs -pr || true)
    if [[ "${#local_jobs[@]}" -gt 0 ]]; then
      echo "stage3 cleanup killing_local_jobs=${#local_jobs[@]}"
      kill "${local_jobs[@]}" 2>/dev/null || true
    fi
    for job_id in "${ACTIVE_REMOTE_JOB_IDS[@]}"; do
      if [[ -n "$job_id" ]]; then
        cleanup_job_ids["$job_id"]=1
      fi
    done
    if [[ -n "${ACTIVE_REMOTE_JOB_IDS_FILE:-}" ]] && [[ -f "$ACTIVE_REMOTE_JOB_IDS_FILE" ]]; then
      while IFS= read -r job_id; do
        if [[ -n "$job_id" ]]; then
          cleanup_job_ids["$job_id"]=1
        fi
      done < "$ACTIVE_REMOTE_JOB_IDS_FILE"
    fi
    if (( ${#cleanup_job_ids[@]} > 0 )); then
      echo "stage3 cleanup scancel_remote_jobs=${#cleanup_job_ids[@]}"
      for job_id in "${!cleanup_job_ids[@]}"; do
        scancel "$job_id" >/dev/null 2>&1 || true
      done
    fi
    wait || true
  fi
  if [[ -n "${ACTIVE_REMOTE_JOB_IDS_FILE:-}" ]]; then
    rm -f "$ACTIVE_REMOTE_JOB_IDS_FILE"
  fi
}

trap on_interrupt INT TERM HUP
trap cleanup_stage3 EXIT

EPOCH=$(python - <<'PY'
from pathlib import Path
import utils
import os
run_dir = Path(os.environ["NANODISCOVER_RESUME_DIR"]).resolve()
last = -1
for p in sorted(run_dir.glob("epoch*")):
    if not p.is_dir():
        continue
    name = p.name
    if not name.startswith("epoch"):
        continue
    epoch = int(name.replace("epoch", ""))
    sub = utils.epoch_subdir(run_dir, epoch)
    if sub.has_generation() and not sub.has_evaluation():
        print(epoch)
        break
else:
    raise SystemExit("No epoch with generation.json and missing evaluation.json")
PY
)

TOTAL=$(python - <<PY
from pathlib import Path
import utils
run_dir = Path("$RUN_DIR").resolve()
epoch = int("$EPOCH")
sub = utils.epoch_subdir(run_dir, epoch)
print(len(utils.load_generations(sub)))
PY
)

if [[ "$TOTAL" -le 0 ]]; then
  echo "No generations to evaluate for epoch $EPOCH" >&2
  exit 2
fi

SHARD_DIR="$RUN_DIR/epoch$(printf "%03d" "$EPOCH")/evaluation_shards"
mkdir -p "$SHARD_DIR"
export SHARD_DIR TOTAL
EVALUATOR_TMPDIR="${NANODISCOVER_EVALUATOR_TMPDIR:-/tmp/${USER:-nanodiscover}}"
mkdir -p "$EVALUATOR_TMPDIR"
echo "stage3 evaluator_tmpdir=$EVALUATOR_TMPDIR"

LOCAL_LIMIT=$(( SLURM_ATTACHED_LOCAL_INITIAL_SHARDS ))
if (( LOCAL_LIMIT > TOTAL )); then
  LOCAL_LIMIT=$TOTAL
fi

LOCAL_PARALLEL=$(( SLURM_ATTACHED_LOCAL_PARALLEL_JOBS ))
if (( LOCAL_PARALLEL < 0 )); then
  LOCAL_PARALLEL=1
fi

declare -a LOCAL_CPU_PACK_FREE_SLOTS=()
declare -A LOCAL_CPU_PACK_PID_SLOT=()
local_cpu_pack_slots_reset() {
  local i
  LOCAL_CPU_PACK_FREE_SLOTS=()
  for ((i=0; i<LOCAL_PARALLEL; i++)); do
    LOCAL_CPU_PACK_FREE_SLOTS+=("$i")
  done
}
local_cpu_pack_slots_reset

REMOTE_SHARDS_PER_TASK=$(( SLURM_ATTACHED_EVAL_SHARD_SIZE ))
if (( REMOTE_SHARDS_PER_TASK < 1 )); then
  REMOTE_SHARDS_PER_TASK=1
fi

REMOTE_MAX_ARRAY_TASKS=$(( SLURM_ATTACHED_REMOTE_MAX_ARRAY_TASKS ))
if (( REMOTE_MAX_ARRAY_TASKS < 1 )); then
  REMOTE_MAX_ARRAY_TASKS=1
fi

scale_mem_spec() {
  local spec="$1"
  local factor="$2"
  if [[ "$spec" =~ ^([0-9]+)([KkMmGgTt])$ ]]; then
    local amount="${BASH_REMATCH[1]}"
    local unit="${BASH_REMATCH[2]}"
    echo "$((amount * factor))${unit}"
    return 0
  fi
  if [[ "$spec" =~ ^([0-9]+)$ ]]; then
    local amount="${BASH_REMATCH[1]}"
    echo "$((amount * factor))"
    return 0
  fi
  echo "$spec"
  return 0
}

echo "stage3 start epoch=$EPOCH total_rollouts=$TOTAL local_initial_shards=$LOCAL_LIMIT local_parallel_jobs=$LOCAL_PARALLEL remote_partitions=$SLURM_ATTACHED_REMOTE_PARTITIONS remote_qos=${SLURM_ATTACHED_REMOTE_QOS:-none} remote_shards_per_task=$REMOTE_SHARDS_PER_TASK remote_parallel_evals_cfg=$SLURM_ATTACHED_REMOTE_PARALLEL_EVALS_PER_TASK remote_max_array_tasks=$REMOTE_MAX_ARRAY_TASKS"

QUEUE_DIR="$SHARD_DIR/stage3_state"
REMOTE_WAVE_DIR="$QUEUE_DIR/remote_waves"
mkdir -p "$QUEUE_DIR" "$REMOTE_WAVE_DIR"
ACTIVE_REMOTE_JOB_IDS_FILE="$QUEUE_DIR/active_remote_job_ids.txt"
: > "$ACTIVE_REMOTE_JOB_IDS_FILE"
PENDING_FILE="$QUEUE_DIR/pending_indices.txt"
export PENDING_FILE

resume_counts=$(python - <<'PY'
from pathlib import Path
import os

from core.evaluator import scan_shard_status

pending_file = Path(os.environ["PENDING_FILE"])
status = scan_shard_status(
    Path(os.environ["SHARD_DIR"]),
    expected_total=int(os.environ["TOTAL"]),
    delete_invalid=True,
)
pending = [str(idx) for idx in status["pending_indices"]]
pending_file.write_text(("\n".join(pending) + "\n") if pending else "", encoding="utf-8")
print(f"{status['complete_count']} {status['invalid_count']} {len(status['pending_indices'])}")
PY
)

read -r COMPLETE_SHARDS INVALID_SHARDS PENDING_SHARDS <<< "$resume_counts"
echo "stage3 resume complete_shards=$COMPLETE_SHARDS invalid_shards=$INVALID_SHARDS pending_shards=$PENDING_SHARDS"

mapfile -t pending_indices < "$PENDING_FILE"
filtered_pending=()
for idx in "${pending_indices[@]}"; do
  if [[ "$idx" =~ ^[0-9]+$ ]]; then
    filtered_pending+=("$idx")
  fi
done
pending_indices=("${filtered_pending[@]}")

expand_array_ids() {
  local raw="$1"
  raw="${raw//[[:space:]]/}"
  raw="${raw#[}"
  raw="${raw%]}"
  if [[ -z "$raw" ]]; then
    return 0
  fi
  local part
  IFS=',' read -ra parts <<< "$raw"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^[0-9]+$ ]]; then
      echo "$part"
    elif [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local start="${BASH_REMATCH[1]}"
      local end="${BASH_REMATCH[2]}"
      if (( end >= start )); then
        for ((i=start; i<=end; i++)); do
          echo "$i"
        done
      else
        for ((i=start; i>=end; i--)); do
          echo "$i"
        done
      fi
    fi
  done
}

declare -a PENDING_QUEUE=()
declare -A PENDING_QUEUED_INDEX=()
PENDING_QUEUE_HEAD=0

declare -a LOCAL_RUNNING_PIDS=()
declare -A LOCAL_RUNNING_BY_PID=()
declare -A LOCAL_RUNNING_BY_INDEX=()
LOCAL_LAUNCHED=0
LOCAL_COMPLETED=0

declare -A REMOTE_BAD_NODES=()
declare -A SHARD_FORCE_SINGLE_REMOTE=()
declare -A SHARD_FINALIZED_REASON=()

declare -A REMOTE_JOB_INDEX_FILE=()
declare -A REMOTE_JOB_SHARDS_PER_TASK=()
declare -A REMOTE_JOB_PARALLEL_EVALS=()
declare -A REMOTE_JOB_ARRAY_TASK_COUNT=()
declare -A REMOTE_JOB_KIND=()
declare -A REMOTE_JOB_IDLE_LOOPS=()
declare -A REMOTE_JOB_PENDING_COUNT=()
declare -A REMOTE_JOB_RUNNING_COUNT=()
declare -A REMOTE_JOB_ACTIVE_COUNT=()
declare -A REMOTE_JOB_HANDLED_COUNT=()

declare -A REMOTE_TASK_ACTION=()
declare -A REMOTE_INDEX_OWNER=()

PENDING_DEQUEUED_INDEX=""
TAKEN_PENDING_INDICES=()
QUEUE_SLICE_ADDED=0
QUEUE_SLICE_TOTAL=0
WAVE_SEQUENCE=0
FINALIZED_SYNTHETIC_FAILURES=0
RECLAIMED_PENDING_THIS_LOOP=0
REMOTE_STATUS_LAST=""
MONITOR_SLEEP_SECONDS="${SLURM_ATTACHED_MONITOR_SLEEP_SECONDS:-2}"
if (( MONITOR_SLEEP_SECONDS < 1 )); then
  MONITOR_SLEEP_SECONDS=1
fi

pending_queue_depth() {
  echo "${#PENDING_QUEUED_INDEX[@]}"
}

local_slots_free() {
  local free=$(( LOCAL_PARALLEL - ${#LOCAL_RUNNING_PIDS[@]} ))
  if (( free < 0 )); then
    free=0
  fi
  echo "$free"
}

local_reserved_pending_floor() {
  local reserve=$(( LOCAL_LIMIT - LOCAL_LAUNCHED ))
  local depth
  if (( reserve < 0 )); then
    reserve=0
  fi
  depth=$(pending_queue_depth)
  if (( reserve > depth )); then
    reserve=$depth
  fi
  echo "$reserve"
}

shard_file_is_complete() {
  local idx="$1"
  local shard_path="$SHARD_DIR/shard_${idx}.json"
  if [[ ! -f "$shard_path" ]]; then
    return 1
  fi
  SHARD_VALIDATE_INDEX="$idx" SHARD_VALIDATE_PATH="$shard_path" python - <<'PY'
from pathlib import Path
import os

from core.evaluator import load_single_rollout_shard

try:
    load_single_rollout_shard(
        Path(os.environ["SHARD_VALIDATE_PATH"]),
        expected_index=int(os.environ["SHARD_VALIDATE_INDEX"]),
    )
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

queue_pending_index() {
  local idx="$1"
  if [[ ! "$idx" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  if [[ -f "$SHARD_DIR/shard_${idx}.json" ]]; then
    if shard_file_is_complete "$idx"; then
      return 1
    fi
    echo "stage3 shard invalid idx=$idx removing_for_retry=1"
    rm -f "$SHARD_DIR/shard_${idx}.json"
  fi
  if [[ -n "${PENDING_QUEUED_INDEX[$idx]:-}" ]]; then
    return 1
  fi
  if [[ -n "${LOCAL_RUNNING_BY_INDEX[$idx]:-}" ]]; then
    return 1
  fi
  if [[ -n "${REMOTE_INDEX_OWNER[$idx]:-}" ]]; then
    return 1
  fi
  PENDING_QUEUE+=("$idx")
  PENDING_QUEUED_INDEX["$idx"]=1
  return 0
}

dequeue_pending_index() {
  local idx
  PENDING_DEQUEUED_INDEX=""
  while (( PENDING_QUEUE_HEAD < ${#PENDING_QUEUE[@]} )); do
    idx="${PENDING_QUEUE[$PENDING_QUEUE_HEAD]}"
    PENDING_QUEUE_HEAD=$((PENDING_QUEUE_HEAD + 1))
    if [[ -z "${PENDING_QUEUED_INDEX[$idx]:-}" ]]; then
      continue
    fi
    unset 'PENDING_QUEUED_INDEX[$idx]'
    PENDING_DEQUEUED_INDEX="$idx"
    return 0
  done
  return 1
}

count_pending_indices_matching() {
  local mode="$1"
  local reserve_floor="$2"
  local reserved_left="$reserve_floor"
  local count=0
  local idx
  local scan
  for ((scan=PENDING_QUEUE_HEAD; scan<${#PENDING_QUEUE[@]}; scan++)); do
    idx="${PENDING_QUEUE[$scan]}"
    if [[ -z "${PENDING_QUEUED_INDEX[$idx]:-}" ]]; then
      continue
    fi
    if (( reserved_left > 0 )); then
      reserved_left=$((reserved_left - 1))
      continue
    fi
    case "$mode" in
      single)
        if [[ -z "${SHARD_FORCE_SINGLE_REMOTE[$idx]:-}" ]]; then
          continue
        fi
        ;;
      normal)
        if [[ -n "${SHARD_FORCE_SINGLE_REMOTE[$idx]:-}" ]]; then
          continue
        fi
        ;;
      any)
        ;;
      *)
        return 1
        ;;
    esac
    count=$((count + 1))
  done
  echo "$count"
}

take_pending_indices_for_remote_wave() {
  local limit="$1"
  local mode="$2"
  local reserve_floor="$3"
  local reserved_left="$reserve_floor"
  local idx
  local scan
  TAKEN_PENDING_INDICES=()
  if (( limit < 1 )); then
    return 0
  fi
  for ((scan=PENDING_QUEUE_HEAD; scan<${#PENDING_QUEUE[@]}; scan++)); do
    if (( ${#TAKEN_PENDING_INDICES[@]} >= limit )); then
      break
    fi
    idx="${PENDING_QUEUE[$scan]}"
    if [[ -z "${PENDING_QUEUED_INDEX[$idx]:-}" ]]; then
      continue
    fi
    if (( reserved_left > 0 )); then
      reserved_left=$((reserved_left - 1))
      continue
    fi
    case "$mode" in
      single)
        if [[ -z "${SHARD_FORCE_SINGLE_REMOTE[$idx]:-}" ]]; then
          continue
        fi
        ;;
      normal)
        if [[ -n "${SHARD_FORCE_SINGLE_REMOTE[$idx]:-}" ]]; then
          continue
        fi
        ;;
      any)
        ;;
      *)
        return 1
        ;;
    esac
    unset 'PENDING_QUEUED_INDEX[$idx]'
    TAKEN_PENDING_INDICES+=("$idx")
  done
}

launch_local_from_pending() {
  local idx
  local pid
  while (( $(local_slots_free) > 0 )); do
    if ! dequeue_pending_index; then
      break
    fi
    idx="$PENDING_DEQUEUED_INDEX"
    if [[ -f "$SHARD_DIR/shard_${idx}.json" ]]; then
      continue
    fi
    if [[ -n "${LOCAL_RUNNING_BY_INDEX[$idx]:-}" ]]; then
      continue
    fi
    _local_cpu_pack_slot="${LOCAL_CPU_PACK_FREE_SLOTS[0]}"
    LOCAL_CPU_PACK_FREE_SLOTS=("${LOCAL_CPU_PACK_FREE_SLOTS[@]:1}")
    TMPDIR="$EVALUATOR_TMPDIR" "${NANODISCOVER_EVAL_PYTHON:?NANODISCOVER_EVAL_PYTHON must point to the task evaluator python}" -m core.evaluator evaluate-shard \
      --task "$TASK_NAME" \
      --run-dir "$RUN_DIR" \
      --epoch "$EPOCH" \
      --start "$idx" \
      --stop "$((idx + 1))" \
      --workers 1 \
      --cpu-pack-slot "$_local_cpu_pack_slot" \
      --output "$SHARD_DIR/shard_${idx}.json" &
    pid="$!"
    LOCAL_CPU_PACK_PID_SLOT[$pid]="$_local_cpu_pack_slot"
    LOCAL_RUNNING_PIDS+=("$pid")
    LOCAL_RUNNING_BY_PID["$pid"]="$idx"
    LOCAL_RUNNING_BY_INDEX["$idx"]="$pid"
    LOCAL_LAUNCHED=$((LOCAL_LAUNCHED + 1))
  done
}

reap_finished_local_jobs() {
  local kept=()
  local pid
  local idx
  for pid in "${LOCAL_RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kept+=("$pid")
      continue
    fi
    idx="${LOCAL_RUNNING_BY_PID[$pid]:-unknown}"
    if ! wait "$pid"; then
      echo "stage3 local job failed idx=$idx"
      return 1
    fi
    if [[ -n "${LOCAL_CPU_PACK_PID_SLOT[$pid]+x}" ]]; then
      LOCAL_CPU_PACK_FREE_SLOTS+=("${LOCAL_CPU_PACK_PID_SLOT[$pid]}")
      unset 'LOCAL_CPU_PACK_PID_SLOT[$pid]'
    fi
    unset 'LOCAL_RUNNING_BY_PID[$pid]'
    unset 'LOCAL_RUNNING_BY_INDEX[$idx]'
    LOCAL_COMPLETED=$((LOCAL_COMPLETED + 1))
    if (( LOCAL_COMPLETED % 10 == 0 )) || (( ${#kept[@]} == 0 && $(pending_queue_depth) == 0 )); then
      echo "stage3 local progress completed=$LOCAL_COMPLETED launched=$LOCAL_LAUNCHED pending_queue=$(pending_queue_depth) running=$(( ${#kept[@]} ))"
    fi
  done
  LOCAL_RUNNING_PIDS=("${kept[@]}")
}

drain_local_queue() {
  launch_local_from_pending
  while (( $(pending_queue_depth) > 0 )) || (( ${#LOCAL_RUNNING_PIDS[@]} > 0 )); do
    sleep "$MONITOR_SLEEP_SECONDS"
    reap_finished_local_jobs
    launch_local_from_pending
  done
}

load_missing_indices() {
  python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["SHARD_DIR"])
total = int(os.environ["TOTAL"])
missing = []
for idx in range(total):
    if not (root / f"shard_{idx}.json").exists():
        missing.append(str(idx))
print("\n".join(missing))
PY
}

slice_indices_for_remote_task() {
  local job_id="$1"
  local arr_id="$2"
  local start_line
  local end_line
  local index_file="${REMOTE_JOB_INDEX_FILE[$job_id]}"
  local shards_per_task="${REMOTE_JOB_SHARDS_PER_TASK[$job_id]}"
  if [[ -z "$index_file" || -z "$shards_per_task" ]]; then
    return 0
  fi
  start_line=$((arr_id * shards_per_task + 1))
  end_line=$((start_line + shards_per_task - 1))
  sed -n "${start_line},${end_line}p" "$index_file"
}

clear_remote_slice_owners() {
  local job_id="$1"
  local arr_id="$2"
  local idx
  mapfile -t slice_indices < <(slice_indices_for_remote_task "$job_id" "$arr_id")
  for idx in "${slice_indices[@]}"; do
    if [[ "${REMOTE_INDEX_OWNER[$idx]:-}" == "$job_id:$arr_id" ]]; then
      unset 'REMOTE_INDEX_OWNER[$idx]'
    fi
  done
}

queue_remote_slice_missing() {
  local job_id="$1"
  local arr_id="$2"
  local force_single="$3"
  local idx
  QUEUE_SLICE_ADDED=0
  QUEUE_SLICE_TOTAL=0
  mapfile -t slice_indices < <(slice_indices_for_remote_task "$job_id" "$arr_id")
  for idx in "${slice_indices[@]}"; do
    if [[ ! "$idx" =~ ^[0-9]+$ ]]; then
      continue
    fi
    QUEUE_SLICE_TOTAL=$((QUEUE_SLICE_TOTAL + 1))
    if [[ "${REMOTE_INDEX_OWNER[$idx]:-}" == "$job_id:$arr_id" ]]; then
      unset 'REMOTE_INDEX_OWNER[$idx]'
    fi
    if [[ -f "$SHARD_DIR/shard_${idx}.json" ]]; then
      continue
    fi
    if (( force_single )); then
      SHARD_FORCE_SINGLE_REMOTE["$idx"]=1
    fi
    if queue_pending_index "$idx"; then
      QUEUE_SLICE_ADDED=$((QUEUE_SLICE_ADDED + 1))
    fi
  done
}

parse_slurm_exit_code() {
  local raw="$1"
  SLURM_EXIT_STATUS=0
  SLURM_EXIT_SIGNAL=0
  if [[ "$raw" =~ ^([0-9]+):([0-9]+)$ ]]; then
    SLURM_EXIT_STATUS="${BASH_REMATCH[1]}"
    SLURM_EXIT_SIGNAL="${BASH_REMATCH[2]}"
  fi
}

record_bad_remote_node() {
  local node="$1"
  local state="$2"
  local exit_code="$3"
  local should_record=0
  case "$state" in
    NODE_FAIL*|BOOT_FAIL*)
      should_record=1
      ;;
    FAILED*)
      parse_slurm_exit_code "$exit_code"
      if (( SLURM_EXIT_SIGNAL > 0 )); then
        should_record=1
      fi
      ;;
  esac
  if (( should_record )) && [[ -n "$node" && "$node" != "None assigned" ]] && [[ -z "${REMOTE_BAD_NODES[$node]:-}" ]]; then
    REMOTE_BAD_NODES["$node"]="$state:$exit_code"
    echo "stage3 remote bad_node node=$node state=$state exit_code=$exit_code excluded_for_future=1"
  fi
}

bad_nodes_csv() {
  local nodes=("${!REMOTE_BAD_NODES[@]}")
  if (( ${#nodes[@]} == 0 )); then
    return 0
  fi
  local IFS=,
  echo "${nodes[*]}"
}

remote_active_array_task_count() {
  local used=0
  local job_id
  local arr_id
  local total
  for job_id in "${ACTIVE_REMOTE_JOB_IDS[@]}"; do
    total="${REMOTE_JOB_ARRAY_TASK_COUNT[$job_id]:-0}"
    for ((arr_id=0; arr_id<total; arr_id++)); do
      if [[ -z "${REMOTE_TASK_ACTION["$job_id:$arr_id"]:-}" ]]; then
        used=$((used + 1))
      fi
    done
  done
  echo "$used"
}

available_remote_array_task_slots() {
  local used
  local free
  used=$(remote_active_array_task_count)
  free=$(( REMOTE_MAX_ARRAY_TASKS - used ))
  if (( free < 0 )); then
    free=0
  fi
  echo "$free"
}

write_synthetic_failure_shard() {
  local idx="$1"
  local state="$2"
  local exit_code="$3"
  local node="$4"
  local remote_task_id="$5"
  STAGE3_FORCE_FAIL_INDEX="$idx" \
  STAGE3_FORCE_FAIL_EPOCH="$EPOCH" \
  STAGE3_FORCE_FAIL_RUN_DIR="$RUN_DIR" \
  STAGE3_FORCE_FAIL_OUTPUT="$SHARD_DIR/shard_${idx}.json" \
  STAGE3_FORCE_FAIL_STATE="$state" \
  STAGE3_FORCE_FAIL_EXIT_CODE="$exit_code" \
  STAGE3_FORCE_FAIL_NODE="$node" \
  STAGE3_FORCE_FAIL_REMOTE_TASK_ID="$remote_task_id" \
  STAGE3_FORCE_FAIL_TASK="circle_packing" \
  "${NANODISCOVER_EVAL_PYTHON:?NANODISCOVER_EVAL_PYTHON must point to the task evaluator python}" - <<'PY'
from pathlib import Path
import os

from core.evaluator import (
    EvaluatedRollout,
    build_rollout_shard_payload,
    build_task,
    expand_seed_states_and_prompts_for_generations,
    load_generation_payload,
    load_sample_payload,
    write_json_payload,
)

run_dir = Path(os.environ["STAGE3_FORCE_FAIL_RUN_DIR"]).resolve()
epoch = int(os.environ["STAGE3_FORCE_FAIL_EPOCH"])
idx = int(os.environ["STAGE3_FORCE_FAIL_INDEX"])
output_path = Path(os.environ["STAGE3_FORCE_FAIL_OUTPUT"]).resolve()
state = os.environ["STAGE3_FORCE_FAIL_STATE"]
exit_code = os.environ["STAGE3_FORCE_FAIL_EXIT_CODE"]
node = os.environ["STAGE3_FORCE_FAIL_NODE"]
remote_task_id = os.environ["STAGE3_FORCE_FAIL_REMOTE_TASK_ID"]
task_name = os.environ["STAGE3_FORCE_FAIL_TASK"]

seed_states, prompts = load_sample_payload(run_dir, epoch)
generations = load_generation_payload(run_dir, epoch)
repeated_seed_states, repeated_prompts, _ = expand_seed_states_and_prompts_for_generations(
    seed_states,
    prompts,
    generations,
)
task = build_task(task_name)
generation = generations[idx]
parsed_code = str(task.parse_code(generation.response_text) or "")
message = (
    f"isolated remote {state} after OOM split; treating as candidate fault "
    f"(task={remote_task_id}, exitcode={exit_code}, node={node})"
)
rollout = EvaluatedRollout(
    seed_state=repeated_seed_states[idx],
    prompt_text=repeated_prompts[idx],
    response_text=generation.response_text,
    prompt_token_ids=list(generation.prompt_token_ids),
    completion_token_ids=list(generation.completion_token_ids),
    completion_logprobs=list(generation.completion_logprobs),
    completion_mask=list(generation.completion_mask),
    finish_reason=generation.finish_reason,
    parsed_code=parsed_code,
    reward=0.0,
    correctness=0.0,
    performance=0.0,
    raw_score=0.0,
    archive_value=None,
    next_state=None,
    msg=message,
    result_payload={
        "remote_failure": {
            "state": state,
            "exit_code": exit_code,
            "node": node,
            "task_id": remote_task_id,
            "reason": "isolated_remote_oom_candidate_fault",
        }
    },
    stdout=f"[stage3]\n{message}\n",
)
payload = build_rollout_shard_payload(epoch=epoch, start_index=idx, evaluated=[rollout])
write_json_payload(output_path, payload)
PY
  SHARD_FINALIZED_REASON["$idx"]="isolated_remote_oom"
  unset 'SHARD_FORCE_SINGLE_REMOTE[$idx]'
  FINALIZED_SYNTHETIC_FAILURES=$((FINALIZED_SYNTHETIC_FAILURES + 1))
  echo "stage3 finalize source=remote_oom_single idx=$idx task=$remote_task_id node=$node exit_code=$exit_code synthetic_failures=$FINALIZED_SYNTHETIC_FAILURES"
}

process_remote_terminal_task() {
  local job_id="$1"
  local arr_id="$2"
  local state="$3"
  local exit_code="$4"
  local node="$5"
  local key="$job_id:$arr_id"
  local idx
  if [[ -n "${REMOTE_TASK_ACTION[$key]:-}" ]]; then
    return 0
  fi
  record_bad_remote_node "$node" "$state" "$exit_code"
  case "$state" in
    COMPLETED*)
      queue_remote_slice_missing "$job_id" "$arr_id" 0
      if (( QUEUE_SLICE_ADDED > 0 )); then
        REMOTE_TASK_ACTION["$key"]="completed_missing_requeued"
        echo "stage3 remote retry source=completed_missing job=${job_id}_${arr_id} shards=$QUEUE_SLICE_ADDED/$QUEUE_SLICE_TOTAL"
      else
        REMOTE_TASK_ACTION["$key"]="completed"
      fi
      ;;
    OUT_OF_MEMORY*)
      mapfile -t slice_indices < <(slice_indices_for_remote_task "$job_id" "$arr_id")
      QUEUE_SLICE_TOTAL=0
      for idx in "${slice_indices[@]}"; do
        if [[ "$idx" =~ ^[0-9]+$ ]]; then
          QUEUE_SLICE_TOTAL=$((QUEUE_SLICE_TOTAL + 1))
        fi
      done
      if (( QUEUE_SLICE_TOTAL > 1 )); then
        queue_remote_slice_missing "$job_id" "$arr_id" 1
        REMOTE_TASK_ACTION["$key"]="retry_oom_split"
        echo "stage3 remote retry source=oom_split job=${job_id}_${arr_id} shards=$QUEUE_SLICE_ADDED/$QUEUE_SLICE_TOTAL"
      else
        REMOTE_TASK_ACTION["$key"]="finalized_oom_single"
        clear_remote_slice_owners "$job_id" "$arr_id"
        for idx in "${slice_indices[@]}"; do
          if [[ ! "$idx" =~ ^[0-9]+$ ]]; then
            continue
          fi
          if [[ -f "$SHARD_DIR/shard_${idx}.json" ]]; then
            continue
          fi
          write_synthetic_failure_shard "$idx" "$state" "$exit_code" "$node" "${job_id}_${arr_id}"
        done
      fi
      ;;
    PREEMPTED*|NODE_FAIL*|BOOT_FAIL*|FAILED*|TIMEOUT*|CANCELLED*)
      queue_remote_slice_missing "$job_id" "$arr_id" 0
      REMOTE_TASK_ACTION["$key"]="retry_${state//[^[:alnum:]]/_}"
      echo "stage3 remote retry source=terminal job=${job_id}_${arr_id} state=$state shards=$QUEUE_SLICE_ADDED/$QUEUE_SLICE_TOTAL node=$node exit_code=$exit_code"
      ;;
    *)
      clear_remote_slice_owners "$job_id" "$arr_id"
      REMOTE_TASK_ACTION["$key"]="terminal_${state//[^[:alnum:]]/_}"
      ;;
  esac
}

monitor_remote_job() {
  local job_id="$1"
  local line
  local task_id
  local state
  local arr_raw
  local arr_id
  local exit_code
  local node
  local pending_count=0
  local running_count=0
  local active_count=0
  local handled_count=0
  local total="${REMOTE_JOB_ARRAY_TASK_COUNT[$job_id]:-0}"
  local key
  local limbo_tasks=0
  local limbo_shards=0
  local accounting_job_id
  declare -A active_arr=()
  mapfile -t queue_lines < <(squeue -h -j "$job_id" -o "%i|%T" 2>/dev/null || true)
  for line in "${queue_lines[@]}"; do
    IFS='|' read -r task_id state <<< "$line"
    arr_raw="${task_id#${job_id}_}"
    if [[ "$arr_raw" == "$task_id" ]]; then
      continue
    fi
    mapfile -t arr_ids < <(expand_array_ids "$arr_raw")
    for arr_id in "${arr_ids[@]}"; do
      active_arr["$arr_id"]=1
      active_count=$((active_count + 1))
      case "$state" in
        PD*|PENDING*)
          pending_count=$((pending_count + 1))
          ;;
        R*|RUNNING*)
          running_count=$((running_count + 1))
          ;;
      esac
    done
  done
  mapfile -t accounting_lines < <(sacct -X -n -P -j "$job_id" --format=JobID,State,ExitCode,NodeList 2>/dev/null || true)
  for line in "${accounting_lines[@]}"; do
    IFS='|' read -r accounting_job_id state exit_code node <<< "$line"
    arr_raw="${accounting_job_id#${job_id}_}"
    if [[ "$arr_raw" == "$accounting_job_id" ]]; then
      continue
    fi
    mapfile -t arr_ids < <(expand_array_ids "$arr_raw")
    for arr_id in "${arr_ids[@]}"; do
      if [[ -n "${active_arr[$arr_id]:-}" ]]; then
        continue
      fi
      process_remote_terminal_task "$job_id" "$arr_id" "$state" "$exit_code" "$node"
    done
  done
  for ((arr_id=0; arr_id<total; arr_id++)); do
    if [[ -n "${REMOTE_TASK_ACTION["$job_id:$arr_id"]:-}" ]]; then
      handled_count=$((handled_count + 1))
    fi
  done
  if (( active_count == 0 )) && (( handled_count < total )); then
    REMOTE_JOB_IDLE_LOOPS["$job_id"]=$(( ${REMOTE_JOB_IDLE_LOOPS[$job_id]:-0} + 1 ))
    if (( ${REMOTE_JOB_IDLE_LOOPS[$job_id]} >= 3 )); then
      for ((arr_id=0; arr_id<total; arr_id++)); do
        key="$job_id:$arr_id"
        if [[ -n "${REMOTE_TASK_ACTION[$key]:-}" ]]; then
          continue
        fi
        REMOTE_TASK_ACTION["$key"]="limbo_requeued"
        queue_remote_slice_missing "$job_id" "$arr_id" 0
        limbo_tasks=$((limbo_tasks + 1))
        limbo_shards=$((limbo_shards + QUEUE_SLICE_ADDED))
      done
      echo "stage3 remote retry source=limbo job=$job_id array_tasks=$limbo_tasks shards=$limbo_shards"
      scancel "$job_id" >/dev/null 2>&1 || true
      handled_count=0
      for ((arr_id=0; arr_id<total; arr_id++)); do
        if [[ -n "${REMOTE_TASK_ACTION["$job_id:$arr_id"]:-}" ]]; then
          handled_count=$((handled_count + 1))
        fi
      done
    fi
  else
    REMOTE_JOB_IDLE_LOOPS["$job_id"]=0
  fi
  REMOTE_JOB_PENDING_COUNT["$job_id"]="$pending_count"
  REMOTE_JOB_RUNNING_COUNT["$job_id"]="$running_count"
  REMOTE_JOB_ACTIVE_COUNT["$job_id"]="$active_count"
  REMOTE_JOB_HANDLED_COUNT["$job_id"]="$handled_count"
}

reclaim_pending_remote_tasks() {
  local free_slots
  local queue_depth
  local needed
  local reclaimed_tasks=0
  local added_total=0
  local job_id
  local task_id
  local arr_raw
  local arr_id
  local key
  free_slots=$(local_slots_free)
  if (( free_slots < 1 )); then
    return 0
  fi
  queue_depth=$(pending_queue_depth)
  if (( queue_depth >= free_slots )); then
    return 0
  fi
  needed=$(( free_slots - queue_depth ))
  for job_id in "${ACTIVE_REMOTE_JOB_IDS[@]}"; do
    mapfile -t pending_tasks < <(squeue -h -j "$job_id" -t PD -o "%i" 2>/dev/null || true)
    for task_id in "${pending_tasks[@]}"; do
      arr_raw="${task_id#${job_id}_}"
      if [[ "$arr_raw" == "$task_id" ]]; then
        continue
      fi
      mapfile -t arr_ids < <(expand_array_ids "$arr_raw")
      for arr_id in "${arr_ids[@]}"; do
        key="$job_id:$arr_id"
        if [[ -n "${REMOTE_TASK_ACTION[$key]:-}" ]]; then
          continue
        fi
        REMOTE_TASK_ACTION["$key"]="reclaimed_pending"
        scancel "${job_id}_${arr_id}" >/dev/null 2>&1 || true
        queue_remote_slice_missing "$job_id" "$arr_id" 0
        if (( QUEUE_SLICE_ADDED > 0 )); then
          reclaimed_tasks=$((reclaimed_tasks + 1))
          added_total=$((added_total + QUEUE_SLICE_ADDED))
        fi
        if (( $(pending_queue_depth) >= needed )); then
          break 3
        fi
      done
    done
  done
  if (( added_total > 0 )); then
    RECLAIMED_PENDING_THIS_LOOP=1
    echo "stage3 remote reclaim source=pending array_tasks=$reclaimed_tasks shards=$added_total pending_queue=$(pending_queue_depth) running_local=${#LOCAL_RUNNING_PIDS[@]}"
  fi
}

submit_remote_wave_from_indices() {
  local kind="$1"
  local shards_per_task="$2"
  local parallel_evals="$3"
  shift 3
  local indices=("$@")
  local count="${#indices[@]}"
  local task_count
  local cpus_per_task
  local mem_per_task
  local wave_name
  local index_file
  local remote_out_pattern
  local remote_err_pattern
  local exclude_csv
  local idx_offset=0
  local arr_id
  local idx
  local job_id
  local qos_args=()
  local exclude_args=()
  local nodelist_args=()
  if (( count < 1 )); then
    return 0
  fi
  if (( shards_per_task < 1 )); then
    shards_per_task=1
  fi
  task_count=$(( (count + shards_per_task - 1) / shards_per_task ))
  if (( parallel_evals < 1 )); then
    parallel_evals="$shards_per_task"
  fi
  if (( parallel_evals > shards_per_task )); then
    parallel_evals="$shards_per_task"
  fi
  cpus_per_task=$(( SLURM_ATTACHED_CPUS_PER_EVAL * parallel_evals ))
  mem_per_task=$(scale_mem_spec "$SLURM_ATTACHED_REMOTE_MEM_PER_TASK" "$parallel_evals")
  WAVE_SEQUENCE=$((WAVE_SEQUENCE + 1))
  wave_name="$(printf "wave%03d_%s" "$WAVE_SEQUENCE" "$kind")"
  index_file="$REMOTE_WAVE_DIR/${wave_name}.indices.txt"
  printf "%s\n" "${indices[@]}" > "$index_file"
  exclude_csv="$(bad_nodes_csv)"
  if [[ -n "${SLURM_ATTACHED_REMOTE_QOS:-}" ]]; then
    qos_args=(--qos "$SLURM_ATTACHED_REMOTE_QOS")
  fi
  if [[ -n "$exclude_csv" ]]; then
    exclude_args=(--exclude "$exclude_csv")
  fi

  # Restrict eval spillover to known-good CPU nodes for score reproducibility.
  # Filter out draining/down nodes dynamically to avoid ReqNodeNotAvail blocking.
  local nodelist="${SLURM_ATTACHED_REMOTE_NODELIST:-}"
  if [[ -n "$nodelist" ]]; then
    local _active_nodes
    _active_nodes=$(sinfo -N -p "${SLURM_ATTACHED_REMOTE_PARTITIONS}" --noheader -n "$nodelist" --format="%N %T" 2>/dev/null | awk '$2 !~ /drain|down/ {print $1}' | paste -sd, -)
    if [[ -n "$_active_nodes" ]]; then
      nodelist="$_active_nodes"
    fi
    nodelist_args=(--nodelist "$nodelist")
  fi
  remote_out_pattern="$SHARD_DIR/slurm-${wave_name}-%A_%a.out"
  remote_err_pattern="$SHARD_DIR/slurm-${wave_name}-%A_%a.err"
  echo "stage3 remote submit wave=$wave_name kind=$kind shards=$count array_tasks=$task_count shards_per_task=$shards_per_task parallel_evals=$parallel_evals cpus_per_task=$cpus_per_task mem_per_task=$mem_per_task exclude=${exclude_csv:-none}"
  if ! job_id=$(sbatch --parsable \
    --partition="$SLURM_ATTACHED_REMOTE_PARTITIONS" \
    "${qos_args[@]}" \
    "${exclude_args[@]}" \
    "${nodelist_args[@]}" \
    --export="ALL,TMPDIR=$EVALUATOR_TMPDIR" \
    --time="$SLURM_ATTACHED_REMOTE_TIME" \
    --cpus-per-task="$cpus_per_task" \
    --mem="$mem_per_task" \
    --output="$remote_out_pattern" \
    --error="$remote_err_pattern" \
    --array="0-$((task_count - 1))" \
    tasks/circle_packing/launchers/slurm_attached/stage3_array.sbatch.sh \
    "$PWD" "$RUN_DIR" "$EPOCH" "$index_file" "$SHARD_DIR" "$SLURM_ATTACHED_REMOTE_WORKERS_PER_TASK" "$shards_per_task" "$parallel_evals"); then
    echo "stage3 remote submit failed wave=$wave_name; requeueing $count shards"
    for idx in "${indices[@]}"; do
      queue_pending_index "$idx" || true
    done
    return 1
  fi
  printf "%s\n" "$job_id" >> "$ACTIVE_REMOTE_JOB_IDS_FILE"
  ACTIVE_REMOTE_JOB_IDS+=("$job_id")
  REMOTE_JOB_INDEX_FILE["$job_id"]="$index_file"
  REMOTE_JOB_SHARDS_PER_TASK["$job_id"]="$shards_per_task"
  REMOTE_JOB_PARALLEL_EVALS["$job_id"]="$parallel_evals"
  REMOTE_JOB_ARRAY_TASK_COUNT["$job_id"]="$task_count"
  REMOTE_JOB_KIND["$job_id"]="$kind"
  REMOTE_JOB_IDLE_LOOPS["$job_id"]=0
  REMOTE_JOB_PENDING_COUNT["$job_id"]="$task_count"
  REMOTE_JOB_RUNNING_COUNT["$job_id"]=0
  REMOTE_JOB_ACTIVE_COUNT["$job_id"]="$task_count"
  REMOTE_JOB_HANDLED_COUNT["$job_id"]=0
  for ((arr_id=0; arr_id<task_count; arr_id++)); do
    for ((slot=0; slot<shards_per_task && idx_offset<count; slot++)); do
      idx="${indices[$idx_offset]}"
      REMOTE_INDEX_OWNER["$idx"]="$job_id:$arr_id"
      idx_offset=$((idx_offset + 1))
    done
  done
  echo "submitted remote eval array job $job_id wave=$wave_name kind=$kind"
}

submit_remote_single_wave() {
  local available_slots
  local reserve_floor
  local single_pending
  available_slots=$(available_remote_array_task_slots)
  if (( available_slots < 1 )); then
    return 0
  fi
  reserve_floor=$(local_reserved_pending_floor)
  single_pending=$(count_pending_indices_matching single "$reserve_floor")
  if (( single_pending < 1 )); then
    return 0
  fi
  if (( single_pending > available_slots )); then
    single_pending="$available_slots"
  fi
  take_pending_indices_for_remote_wave "$single_pending" single "$reserve_floor"
  submit_remote_wave_from_indices single 1 1 "${TAKEN_PENDING_INDICES[@]}"
}

submit_remote_packed_wave() {
  local available_slots
  local reserve_floor
  local normal_pending
  local shards_per_task
  local max_indices
  local parallel_evals
  available_slots=$(available_remote_array_task_slots)
  if (( available_slots < 1 )); then
    return 0
  fi
  reserve_floor=$(local_reserved_pending_floor)
  normal_pending=$(count_pending_indices_matching normal "$reserve_floor")
  if (( normal_pending < 1 )); then
    return 0
  fi
  shards_per_task=$(( (normal_pending + available_slots - 1) / available_slots ))
  if (( shards_per_task < REMOTE_SHARDS_PER_TASK )); then
    shards_per_task="$REMOTE_SHARDS_PER_TASK"
  fi
  if (( shards_per_task < 1 )); then
    shards_per_task=1
  fi
  max_indices=$(( available_slots * shards_per_task ))
  take_pending_indices_for_remote_wave "$max_indices" normal "$reserve_floor"
  if (( ${#TAKEN_PENDING_INDICES[@]} == 0 )); then
    return 0
  fi
  parallel_evals=$(( SLURM_ATTACHED_REMOTE_PARALLEL_EVALS_PER_TASK ))
  if (( parallel_evals <= 0 )); then
    parallel_evals="$shards_per_task"
  fi
  if (( parallel_evals > shards_per_task )); then
    parallel_evals="$shards_per_task"
  fi
  if (( parallel_evals < 1 )); then
    parallel_evals="$shards_per_task"
  fi
  submit_remote_wave_from_indices packed "$shards_per_task" "$parallel_evals" "${TAKEN_PENDING_INDICES[@]}"
}

for idx in "${pending_indices[@]}"; do
  queue_pending_index "$idx" || true
done
echo "stage3 pending queue initial=$(pending_queue_depth) reserved_local=$(local_reserved_pending_floor) local_parallelism=$LOCAL_PARALLEL"

while true; do
  reap_finished_local_jobs
  RECLAIMED_PENDING_THIS_LOOP=0
  updated_remote_jobs=()
  for job_id in "${ACTIVE_REMOTE_JOB_IDS[@]}"; do
    monitor_remote_job "$job_id"
    if (( ${REMOTE_JOB_HANDLED_COUNT[$job_id]:-0} < ${REMOTE_JOB_ARRAY_TASK_COUNT[$job_id]:-0} )) || (( ${REMOTE_JOB_ACTIVE_COUNT[$job_id]:-0} > 0 )); then
      updated_remote_jobs+=("$job_id")
    else
      echo "stage3 remote job complete job=$job_id kind=${REMOTE_JOB_KIND[$job_id]} array_tasks=${REMOTE_JOB_ARRAY_TASK_COUNT[$job_id]}"
    fi
  done
  ACTIVE_REMOTE_JOB_IDS=("${updated_remote_jobs[@]}")
  reclaim_pending_remote_tasks
  launch_local_from_pending
  if (( RECLAIMED_PENDING_THIS_LOOP == 0 )); then
    submit_remote_single_wave
    submit_remote_packed_wave
  fi

  mapfile -t missing < <(load_missing_indices)
  filtered_missing=()
  for idx in "${missing[@]}"; do
    if [[ "$idx" =~ ^[0-9]+$ ]]; then
      filtered_missing+=("$idx")
    fi
  done
  missing=("${filtered_missing[@]}")

  total_remote_pending=0
  total_remote_running=0
  total_remote_active=0
  for job_id in "${ACTIVE_REMOTE_JOB_IDS[@]}"; do
    total_remote_pending=$((total_remote_pending + ${REMOTE_JOB_PENDING_COUNT[$job_id]:-0}))
    total_remote_running=$((total_remote_running + ${REMOTE_JOB_RUNNING_COUNT[$job_id]:-0}))
    total_remote_active=$((total_remote_active + ${REMOTE_JOB_ACTIVE_COUNT[$job_id]:-0}))
  done
  remote_status="missing=${#missing[@]} queue=$(pending_queue_depth) reserved_local=$(local_reserved_pending_floor) remote_pending=$total_remote_pending remote_running=$total_remote_running remote_active=$total_remote_active remote_jobs=${#ACTIVE_REMOTE_JOB_IDS[@]} local_running=${#LOCAL_RUNNING_PIDS[@]} bad_nodes=${#REMOTE_BAD_NODES[@]} synthetic_failures=$FINALIZED_SYNTHETIC_FAILURES"
  if [[ "$remote_status" != "$REMOTE_STATUS_LAST" ]]; then
    echo "stage3 remote monitor $remote_status"
    REMOTE_STATUS_LAST="$remote_status"
  fi
  if [[ "${#missing[@]}" -eq 0 ]] && (( $(pending_queue_depth) == 0 )) && (( ${#LOCAL_RUNNING_PIDS[@]} == 0 )) && (( total_remote_active == 0 )); then
    echo "stage3 remote monitor complete missing=0"
    break
  fi
  if [[ "${#missing[@]}" -gt 0 ]] && (( $(pending_queue_depth) == 0 )) && (( ${#LOCAL_RUNNING_PIDS[@]} == 0 )) && (( total_remote_active == 0 )); then
    echo "stage3 queue repair missing=${#missing[@]} requeueing=1"
    for idx in "${missing[@]}"; do
      queue_pending_index "$idx" || true
    done
    launch_local_from_pending
    submit_remote_single_wave
    submit_remote_packed_wave
    continue
  fi
  sleep "$MONITOR_SLEEP_SECONDS"
done

if [[ "${#ACTIVE_REMOTE_JOB_IDS[@]}" -gt 0 ]]; then
  for job_id in "${ACTIVE_REMOTE_JOB_IDS[@]}"; do
    scancel "$job_id" >/dev/null 2>&1 || true
  done
  ACTIVE_REMOTE_JOB_IDS=()
fi
if [[ -n "${ACTIVE_REMOTE_JOB_IDS_FILE:-}" ]]; then
  rm -f "$ACTIVE_REMOTE_JOB_IDS_FILE"
fi

# 4) Safety pass: evaluate anything still missing locally.
final_scan_counts=$(python - <<'PY'
from pathlib import Path
import os

from core.evaluator import scan_shard_status

pending_file = Path(os.environ["PENDING_FILE"])
status = scan_shard_status(
    Path(os.environ["SHARD_DIR"]),
    expected_total=int(os.environ["TOTAL"]),
    delete_invalid=True,
)
pending = [str(idx) for idx in status["pending_indices"]]
pending_file.write_text(("\n".join(pending) + "\n") if pending else "", encoding="utf-8")
print(f"{status['complete_count']} {status['invalid_count']} {len(status['pending_indices'])}")
PY
)
read -r FINAL_COMPLETE_SHARDS FINAL_INVALID_SHARDS FINAL_PENDING_SHARDS <<< "$final_scan_counts"
echo "stage3 safety pass complete_shards=$FINAL_COMPLETE_SHARDS invalid_shards=$FINAL_INVALID_SHARDS pending_shards=$FINAL_PENDING_SHARDS"
mapfile -t still_missing < "$PENDING_FILE"
filtered_still_missing=()
for idx in "${still_missing[@]}"; do
  if [[ "$idx" =~ ^[0-9]+$ ]]; then
    filtered_still_missing+=("$idx")
  fi
done
still_missing=("${filtered_still_missing[@]}")
if [[ "${#still_missing[@]}" -gt 0 ]]; then
  echo "stage3 safety pass requeueing=${#still_missing[@]}"
  for idx in "${still_missing[@]}"; do
    queue_pending_index "$idx" || true
  done
  drain_local_queue
fi

final_count=$(find "$SHARD_DIR" -maxdepth 1 -name 'shard_*.json' | wc -l)
echo "stage3 merge start shards_present=$final_count expected=$TOTAL"

python -m core.evaluator merge-shards \
  --run-dir "$RUN_DIR" \
  --epoch "$EPOCH" \
  --shard-dir "$SHARD_DIR" \
  --expected-total "$TOTAL"
