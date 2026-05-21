#!/usr/bin/env bash
set -euo pipefail

slurm_alloc_fail() {
  echo "$1" >&2
  return 2 2>/dev/null || exit 2
}

slurm_alloc_require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    slurm_alloc_fail "$name must be set before launching this mode."
  fi
}

slurm_alloc_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    slurm_alloc_fail "$label does not exist: $path"
  fi
}

slurm_alloc_require_exec() {
  local path="$1"
  local label="$2"
  if [[ ! -x "$path" ]]; then
    slurm_alloc_fail "$label is not executable: $path"
  fi
}

slurm_alloc_prepare_submission_context() {
  local repo_root="$1"
  local common_env_path="$2"
  local scheduler_log_dir

  # Validate the task config and evaluator path before submission so failures
  # happen on the submit side rather than after allocation.
  # shellcheck disable=SC1090
  source "$common_env_path"
  # shellcheck disable=SC1090
  source "${repo_root}/tasks/${NANODISCOVER_ALLOC_TASK_NAME}/launchers/slurm_alloc/profile_env.sh"

  : "${SLURM_ALLOC_ALWAYS_RETRY:=0}"
  : "${SLURM_ALLOC_ENABLE_REQUEUE:=1}"

  if [[ "${SLURM_ALLOC_ALWAYS_RETRY}" == "1" ]] && [[ -z "${NANODISCOVER_RESUME_DIR:-}" ]]; then
    : "${NANODISCOVER_LOG_ROOT:=./logs}"
    export NANODISCOVER_RESUME_DIR="${NANODISCOVER_LOG_ROOT}/${NANODISCOVER_ALLOC_TASK_NAME}-$(date +%Y%m%d-%H%M%S)"
  fi

  if [[ -n "${NANODISCOVER_RESUME_DIR:-}" ]]; then
    mkdir -p "$NANODISCOVER_RESUME_DIR"
    scheduler_log_dir="${NANODISCOVER_RESUME_DIR}/slurm"
  else
    : "${NANODISCOVER_LOG_ROOT:=./logs}"
    scheduler_log_dir="${NANODISCOVER_LOG_ROOT}/slurm_alloc"
  fi
  : "${SLURM_ALLOC_STDOUT_PATH:=${SLURM_ALLOC_OUTPUT_PATTERN:-${scheduler_log_dir}/%x-%j.out}}"
  : "${SLURM_ALLOC_STDERR_PATH:=${SLURM_ALLOC_ERROR_PATTERN:-${scheduler_log_dir}/%x-%j.err}}"
  mkdir -p "$(dirname "$SLURM_ALLOC_STDOUT_PATH")" "$(dirname "$SLURM_ALLOC_STDERR_PATH")"

  : "${SLURM_ALLOC_GPU_JOB_NAME:=nanodisc-${NANODISCOVER_ALLOC_TASK_NAME}-${NANODISCOVER_ALLOC_JOB_NAME_SUFFIX}}"
  : "${SLURM_ALLOC_GPU_PARTITION:=preempt}"
  : "${SLURM_ALLOC_GPU_QOS:=}"
  : "${SLURM_ALLOC_GPU_TIME:=2-00:00:00}"
  : "${SLURM_ALLOC_GPU_GRES:=gpu:RTX_PRO_6000:4}"
  : "${SLURM_ALLOC_GPU_CPUS_PER_TASK:=32}"
  : "${SLURM_ALLOC_GPU_MEM:=96G}"
  : "${SLURM_ALLOC_GPU_NODELIST:=}"
  : "${SLURM_ALLOC_GPU_COMMENT:=PROFILER_DISABLE}"
  : "${SLURM_ALLOC_SIGNAL_SECONDS:=120}"

  if [[ "${SLURM_ALLOC_ALWAYS_RETRY}" == "1" ]] && [[ "${SLURM_ALLOC_ENABLE_REQUEUE}" == "1" ]]; then
    echo "slurm_alloc note: SLURM_ALLOC_ALWAYS_RETRY=1 disables native Slurm requeue." >&2
    SLURM_ALLOC_ENABLE_REQUEUE=0
  fi
}

slurm_alloc_submit_wrapper() {
  local wrapper_path="$1"
  local repo_root="$2"
  local job_id
  local -a sbatch_args

  sbatch_args=(
    --parsable
    --nodes=1
    --ntasks-per-node=1
    --partition="$SLURM_ALLOC_GPU_PARTITION"
    --time="$SLURM_ALLOC_GPU_TIME"
    --cpus-per-task="$SLURM_ALLOC_GPU_CPUS_PER_TASK"
    --mem="$SLURM_ALLOC_GPU_MEM"
    --job-name="$SLURM_ALLOC_GPU_JOB_NAME"
    --output="$SLURM_ALLOC_STDOUT_PATH"
    --error="$SLURM_ALLOC_STDERR_PATH"
    --chdir="$repo_root"
    --signal="B:TERM@${SLURM_ALLOC_SIGNAL_SECONDS}"
    --export="ALL,NANODISCOVER_ALLOC_INNER=1,NANODISCOVER_ROOT=${repo_root}"
  )

  if [[ -n "$SLURM_ALLOC_GPU_GRES" ]]; then
    sbatch_args+=(--gres="$SLURM_ALLOC_GPU_GRES")
  fi
  if [[ -n "${SLURM_ALLOC_GPU_QOS:-}" ]]; then
    sbatch_args+=(--qos="$SLURM_ALLOC_GPU_QOS")
  fi
  if [[ -n "${SLURM_ALLOC_GPU_NODELIST:-}" ]]; then
    sbatch_args+=(--nodelist="$SLURM_ALLOC_GPU_NODELIST")
  fi
  if [[ -n "${SLURM_ALLOC_GPU_COMMENT:-}" ]]; then
    sbatch_args+=(--comment="$SLURM_ALLOC_GPU_COMMENT")
  fi
  if [[ "${SLURM_ALLOC_ALWAYS_RETRY}" == "1" ]]; then
    sbatch_args+=(--no-requeue)
  elif [[ "${SLURM_ALLOC_ENABLE_REQUEUE}" == "1" ]]; then
    sbatch_args+=(--requeue)
  fi

  job_id="$(sbatch "${sbatch_args[@]}" "$wrapper_path")"
  printf '%s\n' "$job_id"
}

slurm_alloc_handle_term() {
  local child_pid="$1"
  local wrapper_path="$2"
  local repo_root="$3"
  local common_env_path="$4"
  local new_job_id

  if [[ -z "${child_pid:-}" ]] || ! kill -0 "$child_pid" 2>/dev/null; then
    return 0
  fi

  if [[ "${SLURM_ALLOC_ALWAYS_RETRY:-0}" == "1" ]] && [[ "${SLURM_ALLOC_RETRY_SUBMITTED:-0}" != "1" ]]; then
    SLURM_ALLOC_RETRY_SUBMITTED=1
    slurm_alloc_prepare_submission_context "$repo_root" "$common_env_path"
    if new_job_id="$(slurm_alloc_submit_wrapper "$wrapper_path" "$repo_root")"; then
      echo "slurm_alloc always_retry old_job=${SLURM_JOB_ID:-unknown} new_job=${new_job_id} signal=TERM" >&2
    else
      echo "slurm_alloc always_retry submit_failed old_job=${SLURM_JOB_ID:-unknown} signal=TERM" >&2
    fi
  fi

  kill -TERM -- "-${child_pid}" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
}

slurm_alloc_run() {
  slurm_alloc_require_env NANODISCOVER_ALLOC_TASK_NAME
  slurm_alloc_require_env NANODISCOVER_ALLOC_COMMON_ENV
  slurm_alloc_require_env NANODISCOVER_ALLOC_INNER_MODE
  slurm_alloc_require_env NANODISCOVER_ALLOC_INNER_ENTRYPOINT
  slurm_alloc_require_env NANODISCOVER_ALLOC_JOB_NAME_SUFFIX
  slurm_alloc_require_env NANODISCOVER_AC1_CONFIG
  slurm_alloc_require_env NANODISCOVER_EVAL_PYTHON
  slurm_alloc_require_env NANODISCOVER_RUNTIME_VENV

  local script_dir repo_root activate_path inner_entrypoint common_env_path scheduler_log_dir wrapper_source wrapper_path
  local -a sbatch_args
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  wrapper_source="${BASH_SOURCE[1]:-$0}"
  if [[ "$wrapper_source" == /* ]]; then
    wrapper_path="$wrapper_source"
  else
    wrapper_path="$(cd "$(dirname "$wrapper_source")" && pwd)/$(basename "$wrapper_source")"
  fi
  repo_root="${NANODISCOVER_ROOT:-$(cd "${script_dir}/../../../../.." && pwd)}"
  export NANODISCOVER_ROOT="$repo_root"

  common_env_path="${repo_root}/${NANODISCOVER_ALLOC_COMMON_ENV}"
  inner_entrypoint="${repo_root}/tasks/${NANODISCOVER_ALLOC_TASK_NAME}/launchers/${NANODISCOVER_ALLOC_INNER_MODE}/${NANODISCOVER_ALLOC_INNER_ENTRYPOINT}"
  activate_path="${NANODISCOVER_RUNTIME_VENV}/bin/activate"

  slurm_alloc_require_file "$common_env_path" "task launcher env"
  slurm_alloc_require_file "$inner_entrypoint" "inner launcher entrypoint"
  slurm_alloc_require_file "$activate_path" "runtime venv activate script"
  slurm_alloc_require_exec "${NANODISCOVER_EVAL_PYTHON}" "NANODISCOVER_EVAL_PYTHON"

  if [[ "${NANODISCOVER_ALLOC_INNER:-0}" == "1" ]]; then
    cd "$repo_root"
    slurm_alloc_prepare_submission_context "$repo_root" "$common_env_path"
    # shellcheck disable=SC1090
    source "$activate_path"
    if [[ "${SLURM_ALLOC_ALWAYS_RETRY}" != "1" ]]; then
      exec bash "$inner_entrypoint"
    fi

    local child_pid child_status
    child_pid=""
    SLURM_ALLOC_RETRY_SUBMITTED=0
    setsid bash "$inner_entrypoint" &
    child_pid=$!
    trap 'slurm_alloc_handle_term "$child_pid" "$wrapper_path" "$repo_root" "$common_env_path"' TERM
    set +e
    wait "$child_pid"
    child_status=$?
    set -e
    trap - TERM
    return "$child_status"
  fi

  cd "$repo_root"
  slurm_alloc_prepare_submission_context "$repo_root" "$common_env_path"

  local job_id
  job_id="$(slurm_alloc_submit_wrapper "$wrapper_path" "$repo_root")"

  echo "task=${NANODISCOVER_ALLOC_TASK_NAME}"
  echo "family=${NANODISCOVER_ALLOC_INNER_MODE}"
  echo "entrypoint=${NANODISCOVER_ALLOC_INNER_ENTRYPOINT}"
  if [[ -n "${NANODISCOVER_RESUME_DIR:-}" ]]; then
    echo "run_dir=$NANODISCOVER_RESUME_DIR"
  fi
  echo "job_id=$job_id"
  echo "stdout=$SLURM_ALLOC_STDOUT_PATH"
  echo "stderr=$SLURM_ALLOC_STDERR_PATH"
}

slurm_alloc_run
