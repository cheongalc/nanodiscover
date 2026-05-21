#!/usr/bin/env bash
# shellcheck shell=bash

erdos_common_env_fail() {
  echo "$1" >&2
  return 2 2>/dev/null || exit 2
}

if [[ -z "${NANODISCOVER_ERDOS_CONFIG:-}" ]]; then
  erdos_common_env_fail "NANODISCOVER_ERDOS_CONFIG must be set to an Erdos config under tasks/erdos/launchers/configs/ (for example: qwen3_8b_8xL40S or gpt_oss_120b_8xH100)."
fi

erdos_config_path="tasks/erdos/launchers/configs/${NANODISCOVER_ERDOS_CONFIG}.sh"
if [[ ! -f "${erdos_config_path}" ]]; then
  erdos_common_env_fail "Unknown Erdos launcher config: ${NANODISCOVER_ERDOS_CONFIG} (expected ${erdos_config_path} to exist)."
fi

export NANODISCOVER_ERDOS_CONFIG
source "${erdos_config_path}"

if [[ -z "${NANODISCOVER_EVAL_PYTHON:-}" ]]; then
  erdos_common_env_fail "NANODISCOVER_EVAL_PYTHON must point to the dedicated math evaluator python for Erdos parity."
fi
if [[ ! -x "${NANODISCOVER_EVAL_PYTHON}" ]]; then
  erdos_common_env_fail "NANODISCOVER_EVAL_PYTHON is not executable: ${NANODISCOVER_EVAL_PYTHON}"
fi

export NANODISCOVER_EVAL_PYTHON

# Safety net: reduce allocator fragmentation risk on long-sequence training workloads.
# Custom configs that do not set this will get the safe default here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
