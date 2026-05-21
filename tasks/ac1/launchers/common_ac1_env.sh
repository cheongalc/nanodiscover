#!/usr/bin/env bash
# shellcheck shell=bash

ac1_common_env_fail() {
  echo "$1" >&2
  return 2 2>/dev/null || exit 2
}

if [[ -z "${NANODISCOVER_AC1_CONFIG:-}" ]]; then
  ac1_common_env_fail "NANODISCOVER_AC1_CONFIG must be set to an AC1 config under tasks/ac1/launchers/configs/ (for example: qwen3_8b_8xL40S or gpt_oss_120b_8xH100)."
fi

ac1_config_path="tasks/ac1/launchers/configs/${NANODISCOVER_AC1_CONFIG}.sh"
if [[ ! -f "${ac1_config_path}" ]]; then
  ac1_common_env_fail "Unknown AC1 launcher config: ${NANODISCOVER_AC1_CONFIG} (expected ${ac1_config_path} to exist)."
fi

export NANODISCOVER_AC1_CONFIG
source "${ac1_config_path}"

if [[ -z "${NANODISCOVER_EVAL_PYTHON:-}" ]]; then
  ac1_common_env_fail "NANODISCOVER_EVAL_PYTHON must point to the dedicated math evaluator python for AC1 parity."
fi
if [[ ! -x "${NANODISCOVER_EVAL_PYTHON}" ]]; then
  ac1_common_env_fail "NANODISCOVER_EVAL_PYTHON is not executable: ${NANODISCOVER_EVAL_PYTHON}"
fi

export NANODISCOVER_EVAL_PYTHON
# Safety net: reduce allocator fragmentation risk on long-sequence training workloads.
# Custom configs that do not set this will get the safe default here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
