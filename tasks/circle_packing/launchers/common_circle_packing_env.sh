#!/usr/bin/env bash
# shellcheck shell=bash

circle_packing_common_env_fail() {
  echo "$1" >&2
  return 2 2>/dev/null || exit 2
}

if [[ -z "${NANODISCOVER_CIRCLE_PACKING_CONFIG:-}" ]]; then
  circle_packing_common_env_fail "NANODISCOVER_CIRCLE_PACKING_CONFIG must be set to a Circle Packing config under tasks/circle_packing/launchers/configs/ (for example: qwen3_8b_8xL40S or gpt_oss_120b_8xH100)."
fi

circle_packing_config_path="tasks/circle_packing/launchers/configs/${NANODISCOVER_CIRCLE_PACKING_CONFIG}.sh"
if [[ ! -f "${circle_packing_config_path}" ]]; then
  circle_packing_common_env_fail "Unknown Circle Packing launcher config: ${NANODISCOVER_CIRCLE_PACKING_CONFIG} (expected ${circle_packing_config_path} to exist)."
fi

export NANODISCOVER_CIRCLE_PACKING_CONFIG
source "${circle_packing_config_path}"

if [[ -z "${NANODISCOVER_EVAL_PYTHON:-}" ]]; then
  circle_packing_common_env_fail "NANODISCOVER_EVAL_PYTHON must point to the dedicated math evaluator python for Circle Packing parity."
fi
if [[ ! -x "${NANODISCOVER_EVAL_PYTHON}" ]]; then
  circle_packing_common_env_fail "NANODISCOVER_EVAL_PYTHON is not executable: ${NANODISCOVER_EVAL_PYTHON}"
fi

export NANODISCOVER_EVAL_PYTHON

# Safety net: reduce allocator fragmentation risk on long-sequence training workloads.
# Custom configs that do not set this will get the safe default here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
