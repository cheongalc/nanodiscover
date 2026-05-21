#!/usr/bin/env bash
# shellcheck shell=bash

# These defaults were set for a specific cluster configuration.
# You MUST override these values to match your own cluster before using this mode.
# Set the relevant variables in your shell before calling the launcher script,
# or edit this file directly.

: "${SLURM_ALLOC_STAGED_GPU_PARTITION:=${SLURM_BATCH_GPU_PARTITION:-general}}"
: "${SLURM_ALLOC_STAGED_CPU_PARTITION:=${SLURM_BATCH_CPU_PARTITION:-array}}"
: "${SLURM_ALLOC_STAGED_GPU_TIME:=${SLURM_BATCH_GPU_TIME:-08:00:00}}"
: "${SLURM_ALLOC_STAGED_EVAL_TIME:=${SLURM_BATCH_EVAL_TIME:-00:40:00}}"
: "${SLURM_ALLOC_STAGED_EVAL_CPUS_PER_TASK:=${SLURM_BATCH_EVAL_CPUS_PER_TASK:-2}}"
: "${SLURM_ALLOC_STAGED_EVAL_MEM_PER_TASK:=${SLURM_BATCH_EVAL_MEM_PER_TASK:-1G}}"

export SLURM_ALLOC_STAGED_GPU_PARTITION
export SLURM_ALLOC_STAGED_CPU_PARTITION
export SLURM_ALLOC_STAGED_GPU_TIME
export SLURM_ALLOC_STAGED_EVAL_TIME
export SLURM_ALLOC_STAGED_EVAL_CPUS_PER_TASK
export SLURM_ALLOC_STAGED_EVAL_MEM_PER_TASK
