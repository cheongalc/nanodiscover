#!/usr/bin/env bash
# shellcheck shell=bash

# These defaults were set for a specific cluster configuration.
# You MUST override these values to match your own cluster before using this mode.
# Set the relevant variables in your shell before calling the launcher script,
# or edit this file directly.

: "${SLURM_ALLOC_GPU_PARTITION:=preempt}"
: "${SLURM_ALLOC_GPU_QOS:=}"
: "${SLURM_ALLOC_GPU_TIME:=2-00:00:00}"
: "${SLURM_ALLOC_GPU_GRES:=gpu:L40S:4}"
: "${SLURM_ALLOC_GPU_CPUS_PER_TASK:=32}"
: "${SLURM_ALLOC_GPU_MEM:=96G}"
: "${SLURM_ALLOC_GPU_NODELIST:=}"
: "${SLURM_ALLOC_GPU_COMMENT:=PROFILER_DISABLE}"
: "${SLURM_ALLOC_ENABLE_REQUEUE:=1}"
: "${SLURM_ALLOC_ALWAYS_RETRY:=0}"
: "${SLURM_ALLOC_SIGNAL_SECONDS:=120}"
: "${SLURM_ALLOC_OUTPUT_PATTERN:=}"

export SLURM_ALLOC_GPU_PARTITION
export SLURM_ALLOC_GPU_QOS
export SLURM_ALLOC_GPU_TIME
export SLURM_ALLOC_GPU_GRES
export SLURM_ALLOC_GPU_CPUS_PER_TASK
export SLURM_ALLOC_GPU_MEM
export SLURM_ALLOC_GPU_NODELIST
export SLURM_ALLOC_GPU_COMMENT
export SLURM_ALLOC_ENABLE_REQUEUE
export SLURM_ALLOC_ALWAYS_RETRY
export SLURM_ALLOC_SIGNAL_SECONDS
export SLURM_ALLOC_OUTPUT_PATTERN
