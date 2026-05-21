# `slurm_alloc/`

Use this launcher family when you are **not** already sitting on a GPU compute node and need Slurm to allocate resources for you (e.g. you are on a login node).

Three sub-modes are available:

- **`attached/`** — submits one long-lived GPU allocation job; once it starts on a compute node, runs the `slurm_attached` launcher inside that allocation.
- **`single_node/`** — submits one long-lived GPU allocation job; once it starts, runs the `single_node` launcher inside that allocation.
- **`staged/`** — fully scheduler-driven staged pipeline: separate GPU job for generation, CPU array jobs for evaluation, then GPU job for archive update and training. See `staged/README.md` for details.

## Required environment

Before submitting any `attached/` or `single_node/` job, export:

```bash
export NANODISCOVER_RUNTIME_VENV=/path/to/nanodiscover-runtime-venv
export NANODISCOVER_ERDOS_CONFIG=qwen3_8b_4xL40S
export NANODISCOVER_EVAL_PYTHON=/path/to/nanodiscover-eval-erdos-venv/bin/python
```

Any other variables you set before submission (`NANODISCOVER_LOG_ROOT`, `NANODISCOVER_RESUME_DIR`, task-specific overrides) are forwarded to the allocation job via `sbatch --export=ALL`.

## Scheduler log paths

To control where Slurm writes the allocation job's stdout/stderr:

```bash
export SLURM_ALLOC_STDOUT_PATH=/path/to/stdout.log
export SLURM_ALLOC_STDERR_PATH=/path/to/stderr.log
```

When unset, `attached/` and `single_node/` default to `${NANODISCOVER_RESUME_DIR}/slurm/` for resumed runs or `${NANODISCOVER_LOG_ROOT}/slurm_alloc/` for fresh runs.

## Preemption handling

For preempt-style queues, set `SLURM_ALLOC_ALWAYS_RETRY=1`. This disables native Slurm requeue and instead makes the wrapper submit a fresh copy of itself when the allocation receives a TERM warning. A stable `NANODISCOVER_RESUME_DIR` is pinned automatically for fresh launches so the replacement allocation resumes the same run instead of starting over.

## Scheduler defaults

> **Cluster-specific:** `profile_env.sh` in this directory contains defaults that were set for a specific cluster (partition names, GRES strings, memory). You **must** override these to match your own cluster. Either set the relevant variables in your shell before calling the launcher, or edit `profile_env.sh` directly. The key variables to check are `SLURM_ALLOC_GPU_PARTITION`, `SLURM_ALLOC_GPU_GRES`, and `SLURM_ALLOC_GPU_MEM`.

- Defaults for `attached/` and `single_node/` live in `profile_env.sh` in this directory.
- Defaults for `staged/` live in `staged/profile_env.sh`.
