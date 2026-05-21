# `slurm_attached/`

Use this mode when:

- you are already on a GPU node inside a Slurm cluster (interactively)
- you can also launch CPU array jobs from that node

Generation and training run directly on the GPU node you are on. For evaluation, the script splits work between the node's local CPUs and remote Slurm CPU array jobs, so all 512 rollouts can be evaluated in parallel. This is the mode that has been most tested.

Note: each Erdős rollout uses 1 CPU core and can evaluate for up to ~530 seconds, and a full epoch has 512 rollouts. Make sure your CPU array partition has enough cores to keep evaluation time reasonable.

## Quick start

```bash
cd /path/to/nanodiscover
source /path/to/nanodiscover-runtime-venv/bin/activate

export NANODISCOVER_ERDOS_CONFIG=qwen3_8b_4xL40S
export NANODISCOVER_EVAL_PYTHON=/path/to/nanodiscover-eval-erdos-venv/bin/python
export NANODISCOVER_LOG_ROOT=/path/to/your/log/root

# Fresh run
bash tasks/erdos/launchers/slurm_attached/run_all.sh

# Resume an existing run
export NANODISCOVER_RESUME_DIR=/path/to/existing/run
bash tasks/erdos/launchers/slurm_attached/run_all.sh

# Run exactly one epoch
bash tasks/erdos/launchers/slurm_attached/run_one_epoch.sh
```

## Entrypoints

- `run_all.sh` — recommended for most users. Keeps calling `run_one_epoch.sh` until the run is complete.
- `run_one_epoch.sh` — runs exactly one epoch. Useful for step-by-step inspection.
- `stage12.sh` — advanced: sampling and generation only.
- `stage3.sh` — advanced: evaluation only (handles Slurm spillover automatically).
- `stage45.sh` — advanced: archive update and training only.

Stage scripts assume earlier pipeline stages have already run for the target epoch. Use them only for manual recovery or debugging, not as a normal entrypoint.

## Scheduler knobs

This mode reads scheduler and evaluation settings from `profile_env.sh`.

> **Cluster-specific:** the defaults in `profile_env.sh` were set for a specific cluster. You **must** override `SLURM_ATTACHED_REMOTE_PARTITIONS` and `SLURM_ATTACHED_REMOTE_QOS` to match your cluster's CPU partition and QOS names. The other values may also need adjustment. Set them in your shell before calling the launcher, or edit `profile_env.sh` directly.

The main knobs are:

| Variable | What it controls |
|---|---|
| `SLURM_ATTACHED_REMOTE_PARTITIONS` | CPU partition(s) to submit eval array jobs to |
| `SLURM_ATTACHED_REMOTE_QOS` | QOS for remote eval jobs |
| `SLURM_ATTACHED_REMOTE_TIME` | Wall time limit for each remote eval job |
| `SLURM_ATTACHED_REMOTE_MEM_PER_TASK` | Memory per task for remote eval jobs |
| `SLURM_ATTACHED_CPUS_PER_EVAL` | CPUs per eval shard (1 for Erdős) |
| `SLURM_ATTACHED_REMOTE_WORKERS_PER_TASK` | Eval workers per remote array task |
| `SLURM_ATTACHED_REMOTE_PARALLEL_EVALS_PER_TASK` | Parallel evals per remote array task |
| `SLURM_ATTACHED_REMOTE_MAX_ARRAY_TASKS` | Max concurrent remote array tasks |
| `SLURM_ATTACHED_LOCAL_CPUS` | CPUs available locally for eval |
| `SLURM_ATTACHED_LOCAL_INITIAL_SHARDS` | Initial shards dispatched to local CPUs |
| `SLURM_ATTACHED_LOCAL_PARALLEL_JOBS` | Parallel local eval jobs |
| `SLURM_ATTACHED_EVAL_SHARD_SIZE` | Rollouts per eval shard |

## Notes

- `stage3.sh` handles Slurm spillover automatically — you do not need to submit eval jobs by hand.
- Evaluator logs go to `epochXXX/evaluator.log`; per-rollout stdout and failures go to `epochXXX/evaluation_logs/`.
- Both local and remote stage-3 shard workers use `NANODISCOVER_EVAL_PYTHON` for `python -m core.evaluator evaluate-shard`.
- Common config loading happens through `tasks/erdos/launchers/common_erdos_env.sh`.
