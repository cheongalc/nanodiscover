# `slurm_alloc/staged/`

Use this mode when you have access to a Slurm cluster but cannot sit interactively on the GPU node. Everything is submitted through the scheduler as a dependency chain.

A typical epoch looks like:

1. GPU job: sampling and generation (`stage12.sbatch.sh`)
2. CPU array job: evaluation shards (`stage3_array.sbatch.sh`)
3. CPU job: merge evaluation shards (`stage3_merge.sbatch.sh`)
4. GPU job: archive update and training (`stage45.sbatch.sh`)

`run_one_epoch.sh` builds and submits this chain for you. `run_all.sh` calls it repeatedly until the configured number of epochs is done.

## Quick start

```bash
cd /path/to/nanodiscover
source /path/to/nanodiscover-runtime-venv/bin/activate

export NANODISCOVER_AC1_CONFIG=qwen3_8b_4xL40S
export NANODISCOVER_EVAL_PYTHON=/path/to/nanodiscover-eval-ac1-venv/bin/python
export NANODISCOVER_LOG_ROOT=/path/to/your/log/root

# Submit all remaining epochs for a fresh run
bash tasks/ac1/launchers/slurm_alloc/staged/run_all.sh

# Resume an existing run and submit the remaining epochs
export NANODISCOVER_RESUME_DIR=/path/to/existing/run
bash tasks/ac1/launchers/slurm_alloc/staged/run_all.sh

# Submit exactly one epoch
bash tasks/ac1/launchers/slurm_alloc/staged/run_one_epoch.sh
```

## Entrypoints

- `run_all.sh` — recommended for most users. Chains epochs by making each epoch depend on the previous epoch's final training job.
- `run_one_epoch.sh` — submits exactly one epoch as a dependency chain.
- `stage12.sbatch.sh`, `stage3_array.sbatch.sh`, `stage3_merge.sbatch.sh`, `stage45.sbatch.sh` — lower-level scheduler helpers used internally by `run_one_epoch.sh`. Most users should not need to call these directly.

## Scheduler knobs

Edit `profile_env.sh` to match your cluster. The main knobs are:

| Variable | What it controls |
|---|---|
| `SLURM_ALLOC_STAGED_GPU_PARTITION` | Partition for GPU jobs |
| `SLURM_ALLOC_STAGED_CPU_PARTITION` | Partition for CPU eval array jobs |
| `SLURM_ALLOC_STAGED_GPU_TIME` | Wall time limit for GPU jobs |
| `SLURM_ALLOC_STAGED_EVAL_TIME` | Wall time limit for each eval array task |
| `SLURM_ALLOC_STAGED_EVAL_CPUS_PER_TASK` | CPUs per eval task (2 for AC1) |
| `SLURM_ALLOC_STAGED_EVAL_MEM_PER_TASK` | Memory per eval task |
| `SLURM_ALLOC_STAGED_JOB_PREFIX` | Prefix for submitted job names |
| `SLURM_ALLOC_STAGED_STDOUT_PATH` | Path for job stdout logs |
| `SLURM_ALLOC_STAGED_STDERR_PATH` | Path for job stderr logs |

The old `SLURM_BATCH_*` variable names are accepted as fallbacks for compatibility.

## Notes

- When `SLURM_ALLOC_STAGED_STDOUT_PATH` and `SLURM_ALLOC_STAGED_STDERR_PATH` are unset, scheduler logs are written to `NANODISCOVER_RESUME_DIR/slurm/`.
- Common config loading happens through `tasks/ac1/launchers/common_ac1_env.sh`.
