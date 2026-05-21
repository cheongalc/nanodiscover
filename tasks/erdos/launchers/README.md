# Erdős Launchers

This folder contains launchers for running `nanodiscover` on the Erdős minimum overlap problem task.

## Step 1: Pick a config

A config is a shell file under `configs/` that sets the model, hardware sizing, and key training hyperparameters together. Pick the one that matches your hardware, or make your own.

Configs ending in `_notrain.sh` disable TTT (no RL-based LoRA training). Configs ending in `_hightemp.sh` use higher sampling temperature.

> **VRAM warning:** make sure your GPUs have enough VRAM to cover the model's full context window during training. For Qwen3-8B, our testing found that 4x L40S (48 GB each) works well. On 2x L40S with `NANODISCOVER_SEQUENCE_PARALLEL_SIZE=2`, you may hit OOM if `NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK` is set above ~8–9K. If you are running on 2-GPU hardware, lower that value in the config before launching. See the inline comments in `configs/` for a full explanation of what these variables mean.

## Step 2: Pick an execution mode

- **`slurm_attached/`** *(recommended, most tested)*: use this if you have interactive access to a GPU node on a SLURM cluster and can submit CPU array jobs from that node. Generation and training run on the GPU node while evaluation is parallelised across CPU array jobs.
  - Each Erdős rollout uses 1 CPU core and can evaluate for up to ~530 seconds, and a full epoch has 512 rollouts. A large CPU array partition keeps evaluation time reasonable.
- **`single_node/`**: runs everything on one machine with no cluster scheduler. Simple to set up, but evaluation will be slow — 512 rollouts x up to 530s each on a single machine's CPUs.
- **`slurm_alloc/`**: use this if you are submitting from a login node and do not have interactive GPU access. Three sub-modes:
  - `attached/`: submits a long-lived GPU job and runs `slurm_attached` inside it once it starts.
  - `single_node/`: submits a long-lived GPU job and runs `single_node` inside it once it starts.
  - `staged/`: fully scheduler-driven staged pipeline (submits GPU job for generation, then CPU array job for evaluation, then GPU job for archive update and training, then repeats).

## Quick start

```bash
cd /path/to/nanodiscover
source /path/to/nanodiscover-runtime-venv/bin/activate

export NANODISCOVER_ERDOS_CONFIG=qwen3_8b_4xL40S
export NANODISCOVER_EVAL_PYTHON=/path/to/nanodiscover-eval-erdos-env/bin/python
export NANODISCOVER_LOG_ROOT=/path/to/your/log/root

bash tasks/erdos/launchers/slurm_attached/run_all.sh
```

To resume an existing run, set `NANODISCOVER_RESUME_DIR` instead of `NANODISCOVER_LOG_ROOT`:

```bash
export NANODISCOVER_RESUME_DIR=/path/to/existing/run
bash tasks/erdos/launchers/slurm_attached/run_all.sh
```

Swap `slurm_attached` for any other execution mode as needed.

## Entrypoints

- `run_all.sh` — runs until the configured number of epochs is complete. If the run dies, set `NANODISCOVER_RESUME_DIR` and call it again.
- `run_one_epoch.sh` — runs or submits exactly one epoch. Useful for step-by-step inspection.
- `run_smoke.sh` *(single_node only)* — 1 epoch, minimal rollouts, for verifying the stack works end-to-end.
- Stage scripts (`stage12.sh`, `stage3.sh`, `stage45.sh`, `stage*.sbatch.sh`) — advanced entrypoints for manual orchestration or recovery. Each assumes earlier pipeline stages have already run; prefer `run_all.sh` unless you are debugging.

## Run directories

Logs and artifacts are written under `NANODISCOVER_LOG_ROOT` for fresh runs, or `NANODISCOVER_RESUME_DIR` for resumed runs. On a cluster, symlink `logs/` to networked storage or scratch, as Erdős runs can produce large artifacts.
