# AC1 Launchers

This folder contains launchers for running `nanodiscover` on the AC1 (1st autocorrelation inequality) task.

## Step 1: Pick a config

A config is a shell file under `configs/` that sets the model, hardware sizing, and key training hyperparameters together. Pick the one that matches your hardware, or make your own.

Configs ending in `_notrain.sh` disable TTT (no RL-based LoRA training). Useful for debugging or running ablations.

> **VRAM warning:** make sure your GPUs have enough VRAM to cover the model's full context window during training. For Qwen3-8B, our testing found that 4x L40S (48 GB each) works well. On 2x L40S with `NANODISCOVER_SEQUENCE_PARALLEL_SIZE=2`, you may hit OOM if `NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK` is set above ~8–9K. If you are running on 2-GPU hardware, lower that value in the config before launching. See the inline comments in `configs/` for a full explanation of what these variables mean.

## Step 2: Pick an execution mode

- **`slurm_attached/`** *(recommended, most tested)*: use this if you have interactive access to a GPU node on a SLURM cluster and can submit CPU array jobs from that node. Generation and training run on the GPU node while evaluation is parallelised across CPU array jobs.
  - It's important that your CPU array partition has a large number of cores. This keeps each epoch's evaluation time reasonable, as otherwise if you were to follow TTT-Discover's recipe, there are 512 rollouts generated per epoch. Each `ac1` rollout needs 2 CPU cores and can evaluate for up to ~20 minutes, so the wall time for just the evaluation stage of an epoch can quickly balloon to hours if you don't have enough cores.
  - Also, [as per the TTT-Discover authors' guidance](https://github.com/test-time-training/discover/blob/6c40e82dab9d5de7416ac873ad5cd3106084aaed/docs/reproducing.md?plain=1#L43), try to use server-grade CPUs as their clock speeds can affect the performance of the generated solutions.
- **`single_node/`**: runs everything on one machine with no cluster scheduler. Simple to set up, but evaluation can be very slow for the reason mentioned above.
- **`slurm_alloc/`**: use this if you are submitting from a login node and do not have interactive GPU access. Three sub-modes:
  - `attached/`: submits a long-lived GPU job and runs `slurm_attached` inside it once it starts.
  - `single_node/`: submits a long-lived GPU job and runs `single_node` inside it once it starts.
  - `staged/`: fully scheduler-driven staged pipeline (submits GPU job for generation, then CPU array job for evaluation, then GPU job for archive update and training, then repeats).

## Quick start

```bash
cd /path/to/nanodiscover
source /path/to/nanodiscover-runtime-venv/bin/activate

export NANODISCOVER_AC1_CONFIG=qwen3_8b_4xL40S
export NANODISCOVER_EVAL_PYTHON=/path/to/nanodiscover-eval-ac1-env/bin/python
export NANODISCOVER_LOG_ROOT=/path/to/your/log/root

bash tasks/ac1/launchers/slurm_attached/run_all.sh
```

To resume an existing run, set `NANODISCOVER_RESUME_DIR` instead of `NANODISCOVER_LOG_ROOT`:

```bash
export NANODISCOVER_RESUME_DIR=/path/to/existing/run
bash tasks/ac1/launchers/slurm_attached/run_all.sh
```

Swap `slurm_attached` for any other execution mode as needed.

## Entrypoints

- `run_all.sh` — runs until the configured number of epochs is complete. If the run dies, set `NANODISCOVER_RESUME_DIR` and call it again.
- `run_one_epoch.sh` — runs or submits exactly one epoch. Useful for step-by-step inspection.
- `run_smoke.sh` *(single_node only)* — 1 epoch, minimal rollouts, for verifying the stack works end-to-end.
- Stage scripts (`stage12.sh`, `stage3.sh`, `stage45.sh`, `stage*.sbatch.sh`) — advanced entrypoints for manual orchestration or recovery. Each assumes earlier pipeline stages have already run; prefer `run_all.sh` unless you are debugging.

## Run directories

Logs and artifacts are written under `NANODISCOVER_LOG_ROOT` for fresh runs, or `NANODISCOVER_RESUME_DIR` for resumed runs. On a cluster, symlink `logs/` to networked storage or scratch, as AC1 runs can produce large artifacts. For reference, a full 50-epoch run with checkpoints saved for the last 2 epochs can take around 120GB of space.
