# `single_node/`

This execution mode runs AC1 on one machine with no cluster scheduler. Sampling, generation, evaluation, archive update, and training all run on the same node.

Use this mode if you want the simplest setup or you are testing on a local machine. Be aware that evaluation can be slow: each AC1 rollout needs 2 CPU cores and can run for up to ~18 minutes, and a standard TTT-Discover-style run evaluates 512 rollouts per epoch. If you have many CPU cores available, most of that time can be parallelised.

## Quick start

```bash
cd /path/to/nanodiscover
source /path/to/nanodiscover-runtime-venv/bin/activate

export NANODISCOVER_AC1_CONFIG=qwen3_8b_4xL40S
export NANODISCOVER_EVAL_PYTHON=/path/to/nanodiscover-eval-ac1-venv/bin/python
export NANODISCOVER_LOG_ROOT=/path/to/your/log/root

# Fresh run
bash tasks/ac1/launchers/single_node/run_all.sh

# Resume an existing run
export NANODISCOVER_RESUME_DIR=/path/to/existing/run
bash tasks/ac1/launchers/single_node/run_all.sh
```

## Entrypoints

- `run_all.sh` — runs the full pipeline until `NANODISCOVER_NUM_EPOCHS` is reached. If the run dies, set `NANODISCOVER_RESUME_DIR` and call it again.
- `run_one_epoch.sh` — runs exactly one full epoch locally.
- `run_smoke.sh` — minimal one-epoch smoke test (1 seed, 2 rollouts) for validating that the stack works end-to-end before spending real compute.

## Notes

- `run_smoke.sh` overrides `NANODISCOVER_STAGE_STOP=train`, `NANODISCOVER_SEEDS_PER_EPOCH=1`, `NANODISCOVER_ROLLOUTS_PER_SEED=2`, `NANODISCOVER_SEQUENCE_PARALLEL_SIZE=2`, `NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE=2`, and `NANODISCOVER_GENERATOR_BATCH_SIZE=2` to make the run as fast as possible.
- This mode has no per-stage scripts. If you need stage-level control, use `slurm_attached/`.
- Common config loading happens through `tasks/ac1/launchers/common_ac1_env.sh`.
