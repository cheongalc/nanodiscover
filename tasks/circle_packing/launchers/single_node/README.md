# `single_node/`

This execution mode runs circle packing on one machine with no cluster scheduler. Sampling, generation, evaluation, archive update, and training all run on the same node.

Use this mode if you want the simplest setup or you are testing on a local machine. Be aware that evaluation can be slow: each circle packing rollout uses 1 CPU core and can evaluate for up to ~530 seconds, and a full epoch has 512 rollouts. If you have many CPU cores available, most of that time can be parallelised.

## Quick start

```bash
cd /path/to/nanodiscover
source /path/to/nanodiscover-runtime-venv/bin/activate

export NANODISCOVER_CIRCLE_PACKING_CONFIG=qwen3_8b_4xL40S
export NANODISCOVER_EVAL_PYTHON=/path/to/nanodiscover-eval-circle-packing-venv/bin/python
export NANODISCOVER_LOG_ROOT=/path/to/your/log/root
export NANODISCOVER_CIRCLE_PACKING_N=26  # or 32
# Fresh run
bash tasks/circle_packing/launchers/single_node/run_all.sh

# Resume an existing run
export NANODISCOVER_RESUME_DIR=/path/to/existing/run
bash tasks/circle_packing/launchers/single_node/run_all.sh
```

## Entrypoints

- `run_all.sh` — runs the full pipeline until `NANODISCOVER_NUM_EPOCHS` is reached. If the run dies, set `NANODISCOVER_RESUME_DIR` and call it again.
- `run_one_epoch.sh` — runs exactly one full epoch locally.
- `run_smoke.sh` — minimal one-epoch smoke test for validating that the stack works end-to-end before spending real compute.

## Notes

- This mode has no per-stage scripts. If you need stage-level control, use `slurm_attached/`.
- Common config loading happens through `tasks/circle_packing/launchers/common_circle_packing_env.sh`.
- Also set `NANODISCOVER_CIRCLE_PACKING_N` to choose the problem size (default: 26).
