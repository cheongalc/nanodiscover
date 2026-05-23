# nanodiscover

`nanodiscover` is a reimplementation of [TTT-Discover](https://arxiv.org/abs/2601.16175) ([original code](https://github.com/test-time-training/discover)) that does not depend on the [Tinker API](https://thinkingmachines.ai/tinker/). It runs the same TTT-Discover-style automated discovery loop (sample parent solutions, generate children, evaluate, update the archive, train the proposer LLM via RL) while running on local GPUs.

We built it to make TTT-Discover-style research easier to run, inspect, and extend.

Instead of Tinker, `nanodiscover` uses [Ray Data LLM](https://docs.ray.io/en/latest/data/working-with-llms.html) for inference and [DeepSpeed](https://github.com/deepspeedai/deepspeed) for distributed training. Tensor parallel size > 1 is not yet supported. Our testing has used `Qwen3-8B`.

## Tasks

`nanodiscover` currently implements the 4 math tasks from the original TTT-Discover paper:

| Task | Description | Objective |
|------|-------------|-----------|
| `ac1` | 1st autocorrelation inequality | Minimize |
| `ac2` | 2nd autocorrelation inequality | Maximize |
| `circle_packing` | Circle packing (n=26 and n=32) | Maximize |
| `erdos` | Erdős minimum overlap problem | Minimize |

More tasks are in progress.

## Modes

**Non-strict mode (`main`):** This mode is designed to match TTT-Discover's behavior for the `ac1`, `ac2` and `erdos` tasks. The parent program's mathematical construction can be used directly by the child program.

**Strict mode (this branch, `strict`):** In this mode, the child program must produce its construction from scratch, without inheriting the parent construction.

## Results

We track results in a [live Google Sheet](https://docs.google.com/spreadsheets/d/1CeOXrNqZEOcNO3rZT8V8yalzYd5o2QJG8S-TZPL8n_w/).

This is a v0.1 infrastructure release. The sheet separates results by evidence status (live runs, audited runs, development snapshot, and pending runs) because prompt/config parity with the original TTT-Discover is still under audit. Treat the sheet as a live progress tracker, not a final benchmark table.

See [`KNOWN_DIFFERENCES.md`](KNOWN_DIFFERENCES.md) for a detailed account of known prompt and behavior differences from the original TTT-Discover.

## Setup

### Requirements

- Python 3.12.4 (we recommend using `uv` to manage your environment and tested with `uv` 0.11.5)
- One or more GPUs where your model of choice fits on a single card (no tensor parallelism yet)
- A separate evaluator Python environment per task (see below)

### Environment split

`nanodiscover` uses two separate Python environments on purpose:

- **Runtime env** (`requirements.txt`): used to launch generation and training. Install this in one venv.
- **Evaluator env** (`tasks/<task>/requirements.txt`): used to evaluate candidate solutions. Install this in a separate venv per task.

For the current math tasks the evaluator requirements are intentionally lightweight. Keeping them separate from the runtime stack ensures evaluator parity is not broken by runtime dependency updates.

```bash
# Create the runtime env (example using uv)
uv venv .venvs/nanodiscover-runtime --python 3.12
uv pip install -r requirements.txt --venv .venvs/nanodiscover-runtime

# Create the evaluator env for a given task (example: erdos)
uv venv .venvs/nanodiscover-eval-erdos --python 3.12
uv pip install -r tasks/erdos/requirements.txt --venv .venvs/nanodiscover-eval-erdos
```

### Required patch: Ray Data LLM LoRA fix

**This patch is required for TTT to work.** `ray[llm]==2.54.0` has a bug where LoRA adapters have no effect during generation, meaning the model never learns from training. With the runtime venv activated, apply the patch:

```bash
cp patches/vllm_engine_stage.py \
  "$(python -c 'import ray, os; print(os.path.dirname(ray.__file__))')/llm/_internal/batch/stages/vllm_engine_stage.py"
```

See [`patches/README.md`](patches/README.md) for details and the upstream fix.

## Running

Each task has launchers under `tasks/<task>/launchers/`. The launcher families are:

- `single_node/`: use this if you're running on a single node (can be on a SLURM cluster or not)
- `slurm_attached/`: use this if you're running on a SLURM cluster, you have interactive access to a GPU node, and your cluster also has an array partition with many CPU cores (this is important for running evaluation jobs in parallel so they take a reasonable amount of time)
- `slurm_alloc/`: use this if you're running on a SLURM cluster and you don't have interactive access to a GPU node, but your cluster still has the array partition with many CPU cores

Each launcher family has a `run_all.sh` (full multi-epoch run), `run_one_epoch.sh`, and for single-node a `run_smoke.sh` (1 epoch, minimal config, for verifying the stack works).

GPU config presets live in `tasks/<task>/launchers/configs/`. Pick the one matching your hardware (e.g. `qwen3_8b_4xH100.sh`), or make your own.

### Quick smoke test

Quick smoke test for Erdos on a single node:

```bash
cd /path/to/nanodiscover

# Activate runtime env
source .venvs/nanodiscover-runtime/bin/activate

# Set evaluator python
export NANODISCOVER_EVAL_PYTHON=/path/to/.venvs/nanodiscover-eval-erdos/bin/python

# Set Erdos GPU config
export NANODISCOVER_ERDOS_CONFIG=tasks/erdos/launchers/configs/qwen3_8b_4xH100.sh

# Set where you want the run to save its artifacts
export NANODISCOVER_LOG_ROOT=/path/to/your/log/root
# Alternatively, you can export NANODISCOVER_RESUME_DIR=/path/to/your/existing/run to resume from an existing run. This will automatically set NANODISCOVER_LOG_ROOT to the parent directory of the existing run.

bash tasks/erdos/launchers/single_node/run_smoke.sh
```

## Ray temp directory

`nanodiscover` uses Ray for inference. Ray uses UNIX-domain sockets under its temp root, so keep the temp path short. `nanodiscover` picks a temp root in this order: `NANODISCOVER_RAY_TMPDIR` (if set), `SLURM_TMPDIR`, `TMPDIR` and finally `/tmp/$USER`. Automatic temp roots are cleaned up on teardown; explicit overrides are left as-is.

## Acknowledgements

`nanodiscover` is built on top of the ideas introduced in [TTT-Discover](https://arxiv.org/abs/2601.16175). We are grateful for their work and for their openness to follow-on implementations.
