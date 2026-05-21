# Tasks

Each task lives in its own subdirectory under `tasks/` and follows a consistent layout. This makes it straightforward to add new tasks and to understand what each one does.

## Adding a New Task

A new task needs:

1. `env.py`: must define a `build_task()` function. `main.py` loads tasks via `tasks.<task_name>.env:build_task()`.
2. `prompt.py`: this defines how the prompt construction logic.
3. `evaluator.py`: evaluation logic for candidate solutions.
4. `requirements.txt`: evaluator dependencies. Keep this separate from the root `requirements.txt`, which covers the generation/training runtime stack.
5. `launchers/`: at minimum a `single_node/` launcher family with `run_all.sh` and `run_one_epoch.sh`.

The test suite enforces that every task directory contains exactly the required Python files.

## Requirements Split

Each task has its own `requirements.txt` for the evaluator environment. This is kept separate from the root `requirements.txt` (the generation/training runtime) so that evaluator dependencies stay isolated and task-specific. For the current math tasks these evaluator requirements are intentionally lightweight and currently identical because they are based off of TTT-Discover's [requirements-math.txt](https://github.com/test-time-training/discover/blob/6c40e82dab9d5de7416ac873ad5cd3106084aaed/requirements/requirements-math.txt), but they remain task-local by design.