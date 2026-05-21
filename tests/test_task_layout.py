import importlib
from pathlib import Path


TASKS_ROOT = Path(__file__).resolve().parents[1] / "tasks"
REQUIRED_PY_FILES = {"env.py", "prompt.py", "evaluator.py"}
REQUIRED_NON_PY_FILES = {"requirements.txt"}


def _task_dirs() -> list[Path]:
	return sorted(
		path
		for path in TASKS_ROOT.iterdir()
		if path.is_dir() and not path.name.startswith("__")
	)


def test_all_tasks_share_same_python_file_set():
	for task_dir in _task_dirs():
		py_files = {path.name for path in task_dir.iterdir() if path.is_file() and path.suffix == ".py"}
		assert py_files == REQUIRED_PY_FILES, f"{task_dir.name} has python files {sorted(py_files)} instead of {sorted(REQUIRED_PY_FILES)}"


def test_all_tasks_have_required_non_python_files():
	for task_dir in _task_dirs():
		file_names = {path.name for path in task_dir.iterdir() if path.is_file()}
		missing = REQUIRED_NON_PY_FILES - file_names
		assert not missing, f"{task_dir.name} is missing required files: {sorted(missing)}"


def test_all_tasks_expose_build_task_factory():
	for task_dir in _task_dirs():
		module = importlib.import_module(f"tasks.{task_dir.name}.env")
		task = module.build_task()
		assert getattr(task, "name", "").strip().lower() == task_dir.name