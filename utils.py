from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from config import epoch_subdir_path, latest_epoch
from core.archive import Archive, ArchiveNode
from core.evaluator import EvaluatedRollout, json_default_for_numpy
from core.generator import GenerationOutput


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


@dataclass(frozen=True)
class EpochSubdir:
    """Canonical file layout for a single epoch directory."""

    root: Path
    sample: Path
    generation: Path
    evaluation: Path
    evaluation_logs: Path
    archive: Path
    sampler: Path
    training: Path
    adapter: Path

    def has_sample(self) -> bool:
        return self.sample.exists()

    def has_generation(self) -> bool:
        return self.generation.exists()

    def has_evaluation(self) -> bool:
        return self.evaluation.exists()

    def has_archive_checkpoint(self) -> bool:
        return self.archive.exists()

    def has_sampler_checkpoint(self) -> bool:
        return self.sampler.exists()

    def has_state_checkpoints(self) -> bool:
        return self.has_archive_checkpoint() and self.has_sampler_checkpoint()

    def has_training_result(self) -> bool:
        return self.training.exists()

    def has_adapter_dir(self) -> bool:
        return self.adapter.exists()


def epoch_subdir(run_dir: str | Path, epoch: int) -> EpochSubdir:
    """Return the canonical file bundle for one epoch directory."""

    root = epoch_subdir_path(run_dir, epoch)
    return EpochSubdir(
        root=root,
        sample=root / "sample.json",
        generation=root / "generation.json",
        evaluation=root / "evaluation.json",
        evaluation_logs=root / "evaluation_logs",
        archive=root / "archive.json",
        sampler=root / "sampler.json",
        training=root / "training.json",
        adapter=root / "adapter",
    )


def epoch_indices(run_dir: Path, *, reverse: bool = False) -> list[int]:
    """Return the known epoch indices for a run directory."""

    last_epoch = latest_epoch(run_dir)
    if last_epoch is None:
        return []
    indices = list(range(last_epoch + 1))
    return list(reversed(indices)) if reverse else indices


def configure_run_logging(run_dir: Path) -> None:
    """Redirect process stdout, stderr, and logging into `run_dir/log.txt`."""

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = (run_dir / "log.txt").open("a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(log_file)],
        force=True,
    )


@contextmanager
def capture_stage_output(log_path: Path):
    """Route process output and Python logging to a stage-specific log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    old_handlers = list(root_logger.handlers)
    old_level = root_logger.level
    old_stdout_obj = sys.stdout
    old_stderr_obj = sys.stderr
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    stage_file = log_path.open("a", encoding="utf-8", buffering=1)
    try:
        os.dup2(stage_file.fileno(), 1)
        os.dup2(stage_file.fileno(), 2)
        sys.stdout = stage_file
        sys.stderr = stage_file

        handler = logging.StreamHandler(stage_file)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.handlers = [handler]
        root_logger.setLevel(logging.INFO)

        yield
    finally:
        for handler in root_logger.handlers:
            handler.flush()
        root_logger.handlers = old_handlers
        root_logger.setLevel(old_level)

        sys.stdout = old_stdout_obj
        sys.stderr = old_stderr_obj

        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        stage_file.flush()
        stage_file.close()


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default_for_numpy),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def archive_snapshot(archive: Archive) -> dict[str, object]:
    """Serialize the archive state needed for epoch checkpoints."""

    return {
        "epoch": archive.current_epoch,
        "states": [state.to_dict() for state in archive.states],
        "initial_states": [state.to_dict() for state in archive.initial_states],
    }


def restore_archive_snapshot(archive: Archive, payload: dict[str, object]) -> None:
    """Restore archive state from an epoch checkpoint payload."""

    archive.current_epoch = int(payload.get("epoch", archive.current_epoch))
    archive.states = [ArchiveNode.from_dict(item) for item in payload.get("states", [])]
    archive.initial_states = [ArchiveNode.from_dict(item) for item in payload.get("initial_states", [])]


def save_sample(subdir: EpochSubdir, *, epoch: int, archive: Archive, seed_states: list[ArchiveNode], prompts: list[str]) -> None:
    """Persist sample-stage outputs for later resume."""

    write_json(
        subdir.sample,
        {
            "epoch": epoch,
            "archive": archive_snapshot(archive),
            "seed_states": [state.to_dict() for state in seed_states],
            "prompts": list(prompts),
        },
    )


def load_sample(subdir: EpochSubdir) -> tuple[dict[str, object], list[ArchiveNode], list[str]]:
    """Load sample-stage artifacts from disk."""

    payload = read_json(subdir.sample)
    return (
        dict(payload.get("archive") or {}),
        [ArchiveNode.from_dict(item) for item in payload.get("seed_states", [])],
        [str(item) for item in payload.get("prompts", [])],
    )


def save_generations(subdir: EpochSubdir, *, epoch: int, generations: list[GenerationOutput]) -> None:
    """Persist generation-stage outputs for an epoch."""

    write_json(subdir.generation, {"epoch": epoch, "generations": [item.to_dict() for item in generations]})


def load_generations(subdir: EpochSubdir) -> list[GenerationOutput]:
    """Load generation-stage outputs from disk."""

    return [GenerationOutput.from_dict(item) for item in read_json(subdir.generation).get("generations", [])]


def save_evaluations(subdir: EpochSubdir, *, epoch: int, evaluated: list[EvaluatedRollout]) -> None:
    """Persist evaluator outputs for an epoch."""

    write_json(subdir.evaluation, {"epoch": epoch, "evaluated": [item.to_dict() for item in evaluated]})


def resolve_rollout_stream_text(rollout: EvaluatedRollout) -> str:
    """Return rollout stdout, synthesizing a failure message when needed."""

    stream_text = str(rollout.stdout or "")
    if stream_text.strip():
        return stream_text
    message = str(rollout.msg or "").strip()
    if rollout.correctness <= 0 and message:
        synthesized = f"[evaluator]\n{message}\n"
        rollout.stdout = synthesized
        return synthesized
    return ""


def persist_evaluation_stream(
    subdir: EpochSubdir,
    *,
    index: int,
    rollout: EvaluatedRollout,
    inline_limit_chars: int = 8192,
) -> int:
    """Write one rollout's evaluation stream to disk and annotate its payload."""

    subdir.evaluation_logs.mkdir(parents=True, exist_ok=True)
    stream_text = resolve_rollout_stream_text(rollout)
    if not stream_text:
        return 0

    log_name = f"rollout_{index:06d}.log"
    log_path = subdir.evaluation_logs / log_name
    log_path.write_text(stream_text, encoding="utf-8")
    stream_bytes = len(stream_text.encode("utf-8"))

    payload = dict(rollout.result_payload or {})
    payload["evaluation_log"] = str(log_path.relative_to(subdir.root))
    rollout.result_payload = payload

    if len(stream_text) > inline_limit_chars:
        tail = stream_text[-inline_limit_chars:]
        rollout.stdout = (
            f"[truncated; full output in {payload['evaluation_log']}]\n"
            f"... tail ({inline_limit_chars} chars) ...\n{tail}"
        )
    return stream_bytes


def persist_evaluation_streams(
    subdir: EpochSubdir,
    *,
    epoch: int,
    evaluated: list[EvaluatedRollout],
    inline_limit_chars: int = 8192,
) -> tuple[int, int]:
    """Persist all rollout streams for an epoch and return file/byte counts."""

    files_written = 0
    total_bytes = 0
    for index, rollout in enumerate(evaluated):
        stream_bytes = persist_evaluation_stream(
            subdir,
            index=index,
            rollout=rollout,
            inline_limit_chars=inline_limit_chars,
        )
        if stream_bytes > 0:
            files_written += 1
            total_bytes += stream_bytes
    return files_written, total_bytes


def load_evaluations(subdir: EpochSubdir) -> list[EvaluatedRollout]:
    """Load evaluator outputs from disk."""

    return [EvaluatedRollout.from_dict(item) for item in read_json(subdir.evaluation).get("evaluated", [])]


def save_training_result(subdir: EpochSubdir, *, epoch: int, result) -> None:
    """Persist the trainer result payload for an epoch."""

    write_json(
        subdir.training,
        {
            "epoch": epoch,
            "metrics": dict(getattr(result, "metrics", {}) or {}),
            "adapter_path": getattr(result, "adapter_path", None),
            "optimizer_state_dir": getattr(result, "optimizer_state_dir", None),
            "skipped": bool(getattr(result, "skipped", False)),
        },
    )


def load_training_result(subdir: EpochSubdir) -> dict[str, object]:
    """Load the trainer result payload for an epoch."""

    return read_json(subdir.training)


def stage_stop_reached(subdir: EpochSubdir, stage_stop: str) -> bool:
    """Return whether the epoch has reached the configured terminal stage."""

    if stage_stop == "sample":
        return subdir.has_sample()
    if stage_stop == "generate":
        return subdir.has_generation()
    if stage_stop == "evaluate":
        return subdir.has_evaluation()
    if stage_stop == "archive_update":
        return subdir.has_state_checkpoints()
    if stage_stop == "train":
        return subdir.has_training_result()
    raise ValueError(f"Unsupported stage_stop: {stage_stop!r}")


def resume_epoch(run_dir: Path, stage_stop: str = "train") -> int:
    """Return the next epoch index that should run for a resume."""

    epochs = epoch_indices(run_dir)
    if not epochs:
        return 0
    last_epoch = epochs[-1]
    subdir = epoch_subdir(run_dir, last_epoch)
    return last_epoch + 1 if stage_stop_reached(subdir, stage_stop) else last_epoch


def resume_adapter(run_dir: Path) -> str | None:
    """Return the latest canonical epoch-local adapter path for a resume."""

    for epoch in epoch_indices(run_dir, reverse=True):
        subdir = epoch_subdir(run_dir, epoch)
        training_path = subdir.training
        if not training_path.exists():
            continue
        if subdir.has_adapter_dir():
            return str(subdir.adapter)
    return None


def resume_optimizer_state(run_dir: Path) -> str | None:
    """Return the latest canonical epoch-local optimizer state directory for a resume."""

    for epoch in epoch_indices(run_dir, reverse=True):
        subdir = epoch_subdir(run_dir, epoch)
        training_path = subdir.training
        if not training_path.exists():
            continue
        optimizer_state_dir = subdir.root / "optimizer_state"
        if optimizer_state_dir.exists():
            return str(optimizer_state_dir)
    return None


def load_best_raw_score(run_dir: Path, maximize_raw_score: bool) -> float | None:
    """Replay stored evaluations to recover the best raw score so far."""

    best_raw_score: float | None = None
    for epoch in epoch_indices(run_dir):
        evaluation_path = epoch_subdir(run_dir, epoch).evaluation
        if not evaluation_path.exists():
            continue
        for item in read_json(evaluation_path).get("evaluated", []):
            correctness = item.get("correctness") if isinstance(item, dict) else None
            if correctness is None or float(correctness) <= 0:
                continue
            raw_score = item.get("raw_score") if isinstance(item, dict) else None
            if raw_score is None:
                continue
            value = float(raw_score)
            if best_raw_score is None:
                best_raw_score = value
            elif maximize_raw_score and value > best_raw_score:
                best_raw_score = value
            elif not maximize_raw_score and value < best_raw_score:
                best_raw_score = value
    return best_raw_score