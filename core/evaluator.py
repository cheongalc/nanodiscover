from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from core.archive import ArchiveNode


logger = logging.getLogger(__name__)


def coerce_int_list(values: Any) -> list[int]:
    """Return a list coerced to integers."""

    return [int(value) for value in values or []]


def coerce_float_list(values: Any) -> list[float]:
    """Return a list coerced to floats."""

    return [float(value) for value in values or []]


def read_json_payload(path: Path) -> dict[str, Any]:
    """Read a JSON payload from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def json_default_for_numpy(obj: Any) -> Any:
    """Coerce numpy scalars and arrays to native Python types for json.dumps.

    User-provided candidate code (called from task evaluator workers) commonly
    returns lists containing numpy scalars (e.g. ``list(np.array(...))``), which
    leak through ``result_payload`` into the shard JSON written here. ``np.int64``
    in particular is not a subclass of ``int`` on every platform and crashes the
    default ``json`` encoder.
    """

    try:
        import numpy as np
    except ImportError:  # numpy is part of every task eval env, but be defensive.
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload to disk with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default_for_numpy),
        encoding="utf-8",
    )


def update_best_raw_score(
    best_raw_score: float | None,
    raw_score: float,
    *,
    maximize_raw_score: bool,
) -> float:
    """Update a tracked best raw score with one candidate value."""

    if best_raw_score is None:
        return raw_score
    if maximize_raw_score and raw_score > best_raw_score:
        return raw_score
    if not maximize_raw_score and raw_score < best_raw_score:
        return raw_score
    return best_raw_score


@dataclass
class EvaluatorConfig:
    """Executor configuration for rollout evaluation."""

    max_workers: int
    #: When set, this process's worker slots are sliced from ``available_cpu_ids()`` starting at
    #: ``cpu_pack_slot * (max_workers * cpus_per_eval)``, so sibling ``evaluate-shard`` jobs under
    #: one Slurm step (each with ``--workers 1``) map to disjoint CPUs instead of all using slot 0.
    cpu_pack_slot: int | None = None


@dataclass(frozen=True)
class TaskEvaluationRequirements:
    """Task-declared resource limits for one rollout evaluation."""

    cpus_per_eval: int = 1
    max_concurrent_evaluations: int | None = None


@dataclass(frozen=True)
class AllocatedEvaluationResources:
    """Concrete CPU resources allocated to one evaluation worker."""

    cpu_ids: tuple[int, ...] = ()
    slot_index: int = 0

    @property
    def thread_count(self) -> int:
        return max(1, len(self.cpu_ids))


@dataclass
class EvaluatedRollout:
    """Evaluation result for a single generated rollout.

    ``eval_wall_time_s`` is set by :func:`evaluate_rollout` and measures wall time for
    ``task.evaluate_code`` only (not parsing, reward, or next-state construction).
    """

    seed_state: ArchiveNode
    prompt_text: str
    response_text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    completion_mask: list[float]
    finish_reason: str | None
    parsed_code: str
    reward: float
    correctness: float
    performance: float
    raw_score: float | None
    archive_value: float | None
    next_state: ArchiveNode | None
    msg: str
    result_payload: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    #: Wall time in seconds for ``task.evaluate_code`` only (excludes ``parse_code`` and reward / next_state).
    eval_wall_time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_state": self.seed_state.to_dict(),
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "completion_logprobs": list(self.completion_logprobs),
            "completion_mask": list(self.completion_mask),
            "finish_reason": self.finish_reason,
            "parsed_code": self.parsed_code,
            "reward": float(self.reward),
            "correctness": float(self.correctness),
            "performance": float(self.performance),
            "raw_score": self.raw_score,
            "archive_value": self.archive_value,
            "next_state": self.next_state.to_dict() if self.next_state is not None else None,
            "msg": self.msg,
            "result_payload": dict(self.result_payload),
            "stdout": self.stdout,
            "eval_wall_time_s": self.eval_wall_time_s,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluatedRollout":
        next_state_payload = payload.get("next_state")
        return cls(
            seed_state=ArchiveNode.from_dict(dict(payload.get("seed_state") or {})),
            prompt_text=str(payload.get("prompt_text", "")),
            response_text=str(payload.get("response_text", "")),
            prompt_token_ids=coerce_int_list(payload.get("prompt_token_ids")),
            completion_token_ids=coerce_int_list(payload.get("completion_token_ids")),
            completion_logprobs=coerce_float_list(payload.get("completion_logprobs")),
            completion_mask=coerce_float_list(payload.get("completion_mask")),
            finish_reason=(str(payload["finish_reason"]) if payload.get("finish_reason") is not None else None),
            parsed_code=str(payload.get("parsed_code", "")),
            reward=float(payload.get("reward", 0.0)),
            correctness=float(payload.get("correctness", 0.0)),
            performance=float(payload.get("performance", 0.0)),
            raw_score=(float(payload["raw_score"]) if payload.get("raw_score") is not None else None),
            archive_value=(float(payload["archive_value"]) if payload.get("archive_value") is not None else None),
            next_state=(ArchiveNode.from_dict(dict(next_state_payload)) if isinstance(next_state_payload, dict) else None),
            msg=str(payload.get("msg", "")),
            result_payload=dict(payload.get("result_payload") or {}),
            stdout=str(payload.get("stdout", "")),
            eval_wall_time_s=(
                float(payload["eval_wall_time_s"]) if payload.get("eval_wall_time_s") is not None else None
            ),
        )


def normalize_task_evaluation_requirements(value: Any) -> TaskEvaluationRequirements:
    """Normalize task-provided evaluation resource hints."""

    if isinstance(value, TaskEvaluationRequirements):
        return TaskEvaluationRequirements(
            cpus_per_eval=max(1, int(value.cpus_per_eval)),
            max_concurrent_evaluations=(
                None
                if value.max_concurrent_evaluations is None
                else max(1, int(value.max_concurrent_evaluations))
            ),
        )
    # Two loader paths (e.g. mixed site-packages + PYTHONPATH) can produce distinct
    # ``TaskEvaluationRequirements`` class objects; accept duck-typed values.
    if value is not None and hasattr(value, "cpus_per_eval"):
        try:
            max_concurrent = getattr(value, "max_concurrent_evaluations", None)
            return TaskEvaluationRequirements(
                cpus_per_eval=max(1, int(getattr(value, "cpus_per_eval"))),
                max_concurrent_evaluations=(
                    None if max_concurrent is None else max(1, int(max_concurrent))
                ),
            )
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        max_concurrent = value.get("max_concurrent_evaluations")
        return TaskEvaluationRequirements(
            cpus_per_eval=max(1, int(value.get("cpus_per_eval", 1))),
            max_concurrent_evaluations=(None if max_concurrent is None else max(1, int(max_concurrent))),
        )
    return TaskEvaluationRequirements()


def resolve_task_evaluation_requirements(task: Any) -> TaskEvaluationRequirements:
    """Resolve task-specific evaluation resource hints with safe defaults."""

    getter = getattr(task, "evaluation_resources", None)
    if callable(getter):
        try:
            return normalize_task_evaluation_requirements(getter())
        except Exception:  # noqa: BLE001
            logger.exception("failed to resolve task evaluation resources; falling back to defaults")
    return TaskEvaluationRequirements()


def available_cpu_ids() -> tuple[int, ...]:
    """Return the CPU ids visible to the current process."""

    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0)))
    cpu_count = os.cpu_count() or 1
    return tuple(range(max(1, int(cpu_count))))


def resolve_cpu_pack_slot(cli_value: int | None) -> int | None:
    """Return shard sibling CPU slice index from CLI or ``NANODISCOVER_EVAL_CPU_PACK_SLOT``."""

    if cli_value is not None:
        return int(cli_value)
    raw = os.environ.get("NANODISCOVER_EVAL_CPU_PACK_SLOT", "").strip()
    if not raw:
        return None
    return int(raw)


class EvaluationResourcePool:
    """Allocate and recycle CPU slots for concurrent evaluations."""

    def __init__(
        self,
        requirements: TaskEvaluationRequirements,
        *,
        requested_workers: int,
        cpu_pack_slot: int | None = None,
    ) -> None:
        self.requirements = requirements
        self.cpu_ids = available_cpu_ids()
        slot_count = max(1, len(self.cpu_ids) // max(1, requirements.cpus_per_eval))
        if requirements.max_concurrent_evaluations is not None:
            slot_count = min(slot_count, max(1, int(requirements.max_concurrent_evaluations)))
        self.max_workers = max(1, min(max(1, int(requested_workers)), slot_count))
        block_cpus = self.max_workers * requirements.cpus_per_eval
        base = 0
        if cpu_pack_slot is not None:
            if int(cpu_pack_slot) < 0:
                raise ValueError(f"cpu_pack_slot must be >= 0, got {cpu_pack_slot!r}")
            base = int(cpu_pack_slot) * block_cpus
            need = base + block_cpus
            if need > len(self.cpu_ids):
                raise ValueError(
                    f"cpu_pack_slot={cpu_pack_slot} needs {need} visible cpus "
                    f"(base={base}, block_cpus={block_cpus}) but affinity has only {len(self.cpu_ids)}; "
                    "check Slurm --cpus-per-task vs parallel shard count and --workers"
                )
        self._slots: queue.Queue[AllocatedEvaluationResources] = queue.Queue()
        for slot_index in range(self.max_workers):
            start = base + slot_index * requirements.cpus_per_eval
            stop = start + requirements.cpus_per_eval
            slot_cpu_ids = self.cpu_ids[start:stop]
            if not slot_cpu_ids:
                slot_cpu_ids = self.cpu_ids[:1]
            self._slots.put(AllocatedEvaluationResources(cpu_ids=tuple(slot_cpu_ids), slot_index=slot_index))

    @contextmanager
    def acquire(self):
        """Yield one reusable evaluation slot from the pool."""

        slot = self._slots.get()
        try:
            yield slot
        finally:
            self._slots.put(slot)


def call_task_evaluate_code(
    *,
    task: Any,
    parsed_code: str,
    seed_state: ArchiveNode,
    epoch: int,
    seed: int,
    resources: AllocatedEvaluationResources,
) -> dict[str, Any]:
    """Call `task.evaluate_code`, passing resources only when supported."""

    evaluate_code = task.evaluate_code
    kwargs = {
        "parsed_code": parsed_code,
        "state": seed_state,
        "epoch": epoch,
        "seed": seed,
    }
    try:
        signature = inspect.signature(evaluate_code)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "resources" in signature.parameters:
        kwargs["resources"] = resources
    return evaluate_code(**kwargs)


def evaluate_rollout(
    *,
    task: Any,
    seed_state: ArchiveNode,
    prompt_text: str,
    generation: Any,
    epoch: int,
    seed: int,
    resources: AllocatedEvaluationResources,
) -> EvaluatedRollout:
    """Evaluate one rollout and convert it into an `EvaluatedRollout`."""

    eval_wall_time_s: float | None = None
    parsed_code = ""
    next_state = None
    eval_output: dict[str, Any]
    try:
        parsed_code = str(task.parse_code(generation.response_text) or "")
        _eval_t0 = time.perf_counter()
        try:
            eval_output = call_task_evaluate_code(
                task=task,
                parsed_code=parsed_code,
                seed_state=seed_state,
                epoch=epoch,
                seed=seed,
                resources=resources,
            )
        finally:
            eval_wall_time_s = time.perf_counter() - _eval_t0
        reward = float(task.compute_reward(eval_output))
    except Exception as exc:  # noqa: BLE001
        eval_output = {
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": None,
            "msg": f"evaluation failed: {exc}",
            "result_payload": {},
            "stdout": "",
        }
        reward = 0.0

    if float(eval_output.get("correctness", 0.0)) > 0:
        try:
            next_state = task.make_next_state(
                parent_state=seed_state,
                parsed_code=parsed_code,
                eval_output=eval_output,
                epoch=epoch,
            )
        except Exception as exc:  # noqa: BLE001
            eval_output = dict(eval_output)
            detail = str(eval_output.get("msg", "") or "")
            suffix = f"next_state creation failed: {exc}"
            eval_output["msg"] = f"{detail}; {suffix}" if detail else suffix
    return EvaluatedRollout(
        seed_state=seed_state,
        prompt_text=prompt_text,
        response_text=generation.response_text,
        prompt_token_ids=list(generation.prompt_token_ids),
        completion_token_ids=list(generation.completion_token_ids),
        completion_logprobs=list(generation.completion_logprobs),
        completion_mask=list(generation.completion_mask),
        finish_reason=generation.finish_reason,
        parsed_code=parsed_code,
        reward=reward,
        correctness=float(eval_output.get("correctness", 0.0)),
        performance=float(eval_output.get("performance", 0.0)),
        raw_score=(float(eval_output["raw_score"]) if eval_output.get("raw_score") is not None else None),
        archive_value=(float(next_state.value) if next_state is not None and next_state.value is not None else None),
        next_state=next_state,
        msg=str(eval_output.get("msg", "")),
        result_payload=dict(eval_output.get("result_payload") or {}),
        stdout=str(eval_output.get("stdout", "")),
        eval_wall_time_s=eval_wall_time_s,
    )


def evaluate_rollout_with_resource_pool(
    *,
    resource_pool: EvaluationResourcePool,
    task: Any,
    seed_state: ArchiveNode,
    prompt_text: str,
    generation: Any,
    epoch: int,
    seed: int,
) -> EvaluatedRollout:
    """Evaluate one rollout after acquiring resources from the shared pool."""

    with resource_pool.acquire() as resources:
        return evaluate_rollout(
            task=task,
            seed_state=seed_state,
            prompt_text=prompt_text,
            generation=generation,
            epoch=epoch,
            seed=seed,
            resources=resources,
        )


class Evaluator:
    """Run rollout evaluation concurrently on local worker threads."""

    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config

    def evaluate_batch(
        self,
        *,
        task: Any,
        seed_states: list[ArchiveNode],
        prompts: list[str],
        generations: list[Any],
        epoch: int,
        base_seed: int,
        on_rollout_completed: Callable[[int, EvaluatedRollout], None] | None = None,
    ) -> list[EvaluatedRollout]:
        """Evaluate a batch of generations and return ordered rollout results."""

        if not (len(seed_states) == len(prompts) == len(generations)):
            raise ValueError("seed_states, prompts, and generations must align")
        maximize_raw_score = bool(getattr(task, "maximize_raw_score", True))
        total = len(generations)
        results: list[EvaluatedRollout | None] = [None] * total
        completed = 0
        progress_interval = max(1, total // 10)
        started_at = time.perf_counter()
        reward_sum = 0.0
        correctness_sum = 0.0
        next_state_count = 0
        timeout_count = 0
        parse_failure_count = 0
        evaluation_failure_count = 0
        best_raw_score: float | None = None
        requirements = resolve_task_evaluation_requirements(task)
        resource_pool = EvaluationResourcePool(
            requirements,
            requested_workers=self.config.max_workers,
            cpu_pack_slot=self.config.cpu_pack_slot,
        )
        logger.info(
            "evaluator_progress start total=%d requested_max_workers=%d effective_max_workers=%d cpus_per_eval=%d reserved_cpus=%d progress_interval=%d epoch=%d",
            total,
            self.config.max_workers,
            resource_pool.max_workers,
            requirements.cpus_per_eval,
            len(resource_pool.cpu_ids),
            progress_interval,
            epoch,
        )
        with ThreadPoolExecutor(max_workers=resource_pool.max_workers) as executor:
            submitted = {
                executor.submit(
                    evaluate_rollout_with_resource_pool,
                    resource_pool=resource_pool,
                    task=task,
                    seed_state=seed_state,
                    prompt_text=prompt_text,
                    generation=generation,
                    epoch=epoch,
                    seed=base_seed + index,
                ): index
                for index, (seed_state, prompt_text, generation) in enumerate(zip(seed_states, prompts, generations, strict=True))
            }
            for future in as_completed(submitted):
                index = submitted[future]
                results[index] = future.result()
                result = results[index]
                if result is not None and on_rollout_completed is not None:
                    try:
                        on_rollout_completed(index, result)
                    except Exception:  # noqa: BLE001
                        logger.exception("evaluator_progress failed to persist rollout stream index=%d", index)
                completed += 1
                if result is not None:
                    reward_sum += float(result.reward)
                    correctness_sum += float(result.correctness)
                    next_state_count += int(result.next_state is not None)
                    if not result.parsed_code.strip():
                        parse_failure_count += 1
                    elif result.correctness <= 0:
                        evaluation_failure_count += 1
                    if str(result.msg).startswith("timeout after "):
                        timeout_count += 1
                    if result.raw_score is not None:
                        best_raw_score = update_best_raw_score(
                            best_raw_score,
                            float(result.raw_score),
                            maximize_raw_score=maximize_raw_score,
                        )
                if completed == total or completed % progress_interval == 0:
                    logger.info(
                        "evaluator_progress completed=%d total=%d pct=%.1f elapsed_s=%.2f reward_mean=%.4f correctness_mean=%.4f next_states=%d timeouts=%d parse_failures=%d evaluation_failures=%d best_raw_score=%s",
                        completed,
                        total,
                        100.0 * completed / max(1, total),
                        time.perf_counter() - started_at,
                        reward_sum / max(1, completed),
                        correctness_sum / max(1, completed),
                        next_state_count,
                        timeout_count,
                        parse_failure_count,
                        evaluation_failure_count,
                        "none" if best_raw_score is None else f"{best_raw_score:.4f}",
                    )
        return [result for result in results if result is not None]

    def teardown(self) -> None:
        pass


@dataclass
class ShardGeneration:
    """Minimal generation payload used by the external evaluator CLI."""

    response_text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    completion_mask: list[float]
    finish_reason: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ShardGeneration":
        """Build a shard generation object from serialized generation output."""

        return cls(
            response_text=str(payload.get("response_text", "")),
            prompt_token_ids=coerce_int_list(payload.get("prompt_token_ids")),
            completion_token_ids=coerce_int_list(payload.get("completion_token_ids")),
            completion_logprobs=coerce_float_list(payload.get("completion_logprobs")),
            completion_mask=coerce_float_list(payload.get("completion_mask")),
            finish_reason=(str(payload["finish_reason"]) if payload.get("finish_reason") is not None else None),
        )


def build_task(task_name: str) -> Any:
    """Import a task module and call its public `build_task` factory."""

    module = importlib.import_module(f"tasks.{task_name}.env")
    build_fn = getattr(module, "build_task")
    return build_fn()


def epoch_dir(run_dir: Path, epoch: int) -> Path:
    """Return the canonical epoch directory path."""

    return run_dir / f"epoch{int(epoch):03d}"


def load_sample_payload(run_dir: Path, epoch: int) -> tuple[list[ArchiveNode], list[str]]:
    """Load seed states and prompts for one epoch."""

    payload = read_json_payload(epoch_dir(run_dir, epoch) / "sample.json")
    seed_states = [ArchiveNode.from_dict(item) for item in payload.get("seed_states", [])]
    prompts = [str(item) for item in payload.get("prompts", [])]
    return seed_states, prompts


def load_generation_payload(run_dir: Path, epoch: int) -> list[ShardGeneration]:
    """Load serialized generations for one epoch."""

    payload = read_json_payload(epoch_dir(run_dir, epoch) / "generation.json")
    return [ShardGeneration.from_dict(item) for item in payload.get("generations", [])]


def expand_seed_states_and_prompts_for_generations(
    seed_states: list[ArchiveNode],
    prompts: list[str],
    generations: list[ShardGeneration],
) -> tuple[list[ArchiveNode], list[str], int]:
    """Repeat seed states and prompts so they align one-to-one with generations."""

    if not seed_states or not prompts:
        raise RuntimeError("sample.json does not contain seed states/prompts")
    if len(seed_states) != len(prompts):
        raise RuntimeError(f"seed/prompt mismatch: {len(seed_states)} vs {len(prompts)}")
    if len(generations) % len(seed_states) != 0:
        raise RuntimeError(
            f"generation count {len(generations)} must be divisible by seeds_per_epoch {len(seed_states)}"
        )
    rollouts_per_seed = len(generations) // len(seed_states)
    repeated_seed_states = [state for state in seed_states for _ in range(rollouts_per_seed)]
    repeated_prompts = [prompt for prompt in prompts for _ in range(rollouts_per_seed)]
    if len(repeated_seed_states) != len(generations):
        raise RuntimeError(
            f"sample/generation cardinality mismatch: seeds={len(seed_states)} prompts={len(prompts)} generations={len(generations)}"
        )
    return repeated_seed_states, repeated_prompts, rollouts_per_seed


def build_rollout_shard_payload(
    *,
    epoch: int,
    start_index: int,
    evaluated: list[EvaluatedRollout],
) -> dict[str, Any]:
    """Serialize evaluated rollouts into the shard payload schema."""

    return {
        "epoch": int(epoch),
        "items": [
            {
                "index": start_index + index,
                "rollout": rollout.to_dict(),
            }
            for index, rollout in enumerate(evaluated)
        ],
    }


def load_rollouts_by_index(payload: dict[str, Any], *, source: str) -> dict[int, EvaluatedRollout]:
    """Parse the common indexed-rollout payload schema into a dict keyed by index."""

    by_index: dict[int, EvaluatedRollout] = {}
    for item in payload.get("items", []):
        try:
            index = int(item["index"])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{source} contains an item without a valid integer index") from exc
        if index in by_index:
            raise RuntimeError(f"duplicate rollout index in {source}: {index}")
        rollout_payload = item.get("rollout")
        if not isinstance(rollout_payload, dict):
            raise RuntimeError(f"{source} is missing a rollout payload for index {index}")
        by_index[index] = EvaluatedRollout.from_dict(dict(rollout_payload))
    return by_index


def load_single_rollout_shard(
    path: Path,
    *,
    expected_index: int | None = None,
) -> EvaluatedRollout:
    """Load a shard file that must contain exactly one indexed rollout.

    Args:
        path: Path to the shard JSON file.
        expected_index: Optional rollout index that must be present in the shard.

    Returns:
        The single evaluated rollout carried by the shard file.

    Raises:
        RuntimeError: If the shard does not contain exactly one rollout or the
            rollout index does not match ``expected_index``.
    """

    by_index = load_rollouts_by_index(read_json_payload(path), source=str(path))
    if len(by_index) != 1:
        raise RuntimeError(f"{path} must contain exactly one rollout item")
    index, rollout = next(iter(by_index.items()))
    if expected_index is not None and index != int(expected_index):
        raise RuntimeError(
            f"{path} contains rollout index {index}, expected {int(expected_index)}"
        )
    return rollout


def scan_shard_status(
    shard_dir: Path,
    *,
    expected_total: int,
    delete_invalid: bool = False,
) -> dict[str, Any]:
    """Classify shard files into complete, invalid, and pending indices.

    Args:
        shard_dir: Directory containing ``shard_<idx>.json`` files.
        expected_total: Number of rollout indices that must exist.
        delete_invalid: Whether to delete invalid shard files after detecting
            them so the caller can safely requeue those indices.

    Returns:
        A dict containing counts plus the pending indices list.
    """

    resolved_shard_dir = shard_dir.resolve()
    complete_indices: list[int] = []
    invalid_indices: list[int] = []
    pending_indices: list[int] = []
    for index in range(int(expected_total)):
        shard_path = resolved_shard_dir / f"shard_{index}.json"
        if not shard_path.exists():
            pending_indices.append(index)
            continue
        try:
            load_single_rollout_shard(shard_path, expected_index=index)
        except Exception:
            invalid_indices.append(index)
            pending_indices.append(index)
            if delete_invalid:
                shard_path.unlink(missing_ok=True)
            continue
        complete_indices.append(index)
    return {
        "complete_count": len(complete_indices),
        "invalid_count": len(invalid_indices),
        "pending_indices": pending_indices,
    }


def order_rollouts_by_index(
    by_index: dict[int, EvaluatedRollout],
    *,
    expected_total: int | None,
    source: str,
) -> list[EvaluatedRollout]:
    """Validate completeness and return rollouts ordered by rollout index."""

    if expected_total is not None:
        missing = [index for index in range(int(expected_total)) if index not in by_index]
        if missing:
            preview = ", ".join(str(index) for index in missing[:10])
            raise RuntimeError(f"{source} is missing rollout indices: {preview}")
        if len(by_index) != int(expected_total):
            raise RuntimeError(
                f"{source} returned {len(by_index)} rollouts, expected {int(expected_total)}"
            )
    return [rollout for _, rollout in sorted(by_index.items(), key=lambda item: item[0])]


def load_evaluated_rollouts_from_path(
    path: Path,
    *,
    expected_total: int | None = None,
) -> list[EvaluatedRollout]:
    """Load indexed evaluated rollouts from a shard-style payload file."""

    payload = read_json_payload(path)
    by_index = load_rollouts_by_index(payload, source=str(path))
    return order_rollouts_by_index(by_index, expected_total=expected_total, source=str(path))


def evaluate_external_shard(
    *,
    task_name: str,
    run_dir: Path,
    epoch: int,
    start: int,
    stop: int,
    workers: int,
    output_path: Path,
    cpu_pack_slot: int | None = None,
) -> dict[str, Any]:
    """Evaluate one rollout index slice and write the shard payload to disk."""

    started_at = time.perf_counter()
    task = build_task(task_name)
    resolved_run_dir = run_dir.resolve()
    resolved_output_path = output_path.resolve()
    pack_slot = resolve_cpu_pack_slot(cpu_pack_slot)
    try:
        _tasks_env_py = getattr(importlib.import_module(f"tasks.{task_name}.env"), "__file__", "?")
    except Exception:
        _tasks_env_py = "?"
    _req_preview = resolve_task_evaluation_requirements(task)
    _eval_res = getattr(task, "evaluation_resources", None)
    logger.info(
        "external_eval_task_meta task=%s cpus_per_eval=%d tasks_env_py=%s evaluation_resources_callable=%s",
        task_name,
        int(_req_preview.cpus_per_eval),
        _tasks_env_py,
        callable(_eval_res),
    )
    logger.info(
        "external_eval_start task=%s run_dir=%s epoch=%d start=%d stop=%d workers=%d cpu_pack_slot=%s output=%s",
        task_name,
        str(resolved_run_dir),
        int(epoch),
        int(start),
        int(stop),
        max(1, int(workers)),
        "none" if pack_slot is None else str(pack_slot),
        str(resolved_output_path),
    )

    seed_states, prompts = load_sample_payload(resolved_run_dir, epoch)
    generations = load_generation_payload(resolved_run_dir, epoch)
    repeated_seed_states, repeated_prompts, rollouts_per_seed = expand_seed_states_and_prompts_for_generations(
        seed_states,
        prompts,
        generations,
    )
    logger.info(
        "external_eval_loaded seeds=%d prompts=%d generations=%d rollouts_per_seed=%d",
        len(seed_states),
        len(prompts),
        len(generations),
        rollouts_per_seed,
    )

    shard_start = max(0, int(start))
    shard_stop = min(len(generations), int(stop))
    if shard_stop <= shard_start:
        payload = {"epoch": int(epoch), "items": []}
        write_json_payload(resolved_output_path, payload)
        return payload

    evaluator = Evaluator(
        EvaluatorConfig(max_workers=max(1, int(workers)), cpu_pack_slot=pack_slot),
    )
    try:
        evaluated = evaluator.evaluate_batch(
            task=task,
            seed_states=repeated_seed_states[shard_start:shard_stop],
            prompts=repeated_prompts[shard_start:shard_stop],
            generations=generations[shard_start:shard_stop],
            epoch=int(epoch),
            base_seed=int(epoch) * len(seed_states) * rollouts_per_seed + shard_start,
        )
    finally:
        evaluator.teardown()

    payload = build_rollout_shard_payload(
        epoch=int(epoch),
        start_index=shard_start,
        evaluated=evaluated,
    )
    write_json_payload(resolved_output_path, payload)
    logger.info(
        "external_eval_complete epoch=%d evaluated=%d output=%s elapsed_s=%.3f",
        int(epoch),
        len(evaluated),
        str(resolved_output_path),
        time.perf_counter() - started_at,
    )
    return payload


def merge_evaluation_shards(
    *,
    run_dir: Path,
    epoch: int,
    shard_dir: Path,
    expected_total: int | None = None,
) -> dict[str, Any]:
    """Merge shard payload files into the canonical epoch evaluation artifacts."""

    import utils

    subdir = utils.epoch_subdir(run_dir.resolve(), int(epoch))
    resolved_shard_dir = shard_dir.resolve()
    by_index: dict[int, EvaluatedRollout] = {}
    source_by_index: dict[int, Path] = {}
    for path in sorted(resolved_shard_dir.glob("shard_*.json")):
        shard_rollouts = load_rollouts_by_index(read_json_payload(path), source=str(path))
        for index, rollout in shard_rollouts.items():
            if index in by_index:
                raise RuntimeError(
                    f"Duplicate rollout index {index} found in shard files "
                    f"{source_by_index[index]} and {path}"
                )
            by_index[index] = rollout
            source_by_index[index] = path

    evaluated = order_rollouts_by_index(
        by_index,
        expected_total=expected_total,
        source=str(resolved_shard_dir),
    )
    files_written, total_bytes = utils.persist_evaluation_streams(
        subdir,
        epoch=int(epoch),
        evaluated=evaluated,
    )
    utils.save_evaluations(subdir, epoch=int(epoch), evaluated=evaluated)
    return {
        "epoch": int(epoch),
        "evaluated": len(evaluated),
        "stream_files": files_written,
        "stream_mb": total_bytes / (1024.0 * 1024.0),
        "output": str(subdir.evaluation),
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the evaluator CLI parser."""

    parser = argparse.ArgumentParser(description="Nanodiscover evaluation helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate-shard",
        help="Evaluate one rollout shard for a nanodiscover epoch",
    )
    evaluate_parser.add_argument("--task", default="ac1", help="Task module name under tasks/<task>/env.py")
    evaluate_parser.add_argument("--run-dir", required=True)
    evaluate_parser.add_argument("--epoch", required=True, type=int)
    evaluate_parser.add_argument("--start", required=True, type=int, help="Inclusive rollout index")
    evaluate_parser.add_argument("--stop", required=True, type=int, help="Exclusive rollout index")
    evaluate_parser.add_argument("--workers", default=1, type=int)
    evaluate_parser.add_argument(
        "--cpu-pack-slot",
        default=None,
        type=int,
        help=(
            "Index (0..parallel-1) for this evaluate-shard among concurrent siblings on one allocation; "
            "offsets CPU slices so packed Slurm tasks do not all pin to the first cores. "
            "If omitted, uses NANODISCOVER_EVAL_CPU_PACK_SLOT when set."
        ),
    )
    evaluate_parser.add_argument("--output", required=True)

    merge_parser = subparsers.add_parser(
        "merge-shards",
        help="Merge rollout-evaluation shard files into evaluation.json",
    )
    merge_parser.add_argument("--run-dir", required=True)
    merge_parser.add_argument("--epoch", required=True, type=int)
    merge_parser.add_argument("--shard-dir", required=True)
    merge_parser.add_argument("--expected-total", type=int, default=None)
    return parser


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluator CLI arguments."""

    return build_cli_parser().parse_args(argv)


def run_cli_command(args: argparse.Namespace) -> None:
    """Run one evaluator CLI subcommand after output has been routed."""

    if args.command == "evaluate-shard":
        evaluate_external_shard(
            task_name=str(args.task),
            run_dir=Path(args.run_dir),
            epoch=int(args.epoch),
            start=int(args.start),
            stop=int(args.stop),
            workers=max(1, int(args.workers)),
            output_path=Path(args.output),
            cpu_pack_slot=args.cpu_pack_slot,
        )
        return

    summary = merge_evaluation_shards(
        run_dir=Path(args.run_dir),
        epoch=int(args.epoch),
        shard_dir=Path(args.shard_dir),
        expected_total=(
            None if args.expected_total is None else int(args.expected_total)
        ),
    )
    print(json.dumps(summary, sort_keys=True))


def run_cli_command_with_epoch_logging(args: argparse.Namespace) -> None:
    """Route evaluator CLI output into the epoch-local `evaluator.log` file."""

    import utils

    run_dir = Path(args.run_dir).resolve()
    log_path = epoch_dir(run_dir, int(args.epoch)) / "evaluator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with utils.capture_stage_output(log_path):
        run_cli_command(args)


def main(argv: list[str] | None = None) -> None:
    """Run the evaluator CLI."""

    args = parse_cli_args(argv)
    run_cli_command_with_epoch_logging(args)


if __name__ == "__main__":
    main()
