from __future__ import annotations

import importlib
import logging
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)

from core.archive import ArchiveConfig, Archive
from core.evaluator import EvaluatedRollout, EvaluatorConfig, Evaluator, load_evaluated_rollouts_from_path
from core.generator import GeneratorConfig, Generator
from core.sampler import SamplerConfig, Sampler
from core.trainer import TrainerConfig, Trainer, validate_adapter_checkpoint
from config import (
    RunConfig,
    STAGE_ORDER,
    load_run_config,
    resolve_reference_scoring_model_parallel_size,
    validate_run_config,
)
import utils


def configure_runtime_compat_env() -> None:
    """Set runtime environment defaults needed by the public launch path."""

    # Ray worker startup installs uvloop by default when available. Disabling it
    # avoids "no current event loop" crashes in Ray Data actor initialization.
    os.environ.setdefault("RAY_USE_UVLOOP", "0")
    # Keep accelerator visibility behavior stable and silence upcoming-default warning.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    # Pydantic plugin auto-discovery can pull in incompatible optional plugins.
    os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")

    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = transformers_cache
        # Remove deprecated env var to avoid repeated FutureWarning spam.
        os.environ.pop("TRANSFORMERS_CACHE", None)


def stage_enabled(config: RunConfig, stage_name: str) -> bool:
    """Return whether the named stage is inside the requested stage scope."""

    stage_index = STAGE_ORDER.index(stage_name)
    return STAGE_ORDER.index(config.stage_start) <= stage_index <= STAGE_ORDER.index(config.stage_stop)


def stage_stops_before(config: RunConfig, stage_name: str) -> bool:
    """Return whether the configured stage scope ends before the named stage."""

    return STAGE_ORDER.index(config.stage_stop) < STAGE_ORDER.index(stage_name)


def resolve_evaluator_python() -> str | None:
    """Return the configured external evaluator interpreter, if any."""
    value = os.environ.get("NANODISCOVER_EVAL_PYTHON")
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def validate_evaluator_python(task, config: RunConfig) -> str | None:
    """Validate the external evaluator interpreter contract for the active task."""

    if stage_stops_before(config, "evaluate"):
        return None
    evaluator_python = resolve_evaluator_python()
    requires_external = bool(getattr(task, "requires_external_evaluator_python", False))
    if requires_external and evaluator_python is None:
        raise RuntimeError(
            f"Task {getattr(task, 'name', config.task_name)!r} requires "
            "NANODISCOVER_EVAL_PYTHON to point at the dedicated evaluator environment."
        )
    if evaluator_python is None:
        return None
    resolved = Path(evaluator_python).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"NANODISCOVER_EVAL_PYTHON does not exist: {resolved}")
    if resolved.is_dir():
        raise IsADirectoryError(f"NANODISCOVER_EVAL_PYTHON must point to a python executable, not a directory: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise PermissionError(f"NANODISCOVER_EVAL_PYTHON is not executable: {resolved}")
    return str(resolved.absolute())


def read_text_tail(path: Path, *, max_lines: int = 40) -> str:
    """Return a small trailing window of a text file for error reporting."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    return "\n".join(lines[-max(1, int(max_lines)) :])


def run_external_evaluation(
    *,
    evaluator_python: str,
    task_name: str,
    run_dir: Path,
    epoch: int,
    total_rollouts: int,
    workers: int,
) -> list[EvaluatedRollout]:
    """Run the shared evaluator CLI under the external evaluator interpreter."""
    repo_root = Path(__file__).resolve().parent
    subdir = utils.epoch_subdir(run_dir, epoch)
    output_path = subdir.root / "_external_evaluation.json"
    log_path = subdir.root / "evaluator.log"
    command = [
        evaluator_python,
        "-m",
        "core.evaluator",
        "evaluate-shard",
        "--task",
        task_name,
        "--run-dir",
        str(run_dir),
        "--epoch",
        str(epoch),
        "--start",
        "0",
        "--stop",
        str(total_rollouts),
        "--workers",
        str(max(1, int(workers))),
        "--output",
        str(output_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "external_eval_subprocess_start epoch=%d workers=%d log=%s output=%s",
        epoch,
        max(1, int(workers)),
        str(log_path),
        str(output_path),
    )
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        details = read_text_tail(log_path) or f"see {log_path}"
        raise RuntimeError(
            f"external evaluator exited with code {completed.returncode}: {details}"
        )
    logger.info(
        "external_eval_subprocess_complete epoch=%d log=%s output=%s",
        epoch,
        str(log_path),
        str(output_path),
    )
    evaluated = load_evaluated_rollouts_from_path(output_path, expected_total=total_rollouts)
    output_path.unlink(missing_ok=True)
    return evaluated


def resolve_task_instance(config: RunConfig, task=None):
    """Return the task instance for this run."""

    if task is not None:
        return task
    key = config.task_name.strip().lower()
    module_name = f"tasks.{key}.env"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        if missing_name == module_name or missing_name.startswith(f"tasks.{key}"):
            raise ValueError(f"Unsupported task: {config.task_name}") from exc
        raise
    build_fn = getattr(module, "build_task", None)
    if not callable(build_fn):
        raise ValueError(f"Task module {module_name!r} must define a build_task() factory")
    return build_fn()


def build_runtime_components(
    config: RunConfig,
    task,
    *,
    archive=None,
    sampler=None,
    generator=None,
    evaluator=None,
    trainer=None,
) -> tuple[Archive, Sampler, Generator, Evaluator, Trainer]:
    """Instantiate any runtime components the caller did not provide."""

    if archive is None:
        archive = Archive(
            ArchiveConfig(
                max_archive_size=config.max_archive_size,
                topk_children=config.topk_children,
            ),
            initial_state_factory=lambda: task.make_initial_state(),
            refresh_initial_state_fn=getattr(task, "refresh_initial_state", None),
            dedupe_key_fn=task.dedupe_key,
            is_state_valid_fn=task.is_state_valid,
        )
    if sampler is None:
        sampler = Sampler(
            SamplerConfig(
                puct_c=config.puct_c,
                batch_size=config.seeds_per_epoch,
            )
        )
    if generator is None:
        generator = Generator(
            GeneratorConfig(
                model_name_or_path=config.model_name_or_path,
                tokenizer_name_or_path=config.tokenizer_name_or_path,
                renderer_name=config.renderer_name,
                renderer_system_prompt=config.renderer_system_prompt,
                renderer_stop_sequence=config.renderer_stop_sequence,
                temperature=config.temperature,
                phase1_max_tokens=config.phase1_max_tokens,
                context_window=config.context_window,
                context_buffer=config.context_buffer,
                gpu_memory_utilization=config.generator_gpu_memory_utilization,
                max_num_batched_tokens=config.generator_max_num_batched_tokens,
                max_num_seqs=config.generator_max_num_seqs,
                request_parallelism=config.generator_request_parallelism,
                request_timeout_s=config.generator_request_timeout_s,
                backend_name=config.generator_backend_name,
                final_answer_marker=config.final_answer_marker,
                forced_final_suffix=config.forced_final_suffix,
                phase1_end_marker=config.phase1_end_marker,
                forced_final_suffix_after_phase1_end_marker=config.forced_final_suffix_after_phase1_end_marker,
                data_parallel_size=config.generator_data_parallel_size,
                tensor_parallel_size=config.generator_tensor_parallel_size,
                batch_size=config.generator_batch_size,
                lora_rank=config.lora_rank,
                ray_temp_dir=config.ray_temp_dir,
                run_dir=config.run_dir,
            )
        )
    if evaluator is None:
        evaluator = Evaluator(EvaluatorConfig(max_workers=config.evaluator_num_workers))
    if trainer is None:
        trainer = Trainer(
            TrainerConfig(
                backend_name=config.train_backend,
                model_name_or_path=config.model_name_or_path,
                tokenizer_name_or_path=config.tokenizer_name_or_path,
                run_dir=config.run_dir,
                learning_rate=config.learning_rate,
                adam_beta1=config.adam_beta1,
                adam_beta2=config.adam_beta2,
                adam_eps=config.adam_eps,
                weight_decay=config.weight_decay,
                kl_penalty_coef=config.kl_penalty_coef,
                remove_constant_reward_groups=config.remove_constant_reward_groups,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                lora_target_modules=config.lora_target_modules,
                num_substeps=config.num_substeps,
                trainer_num_workers=config.trainer_num_workers,
                trainer_max_tokens_per_rank=config.trainer_max_tokens_per_rank,
                reference_scoring_max_tokens_per_rank=config.reference_scoring_max_tokens_per_rank,
                distributed_strategy=config.distributed_strategy,
                sequence_parallel_size=config.sequence_parallel_size,
                use_remove_padding=config.use_remove_padding,
                gradient_checkpointing=config.gradient_checkpointing,
                logprob_compute_dtype=config.trainer_logprob_compute_dtype,
                reference_logprob_vocab_chunk_size=config.reference_logprob_vocab_chunk_size,
                reference_scoring_model_parallel_size=config.reference_scoring_model_parallel_size,
            )
        )
    return archive, sampler, generator, evaluator, trainer


def restore_run_state(
    config: RunConfig,
    *,
    run_dir: Path,
    archive: Archive,
    sampler: Sampler,
    maximize_raw_score: bool,
) -> tuple[int, str | None, str | None, float | None]:
    """Restore or initialize run-local state before the epoch loop."""

    current_epoch = utils.resume_epoch(run_dir, stage_stop=config.stage_stop) if config.resume_dir else 0
    current_adapter_path = utils.resume_adapter(run_dir) if config.resume_dir else None
    current_optimizer_state_dir = utils.resume_optimizer_state(run_dir) if config.resume_dir else None
    best_raw_score = utils.load_best_raw_score(run_dir, maximize_raw_score) if config.resume_dir else None
    if config.resume_dir and current_epoch > 0:
        previous_epoch_subdir = utils.epoch_subdir(run_dir, current_epoch - 1)
        if not previous_epoch_subdir.has_state_checkpoints():
            raise FileNotFoundError(f"Missing checkpoint files for completed epoch {current_epoch - 1}")
        archive.load(path=previous_epoch_subdir.archive)
        sampler.load(path=previous_epoch_subdir.sampler)
        return current_epoch, current_adapter_path, current_optimizer_state_dir, best_raw_score
    archive.initialize(num_seeds=config.seeds_per_epoch)
    sampler.current_epoch = 0
    return current_epoch, current_adapter_path, current_optimizer_state_dir, best_raw_score


def cleanup_old_optimizer_states(run_dir: Path, current_epoch: int, keep_window: int) -> None:
    """Delete optimizer state directories older than the keep window."""

    import shutil

    if keep_window <= 0:
        return
    cutoff = current_epoch - keep_window
    if cutoff < 0:
        return
    for old_epoch in range(0, cutoff + 1):
        old_dir = utils.epoch_subdir(run_dir, old_epoch).root / "optimizer_state"
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)
            logger.info("optimizer_state_cleanup epoch=%d removed=%s", old_epoch, str(old_dir))


def format_epoch_best_raw_score(evaluated: list[EvaluatedRollout], *, maximize_raw_score: bool) -> str:
    """Return the epoch-local best raw score string used in logging."""

    valid_raw_scores = [
        float(rollout.raw_score)
        for rollout in evaluated
        if float(rollout.correctness) > 0 and rollout.raw_score is not None
    ]
    if not valid_raw_scores:
        return "none"
    value = max(valid_raw_scores) if maximize_raw_score else min(valid_raw_scores)
    return f"{value:.6f}"


def update_best_raw_score(
    best_raw_score: float | None,
    evaluated: list[EvaluatedRollout],
    *,
    maximize_raw_score: bool,
) -> float | None:
    """Update the run-global best raw score with one epoch of rollouts."""

    for rollout in evaluated:
        if float(rollout.correctness) <= 0 or rollout.raw_score is None:
            continue
        if best_raw_score is None:
            best_raw_score = rollout.raw_score
            continue
        if maximize_raw_score and rollout.raw_score > best_raw_score:
            best_raw_score = rollout.raw_score
        elif not maximize_raw_score and rollout.raw_score < best_raw_score:
            best_raw_score = rollout.raw_score
    return best_raw_score


def group_rollouts_by_seed(
    evaluated: list[EvaluatedRollout],
    *,
    rollouts_per_seed: int,
) -> list[list[EvaluatedRollout]]:
    """Return evaluated rollouts grouped in rollout-order by seed."""

    return [
        evaluated[index : index + rollouts_per_seed]
        for index in range(0, len(evaluated), rollouts_per_seed)
    ]


def run(config: RunConfig, *, task=None, archive=None, sampler=None, generator=None, evaluator=None, trainer=None) -> dict[str, object]:
    """Run one NanoDiscover search session for the configured stage scope."""

    validate_run_config(config)
    task = resolve_task_instance(config, task)
    external_evaluator_python = validate_evaluator_python(task, config)
    run_dir = Path(config.run_dir).resolve()
    archive, sampler, generator, evaluator, trainer = build_runtime_components(
        config,
        task,
        archive=archive,
        sampler=sampler,
        generator=generator,
        evaluator=evaluator,
        trainer=trainer,
    )
    maximize_raw_score = bool(getattr(task, "maximize_raw_score", True))
    current_epoch, current_adapter_path, current_optimizer_state_dir, best_raw_score = restore_run_state(
        config,
        run_dir=run_dir,
        archive=archive,
        sampler=sampler,
        maximize_raw_score=maximize_raw_score,
    )
    trainer.set_resume_adapter(current_adapter_path)
    trainer.set_resume_optimizer(current_optimizer_state_dir)

    logger.info(
        "run_start task=%s epochs=%d seeds_per_epoch=%d rollouts_per_seed=%d resume=%s run_dir=%s",
        config.task_name,
        config.num_epochs,
        config.seeds_per_epoch,
        config.rollouts_per_seed,
        bool(config.resume_dir),
        config.run_dir,
    )
    reference_scoring_model_parallel_size = resolve_reference_scoring_model_parallel_size(config)
    logger.info(
        "runtime_topology generator_mode=%s generator_data_parallel_size=%d generator_tensor_parallel_size=%d evaluator_num_workers=%d train_backend=%s trainer_num_workers=%d sequence_parallel_size=%d reference_scoring_model_parallel_size=%d reference_scoring_data_parallel_size=%d",
        config.generator_backend_name,
        config.generator_data_parallel_size,
        config.generator_tensor_parallel_size,
        config.evaluator_num_workers,
        config.train_backend,
        config.trainer_num_workers,
        config.sequence_parallel_size,
        reference_scoring_model_parallel_size,
        int(config.trainer_num_workers) // max(1, reference_scoring_model_parallel_size),
    )
    logger.info(
        "ray_data_llm configured with generator_data_parallel_size=%d Ray LLM replicas and generator_tensor_parallel_size=%d in-engine tensor parallelism; prompts are pre-rendered and pre-tokenized, so Ray chat_template/tokenize/detokenize stages are disabled.",
        config.generator_data_parallel_size,
        config.generator_tensor_parallel_size,
    )

    logger.info(
        "stage_scope start=%s stop=%s max_epochs=%d",
        config.stage_start,
        config.stage_stop,
        config.stage_max_epochs,
    )

    processed_epochs = 0
    for epoch in range(current_epoch, config.num_epochs):
        if config.stage_max_epochs > 0 and processed_epochs >= config.stage_max_epochs:
            logger.info("stage_scope_limit_reached processed_epochs=%d", processed_epochs)
            break

        epoch_subdir = utils.epoch_subdir(run_dir, epoch)
        epoch_subdir.root.mkdir(parents=True, exist_ok=True)

        if epoch_subdir.has_state_checkpoints():
            archive.load(path=epoch_subdir.archive)
            sampler.load(path=epoch_subdir.sampler)

        epoch_started_at = time.perf_counter()
        logger.info(
            "epoch_start epoch=%d/%d archive_size=%d T=%d current_adapter=%s",
            epoch + 1,
            config.num_epochs,
            len(archive.states),
            sampler.T,
            current_adapter_path or "none",
        )

        if stage_enabled(config, "sample"):
            if epoch_subdir.has_sample():
                archive_payload, seed_states, prompts = utils.load_sample(epoch_subdir)
                if not epoch_subdir.has_archive_checkpoint():
                    utils.restore_archive_snapshot(archive, archive_payload)
                logger.info("stage=sample epoch=%d restored=1 seeds=%d", epoch, len(seed_states))
            else:
                sample_started_at = time.perf_counter()
                picked = sampler.sample(archive)
                seed_states = [item.state for item in picked]
                prompts = [task.render_prompt(state) for state in seed_states]
                utils.save_sample(epoch_subdir, epoch=epoch, archive=archive, seed_states=seed_states, prompts=prompts)
                logger.info(
                    "stage=sample epoch=%d seeds=%d rollouts=%d elapsed_s=%.3f",
                    epoch,
                    len(seed_states),
                    len(prompts) * config.rollouts_per_seed,
                    time.perf_counter() - sample_started_at,
                )
        else:
            if not epoch_subdir.has_sample():
                raise FileNotFoundError(f"Missing sample checkpoint for epoch {epoch}: {epoch_subdir.sample}")
            archive_payload, seed_states, prompts = utils.load_sample(epoch_subdir)
            if not epoch_subdir.has_archive_checkpoint():
                utils.restore_archive_snapshot(archive, archive_payload)
            logger.info("stage=sample epoch=%d skipped=1 restored=1 seeds=%d", epoch, len(seed_states))

        repeated_seed_states = [state for state in seed_states for _ in range(config.rollouts_per_seed)]
        repeated_prompts = [prompt for prompt in prompts for _ in range(config.rollouts_per_seed)]

        if stage_enabled(config, "generate"):
            if epoch_subdir.has_generation():
                generations = utils.load_generations(epoch_subdir)
                logger.info("stage=generate epoch=%d restored=1 generations=%d", epoch, len(generations))
            else:
                generator_log_path = epoch_subdir.root / "generator.log"
                generation_started_at = time.perf_counter()
                logger.info("stage=generate epoch=%d stream_log=%s", epoch, generator_log_path)
                with utils.capture_stage_output(generator_log_path):
                    try:
                        if current_adapter_path:
                            weights_path, tensor_count, size_bytes = validate_adapter_checkpoint(current_adapter_path)
                            logger.info(
                                "stage=generate epoch=%d adapter_verified=1 weights=%s tensors=%d size_bytes=%d",
                                epoch,
                                str(weights_path),
                                tensor_count,
                                size_bytes,
                            )
                        generator.reload_adapter(current_adapter_path)
                        generations = generator.generate(repeated_prompts)
                    finally:
                        generator.teardown()
                utils.save_generations(epoch_subdir, epoch=epoch, generations=generations)
                logger.info(
                    "stage=generate epoch=%d prompts=%d generations=%d elapsed_s=%.3f",
                    epoch,
                    len(repeated_prompts),
                    len(generations),
                    time.perf_counter() - generation_started_at,
                )
        else:
            if not epoch_subdir.has_generation():
                raise FileNotFoundError(f"Missing generation checkpoint for epoch {epoch}: {epoch_subdir.generation}")
            generations = utils.load_generations(epoch_subdir)
            logger.info("stage=generate epoch=%d skipped=1 restored=1 generations=%d", epoch, len(generations))

        if stage_stops_before(config, "evaluate"):
            logger.info("epoch_done epoch=%d stage_scope_stop=%s", epoch, config.stage_stop)
            processed_epochs += 1
            continue

        if stage_enabled(config, "evaluate"):
            if epoch_subdir.has_evaluation():
                evaluated = utils.load_evaluations(epoch_subdir)
                logger.info("stage=evaluate epoch=%d restored=1 evaluated=%d", epoch, len(evaluated))
            else:
                evaluation_started_at = time.perf_counter()

                epoch_subdir.evaluation_logs.mkdir(parents=True, exist_ok=True)
                eval_stream_files = 0
                eval_stream_bytes = 0
                if external_evaluator_python is not None:
                    logger.info(
                        "stage=evaluate epoch=%d mode=external_evaluator_cli evaluator_python=%s",
                        epoch,
                        external_evaluator_python,
                    )
                    evaluated = run_external_evaluation(
                        evaluator_python=external_evaluator_python,
                        task_name=config.task_name,
                        run_dir=run_dir,
                        epoch=epoch,
                        total_rollouts=len(generations),
                        workers=config.evaluator_num_workers,
                    )
                    eval_stream_files, eval_stream_bytes = utils.persist_evaluation_streams(
                        epoch_subdir,
                        epoch=epoch,
                        evaluated=evaluated,
                    )
                else:
                    def persist_rollout_stream(index: int, rollout) -> None:
                        nonlocal eval_stream_files, eval_stream_bytes
                        stream_bytes = utils.persist_evaluation_stream(epoch_subdir, index=index, rollout=rollout)
                        if stream_bytes > 0:
                            eval_stream_files += 1
                            eval_stream_bytes += stream_bytes

                    try:
                        evaluated = evaluator.evaluate_batch(
                            task=task,
                            seed_states=repeated_seed_states,
                            prompts=repeated_prompts,
                            generations=generations,
                            epoch=epoch,
                            base_seed=epoch * config.seeds_per_epoch * config.rollouts_per_seed,
                            on_rollout_completed=persist_rollout_stream,
                        )
                    finally:
                        evaluator.teardown()
                utils.save_evaluations(epoch_subdir, epoch=epoch, evaluated=evaluated)
                logger.info(
                    "stage=evaluate epoch=%d evaluated=%d correct=%d reward_sum=%.6f best_raw_score_for_this_epoch=%s eval_stream_files=%d eval_stream_mb=%.3f elapsed_s=%.3f",
                    epoch,
                    len(evaluated),
                    sum(1 for rollout in evaluated if rollout.correctness > 0),
                    sum(float(rollout.reward) for rollout in evaluated),
                    format_epoch_best_raw_score(evaluated, maximize_raw_score=maximize_raw_score),
                    eval_stream_files,
                    eval_stream_bytes / (1024.0 * 1024.0),
                    time.perf_counter() - evaluation_started_at,
                )
        else:
            if not epoch_subdir.has_evaluation():
                raise FileNotFoundError(f"Missing evaluation checkpoint for epoch {epoch}: {epoch_subdir.evaluation}")
            evaluated = utils.load_evaluations(epoch_subdir)
            logger.info(
                "stage=evaluate epoch=%d skipped=1 restored=1 evaluated=%d best_raw_score_for_this_epoch=%s",
                epoch,
                len(evaluated),
                format_epoch_best_raw_score(evaluated, maximize_raw_score=maximize_raw_score),
            )

        if stage_stops_before(config, "archive_update"):
            logger.info("epoch_done epoch=%d stage_scope_stop=%s", epoch, config.stage_stop)
            processed_epochs += 1
            continue

        next_states = [rollout.next_state if rollout.correctness > 0 else None for rollout in evaluated]
        if stage_enabled(config, "archive_update"):
            if not epoch_subdir.has_state_checkpoints():
                update_started_at = time.perf_counter()
                sampler.update(
                    repeated_seed_states,
                    next_states,
                    epoch=epoch + 1,
                    checkpoint=False,
                )
                archive.update(
                    repeated_seed_states,
                    next_states,
                    epoch=epoch + 1,
                    checkpoint=False,
                )
                sampler.checkpoint(path=epoch_subdir.sampler, epoch=epoch + 1)
                archive.checkpoint(path=epoch_subdir.archive, epoch=epoch + 1)
                logger.info(
                    "stage=archive_update epoch=%d inserted=%d archive_size=%d T=%d elapsed_s=%.3f",
                    epoch,
                    sum(1 for state in next_states if state is not None),
                    len(archive.states),
                    sampler.T,
                    time.perf_counter() - update_started_at,
                )
            else:
                logger.info("stage=archive_update epoch=%d restored=1 archive_size=%d T=%d", epoch, len(archive.states), sampler.T)
        else:
            if not epoch_subdir.has_state_checkpoints():
                raise FileNotFoundError(
                    f"Missing archive/sampler checkpoints for epoch {epoch}: {epoch_subdir.archive}, {epoch_subdir.sampler}"
                )
            logger.info("stage=archive_update epoch=%d skipped=1 restored=1 archive_size=%d T=%d", epoch, len(archive.states), sampler.T)

        if stage_stops_before(config, "train"):
            logger.info("epoch_done epoch=%d stage_scope_stop=%s", epoch, config.stage_stop)
            processed_epochs += 1
            continue

        best_raw_score = update_best_raw_score(
            best_raw_score,
            evaluated,
            maximize_raw_score=maximize_raw_score,
        )
        rollout_groups = group_rollouts_by_seed(
            evaluated,
            rollouts_per_seed=config.rollouts_per_seed,
        )

        if stage_enabled(config, "train"):
            if epoch_subdir.has_training_result():
                training_payload = utils.load_training_result(epoch_subdir)
                current_adapter_path = str(training_payload.get("adapter_path") or current_adapter_path or "") or current_adapter_path
                current_optimizer_state_dir = str(training_payload.get("optimizer_state_dir") or current_optimizer_state_dir or "") or current_optimizer_state_dir
                train_metrics = dict(training_payload.get("metrics") or {})
                logger.info("stage=train epoch=%d restored=1 adapter=%s", epoch, current_adapter_path or "none")
            else:
                trainer_log_path = epoch_subdir.root / "trainer.log"
                training_started_at = time.perf_counter()
                logger.info("stage=train epoch=%d stream_log=%s", epoch, trainer_log_path)
                with utils.capture_stage_output(trainer_log_path):
                    try:
                        trainer.set_resume_adapter(current_adapter_path)
                        trainer.set_resume_optimizer(current_optimizer_state_dir)
                        training_result = trainer.train(rollout_groups, epoch=epoch, output_dir=epoch_subdir.root)
                    finally:
                        trainer.teardown()
                utils.save_training_result(epoch_subdir, epoch=epoch, result=training_result)
                current_adapter_path = training_result.adapter_path or current_adapter_path
                current_optimizer_state_dir = training_result.optimizer_state_dir or current_optimizer_state_dir
                train_metrics = getattr(training_result, "metrics", {}) or {}
                logger.info(
                    "stage=train epoch=%d groups=%d loss_samples=%s dropped_constant_groups=%s adapter=%s elapsed_s=%.3f",
                    epoch,
                    len(rollout_groups),
                    train_metrics.get("num_loss_samples", "n/a"),
                    train_metrics.get("dropped_constant_reward_groups", "n/a"),
                    current_adapter_path or "none",
                    time.perf_counter() - training_started_at,
                )
            if config.optimizer_state_keep_window > 0:
                cleanup_old_optimizer_states(run_dir, epoch, config.optimizer_state_keep_window)
        else:
            train_metrics = {}
            logger.info("stage=train epoch=%d skipped=1", epoch)

        logger.info(
            "epoch_done epoch=%d archive_size=%d best_raw_score_since_beginning=%s total_elapsed_s=%.3f",
            epoch,
            len(archive.states),
            "none" if best_raw_score is None else f"{best_raw_score:.6f}",
            time.perf_counter() - epoch_started_at,
        )
        processed_epochs += 1

    return {
        "best_raw_score": best_raw_score,
        "archive_size": len(archive.states),
        "run_dir": config.run_dir,
    }


def main() -> None:
    """Load config, initialize logging, and run NanoDiscover."""

    configure_runtime_compat_env()
    config = load_run_config()

    run_dir = Path(config.run_dir)
    utils.configure_run_logging(run_dir)
    logger.info("logging_to %s", run_dir / "log.txt")
    logger.info(
        "run_session_start pid=%d resume=%s run_dir=%s",
        os.getpid(),
        bool(config.resume_dir),
        config.run_dir,
    )
    logger.info("run_config start")
    for key, value in sorted(asdict(config).items()):
        if value is None:
            continue
        logger.info("  %s=%r", key, value)
    logger.info("run_config end")

    run(config)


if __name__ == "__main__":
    main()
