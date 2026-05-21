from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.renderer import KNOWN_RENDERER_NAMES


STAGE_SAMPLE = "sample"
STAGE_GENERATE = "generate"
STAGE_EVALUATE = "evaluate"
STAGE_ARCHIVE_UPDATE = "archive_update"
STAGE_TRAIN = "train"
STAGE_ORDER = (
    STAGE_SAMPLE,
    STAGE_GENERATE,
    STAGE_EVALUATE,
    STAGE_ARCHIVE_UPDATE,
    STAGE_TRAIN,
)


@dataclass
class RunConfig:
    """Configuration for a single NanoDiscover run."""

    task_name: str
    num_epochs: int
    seeds_per_epoch: int
    rollouts_per_seed: int
    evaluator_num_workers: int
    run_dir: str
    resume_dir: str | None
    max_archive_size: int
    topk_children: int
    puct_c: float
    model_name_or_path: str
    tokenizer_name_or_path: str | None
    renderer_name: str
    renderer_system_prompt: str
    renderer_stop_sequence: str
    temperature: float
    phase1_max_tokens: int
    context_window: int
    context_buffer: int
    final_answer_marker: str | None
    forced_final_suffix: str | None
    phase1_end_marker: str | None
    forced_final_suffix_after_phase1_end_marker: str | None
    train_backend: str
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    weight_decay: float
    kl_penalty_coef: float
    remove_constant_reward_groups: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    num_substeps: int
    trainer_num_workers: int
    trainer_max_tokens_per_rank: int | None
    distributed_strategy: str
    sequence_parallel_size: int
    use_remove_padding: bool
    generator_data_parallel_size: int
    generator_tensor_parallel_size: int
    generator_gpu_memory_utilization: float | None
    generator_max_num_batched_tokens: int | None
    generator_max_num_seqs: int | None
    generator_request_parallelism: int | None
    generator_request_timeout_s: float | None
    generator_backend_name: str
    trainer_logprob_compute_dtype: str
    reference_logprob_vocab_chunk_size: int
    reference_scoring_max_tokens_per_rank: int
    reference_scoring_model_parallel_size: int
    gradient_checkpointing: bool
    optimizer_state_keep_window: int
    generator_batch_size: int | None = None
    stage_start: str = STAGE_SAMPLE
    stage_stop: str = STAGE_TRAIN
    stage_max_epochs: int = 0
    ray_temp_dir: str | None = None


def lookup_env_value(name: str, *aliases: str) -> str | None:
    """Return the first defined environment value among the given names."""

    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return None


def env_str(name: str, *aliases: str) -> str:
    """Return a required environment value as a raw string."""

    value = lookup_env_value(name, *aliases)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def env_int(name: str, *aliases: str) -> int:
    """Return a required environment value parsed as an integer."""

    return int(env_str(name, *aliases))


def env_float(name: str, *aliases: str) -> float:
    """Return a required environment value parsed as a float."""

    return float(env_str(name, *aliases))


def env_int_with_default(name: str, default: int, *aliases: str) -> int:
    """Return an integer environment value, falling back to a default."""

    value = lookup_env_value(name, *aliases)
    if value is None:
        return int(default)
    value = value.strip()
    return int(value) if value else int(default)


def env_float_with_default(name: str, default: float, *aliases: str) -> float:
    """Return a float environment value, falling back to a default."""

    value = lookup_env_value(name, *aliases)
    if value is None:
        return float(default)
    value = value.strip()
    return float(value) if value else float(default)


def env_optional_int(name: str, *aliases: str) -> int | None:
    """Return an optional integer environment value."""

    value = lookup_env_value(name, *aliases)
    if value is None:
        return None
    value = value.strip()
    return int(value) if value else None


def env_bool(name: str, *aliases: str) -> bool:
    """Return a required boolean environment value."""

    raw = env_str(name, *aliases).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def env_optional_str(name: str, *aliases: str) -> str | None:
    """Return an optional environment value stripped to `None` when blank."""

    value = lookup_env_value(name, *aliases)
    if value is None:
        return None
    value = value.strip()
    return value or None


def decode_optional_multiline_env(name: str, *aliases: str) -> str | None:
    """Decode an optional escaped multiline environment value."""

    value = env_optional_str(name, *aliases)
    if value is None:
        return None
    return bytes(value, "utf-8").decode("unicode_escape")


def decode_required_multiline_env(name: str, *aliases: str) -> str:
    """Decode a required escaped multiline environment value."""

    return bytes(env_str(name, *aliases), "utf-8").decode("unicode_escape")


def env_stage(name: str, default: str) -> str:
    """Return a validated stage name from the environment."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized not in STAGE_ORDER:
        expected = ", ".join(STAGE_ORDER)
        raise RuntimeError(f"Invalid {name}={value!r}; expected one of: {expected}")
    return normalized


def resolve_run_dir(task_name: str, log_root: str | None, resume_dir: str | None) -> str:
    """Resolve the active run directory for a fresh or resumed run."""

    if resume_dir:
        path = Path(resume_dir).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Resume directory does not exist: {path}")
        return str(path)
    if log_root is None:
        raise RuntimeError("Missing required environment variable: NANODISCOVER_LOG_ROOT")
    root = Path(log_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str((root / f"{task_name}-{timestamp}").resolve())


def epoch_subdir_name(epoch: int) -> str:
    """Return the canonical on-disk subdirectory name for an epoch."""

    return f"epoch{int(epoch):03d}"


def epoch_subdir_path(run_dir: str | Path, epoch: int) -> Path:
    """Return the absolute path for an epoch subdirectory."""

    return Path(run_dir).resolve() / epoch_subdir_name(epoch)


def parse_epoch_subdir_name(name: str) -> int | None:
    """Parse an `epochNNN` directory name into its integer epoch."""

    match = re.fullmatch(r"epoch(\d{3})", name)
    if match is None:
        return None
    return int(match.group(1))


def latest_epoch(run_dir: str | Path) -> int | None:
    """Return the highest completed epoch directory present under a run."""

    root = Path(run_dir)
    epochs = sorted(
        epoch
        for child in root.iterdir()
        if child.is_dir()
        for epoch in [parse_epoch_subdir_name(child.name)]
        if epoch is not None
    ) if root.exists() else []
    return epochs[-1] if epochs else None


def stage_scope_includes_training(config: RunConfig) -> bool:
    """Return whether the configured stage window includes training."""

    return (
        STAGE_ORDER.index(config.stage_start)
        <= STAGE_ORDER.index(STAGE_TRAIN)
        <= STAGE_ORDER.index(config.stage_stop)
    )


def resolve_reference_scoring_model_parallel_size(config: RunConfig) -> int:
    """Return the effective model-parallel width for reference scoring."""

    return max(
        1,
        int(
            config.reference_scoring_model_parallel_size
            if config.reference_scoring_model_parallel_size is not None
            else min(config.trainer_num_workers, config.sequence_parallel_size)
        ),
    )


def validate_run_config(config: RunConfig) -> None:
    """Validate the user-facing runtime contract for a run configuration."""

    generator_backend_name = str(config.generator_backend_name or "").strip().lower()
    if generator_backend_name != "ray_data_llm":
        raise RuntimeError(
            "Invalid NANODISCOVER_GENERATOR_BACKEND="
            f"{config.generator_backend_name!r}; expected: ray_data_llm"
        )
    train_backend = str(config.train_backend or "").strip().lower()
    if train_backend not in {"deepspeed", "dry-run"}:
        raise RuntimeError(
            f"Invalid NANODISCOVER_TRAIN_BACKEND={config.train_backend!r}; expected one of: deepspeed, dry-run"
        )
    renderer_name = str(config.renderer_name or "").strip().lower()
    valid_renderer_names = set(KNOWN_RENDERER_NAMES) | {"qwen_chat_instruct"}
    if renderer_name not in valid_renderer_names:
        expected = ", ".join(sorted(valid_renderer_names))
        raise RuntimeError(
            f"Invalid NANODISCOVER_RENDERER_NAME={config.renderer_name!r}; expected one of: {expected}"
        )
    if renderer_name in {"qwen_chat", "qwen_instruct_chat", "qwen_chat_instruct", "gpt_oss_harmony"} and not config.renderer_stop_sequence:
        raise RuntimeError(
            f"NANODISCOVER_RENDERER_STOP_SEQUENCE is required when NANODISCOVER_RENDERER_NAME={config.renderer_name!r}"
        )
    if renderer_name == "gpt_oss_harmony":
        required_fields = {
            "NANODISCOVER_RENDERER_SYSTEM_PROMPT": config.renderer_system_prompt,
            "NANODISCOVER_FINAL_ANSWER_MARKER": config.final_answer_marker,
            "NANODISCOVER_FORCED_FINAL_SUFFIX": config.forced_final_suffix,
            "NANODISCOVER_PHASE1_END_MARKER": config.phase1_end_marker,
            "NANODISCOVER_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER": config.forced_final_suffix_after_phase1_end_marker,
        }
        missing = [name for name, value in required_fields.items() if not value]
        if missing:
            raise RuntimeError(
                "GPT-OSS Harmony configs require explicit non-empty launcher values for: "
                + ", ".join(missing)
            )
    if not stage_scope_includes_training(config):
        return
    trainer_num_workers = int(config.trainer_num_workers)
    reference_scoring_model_parallel_size = resolve_reference_scoring_model_parallel_size(config)
    if trainer_num_workers % reference_scoring_model_parallel_size != 0:
        raise RuntimeError(
            "trainer_num_workers must be divisible by reference_scoring_model_parallel_size. "
            f"Current config: trainer_num_workers={config.trainer_num_workers}, "
            "reference_scoring_model_parallel_size="
            f"{reference_scoring_model_parallel_size}."
        )
    if trainer_num_workers <= 1:
        return
    sequence_parallel_size = max(1, int(config.sequence_parallel_size))
    if trainer_num_workers % sequence_parallel_size != 0:
        return
    dp_group_count = max(1, trainer_num_workers // sequence_parallel_size)
    nominal_training_batch_size = int(config.seeds_per_epoch) * int(config.rollouts_per_seed)
    if nominal_training_batch_size % dp_group_count != 0:
        raise RuntimeError(
            "Nominal training batch size "
            f"(seeds_per_epoch * rollouts_per_seed = {nominal_training_batch_size}) "
            f"must be divisible by the data-parallel group count {dp_group_count}. "
            f"Current config: seeds_per_epoch={config.seeds_per_epoch}, "
            f"rollouts_per_seed={config.rollouts_per_seed}, "
            f"trainer_num_workers={config.trainer_num_workers}, "
            f"sequence_parallel_size={config.sequence_parallel_size}."
        )


def load_run_config() -> RunConfig:
    """Load and validate the run configuration from environment variables."""

    task_name = env_str("NANODISCOVER_TASK_NAME")
    resume_dir = env_optional_str("NANODISCOVER_RESUME_DIR")
    generator_backend_name = env_str("NANODISCOVER_GENERATOR_BACKEND")
    train_backend = env_str("NANODISCOVER_TRAIN_BACKEND")
    run_dir = resolve_run_dir(
        task_name,
        None if resume_dir else env_str("NANODISCOVER_LOG_ROOT"),
        resume_dir,
    )
    (
        generator_gpu_memory_utilization,
        generator_max_num_batched_tokens,
        generator_max_num_seqs,
        generator_request_parallelism,
        generator_request_timeout_s,
    ) = (None, None, None, None, None)
    generator_batch_size = env_int("NANODISCOVER_GENERATOR_BATCH_SIZE")
    stage_start = env_stage("NANODISCOVER_STAGE_START", STAGE_SAMPLE)
    stage_stop = env_stage("NANODISCOVER_STAGE_STOP", STAGE_TRAIN)
    if STAGE_ORDER.index(stage_start) > STAGE_ORDER.index(stage_stop):
        raise RuntimeError(
            "NANODISCOVER_STAGE_START must be earlier than or equal to NANODISCOVER_STAGE_STOP"
        )
    distributed_strategy = env_optional_str("NANODISCOVER_DISTRIBUTED_STRATEGY") or "ddp"
    config = RunConfig(
        task_name=task_name,
        num_epochs=env_int("NANODISCOVER_NUM_EPOCHS"),
        seeds_per_epoch=env_int("NANODISCOVER_SEEDS_PER_EPOCH"),
        rollouts_per_seed=env_int("NANODISCOVER_ROLLOUTS_PER_SEED"),
        evaluator_num_workers=env_int("NANODISCOVER_EVALUATOR_NUM_WORKERS"),
        generator_data_parallel_size=env_int("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE"),
        generator_tensor_parallel_size=env_int("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE"),
        generator_gpu_memory_utilization=generator_gpu_memory_utilization,
        generator_max_num_batched_tokens=generator_max_num_batched_tokens,
        generator_max_num_seqs=generator_max_num_seqs,
        generator_request_parallelism=generator_request_parallelism,
        generator_request_timeout_s=generator_request_timeout_s,
        generator_backend_name=generator_backend_name,
        run_dir=run_dir,
        resume_dir=resume_dir,
        max_archive_size=env_int("NANODISCOVER_MAX_ARCHIVE_SIZE"),
        topk_children=env_int("NANODISCOVER_TOPK_CHILDREN"),
        puct_c=env_float("NANODISCOVER_PUCT_C"),
        model_name_or_path=env_str("NANODISCOVER_MODEL_NAME_OR_PATH"),
        tokenizer_name_or_path=env_optional_str("NANODISCOVER_TOKENIZER_NAME_OR_PATH"),
        renderer_name=env_str("NANODISCOVER_RENDERER_NAME"),
        renderer_system_prompt=decode_required_multiline_env("NANODISCOVER_RENDERER_SYSTEM_PROMPT"),
        renderer_stop_sequence=decode_required_multiline_env("NANODISCOVER_RENDERER_STOP_SEQUENCE"),
        temperature=env_float("NANODISCOVER_TEMPERATURE"),
        phase1_max_tokens=env_int("NANODISCOVER_PHASE1_MAX_TOKENS"),
        context_window=env_int("NANODISCOVER_CONTEXT_WINDOW"),
        context_buffer=env_int("NANODISCOVER_CONTEXT_BUFFER"),
        final_answer_marker=decode_required_multiline_env("NANODISCOVER_FINAL_ANSWER_MARKER"),
        forced_final_suffix=decode_required_multiline_env("NANODISCOVER_FORCED_FINAL_SUFFIX"),
        phase1_end_marker=decode_required_multiline_env("NANODISCOVER_PHASE1_END_MARKER"),
        forced_final_suffix_after_phase1_end_marker=decode_required_multiline_env(
            "NANODISCOVER_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER"
        ),
        train_backend=train_backend,
        learning_rate=env_float("NANODISCOVER_LEARNING_RATE"),
        adam_beta1=env_float("NANODISCOVER_ADAM_BETA1"),
        adam_beta2=env_float("NANODISCOVER_ADAM_BETA2"),
        adam_eps=env_float("NANODISCOVER_ADAM_EPS"),
        weight_decay=env_float("NANODISCOVER_WEIGHT_DECAY"),
        kl_penalty_coef=env_float("NANODISCOVER_KL_PENALTY_COEF"),
        remove_constant_reward_groups=env_bool("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS"),
        lora_rank=env_int("NANODISCOVER_LORA_RANK"),
        lora_alpha=env_int("NANODISCOVER_LORA_ALPHA"),
        lora_dropout=env_float("NANODISCOVER_LORA_DROPOUT"),
        lora_target_modules=[
            m.strip() for m in env_str("NANODISCOVER_LORA_TARGET_MODULES").split(",") if m.strip()
        ],
        num_substeps=env_int("NANODISCOVER_NUM_SUBSTEPS"),
        trainer_num_workers=env_int("NANODISCOVER_TRAINER_NUM_WORKERS"),
        trainer_max_tokens_per_rank=env_optional_int("NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK"),
        reference_scoring_max_tokens_per_rank=env_int("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK"),
        distributed_strategy=distributed_strategy,
        sequence_parallel_size=env_int("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "NANODISCOVER_ULYSSES_SEQUENCE_PARALLEL_SIZE"),
        use_remove_padding=env_bool("NANODISCOVER_USE_REMOVE_PADDING"),
        gradient_checkpointing=env_bool("NANODISCOVER_GRADIENT_CHECKPOINTING"),
        optimizer_state_keep_window=env_int("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW"),
        trainer_logprob_compute_dtype=env_str("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE"),
        reference_logprob_vocab_chunk_size=env_int("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE"),
        generator_batch_size=generator_batch_size,
        reference_scoring_model_parallel_size=env_int("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE"),
        stage_start=stage_start,
        stage_stop=stage_stop,
        stage_max_epochs=env_int_with_default("NANODISCOVER_STAGE_MAX_EPOCHS", 0),
        ray_temp_dir=env_optional_str("NANODISCOVER_RAY_TMPDIR"),
    )
    validate_run_config(config)
    return config
