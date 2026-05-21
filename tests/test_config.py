from pathlib import Path

import pytest

from config import latest_epoch, load_run_config, resolve_run_dir


def _set_renderer_env(
    monkeypatch,
    *,
    renderer_name: str = "qwen_chat",
    system_prompt: str = "",
    stop_sequence: str = "<|im_end|>",
    final_answer_marker: str = "",
    forced_final_suffix: str = "",
    phase1_end_marker: str = "",
    forced_final_suffix_after_phase1_end_marker: str = "",
) -> None:
    monkeypatch.setenv("NANODISCOVER_RENDERER_NAME", renderer_name)
    monkeypatch.setenv("NANODISCOVER_RENDERER_SYSTEM_PROMPT", system_prompt)
    monkeypatch.setenv("NANODISCOVER_RENDERER_STOP_SEQUENCE", stop_sequence)
    monkeypatch.setenv("NANODISCOVER_FINAL_ANSWER_MARKER", final_answer_marker)
    monkeypatch.setenv("NANODISCOVER_FORCED_FINAL_SUFFIX", forced_final_suffix)
    monkeypatch.setenv("NANODISCOVER_PHASE1_END_MARKER", phase1_end_marker)
    monkeypatch.setenv(
        "NANODISCOVER_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER",
        forced_final_suffix_after_phase1_end_marker,
    )


def test_resolve_run_dir_prefers_resume_dir(tmp_path):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()

    resolved = resolve_run_dir("ac1", None, str(resume_dir))

    assert resolved == str(resume_dir.resolve())


def test_resolve_run_dir_builds_timestamped_directory_under_log_root(tmp_path):
    log_root = tmp_path / "logs"

    resolved = resolve_run_dir("ac1", str(log_root), None)

    assert Path(resolved).parent == log_root.resolve()
    assert Path(resolved).name.startswith("ac1-")


def test_latest_epoch_reads_highest_epoch_directory(tmp_path):
    (tmp_path / "epoch000").mkdir()
    (tmp_path / "epoch003").mkdir()
    (tmp_path / "epoch002").mkdir()

    assert latest_epoch(tmp_path) == 3


def test_load_run_config_builds_explicit_config(monkeypatch, tmp_path):
    log_root = tmp_path / "logs"
    monkeypatch.setenv("NANODISCOVER_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "5")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "8")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "64")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "4")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "1000")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "2")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "Qwen/Qwen3-8B")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "Qwen/Qwen3-8B")
    _set_renderer_env(monkeypatch)
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "26000")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "50")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "4e-5")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.1")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK", "65536")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "bf16")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_ULYSSES_SEQUENCE_PARALLEL_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    config = load_run_config()

    assert config.task_name == "ac1"
    assert config.num_epochs == 5
    assert config.seeds_per_epoch == 8
    assert config.train_backend == "deepspeed"
    assert config.generator_data_parallel_size == 2
    assert config.generator_tensor_parallel_size == 1
    assert config.generator_gpu_memory_utilization is None
    assert config.generator_max_num_batched_tokens is None
    assert config.generator_max_num_seqs is None
    assert config.generator_request_parallelism is None
    assert config.generator_request_timeout_s is None
    assert config.generator_backend_name == "ray_data_llm"
    assert config.renderer_name == "qwen_chat"
    assert config.renderer_stop_sequence == "<|im_end|>"
    assert config.renderer_system_prompt == ""
    assert config.trainer_max_tokens_per_rank == 65536
    assert config.trainer_logprob_compute_dtype == "bf16"
    assert config.reference_logprob_vocab_chunk_size == 4096
    assert config.reference_scoring_model_parallel_size == 1
    assert config.distributed_strategy == "ddp"
    assert config.sequence_parallel_size == 4
    assert config.use_remove_padding is True
    assert Path(config.run_dir).parent == log_root.resolve()


def test_load_run_config_resume_does_not_require_log_root(monkeypatch, tmp_path):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_RESUME_DIR", str(resume_dir))
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "5")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "8")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "64")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "4")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "8")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "1000")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "2")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "Qwen/Qwen3-8B")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "Qwen/Qwen3-8B")
    _set_renderer_env(monkeypatch)
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "26000")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "50")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "4e-5")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.1")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK", "65536")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    config = load_run_config()

    assert config.run_dir == str(resume_dir.resolve())
    assert config.generator_tensor_parallel_size == 1
    assert config.generator_max_num_seqs is None
    assert config.generator_request_parallelism is None
    assert config.generator_batch_size == 8
    assert config.generator_backend_name == "ray_data_llm"
    assert config.trainer_logprob_compute_dtype == "float32"
    assert config.reference_logprob_vocab_chunk_size == 4096


def test_load_run_config_ray_backend_does_not_require_nonpublic_generator_service_knobs(monkeypatch, tmp_path):
    log_root = tmp_path / "logs"
    monkeypatch.setenv("NANODISCOVER_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "1")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "1")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "1")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "10")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "2")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "Qwen/Qwen3-8B")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "Qwen/Qwen3-8B")
    _set_renderer_env(monkeypatch)
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "26000")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "50")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "4e-5")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.1")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK", "65536")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    config = load_run_config()

    assert config.generator_backend_name == "ray_data_llm"
    assert config.generator_gpu_memory_utilization is None
    assert config.generator_max_num_batched_tokens is None
    assert config.generator_max_num_seqs is None
    assert config.generator_request_parallelism is None
    assert config.generator_request_timeout_s is None


def test_load_run_config_rejects_non_divisible_nominal_training_batch(monkeypatch, tmp_path):
    log_root = tmp_path / "logs"
    monkeypatch.setenv("NANODISCOVER_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "1")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "1")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "64")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "10")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "2")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "Qwen/Qwen3-8B")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "Qwen/Qwen3-8B")
    _set_renderer_env(monkeypatch)
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "26000")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "50")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "4e-5")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.1")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "6")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    with pytest.raises(RuntimeError, match="Nominal training batch size"):
        load_run_config()


def test_load_run_config_rejects_reference_scoring_model_parallel_size_that_does_not_divide_workers(monkeypatch, tmp_path):
    log_root = tmp_path / "logs"
    monkeypatch.setenv("NANODISCOVER_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "1")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "2")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "64")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "10")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "2")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "Qwen/Qwen3-8B")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "Qwen/Qwen3-8B")
    _set_renderer_env(monkeypatch)
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "26000")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "50")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "4e-5")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.1")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "8")
    monkeypatch.setenv("NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK", "8192")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "3")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "8")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    with pytest.raises(RuntimeError, match="reference_scoring_model_parallel_size"):
        load_run_config()


def test_load_run_config_builds_explicit_gpt_oss_renderer_contract(monkeypatch, tmp_path):
    log_root = tmp_path / "logs"
    monkeypatch.setenv("NANODISCOVER_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "1")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "1")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "1")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "8")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "8")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "10")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "2")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "openai/gpt-oss-120b")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "openai/gpt-oss-120b")
    _set_renderer_env(
        monkeypatch,
        renderer_name="gpt_oss_harmony",
        system_prompt="<|start|>system<|message|>sys<|end|>",
        stop_sequence="<|return|>",
        final_answer_marker="<|channel|>final<|message|>",
        forced_final_suffix="\\n\\nforced<|end|><|start|>assistant<|channel|>final<|message|>",
        phase1_end_marker="<|end|>",
        forced_final_suffix_after_phase1_end_marker="\\n\\nforced<|start|>assistant<|channel|>final<|message|>",
    )
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "26000")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "50")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "4e-5")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.1")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK", "8192")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    config = load_run_config()

    assert config.renderer_name == "gpt_oss_harmony"
    assert config.renderer_system_prompt == "<|start|>system<|message|>sys<|end|>"
    assert config.renderer_stop_sequence == "<|return|>"
    assert config.final_answer_marker == "<|channel|>final<|message|>"
    assert config.forced_final_suffix.startswith("\n\nforced")


def test_load_run_config_rejects_gpt_oss_with_blank_required_knob(monkeypatch, tmp_path):
    log_root = tmp_path / "logs"
    monkeypatch.setenv("NANODISCOVER_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "1")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "1")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "1")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "8")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "8")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "10")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "2")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "openai/gpt-oss-120b")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "openai/gpt-oss-120b")
    _set_renderer_env(
        monkeypatch,
        renderer_name="gpt_oss_harmony",
        system_prompt="<|start|>system<|message|>sys<|end|>",
        stop_sequence="<|return|>",
        final_answer_marker="<|channel|>final<|message|>",
        forced_final_suffix="",
        phase1_end_marker="<|end|>",
        forced_final_suffix_after_phase1_end_marker="\\n\\nforced<|start|>assistant<|channel|>final<|message|>",
    )
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "26000")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "50")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "4e-5")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.1")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_MAX_TOKENS_PER_RANK", "8192")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    with pytest.raises(RuntimeError, match="GPT-OSS Harmony configs require explicit non-empty launcher values"):
        load_run_config()


def test_load_run_config_allows_non_divisible_training_shape_when_train_stage_disabled(monkeypatch, tmp_path):
    log_root = tmp_path / "logs"
    monkeypatch.setenv("NANODISCOVER_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "1")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "1")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "64")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "10")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "2")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "Qwen/Qwen3-8B")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "Qwen/Qwen3-8B")
    _set_renderer_env(monkeypatch)
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "26000")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "50")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "4e-5")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.1")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "32")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "6")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")
    monkeypatch.setenv("NANODISCOVER_STAGE_STOP", "generate")

    config = load_run_config()

    assert config.stage_stop == "generate"