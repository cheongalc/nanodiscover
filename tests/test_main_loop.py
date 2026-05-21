import json
from pathlib import Path

import pytest

from core.archive import ArchiveNode
from core.evaluator import EvaluatedRollout
from core.generator import GenerationOutput
import main as main_module
from main import RunConfig, run


class _FakeTask:
    maximize_raw_score = True

    def make_initial_state(self):
        # ignore the seed index; the framework used to supply one but no
        # real tasks in the current tree depend on it.
        return ArchiveNode(epoch=-1, value=1.0, task_payload={"construction": [0], "code": ""})

    def dedupe_key(self, state):
        return tuple(state.task_payload.get("construction") or [])

    def is_state_valid(self, state):
        return True

    def render_prompt(self, state):
        return f"prompt-{state.id}"

    def parse_code(self, response_text):
        return response_text

    def evaluate_code(self, *, parsed_code, state, epoch, seed):
        _ = (parsed_code, state, epoch, seed)
        return {"correctness": 1.0, "performance": 1.0, "raw_score": 1.0, "msg": "ok", "result_payload": {"result_construction": [1.0]}, "stdout": ""}

    def compute_reward(self, eval_output):
        _ = eval_output
        return 1.0

    def make_next_state(self, *, parent_state, parsed_code, eval_output, epoch):
        _ = (parent_state, parsed_code, eval_output)
        return ArchiveNode(epoch=epoch, value=1.0, task_payload={"construction": [1.0], "code": "x"})


class _FakeGenerator:
    def __init__(self):
        self.events = []

    def reload_adapter(self, adapter_path):
        self.events.append("reload")
        _ = adapter_path

    def generate(self, prompts):
        self.events.append("generate")
        return [GenerationOutput(prompt, "resp", [1], [2], [-1.0], [1.0], "stop") for prompt in prompts]

    def teardown(self):
        self.events.append("generate_teardown")


class _FakeEvaluator:
    def __init__(self):
        self.events = []

    def evaluate_batch(self, **kwargs):
        self.events.append("evaluate")
        task = kwargs["task"]
        rollouts = []
        for seed_state, prompt, generation in zip(kwargs["seed_states"], kwargs["prompts"], kwargs["generations"], strict=True):
            eval_output = task.evaluate_code(parsed_code=generation.response_text, state=seed_state, epoch=0, seed=0)
            next_state = task.make_next_state(parent_state=seed_state, parsed_code=generation.response_text, eval_output=eval_output, epoch=0)
            from core.evaluator import EvaluatedRollout

            rollouts.append(
                EvaluatedRollout(
                    seed_state=seed_state,
                    prompt_text=prompt,
                    response_text=generation.response_text,
                    prompt_token_ids=generation.prompt_token_ids,
                    completion_token_ids=generation.completion_token_ids,
                    completion_logprobs=generation.completion_logprobs,
                    completion_mask=generation.completion_mask,
                    finish_reason=generation.finish_reason,
                    parsed_code=generation.response_text,
                    reward=float(task.compute_reward(eval_output)),
                    correctness=float(eval_output.get("correctness", 0.0)),
                    performance=float(eval_output.get("performance", 0.0)),
                    raw_score=(float(eval_output["raw_score"]) if eval_output.get("raw_score") is not None else None),
                    archive_value=(float(next_state.value) if next_state is not None and next_state.value is not None else None),
                    next_state=next_state,
                    msg="ok",
                )
            )
        return rollouts

    def teardown(self):
        self.events.append("evaluate_teardown")


class _FakeTrainer:
    def __init__(self):
        self.events = []

    def train(self, rollout_groups, *, epoch, output_dir=None):
        self.events.append("train")
        _ = (rollout_groups, epoch, output_dir)

        class _Result:
            metrics = {}
            adapter_path = None
            optimizer_state_dir = None
            skipped = False

        return _Result()

    def teardown(self):
        self.events.append("train_teardown")

    def set_resume_adapter(self, adapter_path):
        _ = adapter_path

    def set_resume_optimizer(self, optimizer_state_dir):
        _ = optimizer_state_dir


def _test_run_config(tmp_path, run_name: str) -> RunConfig:
    """Build a small run config for main-loop regression tests."""
    return RunConfig(
        task_name="ac1",
        num_epochs=1,
        seeds_per_epoch=1,
        rollouts_per_seed=1,
        evaluator_num_workers=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / run_name),
        resume_dir=None,
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="deepspeed",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )


def test_main_loop_orders_stages(tmp_path):
    cfg = RunConfig(
        task_name="ac1",
            num_epochs=1,
            seeds_per_epoch=1,
        rollouts_per_seed=1,
        evaluator_num_workers=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / "run"),
        resume_dir=None,
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="deepspeed",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )
    fake_gen = _FakeGenerator()
    fake_eval = _FakeEvaluator()
    fake_train = _FakeTrainer()
    result = run(cfg, task=_FakeTask(), generator=fake_gen, evaluator=fake_eval, trainer=fake_train)

    # the run() return value no longer contains stage events, but the fake
    # components still record their own calls; ensure they were invoked in the
    # expected order.
    assert fake_gen.events == ["reload", "generate", "generate_teardown"]
    assert fake_eval.events == ["evaluate", "evaluate_teardown"]
    assert fake_train.events == ["train", "train_teardown"]
    assert (tmp_path / "run" / "epoch000" / "sample.json").exists()
    assert (tmp_path / "run" / "epoch000" / "generation.json").exists()
    assert (tmp_path / "run" / "epoch000" / "evaluation.json").exists()
    assert (tmp_path / "run" / "epoch000" / "archive.json").exists()
    assert (tmp_path / "run" / "epoch000" / "sampler.json").exists()
    assert (tmp_path / "run" / "epoch000" / "training.json").exists()
    # and the returned dict contains the usual summary keys
    assert set(result.keys()) == {"best_raw_score", "archive_size", "run_dir"}


def test_main_loop_can_use_external_evaluator_python(monkeypatch, tmp_path):
    cfg = _test_run_config(tmp_path, "external-eval")
    fake_gen = _FakeGenerator()
    fake_eval = _FakeEvaluator()
    fake_train = _FakeTrainer()
    evaluator_python = tmp_path / "math-eval-python"
    evaluator_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    evaluator_python.chmod(0o755)
    monkeypatch.setenv("NANODISCOVER_EVAL_PYTHON", str(evaluator_python))

    calls: list[list[str]] = []

    def fake_subprocess_run(command, **kwargs):
        calls.append(list(command))
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rollout = EvaluatedRollout(
            seed_state=ArchiveNode(epoch=-1, value=1.0, task_payload={"construction": [0], "code": ""}),
            prompt_text="prompt",
            response_text="resp",
            prompt_token_ids=[1],
            completion_token_ids=[2],
            completion_logprobs=[-1.0],
            completion_mask=[1.0],
            finish_reason="stop",
            parsed_code="resp",
            reward=1.0,
            correctness=1.0,
            performance=1.0,
            raw_score=1.0,
            archive_value=1.0,
            next_state=ArchiveNode(epoch=0, value=1.0, task_payload={"construction": [1.0], "code": "x"}),
            msg="ok",
            result_payload={"result_construction": [1.0]},
            stdout="",
        )
        payload = {"epoch": 0, "items": [{"index": 0, "rollout": rollout.to_dict()}]}
        output_path.write_text(json.dumps(payload), encoding="utf-8")

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        _ = kwargs
        return _Completed()

    monkeypatch.setattr(main_module.subprocess, "run", fake_subprocess_run)

    run(cfg, task=_FakeTask(), generator=fake_gen, evaluator=fake_eval, trainer=fake_train)

    assert fake_gen.events == ["reload", "generate", "generate_teardown"]
    assert fake_eval.events == []
    assert fake_train.events == ["train", "train_teardown"]
    assert calls
    assert calls[0][0] == str(evaluator_python.absolute())
    assert calls[0][1:4] == ["-m", "core.evaluator", "evaluate-shard"]
    assert "--task" in calls[0]
    assert (tmp_path / "external-eval" / "epoch000" / "evaluation.json").exists()


def test_main_loop_requires_external_evaluator_python_for_math_tasks(tmp_path):
    cfg = _test_run_config(tmp_path, "missing-external-eval")

    class _MathLikeTask(_FakeTask):
        name = "ac1"
        requires_external_evaluator_python = True

    with pytest.raises(RuntimeError, match="NANODISCOVER_EVAL_PYTHON"):
        run(
            cfg,
            task=_MathLikeTask(),
            generator=_FakeGenerator(),
            evaluator=_FakeEvaluator(),
            trainer=_FakeTrainer(),
        )


def test_main_loop_tears_down_generator_on_failure(tmp_path):
    class _FailingGenerator(_FakeGenerator):
        def generate(self, prompts):
            self.events.append("generate")
            _ = prompts
            raise RuntimeError("generator failed")

    cfg = _test_run_config(tmp_path, "run-generate-fail")
    failing_gen = _FailingGenerator()

    with pytest.raises(RuntimeError, match="generator failed"):
        run(cfg, task=_FakeTask(), generator=failing_gen, evaluator=_FakeEvaluator(), trainer=_FakeTrainer())

    assert failing_gen.events == ["reload", "generate", "generate_teardown"]
    epoch_dir = tmp_path / "run-generate-fail" / "epoch000"
    assert (epoch_dir / "sample.json").exists()
    assert not (epoch_dir / "generation.json").exists()


def test_main_loop_tears_down_trainer_on_failure(tmp_path):
    class _FailingTrainer(_FakeTrainer):
        def train(self, rollout_groups, *, epoch, output_dir=None):
            self.events.append("train")
            _ = (rollout_groups, epoch, output_dir)
            raise RuntimeError("trainer failed")

    cfg = _test_run_config(tmp_path, "run-train-fail")
    failing_trainer = _FailingTrainer()

    with pytest.raises(RuntimeError, match="trainer failed"):
        run(cfg, task=_FakeTask(), generator=_FakeGenerator(), evaluator=_FakeEvaluator(), trainer=failing_trainer)

    assert failing_trainer.events == ["train", "train_teardown"]
    epoch_dir = tmp_path / "run-train-fail" / "epoch000"
    assert (epoch_dir / "generation.json").exists()
    assert (epoch_dir / "evaluation.json").exists()
    assert not (epoch_dir / "training.json").exists()


def test_main_sets_up_logging_and_redirects(tmp_path, monkeypatch):
    log_root = tmp_path / "logs"
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "0")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "1")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "1")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "10")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "1")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "fake")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "")
    monkeypatch.setenv("NANODISCOVER_RENDERER_NAME", "plain_text")
    monkeypatch.setenv("NANODISCOVER_RENDERER_SYSTEM_PROMPT", "")
    monkeypatch.setenv("NANODISCOVER_RENDERER_STOP_SEQUENCE", "")
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "1")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "1")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "0")
    monkeypatch.setenv("NANODISCOVER_FINAL_ANSWER_MARKER", "")
    monkeypatch.setenv("NANODISCOVER_FORCED_FINAL_SUFFIX", "")
    monkeypatch.setenv("NANODISCOVER_PHASE1_END_MARKER", "")
    monkeypatch.setenv("NANODISCOVER_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER", "")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "1e-4")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.0")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "0")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    # stub out the heavy run() implementation
    import main as main_module
    monkeypatch.setattr(main_module, "run", lambda cfg: {})

    # call main; it should create the directory and write the log file
    main_module.main()

    run_dirs = list(log_root.glob("ac1-*"))
    assert len(run_dirs) == 1
    log_file = run_dirs[0] / "log.txt"
    assert log_file.exists(), "log.txt should be created in run_dir"
    content = log_file.read_text()
    assert "logging_to" in content
    assert "run_session_start" in content
    assert "run_config start" in content
    assert "task_name='ac1'" in content


def test_logfile_appends_on_resume(tmp_path, monkeypatch):
    base = tmp_path / "run"
    base.mkdir()
    orig = base / "log.txt"
    orig.write_text("start\n")

    # set env to resume
    monkeypatch.setenv("NANODISCOVER_TASK_NAME", "ac1")
    monkeypatch.setenv("NANODISCOVER_RESUME_DIR", str(base))
    monkeypatch.setenv("NANODISCOVER_NUM_EPOCHS", "0")
    monkeypatch.setenv("NANODISCOVER_SEEDS_PER_EPOCH", "1")
    monkeypatch.setenv("NANODISCOVER_ROLLOUTS_PER_SEED", "1")
    monkeypatch.setenv("NANODISCOVER_EVALUATOR_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_DATA_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_TENSOR_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BATCH_SIZE", "4")
    monkeypatch.setenv("NANODISCOVER_GENERATOR_BACKEND", "ray_data_llm")
    monkeypatch.setenv("NANODISCOVER_MAX_ARCHIVE_SIZE", "10")
    monkeypatch.setenv("NANODISCOVER_TOPK_CHILDREN", "1")
    monkeypatch.setenv("NANODISCOVER_PUCT_C", "1.0")
    monkeypatch.setenv("NANODISCOVER_MODEL_NAME_OR_PATH", "fake")
    monkeypatch.setenv("NANODISCOVER_TOKENIZER_NAME_OR_PATH", "")
    monkeypatch.setenv("NANODISCOVER_RENDERER_NAME", "plain_text")
    monkeypatch.setenv("NANODISCOVER_RENDERER_SYSTEM_PROMPT", "")
    monkeypatch.setenv("NANODISCOVER_RENDERER_STOP_SEQUENCE", "")
    monkeypatch.setenv("NANODISCOVER_TEMPERATURE", "1.0")
    monkeypatch.setenv("NANODISCOVER_PHASE1_MAX_TOKENS", "1")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_WINDOW", "1")
    monkeypatch.setenv("NANODISCOVER_CONTEXT_BUFFER", "0")
    monkeypatch.setenv("NANODISCOVER_FINAL_ANSWER_MARKER", "")
    monkeypatch.setenv("NANODISCOVER_FORCED_FINAL_SUFFIX", "")
    monkeypatch.setenv("NANODISCOVER_PHASE1_END_MARKER", "")
    monkeypatch.setenv("NANODISCOVER_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER", "")
    monkeypatch.setenv("NANODISCOVER_TRAIN_BACKEND", "deepspeed")
    monkeypatch.setenv("NANODISCOVER_LEARNING_RATE", "1e-4")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA1", "0.9")
    monkeypatch.setenv("NANODISCOVER_ADAM_BETA2", "0.95")
    monkeypatch.setenv("NANODISCOVER_ADAM_EPS", "1e-8")
    monkeypatch.setenv("NANODISCOVER_WEIGHT_DECAY", "0.0")
    monkeypatch.setenv("NANODISCOVER_KL_PENALTY_COEF", "0.0")
    monkeypatch.setenv("NANODISCOVER_REMOVE_CONSTANT_REWARD_GROUPS", "0")
    monkeypatch.setenv("NANODISCOVER_LORA_RANK", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_ALPHA", "1")
    monkeypatch.setenv("NANODISCOVER_LORA_DROPOUT", "0.0")
    monkeypatch.setenv("NANODISCOVER_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head")
    monkeypatch.setenv("NANODISCOVER_NUM_SUBSTEPS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_NUM_WORKERS", "1")
    monkeypatch.setenv("NANODISCOVER_TRAINER_LOGPROB_COMPUTE_DTYPE", "float32")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE", "4096")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_REFERENCE_SCORING_MAX_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("NANODISCOVER_GRADIENT_CHECKPOINTING", "1")
    monkeypatch.setenv("NANODISCOVER_OPTIMIZER_STATE_KEEP_WINDOW", "2")
    monkeypatch.setenv("NANODISCOVER_DISTRIBUTED_STRATEGY", "ddp")
    monkeypatch.setenv("NANODISCOVER_SEQUENCE_PARALLEL_SIZE", "1")
    monkeypatch.setenv("NANODISCOVER_USE_REMOVE_PADDING", "1")

    import main as main_module
    monkeypatch.setattr(main_module, "run", lambda cfg: {})
    main_module.main()

    content = orig.read_text()
    assert "start" in content
    assert "logging_to" in content
    assert "run_session_start" in content
    assert not any(base.glob("log.txt.*"))


def test_run_uses_dynamic_import_for_ac1(monkeypatch, tmp_path):
    evaluator_python = tmp_path / "math-eval-python"
    evaluator_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    evaluator_python.chmod(0o755)
    monkeypatch.setenv("NANODISCOVER_EVAL_PYTHON", str(evaluator_python))

    cfg = RunConfig(
        task_name="ac1",
        num_epochs=0,
        seeds_per_epoch=1,
        rollouts_per_seed=1,
        evaluator_num_workers=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / "run-import"),
        resume_dir=None,
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="dry-run",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )

    result = run(cfg)

    assert result["run_dir"] == str((tmp_path / "run-import").resolve())


def test_main_loop_tracks_best_raw_score_for_maximize_task(tmp_path):
    cfg = RunConfig(
        task_name="ac1",
        num_epochs=1,
        seeds_per_epoch=1,
        rollouts_per_seed=2,
        evaluator_num_workers=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / "run-max"),
        resume_dir=None,
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="deepspeed",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )

    class _MaxTask(_FakeTask):
        def __init__(self):
            self._seen = 0

        def evaluate_code(self, *, parsed_code, state, epoch, seed):
            _ = (parsed_code, state, epoch, seed)
            self._seen += 1
            raw_score = 1.0 if self._seen == 1 else 3.0
            return {
                "correctness": 1.0,
                "performance": raw_score,
                "raw_score": raw_score,
                "msg": "ok",
                "result_payload": {"result_construction": [raw_score]},
                "stdout": "",
            }

        def compute_reward(self, eval_output):
            return float(eval_output["raw_score"])

        def make_next_state(self, *, parent_state, parsed_code, eval_output, epoch):
            _ = (parent_state, parsed_code)
            return ArchiveNode(epoch=epoch, value=float(eval_output["raw_score"]), task_payload={"construction": [1.0], "code": "x"})

    result = run(cfg, task=_MaxTask(), generator=_FakeGenerator(), evaluator=_FakeEvaluator(), trainer=_FakeTrainer())

    assert result["best_raw_score"] == 3.0


def test_main_loop_tracks_best_raw_score_for_minimize_task(tmp_path):
    cfg = RunConfig(
        task_name="ac1",
        num_epochs=1,
        seeds_per_epoch=1,
        rollouts_per_seed=2,
        evaluator_num_workers=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / "run-min"),
        resume_dir=None,
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="deepspeed",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )

    class _MinTask(_FakeTask):
        maximize_raw_score = False

        def __init__(self):
            self._seen = 0

        def evaluate_code(self, *, parsed_code, state, epoch, seed):
            _ = (parsed_code, state, epoch, seed)
            self._seen += 1
            raw_score = 3.0 if self._seen == 1 else 1.0
            return {
                "correctness": 1.0,
                "performance": -raw_score,
                "raw_score": raw_score,
                "msg": "ok",
                "result_payload": {"result_construction": [raw_score]},
                "stdout": "",
            }

        def compute_reward(self, eval_output):
            return 1.0 / float(eval_output["raw_score"])

        def make_next_state(self, *, parent_state, parsed_code, eval_output, epoch):
            _ = (parent_state, parsed_code)
            return ArchiveNode(epoch=epoch, value=-float(eval_output["raw_score"]), task_payload={"construction": [1.0], "code": "x"})

    result = run(cfg, task=_MinTask(), generator=_FakeGenerator(), evaluator=_FakeEvaluator(), trainer=_FakeTrainer())

    assert result["best_raw_score"] == 1.0


def test_main_loop_ignores_failed_raw_score_in_minimize_tracking(tmp_path):
    cfg = RunConfig(
        task_name="ac1",
        num_epochs=1,
        seeds_per_epoch=1,
        rollouts_per_seed=2,
        evaluator_num_workers=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / "run-min-failure-filter"),
        resume_dir=None,
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="deepspeed",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )

    class _MinTask(_FakeTask):
        maximize_raw_score = False

        def __init__(self):
            self._seen = 0

        def evaluate_code(self, *, parsed_code, state, epoch, seed):
            _ = (parsed_code, state, epoch, seed)
            self._seen += 1
            if self._seen == 1:
                return {
                    "correctness": 0.0,
                    "performance": 0.0,
                    "raw_score": 0.0,
                    "msg": "failed",
                    "result_payload": {},
                    "stdout": "",
                }
            return {
                "correctness": 1.0,
                "performance": -1.0,
                "raw_score": 1.0,
                "msg": "ok",
                "result_payload": {"result_construction": [1.0]},
                "stdout": "",
            }

        def compute_reward(self, eval_output):
            if float(eval_output.get("correctness", 0.0) or 0.0) <= 0:
                return 0.0
            return 1.0 / float(eval_output["raw_score"])

        def make_next_state(self, *, parent_state, parsed_code, eval_output, epoch):
            _ = (parent_state, parsed_code)
            if float(eval_output.get("correctness", 0.0) or 0.0) <= 0:
                return None
            return ArchiveNode(epoch=epoch, value=-float(eval_output["raw_score"]), task_payload={"construction": [1.0], "code": "x"})

    result = run(cfg, task=_MinTask(), generator=_FakeGenerator(), evaluator=_FakeEvaluator(), trainer=_FakeTrainer())

    assert result["best_raw_score"] == 1.0


def test_run_resumes_from_saved_generation_without_regenerating(tmp_path):
    class _FailingEvaluator:
        def __init__(self):
            self.events = []

        def evaluate_batch(self, **kwargs):
            self.events.append("evaluate")
            raise RuntimeError("stop after generation")

        def teardown(self):
            self.events.append("evaluate_teardown")

    first_cfg = RunConfig(
        task_name="ac1",
        num_epochs=1,
        seeds_per_epoch=1,
        rollouts_per_seed=1,
        evaluator_num_workers=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / "run-resume"),
        resume_dir=None,
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="deepspeed",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )

    first_eval = _FailingEvaluator()
    with pytest.raises(RuntimeError, match="stop after generation"):
        run(first_cfg, task=_FakeTask(), generator=_FakeGenerator(), evaluator=first_eval, trainer=_FakeTrainer())
    assert first_eval.events == ["evaluate", "evaluate_teardown"]

    epoch_dir = tmp_path / "run-resume" / "epoch000"
    assert (epoch_dir / "sample.json").exists()
    assert (epoch_dir / "generation.json").exists()
    assert not (epoch_dir / "evaluation.json").exists()

    resume_cfg = RunConfig(
        task_name="ac1",
        num_epochs=1,
        seeds_per_epoch=1,
        rollouts_per_seed=1,
        evaluator_num_workers=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / "run-resume"),
        resume_dir=str(tmp_path / "run-resume"),
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="deepspeed",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )
    resumed_gen = _FakeGenerator()
    resumed_eval = _FakeEvaluator()
    resumed_train = _FakeTrainer()

    run(resume_cfg, task=_FakeTask(), generator=resumed_gen, evaluator=resumed_eval, trainer=resumed_train)

    assert resumed_gen.events == []
    assert resumed_eval.events == ["evaluate", "evaluate_teardown"]
    assert resumed_train.events == ["train", "train_teardown"]
    assert (epoch_dir / "evaluation.json").exists()
    assert (epoch_dir / "archive.json").exists()
    assert (epoch_dir / "sampler.json").exists()
    assert (epoch_dir / "training.json").exists()


def test_run_uses_epoch_local_state_checkpoints_only(tmp_path):
    cfg = RunConfig(
        task_name="fake",
        num_epochs=1,
        seeds_per_epoch=1,
        rollouts_per_seed=1,
        generator_data_parallel_size=1,
        generator_tensor_parallel_size=1,
        evaluator_num_workers=1,
        generator_gpu_memory_utilization=0.98,
        generator_max_num_batched_tokens=1024,
        generator_max_num_seqs=4,
        generator_request_parallelism=2,
        generator_request_timeout_s=60.0,
        generator_backend_name="ray_data_llm",
        run_dir=str(tmp_path / "run-epoch-local"),
        resume_dir=None,
        max_archive_size=10,
        topk_children=2,
        puct_c=1.0,
        model_name_or_path="fake",
        tokenizer_name_or_path=None,
        renderer_name="plain_text",
        renderer_system_prompt="",
        renderer_stop_sequence="",
        temperature=1.0,
        phase1_max_tokens=8,
        context_window=16,
        context_buffer=0,
        final_answer_marker=None,
        forced_final_suffix=None,
        phase1_end_marker=None,
        forced_final_suffix_after_phase1_end_marker=None,
        train_backend="deepspeed",
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=False,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="ddp",
        sequence_parallel_size=1,
        use_remove_padding=True,
        trainer_logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        optimizer_state_keep_window=2,
    )

    run(cfg, task=_FakeTask(), generator=_FakeGenerator(), evaluator=_FakeEvaluator(), trainer=_FakeTrainer())

    epoch_dir = Path(cfg.run_dir) / "epoch000"
    assert (epoch_dir / "archive.json").exists()
    assert (epoch_dir / "sampler.json").exists()
    assert not (Path(cfg.run_dir) / "archive.json").exists()
    assert not (Path(cfg.run_dir) / "sampler.json").exists()