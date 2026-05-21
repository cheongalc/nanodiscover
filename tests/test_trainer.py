from pathlib import Path

import pytest

from core.archive import ArchiveNode
from core.evaluator import EvaluatedRollout
import core.trainer as trainer_module
from core.trainer import TrainerConfig, Trainer, build_loss_sample, build_training_microbatches, collect_reference_kl_diffs, count_padding_samples, drop_constant_reward_groups, gather_aligned_target_logprobs, gather_aligned_target_logprobs_chunked_float32, gather_target_logprobs, gather_target_logprobs_chunked_float32, resolve_logprob_compute_dtype, resolve_reference_policy_device, save_prepared_peft_adapter, score_reference_policy_logprobs, shard_loss_samples_for_rank, shard_reference_loss_samples_for_rank, solve_adaptive_beta, validate_adapter_checkpoint, validate_training_topology


def _trainer_config(tmp_path, *, backend_name: str = "dry-run", kl_penalty_coef: float = 0.0, model_name_or_path: str = "fake-model") -> TrainerConfig:
    return TrainerConfig(
        backend_name=backend_name,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=None,
        run_dir=str(tmp_path),
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=kl_penalty_coef,
        remove_constant_reward_groups=True,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        num_substeps=1,
        trainer_num_workers=1,
        trainer_max_tokens_per_rank=None,
        distributed_strategy="fsdp2",
        sequence_parallel_size=1,
        use_remove_padding=True,
        logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
    )


def _rollout(reward: float, mask: list[float]) -> EvaluatedRollout:
    return EvaluatedRollout(
        seed_state=ArchiveNode(id="seed", epoch=0, value=1.0, task_payload={}),
        prompt_text="prompt",
        response_text="resp",
        prompt_token_ids=[10, 11],
        completion_token_ids=[12, 13],
        completion_logprobs=[-0.1, -0.2],
        completion_mask=mask,
        finish_reason="stop",
        parsed_code="print(1)",
        reward=reward,
        correctness=1.0,
        performance=-reward,
        raw_score=reward,
        archive_value=-reward,
        next_state=ArchiveNode(epoch=1, value=-reward, task_payload={}),
        msg="ok",
    )


def test_build_loss_sample_respects_prompt_and_forced_mask():
    sample = build_loss_sample(_rollout(1.0, [1.0, 0.0]), 0.7)
    assert sample is not None
    assert sample.mask == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("completion_logprobs", [-0.1], "completion_logprobs"),
        ("completion_mask", [1.0], "completion_mask"),
    ],
)
def test_build_loss_sample_rejects_misaligned_completion_lengths(field_name, field_value, message):
    rollout = _rollout(1.0, [1.0, 1.0])
    setattr(rollout, field_name, field_value)

    with pytest.raises(ValueError, match=message):
        build_loss_sample(rollout, 0.7)


def test_constant_reward_groups_are_dropped_but_one_survives():
    groups, dropped = drop_constant_reward_groups([[ _rollout(1.0, [1.0, 1.0]), _rollout(1.0, [1.0, 1.0]) ]])
    assert len(groups) == 1
    assert dropped == 0


def test_adaptive_beta_is_non_negative():
    beta = solve_adaptive_beta([0.1, 0.2, 0.9, 1.1])
    assert beta >= 0.0


def test_compute_advantages_matches_numeric_fixture():
    advantages = trainer_module.compute_advantages([[0.1, 0.2, 0.9, 1.1]])
    assert advantages == [
        pytest.approx(
            [-0.9780413533693426, -0.9651781686079196, 0.17179206902718258, 6.012514377062389],
            rel=1e-12,
            abs=1e-12,
        )
    ]


def test_trainer_dry_run_writes_adapter(tmp_path):
    trainer = Trainer(_trainer_config(tmp_path, backend_name="dry-run", kl_penalty_coef=0.0))
    result = trainer.train([[ _rollout(1.0, [1.0, 1.0]), _rollout(2.0, [1.0, 1.0]) ]], epoch=0)
    assert result.adapter_path is not None


def test_trainer_uses_adaptive_entropic_advantages_by_default(tmp_path):
    trainer = Trainer(_trainer_config(tmp_path, backend_name="dry-run", kl_penalty_coef=0.0))
    result = trainer.train([[ _rollout(1.0, [1.0, 1.0]), _rollout(2.0, [1.0, 1.0]) ]], epoch=0)
    assert result.loss_samples


def test_trainer_passes_entropic_advantages_into_loss_samples_without_drifting(tmp_path):
    class _CapturingBackend:
        def __init__(self):
            self.loss_samples = None
            self.output_dir = None

        def train_step(self, loss_samples, *, output_dir):
            self.loss_samples = list(loss_samples)
            self.output_dir = output_dir
            return {"train/loss": 0.0}

        def adapter_path(self):
            return str(self.output_dir)

    backend = _CapturingBackend()
    trainer = Trainer(_trainer_config(tmp_path, backend_name="dry-run", kl_penalty_coef=0.0), backend=backend)

    result = trainer.train(
        [[
            _rollout(0.1, [1.0, 1.0]),
            _rollout(0.2, [1.0, 1.0]),
            _rollout(0.9, [1.0, 1.0]),
            _rollout(1.1, [1.0, 1.0]),
        ]],
        epoch=0,
    )

    assert backend.loss_samples is not None
    assert [sample.mask for sample in backend.loss_samples] == [[0.0, 1.0, 1.0]] * 4
    assert [sample.advantages for sample in backend.loss_samples] == [
        pytest.approx([0.0, -0.9780413533693426, -0.9780413533693426], rel=1e-12, abs=1e-12),
        pytest.approx([0.0, -0.9651781686079196, -0.9651781686079196], rel=1e-12, abs=1e-12),
        pytest.approx([0.0, 0.17179206902718258, 0.17179206902718258], rel=1e-12, abs=1e-12),
        pytest.approx([0.0, 6.012514377062389, 6.012514377062389], rel=1e-12, abs=1e-12),
    ]
    assert result.adapter_path == str(backend.output_dir)


def test_trainer_skips_empty_loss_samples_without_inventing_adapter_path(tmp_path):
    trainer = Trainer(_trainer_config(tmp_path, backend_name="dry-run", kl_penalty_coef=0.0))
    trainer.set_resume_adapter("/tmp/existing-adapter")
    rollout = EvaluatedRollout(
        seed_state=ArchiveNode(id="seed", epoch=0, value=1.0, task_payload={}),
        prompt_text="prompt",
        response_text="resp",
        prompt_token_ids=[10, 11],
        completion_token_ids=[],
        completion_logprobs=[],
        completion_mask=[],
        finish_reason="stop",
        parsed_code="print(1)",
        reward=1.0,
        correctness=1.0,
        performance=-1.0,
        raw_score=1.0,
        archive_value=-1.0,
        next_state=ArchiveNode(epoch=1, value=-1.0, task_payload={}),
        msg="ok",
    )

    result = trainer.train([[rollout]], epoch=0, output_dir=tmp_path / "epoch000")

    assert result.skipped is True
    assert result.metrics["num_loss_samples"] == 0.0
    assert result.metrics["skipped_empty_loss_samples"] == 1.0
    assert result.adapter_path == "/tmp/existing-adapter"


def test_trainer_records_dropped_invalid_loss_samples_and_continues(tmp_path):
    trainer = Trainer(_trainer_config(tmp_path, backend_name="dry-run", kl_penalty_coef=0.0))
    valid = _rollout(1.0, [1.0, 1.0])
    invalid = _rollout(2.0, [])
    invalid.completion_token_ids = []
    invalid.completion_logprobs = []

    result = trainer.train([[valid, invalid]], epoch=0)

    assert result.skipped is False
    assert result.metrics["dropped_invalid_loss_samples"] == 1.0
    assert result.metrics["num_loss_samples"] == 1.0
    assert result.adapter_path is not None


def test_trainer_dry_run_applies_kl_with_backend_owned_scorer(tmp_path, monkeypatch):
    created = []

    class _FakeReferenceScorer:
        def __init__(self, model_name_or_path, *, logprob_compute_dtype, vocab_chunk_size):
            created.append(("created", model_name_or_path, logprob_compute_dtype, vocab_chunk_size))

        def score_loss_samples(self, loss_samples):
            return [[-0.3] * len(sample.target_token_ids) for sample in loss_samples]

        def close(self):
            created.append("closed")

    trainer = Trainer(_trainer_config(tmp_path, backend_name="dry-run", kl_penalty_coef=0.1, model_name_or_path="fake-base"))
    monkeypatch.setattr(trainer_module, "ReferencePolicyLogprobScorer", _FakeReferenceScorer)

    result = trainer.train([[ _rollout(1.0, [1.0, 1.0]), _rollout(2.0, [1.0, 1.0]) ]], epoch=0)

    assert ("created", "fake-base", "float32", 4096) in created
    assert "closed" in created
    assert "kl_policy_base" in result.metrics


def test_validate_training_topology_rejects_inactive_ulysses(tmp_path):
    config = _trainer_config(tmp_path, backend_name="deepspeed")
    config.distributed_strategy = "ddp"
    config.sequence_parallel_size = 4
    config.trainer_num_workers = 1

    try:
        validate_training_topology(config)
    except ValueError as exc:
        assert "inactive" in str(exc)
    else:
        raise AssertionError("Expected inactive Ulysses topology to raise")


def test_validate_training_topology_rejects_deepspeed_sp_without_remove_padding(tmp_path):
    config = _trainer_config(tmp_path, backend_name="deepspeed")
    config.distributed_strategy = "ddp"
    config.sequence_parallel_size = 2
    config.trainer_num_workers = 2
    config.use_remove_padding = False

    with pytest.raises(ValueError, match="requires use_remove_padding"):
        validate_training_topology(config)


def test_validate_training_topology_rejects_reference_model_parallel_size_that_does_not_divide_workers(tmp_path):
    config = _trainer_config(tmp_path, backend_name="deepspeed")
    config.distributed_strategy = "ddp"
    config.sequence_parallel_size = 8
    config.trainer_num_workers = 8
    config.reference_scoring_model_parallel_size = 3

    with pytest.raises(ValueError, match="reference_scoring_model_parallel_size"):
        validate_training_topology(config)


def test_validate_training_topology_rejects_removed_verl_backend(tmp_path):
    config = _trainer_config(tmp_path, backend_name="verl")

    with pytest.raises(ValueError, match="Unsupported trainer backend"):
        validate_training_topology(config)


def test_trainer_selects_deepspeed_backend_without_changing_default_path(tmp_path, monkeypatch):
    created = []

    class _FakeDeepSpeedBackend:
        def __init__(self, config):
            self.config = config
            self.adapter_path_value = str(tmp_path / "adapter")
            created.append(config.backend_name)

        def set_resume_adapter(self, adapter_path):
            _ = adapter_path

        def train_step(self, loss_samples, *, output_dir):
            _ = (loss_samples, output_dir)
            return {"train/loss": 1.0}

        def adapter_path(self):
            return self.adapter_path_value

    monkeypatch.setattr(trainer_module, "DeepSpeedBackend", _FakeDeepSpeedBackend)

    config = _trainer_config(tmp_path, backend_name="deepspeed")
    config.use_remove_padding = False
    trainer = Trainer(config)
    result = trainer.train([[ _rollout(1.0, [1.0, 1.0]), _rollout(2.0, [1.0, 1.0]) ]], epoch=0)

    assert created == ["deepspeed"]
    assert isinstance(trainer.backend, _FakeDeepSpeedBackend)
    assert result.adapter_path == str(tmp_path / "adapter")


def test_build_trainer_backend_rejects_removed_verl_backend(tmp_path):
    config = _trainer_config(tmp_path, backend_name="verl")

    with pytest.raises(RuntimeError, match="Unsupported trainer backend"):
        trainer_module.build_trainer_backend(config)


def test_deepspeed_backend_uses_ddp_stage_mode_for_multiworker_runs(tmp_path):
    config = _trainer_config(tmp_path, backend_name="deepspeed")
    config.use_remove_padding = False
    config.trainer_num_workers = 2

    backend = trainer_module.DeepSpeedBackend(config)
    command = backend.build_stage_command(Path("/tmp/stage_payload.pkl"))
    mode_index = command.index("--distributed-mode")

    assert backend.distributed_mode == "ddp"
    assert command[mode_index + 1] == "ddp"


def test_build_training_microbatches_respects_token_budget(tmp_path):
    config = _trainer_config(tmp_path)
    samples = [
        build_loss_sample(_rollout(1.0, [1.0, 1.0]), 0.1),
        build_loss_sample(_rollout(2.0, [1.0, 1.0]), 0.2),
        build_loss_sample(_rollout(3.0, [1.0, 1.0]), 0.3),
    ]
    filtered = [sample for sample in samples if sample is not None]

    microbatches = build_training_microbatches(filtered, num_substeps=1, max_tokens_per_batch=3)

    assert len(microbatches) == 3
    assert all(len(batch) == 1 for batch in microbatches)


def test_reference_policy_device_uses_local_rank(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

    class _FakeTorch:
        cuda = _FakeCuda()

        @staticmethod
        def device(value):
            return value

    monkeypatch.setenv("LOCAL_RANK", "1")

    assert resolve_reference_policy_device(_FakeTorch) == "cuda:1"


def test_gather_target_logprobs_matches_log_softmax():
    import torch

    logits = torch.tensor(
        [[[1.0, 0.0, -1.0], [0.5, -0.5, 0.0], [2.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[0, 2, 1]], dtype=torch.long)

    expected = torch.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    actual = gather_target_logprobs(torch, logits, input_ids)

    assert torch.allclose(actual, expected)


def test_gather_target_logprobs_chunked_float32_matches_full_float32_path():
    import torch

    logits = torch.randn((2, 5, 19), dtype=torch.float32)
    input_ids = torch.tensor(
        [
            [0, 3, 7, 4, 1],
            [2, 5, 6, 8, 9],
        ],
        dtype=torch.long,
    )

    full = gather_target_logprobs(torch, logits, input_ids, compute_dtype=torch.float32)
    chunked = gather_target_logprobs_chunked_float32(torch, logits, input_ids, vocab_chunk_size=4)

    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-6)


def test_gather_aligned_target_logprobs_chunked_float32_matches_full_float32_path():
    import torch

    logits = torch.randn((2, 4, 17), dtype=torch.float32)
    target_ids = torch.tensor(
        [
            [0, 3, 7, 4],
            [2, 5, 6, 8],
        ],
        dtype=torch.long,
    )

    full = gather_aligned_target_logprobs(torch, logits, target_ids, compute_dtype=torch.float32)
    chunked = gather_aligned_target_logprobs_chunked_float32(torch, logits, target_ids, vocab_chunk_size=5)

    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-6)


def test_resolve_logprob_compute_dtype_accepts_aliases_and_cpu_fallback():
    class _FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"

    assert resolve_logprob_compute_dtype(_FakeTorch, "bf16", device_type="cuda") == "bfloat16"
    assert resolve_logprob_compute_dtype(_FakeTorch, "fp16", device_type="cuda") == "float16"
    assert resolve_logprob_compute_dtype(_FakeTorch, "bf16", device_type="cpu") == "float32"


def test_reference_policy_logprobs_respect_token_budget_without_reordering_outputs():
    class _FakeReferenceScorer:
        def __init__(self):
            self.calls = []

        def score_loss_samples(self, loss_samples):
            self.calls.append([len(sample.model_input_ids) for sample in loss_samples])
            return [[float(sample.model_input_ids[0])] * len(sample.target_token_ids) for sample in loss_samples]

    samples = [
        trainer_module.LossSample(model_input_ids=[30, 31, 32], target_token_ids=[31, 32, 33], sampling_logprobs=[0.0] * 3, advantages=[0.0] * 3, mask=[1.0] * 3, full_sequence_ids=[30, 31, 32, 33]),
        trainer_module.LossSample(model_input_ids=[10], target_token_ids=[11], sampling_logprobs=[0.0], advantages=[0.0], mask=[1.0], full_sequence_ids=[10, 11]),
        trainer_module.LossSample(model_input_ids=[20, 21], target_token_ids=[21, 22], sampling_logprobs=[0.0] * 2, advantages=[0.0] * 2, mask=[1.0] * 2, full_sequence_ids=[20, 21, 22]),
    ]
    scorer = _FakeReferenceScorer()

    scored = score_reference_policy_logprobs(scorer, samples, max_tokens_per_batch=3)

    assert scorer.calls == [[1, 2], [3]]
    assert scored == [[30.0, 30.0, 30.0], [10.0], [20.0, 20.0]]
def test_topology_summary_includes_distributed_strategy(tmp_path):
    cfg = _trainer_config(tmp_path, backend_name="dry-run")
    cfg.distributed_strategy = "fsdp2"
    topo = trainer_module.topology_summary(cfg)
    assert topo.get("distributed_strategy") == "fsdp2"


def test_pack_batch_tensors_preserves_loss_sample_alignment():
    import torch

    runner = object.__new__(trainer_module.StageRunnerBase)
    runner.torch = torch
    runner.input_device = torch.device("cpu")
    runner.pad_token_id = 0
    runner.sequence_parallel_size = 2
    runner.distributed_active = True

    samples = [
        trainer_module.LossSample(
            model_input_ids=[10, 11, 12],
            target_token_ids=[11, 12, 13],
            sampling_logprobs=[-0.1, -0.2, -0.3],
            advantages=[0.0, 1.0, 1.0],
            mask=[0.0, 1.0, 1.0],
            full_sequence_ids=[10, 11, 12, 13],
        ),
        trainer_module.LossSample(
            model_input_ids=[20, 21],
            target_token_ids=[21, 22],
            sampling_logprobs=[-0.4, -0.5],
            advantages=[0.0, 2.0],
            mask=[0.0, 1.0],
            full_sequence_ids=[20, 21, 22],
        ),
    ]

    batch = runner.pack_batch_tensors(samples)

    assert batch.input_ids.tolist() == [[10, 11, 12, 20, 21, 0]]
    assert batch.target_ids.tolist() == [[11, 12, 13, 21, 22, 0]]
    assert batch.old_logprobs.tolist()[0] == pytest.approx([-0.1, -0.2, -0.3, -0.4, -0.5, 0.0])
    assert batch.advantages.tolist()[0] == pytest.approx([0.0, 1.0, 1.0, 0.0, 2.0, 0.0])
    assert batch.mask.tolist()[0] == pytest.approx([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    assert batch.position_ids.tolist() == [[0, 1, 2, 0, 1, 0]]


def test_pack_batch_tensors_position_ids_match_transformers_varlen_boundaries():
    import torch
    flash_attention_utils = pytest.importorskip("transformers.modeling_flash_attention_utils")

    runner = object.__new__(trainer_module.StageRunnerBase)
    runner.torch = torch
    runner.input_device = torch.device("cpu")
    runner.pad_token_id = 0
    runner.sequence_parallel_size = 2
    runner.distributed_active = True

    sample_a = trainer_module.LossSample(
        model_input_ids=[10, 11, 12],
        target_token_ids=[11, 12, 13],
        sampling_logprobs=[-0.1, -0.2, -0.3],
        advantages=[0.0, 1.0, 1.0],
        mask=[0.0, 1.0, 1.0],
        full_sequence_ids=[10, 11, 12, 13],
    )
    sample_b = trainer_module.LossSample(
        model_input_ids=[20, 21],
        target_token_ids=[21, 22],
        sampling_logprobs=[-0.4, -0.5],
        advantages=[0.0, 2.0],
        mask=[0.0, 1.0],
        full_sequence_ids=[20, 21, 22],
    )

    packed = runner.pack_batch_tensors([sample_a, sample_b])
    (cu_q, cu_k), (max_q, max_k) = flash_attention_utils.prepare_fa_kwargs_from_position_ids(
        packed.position_ids
    )

    assert packed.position_ids.tolist() == [[0, 1, 2, 0, 1, 0]]
    assert cu_q.tolist() == [0, 3, 5, 6]
    assert cu_k.tolist() == [0, 3, 5, 6]
    assert max_q == 3
    assert max_k == 3


def test_monotonic_position_ids_encode_one_flash_attention_segment():
    import torch
    flash_attention_utils = pytest.importorskip("transformers.modeling_flash_attention_utils")

    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    (cu_q, cu_k), (max_q, max_k) = flash_attention_utils.prepare_fa_kwargs_from_position_ids(position_ids)

    assert cu_q.tolist() == [0, 6]
    assert cu_k.tolist() == [0, 6]
    assert max_q == 6
    assert max_k == 6


def test_shard_loss_samples_replication_within_sp_group():
    samples = [
        trainer_module.LossSample(
            model_input_ids=[idx, idx + 1],
            target_token_ids=[idx + 1, idx + 2],
            sampling_logprobs=[0.0, 0.0],
            advantages=[0.0, 0.0],
            mask=[1.0, 1.0],
            full_sequence_ids=[idx, idx + 1, idx + 2],
        )
        for idx in range(16)
    ]

    rank0, usable0, padding0 = shard_loss_samples_for_rank(samples, world_size=8, rank=0, sequence_parallel_size=4)
    rank1, usable1, padding1 = shard_loss_samples_for_rank(samples, world_size=8, rank=1, sequence_parallel_size=4)
    rank4, usable4, padding4 = shard_loss_samples_for_rank(samples, world_size=8, rank=4, sequence_parallel_size=4)

    assert usable0 == 16
    assert usable1 == 16
    assert usable4 == 16
    assert padding0 == 0
    assert padding1 == 0
    assert padding4 == 0
    assert [item.model_input_ids for item in rank0] == [item.model_input_ids for item in rank1]
    assert [item.model_input_ids for item in rank0] != [item.model_input_ids for item in rank4]


def test_shard_loss_samples_pads_instead_of_truncating_when_count_is_uneven():
    samples = [
        trainer_module.LossSample(
            model_input_ids=[idx, idx + 1],
            target_token_ids=[idx + 1, idx + 2],
            sampling_logprobs=[0.0, 0.0],
            advantages=[0.0, 0.0],
            mask=[1.0, 1.0],
            full_sequence_ids=[idx, idx + 1, idx + 2],
        )
        for idx in range(5)
    ]

    rank0, sample_count0, padding_count0 = shard_loss_samples_for_rank(samples, world_size=2, rank=0, sequence_parallel_size=1)
    rank1, sample_count1, padding_count1 = shard_loss_samples_for_rank(samples, world_size=2, rank=1, sequence_parallel_size=1)

    assert sample_count0 == 5
    assert sample_count1 == 5
    assert padding_count0 == 1
    assert padding_count1 == 1
    assert len(rank0) + len(rank1) == 6
    assert count_padding_samples(rank0) + count_padding_samples(rank1) == 1


def test_shard_reference_loss_samples_for_rank_splits_work_without_padding():
    samples = [
        trainer_module.LossSample(
            model_input_ids=[idx, idx + 1],
            target_token_ids=[idx + 1, idx + 2],
            sampling_logprobs=[0.0, 0.0],
            advantages=[0.0, 0.0],
            mask=[1.0, 1.0],
            full_sequence_ids=[idx, idx + 1, idx + 2],
        )
        for idx in range(8)
    ]

    rank0_indices, rank0_samples = shard_reference_loss_samples_for_rank(
        samples,
        world_size=8,
        rank=0,
        model_parallel_size=1,
    )
    rank1_indices, rank1_samples = shard_reference_loss_samples_for_rank(
        samples,
        world_size=8,
        rank=1,
        model_parallel_size=1,
    )

    assert rank0_indices
    assert rank1_indices
    assert set(rank0_indices).isdisjoint(set(rank1_indices))
    assert [samples[index].model_input_ids for index in rank0_indices] == [
        sample.model_input_ids for sample in rank0_samples
    ]


def test_collect_reference_kl_diffs_accepts_duplicate_replica_groups():
    merged = collect_reference_kl_diffs(
        3,
        [
            [(0, [0.1]), (1, [0.2])],
            [(0, [0.1]), (1, [0.2])],
            [(2, [0.3])],
            [(2, [0.3])],
        ],
    )

    assert merged == [[0.1], [0.2], [0.3]]


def test_compute_kl_diffs_rejects_misaligned_base_logprobs():
    sample = trainer_module.LossSample(
        model_input_ids=[10, 11],
        target_token_ids=[11, 12],
        sampling_logprobs=[-0.1, -0.2],
        advantages=[0.0, 1.0],
        mask=[0.0, 1.0],
        full_sequence_ids=[10, 11, 12],
    )

    with pytest.raises(ValueError, match="align to target tokens"):
        trainer_module.compute_kl_diffs([sample], [[-0.3]])


def test_apply_kl_adjustment_matches_masked_formula():
    sample = trainer_module.LossSample(
        model_input_ids=[10, 11, 12],
        target_token_ids=[11, 12, 13],
        sampling_logprobs=[-0.1, -0.2, -0.3],
        advantages=[5.0, 6.0, 7.0],
        mask=[0.0, 1.0, 1.0],
        full_sequence_ids=[10, 11, 12, 13],
    )

    metrics = trainer_module.apply_kl_adjustment(
        [sample],
        [[0.25, 0.5, -0.5]],
        average_diff=1.25,
        kl_penalty_coef=0.2,
    )

    assert metrics == {"kl_policy_base": 1.25}
    assert sample.advantages == pytest.approx(
        [
            5.0,
            6.0 + (0.2 * 1.0 * (1.25 - 0.5)),
            7.0 + (0.2 * 1.0 * (1.25 - (-0.5))),
        ]
    )


def test_score_reference_policy_logprobs_splits_cuda_oom_batches_and_preserves_order():
    class _FakeCuda:
        def __init__(self):
            self.empty_cache_calls = 0

        @staticmethod
        def is_available():
            return True

        def empty_cache(self):
            self.empty_cache_calls += 1

    class _FakeTorch:
        def __init__(self):
            self.cuda = _FakeCuda()

    class _FakeReferenceScorer:
        def __init__(self):
            self.calls = []
            self.torch_module = _FakeTorch()

        def score_loss_samples(self, loss_samples):
            lengths = [len(sample.model_input_ids) for sample in loss_samples]
            self.calls.append(lengths)
            if len(loss_samples) > 1:
                raise RuntimeError("CUDA out of memory while scoring reference policy")
            return [[float(loss_samples[0].model_input_ids[0])] * len(loss_samples[0].target_token_ids)]

    samples = [
        trainer_module.LossSample(
            model_input_ids=[30, 31, 32],
            target_token_ids=[31, 32, 33],
            sampling_logprobs=[0.0, 0.0, 0.0],
            advantages=[0.0, 0.0, 0.0],
            mask=[1.0, 1.0, 1.0],
            full_sequence_ids=[30, 31, 32, 33],
        ),
        trainer_module.LossSample(
            model_input_ids=[10],
            target_token_ids=[11],
            sampling_logprobs=[0.0],
            advantages=[0.0],
            mask=[1.0],
            full_sequence_ids=[10, 11],
        ),
        trainer_module.LossSample(
            model_input_ids=[20, 21],
            target_token_ids=[21, 22],
            sampling_logprobs=[0.0, 0.0],
            advantages=[0.0, 0.0],
            mask=[1.0, 1.0],
            full_sequence_ids=[20, 21, 22],
        ),
    ]
    scorer = _FakeReferenceScorer()

    scored = score_reference_policy_logprobs(scorer, samples, max_tokens_per_batch=10)

    assert scorer.calls[0] == [1, 2, 3]
    assert scorer.torch_module.cuda.empty_cache_calls == 2
    assert scored == [[30.0, 30.0, 30.0], [10.0], [20.0, 20.0]]


def test_run_stage_from_payload_trains_on_kl_adjusted_samples(tmp_path, monkeypatch):
    import pickle

    captured: dict[str, object] = {}

    class _FakeRunner:
        def train_step(self, samples):
            captured["advantages"] = [list(sample.advantages) for sample in samples]
            return {"train/loss": 0.0}

        def save_adapter(self, save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            return str(save_dir)

        def save_optimizer_state(self, save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            return str(save_dir)

    sample = trainer_module.LossSample(
        model_input_ids=[10, 11, 12],
        target_token_ids=[11, 12, 13],
        sampling_logprobs=[-0.1, -0.2, -0.3],
        advantages=[5.0, 6.0, 7.0],
        mask=[0.0, 1.0, 1.0],
        full_sequence_ids=[10, 11, 12, 13],
    )
    cfg = _trainer_config(tmp_path, backend_name="deepspeed", kl_penalty_coef=0.2, model_name_or_path="fake-base")
    payload_path = tmp_path / "stage_payload.pkl"
    output_path = tmp_path / "stage_output.json"
    payload = {
        "samples": [sample],
        "trainer_cfg": cfg,
        "adapter_dir": str(tmp_path / "adapter"),
        "optimizer_state_dir": str(tmp_path / "optimizer_state"),
        "output_path": str(output_path),
    }

    monkeypatch.setattr(trainer_module, "configure_stage_process_logging", lambda: None)
    monkeypatch.setattr(trainer_module, "build_stage_runner", lambda cfg: _FakeRunner())
    monkeypatch.setattr(
        trainer_module,
        "score_reference_policy_kl",
        lambda *args, **kwargs: ([[0.25, 0.5, -0.5]], 2.5, 2.0),
    )

    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    trainer_module.run_stage_from_payload(str(payload_path), distributed_mode="none", distributed_backend="nccl")

    assert captured["advantages"] == [
        pytest.approx(
            [
                5.0,
                6.0 + (0.2 * (1.25 - 0.5)),
                7.0 + (0.2 * (1.25 - (-0.5))),
            ]
        )
    ]


def test_save_prepared_peft_adapter_writes_nonempty_weights(tmp_path):
    import torch
    from safetensors.torch import load_file

    class _FakePeftConfig:
        def __init__(self):
            self.base_model_name_or_path = None
            self.inference_mode = False
            self.is_prompt_learning = False
            self.task_type = "CAUSAL_LM"

        def save_pretrained(self, output_dir, auto_mapping_dict=None):
            _ = auto_mapping_dict
            Path(output_dir, "adapter_config.json").write_text("{}", encoding="utf-8")

    class _FakeModel:
        def __init__(self):
            self.peft_config = {"default": _FakePeftConfig()}
            self.base_model = type(
                "BaseModelHolder",
                (),
                {"model": type("WrappedBaseModel", (), {"name_or_path": "fake-base"})()},
            )()

        def create_or_update_model_card(self, output_dir):
            Path(output_dir, "README.md").write_text("card", encoding="utf-8")

    adapter_state = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((2, 3), dtype=torch.float32),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros((3, 2), dtype=torch.float32),
    }

    weights_path, tensor_count, size_bytes = save_prepared_peft_adapter(
        _FakeModel(),
        tmp_path / "adapter",
        adapter_state,
    )

    assert weights_path == tmp_path / "adapter" / "adapter_model.safetensors"
    assert tensor_count == 2
    assert size_bytes > 40
    assert set(load_file(str(weights_path)).keys()) == set(adapter_state.keys())
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "adapter" / "README.md").exists()


def test_validate_adapter_checkpoint_rejects_empty_safetensors(tmp_path):
    from safetensors.torch import save_file

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file({}, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})

    try:
        validate_adapter_checkpoint(adapter_dir)
    except RuntimeError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected empty safetensors adapter to be rejected")
