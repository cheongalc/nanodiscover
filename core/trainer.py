from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from core.evaluator import EvaluatedRollout


logger = logging.getLogger(__name__)


class RankContextFilter(logging.Filter):
    """Inject the distributed rank into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = rank_from_env()
        return True


@dataclass
class TrainerConfig:
    """Configuration for one trainer backend invocation."""

    backend_name: str
    model_name_or_path: str
    tokenizer_name_or_path: str | None
    run_dir: str
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
    logprob_compute_dtype: str
    reference_logprob_vocab_chunk_size: int
    reference_scoring_max_tokens_per_rank: int
    reference_scoring_model_parallel_size: int
    gradient_checkpointing: bool
    distributed_mode: str = "auto"
    resume_adapter_path: str | None = None
    resume_optimizer_path: str | None = None
    distributed_backend: str = "nccl"


@dataclass
class LossSample:
    """Per-token training example derived from one evaluated rollout."""

    model_input_ids: list[int]
    target_token_ids: list[int]
    sampling_logprobs: list[float]
    advantages: list[float]
    mask: list[float]
    full_sequence_ids: list[int] = field(default_factory=list)
    prompt_token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PaddedBatch:
    """Dense padded tensors for the non-packed training path."""

    input_ids: Any
    target_ids: Any
    old_logprobs: Any
    advantages: Any
    mask: Any
    attention_mask: Any


@dataclass
class PackedBatch:
    """Packed tensors for remove-padding training."""

    input_ids: Any
    target_ids: Any
    old_logprobs: Any
    advantages: Any
    mask: Any
    position_ids: Any


@dataclass
class TrainingResult:
    """Trainer output persisted at the end of an epoch."""

    metrics: dict[str, float]
    adapter_path: str | None
    optimizer_state_dir: str | None = None
    loss_samples: list[LossSample] = field(default_factory=list)
    skipped: bool = False


class TrainerBackend(Protocol):
    def train_step(self, loss_samples: list[LossSample], *, output_dir: Path) -> dict[str, float]: ...
    def adapter_path(self) -> str | None: ...


class ReferenceLogprobScorer(Protocol):
    def score_loss_samples(self, loss_samples: list[LossSample]) -> list[list[float]]: ...
    def close(self) -> None: ...


def is_padding_loss_sample(sample: LossSample) -> bool:
    """Return whether a loss sample was synthesized only for DP padding."""

    return bool(sample.metadata.get("padding_sample"))


def count_padding_samples(samples: list[LossSample]) -> int:
    """Count how many samples in a batch are padding placeholders."""

    return int(sum(1 for sample in samples if is_padding_loss_sample(sample)))


def normalized_backend_name(backend_name: str) -> str:
    """Return the normalized trainer backend name."""

    return str(backend_name or "").strip().lower()


def validate_adapter_checkpoint(adapter_path: str | Path) -> tuple[Path, int, int]:
    """Validate an adapter directory and return basic weight-file stats.

    Args:
        adapter_path: Directory containing a PEFT adapter checkpoint.

    Returns:
        Tuple of `(weights_path, tensor_count, size_bytes)`.

    Raises:
        FileNotFoundError: If the adapter directory or weights file is missing.
        RuntimeError: If the weights file exists but contains no tensors.
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            tensor_count = len(list(handle.keys()))
        if tensor_count <= 0:
            raise RuntimeError(f"Adapter checkpoint is empty: {safetensors_path}")
        return safetensors_path, tensor_count, safetensors_path.stat().st_size

    pytorch_path = adapter_dir / "adapter_model.bin"
    if pytorch_path.exists():
        import torch

        state_dict = torch.load(str(pytorch_path), map_location="cpu", weights_only=True)
        tensor_count = len(state_dict) if isinstance(state_dict, dict) else 0
        if tensor_count <= 0:
            raise RuntimeError(f"Adapter checkpoint is empty: {pytorch_path}")
        return pytorch_path, tensor_count, pytorch_path.stat().st_size

    raise FileNotFoundError(f"Adapter weights file is missing from: {adapter_dir}")


def clone_adapter_state_dict_for_save(adapter_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Clone adapter tensors onto CPU for safe serialization.

    Args:
        adapter_state_dict: Adapter tensors already formatted for on-disk PEFT save.

    Returns:
        A detached CPU copy of the adapter state dict.

    Raises:
        TypeError: If a state-dict entry is not a tensor.
    """
    import torch

    serialized_state: dict[str, Any] = {}
    for name, value in adapter_state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Adapter state entry {name!r} is not a tensor: {type(value)!r}")
        serialized_state[name] = value.detach().cpu().clone()
    return serialized_state


def save_prepared_peft_adapter(
    model: Any,
    output_dir: str | Path,
    adapter_state_dict: dict[str, Any],
) -> tuple[Path, int, int]:
    """Persist a PEFT adapter state dict without re-filtering it.

    Args:
        model: The PEFT model whose config/model card should be saved.
        output_dir: Directory where the adapter should be written.
        adapter_state_dict: Adapter-only state dict in final PEFT on-disk key format.

    Returns:
        Tuple of `(weights_path, tensor_count, size_bytes)` for the written checkpoint.
    """
    from peft.utils.other import SAFETENSORS_WEIGHTS_NAME
    from safetensors.torch import save_file as safe_save_file

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not adapter_state_dict:
        raise RuntimeError("Refusing to save an empty adapter state dict.")
    if not hasattr(model, "peft_config") or "default" not in model.peft_config:
        raise TypeError("Expected a PEFT model with a default adapter when saving adapter weights.")

    serialized_state = clone_adapter_state_dict_for_save(adapter_state_dict)

    if hasattr(model, "create_or_update_model_card"):
        model.create_or_update_model_card(str(output_path))

    safe_save_file(
        serialized_state,
        str(output_path / SAFETENSORS_WEIGHTS_NAME),
        metadata={"format": "pt"},
    )

    peft_config = copy.deepcopy(model.peft_config["default"])
    if peft_config.base_model_name_or_path is None:
        peft_config.base_model_name_or_path = (
            model.base_model.__dict__.get("name_or_path", None)
            if peft_config.is_prompt_learning
            else model.base_model.model.__dict__.get("name_or_path", None)
        )
    inference_mode = peft_config.inference_mode
    peft_config.inference_mode = True

    auto_mapping_dict = None
    if peft_config.task_type is None and hasattr(model, "_get_base_model_class"):
        base_model_class = model._get_base_model_class(
            is_prompt_tuning=peft_config.is_prompt_learning,
        )
        auto_mapping_dict = {
            "base_model_class": base_model_class.__name__,
            "parent_library": base_model_class.__module__,
        }

    peft_config.save_pretrained(str(output_path), auto_mapping_dict=auto_mapping_dict)
    peft_config.inference_mode = inference_mode
    return validate_adapter_checkpoint(output_path)


def make_padding_loss_sample(template: LossSample) -> LossSample:
    """Build a zero-masked padding sample that preserves tensor shapes."""

    return LossSample(
        model_input_ids=list(template.model_input_ids),
        target_token_ids=list(template.target_token_ids),
        sampling_logprobs=[0.0] * len(template.sampling_logprobs),
        advantages=[0.0] * len(template.advantages),
        mask=[0.0] * len(template.mask),
        full_sequence_ids=list(template.full_sequence_ids),
        prompt_token_count=int(template.prompt_token_count),
        metadata={"padding_sample": True},
    )


def resolve_reference_policy_device(torch_module):
    """Return the device used for frozen reference-policy scoring."""

    if not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(f"cuda:{local_rank_from_env()}")


def resolve_logprob_compute_dtype(torch_module, dtype_name: str, *, device_type: str):
    normalized = str(dtype_name).strip().lower()
    mapping = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported logprob compute dtype: {dtype_name!r}")
    resolved = mapping[normalized]
    if device_type != "cuda" and resolved != torch_module.float32:
        logger.warning(
            "logprob compute dtype %s requested on %s; using float32 instead",
            dtype_name,
            device_type,
        )
        return torch_module.float32
    return resolved


def select_logprob_tensors(
    torch_module,
    logits,
    target_ids,
    *,
    shift_targets: bool,
    compute_dtype=None,
):
    """Align logits and target ids for causal or already-aligned logprob gathering."""

    selected_logits = logits[:, :-1, :] if shift_targets else logits
    selected_target_ids = target_ids[:, 1:] if shift_targets else target_ids
    selected_logits = selected_logits.to(dtype=compute_dtype or torch_module.float32)
    selected_target_ids = selected_target_ids.to(device=selected_logits.device)
    return selected_logits, selected_target_ids


def gather_selected_target_logprobs(
    torch_module,
    logits,
    target_ids,
    *,
    shift_targets: bool,
    compute_dtype=None,
):
    """Gather logprobs for target ids from aligned logits."""

    selected_logits, selected_target_ids = select_logprob_tensors(
        torch_module,
        logits,
        target_ids,
        shift_targets=shift_targets,
        compute_dtype=compute_dtype,
    )
    target_logits = selected_logits.gather(-1, selected_target_ids.unsqueeze(-1)).squeeze(-1)
    normalizers = torch_module.logsumexp(selected_logits, dim=-1)
    return target_logits - normalizers


def gather_target_logprobs(torch_module, logits, target_ids, *, compute_dtype=None):
    return gather_selected_target_logprobs(
        torch_module,
        logits,
        target_ids,
        shift_targets=True,
        compute_dtype=compute_dtype,
    )


def gather_aligned_target_logprobs(torch_module, logits, target_ids, *, compute_dtype=None):
    return gather_selected_target_logprobs(
        torch_module,
        logits,
        target_ids,
        shift_targets=False,
        compute_dtype=compute_dtype,
    )


def gather_selected_target_logprobs_chunked_float32(
    torch_module,
    logits,
    target_ids,
    *,
    shift_targets: bool,
    vocab_chunk_size: int,
):
    """Chunk vocab-dimension logprob gathering in float32 for memory safety."""

    selected_logits, selected_target_ids = select_logprob_tensors(
        torch_module,
        logits,
        target_ids,
        shift_targets=shift_targets,
        compute_dtype=None,
    )
    vocab_size = int(selected_logits.shape[-1])
    chunk_size = max(1, int(vocab_chunk_size))

    if vocab_size <= chunk_size:
        return gather_selected_target_logprobs(
            torch_module,
            logits,
            target_ids,
            shift_targets=shift_targets,
            compute_dtype=torch_module.float32,
        )

    batch_size = int(selected_logits.shape[0])
    seq_len = int(selected_logits.shape[1])
    device = selected_logits.device
    target_logits = torch_module.full((batch_size, seq_len), float("-inf"), dtype=torch_module.float32, device=device)
    normalizers = torch_module.full((batch_size, seq_len), float("-inf"), dtype=torch_module.float32, device=device)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk_logits = selected_logits[:, :, start:end].to(dtype=torch_module.float32)
        normalizers = torch_module.logaddexp(normalizers, torch_module.logsumexp(chunk_logits, dim=-1))

        in_chunk = (selected_target_ids >= start) & (selected_target_ids < end)
        if bool(in_chunk.any()):
            local_target_ids = (selected_target_ids - start).clamp(min=0, max=(end - start - 1))
            gathered = chunk_logits.gather(-1, local_target_ids.unsqueeze(-1)).squeeze(-1)
            target_logits = torch_module.where(in_chunk, gathered, target_logits)

    return target_logits - normalizers


def gather_target_logprobs_chunked_float32(
    torch_module,
    logits,
    target_ids,
    *,
    vocab_chunk_size: int,
):
    return gather_selected_target_logprobs_chunked_float32(
        torch_module,
        logits,
        target_ids,
        shift_targets=True,
        vocab_chunk_size=vocab_chunk_size,
    )


def gather_aligned_target_logprobs_chunked_float32(
    torch_module,
    logits,
    target_ids,
    *,
    vocab_chunk_size: int,
):
    return gather_selected_target_logprobs_chunked_float32(
        torch_module,
        logits,
        target_ids,
        shift_targets=False,
        vocab_chunk_size=vocab_chunk_size,
    )


def resolve_reference_scoring_max_tokens_per_batch(config: TrainerConfig) -> int:
    """Return the effective token budget for reference-policy scoring."""

    return int(config.reference_scoring_max_tokens_per_rank)


def resolve_reference_scoring_model_parallel_size(config: TrainerConfig) -> int:
    """Return how many ranks one reference-scoring model replica spans."""

    return max(1, int(config.reference_scoring_model_parallel_size))


def resolve_training_data_parallel_size(
    config: TrainerConfig,
    *,
    world_size: int | None = None,
) -> int:
    """Return training DP width after sequence-parallel grouping."""

    active_world_size = int(world_size if world_size is not None else config.trainer_num_workers)
    return data_parallel_size_for_replica_size(
        active_world_size,
        max(1, int(config.sequence_parallel_size)),
    )


def resolve_reference_scoring_data_parallel_size(
    config: TrainerConfig,
    *,
    world_size: int | None = None,
) -> int:
    """Return reference-scoring DP width after model-parallel grouping."""

    active_world_size = int(world_size if world_size is not None else config.trainer_num_workers)
    return data_parallel_size_for_replica_size(
        active_world_size,
        resolve_reference_scoring_model_parallel_size(config),
    )


class ReferencePolicyLogprobScorer:
    """Score rollout tokens under the frozen reference policy for KL shaping."""

    def __init__(self, model_name_or_path: str, *, logprob_compute_dtype: str, vocab_chunk_size: int) -> None:
        self.model_name_or_path = model_name_or_path
        self.logprob_compute_dtype_name = logprob_compute_dtype
        self.vocab_chunk_size = int(vocab_chunk_size)
        self.torch_module = None
        self.model = None
        self.device = None
        self.logprob_compute_dtype = None

    def ensure_model(self):
        """Load the reference policy lazily and cache it for reuse."""

        if self.model is not None:
            return self.torch_module, self.model, self.device
        import torch
        from transformers import AutoModelForCausalLM

        device = resolve_reference_policy_device(torch)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
        )
        model.to(device)
        model.eval()
        self.torch_module = torch
        self.model = model
        self.device = device
        self.logprob_compute_dtype = resolve_logprob_compute_dtype(
            torch,
            self.logprob_compute_dtype_name,
            device_type=device.type,
        )
        logger.info(
            "reference_policy_precision device=%s model_dtype=%s logprob_compute_dtype=%s token_dtype=%s",
            str(device),
            str(dtype),
            str(self.logprob_compute_dtype),
            str(torch.long),
        )
        return torch, model, device

    def score_loss_samples(self, loss_samples: list[LossSample]) -> list[list[float]]:
        """Score a batch of loss samples with the reference policy."""

        torch, model, device = self.ensure_model()
        base_logprobs_by_sample: list[list[float]] = []
        if not loss_samples:
            return base_logprobs_by_sample
        with torch.no_grad():
            input_ids, attention_mask, lengths = self.score_batch_inputs(loss_samples, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            token_logprobs = gather_target_logprobs_chunked_float32(
                torch,
                outputs.logits,
                input_ids,
                vocab_chunk_size=self.vocab_chunk_size,
            )
            for row, seq_len in enumerate(lengths):
                if seq_len < 2:
                    base_logprobs_by_sample.append([])
                    continue
                sample_logprobs = token_logprobs[row, : seq_len - 1]
                base_logprobs_by_sample.append([float(value) for value in sample_logprobs.detach().cpu().tolist()])
        return base_logprobs_by_sample

    def score_batch_inputs(self, loss_samples: list[LossSample], *, device) -> tuple[Any, Any, list[int]]:
        """Build padded tensors for reference-policy scoring."""

        torch = self.torch_module
        if torch is None:
            raise RuntimeError("Reference scorer torch module is not initialized")
        lengths = [len(sample.full_sequence_ids) for sample in loss_samples]
        max_len = max(1, max(lengths, default=1))
        input_ids = torch.zeros((len(loss_samples), max_len), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(loss_samples), max_len), dtype=torch.long, device=device)
        for row, sample in enumerate(loss_samples):
            sequence_ids = list(sample.full_sequence_ids)
            seq_len = len(sequence_ids)
            if seq_len <= 0:
                continue
            input_ids[row, :seq_len] = torch.tensor(sequence_ids, dtype=torch.long, device=device)
            attention_mask[row, :seq_len] = 1
        return input_ids, attention_mask, lengths

    def close(self) -> None:
        if self.model is None or self.torch_module is None:
            return
        # Do NOT move the model to CPU before deleting — on a RAM-constrained
        # machine, .to("cpu") allocates fresh CPU tensors for every parameter
        # (it does not reuse the mmap'd safetensors pages), which can OOM
        # when multiple ranks do this simultaneously.
        del self.model
        self.model = None
        try:
            self.torch_module.cuda.empty_cache()
        except Exception:
            pass


def balanced_partitions_by_length(lengths: list[int], num_partitions: int) -> list[list[int]]:
    if not lengths:
        return []
    partition_count = max(1, min(int(num_partitions), len(lengths)))
    target_sizes = [len(lengths) // partition_count + (1 if index < (len(lengths) % partition_count) else 0) for index in range(partition_count)]
    partitions: list[list[int]] = [[] for _ in range(partition_count)]
    partition_loads = [0] * partition_count
    ranked_indices = sorted(range(len(lengths)), key=lambda index: lengths[index], reverse=True)
    for item_index in ranked_indices:
        choices = [index for index in range(partition_count) if len(partitions[index]) < target_sizes[index]]
        best_partition = min(choices, key=lambda index: (partition_loads[index], len(partitions[index]), index))
        partitions[best_partition].append(item_index)
        partition_loads[best_partition] += lengths[item_index]
    for partition in partitions:
        partition.sort()
    return partitions


def shard_loss_samples_for_rank(
    samples: list[LossSample],
    *,
    world_size: int,
    rank: int,
    sequence_parallel_size: int = 1,
) -> tuple[list[LossSample], int, int]:
    if world_size <= 1:
        return samples, len(samples), 0

    sp_size = max(1, int(sequence_parallel_size))
    if world_size % sp_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by sequence_parallel_size={sp_size}"
        )

    # Ulysses ranks inside the same SP group must consume the same logical batch;
    # they cooperate by slicing the sequence dimension within model forward.
    dp_group_count = max(1, world_size // sp_size)
    if not samples:
        return [], 0, 0
    sharded_samples = list(samples)
    padding_count = 0
    remainder = len(sharded_samples) % dp_group_count
    if remainder != 0:
        padding_count = dp_group_count - remainder
        template = min(sharded_samples, key=lambda sample: len(sample.model_input_ids))
        sharded_samples.extend(make_padding_loss_sample(template) for _ in range(padding_count))
    partitions = balanced_partitions_by_length([len(sample.model_input_ids) for sample in sharded_samples], dp_group_count)
    dp_group_index = int(rank) // sp_size
    return [sharded_samples[index] for index in partitions[dp_group_index]], len(samples), padding_count


def shard_reference_loss_samples_for_rank(
    samples: list[LossSample],
    *,
    world_size: int,
    rank: int,
    model_parallel_size: int,
) -> tuple[list[int], list[LossSample]]:
    """Return the sample indices this rank should score for reference-policy KL."""

    if not samples:
        return [], []
    if world_size <= 1:
        return list(range(len(samples))), list(samples)

    replica_size = max(1, int(model_parallel_size))
    partition_count = data_parallel_size_for_replica_size(world_size, replica_size)
    partitions = balanced_partitions_by_length(
        [len(sample.full_sequence_ids) for sample in samples],
        partition_count,
    )
    if len(partitions) < partition_count:
        partitions.extend([[] for _ in range(partition_count - len(partitions))])
    partition_index = int(rank) // replica_size
    sample_indices = partitions[partition_index]
    return sample_indices, [samples[index] for index in sample_indices]


def collect_reference_kl_diffs(
    sample_count: int,
    updates_by_rank: list[list[tuple[int, list[float]]]],
) -> list[list[float]]:
    """Merge gathered per-sample KL diffs back into full sample order."""

    diffs_by_sample: list[list[float] | None] = [None] * int(sample_count)
    for rank_updates in updates_by_rank:
        for sample_index, diffs in rank_updates:
            if sample_index < 0 or sample_index >= sample_count:
                raise ValueError(f"Reference KL update index out of bounds: {sample_index}")
            current = diffs_by_sample[sample_index]
            normalized_diffs = [float(value) for value in diffs]
            if current is None:
                diffs_by_sample[sample_index] = normalized_diffs
                continue
            if len(current) != len(normalized_diffs):
                raise ValueError(
                    "Reference KL updates for the same sample disagree on token count: "
                    f"sample_index={sample_index}"
                )
    missing = [index for index, value in enumerate(diffs_by_sample) if value is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise ValueError(f"Reference KL updates are missing samples: {preview}")
    return [value for value in diffs_by_sample if value is not None]


def count_active_tokens(samples: list[LossSample]) -> int:
    """Count masked-in training tokens across the provided samples."""

    return int(sum(sum(sample.mask) for sample in samples))


def count_sequence_tokens(samples: list[LossSample]) -> int:
    """Count raw input tokens across the provided samples."""

    return int(sum(len(sample.model_input_ids) for sample in samples))


def sort_samples_by_length(samples: list[LossSample]) -> list[LossSample]:
    """Return samples ordered from shortest to longest sequence."""

    return sorted(samples, key=lambda sample: len(sample.model_input_ids))


def pack_samples_by_token_budget(samples: list[LossSample], max_tokens_per_batch: int | None) -> list[list[LossSample]]:
    """Greedily pack contiguous samples under a token budget."""

    if not samples:
        return []
    if max_tokens_per_batch is None or max_tokens_per_batch <= 0:
        return [list(samples)]

    batches: list[list[LossSample]] = []
    current_batch: list[LossSample] = []
    current_tokens = 0
    for sample in samples:
        sample_tokens = len(sample.model_input_ids)
        if current_batch and current_tokens + sample_tokens > max_tokens_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(sample)
        current_tokens += sample_tokens
    if current_batch:
        batches.append(current_batch)
    return batches


def truncate_loss_sample(sample: LossSample, max_tokens: int) -> LossSample:
    """Truncate a LossSample to at most max_tokens in model_input_ids (and all aligned fields).

    Samples shorter than max_tokens are returned unchanged. Truncation removes
    tokens from the end of the sequence, preserving the prompt prefix intact.
    """
    if len(sample.model_input_ids) <= max_tokens:
        return sample
    logger.warning(
        "truncate_loss_sample original_tokens=%d max_tokens=%d prompt_tokens=%d",
        len(sample.model_input_ids),
        max_tokens,
        sample.prompt_token_count,
    )
    return LossSample(
        model_input_ids=sample.model_input_ids[:max_tokens],
        target_token_ids=sample.target_token_ids[:max_tokens],
        sampling_logprobs=sample.sampling_logprobs[:max_tokens],
        advantages=sample.advantages[:max_tokens],
        mask=sample.mask[:max_tokens],
        full_sequence_ids=sample.full_sequence_ids,
        prompt_token_count=sample.prompt_token_count,
        metadata=sample.metadata,
    )


def build_training_microbatches(
    samples: list[LossSample],
    *,
    num_substeps: int,
    max_tokens_per_batch: int | None,
) -> list[list[LossSample]]:
    """Split sorted loss samples into microbatches for one optimizer step."""

    if not samples:
        return []
    if max_tokens_per_batch is not None and max_tokens_per_batch > 0:
        samples = [truncate_loss_sample(s, max_tokens_per_batch) for s in samples]
    ordered_samples = sort_samples_by_length(samples)
    substeps = split_contiguous(ordered_samples, max(1, int(num_substeps)))
    microbatches: list[list[LossSample]] = []
    for substep in substeps:
        microbatches.extend(pack_samples_by_token_budget(substep, max_tokens_per_batch))
    return [batch for batch in microbatches if batch]


def summarize_microbatches(microbatches: list[list[LossSample]]) -> dict[str, float]:
    """Summarize microbatch shapes for trainer logging."""

    if not microbatches:
        return {
            "train/microbatch_count": 0.0,
            "train/microbatch_tokens_mean": 0.0,
            "train/microbatch_tokens_max": 0.0,
            "train/microbatch_samples_mean": 0.0,
            "train/microbatch_samples_max": 0.0,
        }
    token_counts = [float(count_sequence_tokens(batch)) for batch in microbatches]
    sample_counts = [float(len(batch)) for batch in microbatches]
    return {
        "train/microbatch_count": float(len(microbatches)),
        "train/microbatch_tokens_mean": float(sum(token_counts) / len(token_counts)),
        "train/microbatch_tokens_max": float(max(token_counts)),
        "train/microbatch_samples_mean": float(sum(sample_counts) / len(sample_counts)),
        "train/microbatch_samples_max": float(max(sample_counts)),
    }


def data_parallel_size_for_replica_size(world_size: int, replica_size: int) -> int:
    """Return DP size when one logical replica spans `replica_size` ranks."""

    normalized_world_size = max(1, int(world_size))
    normalized_replica_size = max(1, int(replica_size))
    if normalized_world_size % normalized_replica_size != 0:
        raise ValueError(
            f"world_size={normalized_world_size} must be divisible by replica_size={normalized_replica_size}"
        )
    return max(1, normalized_world_size // normalized_replica_size)


def topology_summary(config: TrainerConfig) -> dict[str, int | str]:
    """Return a stable summary of the effective training topology."""

    distributed_mode = "ddp" if int(config.trainer_num_workers) > 1 else "none"
    reference_scoring_model_parallel_size = resolve_reference_scoring_model_parallel_size(config)
    return {
        "backend": config.backend_name,
        "trainer_num_workers": int(config.trainer_num_workers),
        "distributed_mode": distributed_mode,
        "distributed_strategy": config.distributed_strategy,
        "sequence_parallel_size": int(config.sequence_parallel_size),
        "data_parallel_size": resolve_training_data_parallel_size(config),
        "reference_scoring_model_parallel_size": reference_scoring_model_parallel_size,
        "reference_scoring_data_parallel_size": resolve_reference_scoring_data_parallel_size(config),
    }


def validate_training_topology(config: TrainerConfig) -> None:
    """Validate that the requested trainer topology is internally consistent."""

    trainer_num_workers = int(config.trainer_num_workers)
    sp_size = max(1, int(config.sequence_parallel_size))
    reference_scoring_model_parallel_size = resolve_reference_scoring_model_parallel_size(config)
    if trainer_num_workers < 1:
        raise ValueError("trainer_num_workers must be >= 1")
    if trainer_num_workers % reference_scoring_model_parallel_size != 0:
        raise ValueError(
            "trainer_num_workers must be divisible by reference_scoring_model_parallel_size"
        )
    backend_name = normalized_backend_name(config.backend_name)
    if backend_name == "dry-run":
        return
    if backend_name != "deepspeed":
        raise ValueError(f"Unsupported trainer backend: {config.backend_name!r}")
    strategy = (config.distributed_strategy or "ddp").strip().lower()
    if trainer_num_workers > 1 and strategy != "ddp":
        raise ValueError(
            "deepspeed currently requires distributed_strategy=ddp for multi-worker runs."
        )
    if sp_size > 1 and trainer_num_workers <= 1:
        raise ValueError(
            "sequence_parallel_size is configured but inactive because trainer_num_workers=1"
        )
    if trainer_num_workers > 1 and trainer_num_workers % sp_size != 0:
        raise ValueError("trainer_num_workers must be divisible by sequence_parallel_size")
    if sp_size > 1 and not config.use_remove_padding:
        raise ValueError("deepspeed sequence parallel currently requires use_remove_padding=1")


def build_deepspeed_engine_config(torch_module, *, distributed_active: bool, model_dtype) -> dict[str, Any]:
    """Build the minimal DeepSpeed engine config for the training backend."""

    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1, # THIS IS PRETTY MUCH IGNORED, BECAUSE OF THE set_gradient_accumulation_boundary() CALLS WE DO LATER. THIS BEHAVIOR IS INTENTIONAL.
        "steps_per_print": 10**9,
        "wall_clock_breakdown": False,
        "zero_optimization": {"stage": 2 if distributed_active else 0},
    }
    if model_dtype == getattr(torch_module, "bfloat16", None):
        config["bf16"] = {"enabled": True}
    elif model_dtype == getattr(torch_module, "float16", None):
        config["fp16"] = {"enabled": True}
    return config


def deepspeed_packed_attention_requested(config: TrainerConfig) -> bool:
    """Return whether the DeepSpeed path needs packed flash attention."""

    return normalized_backend_name(config.backend_name) == "deepspeed" and bool(config.use_remove_padding)


def ensure_flash_attention_2_ready_for_deepspeed() -> None:
    """Ensure Transformers can serve a flash-attention backend for DeepSpeed packing."""
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    try:
        lazy_import_flash_attention("flash_attention_2")
        return
    except Exception:
        pass

    try:
        from transformers.integrations.hub_kernels import load_and_register_kernel

        kernel_repo = "kernels-community/flash-attn2"
        load_and_register_kernel(kernel_repo)
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = ALL_ATTENTION_FUNCTIONS[kernel_repo]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "DeepSpeed packed/remove-padding training requires a working flash attention backend. "
            "Install `kernels` in the runtime environment or provide a compatible `flash_attention_2` backend."
        ) from exc


def apply_reference_policy_kl(
    loss_samples: list[LossSample],
    *,
    model_name_or_path: str,
    kl_penalty_coef: float,
    max_tokens_per_batch: int | None = None,
    logprob_compute_dtype: str,
    vocab_chunk_size: int,
) -> dict[str, float]:
    if not loss_samples or kl_penalty_coef <= 0:
        return {}
    diffs_by_sample, total_diff, total_mask = score_reference_policy_kl(
        loss_samples,
        model_name_or_path=model_name_or_path,
        max_tokens_per_batch=max_tokens_per_batch,
        logprob_compute_dtype=logprob_compute_dtype,
        vocab_chunk_size=vocab_chunk_size,
    )
    average_diff = total_diff / max(total_mask, 1e-8)
    return apply_kl_adjustment(loss_samples, diffs_by_sample, average_diff=average_diff, kl_penalty_coef=float(kl_penalty_coef))


def compute_kl_diffs(loss_samples: list[LossSample], base_logprobs_by_sample: list[list[float]]) -> tuple[list[list[float]], float, float]:
    if len(loss_samples) != len(base_logprobs_by_sample):
        raise ValueError("base logprobs must align with loss samples")
    total_mask = 0.0
    total_diff = 0.0
    diffs_by_sample: list[list[float]] = []
    for sample, base_logprobs in zip(loss_samples, base_logprobs_by_sample, strict=True):
        if len(base_logprobs) != len(sample.target_token_ids):
            raise ValueError("base logprobs must align to target tokens")
        diffs: list[float] = []
        for old_lp, base_lp, mask_value in zip(sample.sampling_logprobs, base_logprobs, sample.mask, strict=True):
            diff = (old_lp - base_lp) * mask_value
            diffs.append(diff)
            total_diff += diff
            total_mask += mask_value
        diffs_by_sample.append(diffs)
    return diffs_by_sample, total_diff, total_mask


def apply_kl_adjustment(
    loss_samples: list[LossSample],
    diffs_by_sample: list[list[float]],
    *,
    average_diff: float,
    kl_penalty_coef: float,
) -> dict[str, float]:
    for sample, diffs in zip(loss_samples, diffs_by_sample, strict=True):
        sample.advantages = [
            advantage + (kl_penalty_coef * mask_value * (average_diff - diff))
            for advantage, mask_value, diff in zip(sample.advantages, sample.mask, diffs, strict=True)
        ]
    return {"kl_policy_base": float(average_diff)}


def score_reference_policy_kl(
    loss_samples: list[LossSample],
    *,
    model_name_or_path: str,
    max_tokens_per_batch: int | None = None,
    logprob_compute_dtype: str,
    vocab_chunk_size: int,
) -> tuple[list[list[float]], float, float]:
    scorer = ReferencePolicyLogprobScorer(
        model_name_or_path,
        logprob_compute_dtype=logprob_compute_dtype,
        vocab_chunk_size=vocab_chunk_size,
    )
    try:
        base_logprobs = score_reference_policy_logprobs(
            scorer,
            loss_samples,
            max_tokens_per_batch=max_tokens_per_batch,
        )
        return compute_kl_diffs(loss_samples, base_logprobs)
    finally:
        scorer.close()


def score_reference_policy_logprobs(
    scorer: ReferenceLogprobScorer,
    loss_samples: list[LossSample],
    *,
    max_tokens_per_batch: int | None = None,
) -> list[list[float]]:
    if not loss_samples:
        return []
    if max_tokens_per_batch is None or max_tokens_per_batch <= 0:
        logger.info(
            "reference_logprob_progress rank=%d mode=single_batch samples=%d total_tokens=%d",
            rank_from_env(),
            len(loss_samples),
            count_sequence_tokens(loss_samples),
        )
        return scorer.score_loss_samples(loss_samples)

    indexed_samples = list(enumerate(loss_samples))
    indexed_samples.sort(key=lambda item: len(item[1].model_input_ids))
    ordered_logprobs: list[list[float]] = [[] for _ in loss_samples]
    current_batch: list[tuple[int, LossSample]] = []
    current_tokens = 0
    flushed_batches = 0
    total_batches = len(pack_samples_by_token_budget([sample for _, sample in indexed_samples], max_tokens_per_batch))

    logger.info(
        "reference_logprob_progress rank=%d mode=chunked samples=%d total_tokens=%d max_tokens_per_batch=%d estimated_batches=%d",
        rank_from_env(),
        len(loss_samples),
        count_sequence_tokens(loss_samples),
        int(max_tokens_per_batch),
        int(total_batches),
    )

    def is_cuda_oom(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "out of memory" in message and "cuda" in message

    def clear_scorer_cuda_cache() -> None:
        torch_module = getattr(scorer, "torch_module", None)
        if torch_module is None:
            return
        try:
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        except Exception:
            pass

    def flush_batch() -> None:
        nonlocal current_batch, current_tokens
        nonlocal flushed_batches
        if not current_batch:
            return
        pending: list[list[tuple[int, LossSample]]] = [current_batch]
        while pending:
            batch_items = pending.pop(0)
            try:
                batch_scores = scorer.score_loss_samples([sample for _, sample in batch_items])
            except RuntimeError as exc:
                if not is_cuda_oom(exc) or len(batch_items) <= 1:
                    raise
                clear_scorer_cuda_cache()
                mid = max(1, len(batch_items) // 2)
                logger.warning(
                    "reference_policy_oom_split batch_size=%d left=%d right=%d",
                    len(batch_items),
                    mid,
                    len(batch_items) - mid,
                )
                pending = [batch_items[:mid], batch_items[mid:]] + pending
                continue
            for (sample_index, _), sample_scores in zip(batch_items, batch_scores, strict=True):
                ordered_logprobs[sample_index] = sample_scores
        flushed_batches += 1
        logger.info(
            "reference_logprob_progress rank=%d flushed_batch=%d/%d batch_samples=%d batch_tokens=%d",
            rank_from_env(),
            flushed_batches,
            total_batches,
            len(current_batch),
            count_sequence_tokens([sample for _, sample in current_batch]),
        )
        current_batch = []
        current_tokens = 0

    for item in indexed_samples:
        sample_tokens = len(item[1].model_input_ids)
        if current_batch and current_tokens + sample_tokens > max_tokens_per_batch:
            flush_batch()
        current_batch.append(item)
        current_tokens += sample_tokens
    flush_batch()
    return ordered_logprobs


class DryRunBackend:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.adapter_path_value: str | None = None

    def train_step(self, loss_samples: list[LossSample], *, output_dir: Path) -> dict[str, float]:
        metrics = apply_reference_policy_kl(
            loss_samples,
            model_name_or_path=self.config.model_name_or_path,
            kl_penalty_coef=self.config.kl_penalty_coef,
            max_tokens_per_batch=resolve_reference_scoring_max_tokens_per_batch(self.config),
            logprob_compute_dtype=self.config.logprob_compute_dtype,
            vocab_chunk_size=self.config.reference_logprob_vocab_chunk_size,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter.json").write_text(
            json.dumps({"num_samples": len(loss_samples)}, indent=2),
            encoding="utf-8",
        )
        self.adapter_path_value = str(output_dir)
        trained_tokens = float(sum(sum(sample.mask) for sample in loss_samples))
        metrics.update({"train_num_samples": float(len(loss_samples)), "train_masked_tokens": trained_tokens})
        return metrics

    def adapter_path(self) -> str | None:
        return self.adapter_path_value


def softmax_kl_against_uniform(rewards: list[float], beta: float) -> float:
    if len(rewards) < 2:
        return 0.0
    reward_max = max(rewards)
    exp_values = [math.exp(beta * (reward - reward_max)) for reward in rewards]
    total = sum(exp_values)
    if total <= 0:
        return 0.0
    probs = [value / total for value in exp_values]
    log_k = math.log(len(rewards))
    return sum(prob * (math.log(max(prob, 1e-12)) + log_k) for prob in probs)


def solve_adaptive_beta(rewards: list[float], *, delta: float = math.log(2.0), beta_max: float = 1e6) -> float:
    if len(rewards) < 2:
        return 0.0
    lo = 0.0
    hi = 1.0
    while hi < beta_max and softmax_kl_against_uniform(rewards, hi) < delta:
        hi *= 2.0
    if softmax_kl_against_uniform(rewards, hi) < delta:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if softmax_kl_against_uniform(rewards, mid) < delta:
            lo = mid
        else:
            hi = mid
    return hi


def entropic_advantages(rewards: list[float], beta: float) -> list[float]:
    if not rewards:
        return []
    reward_max = max(rewards)
    exp_values = [math.exp(beta * (reward - reward_max)) for reward in rewards]
    if len(exp_values) == 1:
        return [0.0]
    total = sum(exp_values)
    out: list[float] = []
    for current in exp_values:
        leave_one_out = (total - current) / max(len(exp_values) - 1, 1)
        out.append((current / max(leave_one_out, 1e-12)) - 1.0)
    return out


def compute_advantages(reward_groups: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for rewards in reward_groups:
        out.append(entropic_advantages(rewards, solve_adaptive_beta(rewards)))
    return out


def build_loss_sample(rollout: EvaluatedRollout, rollout_advantage: float) -> LossSample | None:
    """Build one per-token training sample from an evaluated rollout.

    Args:
        rollout: Evaluated rollout carrying prompt/completion token data.
        rollout_advantage: Scalar advantage assigned to the rollout.

    Returns:
        A loss sample aligned to next-token prediction, or ``None`` when the
        rollout has no trainable completion tokens.

    Raises:
        ValueError: If completion-side tensors disagree on token count.
    """
    prompt_len = len(rollout.prompt_token_ids)
    completion_len = len(rollout.completion_token_ids)
    if completion_len <= 0:
        return None
    if len(rollout.completion_logprobs) != completion_len:
        raise ValueError("completion_logprobs must align with completion_token_ids")
    if len(rollout.completion_mask) != completion_len:
        raise ValueError("completion_mask must align with completion_token_ids")
    full_sequence_ids = list(rollout.prompt_token_ids) + list(rollout.completion_token_ids)
    if len(full_sequence_ids) < 2:
        return None
    full_logprobs = [0.0] * prompt_len + list(rollout.completion_logprobs)
    full_advantages = [0.0] * prompt_len + ([float(rollout_advantage)] * completion_len)
    full_mask = [0.0] * prompt_len + list(rollout.completion_mask)
    return LossSample(
        model_input_ids=full_sequence_ids[:-1],
        target_token_ids=full_sequence_ids[1:],
        sampling_logprobs=full_logprobs[1:],
        advantages=full_advantages[1:],
        mask=full_mask[1:],
        full_sequence_ids=full_sequence_ids,
        prompt_token_count=prompt_len,
        metadata={
            "reward": float(rollout.reward),
            "correctness": float(rollout.correctness),
            "raw_score": float(rollout.raw_score) if rollout.raw_score is not None else None,
            "archive_value": float(rollout.archive_value) if rollout.archive_value is not None else None,
            "seed_state_id": rollout.seed_state.id,
        },
    )


def build_loss_samples(
    rollout_groups: list[list[EvaluatedRollout]],
    advantage_groups: list[list[float]],
) -> tuple[list[LossSample], int]:
    out: list[LossSample] = []
    dropped = 0
    for group, advantages in zip(rollout_groups, advantage_groups, strict=True):
        for rollout, advantage in zip(group, advantages, strict=True):
            sample = build_loss_sample(rollout, float(advantage))
            if sample is not None:
                out.append(sample)
            else:
                dropped += 1
    return out, dropped


def incorporate_kl_penalty(loss_samples: list[LossSample], base_logprobs_by_sample: list[list[float]], kl_penalty_coef: float) -> dict[str, float]:
    if not loss_samples or kl_penalty_coef <= 0:
        return {}
    diffs_by_sample, total_diff, total_mask = compute_kl_diffs(loss_samples, base_logprobs_by_sample)
    average_diff = total_diff / max(total_mask, 1e-8)
    return apply_kl_adjustment(loss_samples, diffs_by_sample, average_diff=average_diff, kl_penalty_coef=kl_penalty_coef)


def drop_constant_reward_groups(rollout_groups: list[list[EvaluatedRollout]]) -> tuple[list[list[EvaluatedRollout]], int]:
    if not rollout_groups:
        return [], 0
    filtered = []
    dropped = 0
    for group in rollout_groups:
        rewards = [rollout.reward for rollout in group]
        if rewards and all(reward == rewards[0] for reward in rewards):
            dropped += 1
            continue
        filtered.append(group)
    if not filtered and rollout_groups:
        return [rollout_groups[0]], max(0, dropped - 1)
    return filtered, dropped


def split_contiguous(items: list[LossSample], num_chunks: int) -> list[list[LossSample]]:
    if not items:
        return []
    n = max(1, min(int(num_chunks), len(items)))
    base = len(items) // n
    rem = len(items) % n
    out: list[list[LossSample]] = []
    start = 0
    for index in range(n):
        size = base + (1 if index < rem else 0)
        out.append(items[start : start + size])
        start += size
    return out


def percentile(lengths: list[int], q: float) -> int:
    if not lengths:
        return 0
    arr = sorted(lengths)
    idx = int((len(arr) - 1) * q)
    return arr[idx]


def rank_from_env() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank_from_env() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size_from_env() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def distributed_enabled() -> bool:
    return world_size_from_env() > 1


def configure_stage_process_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s [rank=%(rank)s pid=%(process)d]: %(message)s")
    )
    handler.addFilter(RankContextFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


class StageRunnerBase:
    """Shared utilities for stage-local training runners."""

    def load_tokenizer_and_base_model(self, cfg: TrainerConfig, auto_tokenizer, auto_model_for_causal_lm) -> None:
        """Load the tokenizer and base causal LM for the stage runner."""

        self.tokenizer = auto_tokenizer.from_pretrained(
            cfg.tokenizer_name_or_path or cfg.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        self.pad_token_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        model_load_kwargs: dict[str, object] = {}
        if self.torch.cuda.is_available() and not self.distributed_active:
            model_load_kwargs["device_map"] = "auto"
        self.model = auto_model_for_causal_lm.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
            dtype=self.dtype,
            **model_load_kwargs,
        )

    def attach_lora_adapter(self, cfg: TrainerConfig, peft_model, lora_config_cls, task_type, get_peft_model) -> None:
        """Attach a fresh or resumed LoRA adapter to the loaded model."""

        resume_adapter = (cfg.resume_adapter_path or "").strip()
        if resume_adapter:
            resume_path = Path(resume_adapter)
            if not resume_path.exists():
                raise FileNotFoundError(f"resume_adapter_path does not exist: {resume_path}")
            weights_path, tensor_count, size_bytes = validate_adapter_checkpoint(resume_path)
            logger.info(
                "lora_resume_load path=%s weights=%s tensors=%d size_bytes=%d",
                str(resume_path),
                str(weights_path),
                tensor_count,
                size_bytes,
            )
            self.model = peft_model.from_pretrained(self.model, str(resume_path), is_trainable=True)
            return

        logger.info("lora_resume_load path=none; initializing_new_lora=1")
        lora_cfg = lora_config_cls(
            task_type=task_type.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def finalize_model_state(self) -> None:
        """Finalize train/eval flags and device placement after wrapping."""

        def disable_use_cache(model_like) -> None:
            config = getattr(model_like, "config", None)
            if config is None:
                return
            if isinstance(config, dict):
                config["use_cache"] = False
                return
            if hasattr(config, "use_cache"):
                config.use_cache = False

        disable_use_cache(self.model)
        if hasattr(self.model, "module"):
            disable_use_cache(self.model.module)
        self.model.train()
        # Keep training inputs on this rank's compute device unless unavailable.
        if self.device is not None and getattr(self.device, "type", None) == "cuda":
            self.input_device = self.device
        else:
            self.input_device = next(self.model.parameters()).device

    def build_optimizer(self, cfg: TrainerConfig) -> None:
        """Build the parity-preserving optimizer for trainable parameters."""

        self.trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        # Parity: original TTT-Discover uses tinker.AdamParams (standard Adam)
        # with no weight decay for all tasks.  AdamW with weight_decay=0.0 is
        # mathematically identical to Adam.  Do not set weight_decay > 0
        # without understanding that the original never used it.
        #
        # DP > 1 correction: with mpu passed to deepspeed.initialize(),
        # ZeRO-2's reduce-scatter divides gradients by W/SP = DP (not W).
        # prepare_loss_for_backward amplifies gradients by SP (all_gather+sum),
        # reconstructing the full-batch gradient.  After ZeRO division by DP,
        # the optimizer sees full_grad/DP.  The original TTT-Discover (Tinker)
        # sees full_grad with no averaging.  Scale LR by DP to restore parity.
        # At DP=1 (current configs) this is a no-op.
        data_parallel_size = max(1, self.world_size // max(1, self.sequence_parallel_size))
        effective_lr = cfg.learning_rate * data_parallel_size
        if self.rank == 0 and data_parallel_size > 1:
            logger.info("optimizer_lr_dp_scaled nominal_lr=%s dp=%d effective_lr=%s", cfg.learning_rate, data_parallel_size, effective_lr)
        self.optimizer = self.torch.optim.AdamW(
            self.trainable_params,
            lr=effective_lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )
        if self.rank == 0:
            logger.info(
                "trainer_precision device=%s distributed_mode=%s model_dtype=%s logprob_compute_dtype=%s aux_tensor_dtype=%s reduction_dtype=%s token_dtype=%s tf32_matmul=%d tf32_cudnn=%d",
                str(self.device),
                self.distributed_mode,
                str(self.dtype),
                str(self.logprob_compute_dtype),
                str(self.torch.float32),
                str(self.torch.float32),
                str(self.torch.long),
                int(bool(self.torch.cuda.is_available() and self.torch.backends.cuda.matmul.allow_tf32)),
                int(bool(self.torch.cuda.is_available() and self.torch.backends.cudnn.allow_tf32)),
            )



    def split_for_rank(self, samples: list[LossSample]) -> tuple[list[LossSample], int, int]:
        return shard_loss_samples_for_rank(
            samples,
            world_size=self.world_size if self.distributed_active else 1,
            rank=self.rank,
            sequence_parallel_size=self.sequence_parallel_size,
        )

    def reduce_scalar_mean(self, value: float) -> float:
        if not self.distributed_active:
            return value
        tensor = self.torch.tensor([value], dtype=self.torch.float64, device=self.input_device)
        self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM)
        return float((tensor / float(self.world_size)).item())

    def reduce_scalar_sum(self, value: float) -> float:
        if not self.distributed_active:
            return value
        tensor = self.torch.tensor([value], dtype=self.torch.float64, device=self.input_device)
        self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM)
        return float(tensor.item())

    def reduce_scalar_max(self, value: float) -> float:
        if not self.distributed_active:
            return value
        tensor = self.torch.tensor([value], dtype=self.torch.float64, device=self.input_device)
        self.dist.all_reduce(tensor, op=self.dist.ReduceOp.MAX)
        return float(tensor.item())

    def uses_packed_sequence_batch(self) -> bool:
        return deepspeed_packed_attention_requested(self.cfg)

    def sequence_parallel_group_rank(self) -> int:
        return int(getattr(self, "sequence_parallel_rank", 0))

    def slice_for_sequence_parallel(self, tensor):
        if self.sequence_parallel_size <= 1 or not self.distributed_active:
            return tensor
        pad_and_slice_inputs = getattr(self, "ulysses_pad_and_slice_inputs", None)
        if pad_and_slice_inputs is not None:
            sliced, _, _ = pad_and_slice_inputs(tensor, None, self.sequence_parallel_size)
            return sliced
        if getattr(tensor, "ndim", 0) < 2:
            return tensor
        seq_len = int(tensor.shape[1])
        if seq_len % self.sequence_parallel_size != 0:
            raise ValueError(
                f"Packed tensor sequence length {seq_len} must be divisible by sequence_parallel_size={self.sequence_parallel_size}"
            )
        chunk_len = seq_len // self.sequence_parallel_size
        rank = self.sequence_parallel_group_rank()
        start = rank * chunk_len
        stop = start + chunk_len
        return tensor[:, start:stop].contiguous()

    def batch_tensors(self, samples: list[LossSample]) -> PaddedBatch:
        batch_size = len(samples)
        max_len = max(len(sample.model_input_ids) for sample in samples)
        input_ids = self.torch.full((batch_size, max_len), self.pad_token_id, dtype=self.torch.long, device=self.input_device)
        target_ids = self.torch.full((batch_size, max_len), self.pad_token_id, dtype=self.torch.long, device=self.input_device)
        old_logprobs = self.torch.zeros((batch_size, max_len), dtype=self.torch.float32, device=self.input_device)
        advantages = self.torch.zeros((batch_size, max_len), dtype=self.torch.float32, device=self.input_device)
        mask = self.torch.zeros((batch_size, max_len), dtype=self.torch.float32, device=self.input_device)
        attention_mask = self.torch.zeros((batch_size, max_len), dtype=self.torch.long, device=self.input_device)
        for row, sample in enumerate(samples):
            seq_len = len(sample.model_input_ids)
            input_ids[row, :seq_len] = self.torch.as_tensor(sample.model_input_ids, dtype=self.torch.long, device=self.input_device)
            target_ids[row, :seq_len] = self.torch.as_tensor(sample.target_token_ids, dtype=self.torch.long, device=self.input_device)
            old_logprobs[row, :seq_len] = self.torch.as_tensor(sample.sampling_logprobs, dtype=self.torch.float32, device=self.input_device)
            advantages[row, :seq_len] = self.torch.as_tensor(sample.advantages, dtype=self.torch.float32, device=self.input_device)
            mask[row, :seq_len] = self.torch.as_tensor(sample.mask, dtype=self.torch.float32, device=self.input_device)
            attention_mask[row, :seq_len] = 1
        return PaddedBatch(
            input_ids=input_ids,
            target_ids=target_ids,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            attention_mask=attention_mask,
        )

    def pack_batch_tensors(self, samples: list[LossSample]) -> PackedBatch:
        input_ids_values: list[int] = []
        target_ids_values: list[int] = []
        old_logprobs_values: list[float] = []
        advantages_values: list[float] = []
        mask_values: list[float] = []
        position_ids_values: list[int] = []

        for sample in samples:
            input_ids_values.extend(sample.model_input_ids)
            target_ids_values.extend(sample.target_token_ids)
            old_logprobs_values.extend(sample.sampling_logprobs)
            advantages_values.extend(sample.advantages)
            mask_values.extend(sample.mask)
            position_ids_values.extend(range(len(sample.model_input_ids)))

        if self.sequence_parallel_size > 1 and self.distributed_active:
            for _ in range((-len(input_ids_values)) % self.sequence_parallel_size):
                input_ids_values.append(self.pad_token_id)
                target_ids_values.append(self.pad_token_id)
                old_logprobs_values.append(0.0)
                advantages_values.append(0.0)
                mask_values.append(0.0)
                position_ids_values.append(0)

        return PackedBatch(
            input_ids=self.torch.as_tensor([input_ids_values], dtype=self.torch.long, device=self.input_device),
            target_ids=self.torch.as_tensor([target_ids_values], dtype=self.torch.long, device=self.input_device),
            old_logprobs=self.torch.as_tensor([old_logprobs_values], dtype=self.torch.float32, device=self.input_device),
            advantages=self.torch.as_tensor([advantages_values], dtype=self.torch.float32, device=self.input_device),
            mask=self.torch.as_tensor([mask_values], dtype=self.torch.float32, device=self.input_device),
            position_ids=self.torch.as_tensor([position_ids_values], dtype=self.torch.long, device=self.input_device),
        )

    def build_loss_batch(self, samples: list[LossSample]) -> PaddedBatch | PackedBatch:
        """Build the dense or packed training batch for one microbatch."""

        if self.uses_packed_sequence_batch():
            return self.pack_batch_tensors(samples)
        return self.batch_tensors(samples)

    def forward_loss_batch(self, batch: PaddedBatch | PackedBatch) -> tuple[Any, Any, Any, Any, Any]:
        """Run the model on one prepared batch and return aligned loss tensors."""

        sliced_input_ids = self.slice_for_sequence_parallel(batch.input_ids)
        sliced_target_ids = self.slice_for_sequence_parallel(batch.target_ids)
        sliced_old_logprobs = self.slice_for_sequence_parallel(batch.old_logprobs)
        sliced_advantages = self.slice_for_sequence_parallel(batch.advantages)
        sliced_mask = self.slice_for_sequence_parallel(batch.mask)
        model_kwargs: dict[str, Any] = {
            "input_ids": sliced_input_ids,
            "use_cache": False,
        }
        if isinstance(batch, PackedBatch):
            model_kwargs["position_ids"] = self.slice_for_sequence_parallel(batch.position_ids)
        else:
            model_kwargs["attention_mask"] = self.slice_for_sequence_parallel(batch.attention_mask)
        outputs = self.model(**model_kwargs)
        return outputs, sliced_target_ids, sliced_old_logprobs, sliced_advantages, sliced_mask

    def gather_batch_logprobs(self, logits, target_ids):
        """Gather aligned target-token logprobs for one forward pass."""

        if self.logprob_compute_dtype == self.torch.float32:
            return gather_aligned_target_logprobs_chunked_float32(
                self.torch,
                logits,
                target_ids,
                vocab_chunk_size=self.cfg.reference_logprob_vocab_chunk_size,
            )
        return gather_aligned_target_logprobs(
            self.torch,
            logits,
            target_ids,
            compute_dtype=self.logprob_compute_dtype,
        )

    def compute_loss(self, samples: list[LossSample]) -> tuple[Any, int]:
        if not samples:
            return self.torch.zeros((), device=self.input_device), 0
        outputs, sliced_target_ids, sliced_old_logprobs, sliced_advantages, sliced_mask = self.forward_loss_batch(
            self.build_loss_batch(samples)
        )
        token_logprobs = self.gather_batch_logprobs(outputs.logits, sliced_target_ids)
        ratio = self.torch.exp(token_logprobs - sliced_old_logprobs)
        objective = ratio * sliced_advantages
        loss = -(objective * sliced_mask).sum()
        return loss, int(sliced_mask.sum().item())

    def prepare_loss_for_backward(self, loss):
        return loss

    def zero_grad_for_step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def backward_loss(self, loss, is_last: bool = False) -> None:
        loss.backward()

    def run_optimizer_step(self) -> tuple[float, int]:
        optimizer_started_at = time.perf_counter()
        self.optimizer.step()
        return time.perf_counter() - optimizer_started_at, 1

    def train_step(self, samples: list[LossSample]) -> dict[str, float]:
        if not samples:
            return {"train/loss": 0.0, "train/tokens_per_sec": 0.0}
        local_samples, global_sample_count, global_padding_count = self.split_for_rank(samples)
        if not local_samples:
            return {"train/loss": 0.0, "train/tokens_per_sec": 0.0}

        lengths = [len(sample.model_input_ids) for sample in samples]
        local_lengths = [len(sample.model_input_ids) for sample in local_samples]
        total_tokens = int(sum(lengths))
        total_masked_tokens = count_active_tokens(samples)
        local_total_tokens = int(sum(local_lengths))
        local_masked_tokens = count_active_tokens(local_samples)
        local_padding_count = count_padding_samples(local_samples)
        if self.rank == 0:
            logger.info(
                "trainer_progress start global_samples=%d global_padding_samples=%d total_tokens=%d total_masked_tokens=%d local_samples=%d local_padding_samples=%d local_tokens=%d local_masked_tokens=%d p50=%d p90=%d p99=%d microbatch_budget_per_rank=%s sp=%d",
                global_sample_count,
                global_padding_count,
                total_tokens,
                total_masked_tokens,
                len(local_samples),
                local_padding_count,
                local_total_tokens,
                local_masked_tokens,
                percentile(lengths, 0.50),
                percentile(lengths, 0.90),
                percentile(lengths, 0.99),
                self.cfg.trainer_max_tokens_per_rank if self.cfg.trainer_max_tokens_per_rank is not None else "none",
                self.sequence_parallel_size,
            )

        step_started_at = time.perf_counter()
        # trainer_max_tokens_per_rank is the per-GPU budget *after* SP
        # splitting.  Multiply by SP size to get the pre-split packed
        # sequence budget so that each rank actually processes up to
        # trainer_max_tokens_per_rank tokens.
        if self.cfg.trainer_max_tokens_per_rank is not None:
            effective_max_tokens = self.cfg.trainer_max_tokens_per_rank * self.sequence_parallel_size
        else:
            effective_max_tokens = None
        microbatches = build_training_microbatches(
            local_samples,
            num_substeps=self.cfg.num_substeps,
            max_tokens_per_batch=effective_max_tokens,
        )
        microbatch_summary = summarize_microbatches(microbatches)
        if self.rank == 0:
            logger.info(
                "trainer_progress microbatches=%d mean_tokens=%.1f max_tokens=%d mean_samples=%.1f max_samples=%d",
                int(microbatch_summary["train/microbatch_count"]),
                float(microbatch_summary["train/microbatch_tokens_mean"]),
                int(microbatch_summary["train/microbatch_tokens_max"]),
                float(microbatch_summary["train/microbatch_samples_mean"]),
                int(microbatch_summary["train/microbatch_samples_max"]),
            )
        local_loss_sum = 0.0
        local_masked_token_count = 0
        sample_count = 0
        optimizer_time_s = 0.0
        optimizer_steps = 0
        progress_interval = max(1, len(microbatches) // 10) if microbatches else 1
        completed_microbatches = 0

        # Parity: if you set num_substeps > 1 in the original TTT-Discover, 
        # it will cut the RL dataset up into num_substeps chunks and call Tinker's
        # forward_backward + optim_step once per substep chunk.
        # Here, NanoDiscover intentionally departs from this behavior and 
        # accumulates gradients across all microbatches, doing only ONE 
        # optimizer step across all num_substeps chunks.
        # In most cases, this won't matter, because TTT-Discover always set
        # num_substeps to 1. It will matter if num_substeps > 1 but we argue that
        # doing a full batch gradient in that case is probably closer to the 
        # original semantics.
        self.zero_grad_for_step()
        for mb_idx, microbatch in enumerate(microbatches):
            is_last = (mb_idx == len(microbatches) - 1)
            current = list(microbatch)
            current_loss_sum, current_token_count = self.compute_loss(current)
            # Parity: Tinker's importance_sampling loss uses a raw sum
            # over tokens with no normalization.  Microbatch gradient
            # accumulation is still correct because
            # grad(sum_of_parts) == sum(grad(parts)).
            self.backward_loss(self.prepare_loss_for_backward(current_loss_sum), is_last=is_last)
            local_loss_sum += float(current_loss_sum.detach().item())
            local_masked_token_count += current_token_count
            sample_count += len(current)
            completed_microbatches += 1
            should_report_progress = (
                completed_microbatches == len(microbatches)
                or completed_microbatches % progress_interval == 0
            )
            if should_report_progress:
                global_masked_tokens_so_far = int(self.reduce_scalar_sum(float(local_masked_token_count)))
                global_loss_sum_so_far = self.reduce_scalar_sum(local_loss_sum)
                global_mean_loss_so_far = global_loss_sum_so_far / max(1.0, float(global_masked_tokens_so_far))
            if self.rank == 0 and should_report_progress:
                logger.info(
                    "trainer_progress completed_microbatches=%d total_microbatches=%d pct=%.1f masked_tokens=%d/%d rank0_samples=%d mean_loss=%.6f elapsed_s=%.2f",
                    completed_microbatches,
                    len(microbatches),
                    100.0 * completed_microbatches / max(1, len(microbatches)),
                    global_masked_tokens_so_far,
                    total_masked_tokens,
                    sample_count,
                    global_mean_loss_so_far,
                    time.perf_counter() - step_started_at,
                )
        optimizer_time_s, optimizer_steps = self.run_optimizer_step()

        step_time_s = max(1e-8, time.perf_counter() - step_started_at)
        global_loss_sum = self.reduce_scalar_sum(local_loss_sum)
        global_token_count = int(self.reduce_scalar_sum(float(local_masked_token_count)))
        global_mean_loss = float(global_loss_sum / max(1.0, float(global_token_count)))
        max_dt = self.reduce_scalar_max(step_time_s)
        tok_per_sec = float(global_token_count / max(1e-8, max_dt))
        metrics = {
            "train/loss": global_mean_loss,
            "train/tokens_per_sec": tok_per_sec,
            "train/sequence_parallel_size": float(self.sequence_parallel_size),
            "train/max_tokens_per_rank": float(self.cfg.trainer_max_tokens_per_rank or 0),
            "train/local_sample_count": float(len(local_samples)),
            "train/local_padding_sample_count": float(local_padding_count),
            "train/global_padding_sample_count": float(global_padding_count),
            "train/local_sequence_tokens": float(local_total_tokens),
            "train/optimizer_time_s": float(optimizer_time_s),
            "train/optimizer_steps": float(optimizer_steps),
            "train/step_time_s": float(step_time_s),
            "train/use_remove_padding": float(bool(self.cfg.use_remove_padding)),
        }
        metrics.update(microbatch_summary)
        return metrics

    def save_adapter(self, save_dir: str) -> str:
        out_dir = Path(save_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.distributed_active and self.rank != 0:
            return str(out_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        peft_state = self.get_peft_model_state_dict(model_to_save)
        if not peft_state:
            raise RuntimeError("Failed to collect any LoRA adapter tensors; refusing to save an empty adapter checkpoint.")
        save_prepared_peft_adapter(model_to_save, out_dir, peft_state)
        return str(out_dir)


class DeepSpeedStageRunner(StageRunnerBase):
    """DeepSpeed-backed stage runner used inside the subprocess training stage."""

    def __init__(self, cfg: TrainerConfig):
        import deepspeed
        import torch
        import torch.distributed as dist
        import torch.distributed.nn.functional as dist_nn
        import torch.nn.functional as F
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from peft.utils.save_and_load import get_peft_model_state_dict
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.dist = dist
        self.dist_nn = dist_nn
        self.F = F
        self.deepspeed = deepspeed
        self.get_peft_model_state_dict = get_peft_model_state_dict
        self.sequence_parallel_size = max(1, int(cfg.sequence_parallel_size))
        self.sequence_parallel_group = None
        self.sequence_parallel_rank = 0

        self.init_deepspeed_runtime_state(cfg)
        self.load_tokenizer_and_base_model(cfg, AutoTokenizer, AutoModelForCausalLM)
        self.attach_lora_adapter(cfg, PeftModel, LoraConfig, TaskType, get_peft_model)
        self.enable_deepspeed_model_optimizations(cfg)
        if cfg.gradient_checkpointing:
            self.enable_gradient_checkpointing()
        self.build_optimizer(cfg)
        self.wrap_model_for_deepspeed()
        self.finalize_model_state()

    def init_deepspeed_runtime_state(self, cfg: TrainerConfig) -> None:
        """Initialize distributed runtime state for the DeepSpeed stage."""

        self.world_size = world_size_from_env()
        self.rank = rank_from_env()
        self.local_rank = local_rank_from_env()
        self.distributed_mode = "ddp" if self.world_size > 1 or int(cfg.trainer_num_workers) > 1 else "none"
        self.distributed_active = self.world_size > 1 and self.distributed_mode != "none"
        if self.distributed_active and not self.dist.is_initialized():
            raise RuntimeError("Distributed training requested but torch.distributed is not initialized.")
        if self.distributed_active and not self.deepspeed.comm.is_initialized():
            dist_backend = "nccl" if self.torch.cuda.is_available() else "gloo"
            self.deepspeed.init_distributed(
                dist_backend=dist_backend,
                init_method="env://",
                auto_mpi_discovery=False,
            )
            if not self.deepspeed.comm.is_initialized():
                raise RuntimeError("DeepSpeed comm failed to initialize from the active torch.distributed process group.")
        if self.torch.cuda.is_available() and self.distributed_active:
            self.torch.cuda.set_device(self.local_rank)

        if self.torch.cuda.is_available() and self.distributed_active:
            self.device = self.torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.dtype = self.torch.bfloat16 if self.torch.cuda.is_available() else self.torch.float32
        self.logprob_compute_dtype = resolve_logprob_compute_dtype(
            self.torch,
            self.cfg.logprob_compute_dtype,
            device_type=self.device.type,
        )
        if self.torch.cuda.is_available():
            self.torch.backends.cuda.matmul.allow_tf32 = True
            self.torch.backends.cudnn.allow_tf32 = True
        logger.info(
            "deepspeed distributed_mode=%s world_size=%d trainer_num_workers=%d rank=%d local_rank=%d current_cuda_device=%s",
            self.distributed_mode,
            self.world_size,
            int(cfg.trainer_num_workers),
            self.rank,
            self.local_rank,
            str(self.torch.cuda.current_device()) if self.torch.cuda.is_available() else "cpu",
        )

    def enable_deepspeed_model_optimizations(self, cfg: TrainerConfig) -> None:
        """Enable packed attention and optional Ulysses sequence parallelism."""

        if not deepspeed_packed_attention_requested(cfg):
            return
        ensure_flash_attention_2_ready_for_deepspeed()
        if hasattr(self.model, "config"):
            self.model.config._attn_implementation = "flash_attention_2"
        if self.sequence_parallel_size <= 1:
            logger.info("deepspeed_optimizations: enabled packed flash attention remove_padding=%s", cfg.use_remove_padding)
            return

        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF

        mpu = UlyssesSPAttentionHF.register_with_transformers(
            self.model,
            core_attn_implementation="flash_attention_2",
            sequence_parallel_size=self.sequence_parallel_size,
            micro_batch_size=1,
            seq_length_is_variable=True,
            disable_in_eval=True,
        )
        if mpu is None:
            raise RuntimeError("DeepSpeed Ulysses registration unexpectedly returned no sequence parallel state.")
        self.sequence_parallel_group = mpu.get_sequence_parallel_group()
        self.sequence_parallel_rank = mpu.get_sequence_parallel_rank()
        # Store the mpu object so wrap_model_for_deepspeed can pass it to
        # deepspeed.initialize().  This lets ZeRO-2 know the real SP topology
        # and divide gradients by W/SP = DP instead of W.
        self._ulysses_mpu = mpu
        logger.info(
            "deepspeed_optimizations: registered sequence parallel sp=%d remove_padding=%s",
            self.sequence_parallel_size,
            cfg.use_remove_padding,
        )

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce activation memory.

        Trades ~30-40% extra compute for dramatically lower per-layer
        activation storage.  Only boundary hidden states between
        checkpointed segments are kept; intermediates are recomputed
        during backward.  ``use_reentrant=False`` is required for
        compatibility with LoRA (frozen base params have no grad).
        """

        target = self.model
        if hasattr(target, "gradient_checkpointing_enable"):
            target.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            logger.info("gradient_checkpointing: enabled (use_reentrant=False)")
        else:
            logger.warning("gradient_checkpointing: model does not support gradient_checkpointing_enable")

    def wrap_model_for_deepspeed(self) -> None:
        """Wrap the model and optimizer with DeepSpeed.

        When Ulysses SP is active, mpu is passed to deepspeed.initialize()
        so that ZeRO-2 knows the real sequence-parallel topology.  This
        changes the gradient divisor inside ZeRO's reduce-scatter from
        world_size to world_size / SP = DP.

        Gradient flow with mpu passed:
          1. Each SP rank computes loss over 1/SP of the tokens.
          2. prepare_loss_for_backward does all_gather+sum, whose backward
             (reduce_scatter) amplifies gradients by SP, reconstructing
             the full-batch gradient.
          3. ZeRO-2 average_tensor divides by W/SP = DP.
          4. Net gradient at the optimizer: full_grad / DP.
          5. build_optimizer scales LR by DP → effective update = full_grad × lr.
             This matches TTT-Discover (Tinker), which sees full_grad × lr.

        Without mpu, ZeRO's sequence_parallel_size defaults to 1 and
        the divisor becomes W, making the effective LR SP× too small.
        """

        ds_config = build_deepspeed_engine_config(
            self.torch,
            distributed_active=self.distributed_active,
            model_dtype=self.dtype,
        )
        ulysses_mpu = getattr(self, "_ulysses_mpu", None)
        if ulysses_mpu is not None and self.rank == 0:
            logger.info("deepspeed_init: passing ulysses mpu to deepspeed.initialize (sp=%d)", self.sequence_parallel_size)
        self.model, self.optimizer, _, _ = self.deepspeed.initialize(
            model=self.model,
            model_parameters=self.trainable_params,
            optimizer=self.optimizer,
            config=ds_config,
            dist_init_required=False,
            mpu=ulysses_mpu,
        )
        # Log the actual divisor ZeRO-2 will use in its reduce-scatter so we
        # can confirm it matches expectations (should be DP, not world_size).
        if self.rank == 0:
            zero_opt = getattr(self.model, 'optimizer', None)
            zero_sp = getattr(zero_opt, 'sequence_parallel_size', None)
            zero_dp_group = getattr(zero_opt, 'dp_process_group', None)
            if zero_sp is not None and zero_dp_group is not None:
                dp_group_size = self.dist.get_world_size(group=zero_dp_group)
                divisor = dp_group_size / float(zero_sp)
                logger.info(
                    "deepspeed_init: zero_grad_divisor=%.1f "
                    "(dp_group_size=%d zero_sequence_parallel_size=%d world_size=%d "
                    "expected_divisor=%.1f)",
                    divisor, dp_group_size, zero_sp, self.world_size,
                    float(max(1, self.world_size // max(1, self.sequence_parallel_size))),
                )

    def zero_grad_for_step(self) -> None:
        self.model.zero_grad()
        # Tell DeepSpeed that subsequent backward() calls should accumulate
        # gradients without finalizing.  The boundary is flipped back to
        # True on the last backward call (is_last=True) to finalize the
        # accumulated gradients, matching the documented DeepSpeed pattern.
        self.model.set_gradient_accumulation_boundary(False)
        self._backward_call_count = 0

    def _all_grad_tensors_norm(self) -> float | None:
        """L2 norm of ZeRO's in-progress accumulation buffer (all_grad_tensors).

        Returns None if the buffer doesn't exist or is empty.
        """
        optimizer = getattr(self.model, 'optimizer', None)
        buf = getattr(optimizer, 'all_grad_tensors', None)
        if not buf:
            return None
        sq_sum = 0.0
        found = False
        for grad_list in buf.values():
            if grad_list is not None:
                for g in grad_list:
                    if g is not None:
                        sq_sum += g.detach().float().pow(2).sum().item()
                        found = True
        return sq_sum ** 0.5 if found else None

    def backward_loss(self, loss, is_last: bool = False) -> None:
        # With ZeRO-2, reduce-scatter hooks fire on every backward (because
        # partition_gradients=True), but the accumulation buffer is only
        # finalized when is_gradient_accumulation_boundary=True.
        # We set boundary=False for all but the final backward so that
        # gradients accumulate across microbatches, then boundary=True on
        # the last one to finalize — matching the documented DeepSpeed
        # gradient accumulation pattern.
        if is_last:
            self.model.set_gradient_accumulation_boundary(True)
        self.model.backward(loss)
        self._backward_call_count += 1
        # Log the accumulation buffer norm after the 1st backward so we can
        # compare single-microbatch signal vs full-batch signal.
        if self._backward_call_count == 1 and self.rank == 0:
            first_norm = self._all_grad_tensors_norm()
            logger.info("trainer_grad_diag first_backward_accum_norm=%s",
                        f"{first_norm:.6f}" if first_norm is not None else "none")

    def prepare_loss_for_backward(self, loss):
        if self.sequence_parallel_size <= 1 or not self.distributed_active or self.sequence_parallel_group is None:
            return loss
        gathered_losses = self.dist_nn.all_gather(loss, group=self.sequence_parallel_group)
        return self.torch.stack(list(gathered_losses)).sum()

    def _grad_norm(self) -> float:
        """L2 norm of accumulated gradients from ZeRO's internal buffers.

        With ZeRO-2, param.grad is cleared after the epilogue — the real
        gradients live in optimizer.averaged_gradients (the partitioned
        gradient slices this rank owns).  Falls back to param.grad for
        non-ZeRO backends.
        """
        sq_sum = 0.0
        optimizer = getattr(self.model, 'optimizer', None)
        avg_grads = getattr(optimizer, 'averaged_gradients', None)
        if avg_grads:
            for grad_list in avg_grads.values():
                if grad_list is not None:
                    for g in grad_list:
                        if g is not None:
                            sq_sum += g.detach().float().pow(2).sum().item()
        else:
            for p in self.trainable_params:
                if p.grad is not None:
                    sq_sum += p.grad.detach().float().pow(2).sum().item()
        return sq_sum ** 0.5

    def run_optimizer_step(self) -> tuple[float, int]:
        # The last backward was called with boundary=True, so ZeRO-2 has
        # already finalized the accumulated gradients into averaged_gradients.
        # model.step() checks is_gradient_accumulation_boundary() — it must
        # still be True (set during the last backward_loss call).
        final_avg_grad_norm = self._grad_norm()
        accum_buf_norm = self._all_grad_tensors_norm()
        if self.rank == 0:
            logger.info(
                "trainer_grad_sync backward_calls=%d "
                "averaged_gradients_norm=%.6f accum_buf_norm=%s",
                self._backward_call_count,
                final_avg_grad_norm,
                f"{accum_buf_norm:.6f}" if accum_buf_norm is not None else "none(cleared)",
            )
        optimizer_started_at = time.perf_counter()
        self.model.step()
        return time.perf_counter() - optimizer_started_at, 1

    def _zero_dp_rank(self) -> int:
        """Return this rank's position within ZeRO's dp_process_group.

        ZeRO partitions optimizer state across this group.  With Ulysses
        SP the group spans all ranks (seq+data parallel), so dp_rank =
        global rank.  With real tensor parallelism the group would be
        smaller and dp_rank != global rank.  Using this consistently for
        save and load ensures correctness regardless of topology.
        """
        return int(self.dist.get_rank(group=self.model.optimizer.dp_process_group))

    def _zero_dp_world_size(self) -> int:
        """Return the size of ZeRO's dp_process_group."""
        return int(self.dist.get_world_size(group=self.model.optimizer.dp_process_group))

    def save_optimizer_state(self, save_dir: str) -> str:
        """Save ZeRO optimizer state directly, bypassing DeepSpeed's engine checkpoint.

        Parity: the original TTT-Discover keeps a persistent Tinker
        TrainingClient across epochs, so optimizer state (Adam first and
        second moments) naturally carries over.  NanoDiscover spawns a
        fresh subprocess each epoch, so we must explicitly save and
        reload the optimizer state to match.

        We save the ZeRO optimizer's own state_dict (which contains the
        Adam moments, fp32 master weights partition, loss scaler, etc.)
        as one file per dp_rank.  This avoids DeepSpeed's engine-level
        save_checkpoint which conflates mp_world_size with SP via the
        Ulysses mpu aliasing, causing IndexError on load when SP > 1.

        Files are keyed by dp_rank (not global rank) so the scheme
        stays correct if tensor parallelism is later added.

        All ranks participate — each saves its own partition.
        """

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dp_rank = self._zero_dp_rank()
        dp_ws = self._zero_dp_world_size()
        zero_sd = self.model.optimizer.state_dict()
        rank_path = out_dir / f"zero_optim_dp_rank_{dp_rank:03d}.pt"
        self.torch.save(zero_sd, str(rank_path))
        if self.rank == 0:
            # Save a small metadata file so load can verify topology consistency.
            meta = {"dp_world_size": dp_ws, "world_size": self.world_size, "format": "nanodiscover_zero_v1"}
            self.torch.save(meta, str(out_dir / "optim_meta.pt"))
            logger.info("optimizer_state_save dir=%s dp_rank=%d dp_world_size=%d", str(out_dir), dp_rank, dp_ws)
        # Barrier so all ranks finish writing before any rank proceeds.
        if self.distributed_active:
            self.dist.barrier()
        return str(out_dir)

    def load_optimizer_state(self, load_dir: str) -> None:
        """Restore ZeRO optimizer state from a previous epoch's per-rank files.

        Loads optimizer states only; model weights are already correct
        (loaded via PEFT adapter before DeepSpeed wrapping).
        """

        load_path = Path(load_dir)
        dp_rank = self._zero_dp_rank()
        dp_ws = self._zero_dp_world_size()
        rank_file = load_path / f"zero_optim_dp_rank_{dp_rank:03d}.pt"
        if not rank_file.exists():
            # Fall back: check for legacy DeepSpeed engine checkpoint format
            # from runs before this change.
            tag_dir = load_path / "optimizer_state"
            if tag_dir.exists():
                if self.rank == 0:
                    logger.warning(
                        "optimizer_state_load dir=%s found legacy engine checkpoint "
                        "but not per-rank files; skipping (start fresh optimizer)",
                        str(load_dir),
                    )
                return
            if self.rank == 0:
                logger.warning("optimizer_state_load dir=%s rank_file_missing=1 skipping", str(load_dir))
            return
        # Verify dp_world_size matches.
        meta_file = load_path / "optim_meta.pt"
        if meta_file.exists():
            meta = self.torch.load(str(meta_file), map_location="cpu", weights_only=True)
            saved_dp_ws = meta.get("dp_world_size")
            if saved_dp_ws is not None and saved_dp_ws != dp_ws:
                raise RuntimeError(
                    f"Optimizer checkpoint was saved with dp_world_size={saved_dp_ws} but "
                    f"current dp_world_size={dp_ws}. Cannot resume with a "
                    f"different ZeRO partition count."
                )
        zero_sd = self.torch.load(str(rank_file), map_location="cpu", weights_only=False)
        # _load_legacy_checkpoint indexes by dp_rank into state_dict_list.
        # Build a list where only this rank's slot is populated.
        state_dict_list = [None] * dp_ws
        state_dict_list[dp_rank] = zero_sd
        self.model.optimizer.load_state_dict(
            state_dict_list,
            load_optimizer_states=True,
            load_from_fp32_weights=True,
        )
        if self.rank == 0:
            logger.info("optimizer_state_load dir=%s success=1 dp_rank=%d dp_world_size=%d", str(load_dir), dp_rank, dp_ws)


class StageProcessBackend:
    """High-level backend that launches the isolated training stage process."""

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.distributed_mode = self.resolve_stage_distributed_mode(config)
        self.adapter_path_value: str | None = None
        self.optimizer_state_dir_value: str | None = None

    def resolve_stage_distributed_mode(self, config: TrainerConfig) -> str:
        """Return the subprocess launch mode for this backend."""
        _ = config
        return "none"

    def adapter_path(self) -> str | None:
        return self.adapter_path_value

    def set_resume_adapter(self, adapter_path: str | None) -> None:
        candidate = (adapter_path or "").strip()
        if not candidate:
            self.adapter_path_value = None
            return
        path_obj = Path(candidate)
        if not path_obj.exists():
            raise FileNotFoundError(f"resume adapter path does not exist: {path_obj}")
        self.adapter_path_value = str(path_obj)

    def set_resume_optimizer(self, optimizer_state_dir: str | None) -> None:
        """Record the optimizer state directory to resume from."""

        candidate = (optimizer_state_dir or "").strip()
        if not candidate:
            self.optimizer_state_dir_value = None
            return
        path_obj = Path(candidate)
        if not path_obj.exists():
            self.optimizer_state_dir_value = None
            return
        self.optimizer_state_dir_value = str(path_obj)

    def build_stage_config(self) -> TrainerConfig:
        return replace(
            self.config,
            distributed_mode=self.distributed_mode,
            resume_adapter_path=self.adapter_path_value,
            resume_optimizer_path=self.optimizer_state_dir_value,
        )

    def build_stage_command(self, payload_path: Path) -> list[str]:
        python_executable = sys.executable
        distributed_stage = self.distributed_mode != "none" and int(self.config.trainer_num_workers) > 1
        if distributed_stage:
            cmd = [
                python_executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(int(self.config.trainer_num_workers)),
                "-m",
                "core.trainer",
            ]
        else:
            cmd = [python_executable, "-m", "core.trainer"]
        cmd.extend(
            [
                "--stage-payload",
                str(payload_path),
                "--distributed-mode",
                self.distributed_mode,
                "--distributed-backend",
                "nccl",
            ]
        )
        return cmd

    def clean_dist_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
        ):
            env.pop(key, None)
        return env

    def train_step(self, loss_samples: list[LossSample], *, output_dir: Path) -> dict[str, float]:
        if not loss_samples:
            return {"train/loss": 0.0, "train/tokens_per_sec": 0.0}
        output_dir.mkdir(parents=True, exist_ok=True)
        payload_path = output_dir / "stage_payload.pkl"
        output_path = output_dir / "stage_output.json"
        adapter_dir = output_dir / "adapter"
        optimizer_state_dir = output_dir / "optimizer_state"
        payload = {
            "samples": loss_samples,
            "trainer_cfg": self.build_stage_config(),
            "adapter_dir": str(adapter_dir),
            "optimizer_state_dir": str(optimizer_state_dir),
            "output_path": str(output_path),
        }
        with payload_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        subprocess.run(self.build_stage_command(payload_path), check=True, env=self.clean_dist_env())
        result = json.loads(output_path.read_text(encoding="utf-8"))
        metrics_raw = result.get("metrics", {})
        metrics = {str(key): float(value) for key, value in metrics_raw.items() if isinstance(value, (int, float))}
        adapter_path = result.get("adapter_path")
        if not isinstance(adapter_path, str) or not adapter_path:
            raise RuntimeError(f"{self.config.backend_name} stage did not return a valid adapter path")
        self.adapter_path_value = adapter_path
        self.optimizer_state_dir_value = result.get("optimizer_state_dir") or str(optimizer_state_dir)
        return metrics


class DeepSpeedBackend(StageProcessBackend):
    """DeepSpeed process-launch backend for the public training path."""

    def resolve_stage_distributed_mode(self, config: TrainerConfig) -> str:
        return "ddp" if int(config.trainer_num_workers) > 1 else "none"


def build_trainer_backend(config: TrainerConfig) -> TrainerBackend:
    """Instantiate the configured high-level trainer backend."""

    backend_name = normalized_backend_name(config.backend_name)
    if backend_name == "dry-run":
        return DryRunBackend(config)
    if backend_name == "deepspeed":
        return DeepSpeedBackend(config)
    raise RuntimeError(f"Unsupported trainer backend: {config.backend_name!r}")


def build_stage_runner(config: TrainerConfig):
    """Instantiate the stage-local runner for a subprocess training stage."""

    backend_name = normalized_backend_name(config.backend_name)
    if backend_name == "deepspeed":
        return DeepSpeedStageRunner(config)
    raise RuntimeError(f"Unsupported stage runner backend: {config.backend_name!r}")


class Trainer:
    """Convert evaluated rollouts into loss samples and run one training step."""

    def __init__(
        self,
        config: TrainerConfig,
        *,
        backend: TrainerBackend | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.resume_adapter_path: str | None = None
        self.resume_optimizer_state_dir: str | None = None

    def set_resume_adapter(self, adapter_path: str | None) -> None:
        """Record the adapter path that should be loaded before training."""

        self.resume_adapter_path = adapter_path

    def set_resume_optimizer(self, optimizer_state_dir: str | None) -> None:
        """Record the optimizer state directory to resume from."""

        self.resume_optimizer_state_dir = optimizer_state_dir

    def train(self, rollout_groups: list[list[EvaluatedRollout]], *, epoch: int, output_dir: str | Path | None = None) -> TrainingResult:
        """Train on one epoch of evaluated rollout groups without changing parity."""

        validate_training_topology(self.config)
        logger.info("trainer_topology %s", topology_summary(self.config))
        active_groups = rollout_groups
        dropped_groups = 0
        if self.config.remove_constant_reward_groups:
            active_groups, dropped_groups = drop_constant_reward_groups(rollout_groups)
        if not active_groups:
            return TrainingResult(
                metrics={"dropped_constant_reward_groups": float(dropped_groups)},
                adapter_path=self.resume_adapter_path,
                skipped=True,
            )

        reward_groups = [[float(rollout.reward) for rollout in group] for group in active_groups]
        advantage_groups = compute_advantages(
            reward_groups,
        )
        loss_samples, dropped_invalid_loss_samples = build_loss_samples(active_groups, advantage_groups)
        metrics = {
            "dropped_constant_reward_groups": float(dropped_groups),
            "dropped_invalid_loss_samples": float(dropped_invalid_loss_samples),
            "num_loss_samples": float(len(loss_samples)),
        }
        if dropped_invalid_loss_samples > 0:
            logger.warning(
                "trainer_drop_invalid_loss_samples epoch=%d dropped=%d total_rollouts=%d remaining=%d",
                epoch,
                dropped_invalid_loss_samples,
                sum(len(group) for group in active_groups),
                len(loss_samples),
            )
        if not loss_samples:
            logger.warning(
                "trainer_skip_empty_loss_samples epoch=%d active_groups=%d",
                epoch,
                len(active_groups),
            )
            metrics["skipped_empty_loss_samples"] = 1.0
            return TrainingResult(
                metrics=metrics,
                adapter_path=self.resume_adapter_path,
                loss_samples=[],
                skipped=True,
            )

        output_dir = Path(output_dir) if output_dir is not None else (Path(self.config.run_dir) / f"epoch{epoch:03d}")
        if self.backend is None:
            self.backend = build_trainer_backend(self.config)
        backend = self.backend
        if hasattr(backend, "set_resume_adapter"):
            backend.set_resume_adapter(self.resume_adapter_path)
        if hasattr(backend, "set_resume_optimizer"):
            backend.set_resume_optimizer(self.resume_optimizer_state_dir)
        logger.info(
            "trainer_resume epoch=%d adapter=%s optimizer_state=%s",
            epoch,
            self.resume_adapter_path or "none",
            self.resume_optimizer_state_dir or "none",
        )
        backend_metrics = backend.train_step(loss_samples, output_dir=output_dir)
        self.resume_adapter_path = backend.adapter_path() or str(output_dir)
        self.resume_optimizer_state_dir = getattr(backend, "optimizer_state_dir_value", None)
        metrics.update({str(key): float(value) for key, value in backend_metrics.items()})
        return TrainingResult(
            metrics=metrics,
            adapter_path=self.resume_adapter_path,
            optimizer_state_dir=self.resume_optimizer_state_dir,
            loss_samples=loss_samples,
        )

    def teardown(self) -> None:
        """Release trainer resources when the caller owns lifecycle control."""

        pass

def build_stage_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for isolated stage execution."""

    parser = argparse.ArgumentParser(description="Run one isolated trainer stage.")
    parser.add_argument("--stage-payload", type=str, required=True)
    parser.add_argument("--distributed-mode", choices=["auto", "none", "ddp"], default="none")
    parser.add_argument("--distributed-backend", type=str, default="nccl")
    return parser


def run_stage_from_payload(payload_path: str, distributed_mode: str, distributed_backend: str) -> None:
    """Execute one isolated training stage from a serialized payload."""

    import torch
    import torch.distributed as dist

    configure_stage_process_logging()

    world_size = world_size_from_env()
    dist_enabled = world_size > 1 and distributed_mode != "none"
    if dist_enabled and torch.cuda.is_available():
        # Ensure process-local GPU affinity before process-group/model init.
        torch.cuda.set_device(local_rank_from_env())
    stage_failed = False
    if dist_enabled and not dist.is_initialized():
        dist.init_process_group(backend=distributed_backend)
    try:
        with Path(payload_path).open("rb") as handle:
            payload = pickle.load(handle)
        samples = payload["samples"]
        trainer_cfg = payload["trainer_cfg"]
        trainer_cfg.distributed_mode = distributed_mode
        trainer_cfg.distributed_backend = distributed_backend
        validate_training_topology(trainer_cfg)
        adapter_dir = Path(payload["adapter_dir"])
        optimizer_state_dir = Path(payload.get("optimizer_state_dir") or str(Path(payload["adapter_dir"]).parent / "optimizer_state"))
        output_path = Path(payload["output_path"])
        metrics: dict[str, float] = {}
        training_local_samples, global_sample_count, global_padding_count = shard_loss_samples_for_rank(
            samples,
            world_size=world_size if dist_enabled else 1,
            rank=rank_from_env(),
            sequence_parallel_size=max(1, int(trainer_cfg.sequence_parallel_size)),
        )
        training_local_padding_count = count_padding_samples(training_local_samples)
        reference_scoring_model_parallel_size = resolve_reference_scoring_model_parallel_size(trainer_cfg)
        reference_scoring_data_parallel_size = resolve_reference_scoring_data_parallel_size(
            trainer_cfg,
            world_size=world_size if dist_enabled else 1,
        )
        reference_sample_indices, reference_local_samples = shard_reference_loss_samples_for_rank(
            samples,
            world_size=world_size if dist_enabled else 1,
            rank=rank_from_env(),
            model_parallel_size=reference_scoring_model_parallel_size,
        )
        metrics["train/reference_scoring_model_parallel_size"] = float(reference_scoring_model_parallel_size)
        metrics["train/reference_scoring_data_parallel_size"] = float(reference_scoring_data_parallel_size)
        if global_padding_count > 0 and rank_from_env() == 0:
            logger.warning(
                "stage_shard_padding global_samples=%d global_padding_samples=%d trainer_num_workers=%d sequence_parallel_size=%d",
                global_sample_count,
                global_padding_count,
                int(trainer_cfg.trainer_num_workers),
                int(trainer_cfg.sequence_parallel_size),
            )
        logger.info(
            "stage_progress rank=%d local_rank=%d phase=kl_prepare global_samples=%d global_padding_samples=%d training_local_samples=%d training_local_padding_samples=%d reference_local_samples=%d reference_local_tokens=%d reference_scoring_model_parallel_size=%d reference_scoring_data_parallel_size=%d",
            rank_from_env(),
            local_rank_from_env(),
            global_sample_count,
            global_padding_count,
            len(training_local_samples),
            training_local_padding_count,
            len(reference_local_samples),
            count_sequence_tokens(reference_local_samples),
            reference_scoring_model_parallel_size,
            reference_scoring_data_parallel_size,
        )
        if trainer_cfg.kl_penalty_coef > 0 and samples:
            kl_started_at = time.perf_counter()
            logger.info(
                "stage_progress rank=%d local_rank=%d phase=kl_start max_tokens_per_batch=%s",
                rank_from_env(),
                local_rank_from_env(),
                str(resolve_reference_scoring_max_tokens_per_batch(trainer_cfg)),
            )
            if reference_local_samples:
                diffs_by_sample, local_total_diff, local_total_mask = score_reference_policy_kl(
                    reference_local_samples,
                    model_name_or_path=trainer_cfg.model_name_or_path,
                    max_tokens_per_batch=resolve_reference_scoring_max_tokens_per_batch(trainer_cfg),
                    logprob_compute_dtype=trainer_cfg.logprob_compute_dtype,
                    vocab_chunk_size=trainer_cfg.reference_logprob_vocab_chunk_size,
                )
            else:
                diffs_by_sample = []
                local_total_diff = 0.0
                local_total_mask = 0.0
            if dist_enabled:
                device = torch.device(f"cuda:{local_rank_from_env()}") if torch.cuda.is_available() else torch.device("cpu")
                totals = torch.tensor([local_total_diff, local_total_mask], dtype=torch.float64, device=device)
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                average_diff = float(totals[0].item() / max(totals[1].item(), 1e-8))
            else:
                average_diff = float(local_total_diff / max(local_total_mask, 1e-8))
            local_updates = list(zip(reference_sample_indices, diffs_by_sample, strict=True))
            if dist_enabled and reference_scoring_data_parallel_size > 1:
                gathered_updates: list[list[tuple[int, list[float]]]] = [list() for _ in range(world_size)]
                dist.all_gather_object(gathered_updates, local_updates)
                diffs_for_all_samples = collect_reference_kl_diffs(len(samples), gathered_updates)
            else:
                diffs_for_all_samples = collect_reference_kl_diffs(len(samples), [local_updates])
            metrics.update(
                apply_kl_adjustment(
                    samples,
                    diffs_for_all_samples,
                    average_diff=average_diff,
                    kl_penalty_coef=trainer_cfg.kl_penalty_coef,
                )
            )
            metrics["train/kl_score_time_s"] = float(time.perf_counter() - kl_started_at)
            logger.info(
                "stage_progress rank=%d local_rank=%d phase=kl_done elapsed_s=%.2f local_total_mask=%.1f",
                rank_from_env(),
                local_rank_from_env(),
                float(metrics["train/kl_score_time_s"]),
                float(local_total_mask),
            )
        else:
            logger.info(
                "stage_progress rank=%d local_rank=%d phase=kl_skip reason=%s",
                rank_from_env(),
                local_rank_from_env(),
                "no_samples" if not samples else "kl_penalty_disabled",
            )
        if rank_from_env() == 0:
            logger.info("trainer_topology %s", topology_summary(trainer_cfg))
        logger.info(
            "stage_progress rank=%d local_rank=%d phase=trainer_init_start",
            rank_from_env(),
            local_rank_from_env(),
        )
        trainer = build_stage_runner(trainer_cfg)
        logger.info(
            "stage_progress rank=%d local_rank=%d phase=trainer_init_done",
            rank_from_env(),
            local_rank_from_env(),
        )
        # Parity: the original TTT-Discover keeps a persistent Tinker
        # TrainingClient across epochs, so Adam optimizer state (first
        # and second moments) carries over naturally.  NanoDiscover
        # spawns a fresh subprocess each epoch, so we explicitly reload
        # the optimizer state saved from the previous epoch.
        resume_optimizer_dir = (trainer_cfg.resume_optimizer_path or "").strip()
        if resume_optimizer_dir and hasattr(trainer, "load_optimizer_state"):
            trainer.load_optimizer_state(resume_optimizer_dir)
        logger.info("stage_progress rank=%d local_rank=%d phase=train_step_start", rank_from_env(), local_rank_from_env())
        train_started_at = time.perf_counter()
        metrics.update(trainer.train_step(samples))
        metrics["train/stage_runner_time_s"] = float(time.perf_counter() - train_started_at)
        logger.info("stage_progress rank=%d local_rank=%d phase=train_step_done elapsed_s=%.2f", rank_from_env(), local_rank_from_env(), metrics["train/stage_runner_time_s"])
        logger.info("stage_progress rank=%d local_rank=%d phase=save_adapter_start", rank_from_env(), local_rank_from_env())
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = trainer.save_adapter(str(adapter_dir))
        logger.info("stage_progress rank=%d local_rank=%d phase=save_adapter_done", rank_from_env(), local_rank_from_env())
        # Save optimizer state so the next epoch can resume with warm Adam moments.
        logger.info("stage_progress rank=%d local_rank=%d phase=save_optimizer_start", rank_from_env(), local_rank_from_env())
        optimizer_state_path = str(optimizer_state_dir)
        if hasattr(trainer, "save_optimizer_state"):
            optimizer_state_path = trainer.save_optimizer_state(str(optimizer_state_dir))
        if dist_enabled:
            dist.barrier()
        if rank_from_env() == 0:
            output_path.write_text(
                json.dumps({
                    "metrics": metrics,
                    "adapter_path": adapter_path,
                    "optimizer_state_dir": optimizer_state_path,
                    "num_samples": len(samples),
                }, sort_keys=True),
                encoding="utf-8",
            )
        if dist_enabled:
            dist.barrier()
    except Exception:
        stage_failed = True
        logger.exception(
            "stage_rank_failure rank=%d local_rank=%d world_size=%d payload=%s",
            rank_from_env(),
            local_rank_from_env(),
            world_size,
            payload_path,
        )
        raise
    finally:
        if dist_enabled and dist.is_initialized() and not stage_failed:
            dist.destroy_process_group()


def dispatch_stage_payload_cli() -> bool:
    """Dispatch the stage-runner CLI entrypoint when requested."""

    if "--stage-payload" not in sys.argv:
        return False
    args = build_stage_parser().parse_args()
    run_stage_from_payload(args.stage_payload, args.distributed_mode, args.distributed_backend)
    return True


if __name__ == "__main__":
    dispatch_stage_payload_cli()
