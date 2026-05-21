"""Distributed runtime probe for packed-boundary handling under Ulysses.

This helper is launched via ``torch.distributed.run`` from a pytest wrapper.
It exercises the real production stack:

- Qwen/Qwen3-8B
- bfloat16 model weights
- Hugging Face flash_attention_2
- DeepSpeed Ulysses sequence parallel
- packed input_ids plus reset position_ids
"""

from __future__ import annotations

import json
import os

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
from transformers import AutoModelForCausalLM

from core.trainer import ensure_flash_attention_2_ready_for_deepspeed


MODEL_NAME = "Qwen/Qwen3-8B"


def slice_for_rank(full_values: list[int], *, rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    """Return the contiguous per-rank sequence slice used by the trainer."""
    full = torch.tensor([full_values], dtype=torch.long, device=device)
    if full.shape[1] % world_size != 0:
        raise ValueError(f"Sequence length {full.shape[1]} must divide world_size={world_size}")
    chunk = full.shape[1] // world_size
    start = rank * chunk
    stop = start + chunk
    return full[:, start:stop].contiguous()


def forward_full_logits(
    model,
    *,
    full_input_ids: list[int],
    full_position_ids: list[int],
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run one SP forward pass and reconstruct the full global logits."""
    local_input_ids = slice_for_rank(full_input_ids, rank=rank, world_size=world_size, device=device)
    local_position_ids = slice_for_rank(full_position_ids, rank=rank, world_size=world_size, device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=local_input_ids,
            position_ids=local_position_ids,
            use_cache=False,
        )
    local_logits = outputs.logits.detach().to(dtype=torch.float32)
    gathered = [torch.empty_like(local_logits) for _ in range(world_size)]
    dist.all_gather(gathered, local_logits)
    return torch.cat(gathered, dim=1).cpu()


def main() -> None:
    """Probe whether packed reset position IDs block cross-sequence attention."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"This probe expects world_size=4, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    if not deepspeed.comm.is_initialized():
        deepspeed.init_distributed(
            dist_backend="nccl",
            init_method="env://",
            auto_mpi_discovery=False,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            device_map={"": local_rank},
        )
        ensure_flash_attention_2_ready_for_deepspeed()
        model.config._attn_implementation = "flash_attention_2"
        model.train()

        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model,
            core_attn_implementation="flash_attention_2",
            sequence_parallel_size=world_size,
            micro_batch_size=1,
            seq_length_is_variable=True,
            disable_in_eval=True,
        )
        if mpu is None:
            raise RuntimeError("Ulysses registration unexpectedly returned None")

        # Two packed samples of length 6. Total length 12 divides evenly over 4 SP ranks.
        a_left = [101, 102, 103, 104, 105, 106]
        a_right = [201, 202, 203, 204, 205, 206]
        b_shared = [301, 302, 303, 304, 305, 306]
        full_left_ids = a_left + b_shared
        full_right_ids = a_right + b_shared

        packed_position_ids = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        monotonic_position_ids = list(range(12))

        logits_reset_left = forward_full_logits(
            model,
            full_input_ids=full_left_ids,
            full_position_ids=packed_position_ids,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        logits_reset_right = forward_full_logits(
            model,
            full_input_ids=full_right_ids,
            full_position_ids=packed_position_ids,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        logits_mono_left = forward_full_logits(
            model,
            full_input_ids=full_left_ids,
            full_position_ids=monotonic_position_ids,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        logits_mono_right = forward_full_logits(
            model,
            full_input_ids=full_right_ids,
            full_position_ids=monotonic_position_ids,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        a_slice = slice(0, 6)
        b_slice = slice(6, 12)
        result = {
            "attention_dropout": float(model.config.attention_dropout),
            "model_name": MODEL_NAME,
            "monotonic_max_abs_diff_a": float(
                (logits_mono_left[:, a_slice, :] - logits_mono_right[:, a_slice, :]).abs().max().item()
            ),
            "monotonic_max_abs_diff_b": float(
                (logits_mono_left[:, b_slice, :] - logits_mono_right[:, b_slice, :]).abs().max().item()
            ),
            "monotonic_position_ids": monotonic_position_ids,
            "packed_position_ids": packed_position_ids,
            "rank_local_monotonic_position_ids": slice_for_rank(
                monotonic_position_ids,
                rank=rank,
                world_size=world_size,
                device=device,
            ).cpu().tolist(),
            "rank_local_packed_position_ids": slice_for_rank(
                packed_position_ids,
                rank=rank,
                world_size=world_size,
                device=device,
            ).cpu().tolist(),
            "reset_max_abs_diff_a": float(
                (logits_reset_left[:, a_slice, :] - logits_reset_right[:, a_slice, :]).abs().max().item()
            ),
            "reset_max_abs_diff_b": float(
                (logits_reset_left[:, b_slice, :] - logits_reset_right[:, b_slice, :]).abs().max().item()
            ),
            "world_size": world_size,
        }
        if rank == 0:
            print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
