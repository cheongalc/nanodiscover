"""Optional 4-GPU production-path trainer probe.

This helper drives the real DeepSpeedStageRunner on the live production stack:

- Qwen/Qwen3-8B weights
- bf16
- DeepSpeed ZeRO-2
- Ulysses sequence parallel size 4
- packed/remove-padding path

It is launched by pytest wrappers via ``torch.distributed.run``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from core.trainer import DeepSpeedStageRunner, LossSample, TrainerConfig


MODEL_NAME = "Qwen/Qwen3-8B"


def build_config(
    *,
    run_dir: str,
    max_tokens_per_rank: int | None,
    resume_adapter_path: str | None,
    resume_optimizer_path: str | None,
) -> TrainerConfig:
    """Build a production-like trainer config for the runtime probe."""
    return TrainerConfig(
        backend_name="deepspeed",
        model_name_or_path=MODEL_NAME,
        tokenizer_name_or_path=MODEL_NAME,
        run_dir=run_dir,
        learning_rate=4e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=True,
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        num_substeps=1,
        trainer_num_workers=4,
        trainer_max_tokens_per_rank=max_tokens_per_rank,
        distributed_strategy="ddp",
        sequence_parallel_size=4,
        use_remove_padding=True,
        logprob_compute_dtype="float32",
        reference_logprob_vocab_chunk_size=4096,
        reference_scoring_max_tokens_per_rank=16384,
        reference_scoring_model_parallel_size=1,
        gradient_checkpointing=True,
        distributed_mode="ddp",
        resume_adapter_path=resume_adapter_path,
        resume_optimizer_path=resume_optimizer_path,
        distributed_backend="nccl",
    )


def build_samples(batch_name: str) -> list[LossSample]:
    """Build deterministic synthetic LossSamples for prod-path training."""
    if batch_name == "batch1":
        seed = 1234
    elif batch_name == "batch2":
        seed = 5678
    else:
        raise ValueError(f"Unknown batch name: {batch_name}")

    generator = torch.Generator().manual_seed(seed)
    lengths = [20, 24, 28, 32]
    out: list[LossSample] = []
    for index, seq_len in enumerate(lengths):
        input_ids = torch.randint(0, 151643, (seq_len,), generator=generator).tolist()
        target_ids = torch.randint(0, 151643, (seq_len,), generator=generator).tolist()
        old_lps = [-(0.1 + float(torch.rand((), generator=generator).item())) for _ in range(seq_len)]
        prompt_len = 2
        mask = [0.0] * prompt_len + [1.0] * (seq_len - prompt_len)
        advantage_value = 1.0 if index < 2 else -0.5
        advantages = [0.0] * prompt_len + [advantage_value] * (seq_len - prompt_len)
        out.append(
            LossSample(
                model_input_ids=input_ids,
                target_token_ids=target_ids,
                sampling_logprobs=old_lps,
                advantages=advantages,
                mask=mask,
                full_sequence_ids=input_ids + [target_ids[-1]],
                prompt_token_count=prompt_len,
                metadata={"batch_name": batch_name, "sample_index": index},
            )
        )
    return out


def max_trainable_param_delta(runner: DeepSpeedStageRunner, before: dict[str, torch.Tensor]) -> float:
    """Return the max absolute trainable-parameter delta across ranks."""
    local_max = 0.0
    model_like = runner.model.module if hasattr(runner.model, "module") else runner.model
    for name, param in model_like.named_parameters():
        if not param.requires_grad or name not in before:
            continue
        local_max = max(local_max, float((param.detach().float().cpu() - before[name]).abs().max().item()))
    tensor = torch.tensor([local_max], dtype=torch.float64, device=runner.input_device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def snapshot_trainable_params(runner: DeepSpeedStageRunner) -> dict[str, torch.Tensor]:
    """Return a CPU snapshot of trainable parameters."""
    model_like = runner.model.module if hasattr(runner.model, "module") else runner.model
    return {
        name: param.detach().float().cpu().clone()
        for name, param in model_like.named_parameters()
        if param.requires_grad
    }


def main() -> None:
    """Run the requested production-path trainer probe operation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["save_initial", "train_step"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-name", choices=["batch1", "batch2"], default="batch1")
    parser.add_argument("--max-tokens-per-rank", type=int, default=0)
    parser.add_argument("--resume-adapter-path", default="")
    parser.add_argument("--resume-optimizer-path", default="")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    try:
        torch.manual_seed(int(args.seed))
        max_tokens = int(args.max_tokens_per_rank)
        config = build_config(
            run_dir=str(Path(args.run_dir)),
            max_tokens_per_rank=max_tokens if max_tokens > 0 else None,
            resume_adapter_path=(args.resume_adapter_path or "").strip() or None,
            resume_optimizer_path=(args.resume_optimizer_path or "").strip() or None,
        )
        runner = DeepSpeedStageRunner(config)
        result: dict[str, object]

        if args.operation == "save_initial":
            adapter_path = runner.save_adapter(str(Path(args.output_dir) / "adapter"))
            result = {
                "adapter_path": adapter_path,
                "operation": args.operation,
            }
        else:
            samples = build_samples(args.batch_name)
            if config.resume_optimizer_path and hasattr(runner, "load_optimizer_state"):
                runner.load_optimizer_state(config.resume_optimizer_path)
            before = snapshot_trainable_params(runner)
            metrics = runner.train_step(samples)
            adapter_path = runner.save_adapter(str(Path(args.output_dir) / "adapter"))
            optimizer_state_dir = runner.save_optimizer_state(str(Path(args.output_dir) / "optimizer_state"))
            result = {
                "adapter_path": adapter_path,
                "batch_name": args.batch_name,
                "max_param_delta": max_trainable_param_delta(runner, before),
                "metrics": metrics,
                "operation": args.operation,
                "optimizer_state_dir": optimizer_state_dir,
            }
        if rank == 0:
            print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}")
    finally:
        if dist.is_initialized():
            dist.barrier()
        try:
            runner  # type: ignore[name-defined]
        except UnboundLocalError:
            pass
        else:
            del runner
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
