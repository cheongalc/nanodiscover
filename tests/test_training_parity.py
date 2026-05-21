"""Tests for training-path correctness that were identified as coverage gaps.

Gap 1: compute_loss() with real GRPO loss formula (no GPU needed)
Gap 2: save_optimizer_state / load_optimizer_state round-trip (GPU needed)
Gap 3: KL penalty numeric correctness (no GPU needed)
Gap 4: microbatch equivalence using real compute_loss (GPU needed)

All tests call the ACTUAL production code, not reimplementations.
"""

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from core.trainer import (
    LossSample,
    apply_kl_adjustment,
    compute_kl_diffs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loss_sample(
    *,
    model_input_ids: list[int],
    target_token_ids: list[int],
    sampling_logprobs: list[float],
    advantages: list[float],
    mask: list[float],
) -> LossSample:
    """Build a LossSample with minimal metadata."""
    return LossSample(
        model_input_ids=model_input_ids,
        target_token_ids=target_token_ids,
        sampling_logprobs=sampling_logprobs,
        advantages=advantages,
        mask=mask,
        full_sequence_ids=model_input_ids + [target_token_ids[-1]],
        prompt_token_count=0,
        metadata={},
    )


# ===========================================================================
# Gap 3: KL penalty numeric correctness (pure Python, no GPU)
# ===========================================================================

class TestKLPenaltyNumeric:
    """Verify compute_kl_diffs and apply_kl_adjustment produce correct numbers."""

    def test_kl_diffs_basic(self):
        """delta[t] = (sampled_lp - base_lp) * mask. Total and per-sample."""
        sample = _make_loss_sample(
            model_input_ids=[1, 2, 3],
            target_token_ids=[2, 3, 4],
            sampling_logprobs=[-1.0, -2.0, -3.0],
            advantages=[0.5, 0.5, 0.5],
            mask=[1.0, 1.0, 0.0],
        )
        base_logprobs = [-1.5, -2.5, -3.5]

        diffs, total_diff, total_mask = compute_kl_diffs([sample], [base_logprobs])

        # diff[0] = (-1.0 - (-1.5)) * 1.0 = 0.5
        # diff[1] = (-2.0 - (-2.5)) * 1.0 = 0.5
        # diff[2] = (-3.0 - (-3.5)) * 0.0 = 0.0  (masked out)
        assert len(diffs) == 1
        assert len(diffs[0]) == 3
        assert abs(diffs[0][0] - 0.5) < 1e-10
        assert abs(diffs[0][1] - 0.5) < 1e-10
        assert abs(diffs[0][2] - 0.0) < 1e-10
        assert abs(total_diff - 1.0) < 1e-10
        assert abs(total_mask - 2.0) < 1e-10

    def test_kl_adjustment_shifts_advantages(self):
        """advantage += lambda * mask * (avg_diff - diff[t])."""
        sample = _make_loss_sample(
            model_input_ids=[1, 2, 3],
            target_token_ids=[2, 3, 4],
            sampling_logprobs=[-1.0, -2.0, -3.0],
            advantages=[1.0, 1.0, 1.0],
            mask=[1.0, 1.0, 0.0],
        )
        diffs_by_sample = [[0.8, 0.2, 0.0]]
        avg_diff = 0.5
        kl_coef = 0.1

        apply_kl_adjustment(
            [sample], diffs_by_sample,
            average_diff=avg_diff, kl_penalty_coef=kl_coef,
        )

        # token 0: 1.0 + 0.1 * 1.0 * (0.5 - 0.8) = 1.0 + 0.1 * (-0.3) = 0.97
        # token 1: 1.0 + 0.1 * 1.0 * (0.5 - 0.2) = 1.0 + 0.1 * (0.3) = 1.03
        # token 2: 1.0 + 0.1 * 0.0 * (0.5 - 0.0) = 1.0 (masked)
        assert abs(sample.advantages[0] - 0.97) < 1e-10
        assert abs(sample.advantages[1] - 1.03) < 1e-10
        assert abs(sample.advantages[2] - 1.0) < 1e-10

    def test_kl_full_pipeline_two_samples(self):
        """End-to-end: diffs → average → adjustment across two samples."""
        s1 = _make_loss_sample(
            model_input_ids=[10, 11],
            target_token_ids=[11, 12],
            sampling_logprobs=[-1.0, -2.0],
            advantages=[2.0, 2.0],
            mask=[1.0, 1.0],
        )
        s2 = _make_loss_sample(
            model_input_ids=[20, 21],
            target_token_ids=[21, 22],
            sampling_logprobs=[-0.5, -1.5],
            advantages=[3.0, 3.0],
            mask=[1.0, 0.0],
        )
        base_lps_1 = [-1.2, -2.3]
        base_lps_2 = [-0.8, -1.0]
        kl_coef = 0.1

        diffs, total_diff, total_mask = compute_kl_diffs([s1, s2], [base_lps_1, base_lps_2])

        # s1: diff[0] = (-1.0 - (-1.2)) * 1.0 = 0.2
        #     diff[1] = (-2.0 - (-2.3)) * 1.0 = 0.3
        # s2: diff[0] = (-0.5 - (-0.8)) * 1.0 = 0.3
        #     diff[1] = (-1.5 - (-1.0)) * 0.0 = 0.0
        assert abs(diffs[0][0] - 0.2) < 1e-10
        assert abs(diffs[0][1] - 0.3) < 1e-10
        assert abs(diffs[1][0] - 0.3) < 1e-10
        assert abs(diffs[1][1] - 0.0) < 1e-10
        assert abs(total_diff - 0.8) < 1e-10  # 0.2 + 0.3 + 0.3 + 0.0
        assert abs(total_mask - 3.0) < 1e-10  # 1 + 1 + 1 + 0

        avg = total_diff / total_mask  # 0.8 / 3.0 = 0.26667
        apply_kl_adjustment([s1, s2], diffs, average_diff=avg, kl_penalty_coef=kl_coef)

        # s1 token 0: 2.0 + 0.1 * 1.0 * (0.2667 - 0.2) = 2.0 + 0.00667 = 2.00667
        # s1 token 1: 2.0 + 0.1 * 1.0 * (0.2667 - 0.3) = 2.0 - 0.00333 = 1.99667
        # s2 token 0: 3.0 + 0.1 * 1.0 * (0.2667 - 0.3) = 3.0 - 0.00333 = 2.99667
        # s2 token 1: 3.0 + 0.1 * 0.0 * (...) = 3.0
        assert abs(s1.advantages[0] - (2.0 + kl_coef * 1.0 * (avg - 0.2))) < 1e-8
        assert abs(s1.advantages[1] - (2.0 + kl_coef * 1.0 * (avg - 0.3))) < 1e-8
        assert abs(s2.advantages[0] - (3.0 + kl_coef * 1.0 * (avg - 0.3))) < 1e-8
        assert abs(s2.advantages[1] - 3.0) < 1e-8


# ===========================================================================
# Gap 1: compute_loss with real GRPO formula (GPU subprocess)
# Gap 2: save/load optimizer state round-trip (GPU subprocess)
# Gap 4: microbatch equivalence with real compute_loss (GPU subprocess)
# ===========================================================================

def _cuda_available(min_gpus: int = 1):
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() >= min_gpus
    except Exception:
        return False


def _run_gpu_script(tmp_path, script: str, *, timeout: int = 180):
    """Run a script in a subprocess with a single GPU."""
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script)

    import os as _os
    clean_env = {k: v for k, v in _os.environ.items()
                 if k not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                              "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"}}
    clean_env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=clean_env,
    )
    stdout = result.stdout
    stderr = result.stderr
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed (rc={result.returncode}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    return stdout, stderr


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_compute_loss_padded_matches_manual_grpo(tmp_path):
    """Padded batch path: compute_loss must match hand-computed GRPO loss."""
    _run_compute_loss_test(tmp_path, use_packed=False)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_compute_loss_packed_matches_manual_grpo(tmp_path):
    """Packed batch path (production Erdos/AC path): compute_loss must match."""
    _run_compute_loss_test(tmp_path, use_packed=True)


def _run_compute_loss_test(tmp_path, *, use_packed: bool):
    nanodiscover_root = str(Path(__file__).resolve().parent.parent)
    backend_name = "deepspeed" if use_packed else "dry-run"
    use_remove_padding = "True" if use_packed else "False"
    script = textwrap.dedent(f"""\
        import json
        import os
        import sys
        sys.path.insert(0, "{nanodiscover_root}")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29597")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        import torch
        import torch.distributed as dist
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoConfig

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)

        from core.trainer import (
            LossSample, StageRunnerBase, gather_aligned_target_logprobs,
        )

        # ---- Tiny model ----
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.vocab_size = 256
        cfg.max_position_embeddings = 128
        model = AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32, device="cuda:0")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
            lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.train()

        # ---- Build a shim StageRunnerBase ----
        shim = object.__new__(StageRunnerBase)
        shim.torch = torch
        shim.model = model
        shim.input_device = torch.device("cuda:0")
        shim.dtype = torch.float32
        shim.logprob_compute_dtype = torch.float32
        shim.pad_token_id = 0
        shim.sequence_parallel_size = 1
        shim.distributed_active = False
        shim.cfg = type("Cfg", (), {{
            "reference_logprob_vocab_chunk_size": 4096,
            "backend_name": "{backend_name}",
            "use_remove_padding": {use_remove_padding},
        }})()

        # ---- Create LossSamples with known values ----
        torch.manual_seed(42)
        seq_len = 12
        input_ids = torch.randint(0, 256, (seq_len,)).tolist()
        target_ids = torch.randint(0, 256, (seq_len,)).tolist()
        # Fake old logprobs (as if from the generation-time policy)
        old_logprobs = [-0.5, -1.2, -0.8, -2.0, -0.3, -1.5, -0.9, -1.1, -0.7, -1.8, -0.4, -1.3]
        advantages = [0.0, 0.0, 1.5, 1.5, 1.5, 1.5, -0.8, -0.8, -0.8, -0.8, 0.0, 0.0]
        mask = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

        sample = LossSample(
            model_input_ids=input_ids,
            target_token_ids=target_ids,
            sampling_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            full_sequence_ids=input_ids + [target_ids[-1]],
            prompt_token_count=2,
            metadata={{}},
        )

        # ---- Call the REAL compute_loss ----
        loss, token_count = shim.compute_loss([sample])
        production_loss = loss.item()

        # ---- Manually compute the expected loss ----
        # Forward pass to get new logprobs
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
        target_tensor = torch.tensor([target_ids], dtype=torch.long, device="cuda:0")
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, use_cache=False)
        new_logprobs = gather_aligned_target_logprobs(
            torch, outputs.logits, target_tensor, compute_dtype=torch.float32,
        )
        old_lp_tensor = torch.tensor([old_logprobs], dtype=torch.float32, device="cuda:0")
        adv_tensor = torch.tensor([advantages], dtype=torch.float32, device="cuda:0")
        mask_tensor = torch.tensor([mask], dtype=torch.float32, device="cuda:0")

        ratio = torch.exp(new_logprobs - old_lp_tensor)
        expected_loss = -(ratio * adv_tensor * mask_tensor).sum().item()

        # ---- Compare ----
        result = {{
            "production_loss": production_loss,
            "expected_loss": expected_loss,
            "abs_diff": abs(production_loss - expected_loss),
            "token_count": token_count,
            "expected_token_count": int(mask_tensor.sum().item()),
        }}
        print("RESULT:" + json.dumps(result))

        # fp32 on same device — should match exactly
        assert abs(production_loss - expected_loss) < 1e-4, (
            f"Loss mismatch: production={{production_loss}} expected={{expected_loss}}"
        )
        assert token_count == int(mask_tensor.sum().item())
        print("ALL_OK")
        dist.destroy_process_group()
    """)
    stdout, stderr = _run_gpu_script(tmp_path, script)
    batch_mode = "packed" if use_packed else "padded"
    assert "ALL_OK" in stdout, f"compute_loss ({batch_mode}) test failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parsed = json.loads(line[len("RESULT:"):])
            print(f"  compute_loss ({batch_mode}) OK: production={parsed['production_loss']:.6f} "
                  f"expected={parsed['expected_loss']:.6f} "
                  f"abs_diff={parsed['abs_diff']:.2e} "
                  f"tokens={parsed['token_count']}")
            break


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_optimizer_state_save_load_round_trips(tmp_path):
    """Verify the production save_optimizer_state/load_optimizer_state round-trips Adam moments."""
    nanodiscover_root = str(Path(__file__).resolve().parent.parent)
    script = textwrap.dedent(f"""\
        import json
        import os
        import sys
        sys.path.insert(0, "{nanodiscover_root}")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29596")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoConfig
        from core.trainer import DeepSpeedStageRunner

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)

        tmp = "{tmp_path}"
        optim_dir = os.path.join(tmp, "optim_state")
        os.makedirs(optim_dir, exist_ok=True)

        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.vocab_size = 256
        cfg.max_position_embeddings = 128

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
            lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )
        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": 2}},
            "bf16": {{"enabled": True}},
        }}

        def build_engine():
            m = AutoModelForCausalLM.from_config(cfg).to(dtype=torch.bfloat16, device="cuda:0")
            m = get_peft_model(m, lora_cfg)
            m.train()
            trainable = [p for p in m.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
            engine, optimizer, _, _ = deepspeed.initialize(
                model=m, model_parameters=trainable,
                optimizer=opt, config=ds_config,
                dist_init_required=False,
            )
            return engine, optimizer, trainable

        # ---- Engine 1: train one step, save ----
        engine1, opt1, _ = build_engine()
        input_ids = torch.randint(0, 256, (1, 16), device="cuda:0")
        out = engine1(input_ids=input_ids)
        engine1.backward(out.logits.sum())
        engine1.step()

        # Extract trained moments
        moments_before = {{}}
        for i, group in enumerate(opt1.param_groups):
            for j, p in enumerate(group["params"]):
                s = opt1.state.get(p, {{}})
                if "exp_avg" in s:
                    moments_before[(i,j)] = (
                        s["exp_avg"].detach().cpu().clone(),
                        s["exp_avg_sq"].detach().cpu().clone(),
                    )
        assert len(moments_before) > 0, "No optimizer state after 1 step"

        # Use real production save method via shim
        class _Shim:
            pass
        shim = _Shim()
        shim.model = engine1
        shim.rank = 0
        shim.world_size = 1
        shim.distributed_active = False
        shim.torch = torch
        shim.dist = dist
        shim._zero_dp_rank = DeepSpeedStageRunner._zero_dp_rank.__get__(shim)
        shim._zero_dp_world_size = DeepSpeedStageRunner._zero_dp_world_size.__get__(shim)
        save_fn = DeepSpeedStageRunner.save_optimizer_state.__get__(shim)
        save_fn(optim_dir)
        del engine1, opt1

        # ---- Engine 2: fresh init, load ----
        engine2, opt2, _ = build_engine()

        shim2 = _Shim()
        shim2.model = engine2
        shim2.rank = 0
        shim2.distributed_active = False
        shim2.torch = torch
        shim2.dist = dist
        shim2._zero_dp_rank = DeepSpeedStageRunner._zero_dp_rank.__get__(shim2)
        shim2._zero_dp_world_size = DeepSpeedStageRunner._zero_dp_world_size.__get__(shim2)
        shim2.world_size = 1
        load_fn = DeepSpeedStageRunner.load_optimizer_state.__get__(shim2)
        load_fn(optim_dir)

        # Compare moments
        mismatches = []
        restored = 0
        for i, group in enumerate(opt2.param_groups):
            for j, p in enumerate(group["params"]):
                s = opt2.state.get(p, {{}})
                if "exp_avg" in s and (i,j) in moments_before:
                    restored += 1
                    saved_avg, saved_sq = moments_before[(i,j)]
                    loaded_avg = s["exp_avg"].detach().cpu()
                    loaded_sq = s["exp_avg_sq"].detach().cpu()
                    if not torch.equal(loaded_avg, saved_avg):
                        mismatches.append("exp_avg mismatch at (" + str(i) + "," + str(j) + ")")
                    if not torch.equal(loaded_sq, saved_sq):
                        mismatches.append("exp_avg_sq mismatch at (" + str(i) + "," + str(j) + ")")

        # Also verify that per-rank file exists
        rank_file_exists = os.path.exists(os.path.join(optim_dir, "zero_optim_dp_rank_000.pt"))
        meta_file_exists = os.path.exists(os.path.join(optim_dir, "optim_meta.pt"))

        result = {{
            "moments_saved": len(moments_before),
            "moments_restored": restored,
            "mismatches": mismatches,
            "rank_file_exists": rank_file_exists,
            "meta_file_exists": meta_file_exists,
        }}
        print("RESULT:" + json.dumps(result))

        assert restored > 0, "No optimizer state restored"
        assert restored == len(moments_before), f"Restored {{restored}} != saved {{len(moments_before)}}"
        assert len(mismatches) == 0, f"Moment mismatches: {{mismatches}}"
        assert rank_file_exists, "Production save did not create zero_optim_dp_rank_000.pt"
        assert meta_file_exists, "Production save did not create optim_meta.pt"
        print("ALL_OK")
        dist.destroy_process_group()
    """)
    stdout, stderr = _run_gpu_script(tmp_path, script)
    assert "ALL_OK" in stdout, f"Optimizer save/load test failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parsed = json.loads(line[len("RESULT:"):])
            print(f"  optimizer round-trip OK: saved={parsed['moments_saved']} "
                  f"restored={parsed['moments_restored']} "
                  f"mismatches={len(parsed['mismatches'])} "
                  f"rank_file={parsed['rank_file_exists']} meta_file={parsed['meta_file_exists']}")
            break


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_microbatch_equivalence_real_loss_padded(tmp_path):
    """Microbatch accumulation with real GRPO loss, padded batch path."""
    _run_microbatch_real_loss_test(tmp_path, use_packed=False)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_microbatch_equivalence_real_loss_packed(tmp_path):
    """Microbatch accumulation with real GRPO loss, packed batch path (production)."""
    _run_microbatch_real_loss_test(tmp_path, use_packed=True)


def _run_microbatch_real_loss_test(tmp_path, *, use_packed: bool):
    nanodiscover_root = str(Path(__file__).resolve().parent.parent)
    backend_name = "deepspeed" if use_packed else "dry-run"
    use_remove_padding = "True" if use_packed else "False"
    script = textwrap.dedent(f"""\
        import json
        import os
        import sys
        sys.path.insert(0, "{nanodiscover_root}")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29595")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoConfig
        from core.trainer import (
            LossSample, DeepSpeedStageRunner, StageRunnerBase,
        )

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)

        NUM_MICROBATCHES = 4

        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.vocab_size = 256
        cfg.max_position_embeddings = 128

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
            lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )
        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": 0}},
        }}

        # ---- Create deterministic LossSamples ----
        torch.manual_seed(99)
        micro_samples = []
        for _ in range(NUM_MICROBATCHES):
            seq_len = torch.randint(8, 16, (1,)).item()
            input_ids = torch.randint(0, 256, (seq_len,)).tolist()
            target_ids = torch.randint(0, 256, (seq_len,)).tolist()
            old_lps = [-(torch.rand(1).item() * 3 + 0.1) for _ in range(seq_len)]
            # First 2 tokens are prompt (mask=0), rest are completion
            prompt_len = 2
            mask = [0.0] * prompt_len + [1.0] * (seq_len - prompt_len)
            advantages = [0.0] * prompt_len + [(torch.randn(1).item()) for _ in range(seq_len - prompt_len)]
            micro_samples.append(LossSample(
                model_input_ids=input_ids,
                target_token_ids=target_ids,
                sampling_logprobs=old_lps,
                advantages=advantages,
                mask=mask,
                full_sequence_ids=input_ids + [target_ids[-1]],
                prompt_token_count=prompt_len,
                metadata={{}},
            ))

        # ---- Save init state ----
        init_model = AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32, device="cuda:0")
        init_model = get_peft_model(init_model, lora_cfg)
        init_state = {{k: v.clone() for k, v in init_model.state_dict().items()}}
        del init_model

        # ==== PATH A: all samples in one compute_loss call, one backward ====
        model_a = AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32, device="cuda:0")
        model_a = get_peft_model(model_a, lora_cfg)
        model_a.load_state_dict(init_state, strict=True)
        model_a.train()
        trainable_a = [p for p in model_a.parameters() if p.requires_grad]
        opt_a = torch.optim.AdamW(trainable_a, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        engine_a, _, _, _ = deepspeed.initialize(
            model=model_a, model_parameters=trainable_a,
            optimizer=opt_a, config=ds_config, dist_init_required=False,
        )

        # Build shim for compute_loss
        shim_a = object.__new__(StageRunnerBase)
        shim_a.torch = torch
        shim_a.model = engine_a
        shim_a.input_device = torch.device("cuda:0")
        shim_a.dtype = torch.float32
        shim_a.logprob_compute_dtype = torch.float32
        shim_a.pad_token_id = 0
        shim_a.sequence_parallel_size = 1
        shim_a.distributed_active = False
        shim_a.cfg = type("Cfg", (), {{
            "reference_logprob_vocab_chunk_size": 4096,
            "backend_name": "{backend_name}",
            "use_remove_padding": {use_remove_padding},
        }})()

        all_samples = list(micro_samples)
        loss_a, _ = shim_a.compute_loss(all_samples)
        engine_a.backward(loss_a)
        engine_a.step()

        params_a = {{
            k: v.detach().cpu().clone()
            for k, v in engine_a.module.named_parameters()
            if v.requires_grad
        }}
        del engine_a, model_a, opt_a, shim_a

        # ==== PATH B: one sample per compute_loss, production boundary logic ====
        model_b = AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32, device="cuda:0")
        model_b = get_peft_model(model_b, lora_cfg)
        model_b.load_state_dict(init_state, strict=True)
        model_b.train()
        trainable_b = [p for p in model_b.parameters() if p.requires_grad]
        opt_b = torch.optim.AdamW(trainable_b, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        engine_b, _, _, _ = deepspeed.initialize(
            model=model_b, model_parameters=trainable_b,
            optimizer=opt_b, config=ds_config, dist_init_required=False,
        )

        # Build shim for both compute_loss and DeepSpeed boundary methods
        class _Shim:
            pass
        shim_b = _Shim()
        shim_b.model = engine_b
        shim_b.rank = 0
        shim_b._backward_call_count = 0
        shim_b.trainable_params = trainable_b
        shim_b.dist = dist
        shim_b.torch = torch

        # Bind real DeepSpeed production methods
        zero_grad = DeepSpeedStageRunner.zero_grad_for_step.__get__(shim_b)
        backward_loss = DeepSpeedStageRunner.backward_loss.__get__(shim_b)
        run_step = DeepSpeedStageRunner.run_optimizer_step.__get__(shim_b)
        shim_b._all_grad_tensors_norm = DeepSpeedStageRunner._all_grad_tensors_norm.__get__(shim_b)
        shim_b._grad_norm = DeepSpeedStageRunner._grad_norm.__get__(shim_b)

        # Also need compute_loss from StageRunnerBase
        shim_compute = object.__new__(StageRunnerBase)
        shim_compute.torch = torch
        shim_compute.model = engine_b
        shim_compute.input_device = torch.device("cuda:0")
        shim_compute.dtype = torch.float32
        shim_compute.logprob_compute_dtype = torch.float32
        shim_compute.pad_token_id = 0
        shim_compute.sequence_parallel_size = 1
        shim_compute.distributed_active = False
        shim_compute.cfg = type("Cfg", (), {{
            "reference_logprob_vocab_chunk_size": 4096,
            "backend_name": "{backend_name}",
            "use_remove_padding": {use_remove_padding},
        }})()

        zero_grad()
        for mb_idx in range(NUM_MICROBATCHES):
            is_last = (mb_idx == NUM_MICROBATCHES - 1)
            loss_b, _ = shim_compute.compute_loss([micro_samples[mb_idx]])
            backward_loss(loss_b, is_last=is_last)
        run_step()

        params_b = {{
            k: v.detach().cpu().clone()
            for k, v in engine_b.module.named_parameters()
            if v.requires_grad
        }}
        del engine_b, model_b, opt_b

        # ==== Compare ====
        mismatches = []
        max_abs_diff = 0.0
        compared = 0
        ABS_TOL = 1e-4  # fp32, should be very close
        for name in sorted(params_a.keys()):
            if name not in params_b:
                mismatches.append(f"missing: {{name}}")
                continue
            a = params_a[name].float()
            b = params_b[name].float()
            abs_diff = (a - b).abs().max().item()
            max_abs_diff = max(max_abs_diff, abs_diff)
            compared += 1
            if abs_diff > ABS_TOL:
                mismatches.append(f"{{name}}: abs_diff={{abs_diff:.2e}}")

        result = {{
            "compared": compared,
            "max_abs_diff": max_abs_diff,
            "mismatches": mismatches,
        }}
        print("RESULT:" + json.dumps(result))

        if len(mismatches) == 0:
            print("ALL_OK")
        else:
            print("FAILED")
        dist.destroy_process_group()
    """)
    stdout, stderr = _run_gpu_script(tmp_path, script)
    batch_mode = "packed" if use_packed else "padded"
    assert "ALL_OK" in stdout, (
        f"Microbatch equivalence ({batch_mode}) with real compute_loss failed.\n"
        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parsed = json.loads(line[len("RESULT:"):])
            print(f"  real compute_loss equivalence ({batch_mode}) OK: compared={parsed['compared']} "
                  f"max_abs_diff={parsed['max_abs_diff']:.2e}")
            break


# ===========================================================================
# End-to-end train_step test
# ===========================================================================

@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_train_step_end_to_end(tmp_path):
    """Full production train_step: split_for_rank → build_microbatches →
    compute_loss → backward_loss → run_optimizer_step.

    Calls the REAL StageRunnerBase.train_step (inherited by DeepSpeedStageRunner)
    with multiple LossSamples that get split into microbatches by the token budget.
    Verifies that:
      - params change after training (non-trivial update)
      - returned metrics are sane
      - the loss and token count are consistent
    """
    nanodiscover_root = str(Path(__file__).resolve().parent.parent)
    script = textwrap.dedent(f"""\
        import json
        import os
        import sys
        sys.path.insert(0, "{nanodiscover_root}")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29594")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoConfig
        from core.trainer import (
            LossSample, TrainerConfig, StageRunnerBase, DeepSpeedStageRunner,
            build_training_microbatches, shard_loss_samples_for_rank,
        )

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)

        # ---- Tiny model ----
        model_cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        model_cfg.num_hidden_layers = 2
        model_cfg.hidden_size = 64
        model_cfg.intermediate_size = 128
        model_cfg.num_attention_heads = 4
        model_cfg.num_key_value_heads = 2
        model_cfg.vocab_size = 256
        model_cfg.max_position_embeddings = 128

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
            lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )
        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": 0}},
        }}

        model = AutoModelForCausalLM.from_config(model_cfg).to(dtype=torch.float32, device="cuda:0")
        model = get_peft_model(model, lora_cfg)
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable,
            optimizer=opt, config=ds_config, dist_init_required=False,
        )

        # Snapshot params before training
        params_before = {{
            k: v.detach().cpu().clone()
            for k, v in engine.module.named_parameters()
            if v.requires_grad
        }}

        # ---- Create 8 LossSamples with varied lengths (forces multiple microbatches) ----
        torch.manual_seed(77)
        samples = []
        for i in range(8):
            seq_len = 8 + i * 3  # 8, 11, 14, 17, 20, 23, 26, 29
            input_ids = torch.randint(0, 256, (seq_len,)).tolist()
            target_ids = torch.randint(0, 256, (seq_len,)).tolist()
            old_lps = [-(torch.rand(1).item() * 2 + 0.1) for _ in range(seq_len)]
            prompt_len = 2
            mask = [0.0] * prompt_len + [1.0] * (seq_len - prompt_len)
            adv_val = 1.0 if i < 4 else -0.5
            advantages = [0.0] * prompt_len + [adv_val] * (seq_len - prompt_len)
            samples.append(LossSample(
                model_input_ids=input_ids,
                target_token_ids=target_ids,
                sampling_logprobs=old_lps,
                advantages=advantages,
                mask=mask,
                full_sequence_ids=input_ids + [target_ids[-1]],
                prompt_token_count=prompt_len,
                metadata={{}},
            ))

        # ---- Build a shim that has everything train_step needs ----
        class _Shim:
            pass
        shim = _Shim()
        shim.model = engine
        shim.rank = 0
        shim.world_size = 1
        shim.sequence_parallel_size = 1
        shim.distributed_active = False
        shim.torch = torch
        shim.dist = dist
        shim.input_device = torch.device("cuda:0")
        shim.dtype = torch.float32
        shim.logprob_compute_dtype = torch.float32
        shim.pad_token_id = 0
        shim.trainable_params = trainable
        shim.cfg = type("Cfg", (), {{
            "reference_logprob_vocab_chunk_size": 4096,
            "backend_name": "dry-run",
            "use_remove_padding": False,
            "trainer_max_tokens_per_rank": 64,  # small budget to force multiple microbatches
            "num_substeps": 1,
        }})()

        # Bind ALL production methods needed by train_step
        shim.split_for_rank = StageRunnerBase.split_for_rank.__get__(shim)
        shim.reduce_scalar_sum = StageRunnerBase.reduce_scalar_sum.__get__(shim)
        shim.reduce_scalar_max = StageRunnerBase.reduce_scalar_max.__get__(shim)
        shim.uses_packed_sequence_batch = StageRunnerBase.uses_packed_sequence_batch.__get__(shim)
        shim.build_loss_batch = StageRunnerBase.build_loss_batch.__get__(shim)
        shim.batch_tensors = StageRunnerBase.batch_tensors.__get__(shim)
        shim.pack_batch_tensors = StageRunnerBase.pack_batch_tensors.__get__(shim)
        shim.forward_loss_batch = StageRunnerBase.forward_loss_batch.__get__(shim)
        shim.slice_for_sequence_parallel = StageRunnerBase.slice_for_sequence_parallel.__get__(shim)
        shim.gather_batch_logprobs = StageRunnerBase.gather_batch_logprobs.__get__(shim)
        shim.compute_loss = StageRunnerBase.compute_loss.__get__(shim)
        shim.prepare_loss_for_backward = StageRunnerBase.prepare_loss_for_backward.__get__(shim)
        # Use DeepSpeed overrides for gradient handling
        shim.zero_grad_for_step = DeepSpeedStageRunner.zero_grad_for_step.__get__(shim)
        shim.backward_loss = DeepSpeedStageRunner.backward_loss.__get__(shim)
        shim.run_optimizer_step = DeepSpeedStageRunner.run_optimizer_step.__get__(shim)
        shim._all_grad_tensors_norm = DeepSpeedStageRunner._all_grad_tensors_norm.__get__(shim)
        shim._grad_norm = DeepSpeedStageRunner._grad_norm.__get__(shim)
        shim._backward_call_count = 0

        # Call the REAL train_step
        train_step = StageRunnerBase.train_step.__get__(shim)
        metrics = train_step(samples)

        # Snapshot params after
        params_after = {{
            k: v.detach().cpu().clone()
            for k, v in engine.module.named_parameters()
            if v.requires_grad
        }}

        # ---- Verify ----
        # 1. Params actually changed
        total_diff = 0.0
        compared = 0
        for name in params_before:
            diff = (params_after[name].float() - params_before[name].float()).abs().max().item()
            total_diff = max(total_diff, diff)
            compared += 1

        # 2. Metrics are sane
        loss = metrics.get("train/loss", None)
        optimizer_steps = metrics.get("train/optimizer_steps", None)
        microbatch_count = metrics.get("train/microbatch_count", None)

        result = {{
            "compared": compared,
            "max_param_diff": total_diff,
            "loss": loss,
            "optimizer_steps": optimizer_steps,
            "microbatch_count": microbatch_count,
            "metrics_keys": sorted(metrics.keys()),
        }}
        print("RESULT:" + json.dumps(result))

        errors = []
        if total_diff < 1e-8:
            errors.append("params did not change after train_step (trivial update)")
        if loss is None:
            errors.append("train/loss missing from metrics")
        if optimizer_steps != 1.0:
            errors.append("expected optimizer_steps=1, got " + str(optimizer_steps))
        if microbatch_count is None or microbatch_count < 2:
            errors.append("expected multiple microbatches (token budget=64), got " + str(microbatch_count))

        if errors:
            print("ERRORS:" + json.dumps(errors))
        else:
            print("ALL_OK")
        dist.destroy_process_group()
    """)
    stdout, stderr = _run_gpu_script(tmp_path, script)
    assert "ALL_OK" in stdout, (
        f"End-to-end train_step test failed.\n"
        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parsed = json.loads(line[len("RESULT:"):])
            print(f"  train_step e2e OK: compared={parsed['compared']} "
                  f"max_param_diff={parsed['max_param_diff']:.2e} "
                  f"loss={parsed['loss']:.6f} "
                  f"optimizer_steps={parsed['optimizer_steps']} "
                  f"microbatch_count={parsed['microbatch_count']}")
            break


# ===========================================================================
# Multi-GPU SP gradient correctness
# ===========================================================================

@pytest.mark.skipif(not _cuda_available(min_gpus=2), reason="Need >= 2 GPUs")
def test_sp2_train_step_matches_single_gpu_reference(tmp_path):
    """SP=2 on 2 GPUs must produce the same parameter update as 1 GPU on the full sequence.

    This tests prepare_loss_for_backward (all_gather+sum), the mpu-aware
    ZeRO divisor, and the full train_step pipeline under sequence parallelism.
    """
    nanodiscover_root = str(Path(__file__).resolve().parent.parent)

    # ---- Step 1: single-GPU reference ----
    ref_script = textwrap.dedent(f"""\
        import json
        import os
        import sys
        sys.path.insert(0, "{nanodiscover_root}")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29593")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoConfig
        from safetensors.torch import save_file
        from core.trainer import (
            LossSample, StageRunnerBase, DeepSpeedStageRunner,
        )

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)

        model_cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        model_cfg.num_hidden_layers = 2
        model_cfg.hidden_size = 64
        model_cfg.intermediate_size = 128
        model_cfg.num_attention_heads = 4
        model_cfg.num_key_value_heads = 2
        model_cfg.vocab_size = 256
        model_cfg.max_position_embeddings = 128

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
            lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )

        # Create model + save init weights for both paths
        # Use bf16 because flash attention (required by Ulysses SP) only supports fp16/bf16
        model = AutoModelForCausalLM.from_config(model_cfg).to(dtype=torch.bfloat16, device="cuda:0")
        model = get_peft_model(model, lora_cfg)
        init_state = {{k: v.clone() for k, v in model.state_dict().items()}}
        save_file(
            {{k: v.cpu() for k, v in init_state.items()}},
            os.path.join("{tmp_path}", "init_weights.safetensors"),
        )
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": 0}},
            "bf16": {{"enabled": True}},
        }}
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable,
            optimizer=opt, config=ds_config, dist_init_required=False,
        )

        # Create deterministic samples (seq_len must be divisible by SP=2)
        torch.manual_seed(55)
        samples = []
        for i in range(4):
            seq_len = 16 + i * 4  # 16, 20, 24, 28 — all divisible by 2
            input_ids = torch.randint(0, 256, (seq_len,)).tolist()
            target_ids = torch.randint(0, 256, (seq_len,)).tolist()
            old_lps = [-(torch.rand(1).item() * 2 + 0.1) for _ in range(seq_len)]
            prompt_len = 2
            mask = [0.0] * prompt_len + [1.0] * (seq_len - prompt_len)
            adv_val = 1.5 if i < 2 else -0.7
            advantages = [0.0] * prompt_len + [adv_val] * (seq_len - prompt_len)
            samples.append(LossSample(
                model_input_ids=input_ids,
                target_token_ids=target_ids,
                sampling_logprobs=old_lps,
                advantages=advantages,
                mask=mask,
                full_sequence_ids=input_ids + [target_ids[-1]],
                prompt_token_count=prompt_len,
                metadata={{}},
            ))

        # Save samples for the SP path
        torch.save(samples, os.path.join("{tmp_path}", "samples.pt"))

        # Build shim and run train_step on 1 GPU
        class _Shim:
            pass
        shim = _Shim()
        shim.model = engine
        shim.rank = 0
        shim.world_size = 1
        shim.sequence_parallel_size = 1
        shim.distributed_active = False
        shim.torch = torch
        shim.dist = dist
        shim.input_device = torch.device("cuda:0")
        shim.dtype = torch.bfloat16
        shim.logprob_compute_dtype = torch.float32
        shim.pad_token_id = 0
        shim.trainable_params = trainable
        shim._backward_call_count = 0
        shim.cfg = type("Cfg", (), {{
            "reference_logprob_vocab_chunk_size": 4096,
            "backend_name": "deepspeed",
            "use_remove_padding": True,
            "trainer_max_tokens_per_rank": None,
            "num_substeps": 1,
        }})()

        for method_name in [
            "split_for_rank", "reduce_scalar_sum", "reduce_scalar_max",
            "uses_packed_sequence_batch", "build_loss_batch", "batch_tensors",
            "pack_batch_tensors", "forward_loss_batch", "slice_for_sequence_parallel",
            "gather_batch_logprobs", "compute_loss", "prepare_loss_for_backward",
        ]:
            setattr(shim, method_name, getattr(StageRunnerBase, method_name).__get__(shim))
        shim.zero_grad_for_step = DeepSpeedStageRunner.zero_grad_for_step.__get__(shim)
        shim.backward_loss = DeepSpeedStageRunner.backward_loss.__get__(shim)
        shim.run_optimizer_step = DeepSpeedStageRunner.run_optimizer_step.__get__(shim)
        shim._all_grad_tensors_norm = DeepSpeedStageRunner._all_grad_tensors_norm.__get__(shim)
        shim._grad_norm = DeepSpeedStageRunner._grad_norm.__get__(shim)

        train_step = StageRunnerBase.train_step.__get__(shim)
        metrics = train_step(samples)

        # Save reference params
        ref_params = {{
            k: v.detach().cpu()
            for k, v in engine.module.named_parameters()
            if v.requires_grad
        }}
        save_file(ref_params, os.path.join("{tmp_path}", "ref_params.safetensors"))
        print("REF_LOSS:" + str(metrics["train/loss"]))
        print("ALL_OK")
        dist.destroy_process_group()
    """)

    ref_script_path = tmp_path / "ref_script.py"
    ref_script_path.write_text(ref_script)

    import os as _os
    clean_env = {k: v for k, v in _os.environ.items()
                 if k not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                              "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"}}
    clean_env["CUDA_VISIBLE_DEVICES"] = "0"
    ref_result = subprocess.run(
        [sys.executable, str(ref_script_path)],
        capture_output=True, text=True, timeout=180, env=clean_env,
    )
    assert ref_result.returncode == 0 and "ALL_OK" in ref_result.stdout, (
        f"Reference 1-GPU run failed.\nSTDOUT:\n{ref_result.stdout}\nSTDERR:\n{ref_result.stderr}"
    )

    # ---- Step 2: SP=2 on 2 GPUs ----
    sp_script = textwrap.dedent(f"""\
        import json
        import os
        import sys
        sys.path.insert(0, "{nanodiscover_root}")

        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoConfig
        from safetensors.torch import load_file, save_file
        from core.trainer import (
            LossSample, TrainerConfig, StageRunnerBase, DeepSpeedStageRunner,
        )

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        if not deepspeed.comm.is_initialized():
            deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

        model_cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        model_cfg.num_hidden_layers = 2
        model_cfg.hidden_size = 64
        model_cfg.intermediate_size = 128
        model_cfg.num_attention_heads = 4
        model_cfg.num_key_value_heads = 2
        model_cfg.vocab_size = 256
        model_cfg.max_position_embeddings = 128

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
            lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )

        # Load same init weights
        model = AutoModelForCausalLM.from_config(model_cfg).to(
            dtype=torch.bfloat16, device=f"cuda:{{local_rank}}"
        )
        model = get_peft_model(model, lora_cfg)
        init_weights = load_file(os.path.join("{tmp_path}", "init_weights.safetensors"))
        model.load_state_dict(
            {{k: v.to(f"cuda:{{local_rank}}") for k, v in init_weights.items()}},
            strict=True,
        )
        model.train()

        # Enable Ulysses SP=2
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu_module
        from core.trainer import ensure_flash_attention_2_ready_for_deepspeed

        ensure_flash_attention_2_ready_for_deepspeed()
        if hasattr(model, "config"):
            model.config._attn_implementation = "flash_attention_2"
        ulysses_mpu = UlyssesSPAttentionHF.register_with_transformers(
            model,
            core_attn_implementation="flash_attention_2",
            sequence_parallel_size=2,
            micro_batch_size=1,
            seq_length_is_variable=True,
            disable_in_eval=True,
        )
        sp_group = ulysses_mpu.get_sequence_parallel_group()
        sp_rank = ulysses_mpu.get_sequence_parallel_rank()

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": 2}},
            "bf16": {{"enabled": True}},
        }}
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable,
            optimizer=opt, config=ds_config, dist_init_required=False,
            mpu=ulysses_mpu,
        )

        # Load same samples
        samples = torch.load(os.path.join("{tmp_path}", "samples.pt"), weights_only=False)

        # Build shim with SP=2
        import torch.distributed.nn.functional as dist_nn

        class _Shim:
            pass
        shim = _Shim()
        shim.model = engine
        shim.rank = rank
        shim.world_size = world_size
        shim.sequence_parallel_size = 2
        shim.distributed_active = True
        shim.torch = torch
        shim.dist = dist
        shim.dist_nn = dist_nn
        shim.F = torch.nn.functional
        shim.input_device = torch.device(f"cuda:{{local_rank}}")
        shim.dtype = torch.bfloat16
        shim.logprob_compute_dtype = torch.float32
        shim.pad_token_id = 0
        shim.trainable_params = trainable
        shim._backward_call_count = 0
        shim.sequence_parallel_group = sp_group
        shim.sequence_parallel_rank = sp_rank
        shim._ulysses_mpu = ulysses_mpu
        shim.cfg = type("Cfg", (), {{
            "reference_logprob_vocab_chunk_size": 4096,
            "backend_name": "deepspeed",
            "use_remove_padding": True,
            "trainer_max_tokens_per_rank": None,
            "num_substeps": 1,
        }})()

        for method_name in [
            "split_for_rank", "reduce_scalar_sum", "reduce_scalar_max",
            "uses_packed_sequence_batch", "build_loss_batch", "batch_tensors",
            "pack_batch_tensors", "forward_loss_batch", "slice_for_sequence_parallel",
            "sequence_parallel_group_rank",
            "gather_batch_logprobs", "compute_loss",
        ]:
            setattr(shim, method_name, getattr(StageRunnerBase, method_name).__get__(shim))
        shim.prepare_loss_for_backward = DeepSpeedStageRunner.prepare_loss_for_backward.__get__(shim)
        shim.zero_grad_for_step = DeepSpeedStageRunner.zero_grad_for_step.__get__(shim)
        shim.backward_loss = DeepSpeedStageRunner.backward_loss.__get__(shim)
        shim.run_optimizer_step = DeepSpeedStageRunner.run_optimizer_step.__get__(shim)
        shim._all_grad_tensors_norm = DeepSpeedStageRunner._all_grad_tensors_norm.__get__(shim)
        shim._grad_norm = DeepSpeedStageRunner._grad_norm.__get__(shim)

        train_step = StageRunnerBase.train_step.__get__(shim)
        metrics = train_step(samples)

        # Gather params to rank 0 for comparison
        sp_params = {{
            k: v.detach().cpu()
            for k, v in engine.module.named_parameters()
            if v.requires_grad
        }}

        if rank == 0:
            ref_params = load_file(os.path.join("{tmp_path}", "ref_params.safetensors"))
            mismatches = []
            max_abs_diff = 0.0
            compared = 0
            ABS_TOL = 1e-3  # float32 but different computation order across ranks
            for name in sorted(ref_params.keys()):
                if name not in sp_params:
                    mismatches.append("missing: " + name)
                    continue
                a = ref_params[name].float()
                b = sp_params[name].float()
                abs_diff = (a - b).abs().max().item()
                max_abs_diff = max(max_abs_diff, abs_diff)
                compared += 1
                if abs_diff > ABS_TOL:
                    mismatches.append(name + ": abs_diff=" + str(abs_diff))

            result = {{
                "compared": compared,
                "max_abs_diff": max_abs_diff,
                "mismatches": mismatches,
                "sp_loss": metrics["train/loss"],
            }}
            print("RESULT:" + json.dumps(result))
            if len(mismatches) == 0:
                print("ALL_OK")
            else:
                print("FAILED")

        dist.destroy_process_group()
    """)

    sp_script_path = tmp_path / "sp_script.py"
    sp_script_path.write_text(sp_script)

    clean_env2 = {k: v for k, v in _os.environ.items()
                  if k not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                               "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"}}
    clean_env2["CUDA_VISIBLE_DEVICES"] = "0,1"
    sp_result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--standalone", "--nproc_per_node", "2",
         str(sp_script_path)],
        capture_output=True, text=True, timeout=180, env=clean_env2,
    )
    assert sp_result.returncode == 0 and "ALL_OK" in sp_result.stdout, (
        f"SP=2 run failed.\nSTDOUT:\n{sp_result.stdout}\nSTDERR:\n{sp_result.stderr}"
    )
    for line in sp_result.stdout.splitlines():
        if line.startswith("RESULT:"):
            parsed = json.loads(line[len("RESULT:"):])
            print(f"  SP=2 vs 1-GPU OK: compared={parsed['compared']} "
                  f"max_abs_diff={parsed['max_abs_diff']:.2e} "
                  f"sp_loss={parsed['sp_loss']:.6f}")
            break
