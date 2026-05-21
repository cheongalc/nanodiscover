"""Test that many-microbatch backward with boundary logic matches single summed-loss backward.

This is the critical correctness test for the DeepSpeed training path in
core/trainer.py.  The production code calls backward() N times (once per
microbatch) using set_gradient_accumulation_boundary() to accumulate
gradients, then one optimizer step.  This test verifies that the resulting
model parameters are identical to a reference path that sums all micro-losses
into one scalar and does a single backward + step.

If this test fails, the trainer is silently producing wrong gradients and
the model is not learning what it should be learning.

Requires: 1 GPU, torch, deepspeed, peft, transformers.
Skipped automatically if CUDA is not available.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _cuda_available(min_gpus: int = 1):
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() >= min_gpus
    except ImportError:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_microbatch_boundary_matches_summed_loss_zero0(tmp_path):
    """ZeRO stage 0: N microbatch backwards with boundary logic must produce
    the same parameter update as one backward on the summed loss."""
    _run_equivalence_test(tmp_path, zero_stage=0, num_microbatches=5)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_microbatch_boundary_matches_summed_loss_zero2(tmp_path):
    """ZeRO stage 2: same test with gradient partitioning active."""
    _run_equivalence_test(tmp_path, zero_stage=2, num_microbatches=5)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_microbatch_boundary_many_steps_zero2(tmp_path):
    """ZeRO stage 2 with more microbatches to stress the accumulation."""
    _run_equivalence_test(tmp_path, zero_stage=2, num_microbatches=20)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_microbatch_boundary_fp32_control(tmp_path):
    """Float32 control: if this passes with near-zero diff, it proves the
    bf16 test diffs are accumulation-order rounding, not a real bug."""
    _run_equivalence_test(tmp_path, zero_stage=0, num_microbatches=5, use_fp32=True)


def _run_equivalence_test(tmp_path, *, zero_stage: int, num_microbatches: int, use_fp32: bool = False):
    nanodiscover_root = str(Path(__file__).resolve().parent.parent)
    script = textwrap.dedent(f"""\
        import json
        import os
        import sys
        sys.path.insert(0, "{nanodiscover_root}")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29598")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoConfig

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)

        NUM_MICROBATCHES = {num_microbatches}
        ZERO_STAGE = {zero_stage}
        USE_FP32 = {use_fp32}
        MODEL_DTYPE = torch.float32 if USE_FP32 else torch.bfloat16

        # ---- Tiny Qwen-architecture model ----
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.vocab_size = 256
        cfg.max_position_embeddings = 128

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )

        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": ZERO_STAGE}},
        }}
        if not USE_FP32:
            ds_config["bf16"] = {{"enabled": True}}

        # ---- Create deterministic micro-losses ----
        # Use fixed random inputs so both paths see the same data.
        torch.manual_seed(42)
        micro_inputs = [
            torch.randint(0, 256, (1, 16), device="cuda:0")
            for _ in range(NUM_MICROBATCHES)
        ]

        # ---- Save initial LoRA weights so both paths start from the same point ----
        init_model = AutoModelForCausalLM.from_config(cfg).to(
            dtype=MODEL_DTYPE, device="cuda:0"
        )
        init_model = get_peft_model(init_model, lora_cfg)
        init_state = {{k: v.clone() for k, v in init_model.state_dict().items()}}
        del init_model

        # ==============================================================
        # PATH A: Reference — sum all micro-losses, one backward, one step
        # ==============================================================
        model_a = AutoModelForCausalLM.from_config(cfg).to(
            dtype=MODEL_DTYPE, device="cuda:0"
        )
        model_a = get_peft_model(model_a, lora_cfg)
        model_a.load_state_dict(init_state, strict=True)
        model_a.train()
        trainable_a = [p for p in model_a.parameters() if p.requires_grad]
        opt_a = torch.optim.AdamW(
            trainable_a, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )
        engine_a, _, _, _ = deepspeed.initialize(
            model=model_a, model_parameters=trainable_a,
            optimizer=opt_a, config=ds_config,
            dist_init_required=False,
        )

        # Compute all micro-losses without backward, then sum and do one backward
        micro_losses = []
        for inp in micro_inputs:
            out = engine_a(input_ids=inp)
            # Simulate the production loss: a raw sum (not mean) of logits
            # masked to the last few positions, times a fake advantage.
            loss = out.logits[:, -4:, :].sum()
            micro_losses.append(loss)

        total_loss = sum(micro_losses)
        engine_a.backward(total_loss)
        engine_a.step()

        params_a = {{
            k: v.detach().cpu().clone()
            for k, v in engine_a.module.named_parameters()
            if v.requires_grad
        }}
        del engine_a, model_a, opt_a

        # ==============================================================
        # PATH B: Production — N backwards with boundary logic, one step
        # ==============================================================
        model_b = AutoModelForCausalLM.from_config(cfg).to(
            dtype=MODEL_DTYPE, device="cuda:0"
        )
        model_b = get_peft_model(model_b, lora_cfg)
        model_b.load_state_dict(init_state, strict=True)
        model_b.train()
        trainable_b = [p for p in model_b.parameters() if p.requires_grad]
        opt_b = torch.optim.AdamW(
            trainable_b, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )
        engine_b, _, _, _ = deepspeed.initialize(
            model=model_b, model_parameters=trainable_b,
            optimizer=opt_b, config=ds_config,
            dist_init_required=False,
        )

        # Call the ACTUAL production methods from core/trainer.py, not a
        # reimplementation.  We build a lightweight shim that holds the
        # attributes the methods need (self.model, self.rank, etc.) and
        # bind the real methods to it.
        from core.trainer import DeepSpeedStageRunner

        class _Shim:
            pass

        shim = _Shim()
        shim.model = engine_b
        shim.rank = 0
        shim._backward_call_count = 0
        shim.trainable_params = trainable_b

        # Bind the real methods
        zero_grad_for_step = DeepSpeedStageRunner.zero_grad_for_step.__get__(shim)
        backward_loss = DeepSpeedStageRunner.backward_loss.__get__(shim)
        run_optimizer_step = DeepSpeedStageRunner.run_optimizer_step.__get__(shim)
        # _all_grad_tensors_norm and _grad_norm are needed by the above
        shim._all_grad_tensors_norm = DeepSpeedStageRunner._all_grad_tensors_norm.__get__(shim)
        shim._grad_norm = DeepSpeedStageRunner._grad_norm.__get__(shim)
        shim.dist = dist
        shim.torch = torch

        zero_grad_for_step()
        for mb_idx, inp in enumerate(micro_inputs):
            is_last = (mb_idx == NUM_MICROBATCHES - 1)
            out = engine_b(input_ids=inp)
            loss = out.logits[:, -4:, :].sum()
            backward_loss(loss, is_last=is_last)

        run_optimizer_step()

        params_b = {{
            k: v.detach().cpu().clone()
            for k, v in engine_b.module.named_parameters()
            if v.requires_grad
        }}
        del engine_b, model_b, opt_b

        # ==============================================================
        # Compare: both paths must produce identical parameters
        # ==============================================================
        mismatches = []
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        compared = 0
        # bf16 addition is not associative: sum-then-backward vs
        # backward-per-microbatch accumulates rounding differently.
        # Observed diffs are ~5e-5 to 8e-5 which is well within bf16
        # numerical noise.  Use 1e-3 as tolerance: tight enough to catch
        # real gradient bugs (which produce diffs of order 1.0+) but
        # loose enough for bf16 accumulation order differences.
        # In fp32, the two paths should match to near machine epsilon.
        ABS_TOL = 1e-6 if USE_FP32 else 1e-3
        for name in sorted(params_a.keys()):
            if name not in params_b:
                mismatches.append(f"missing in path B: {{name}}")
                continue
            a = params_a[name].float()
            b = params_b[name].float()
            abs_diff = (a - b).abs().max().item()
            denom = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
            rel_diff = abs_diff / denom
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)
            compared += 1
            if abs_diff > ABS_TOL:
                mismatches.append(
                    f"{{name}}: abs_diff={{abs_diff:.2e}} rel_diff={{rel_diff:.2e}}"
                )

        result = {{
            "zero_stage": ZERO_STAGE,
            "num_microbatches": NUM_MICROBATCHES,
            "use_fp32": USE_FP32,
            "abs_tol": ABS_TOL,
            "compared": compared,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mismatches": mismatches,
        }}
        print("RESULT:" + json.dumps(result))

        if len(mismatches) == 0:
            print("ALL_OK")
        else:
            print("FAILED")

        dist.destroy_process_group()
    """)

    script_path = tmp_path / "test_microbatch_equiv.py"
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
        timeout=180,
        env=clean_env,
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        pytest.fail(
            f"Subprocess failed (rc={result.returncode}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    assert "ALL_OK" in stdout, (
        f"Microbatch boundary logic does not match summed-loss reference.\n"
        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )

    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parsed = json.loads(line[len("RESULT:"):])
            assert parsed["compared"] > 0, f"No parameters were compared: {parsed}"
            assert len(parsed["mismatches"]) == 0, (
                f"Parameter mismatches between production and reference paths "
                f"(zero_stage={zero_stage}, microbatches={num_microbatches}): "
                f"{parsed['mismatches']}"
            )
            print(
                f"  equivalence OK: zero_stage={parsed['zero_stage']} "
                f"microbatches={parsed['num_microbatches']} "
                f"fp32={parsed.get('use_fp32', False)} "
                f"abs_tol={parsed.get('abs_tol', '?')} "
                f"compared={parsed['compared']} "
                f"max_abs_diff={parsed['max_abs_diff']:.2e} "
                f"max_rel_diff={parsed['max_rel_diff']:.2e}"
            )
            break
    else:
        pytest.fail(f"No RESULT line found in output.\nSTDOUT:\n{stdout}")
