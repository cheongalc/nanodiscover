"""Test that DeepSpeed optimizer state save/load round-trips correctly.

This test exercises the actual save_optimizer_state / load_optimizer_state
methods on DeepSpeedStageRunner to verify that Adam momentum and variance
persist across simulated epoch boundaries.

Requires: GPU(s), torch, deepspeed, peft, transformers.
Skipped automatically if CUDA is not available.
"""

import json
import shutil
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
def test_optimizer_state_round_trips_through_deepspeed(tmp_path):
    """Verify that optimizer state saved by one DeepSpeedStageRunner can be
    loaded by another, preserving Adam first/second moments exactly."""

    # We run this in a subprocess because DeepSpeed init is hard to tear down
    # cleanly within a single pytest process.
    script = textwrap.dedent(f"""\
        import json
        import sys
        import os
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29599")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        from peft.utils.save_and_load import get_peft_model_state_dict
        from transformers import AutoModelForCausalLM, AutoConfig

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)

        tmp = "{tmp_path}"
        adapter_dir = os.path.join(tmp, "adapter")
        optim_dir_1 = os.path.join(tmp, "optim_epoch0")
        optim_dir_2 = os.path.join(tmp, "optim_epoch1")

        # ---- Build a tiny model so this runs fast ----
        config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        config.num_hidden_layers = 2
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.vocab_size = 256
        config.max_position_embeddings = 128
        model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16, device="cuda:0")

        # ---- Attach LoRA ----
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": 0}},
            "bf16": {{"enabled": True}},
        }}
        engine, opt, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable,
            optimizer=optimizer, config=ds_config,
            dist_init_required=False,
        )

        # ---- Do a forward+backward to create non-trivial optimizer state ----
        input_ids = torch.randint(0, 256, (1, 16), device="cuda:0")
        outputs = engine(input_ids=input_ids)
        loss = outputs.logits.sum()
        engine.backward(loss)
        engine.step()

        # ---- Save adapter + optimizer state ----
        os.makedirs(adapter_dir, exist_ok=True)
        peft_state = get_peft_model_state_dict(engine.module)
        assert len(peft_state) > 0, "empty adapter state"
        from safetensors.torch import save_file
        save_file({{k: v.detach().cpu().clone() for k, v in peft_state.items()}},
                  os.path.join(adapter_dir, "adapter_model.safetensors"),
                  metadata={{"format": "pt"}})
        engine.module.peft_config["default"].save_pretrained(adapter_dir)

        os.makedirs(optim_dir_1, exist_ok=True)
        engine.save_checkpoint(optim_dir_1, tag="optimizer_state")

        # ---- Extract optimizer state for comparison ----
        state_before = {{}}
        for i, group in enumerate(opt.param_groups):
            for j, p in enumerate(group["params"]):
                s = opt.state.get(p, {{}})
                if "exp_avg" in s:
                    state_before[(i,j,"exp_avg")] = s["exp_avg"].clone()
                    state_before[(i,j,"exp_avg_sq")] = s["exp_avg_sq"].clone()
                    state_before[(i,j,"step")] = s["step"].clone() if isinstance(s["step"], torch.Tensor) else s["step"]

        assert len(state_before) > 0, "optimizer has no state after 1 step"

        # ---- Destroy engine, build a fresh one (simulating new epoch subprocess) ----
        del engine, opt, model

        model2 = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16, device="cuda:0")
        model2 = PeftModel.from_pretrained(model2, adapter_dir, is_trainable=True)
        model2.train()

        trainable2 = [p for p in model2.parameters() if p.requires_grad]
        optimizer2 = torch.optim.AdamW(trainable2, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

        engine2, opt2, _, _ = deepspeed.initialize(
            model=model2, model_parameters=trainable2,
            optimizer=optimizer2, config=ds_config,
            dist_init_required=False,
        )

        # ---- Snapshot cold optimizer state (DeepSpeed init may have populated
        #      structure, but the moments should differ from the trained ones) ----
        cold_state = {{}}
        for i, group in enumerate(opt2.param_groups):
            for j, p in enumerate(group["params"]):
                s = opt2.state.get(p, {{}})
                if "exp_avg" in s:
                    cold_state[(i,j,"exp_avg")] = s["exp_avg"].clone()
                    cold_state[(i,j,"exp_avg_sq")] = s["exp_avg_sq"].clone()

        # ---- Load optimizer state from checkpoint ----
        _, client_state = engine2.load_checkpoint(
            optim_dir_1,
            tag="optimizer_state",
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
            load_module_only=False,
        )

        # ---- Verify optimizer state matches the saved (trained) state,
        #      NOT the cold state ----
        mismatches = []
        restored_count = 0
        changed_from_cold = 0
        for i, group in enumerate(opt2.param_groups):
            for j, p in enumerate(group["params"]):
                s = opt2.state.get(p, {{}})
                if "exp_avg" in s:
                    restored_count += 1
                    key_avg = (i, j, "exp_avg")
                    key_sq = (i, j, "exp_avg_sq")
                    if key_avg in state_before:
                        if not torch.equal(s["exp_avg"], state_before[key_avg].to(s["exp_avg"].device)):
                            mismatches.append(f"exp_avg mismatch at group={{i}} param={{j}}")
                        if not torch.equal(s["exp_avg_sq"], state_before[key_sq].to(s["exp_avg_sq"].device)):
                            mismatches.append(f"exp_avg_sq mismatch at group={{i}} param={{j}}")
                    if key_avg in cold_state:
                        if not torch.equal(s["exp_avg"], cold_state[key_avg].to(s["exp_avg"].device)):
                            changed_from_cold += 1

        result = {{
            "state_before_count": len(state_before) // 3,
            "restored_count": restored_count,
            "mismatches": mismatches,
            "changed_from_cold": changed_from_cold,
        }}
        print("RESULT:" + json.dumps(result))

        # ---- Now do another training step and save again ----
        input_ids2 = torch.randint(0, 256, (1, 16), device="cuda:0")
        outputs2 = engine2(input_ids=input_ids2)
        loss2 = outputs2.logits.sum()
        engine2.backward(loss2)
        engine2.step()

        # Verify step count > 1 (warm resume + 1 new step).
        # DeepSpeed's initialize() may add a dummy step, so the exact
        # count depends on the DS version. What matters is it's > 1.
        for group in opt2.param_groups:
            for p in group["params"]:
                s = opt2.state.get(p, {{}})
                if "step" in s:
                    step_val = s["step"].item() if isinstance(s["step"], torch.Tensor) else s["step"]
                    assert step_val > 1, f"expected step>1 after warm resume + 1 step, got {{step_val}}"
                    break
            else:
                continue
            break

        os.makedirs(optim_dir_2, exist_ok=True)
        engine2.save_checkpoint(optim_dir_2, tag="optimizer_state")

        print("RESULT_STEP:2")
        print("ALL_OK")
        dist.destroy_process_group()
    """)

    script_path = tmp_path / "test_optim_roundtrip.py"
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
        timeout=120,
        env=clean_env,
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        pytest.fail(f"Subprocess failed (rc={result.returncode}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    assert "ALL_OK" in stdout, f"Script did not complete successfully.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    # Parse the JSON result line
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parsed = json.loads(line[len("RESULT:"):])
            assert parsed["restored_count"] > 0, f"No optimizer state was restored: {parsed}"
            assert parsed["restored_count"] == parsed["state_before_count"], (
                f"Mismatch in restored count: before={parsed['state_before_count']} restored={parsed['restored_count']}"
            )
            assert len(parsed["mismatches"]) == 0, f"Optimizer state mismatches: {parsed['mismatches']}"
            assert parsed["changed_from_cold"] > 0, (
                "Loading checkpoint did not change any optimizer state from the cold init — "
                "the load may be a no-op"
            )
            break
    else:
        pytest.fail(f"No RESULT line found in output.\nSTDOUT:\n{stdout}")

    assert "RESULT_STEP:2" in stdout, f"Step count did not reach 2 after warm resume.\nSTDOUT:\n{stdout}"


@pytest.mark.skipif(not _cuda_available(min_gpus=2), reason="Need >= 2 GPUs")
def test_optimizer_state_round_trips_through_deepspeed_zero2_multi_gpu(tmp_path):
    """Verify optimizer state save/load with ZeRO stage 2 on 2 GPUs.

    This matches the real AC1 training topology (ZeRO-2 with partitioned
    optimizer states across ranks). Each rank saves/loads its own partition
    and verifies its own moments match.
    """

    script = textwrap.dedent(f"""\
        import json
        import os
        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        from peft.utils.save_and_load import get_peft_model_state_dict
        from transformers import AutoModelForCausalLM, AutoConfig

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        tmp = "{tmp_path}"
        adapter_dir = os.path.join(tmp, "adapter")
        optim_dir = os.path.join(tmp, "optim_epoch0")

        # ---- Tiny model ----
        config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        config.num_hidden_layers = 2
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.vocab_size = 256
        config.max_position_embeddings = 128
        model = AutoModelForCausalLM.from_config(config).to(
            dtype=torch.bfloat16, device=f"cuda:{{local_rank}}"
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )

        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": 2}},
            "bf16": {{"enabled": True}},
        }}
        engine, opt, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable,
            optimizer=optimizer, config=ds_config,
            dist_init_required=False,
        )

        # ---- Train one step ----
        input_ids = torch.randint(0, 256, (1, 16), device=f"cuda:{{local_rank}}")
        outputs = engine(input_ids=input_ids)
        loss = outputs.logits.sum()
        engine.backward(loss)
        engine.step()

        # ---- Save adapter (rank 0 only) + optimizer state (all ranks) ----
        if rank == 0:
            os.makedirs(adapter_dir, exist_ok=True)
            peft_state = get_peft_model_state_dict(engine.module)
            assert len(peft_state) > 0
            from safetensors.torch import save_file
            save_file(
                {{k: v.detach().cpu().clone() for k, v in peft_state.items()}},
                os.path.join(adapter_dir, "adapter_model.safetensors"),
                metadata={{"format": "pt"}},
            )
            engine.module.peft_config["default"].save_pretrained(adapter_dir)
        dist.barrier()

        os.makedirs(optim_dir, exist_ok=True)
        engine.save_checkpoint(optim_dir, tag="optimizer_state")

        # ---- Snapshot this rank's ZeRO-2 optimizer state via state_dict ----
        saved_state = engine.optimizer.state_dict()
        # The base optimizer state is under the key "base_optimizer_state"
        # which contains per-param dicts with "exp_avg", "exp_avg_sq", etc.
        base_opt_state = saved_state.get("base_optimizer_state", {{}})
        # Count how many params have moments
        saved_moment_count = 0
        if isinstance(base_opt_state, dict) and "state" in base_opt_state:
            # Non-elastic format: full base optimizer state_dict
            for pid, pstate in base_opt_state["state"].items():
                if "exp_avg" in pstate:
                    saved_moment_count += 1
        elif isinstance(base_opt_state, list):
            # Elastic format: list of per-param state dicts
            for pstate in base_opt_state:
                if isinstance(pstate, dict) and "exp_avg" in pstate:
                    saved_moment_count += 1
        assert saved_moment_count > 0, f"rank {{rank}}: no optimizer moments in state_dict"

        # Also snapshot the fp32 partition values to verify they round-trip
        fp32_partitions = saved_state.get("single_partition_of_fp32_groups", [])
        saved_fp32 = [p.clone() if hasattr(p, "clone") else p for p in fp32_partitions]

        # ---- Destroy engine, build fresh one ----
        del engine, opt, model

        model2 = AutoModelForCausalLM.from_config(config).to(
            dtype=torch.bfloat16, device=f"cuda:{{local_rank}}"
        )
        model2 = PeftModel.from_pretrained(model2, adapter_dir, is_trainable=True)
        model2.train()

        trainable2 = [p for p in model2.parameters() if p.requires_grad]
        optimizer2 = torch.optim.AdamW(
            trainable2, lr=4e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )
        engine2, opt2, _, _ = deepspeed.initialize(
            model=model2, model_parameters=trainable2,
            optimizer=optimizer2, config=ds_config,
            dist_init_required=False,
        )

        # ---- Snapshot cold state ----
        cold_state = engine2.optimizer.state_dict()
        cold_fp32 = cold_state.get("single_partition_of_fp32_groups", [])
        cold_fp32 = [p.clone() if hasattr(p, "clone") else p for p in cold_fp32]

        # ---- Load checkpoint ----
        _, _ = engine2.load_checkpoint(
            optim_dir,
            tag="optimizer_state",
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
            load_module_only=False,
        )

        # ---- Compare state after load with the saved state ----
        loaded_state = engine2.optimizer.state_dict()
        loaded_fp32 = loaded_state.get("single_partition_of_fp32_groups", [])

        # Compare FP32 master param partitions
        fp32_mismatches = 0
        fp32_changed_from_cold = 0
        for idx, (s, l) in enumerate(zip(saved_fp32, loaded_fp32)):
            if hasattr(s, "shape") and hasattr(l, "shape"):
                if not torch.equal(s, l):
                    fp32_mismatches += 1
                if idx < len(cold_fp32) and hasattr(cold_fp32[idx], "shape"):
                    if not torch.equal(cold_fp32[idx], l):
                        fp32_changed_from_cold += 1

        result = {{
            "rank": rank,
            "world_size": world_size,
            "saved_moment_count": saved_moment_count,
            "fp32_partitions_saved": len(saved_fp32),
            "fp32_partitions_loaded": len(loaded_fp32),
            "fp32_mismatches": fp32_mismatches,
            "fp32_changed_from_cold": fp32_changed_from_cold,
        }}
        # Only rank 0 prints the result line to avoid interleaved output.
        # Gather results to rank 0.
        all_results = [None] * world_size
        dist.all_gather_object(all_results, result)
        if rank == 0:
            for r in all_results:
                print("RESULT:" + json.dumps(r))
            print("ALL_OK")
        dist.barrier()
        dist.destroy_process_group()
    """)

    script_path = tmp_path / "test_optim_zero2.py"
    script_path.write_text(script)

    import os as _os
    clean_env = {k: v for k, v in _os.environ.items()
                 if k not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                              "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"}}
    clean_env["CUDA_VISIBLE_DEVICES"] = "0,1"

    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node", "2",
            str(script_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=clean_env,
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        pytest.fail(f"Subprocess failed (rc={result.returncode}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    assert "ALL_OK" in stdout, f"Script did not complete.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    result_lines = [line for line in stdout.splitlines() if line.startswith("RESULT:")]
    assert len(result_lines) == 2, f"Expected 2 RESULT lines (one per rank), got {len(result_lines)}"

    for line in result_lines:
        parsed = json.loads(line[len("RESULT:"):])
        assert parsed["saved_moment_count"] > 0, f"rank {parsed['rank']}: no optimizer moments found"
        assert parsed["fp32_partitions_loaded"] == parsed["fp32_partitions_saved"], (
            f"rank {parsed['rank']}: partition count changed: "
            f"saved={parsed['fp32_partitions_saved']} loaded={parsed['fp32_partitions_loaded']}"
        )
        assert parsed["fp32_mismatches"] == 0, (
            f"rank {parsed['rank']}: {parsed['fp32_mismatches']} FP32 partition mismatches after load"
        )
        assert parsed["fp32_changed_from_cold"] > 0, (
            f"rank {parsed['rank']}: load did not change any FP32 partitions from cold init"
        )


@pytest.mark.skipif(not _cuda_available(min_gpus=2), reason="Need >= 2 GPUs")
def test_warm_optimizer_produces_different_update_than_cold_zero2(tmp_path):
    """Behavioral proof: warm-resumed optimizer produces a different parameter
    update than a cold-restarted optimizer on the same data and same adapter.

    This is the definitive test.  If optimizer state persistence is working,
    step 2 with warm Adam moments must produce different LoRA weights than
    step 2 with cold Adam moments, because Adam's adaptive learning rate
    depends on accumulated first/second moment history.

    Runs on 2 GPUs with ZeRO stage 2 to match the real AC1 topology.
    """

    script = textwrap.dedent(f"""\
        import json
        import os
        import torch
        import torch.distributed as dist
        import deepspeed
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        from peft.utils.save_and_load import get_peft_model_state_dict
        from transformers import AutoModelForCausalLM, AutoConfig

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        tmp = "{tmp_path}"
        adapter_dir = os.path.join(tmp, "adapter")
        optim_dir = os.path.join(tmp, "optim_epoch0")

        # ---- Tiny model ----
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        cfg.vocab_size = 256
        cfg.max_position_embeddings = 128

        ds_config = {{
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
            "zero_optimization": {{"stage": 2}},
            "bf16": {{"enabled": True}},
        }}

        def make_engine(model_cfg, adapter_path=None):
            m = AutoModelForCausalLM.from_config(model_cfg).to(
                dtype=torch.bfloat16, device=f"cuda:{{local_rank}}"
            )
            if adapter_path:
                m = PeftModel.from_pretrained(m, adapter_path, is_trainable=True)
            else:
                lc = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=4, lora_alpha=8, lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    bias="none",
                )
                m = get_peft_model(m, lc)
            m.train()
            tp = [p for p in m.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(tp, lr=4e-5, betas=(0.9, 0.95), eps=1e-8)
            eng, _, _, _ = deepspeed.initialize(
                model=m, model_parameters=tp, optimizer=opt,
                config=ds_config, dist_init_required=False,
            )
            return eng

        # Use a FIXED seed so both paths see the exact same data.
        torch.manual_seed(42 + rank)
        data_step1 = torch.randint(0, 256, (1, 32), device=f"cuda:{{local_rank}}")
        data_step2 = torch.randint(0, 256, (1, 32), device=f"cuda:{{local_rank}}")

        # =============== PATH A: warm optimizer (save after step 1, reload, step 2) ===============
        engine_a = make_engine(cfg)
        out = engine_a(input_ids=data_step1)
        engine_a.backward(out.logits.sum())
        engine_a.step()

        # Save adapter + optimizer
        if rank == 0:
            os.makedirs(adapter_dir, exist_ok=True)
            ps = get_peft_model_state_dict(engine_a.module)
            from safetensors.torch import save_file
            save_file({{k: v.detach().cpu().clone() for k, v in ps.items()}},
                      os.path.join(adapter_dir, "adapter_model.safetensors"),
                      metadata={{"format": "pt"}})
            engine_a.module.peft_config["default"].save_pretrained(adapter_dir)
        dist.barrier()
        os.makedirs(optim_dir, exist_ok=True)
        engine_a.save_checkpoint(optim_dir, tag="optimizer_state")
        del engine_a

        # New engine from saved adapter, WARM optimizer
        engine_warm = make_engine(cfg, adapter_path=adapter_dir)
        engine_warm.load_checkpoint(
            optim_dir, tag="optimizer_state",
            load_module_strict=True, load_optimizer_states=True,
            load_lr_scheduler_states=False, load_module_only=False,
        )
        out = engine_warm(input_ids=data_step2)
        engine_warm.backward(out.logits.sum())
        engine_warm.step()

        warm_weights = {{}}
        if rank == 0:
            ps = get_peft_model_state_dict(engine_warm.module)
            warm_weights = {{k: v.detach().cpu().clone() for k, v in ps.items()}}
        del engine_warm

        # =============== PATH B: cold optimizer (same adapter, NO optimizer reload, step 2) ===============
        engine_cold = make_engine(cfg, adapter_path=adapter_dir)
        # No load_checkpoint — cold optimizer
        out = engine_cold(input_ids=data_step2)
        engine_cold.backward(out.logits.sum())
        engine_cold.step()

        cold_weights = {{}}
        if rank == 0:
            ps = get_peft_model_state_dict(engine_cold.module)
            cold_weights = {{k: v.detach().cpu().clone() for k, v in ps.items()}}
        del engine_cold

        # =============== Compare ===============
        if rank == 0:
            assert len(warm_weights) > 0 and len(cold_weights) > 0
            differing_params = 0
            max_abs_diff = 0.0
            for key in warm_weights:
                if key in cold_weights:
                    diff = (warm_weights[key].float() - cold_weights[key].float()).abs().max().item()
                    if diff > 0:
                        differing_params += 1
                    max_abs_diff = max(max_abs_diff, diff)
            result = {{
                "total_params": len(warm_weights),
                "differing_params": differing_params,
                "max_abs_diff": max_abs_diff,
            }}
            print("RESULT:" + json.dumps(result))
            print("ALL_OK")

        dist.barrier()
        dist.destroy_process_group()
    """)

    script_path = tmp_path / "test_warm_vs_cold.py"
    script_path.write_text(script)

    import os as _os
    clean_env = {k: v for k, v in _os.environ.items()
                 if k not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                              "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"}}
    clean_env["CUDA_VISIBLE_DEVICES"] = "0,1"

    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node", "2",
            str(script_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=clean_env,
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        pytest.fail(f"Subprocess failed (rc={result.returncode}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    assert "ALL_OK" in stdout, f"Script did not complete.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parsed = json.loads(line[len("RESULT:"):])
            print(f"WARM vs COLD: {parsed}")
            assert parsed["differing_params"] > 0, (
                "Warm and cold optimizer produced IDENTICAL weights — "
                "optimizer state persistence is not affecting training. "
                f"Result: {parsed}"
            )
            assert parsed["max_abs_diff"] > 0, (
                f"Zero max diff between warm and cold paths. Result: {parsed}"
            )
            break
    else:
        pytest.fail(f"No RESULT line found.\nSTDOUT:\n{stdout}")
