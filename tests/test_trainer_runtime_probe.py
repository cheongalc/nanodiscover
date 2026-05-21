from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def gpu_runtime_probe_enabled() -> bool:
    """Return whether the expensive 4-GPU runtime probe should run."""
    return os.environ.get("NANODISCOVER_RUN_GPU_RUNTIME_TESTS", "0") == "1"


def run_distributed_probe(
    probe_path: Path,
    *,
    extra_args: list[str] | None = None,
) -> dict[str, object]:
    """Run one 4-GPU distributed probe helper and return its JSON payload."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(probe_path.parent.parent)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TRITON_CACHE_DIR", "/tmp/nanodiscover-triton-cache")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "4",
            str(probe_path),
            *(extra_args or []),
        ],
        cwd=str(probe_path.parent.parent),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            f"Runtime probe {probe_path.name} failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    payload_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("RESULT_JSON=")),
        None,
    )
    if payload_line is None:
        raise AssertionError(f"Probe did not emit RESULT_JSON.\nstdout:\n{result.stdout}")
    return json.loads(payload_line.removeprefix("RESULT_JSON="))


@pytest.mark.skipif(
    not gpu_runtime_probe_enabled(),
    reason="Set NANODISCOVER_RUN_GPU_RUNTIME_TESTS=1 to run the 4-GPU runtime trainer probe.",
)
def test_ulysses_flash_attention_runtime_blocks_cross_sequence_attention():
    """Verify the production Ulysses + FA2 path blocks packed cross-sequence leakage."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepspeed")
    pytest.importorskip("transformers")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("Requires at least 4 visible CUDA GPUs.")

    probe_path = Path(__file__).with_name("_ulysses_boundary_probe.py")
    payload = run_distributed_probe(probe_path)
    assert payload["reset_max_abs_diff_a"] > 0.0
    assert payload["reset_max_abs_diff_b"] == pytest.approx(0.0)
    assert payload["monotonic_max_abs_diff_b"] > 0.0


@pytest.mark.skipif(
    not gpu_runtime_probe_enabled(),
    reason="Set NANODISCOVER_RUN_GPU_RUNTIME_TESTS=1 to run the 4-GPU runtime trainer probes.",
)
def test_qwen3_8b_ulysses_train_step_matches_single_vs_multi_microbatch(tmp_path):
    """Verify actual prod-path train_step is microbatch-equivalent on 4 GPUs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepspeed")
    pytest.importorskip("transformers")
    load_file = pytest.importorskip("safetensors.torch").load_file

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("Requires at least 4 visible CUDA GPUs.")

    probe_path = Path(__file__).with_name("_prod_train_step_probe.py")
    run_root = tmp_path / "runtime_prod_microbatch"
    run_root.mkdir(parents=True, exist_ok=True)
    initial_dir = run_root / "initial"
    single_dir = run_root / "single_batch"
    multi_dir = run_root / "multi_batch"

    initial = run_distributed_probe(
        probe_path,
        extra_args=[
            "--operation", "save_initial",
            "--run-dir", str(run_root / "run"),
            "--output-dir", str(initial_dir),
            "--seed", "777",
        ],
    )
    single = run_distributed_probe(
        probe_path,
        extra_args=[
            "--operation", "train_step",
            "--run-dir", str(run_root / "run_single"),
            "--output-dir", str(single_dir),
            "--seed", "777",
            "--batch-name", "batch1",
            "--max-tokens-per-rank", "64",
            "--resume-adapter-path", str(initial["adapter_path"]),
        ],
    )
    multi = run_distributed_probe(
        probe_path,
        extra_args=[
            "--operation", "train_step",
            "--run-dir", str(run_root / "run_multi"),
            "--output-dir", str(multi_dir),
            "--seed", "777",
            "--batch-name", "batch1",
            "--max-tokens-per-rank", "10",
            "--resume-adapter-path", str(initial["adapter_path"]),
        ],
    )

    assert float(single["max_param_delta"]) > 0.0
    assert float(multi["max_param_delta"]) > 0.0
    single_metrics = dict(single["metrics"])
    multi_metrics = dict(multi["metrics"])
    assert float(single_metrics["train/optimizer_steps"]) == 1.0
    assert float(multi_metrics["train/optimizer_steps"]) == 1.0
    assert float(single_metrics["train/microbatch_count"]) == 1.0
    assert float(multi_metrics["train/microbatch_count"]) > 1.0
    assert float(single_metrics["train/sequence_parallel_size"]) == 4.0
    assert float(multi_metrics["train/sequence_parallel_size"]) == 4.0
    assert float(single_metrics["train/use_remove_padding"]) == 1.0
    assert float(multi_metrics["train/use_remove_padding"]) == 1.0

    single_weights = load_file(str(Path(str(single["adapter_path"])) / "adapter_model.safetensors"))
    multi_weights = load_file(str(Path(str(multi["adapter_path"])) / "adapter_model.safetensors"))
    assert single_weights.keys() == multi_weights.keys()

    max_abs_diff = 0.0
    for name in single_weights:
        max_abs_diff = max(
            max_abs_diff,
            float((single_weights[name].float() - multi_weights[name].float()).abs().max().item()),
        )
    assert max_abs_diff < 1e-3


@pytest.mark.skipif(
    not gpu_runtime_probe_enabled(),
    reason="Set NANODISCOVER_RUN_GPU_RUNTIME_TESTS=1 to run the 4-GPU runtime trainer probes.",
)
def test_qwen3_8b_ulysses_warm_optimizer_resume_changes_second_step(tmp_path):
    """Verify prod-path warm optimizer resume changes step-2 behavior on 4 GPUs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepspeed")
    pytest.importorskip("transformers")
    load_file = pytest.importorskip("safetensors.torch").load_file

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("Requires at least 4 visible CUDA GPUs.")

    probe_path = Path(__file__).with_name("_prod_train_step_probe.py")
    run_root = tmp_path / "runtime_prod_resume"
    run_root.mkdir(parents=True, exist_ok=True)
    initial_dir = run_root / "initial"
    step1_dir = run_root / "step1"
    warm_dir = run_root / "warm"
    cold_dir = run_root / "cold"

    initial = run_distributed_probe(
        probe_path,
        extra_args=[
            "--operation", "save_initial",
            "--run-dir", str(run_root / "run"),
            "--output-dir", str(initial_dir),
            "--seed", "888",
        ],
    )
    step1 = run_distributed_probe(
        probe_path,
        extra_args=[
            "--operation", "train_step",
            "--run-dir", str(run_root / "run_step1"),
            "--output-dir", str(step1_dir),
            "--seed", "888",
            "--batch-name", "batch1",
            "--max-tokens-per-rank", "10",
            "--resume-adapter-path", str(initial["adapter_path"]),
        ],
    )
    warm = run_distributed_probe(
        probe_path,
        extra_args=[
            "--operation", "train_step",
            "--run-dir", str(run_root / "run_warm"),
            "--output-dir", str(warm_dir),
            "--seed", "888",
            "--batch-name", "batch2",
            "--max-tokens-per-rank", "10",
            "--resume-adapter-path", str(step1["adapter_path"]),
            "--resume-optimizer-path", str(step1["optimizer_state_dir"]),
        ],
    )
    cold = run_distributed_probe(
        probe_path,
        extra_args=[
            "--operation", "train_step",
            "--run-dir", str(run_root / "run_cold"),
            "--output-dir", str(cold_dir),
            "--seed", "888",
            "--batch-name", "batch2",
            "--max-tokens-per-rank", "10",
            "--resume-adapter-path", str(step1["adapter_path"]),
        ],
    )

    warm_metrics = dict(warm["metrics"])
    cold_metrics = dict(cold["metrics"])
    assert float(warm["max_param_delta"]) > 0.0
    assert float(cold["max_param_delta"]) > 0.0
    assert float(warm_metrics["train/optimizer_steps"]) == 1.0
    assert float(cold_metrics["train/optimizer_steps"]) == 1.0
    assert float(warm_metrics["train/sequence_parallel_size"]) == 4.0
    assert float(cold_metrics["train/sequence_parallel_size"]) == 4.0

    warm_weights = load_file(str(Path(str(warm["adapter_path"])) / "adapter_model.safetensors"))
    cold_weights = load_file(str(Path(str(cold["adapter_path"])) / "adapter_model.safetensors"))
    assert warm_weights.keys() == cold_weights.keys()

    differing_params = 0
    max_abs_diff = 0.0
    for name in warm_weights:
        diff = float((warm_weights[name].float() - cold_weights[name].float()).abs().max().item())
        if diff > 0.0:
            differing_params += 1
        max_abs_diff = max(max_abs_diff, diff)
    assert differing_params > 0
    assert max_abs_diff > 0.0
