from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.archive import ArchiveNode
from core.evaluator import (
    AllocatedEvaluationResources,
    EvaluatedRollout,
    evaluate_rollout,
    load_single_rollout_shard,
    scan_shard_status,
)
from core.generator import GenerationOutput


def build_rollout_dict(raw_score: float) -> dict:
    """Build a minimal evaluated-rollout payload for shard-merge tests."""
    return EvaluatedRollout(
        seed_state=ArchiveNode(id="seed", epoch=0, value=1.0, task_payload={}),
        prompt_text="prompt",
        response_text="resp",
        prompt_token_ids=[10, 11],
        completion_token_ids=[12, 13],
        completion_logprobs=[-0.1, -0.2],
        completion_mask=[1.0, 1.0],
        finish_reason="stop",
        parsed_code="print(1)",
        reward=1.0,
        correctness=1.0,
        performance=raw_score,
        raw_score=raw_score,
        archive_value=raw_score,
        next_state=ArchiveNode(epoch=1, value=raw_score, task_payload={}),
        msg="ok",
    ).to_dict()


def test_evaluated_rollout_roundtrip_preserves_eval_wall_time():
    r = EvaluatedRollout(
        seed_state=ArchiveNode(id="seed", epoch=0, value=1.0, task_payload={}),
        prompt_text="prompt",
        response_text="resp",
        prompt_token_ids=[10, 11],
        completion_token_ids=[12, 13],
        completion_logprobs=[-0.1, -0.2],
        completion_mask=[1.0, 1.0],
        finish_reason="stop",
        parsed_code="print(1)",
        reward=1.0,
        correctness=1.0,
        performance=1.5,
        raw_score=1.5,
        archive_value=1.5,
        next_state=ArchiveNode(epoch=1, value=1.5, task_payload={}),
        msg="ok",
        eval_wall_time_s=42.25,
    )
    r2 = EvaluatedRollout.from_dict(r.to_dict())
    assert r2.eval_wall_time_s == 42.25


def test_evaluate_rollout_sets_eval_wall_time_around_evaluate_code():
    """``eval_wall_time_s`` spans ``task.evaluate_code`` only (not parse/reward/next_state)."""

    class _StubTask:
        maximize_raw_score = True

        def parse_code(self, response_text: str) -> str:
            _ = response_text
            return "pass"

        def evaluate_code(self, **_kwargs):
            time.sleep(0.06)
            return {
                "correctness": 0.0,
                "performance": 0.0,
                "raw_score": None,
                "msg": "stub",
                "result_payload": {},
                "stdout": "",
            }

        def compute_reward(self, eval_output):
            _ = eval_output
            return 0.0

    gen = GenerationOutput("p", "```python\nx=1\n```", [1], [2], [-1.0], [1.0], "stop")
    resources = AllocatedEvaluationResources(cpu_ids=(0, 1), slot_index=0)
    rollout = evaluate_rollout(
        task=_StubTask(),
        seed_state=ArchiveNode(epoch=0, value=0.0, task_payload={}),
        prompt_text="p",
        generation=gen,
        epoch=0,
        seed=0,
        resources=resources,
    )
    assert rollout.eval_wall_time_s is not None
    assert rollout.eval_wall_time_s >= 0.05


def test_merge_eval_shards_rejects_duplicate_indices(tmp_path):
    """Reject duplicate rollout indices when merging shard payloads."""

    run_dir = tmp_path / "run"
    shard_dir = tmp_path / "evaluation_shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "shard_0.json").write_text(
        json.dumps({"items": [{"index": 0, "rollout": build_rollout_dict(1.0)}]}),
        encoding="utf-8",
    )
    (shard_dir / "shard_1.json").write_text(
        json.dumps({"items": [{"index": 0, "rollout": build_rollout_dict(2.0)}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.evaluator",
            "merge-shards",
            "--run-dir",
            str(run_dir),
            "--epoch",
            "0",
            "--shard-dir",
            str(shard_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Duplicate rollout index 0" in (result.stdout + result.stderr)


def test_load_single_rollout_shard_rejects_wrong_index(tmp_path):
    """Reject shard files whose single rollout index does not match the filename."""

    shard_path = tmp_path / "shard_1.json"
    shard_path.write_text(
        json.dumps({"items": [{"index": 0, "rollout": build_rollout_dict(1.0)}]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected 1"):
        load_single_rollout_shard(shard_path, expected_index=1)


def test_scan_shard_status_deletes_invalid_shards(tmp_path):
    """Treat invalid shard files as pending and delete them for retry."""

    shard_dir = tmp_path / "evaluation_shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "shard_0.json").write_text(
        json.dumps({"items": [{"index": 0, "rollout": build_rollout_dict(1.0)}]}),
        encoding="utf-8",
    )
    (shard_dir / "shard_1.json").write_text(
        json.dumps({"items": [{"index": 0, "rollout": build_rollout_dict(2.0)}]}),
        encoding="utf-8",
    )

    status = scan_shard_status(shard_dir, expected_total=2, delete_invalid=True)

    assert status == {
        "complete_count": 1,
        "invalid_count": 1,
        "pending_indices": [1],
    }
    assert not (shard_dir / "shard_1.json").exists()
