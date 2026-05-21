from contextlib import contextmanager
import json
import threading
import time

from core.archive import ArchiveNode
import core.evaluator as evaluator_module
from core.evaluator import (
    EvaluatedRollout,
    Evaluator,
    EvaluatorConfig,
    EvaluationResourcePool,
    TaskEvaluationRequirements,
    load_evaluated_rollouts_from_path,
    merge_evaluation_shards,
)
from core.generator import GenerationOutput
import pytest
from tasks.ac1.env import AC1_BUDGET_SECONDS, AC1Task
from tasks.ac1.evaluator import evaluate_candidate_code
from tasks.ac1.prompt import default_initial_code
import utils


def make_evaluated_rollout(response_text: str, *, raw_score: float = 1.0, stdout: str = "ok") -> EvaluatedRollout:
    """Build a small valid evaluated rollout for evaluator persistence tests."""

    return EvaluatedRollout(
        seed_state=ArchiveNode(epoch=0, value=1.0, task_payload={"construction": [0.0], "code": "seed"}),
        prompt_text="prompt",
        response_text=response_text,
        prompt_token_ids=[1],
        completion_token_ids=[2],
        completion_logprobs=[-1.0],
        completion_mask=[1.0],
        finish_reason="stop",
        parsed_code=response_text,
        reward=raw_score,
        correctness=1.0,
        performance=raw_score,
        raw_score=raw_score,
        archive_value=raw_score,
        next_state=ArchiveNode(
            epoch=0,
            value=raw_score,
            task_payload={"construction": [raw_score], "code": response_text},
        ),
        msg="ok",
        result_payload={"result_construction": [raw_score]},
        stdout=stdout,
    )


def test_ac1_parse_last_code_block_and_reward_formula():
    task = AC1Task()
    response = "ignore\n```python\nprint(1)\n```\ntext\n```python\ndef propose_candidate(seed: int, budget_s: int):\n    return [1.0, 1.0]\n```"
    code = task.parse_code(response)
    assert code.strip() == "def propose_candidate(seed: int, budget_s: int):\n    return [1.0, 1.0]"
    reward = task.compute_reward({"correctness": 1.0, "raw_score": 2.0})
    assert abs(reward - (1.0 / (1e-8 + 2.0))) < 1e-9


def test_evaluator_only_creates_next_state_for_correct_rollouts():
    class _StubTask:
        def parse_code(self, response_text):
            return response_text

        def evaluate_code(self, *, parsed_code, state, epoch, seed):
            _ = (parsed_code, state, epoch, seed)
            return {"correctness": 0.0, "performance": 0.0, "raw_score": None, "msg": "bad", "result_payload": {}, "stdout": ""}

        def compute_reward(self, eval_output):
            return 0.0

        def make_next_state(self, *, parent_state, parsed_code, eval_output, epoch):
            _ = (parent_state, parsed_code, eval_output, epoch)
            raise AssertionError("should not be called")

    evaluator = Evaluator(EvaluatorConfig(max_workers=1))
    rollout = evaluator.evaluate_batch(
        task=_StubTask(),
        seed_states=[ArchiveNode(epoch=0, value=1.0, task_payload={})],
        prompts=["p"],
        generations=[GenerationOutput("p", "resp", [1], [2], [-1.0], [1.0], "stop")],
        epoch=0,
        base_seed=0,
    )[0]
    assert rollout.next_state is None


def test_evaluator_converts_task_exceptions_into_failed_rollouts():
    class _StubTask:
        def parse_code(self, response_text):
            return response_text

        def evaluate_code(self, *, parsed_code, state, epoch, seed):
            _ = (parsed_code, state, epoch, seed)
            raise RuntimeError("boom")

        def compute_reward(self, eval_output):
            _ = eval_output
            raise AssertionError("should not be called after evaluator failure")

        def make_next_state(self, *, parent_state, parsed_code, eval_output, epoch):
            _ = (parent_state, parsed_code, eval_output, epoch)
            raise AssertionError("should not be called")

    evaluator = Evaluator(EvaluatorConfig(max_workers=1))
    rollout = evaluator.evaluate_batch(
        task=_StubTask(),
        seed_states=[ArchiveNode(epoch=0, value=1.0, task_payload={})],
        prompts=["p"],
        generations=[GenerationOutput("p", "resp", [1], [2], [-1.0], [1.0], "stop")],
        epoch=0,
        base_seed=0,
    )[0]

    assert rollout.reward == 0.0
    assert rollout.correctness == 0.0
    assert rollout.next_state is None
    assert rollout.raw_score is None
    assert rollout.msg == "evaluation failed: boom"


def test_load_evaluated_rollouts_from_path_rejects_missing_indices(tmp_path):
    payload = {
        "epoch": 0,
        "items": [
            {"index": 0, "rollout": make_evaluated_rollout("resp-0").to_dict()},
            {"index": 2, "rollout": make_evaluated_rollout("resp-2").to_dict()},
        ],
    }
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing rollout indices: 1"):
        load_evaluated_rollouts_from_path(path, expected_total=3)


def test_merge_evaluation_shards_writes_ordered_epoch_outputs(tmp_path):
    run_dir = tmp_path / "run"
    shard_dir = run_dir / "epoch000" / "evaluation_shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "shard_1.json").write_text(
        json.dumps(
            {
                "epoch": 0,
                "items": [{"index": 1, "rollout": make_evaluated_rollout("resp-1", raw_score=2.0).to_dict()}],
            }
        ),
        encoding="utf-8",
    )
    (shard_dir / "shard_0.json").write_text(
        json.dumps(
            {
                "epoch": 0,
                "items": [{"index": 0, "rollout": make_evaluated_rollout("resp-0", raw_score=1.0).to_dict()}],
            }
        ),
        encoding="utf-8",
    )

    summary = merge_evaluation_shards(
        run_dir=run_dir,
        epoch=0,
        shard_dir=shard_dir,
        expected_total=2,
    )

    subdir = utils.epoch_subdir(run_dir, 0)
    loaded = utils.load_evaluations(subdir)
    assert summary["evaluated"] == 2
    assert summary["stream_files"] == 2
    assert [rollout.response_text for rollout in loaded] == ["resp-0", "resp-1"]
    assert (subdir.evaluation_logs / "rollout_000000.log").exists()
    assert (subdir.evaluation_logs / "rollout_000001.log").exists()


def test_evaluator_cli_routes_output_to_epoch_local_log(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    @contextmanager
    def fake_capture_stage_output(path):
        captured["path"] = path
        yield

    def fake_run_cli_command(args):
        captured["args"] = args

    monkeypatch.setattr(utils, "capture_stage_output", fake_capture_stage_output)
    monkeypatch.setattr(evaluator_module, "run_cli_command", fake_run_cli_command)

    run_dir = tmp_path / "run"
    evaluator_module.main(
        [
            "evaluate-shard",
            "--task",
            "ac1",
            "--run-dir",
            str(run_dir),
            "--epoch",
            "3",
            "--start",
            "0",
            "--stop",
            "1",
            "--workers",
            "1",
            "--output",
            str(run_dir / "epoch003" / "evaluation_shards" / "shard_0.json"),
        ]
    )

    assert captured["path"] == run_dir / "epoch003" / "evaluator.log"
    assert getattr(captured["args"], "command") == "evaluate-shard"


def test_evaluator_preserves_reward_when_next_state_creation_fails():
    class _StubTask:
        def parse_code(self, response_text):
            return response_text

        def evaluate_code(self, *, parsed_code, state, epoch, seed):
            _ = (parsed_code, state, epoch, seed)
            return {
                "correctness": 1.0,
                "performance": 7.0,
                "raw_score": 7.0,
                "msg": "ok",
                "result_payload": {"result_construction": [1.0]},
                "stdout": "",
            }

        def compute_reward(self, eval_output):
            return float(eval_output["raw_score"])

        def make_next_state(self, *, parent_state, parsed_code, eval_output, epoch):
            _ = (parent_state, parsed_code, eval_output, epoch)
            raise RuntimeError("cannot build state")

    evaluator = Evaluator(EvaluatorConfig(max_workers=1))
    rollout = evaluator.evaluate_batch(
        task=_StubTask(),
        seed_states=[ArchiveNode(epoch=0, value=1.0, task_payload={})],
        prompts=["p"],
        generations=[GenerationOutput("p", "resp", [1], [2], [-1.0], [1.0], "stop")],
        epoch=0,
        base_seed=0,
    )[0]

    assert rollout.reward == 7.0
    assert rollout.correctness == 1.0
    assert rollout.raw_score == 7.0
    assert rollout.next_state is None
    assert "next_state creation failed: cannot build state" in rollout.msg


def test_evaluator_progress_log_uses_minimize_direction(caplog):
    class _StubTask:
        maximize_raw_score = False

        def __init__(self):
            self._seen = 0

        def parse_code(self, response_text):
            return response_text

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
            return ArchiveNode(epoch=epoch, value=-float(eval_output["raw_score"]), task_payload={})

    evaluator = Evaluator(EvaluatorConfig(max_workers=1))
    caplog.set_level("INFO", logger=evaluator_module.logger.name)

    evaluator.evaluate_batch(
        task=_StubTask(),
        seed_states=[ArchiveNode(epoch=0, value=1.0, task_payload={}) for _ in range(2)],
        prompts=["p1", "p2"],
        generations=[
            GenerationOutput("p1", "resp1", [1], [2], [-1.0], [1.0], "stop"),
            GenerationOutput("p2", "resp2", [1], [2], [-1.0], [1.0], "stop"),
        ],
        epoch=0,
        base_seed=0,
    )

    progress_messages = [record.getMessage() for record in caplog.records if "evaluator_progress completed=" in record.getMessage()]

    assert progress_messages
    assert "best_raw_score=1.0000" in progress_messages[-1]


def test_ac1_initial_state_matches_discover_seed_and_starter_code():
    task = AC1Task()
    state_a = task.make_initial_state()
    state_b = task.make_initial_state()

    assert state_a.task_payload["construction"] == state_b.task_payload["construction"]
    assert state_a.task_payload["code"] == default_initial_code(AC1_BUDGET_SECONDS)
    assert len(state_a.task_payload["construction"]) >= 1000


def test_ac1_prompt_uses_discover_template_and_state_context():
    task = AC1Task()
    state = ArchiveNode(
        epoch=0,
        value=-1.75,
        parent_values=[-1.80],
        task_payload={
            "construction": [0.5] * 1000,
            "code": "```python\ndef propose_candidate(seed=0, budget_s=1, **kwargs):\n    return [0.5] * 1000\n```",
            "stdout": "best=1.75",
        },
    )

    prompt = task.render_prompt(state)

    assert prompt.startswith(
        "Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties."
    )
    assert "You are iteratively optimizing upper bound." in prompt
    assert "Here is the last code we ran:" in prompt
    assert "Here is the upper bound before and after running the code above (lower is better): 1.800000 -> 1.750000" in prompt
    assert "Target: 1.503. Current gap: 0.247000." in prompt
    assert "Length of the construction: 1000" in prompt
    assert "--- Previous Program Output ---" in prompt
    assert "height_sequence_1" in prompt


def test_ac1_initial_prompt_uses_lower_is_better_wording():
    task = AC1Task()
    state = task.make_initial_state()

    prompt = task.render_prompt(state)

    assert "Current upper bound (lower is better):" in prompt


def test_ac1_validity_matches_original_length_limits():
    task = AC1Task()
    short_state = ArchiveNode(epoch=0, value=-1.0, task_payload={"construction": [0.1] * 999})
    long_state = ArchiveNode(epoch=0, value=-1.0, task_payload={"construction": [0.1] * 100001})
    valid_state = ArchiveNode(epoch=0, value=-1.0, task_payload={"construction": [0.1] * 1000})

    assert not task.is_state_valid(short_state)
    assert not task.is_state_valid(long_state)
    assert task.is_state_valid(valid_state)


def test_ac1_refresh_mutates_seed_in_place_but_preserves_code():
    task = AC1Task()
    state = task.make_initial_state()
    original_id = state.id
    original_code = state.task_payload["code"]
    original_construction = list(state.task_payload["construction"])

    task.refresh_initial_state(state)

    assert state.id == original_id
    assert state.task_payload["code"] == original_code
    assert state.task_payload["construction"] != original_construction
    assert state.value is not None


def test_ac1_next_state_uses_raw_score_for_archive_value():
    task = AC1Task()

    next_state = task.make_next_state(
        parent_state=ArchiveNode(epoch=0, value=-1.9, task_payload={"construction": [0.1] * 1000}),
        parsed_code="def propose_candidate(seed=0, budget_s=1, **kwargs):\n    return [1.0] * 1000",
        eval_output={
            "correctness": 1.0,
            "raw_score": 1.75,
            "performance": -999.0,
            "result_payload": {"result_construction": [1.0] * 1000},
            "stdout": "ok",
        },
        epoch=3,
    )

    assert next_state is not None
    assert next_state.value == -1.75


def test_ac1_declares_two_cpus_per_evaluation():
    task = AC1Task()

    requirements = task.evaluation_resources()

    assert requirements == TaskEvaluationRequirements(cpus_per_eval=2)


def test_evaluation_resource_pool_cpu_pack_slot_offsets_first_slot(monkeypatch):
    monkeypatch.setattr(evaluator_module, "available_cpu_ids", lambda: tuple(range(40)))
    req = TaskEvaluationRequirements(cpus_per_eval=2)
    pool = EvaluationResourcePool(req, requested_workers=1, cpu_pack_slot=5)
    with pool.acquire() as slot:
        assert slot.cpu_ids == (10, 11)


def test_evaluation_resource_pool_cpu_pack_slot_rejects_out_of_range(monkeypatch):
    monkeypatch.setattr(evaluator_module, "available_cpu_ids", lambda: tuple(range(10)))
    req = TaskEvaluationRequirements(cpus_per_eval=2)
    with pytest.raises(ValueError, match="cpu_pack_slot"):
        EvaluationResourcePool(req, requested_workers=1, cpu_pack_slot=5)


def test_evaluator_limits_concurrency_from_task_cpu_requirements(monkeypatch):
    monkeypatch.setattr(evaluator_module, "available_cpu_ids", lambda: (0, 1, 2, 3))

    concurrency_lock = threading.Lock()
    active = 0
    max_active = 0
    observed_thread_counts: list[int] = []

    class _StubTask:
        def evaluation_resources(self):
            return TaskEvaluationRequirements(cpus_per_eval=2)

        def parse_code(self, response_text):
            return response_text

        def evaluate_code(self, *, parsed_code, state, epoch, seed, resources):
            nonlocal active, max_active
            _ = (parsed_code, state, epoch, seed)
            with concurrency_lock:
                active += 1
                max_active = max(max_active, active)
                observed_thread_counts.append(resources.thread_count)
            time.sleep(0.02)
            with concurrency_lock:
                active -= 1
            return {"correctness": 0.0, "performance": 0.0, "raw_score": None, "msg": "ok", "result_payload": {}, "stdout": ""}

        def compute_reward(self, eval_output):
            return 0.0

        def make_next_state(self, *, parent_state, parsed_code, eval_output, epoch):
            _ = (parent_state, parsed_code, eval_output, epoch)
            return None

    evaluator = Evaluator(EvaluatorConfig(max_workers=8))
    evaluator.evaluate_batch(
        task=_StubTask(),
        seed_states=[ArchiveNode(epoch=0, value=1.0, task_payload={}) for _ in range(4)],
        prompts=["p"] * 4,
        generations=[GenerationOutput("p", "resp", [1], [2], [-1.0], [1.0], "stop") for _ in range(4)],
        epoch=0,
        base_seed=0,
    )

    assert max_active <= 2
    assert observed_thread_counts == [2, 2, 2, 2]


def test_ac1_worker_failure_returns_quickly_without_waiting_for_timeout():
    started_at = time.perf_counter()

    result = evaluate_candidate_code(
        code="print('hello from worker')",
        parent_construction=[0.5, 0.5],
        timeout_s=5,
        budget_s=1,
        seed=0,
    )

    elapsed_s = time.perf_counter() - started_at
    assert elapsed_s < 2.0
    assert result["correctness"] == 0.0
    assert "Generated code must define propose_candidate" in str(result["msg"])