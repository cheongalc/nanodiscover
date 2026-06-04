import numpy as np
import pytest

from core.archive import ArchiveNode
from core.evaluator import TaskEvaluationRequirements
from tasks.erdos.env import ERDOS_EVAL_TIMEOUT_SECONDS, MAX_CONSTRUCTION_LEN, ErdosTask
from tasks.erdos.evaluator import (
    evaluate_candidate_code,
    evaluate_erdos_solution,
    verify_c5_solution,
    verify_erdos_solution,
)


# ---------------------------------------------------------------------------
# verify_c5_solution / evaluate_erdos_solution numeric correctness
# ---------------------------------------------------------------------------


def _make_valid_h(n: int = 100) -> np.ndarray:
    """Build a valid h in [0,1] that sums to n/2."""
    h = np.ones(n) * 0.5
    return h


def test_verify_c5_solution_accepts_valid():
    h = _make_valid_h(100)
    dx = 2.0 / len(h)
    j = 1.0 - h
    corr = np.correlate(h, j, mode="full") * dx
    c5 = float(np.max(corr))
    result = verify_c5_solution(h, c5, len(h))
    assert np.isfinite(result)
    assert abs(result - c5) < 1e-4


def test_verify_c5_solution_rejects_wrong_c5():
    h = _make_valid_h(100)
    dx = 2.0 / len(h)
    j = 1.0 - h
    corr = np.correlate(h, j, mode="full") * dx
    c5 = float(np.max(corr))
    with pytest.raises(ValueError, match="C5 mismatch"):
        verify_c5_solution(h, c5 + 1.0, len(h))


def test_verify_c5_solution_rejects_out_of_range():
    h = np.ones(50) * 1.5  # out of [0,1]
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        verify_c5_solution(h, 0.5, 50)


def test_verify_c5_solution_rejects_wrong_shape():
    h = np.ones((10, 2))
    with pytest.raises(ValueError, match="must be 1D"):
        verify_c5_solution(h, 0.5, 10)


def test_verify_c5_solution_rejects_wrong_length():
    h = _make_valid_h(50)
    dx = 2.0 / 50
    j = 1.0 - h
    corr = np.correlate(h, j, mode="full") * dx
    c5 = float(np.max(corr))
    with pytest.raises(ValueError, match="Expected h shape"):
        verify_c5_solution(h, c5, 100)


def test_verify_c5_solution_rejects_nan():
    h = np.array([0.5] * 49 + [float("nan")])
    with pytest.raises(ValueError, match="NaN or inf"):
        verify_c5_solution(h, 0.5, 50)


def test_verify_erdos_solution_tuple_interface():
    h = _make_valid_h(80)
    dx = 2.0 / len(h)
    j = 1.0 - h
    c5 = float(np.max(np.correlate(h, j, mode="full") * dx))
    assert verify_erdos_solution((h, c5, len(h))) is True


def test_verify_erdos_solution_rejects_invalid():
    assert verify_erdos_solution((np.ones(10) * 2.0, 0.5, 10)) is False


def test_evaluate_erdos_solution_returns_computed_c5():
    h = _make_valid_h(60)
    dx = 2.0 / len(h)
    j = 1.0 - h
    computed = float(np.max(np.correlate(h, j, mode="full") * dx))
    result = evaluate_erdos_solution(h, computed, len(h))
    assert abs(result - computed) < 1e-6


def test_evaluate_erdos_solution_returns_computed_not_reported_when_close():
    h = _make_valid_h(60)
    dx = 2.0 / len(h)
    j = 1.0 - h
    computed = float(np.max(np.correlate(h, j, mode="full") * dx))
    reported = computed - 9e-5
    result = evaluate_erdos_solution(h, reported, len(h))
    assert abs(result - computed) < 1e-6
    assert abs(result - reported) > 1e-5


# ---------------------------------------------------------------------------
# Code extraction: LAST fenced python block
# ---------------------------------------------------------------------------


def test_erdos_parse_last_code_block():
    task = ErdosTask()
    response = (
        "first\n```python\nprint(1)\n```\n"
        "second\n```python\ndef run():\n    return (h, c5, n)\n```"
    )
    code = task.parse_code(response)
    assert "def run()" in code
    assert "print(1)" not in code


# ---------------------------------------------------------------------------
# Reward formula: 1 / (1e-8 + raw_score)
# ---------------------------------------------------------------------------


def test_erdos_reward_formula():
    task = ErdosTask()
    reward = task.compute_reward({"correctness": 1.0, "raw_score": 0.25})
    expected = 1.0 / (1e-8 + 0.25)
    assert abs(reward - expected) < 1e-9


def test_erdos_reward_zero_when_incorrect():
    task = ErdosTask()
    reward = task.compute_reward({"correctness": 0.0, "raw_score": 0.25})
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Initial state: unseeded (non-deterministic)
# ---------------------------------------------------------------------------


def test_erdos_initial_state_has_negative_value():
    """Erdos minimizes c5_bound, so state.value = -c5_bound (negative)."""
    task = ErdosTask()
    state = task.make_initial_state()
    assert state.value is not None
    assert state.value < 0


def test_erdos_initial_state_construction_within_limits():
    task = ErdosTask()
    state = task.make_initial_state()
    length = len(state.task_payload["construction"])
    assert 0 < length <= MAX_CONSTRUCTION_LEN


# ---------------------------------------------------------------------------
# make_next_state: value = -raw_score (negated for minimize)
# ---------------------------------------------------------------------------


def test_erdos_next_state_value_is_negated_raw_score():
    task = ErdosTask()
    next_state = task.make_next_state(
        parent_state=ArchiveNode(epoch=0, value=-0.3, task_payload={"construction": [0.5] * 50}),
        parsed_code="def run():\n    pass",
        eval_output={
            "correctness": 1.0,
            "raw_score": 0.22,
            "performance": -0.22,
            "result_payload": {"result_construction": [0.5] * 50},
            "stdout": "ok",
        },
        epoch=1,
    )
    assert next_state is not None
    assert next_state.value == pytest.approx(-0.22)


# ---------------------------------------------------------------------------
# Validity: 0 < len <= 1000
# ---------------------------------------------------------------------------


def test_erdos_validity_accepts_empty_construction():
    """Parity: original only checks max_construction_len upper bound, no lower bound."""
    task = ErdosTask()
    state = ArchiveNode(epoch=0, value=-0.5, task_payload={"construction": []})
    assert task.is_state_valid(state)


def test_erdos_validity_rejects_too_long():
    task = ErdosTask()
    state = ArchiveNode(epoch=0, value=-0.5, task_payload={"construction": [0.5] * 1001})
    assert not task.is_state_valid(state)


def test_erdos_validity_accepts_within_limits():
    task = ErdosTask()
    state = ArchiveNode(epoch=0, value=-0.5, task_payload={"construction": [0.5] * 500})
    assert task.is_state_valid(state)


# ---------------------------------------------------------------------------
# No refresh_initial_state (erdos has no construction_length_limits)
# ---------------------------------------------------------------------------


def test_erdos_has_no_refresh():
    assert not hasattr(ErdosTask, "refresh_initial_state")


# ---------------------------------------------------------------------------
# Resource requirements
# ---------------------------------------------------------------------------


def test_erdos_declares_one_cpu_per_evaluation():
    task = ErdosTask()
    assert task.evaluation_resources() == TaskEvaluationRequirements(cpus_per_eval=1)


# ---------------------------------------------------------------------------
# minimize direction
# ---------------------------------------------------------------------------


def test_erdos_maximize_raw_score_is_false():
    assert ErdosTask.maximize_raw_score is False


# ---------------------------------------------------------------------------
# Timeout constant
# ---------------------------------------------------------------------------


def test_erdos_eval_timeout_is_1100():
    assert ERDOS_EVAL_TIMEOUT_SECONDS == 1100


# ---------------------------------------------------------------------------
# Worker: quick failure for bad code
# ---------------------------------------------------------------------------


def test_erdos_worker_failure_returns_quickly():
    import time

    started = time.perf_counter()
    result = evaluate_candidate_code(
        code="print('hello')",
        parent_construction=[0.5] * 50,
        timeout_s=5,
        budget_s=1,
        seed=0,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0
    assert result["correctness"] == 0.0
    assert "run" in str(result["msg"]).lower()
