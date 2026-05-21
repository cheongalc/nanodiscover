import numpy as np
import pytest

from core.archive import ArchiveNode
from core.evaluator import TaskEvaluationRequirements
from tasks.ac2.env import AC2Task
from tasks.ac2.evaluator import evaluate_candidate_code, evaluate_sequence


# ---------------------------------------------------------------------------
# evaluate_sequence numeric correctness
# ---------------------------------------------------------------------------


def test_ac2_evaluate_sequence_known_constant():
    """A constant sequence [c, c, ...] should produce a finite, positive bound."""
    seq = [1.0] * 100
    bound = evaluate_sequence(seq)
    assert np.isfinite(bound)
    assert bound > 0


def test_ac2_evaluate_sequence_rejects_empty():
    with pytest.raises(ValueError, match="Empty sequence"):
        evaluate_sequence([])


def test_ac2_evaluate_sequence_rejects_nan():
    with pytest.raises(ValueError, match="Invalid sequence element"):
        evaluate_sequence([1.0, float("nan"), 1.0])


def test_ac2_evaluate_sequence_rejects_inf():
    with pytest.raises(ValueError, match="Invalid sequence element"):
        evaluate_sequence([1.0, float("inf"), 1.0])


def test_ac2_evaluate_sequence_rejects_bool():
    with pytest.raises(ValueError, match="Invalid sequence element"):
        evaluate_sequence([True, False, True])


def test_ac2_evaluate_sequence_rejects_near_zero_sum():
    with pytest.raises(ValueError, match="too close to zero"):
        evaluate_sequence([0.0, 0.0, 0.0])


def test_ac2_evaluate_sequence_formula_matches_original():
    """Verify the L2/L1/Linf formula against a hand-computed example.

    For sequence [1, 2, 3]:
      conv = convolve([1,2,3], [1,2,3]) = [1, 4, 10, 12, 9]
    We then compute L2^2 via trapezoidal-like piecewise-linear integration,
    L1 = sum(|conv|) / (len(conv)+1), Linf = max(|conv|),
    and return L2^2 / (L1 * Linf).
    """
    seq = [1.0, 2.0, 3.0]
    bound = evaluate_sequence(seq)
    assert np.isfinite(bound)
    assert bound > 0

    # Cross-check: manually compute the same formula
    conv = np.convolve(seq, seq)
    num_points = len(conv)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], conv, [0]))
    l2_sq = 0.0
    for i in range(num_points + 1):
        y1, y2, h = y_points[i], y_points[i + 1], x_intervals[i]
        l2_sq += (h / 3) * (y1**2 + y1 * y2 + y2**2)
    norm_1 = np.sum(np.abs(conv)) / (num_points + 1)
    norm_inf = np.max(np.abs(conv))
    expected = l2_sq / (norm_1 * norm_inf)
    assert abs(bound - expected) < 1e-12


# ---------------------------------------------------------------------------
# Code extraction: LAST fenced python block
# ---------------------------------------------------------------------------


def test_ac2_parse_last_code_block():
    task = AC2Task()
    response = (
        "first block\n```python\nprint(1)\n```\n"
        "second block\n```python\ndef construct_function():\n    return [1.0]\n```"
    )
    code = task.parse_code(response)
    assert "construct_function" in code
    assert "print(1)" not in code


# ---------------------------------------------------------------------------
# Reward formula
# ---------------------------------------------------------------------------


def test_ac2_reward_equals_raw_score():
    """AC2 reward = raw_score directly (maximize)."""
    task = AC2Task()
    reward = task.compute_reward({"correctness": 1.0, "raw_score": 4.2})
    assert abs(reward - 4.2) < 1e-12


def test_ac2_reward_zero_when_incorrect():
    task = AC2Task()
    reward = task.compute_reward({"correctness": 0.0, "raw_score": 4.2})
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Initial state: deterministic seed 12345
# ---------------------------------------------------------------------------


def test_ac2_initial_state_is_deterministic():
    task_a = AC2Task()
    task_b = AC2Task()
    state_a = task_a.make_initial_state()
    state_b = task_b.make_initial_state()
    assert state_a.task_payload["construction"] == state_b.task_payload["construction"]
    assert state_a.value == state_b.value


def test_ac2_initial_state_construction_within_length_limits():
    task = AC2Task()
    state = task.make_initial_state()
    length = len(state.task_payload["construction"])
    assert 1000 <= length <= 100000


def test_ac2_initial_state_value_is_positive():
    """AC2 maximizes, so state.value = raw_score (positive)."""
    task = AC2Task()
    state = task.make_initial_state()
    assert state.value is not None
    assert state.value > 0


# ---------------------------------------------------------------------------
# Refresh initial state (AC2 has construction_length_limits)
# ---------------------------------------------------------------------------


def test_ac2_refresh_mutates_construction_preserves_code():
    task = AC2Task()
    state = task.make_initial_state()
    original_id = state.id
    original_code = state.task_payload["code"]
    original_construction = list(state.task_payload["construction"])

    task.refresh_initial_state(state)

    assert state.id == original_id
    assert state.task_payload["code"] == original_code
    assert state.task_payload["construction"] != original_construction
    assert state.value is not None
    assert state.value > 0


# ---------------------------------------------------------------------------
# make_next_state: value = raw_score (positive, maximize)
# ---------------------------------------------------------------------------


def test_ac2_next_state_value_equals_raw_score():
    task = AC2Task()
    next_state = task.make_next_state(
        parent_state=ArchiveNode(epoch=0, value=3.0, task_payload={"construction": [1.0] * 1000}),
        parsed_code="def construct_function():\n    return [1.0] * 1000",
        eval_output={
            "correctness": 1.0,
            "raw_score": 5.5,
            "performance": 5.5,
            "result_payload": {"result_construction": [1.0] * 1000},
            "stdout": "ok",
        },
        epoch=1,
    )
    assert next_state is not None
    assert next_state.value == 5.5


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def test_ac2_validity_matches_original_length_limits():
    task = AC2Task()
    short = ArchiveNode(epoch=0, value=1.0, task_payload={"construction": [0.1] * 999})
    long = ArchiveNode(epoch=0, value=1.0, task_payload={"construction": [0.1] * 100001})
    valid = ArchiveNode(epoch=0, value=1.0, task_payload={"construction": [0.1] * 1000})

    assert not task.is_state_valid(short)
    assert not task.is_state_valid(long)
    assert task.is_state_valid(valid)


# ---------------------------------------------------------------------------
# Resource requirements
# ---------------------------------------------------------------------------


def test_ac2_declares_two_cpus_per_evaluation():
    task = AC2Task()
    assert task.evaluation_resources() == TaskEvaluationRequirements(cpus_per_eval=2)


# ---------------------------------------------------------------------------
# maximize direction
# ---------------------------------------------------------------------------


def test_ac2_maximize_raw_score_is_true():
    assert AC2Task.maximize_raw_score is True


# ---------------------------------------------------------------------------
# Worker: quick failure for bad code
# ---------------------------------------------------------------------------


def test_ac2_worker_failure_returns_quickly():
    import time

    started = time.perf_counter()
    result = evaluate_candidate_code(
        code="print('hello')",
        parent_construction=[0.5, 0.5],
        timeout_s=5,
        budget_s=1,
        seed=0,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0
    assert result["correctness"] == 0.0
    assert "construct_function" in str(result["msg"])
