import numpy as np
import pytest

from core.archive import ArchiveNode
from core.evaluator import TaskEvaluationRequirements
from tasks.circle_packing.env import CIRCLE_PACKING_EVAL_TIMEOUT_SECONDS, CirclePackingTask
from tasks.circle_packing.evaluator import (
    check_packing_correctness,
    evaluate_candidate_code,
    validate_packing,
)


# ---------------------------------------------------------------------------
# validate_packing correctness
# ---------------------------------------------------------------------------


def _two_valid_circles():
    """Two non-overlapping circles inside the unit square."""
    centers = np.array([[0.2, 0.2], [0.8, 0.8]])
    radii = np.array([0.15, 0.15])
    return centers, radii


def test_validate_packing_accepts_valid():
    centers, radii = _two_valid_circles()
    assert validate_packing(centers, radii) is True


def test_validate_packing_rejects_overlap():
    centers = np.array([[0.3, 0.3], [0.35, 0.3]])
    radii = np.array([0.2, 0.2])
    assert validate_packing(centers, radii) is False


def test_validate_packing_rejects_outside_unit_square():
    centers = np.array([[0.05, 0.5], [0.5, 0.5]])
    radii = np.array([0.2, 0.1])  # first circle extends past x=0
    assert validate_packing(centers, radii) is False


def test_validate_packing_rejects_negative_radius():
    centers = np.array([[0.5, 0.5]])
    radii = np.array([-0.1])
    assert validate_packing(centers, radii) is False


def test_validate_packing_rejects_nan_center():
    centers = np.array([[float("nan"), 0.5]])
    radii = np.array([0.1])
    assert validate_packing(centers, radii) is False


def test_validate_packing_rejects_nan_radius():
    centers = np.array([[0.5, 0.5]])
    radii = np.array([float("nan")])
    assert validate_packing(centers, radii) is False


def test_validate_packing_tolerance_1e12():
    """Circles are allowed to touch the boundary within 1e-12 tolerance."""
    centers = np.array([[0.1, 0.5]])
    radii = np.array([0.1 + 1e-13])  # just barely within tolerance
    assert validate_packing(centers, radii) is True


def test_check_packing_correctness_rejects_wrong_shape():
    centers = np.array([[0.5, 0.5], [0.3, 0.3]])
    radii = np.array([0.1, 0.1])
    # Claims 3 circles but only provides 2
    assert check_packing_correctness(centers, radii, num_circles=3) is False


def test_check_packing_correctness_accepts_valid():
    centers, radii = _two_valid_circles()
    assert check_packing_correctness(centers, radii, num_circles=2) is True


# ---------------------------------------------------------------------------
# Code extraction: LAST fenced python block
# ---------------------------------------------------------------------------


def test_circle_packing_parse_last_code_block():
    task = CirclePackingTask()
    response = (
        "first\n```python\nprint(1)\n```\n"
        "second\n```python\ndef run_packing():\n    pass\n```"
    )
    code = task.parse_code(response)
    assert "run_packing" in code
    assert "print(1)" not in code


# ---------------------------------------------------------------------------
# Reward formula: reward = raw_score (maximize)
# ---------------------------------------------------------------------------


def test_circle_packing_reward_equals_raw_score():
    task = CirclePackingTask()
    reward = task.compute_reward({"correctness": 1.0, "raw_score": 2.5})
    assert abs(reward - 2.5) < 1e-12


def test_circle_packing_reward_zero_when_incorrect():
    task = CirclePackingTask()
    reward = task.compute_reward({"correctness": 0.0, "raw_score": 2.5})
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_circle_packing_initial_state_has_zero_value():
    task = CirclePackingTask()
    state = task.make_initial_state()
    assert state.value == 0.0
    assert state.task_payload["construction"] == []
    assert state.task_payload["code"] == ""


# ---------------------------------------------------------------------------
# make_next_state: value = raw_score (positive, maximize)
# ---------------------------------------------------------------------------


def test_circle_packing_next_state_value_equals_raw_score():
    task = CirclePackingTask()
    next_state = task.make_next_state(
        parent_state=ArchiveNode(epoch=0, value=0.0, task_payload={"construction": []}),
        parsed_code="def run_packing():\n    pass",
        eval_output={
            "correctness": 1.0,
            "raw_score": 3.14,
            "performance": 3.14,
            "result_payload": {"result_construction": []},
            "stdout": "ok",
        },
        epoch=1,
    )
    assert next_state is not None
    assert next_state.value == pytest.approx(3.14)


def test_circle_packing_next_state_none_when_incorrect():
    task = CirclePackingTask()
    next_state = task.make_next_state(
        parent_state=ArchiveNode(epoch=0, value=0.0, task_payload={"construction": []}),
        parsed_code="def run_packing():\n    pass",
        eval_output={
            "correctness": 0.0,
            "raw_score": 3.14,
            "performance": 3.14,
            "result_payload": {},
            "stdout": "",
        },
        epoch=1,
    )
    assert next_state is None


# ---------------------------------------------------------------------------
# Validity: only requires value is not None
# ---------------------------------------------------------------------------


def test_circle_packing_validity_accepts_any_with_value():
    task = CirclePackingTask()
    state = ArchiveNode(epoch=0, value=1.0, task_payload={"construction": []})
    assert task.is_state_valid(state)


def test_circle_packing_validity_rejects_none_value():
    task = CirclePackingTask()
    state = ArchiveNode(epoch=0, value=None, task_payload={"construction": []})
    assert not task.is_state_valid(state)


# ---------------------------------------------------------------------------
# No dedup key (circle packing construction is always [])
# ---------------------------------------------------------------------------


def test_circle_packing_dedupe_key_is_none():
    task = CirclePackingTask()
    state = ArchiveNode(epoch=0, value=1.0, task_payload={"construction": []})
    assert task.dedupe_key(state) is None


# ---------------------------------------------------------------------------
# No refresh_initial_state (no construction_length_limits)
# ---------------------------------------------------------------------------


def test_circle_packing_has_no_refresh():
    assert not hasattr(CirclePackingTask, "refresh_initial_state")


# ---------------------------------------------------------------------------
# Resource requirements
# ---------------------------------------------------------------------------


def test_circle_packing_declares_one_cpu_per_evaluation():
    task = CirclePackingTask()
    assert task.evaluation_resources() == TaskEvaluationRequirements(cpus_per_eval=1)


# ---------------------------------------------------------------------------
# maximize direction
# ---------------------------------------------------------------------------


def test_circle_packing_maximize_raw_score_is_true():
    assert CirclePackingTask.maximize_raw_score is True


# ---------------------------------------------------------------------------
# Timeout constant
# ---------------------------------------------------------------------------


def test_circle_packing_eval_timeout_is_530():
    assert CIRCLE_PACKING_EVAL_TIMEOUT_SECONDS == 530


# ---------------------------------------------------------------------------
# num_circles from env
# ---------------------------------------------------------------------------


def test_circle_packing_default_num_circles_is_26():
    task = CirclePackingTask()
    assert task.num_circles == 26


# ---------------------------------------------------------------------------
# Worker: quick failure for bad code
# ---------------------------------------------------------------------------


def test_circle_packing_worker_failure_returns_quickly():
    import time

    started = time.perf_counter()
    result = evaluate_candidate_code(
        code="print('hello')",
        num_circles=26,
        timeout_s=5,
        seed=0,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0
    assert result["correctness"] == 0.0
    assert "run_packing" in str(result["msg"])
