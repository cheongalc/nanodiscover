"""Tests for FLOPs estimation module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.estimate_flops import (
    estimate_generation_flops,
    estimate_reference_scoring_flops,
    estimate_training_flops,
    combine_reports,
    FLOPsReport,
)


def test_generation_flops_qwen3_8b() -> None:
    """Verify generation FLOPs for a typical Qwen3-8B epoch.

    Qwen3-8B has ~8.03e9 parameters.
    512 rollouts * 8192 tokens avg = 4,194,304 total tokens.
    Expected: 2 * 8.03e9 * 4194304 ≈ 6.73e16 FLOPs.
    """

    report = estimate_generation_flops(
        model_params=8.03e9,
        total_tokens=512 * 8192,
    )

    assert report.stage == "generation"
    assert report.model_params == 8.03e9
    assert report.total_tokens == 512 * 8192

    # 2 * 8.03e9 * 4194304 = 6.7359e16
    expected = 2.0 * 8.03e9 * (512 * 8192)
    assert abs(report.estimated_flops - expected) < 1e10  # tight tolerance


def test_training_flops_qwen3_8b() -> None:
    """Verify training FLOPs for a typical Qwen3-8B training step.

    512 rollouts * 4096 tokens avg packed = 2,097,152 total tokens.
    Expected: 6 * 8.03e9 * 2097152 ≈ 1.01e17 FLOPs.
    """

    report = estimate_training_flops(
        model_params=8.03e9,
        total_tokens=512 * 4096,
    )

    assert report.stage == "training"
    expected = 6.0 * 8.03e9 * (512 * 4096)
    assert abs(report.estimated_flops - expected) < 1e10


def test_reference_scoring_flops() -> None:
    """Verify reference scoring FLOPs (same as generation: forward only)."""

    report = estimate_reference_scoring_flops(
        model_params=8.03e9,
        total_tokens=512 * 8192,
    )

    assert report.stage == "reference_scoring"
    expected = 2.0 * 8.03e9 * (512 * 8192)
    assert abs(report.estimated_flops - expected) < 1e10


def test_combine_reports() -> None:
    """Verify that combine_reports sums correctly."""

    gen = estimate_generation_flops(8e9, 1000)
    ref = estimate_reference_scoring_flops(8e9, 1000)
    train = estimate_training_flops(8e9, 500)

    combined = combine_reports(gen, ref, train)

    expected_total = (2 * 8e9 * 1000) + (2 * 8e9 * 1000) + (6 * 8e9 * 500)
    assert abs(combined["total_flops"] - expected_total) < 1e5
    assert "generation" in combined["stages"]
    assert "reference_scoring" in combined["stages"]
    assert "training" in combined["stages"]


def test_report_serialization() -> None:
    """Verify FLOPsReport.to_dict() produces clean output."""

    report = FLOPsReport(
        stage="test",
        model_params=1e9,
        total_tokens=100,
        estimated_flops=2e11,
        details={"formula": "test"},
    )
    d = report.to_dict()
    assert d["stage"] == "test"
    assert d["estimated_flops"] == 2e11
    assert "formula" in d["details"]


def test_zero_tokens() -> None:
    """Edge case: zero tokens should produce zero FLOPs."""

    report = estimate_generation_flops(8e9, 0)
    assert report.estimated_flops == 0.0
    assert report.total_tokens == 0


if __name__ == "__main__":
    test_generation_flops_qwen3_8b()
    test_training_flops_qwen3_8b()
    test_reference_scoring_flops()
    test_combine_reports()
    test_report_serialization()
    test_zero_tokens()
    print("All FLOPs tests passed!")
