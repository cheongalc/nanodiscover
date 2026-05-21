from __future__ import annotations

import inspect
import os
from typing import Any

from core.archive import ArchiveNode
from tasks.circle_packing.evaluator import validate_packing


CIRCLE_PACKING_METRIC_NAME = "sum of radii"
CIRCLE_PACKING_IS_MAXIMIZE = True


def _get_target(num_circles: int) -> float:
    if num_circles == 26:
        return 2.636
    elif num_circles == 32:
        return 2.940
    else:
        # Reasonable fallback for other values
        return 0.0


def format_fenced_python(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("```python"):
        return stripped
    return f"```python\n{stripped}\n```"


def build_circle_packing_state_context(
    state: ArchiveNode,
    *,
    code: str,
    num_circles: int,
) -> str:
    target = _get_target(num_circles)
    value_ctx = f"You are iteratively optimizing {CIRCLE_PACKING_METRIC_NAME}."
    improvement_direction = "higher" if CIRCLE_PACKING_IS_MAXIMIZE else "lower"

    has_code = code and code.strip()
    if has_code:
        value_ctx += "\nHere is the last code we ran:\n"
        value_ctx += code if code.strip().startswith("```") else format_fenced_python(code)
    else:
        value_ctx += "\nNo previous code available."

    if state.parent_values and state.value is not None:
        before_value = state.parent_values[0] if CIRCLE_PACKING_IS_MAXIMIZE else -state.parent_values[0]
        after_value = state.value if CIRCLE_PACKING_IS_MAXIMIZE else -state.value
        current_gap = target - after_value if CIRCLE_PACKING_IS_MAXIMIZE else after_value - target
        value_ctx += (
            f"\nHere is the {CIRCLE_PACKING_METRIC_NAME} before and after running the code above "
            f"({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
        )
        if target > 0:
            value_ctx += (
                f"\nTarget: {target}. Current gap: {current_gap:.6f}. "
                "Further improvements will also be generously rewarded."
            )
    elif state.value is not None:
        after_value = state.value if CIRCLE_PACKING_IS_MAXIMIZE else -state.value
        value_ctx += f"\nCurrent {CIRCLE_PACKING_METRIC_NAME} ({improvement_direction} is better): {after_value:.6f}"
        if target > 0:
            current_gap = target - after_value if CIRCLE_PACKING_IS_MAXIMIZE else after_value - target
            value_ctx += (
                f"\nTarget: {target}. Current gap: {current_gap:.6f}. "
                "Further improvements will also be generously rewarded."
            )
    else:
        if target > 0:
            value_ctx += f"\nTarget {CIRCLE_PACKING_METRIC_NAME}: {target}"

    stdout = str(state.task_payload.get("stdout") or "").strip()
    if stdout:
        if len(stdout) > 500:
            stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    return value_ctx


def render_circle_packing_prompt(
    *,
    state_ctx: str,
    num_circles: int,
) -> str:
    validator_src = inspect.getsource(validate_packing)
    return f"""You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack {num_circles} circles in a unit square [0,1]\u00d7[0,1] to maximize the sum of radii.

We will run the below validation function (read-only, do not modify this):
```python
{validator_src}
```

{state_ctx}

Reason about how you could further improve this packing. Consider:
- Are circles placed optimally near boundaries and corners?
- Could a different arrangement (hexagonal, nested, hybrid) yield better results?
- Are there gaps that could be filled with repositioned or resized circles?
- Could optimization parameters or methods be improved?

Rules:
- You must define the run_packing function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape ({num_circles}, 2) and radii has shape ({num_circles},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- You need to get really creative and think from first principles.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""
