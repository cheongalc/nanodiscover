from __future__ import annotations

from typing import Any

from core.archive import ArchiveNode


ERDOS_TARGET = 0.38080
ERDOS_METRIC_NAME = "C\u2085 bound"
ERDOS_IS_MAXIMIZE = False


def format_fenced_python(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("```python"):
        return stripped
    return f"```python\n{stripped}\n```"


def build_erdos_state_context(
    state: ArchiveNode,
    *,
    code: str,
) -> str:
    value_ctx = f"You are iteratively optimizing {ERDOS_METRIC_NAME}."
    improvement_direction = "higher" if ERDOS_IS_MAXIMIZE else "lower"

    has_code = code and code.strip()
    if has_code:
        value_ctx += "\nHere is the last code we ran:\n"
        value_ctx += code if code.strip().startswith("```") else format_fenced_python(code)
    else:
        value_ctx += "\nNo previous code available."

    if state.value is not None:
        current_value = state.value if ERDOS_IS_MAXIMIZE else -state.value
        current_gap = ERDOS_TARGET - current_value if ERDOS_IS_MAXIMIZE else current_value - ERDOS_TARGET
        value_ctx += f"\nCurrent best {ERDOS_METRIC_NAME} ({improvement_direction} is better): {current_value:.6f}"
        value_ctx += (
            f"\nTarget: {ERDOS_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    else:
        value_ctx += f"\nTarget {ERDOS_METRIC_NAME}: {ERDOS_TARGET}"

    stdout = str(state.task_payload.get("stdout") or "").strip()
    if stdout:
        if len(stdout) > 500:
            stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    return value_ctx


def render_erdos_prompt(
    *,
    state_ctx: str,
    code: str,
    budget_s: int,
) -> str:
    # Construct code section
    if code and code.strip():
        code_section = '''Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.
Unless you make a meaningful improvement, you will not be rewarded.'''
    else:
        code_section = '''Write code to optimize this construction.'''

    return f'''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: {budget_s}s for your code to run
- **CPUs**: 1 available

## Rules
- Define `run(seed=42, budget_s={budget_s}, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` is pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

{state_ctx}
{code_section}
'''
