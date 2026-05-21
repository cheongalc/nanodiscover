from __future__ import annotations

import os
from typing import Any

from core.archive import ArchiveNode


AC1_TARGET = 1.5030
AC1_METRIC_NAME = "upper bound"
AC1_IS_MAXIMIZE = False

example_ae_program_random_init = (
    lambda num_seconds: f'''
import numpy as np
import time
from scipy import optimize
linprog = optimize.linprog


def get_good_direction_to_move_into(sequence):
    """Returns a better direction using LP to find g with larger sum while keeping conv bounded."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_g = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
    t = 0.01
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    return new_sequence


def solve_convolution_lp(f_sequence, rhs):
    """Solves the LP: maximize sum(b) s.t. conv(f, b) <= rhs, b >= 0."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)
    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, options={{
        'time_limit': 10.0,   # seconds, make sure we don't get stuck
        'disp': False,
    }})
    if result.success:
        return result.x
    return None


def propose_candidate(seed=42, budget_s={num_seconds}, **kwargs):
    np.random.seed(seed)
    deadline = time.time() + budget_s - 10

    if np.random.rand() < 0.5:
        # Start from the SOTA sequence (already available as height_sequence_1)
        best_sequence = list(height_sequence_1)
    else:
        # Start from random initialization, could help if height_sequence_1 is a local minimum
        best_sequence = [np.random.random()] * np.random.randint(100, 1000)
    curr_sequence = best_sequence.copy()
    best_score = evaluate_sequence(best_sequence)

    while time.time() < deadline:
        h_function = get_good_direction_to_move_into(curr_sequence)
        if h_function is None:
            # Random perturbation if LP fails
            idx = np.random.randint(len(curr_sequence))
            curr_sequence[idx] = max(0, curr_sequence[idx] + np.random.randn() * 0.01)
        else:
            curr_sequence = h_function

        try:
            curr_score = evaluate_sequence(curr_sequence)
            if curr_score < best_score:
                best_score = curr_score
                best_sequence = curr_sequence.copy()
        except:
            pass

    return best_sequence
''')


AC1_LITERATURE = r"""A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.
```latex
Starting from a nonnegative step function $f=(a_0,\dots,a_{n-1})$ normalized so that $\sum_j a_j=\sqrt{2n}$, set $M=\|f*f\|_\infty$. Next compute $g_0=(b_0,\dots,b_{n-1})$ by solving a linear program, i.e.\ maximizing $\sum_j b_j$ subject to $b_j\ge0$ and $\|f*g_0\|_\infty\le M$; as is standard, the optimum is attained at an extreme point determined by an active set of binding inequalities, here corresponding to important constraints where the convolution bound $(f*g_0)(x)\le M$ is tight and limiting. Rescale $g_0$ to match the normalization, $g=\frac{\sqrt{2n}}{\sum_j b_j}g_0$, and update $f\leftarrow (1-t)f+t g$ for a small $t>0$. Repeating this step produces a sequence with nonincreasing $\|f*f\|_\infty$, and the iteration is continued until it stabilizes.
```"""


AC1_EVAL_FUNCTION = '''```python
import numpy as np

def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    # --- Security Checks ---

    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))
```'''


def format_fenced_python(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("```python"):
        return stripped
    return f"```python\n{stripped}\n```"


def default_initial_code(budget_s: int) -> str:
    return format_fenced_python(example_ae_program_random_init(budget_s))


def build_ac1_state_context(
    state: ArchiveNode,
    *,
    code: str,
    construction: list[float] | None,
) -> str:
    value_ctx = f"You are iteratively optimizing {AC1_METRIC_NAME}."
    improvement_direction = "higher" if AC1_IS_MAXIMIZE else "lower"

    has_code = code and code.strip()
    if has_code:
        value_ctx += "\nHere is the last code we ran:\n"
        value_ctx += code if code.strip().startswith("```") else format_fenced_python(code)
    else:
        value_ctx += "\nNo previous code available."

    if state.parent_values and state.value is not None and construction:
        before_value = state.parent_values[0] if AC1_IS_MAXIMIZE else -state.parent_values[0]
        after_value = state.value if AC1_IS_MAXIMIZE else -state.value
        current_gap = AC1_TARGET - after_value if AC1_IS_MAXIMIZE else after_value - AC1_TARGET
        value_ctx += (
            f"\nHere is the {AC1_METRIC_NAME} before and after running the code above "
            f"({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
        )
        value_ctx += (
            f"\nTarget: {AC1_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    elif state.value is not None:
        after_value = state.value if AC1_IS_MAXIMIZE else -state.value
        current_gap = AC1_TARGET - after_value if AC1_IS_MAXIMIZE else after_value - AC1_TARGET
        value_ctx += f"\nCurrent {AC1_METRIC_NAME} ({improvement_direction} is better): {after_value:.6f}"
        value_ctx += (
            f"\nTarget: {AC1_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    else:
        value_ctx += f"\nTarget {AC1_METRIC_NAME}: {AC1_TARGET}"

    stdout = str(state.task_payload.get("stdout") or "").strip()
    if stdout:
        if len(stdout) > 500:
            stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    if construction:
        value_ctx += f"\nLength of the construction: {len(construction)}"
    return value_ctx


def render_ac1_prompt(
    *,
    state_ctx: str,
    budget_s: int,
) -> str:
    return f'''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

{AC1_EVAL_FUNCTION}

{AC1_LITERATURE}

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have {budget_s} seconds to run, and after that it has to have returned the best sequence it found. If after {budget_s} seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with {budget_s}s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

{state_ctx}

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable. 
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.

Rules:
- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''