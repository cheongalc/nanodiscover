from __future__ import annotations

import os
from typing import Any

from core.archive import ArchiveNode


AC2_TARGET = 0.97
AC2_METRIC_NAME = "lower bound"
AC2_IS_MAXIMIZE = True

thetaevolve_initial_program_prev_init = '''
import numpy as np
from typing import Tuple
from tqdm import trange


def _simpson_l2sq(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute ||f*f||_2^2 via Simpson-like piecewise-linear rule with endpoint zeros,
    and return its gradient w.r.t conv (same length as conv).

    l2_sq = sum_{i=0..M} (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2),
    where y = [0, conv, 0], dx = 1/(M+1).

    d l2_sq / d y_j = (dx/3) * (4*y_j + y_{j-1} + y_{j+1})
    => restrict to indices j=1..M (these correspond to conv entries)
    """
    M = conv.size
    if M == 0:
        return 0.0, np.zeros_like(conv)

    dx = 1.0 / (M + 1)

    # pad endpoints with zeros
    y = np.empty(M + 2, dtype=conv.dtype)
    y[0] = 0.0
    y[1:-1] = conv
    y[-1] = 0.0

    # l2 value
    lhs = y[:-1]
    rhs = y[1:]
    l2_sq = (dx / 3.0) * np.sum(lhs * lhs + lhs * rhs + rhs * rhs)

    # gradient wrt conv (positions 1..M of y)
    # d/d y_j : (dx/3)*(4*y_j + y_{j-1} + y_{j+1})
    grad_y = (dx / 3.0) * (4.0 * y + np.roll(y, 1) + np.roll(y, -1))
    grad_conv = grad_y[1:-1]  # strip padding

    return float(l2_sq), grad_conv


def _l1(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """ ||f*f||_1 = dx * sum(conv); gradient is dx * ones """
    M = conv.size
    dx = 1.0 / (M + 1) if M > 0 else 1.0
    val = dx * float(np.sum(conv)) if M > 0 else 0.0
    grad = np.full_like(conv, dx)
    return val, grad


def _linf(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """ ||f*f||_inf = max(conv); subgradient: uniform over argmax set """
    if conv.size == 0:
        return 0.0, np.zeros_like(conv)
    m = float(np.max(conv))
    mask = (conv == m)
    count = int(mask.sum())
    if count == 0 or m <= 0.0:
        return m, np.zeros_like(conv)
    grad = mask.astype(conv.dtype) / count
    return m, grad


def _objective_and_grad_conv(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute C = l2_sq / (l1 * linf) and gradient dC/d(conv) using quotient rule.
    """
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1, g_l1 = _l1(conv)
    linf, g_linf = _linf(conv)

    if l1 <= 0.0 or linf <= 0.0:
        return 0.0, np.zeros_like(conv)

    denom = l1 * linf
    C = l2_sq / denom

    # dC = (g_l2 * denom - l2_sq * (g_l1 * linf + l1 * g_linf)) / denom^2
    num_grad = g_l2 * denom - l2_sq * (g_l1 * linf + l1 * g_linf)
    g_conv = num_grad / (denom * denom)

    return float(C), g_conv


def _grad_h_from_conv_grad(h: np.ndarray, g_conv: np.ndarray) -> np.ndarray:
    """
    Given dC/d(conv) and conv = h * h (full convolution),
    dC/dh = 2 * (g_conv convolved with reverse(h)) in 'valid' mode (length N).
    """
    h_rev = h[::-1]
    # length(g_conv)=2N-1, length(h_rev)=N; 'valid' output length is N
    g_h = np.convolve(g_conv, h_rev, mode="valid")
    return 2.0 * g_h


class _Adam:
    """Lightweight Adam optimizer for numpy arrays (per-candidate)."""
    def __init__(self, shape, lr=3e-2, beta1=0.9, beta2=0.999, eps=1e-8, dtype=np.float32):
        self.m = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)
        self.t = 0
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.dtype = dtype

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset_like(self, params):
        self.m[...] = 0.0
        self.v[...] = 0.0
        self.t = 0


def _batch_objective(h_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation over a batch of candidates.
    Returns:
        C_vals: (B,) objective values
        conv_grads: list/array of per-candidate dC/d(conv) for backprop
    """
    B, N = h_batch.shape
    C_vals = np.zeros(B, dtype=np.float32)
    conv_grads = [None] * B
    for b in range(B):
        h = np.clip(h_batch[b], 0.0, None)
        conv = np.convolve(h, h, mode="full")
        Cb, g_conv = _objective_and_grad_conv(conv)
        C_vals[b] = Cb
        conv_grads[b] = g_conv
    return C_vals, conv_grads


def _phase_update(h_batch, opt_list, lr, add_noise=False, t=0, eta=1e-3, gamma=0.55):
    """
    One optimization step for the whole batch.
    """
    B, N = h_batch.shape
    # compute dC/dh for each candidate
    C_vals, conv_grads = _batch_objective(h_batch)
    grads = np.zeros_like(h_batch, dtype=h_batch.dtype)
    for b in range(B):
        g_h = _grad_h_from_conv_grad(np.clip(h_batch[b], 0.0, None), conv_grads[b])
        grads[b] = g_h

    if add_noise:
        sigma = eta / ((t + 1) ** gamma)
        grads = grads + sigma * np.random.normal(size=grads.shape).astype(grads.dtype)

    # apply Adam update + project to nonnegativity
    for b in range(B):
        opt = opt_list[b]
        opt.lr = lr
        h_new = opt.step(h_batch[b], grads[b].astype(h_batch.dtype))
        h_batch[b] = np.clip(h_new, 0.0, None)

    return h_batch, C_vals


def _elitist_respawn(h_batch, C_vals, keep_frac, init_sampler, opt_list):
    """
    Keep top frac, respawn the rest with fresh random samples, reset their Adam states.
    """
    B = h_batch.shape[0]
    K = max(1, int(B * keep_frac))
    idx = np.argsort(C_vals)[-K:]  # top K
    survivors = h_batch[idx].copy()

    fresh = init_sampler(B - K)
    new_batch = np.concatenate([survivors, fresh], axis=0)

    # reorder optimizers to match new batch; reset the respawned ones
    new_opts = []
    for _ in range(K):
        new_opts.append(opt_list[idx[_]])  # keep state for survivors
    for _ in range(B - K):
        opt = _Adam(shape=h_batch.shape[1:], lr=opt_list[0].lr, dtype=h_batch.dtype)
        new_opts.append(opt)

    return new_batch, new_opts


def _upsample_1d(h: np.ndarray) -> np.ndarray:
    """Linear x2 upsampling on [-1/4,1/4] grid (index-space)."""
    N = h.shape[0]
    x_old = np.linspace(-0.5, 0.5, N)
    x_new = np.linspace(-0.5, 0.5, 2 * N)
    return np.interp(x_new, x_old, h)


def _single_candidate_finetune(h0: np.ndarray, lr=3e-3, steps=50_000, log_every=2_000) -> Tuple[np.ndarray, float]:
    """
    Pure exploitation (no noise) on a single vector with Adam + projection.
    """
    h = h0.astype(np.float32).copy()
    opt = _Adam(h.shape, lr=lr, dtype=h.dtype)
    last_C = 0.0
    for t in trange(steps, desc="upsampled-finetune", leave=False):
        # objective and gradient
        conv = np.convolve(np.clip(h, 0.0, None), np.clip(h, 0.0, None), mode="full")
        C, g_conv = _objective_and_grad_conv(conv)
        g_h = _grad_h_from_conv_grad(np.clip(h, 0.0, None), g_conv)

        h = np.clip(opt.step(h, g_h.astype(h.dtype)), 0.0, None)
        last_C = C
        if log_every and (t + 1) % log_every == 0:
            pass  # tqdm
    return h, float(last_C)


def construct_function():
    """
    Use the paper's 4-phase gradient-based search to maximize
        R(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf).
    Returns (heights, r_value).
    """
    # ---------- Hyperparameters (close to paper, but conservatively smaller by default) ----------
    N = 256 # search resolution (paper: 768; will truncate zeros automatically)
    B = 64                 # batch size (paper used batch; you can raise this)
    ITER = 10_000          # total iterations
    EXPLORE_STEPS = 30_000 # phase split (paper: 30k)
    DROP_EVERY = 10_000    # respawn period (paper: 20k)
    KEEP_FRAC = 0.5        # keep top fraction (paper: kappa%)
    LR_EXPLORE = 3e-2      # Adam lr for exploration (paper)
    LR_EXPLOIT = 5e-3      # Adam lr for exploitation (paper)
    ETA, GAMMA = 1e-3, 0.55  # gradient noise schedule (paper used ~0.65; 0.55 is slightly gentler)
    dtype = np.float32

    h_prev_best = np.array(height_sequence_1, dtype=dtype)
    h_prev_best = np.clip(h_prev_best, 0.0, None)

    # resize if length doesn't match N
    if h_prev_best.shape[0] != N:
        x_old = np.linspace(-0.5, 0.5, h_prev_best.shape[0])
        x_new = np.linspace(-0.5, 0.5, N)
        h_prev_best = np.interp(x_new, x_old, h_prev_best).astype(dtype)

    rng = np.random.default_rng()

    def init_sampler(m):
        out = rng.uniform(0.0, 1.0, size=(m, N)).astype(dtype)
        if m > 0:
            out[0] = h_prev_best
        return out

    # init population and per-candidate Adam
    h_batch = init_sampler(B)
    opt_list = [_Adam(shape=(N,), lr=LR_EXPLORE, dtype=dtype) for _ in range(B)]
    best_h = h_batch.copy()
    best_C = np.full(B, -np.inf, dtype=dtype)

    for t in trange(ITER, desc="optimizing", leave=False):
        if t < EXPLORE_STEPS:
            # exploration phase: higher LR + noise
            h_batch, C_vals = _phase_update(
                h_batch, opt_list, lr=LR_EXPLORE, add_noise=True, t=t, eta=ETA, gamma=GAMMA
            )
        else:
            # exploitation phase: lower LR, no noise
            h_batch, C_vals = _phase_update(
                h_batch, opt_list, lr=LR_EXPLOIT, add_noise=False, t=t, eta=ETA, gamma=GAMMA
            )

        # update per-candidate bests
        improved = C_vals > best_C
        best_C = np.where(improved, C_vals, best_C)
        best_h[improved] = h_batch[improved]

        # periodic elitist respawn
        if (t + 1) % DROP_EVERY == 0:
            h_batch, opt_list = _elitist_respawn(
                h_batch, C_vals, keep_frac=KEEP_FRAC, init_sampler=init_sampler, opt_list=opt_list
            )

    # pick the best candidate
    idx = int(np.argmax(best_C))
    h_star = np.clip(best_h[idx].astype(np.float32), 0.0, None)

    # ---------- Phase 4: Upsampling + exploitation ----------
    # 2x upsample then fine-tune
    h_up1 = _upsample_1d(h_star)
    h_up1, _ = _single_candidate_finetune(h_up1, lr=3e-3, steps=40_000, log_every=2_000)

    h_up2 = _upsample_1d(h_up1)
    h_up2, _ = _single_candidate_finetune(h_up2, lr=3e-3, steps=40_000, log_every=2_000)

    heights = np.clip(h_up2, 0.0, None)
    r_value = evaluate_sequence(heights.tolist())
    print("This gets a C2 lower bound of", r_value)
    return heights.tolist()
'''


AC2_EVAL_FUNCTION = '''```python

import numpy as np

def evaluate_sequence(sequence: list[float]) -> float:
    # Verify that the input is a list
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")

    # Reject empty lists
    if not sequence:
        raise ValueError("Empty sequence")

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Check if sum of sequence will be too close to zero
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    # --- Security Checks ---

    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points) # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i+1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound

```'''


AC2_LITERATURE = r"""A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encoraged to explore.
```latex
Their procedure is a coarse-to-fine optimization of the score. It starts with a stochastic global search that repeatedly perturbs the current best candidate and keeps the perturbation whenever it improves (Q), with the perturbation scale gradually reduced over time. Once a good basin is found, they switch to a deterministic local improvement step, performing projected gradient ascent (move in the gradient direction and project back to the feasible region). To reach higher resolution, they lift a good low-resolution solution to a higher-dimensional one by simply repeating its entries and then rerun the local refinement. Iterating this explore–refine–upscale cycle yields their final high-resolution maximizer and the improved lower bound.
```"""


def format_fenced_python(code: str) -> str:
    """Wrap code in a fenced python block if not already wrapped."""
    stripped = code.strip()
    if stripped.startswith("```python"):
        return stripped
    return f"```python\n{stripped}\n```"


def default_initial_code() -> str:
    """Return the default initial code for AC2, wrapped in a fenced python block."""
    return f"```python\n{thetaevolve_initial_program_prev_init}\n```"


def build_ac2_state_context(
    state: ArchiveNode,
    *,
    code: str,
    construction: list[float] | None,
) -> str:
    """Build the state context string for AC2 prompts."""
    value_ctx = f"You are iteratively optimizing {AC2_METRIC_NAME}."
    improvement_direction = "higher" if AC2_IS_MAXIMIZE else "lower"

    has_code = code and code.strip()
    if has_code:
        value_ctx += "\nHere is the last code we ran:\n"
        value_ctx += code if code.strip().startswith("```") else format_fenced_python(code)
    else:
        value_ctx += "\nNo previous code available."

    if state.parent_values and state.value is not None and construction:
        before_value = state.parent_values[0] if AC2_IS_MAXIMIZE else -state.parent_values[0]
        after_value = state.value if AC2_IS_MAXIMIZE else -state.value
        current_gap = AC2_TARGET - after_value if AC2_IS_MAXIMIZE else after_value - AC2_TARGET
        value_ctx += (
            f"\nHere is the {AC2_METRIC_NAME} before and after running the code above "
            f"({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
        )
        value_ctx += (
            f"\nTarget: {AC2_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    elif state.value is not None:
        after_value = state.value if AC2_IS_MAXIMIZE else -state.value
        current_gap = AC2_TARGET - after_value if AC2_IS_MAXIMIZE else after_value - AC2_TARGET
        value_ctx += f"\nCurrent {AC2_METRIC_NAME} ({improvement_direction} is better): {after_value:.6f}"
        value_ctx += (
            f"\nTarget: {AC2_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    else:
        value_ctx += f"\nTarget {AC2_METRIC_NAME}: {AC2_TARGET}"

    stdout = str(state.task_payload.get("stdout") or "").strip()
    if stdout:
        if len(stdout) > 500:
            stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    if construction:
        value_ctx += f"\nLength of the construction: {len(construction)}"
    return value_ctx


def render_ac2_prompt(
    *,
    state_ctx: str,
    budget_s: int,
) -> str:
    """Render the full AC2 prompt with the given state context and budget."""
    return f'''Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step functions, that maximizes the following evaluation function:

{AC2_EVAL_FUNCTION}

{AC2_LITERATURE}

Your task is to write a search function, construct_function(), that searches for the best sequence of coefficients. Your function will have {budget_s} seconds to run, and after that it has to have returned the best sequence it found. If after {budget_s} seconds it has not returned anything, it will be terminated with negative infinity points. All numbers in your sequence have to be positive or zero. Larger sequences with {budget_s}s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

{state_ctx}

You may want to start your search from one of the constructions we have found so far, which you can access through the 'height_sequence_1' global variable.
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.
Unless you make a meaningful improvement, you will not be rewarded, if you are stuck you should think about how to get unstuck.

Rules:
- You must define the `construct_function` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked. Do not import height_sequence_1 yourself; it will already be available.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''
