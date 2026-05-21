from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.evaluator import AllocatedEvaluationResources


AC2_CPUS_PER_EVAL = 2
# Match ttt-discover paper/evaluator resource envelope: 2 CPUs, 1100s timeout.
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _np():
    import numpy as np

    return np


def normalize_ac2_resources(resources: "AllocatedEvaluationResources | None") -> tuple[tuple[int, ...], int]:
    cpu_ids = tuple(int(cpu_id) for cpu_id in getattr(resources, "cpu_ids", ())[:AC2_CPUS_PER_EVAL])
    thread_limit = max(1, len(cpu_ids) if cpu_ids else AC2_CPUS_PER_EVAL)
    return cpu_ids, thread_limit


def apply_worker_resource_limits(cpu_ids: tuple[int, ...], thread_limit: int) -> None:
    for env_name in THREAD_LIMIT_ENV_VARS:
        os.environ[env_name] = str(max(1, int(thread_limit)))
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))


def evaluate_sequence(sequence: list[float]) -> float:
    np = _np()
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


def evaluate_candidate_in_worker(
    code: str,
    parent_construction: list[float] | None,
    seed: int,
    result_conn: Connection,
    cpu_ids: tuple[int, ...],
    thread_limit: int,
) -> None:
    apply_worker_resource_limits(cpu_ids, thread_limit)
    if hasattr(os, "sched_getaffinity"):
        eff_ids = tuple(sorted(int(x) for x in os.sched_getaffinity(0)))
    else:
        eff_ids = ()
    logger.info(
        "ac2_eval_worker_resources seed=%d effective_cpus=%d effective_cpu_ids=%s blas_omp_thread_cap=%d requested_cpu_ids=%s",
        int(seed),
        len(eff_ids),
        eff_ids,
        int(thread_limit),
        cpu_ids,
    )
    np = _np()
    stdout = io.StringIO()
    stderr = io.StringIO()

    def combined_output() -> str:
        out = stdout.getvalue()
        err = stderr.getvalue()
        if out and err:
            return f"{out}\n\n[stderr]\n{err}"
        if err:
            return f"[stderr]\n{err}"
        return out

    def send_result(payload: dict[str, Any]) -> None:
        try:
            result_conn.send(payload)
        except Exception:
            # If the parent has already stopped waiting, there is nothing else to do.
            pass

    try:
        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "np": np,
            "evaluate_sequence": evaluate_sequence,
        }
        if parent_construction is not None:
            namespace["height_sequence_1"] = np.array(parent_construction, dtype=float)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, namespace, namespace)
            if "construct_function" not in namespace:
                raise ValueError("Generated code must define construct_function")
            # Match ttt-discover: call construct_function() with no args.
            candidate = namespace["construct_function"]()
        # Match ttt-discover: only list outputs are accepted (numpy arrays are invalid).
        if not isinstance(candidate, list):
            raise TypeError("construct_function must return list[float]")
        bound = evaluate_sequence(candidate)
        send_result(
            {
                "ok": True,
                "bound": float(bound),
                "construction": candidate,
                "stdout": combined_output(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        send_result({"ok": False, "msg": str(exc), "stdout": combined_output()})
    finally:
        result_conn.close()


def evaluate_candidate_code(
    code: str,
    parent_construction: list[float] | None,
    timeout_s: int,
    budget_s: int,
    seed: int,
    resources: "AllocatedEvaluationResources | None" = None,
) -> dict[str, Any]:
    _ = budget_s  # Not forwarded; match original ttt-discover which calls construct_function() with no args.
    cpu_ids, thread_limit = normalize_ac2_resources(resources)
    pool_slot = getattr(resources, "slot_index", None) if resources is not None else None
    logger.info(
        "ac2_eval_allocate seed=%d allocated_cpus=%d allocated_cpu_ids=%s blas_omp_thread_cap=%d resource_pool_slot=%s",
        int(seed),
        len(cpu_ids),
        cpu_ids,
        int(thread_limit),
        pool_slot,
    )
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=evaluate_candidate_in_worker,
        args=(code, parent_construction, seed, child_conn, cpu_ids, thread_limit),
    )
    process.start()
    child_conn.close()
    process.join(timeout=timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        process.close()
        parent_conn.close()
        return {
            "score": 0.0,
            "msg": f"timeout after {timeout_s}s",
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": 0.0,
            "result_payload": {},
            "stdout": f"[evaluator]\ntimeout after {timeout_s}s\n",
        }

    if not parent_conn.poll(1.0):
        exitcode = process.exitcode
        process.close()
        parent_conn.close()
        return {
            "score": 0.0,
            "msg": f"empty evaluator result (exitcode={exitcode})",
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": 0.0,
            "result_payload": {},
            "stdout": f"[evaluator]\nempty evaluator result (exitcode={exitcode})\n",
        }
    result = parent_conn.recv()
    process.close()
    parent_conn.close()

    if not result.get("ok", False):
        message = str(result.get("msg", "evaluation failed"))
        stdout_text = str(result.get("stdout", ""))
        if not stdout_text.strip():
            stdout_text = f"[evaluator]\n{message}\n"
        return {
            "score": 0.0,
            "msg": message,
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": 0.0,
            "result_payload": {},
            "stdout": stdout_text,
        }

    bound = float(result["bound"])
    return {
        "score": bound,
        "msg": f"Success; raw_score={bound}",
        "correctness": 1.0,
        "performance": bound,
        "raw_score": bound,
        "result_payload": {"result_construction": result["construction"]},
        "stdout": str(result.get("stdout", "")),
    }
