from __future__ import annotations

import contextlib
import inspect
import io
import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.evaluator import AllocatedEvaluationResources


ERDOS_CPUS_PER_EVAL = 1
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    # Upstream TTT returned run()'s c5_bound after verify; with atol=1e-4 the model
    # can report an optimistically low C5 that still passes. Score from computed C5
    # so raw_score, archive, and global-best tracking use the true bound.
    computed_c5 = verify_c5_solution(h_values, c5_bound, n_points)
    return float(computed_c5)


def verify_erdos_solution(result: tuple[np.ndarray, float, int]) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


def normalize_erdos_resources(resources: "AllocatedEvaluationResources | None") -> tuple[tuple[int, ...], int]:
    cpu_ids = tuple(int(cpu_id) for cpu_id in getattr(resources, "cpu_ids", ())[:ERDOS_CPUS_PER_EVAL])
    thread_limit = max(1, len(cpu_ids) if cpu_ids else ERDOS_CPUS_PER_EVAL)
    return cpu_ids, thread_limit


def apply_worker_resource_limits(cpu_ids: tuple[int, ...], thread_limit: int) -> None:
    for env_name in THREAD_LIMIT_ENV_VARS:
        os.environ[env_name] = str(max(1, int(thread_limit)))
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))


def evaluate_candidate_in_worker(
    code: str,
    seed: int,
    result_conn: Connection,
    cpu_ids: tuple[int, ...],
    thread_limit: int,
) -> None:
    apply_worker_resource_limits(cpu_ids, thread_limit)
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
            pass

    try:
        # Build the preprocessed code: inject numpy and verifier
        verifier_src = inspect.getsource(verify_c5_solution)
        numpy_import = "import numpy as np"
        base = numpy_import + "\n\n" + verifier_src + "\n\n"

        full_code = base + code

        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "np": np,
            "verify_c5_solution": verify_c5_solution,
            "evaluate_erdos_solution": evaluate_erdos_solution,
        }

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(full_code, namespace, namespace)
            if "run" not in namespace:
                raise ValueError("Generated code must define run")
            result = namespace["run"]()

        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("run must return a tuple of (h_values, c5_bound, n_points)")

        h_values, c5_bound, n_points = result

        if not verify_erdos_solution(result):
            send_result({"ok": False, "msg": "Invalid solution.", "stdout": combined_output()})
            return

        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)

        send_result(
            {
                "ok": True,
                "c5_bound": float(c5_bound),
                "construction": list(h_values),
                "stdout": combined_output(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        send_result({"ok": False, "msg": str(exc), "stdout": combined_output()})
    finally:
        result_conn.close()


def evaluate_candidate_code(
    code: str,
    timeout_s: int,
    budget_s: int,
    seed: int,
    resources: "AllocatedEvaluationResources | None" = None,
) -> dict[str, Any]:
    _ = budget_s
    cpu_ids, thread_limit = normalize_erdos_resources(resources)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=evaluate_candidate_in_worker,
        args=(code, seed, child_conn, cpu_ids, thread_limit),
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

    c5_bound = float(result["c5_bound"])
    return {
        "score": 1.0 / (1e-8 + c5_bound),
        "msg": f"Success; raw_score={c5_bound}",
        "correctness": 1.0,
        "performance": -c5_bound,
        "raw_score": c5_bound,
        "result_payload": {"result_construction": result["construction"]},
        "stdout": str(result.get("stdout", "")),
    }
