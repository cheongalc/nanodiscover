from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from core.evaluator import AllocatedEvaluationResources


CIRCLE_PACKING_CPUS_PER_EVAL = 1
# Match ttt-discover paper/evaluator resource envelope: 1 CPU, 530s timeout.
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def normalize_circle_packing_resources(resources: "AllocatedEvaluationResources | None") -> tuple[tuple[int, ...], int]:
    cpu_ids = tuple(int(cpu_id) for cpu_id in getattr(resources, "cpu_ids", ())[:CIRCLE_PACKING_CPUS_PER_EVAL])
    thread_limit = max(1, len(cpu_ids) if cpu_ids else CIRCLE_PACKING_CPUS_PER_EVAL)
    return cpu_ids, thread_limit


def apply_worker_resource_limits(cpu_ids: tuple[int, ...], thread_limit: int) -> None:
    for env_name in THREAD_LIMIT_ENV_VARS:
        os.environ[env_name] = str(max(1, int(thread_limit)))
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True


def check_packing_correctness(centers, radii, num_circles: int) -> bool:
    shape_valid = centers.shape == (num_circles, 2) and radii.shape == (num_circles,)
    if not shape_valid:
        return False

    return validate_packing(centers, radii)


# numpy is available to evaluator code, but not injected into generated code.
import numpy as np


def evaluate_candidate_in_worker(
    code: str,
    num_circles: int,
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
        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
        }
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, namespace, namespace)
            if "run_packing" not in namespace:
                raise ValueError("Generated code must define run_packing")
            output = namespace["run_packing"]()

        centers, radii, sum_radii = output
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        if not check_packing_correctness(centers, radii, num_circles):
            send_result({"ok": False, "msg": "Packing is not valid.", "stdout": combined_output()})
            return

        sum_of_radii = float(np.sum(radii))
        send_result(
            {
                "ok": True,
                "sum_of_radii": sum_of_radii,
                "result_construction": [],
                "stdout": combined_output(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        send_result({"ok": False, "msg": str(exc), "stdout": combined_output()})
    finally:
        result_conn.close()


def evaluate_candidate_code(
    code: str,
    num_circles: int,
    timeout_s: int,
    seed: int,
    resources: "AllocatedEvaluationResources | None" = None,
) -> dict[str, Any]:
    cpu_ids, thread_limit = normalize_circle_packing_resources(resources)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=evaluate_candidate_in_worker,
        args=(code, num_circles, seed, child_conn, cpu_ids, thread_limit),
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

    sum_of_radii = float(result["sum_of_radii"])
    return {
        "score": sum_of_radii,
        "msg": f"Success; raw_score={sum_of_radii}",
        "correctness": 1.0,
        "performance": sum_of_radii,
        "raw_score": sum_of_radii,
        "result_payload": {"result_construction": result["result_construction"]},
        "stdout": str(result.get("stdout", "")),
    }
