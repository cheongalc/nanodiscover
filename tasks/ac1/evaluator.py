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


AC1_CPUS_PER_EVAL = 2

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


def normalize_ac1_resources(resources: "AllocatedEvaluationResources | None") -> tuple[tuple[int, ...], int]:
    cpu_ids = tuple(int(cpu_id) for cpu_id in getattr(resources, "cpu_ids", ())[:AC1_CPUS_PER_EVAL])
    thread_limit = max(1, len(cpu_ids) if cpu_ids else AC1_CPUS_PER_EVAL)
    return cpu_ids, thread_limit


def apply_worker_resource_limits(cpu_ids: tuple[int, ...], thread_limit: int) -> None:
    for env_name in THREAD_LIMIT_ENV_VARS:
        os.environ[env_name] = str(max(1, int(thread_limit)))
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))


def evaluate_sequence(sequence: list[float]) -> float:
    np = _np()
    if not isinstance(sequence, list):
        return np.inf
    if not sequence:
        return np.inf
    for item in sequence:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return np.inf
        if np.isnan(item) or np.isinf(item):
            return np.inf
    sequence = [min(1000.0, max(0.0, float(item))) for item in sequence]
    total = float(np.sum(sequence))
    if total < 0.01:
        return np.inf
    convolution = np.convolve(sequence, sequence)
    return float(2 * len(sequence) * max(convolution) / (total**2))


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
        "ac1_eval_worker_resources seed=%d effective_cpus=%d effective_cpu_ids=%s blas_omp_thread_cap=%d requested_cpu_ids=%s",
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
            if "propose_candidate" not in namespace:
                raise ValueError("Generated code must define propose_candidate")
            # Match ttt-discover: call propose_candidate() with no args.
            # If the model didn't provide defaults, this should error (as in the original).
            candidate = namespace["propose_candidate"]()
        # Match ttt-discover: only list outputs are accepted (numpy arrays are invalid).
        if not isinstance(candidate, list):
            raise TypeError("propose_candidate must return list[float]")
        bound = evaluate_sequence(candidate)
        if not np.isfinite(bound):
            send_result({"ok": False, "msg": "candidate failed evaluation", "stdout": combined_output()})
            return
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
    _ = budget_s  # Not forwarded; match original ttt-discover which calls propose_candidate() with no args.
    cpu_ids, thread_limit = normalize_ac1_resources(resources)
    pool_slot = getattr(resources, "slot_index", None) if resources is not None else None
    logger.info(
        "ac1_eval_allocate seed=%d allocated_cpus=%d allocated_cpu_ids=%s blas_omp_thread_cap=%d resource_pool_slot=%s",
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
        "score": 1.0 / (1e-8 + bound),
        "msg": f"Success; raw_score={bound}",
        "correctness": 1.0,
        "performance": -bound,
        "raw_score": bound,
        "result_payload": {"result_construction": result["construction"]},
        "stdout": str(result.get("stdout", "")),
    }
