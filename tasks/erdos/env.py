from __future__ import annotations

import re

import numpy as np

from core.archive import ArchiveNode
from core.evaluator import TaskEvaluationRequirements
from tasks.erdos.prompt import build_erdos_state_context, format_fenced_python, render_erdos_prompt
from tasks.erdos.evaluator import ERDOS_CPUS_PER_EVAL, evaluate_candidate_code


ERDOS_BUDGET_SECONDS = 1000
ERDOS_EVAL_TIMEOUT_SECONDS = 1100
MAX_CONSTRUCTION_LEN = 1000

CODE_RE = re.compile(r"```python\s*\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)


class ErdosTask:
    name = "erdos"
    maximize_raw_score = False
    requires_external_evaluator_python = True

    def evaluation_resources(self) -> TaskEvaluationRequirements:
        return TaskEvaluationRequirements(cpus_per_eval=ERDOS_CPUS_PER_EVAL)

    def make_initial_state(self) -> ArchiveNode:
        rng = np.random.default_rng()
        n_points = int(rng.integers(40, 100))
        construction = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - np.mean(perturbation)
        construction = construction + perturbation
        dx = 2.0 / n_points
        correlation = np.correlate(construction, 1 - construction, mode="full") * dx
        c5_bound = float(np.max(correlation))
        return ArchiveNode(
            epoch=-1,
            value=-c5_bound,
            task_payload={
                "construction": list(construction),
                "code": "",
            },
        )

    def render_prompt(self, state: ArchiveNode) -> str:
        code = str(state.task_payload.get("code") or "")
        state_ctx = build_erdos_state_context(state, code=code)
        return render_erdos_prompt(
            state_ctx=state_ctx,
            code=code,
            budget_s=ERDOS_BUDGET_SECONDS,
        )

    def parse_code(self, response_text: str) -> str:
        matches = list(CODE_RE.finditer(response_text or ""))
        if not matches:
            return ""
        return matches[-1].group(1).strip()

    def evaluate_code(
        self,
        *,
        parsed_code: str,
        state: ArchiveNode,
        epoch: int,
        seed: int,
        resources=None,
    ) -> dict[str, object]:
        _ = epoch
        if not parsed_code.strip():
            return {
                "score": 0.0,
                "msg": "cannot extract python code from model response",
                "correctness": 0.0,
                "performance": 0.0,
                "raw_score": None,
                "result_payload": {},
                "stdout": "",
            }
        return evaluate_candidate_code(
            code=parsed_code,
            timeout_s=ERDOS_EVAL_TIMEOUT_SECONDS,
            budget_s=ERDOS_BUDGET_SECONDS,
            seed=seed,
            resources=resources,
        )

    def compute_reward(self, eval_output: dict[str, object]) -> float:
        correctness = float(eval_output.get("correctness", 0.0) or 0.0)
        if correctness <= 0:
            return 0.0
        raw_score = eval_output.get("raw_score")
        if raw_score is None:
            return 0.0
        return 1.0 / (1e-8 + float(raw_score))

    def make_next_state(
        self,
        *,
        parent_state: ArchiveNode,
        parsed_code: str,
        eval_output: dict[str, object],
        epoch: int,
    ) -> ArchiveNode | None:
        _ = parent_state
        payload = dict(eval_output.get("result_payload") or {})
        construction = payload.get("result_construction")
        raw_score = eval_output.get("raw_score")
        if construction is None:
            return None
        if raw_score is None:
            return None
        return ArchiveNode(
            epoch=epoch,
            value=-float(raw_score),
            task_payload={
                "construction": list(construction),
                "code": format_fenced_python(parsed_code),
                "stdout": str(eval_output.get("stdout", "")),
            },
        )

    def dedupe_key(self, state: ArchiveNode):
        construction = state.task_payload.get("construction")
        if construction is None:
            return None
        return tuple(float(item) for item in construction)

    def is_state_valid(self, state: ArchiveNode) -> bool:
        # Parity: the original only checks max_construction_len (upper bound).
        # There is no lower-bound check for Erdos — that only exists for AC tasks
        # via construction_length_limits.
        construction = list(state.task_payload.get("construction") or [])
        return len(construction) <= MAX_CONSTRUCTION_LEN and state.value is not None


def build_task() -> ErdosTask:
    return ErdosTask()
