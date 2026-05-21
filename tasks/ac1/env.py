from __future__ import annotations

import re

import numpy as np

from core.archive import ArchiveNode
from core.evaluator import TaskEvaluationRequirements
from tasks.ac1.prompt import build_ac1_state_context, default_initial_code, format_fenced_python, render_ac1_prompt
from tasks.ac1.evaluator import AC1_CPUS_PER_EVAL, evaluate_candidate_code, evaluate_sequence


AC1_INITIAL_STATE_CREATION_SEED = 12345
AC1_BUDGET_SECONDS = 1000
AC1_EVAL_TIMEOUT_SECONDS = 1100
MIN_CONSTRUCTION_LEN = 1000
MAX_CONSTRUCTION_LEN = 100000

CODE_RE = re.compile(r"```python\s*\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)


class AC1Task:
    name = "ac1"
    maximize_raw_score = False
    requires_external_evaluator_python = True

    def evaluation_resources(self) -> TaskEvaluationRequirements:
        return TaskEvaluationRequirements(cpus_per_eval=AC1_CPUS_PER_EVAL)

    def make_initial_state(self) -> ArchiveNode:
        rng = np.random.default_rng(AC1_INITIAL_STATE_CREATION_SEED)
        construction = [float(rng.random())] * int(rng.integers(1000, 8000))
        bound = evaluate_sequence(construction)
        return ArchiveNode(
            epoch=-1,
            value=-float(bound),
            task_payload={
                "construction": construction,
                "code": default_initial_code(AC1_BUDGET_SECONDS),
            },
        )

    def refresh_initial_state(self, state: ArchiveNode) -> None:
        rng = np.random.default_rng()
        construction = [float(rng.random())] * int(rng.integers(1000, 8000))
        bound = evaluate_sequence(construction)
        state.value = -float(bound)
        state.task_payload["construction"] = construction

    def render_prompt(self, state: ArchiveNode) -> str:
        construction = list(state.task_payload.get("construction") or [])
        code = str(state.task_payload.get("code") or "")
        state_ctx = build_ac1_state_context(
            state,
            code=code,
            construction=construction,
        )
        return render_ac1_prompt(state_ctx=state_ctx, budget_s=AC1_BUDGET_SECONDS)

    def parse_code(self, response_text: str) -> str:
        matches = list(CODE_RE.finditer(response_text or ""))
        if not matches:
            return ""
        # Parity: original discover runs last_codeblock_postprocess (LAST block) then
        # _extract_code on the result.  Net effect is the last fenced python block.
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
            parent_construction=list(state.task_payload.get("construction") or []),
            timeout_s=AC1_EVAL_TIMEOUT_SECONDS,
            budget_s=AC1_BUDGET_SECONDS,
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
        construction = list(state.task_payload.get("construction") or [])
        return MIN_CONSTRUCTION_LEN <= len(construction) <= MAX_CONSTRUCTION_LEN and state.value is not None


def build_task() -> AC1Task:
    return AC1Task()

