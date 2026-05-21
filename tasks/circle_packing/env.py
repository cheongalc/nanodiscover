from __future__ import annotations

import os
import re

from core.archive import ArchiveNode
from core.evaluator import TaskEvaluationRequirements
from tasks.circle_packing.prompt import (
    build_circle_packing_state_context,
    format_fenced_python,
    render_circle_packing_prompt,
)
from tasks.circle_packing.evaluator import (
    CIRCLE_PACKING_CPUS_PER_EVAL,
    evaluate_candidate_code,
)


CIRCLE_PACKING_EVAL_TIMEOUT_SECONDS = 530

CODE_RE = re.compile(r"```python\s*\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)


class CirclePackingTask:
    name = "circle_packing"
    maximize_raw_score = True
    requires_external_evaluator_python = True

    def __init__(self):
        self.num_circles = int(os.environ.get("NANODISCOVER_CIRCLE_PACKING_N", "26"))

    def evaluation_resources(self) -> TaskEvaluationRequirements:
        return TaskEvaluationRequirements(cpus_per_eval=CIRCLE_PACKING_CPUS_PER_EVAL)

    def make_initial_state(self) -> ArchiveNode:
        return ArchiveNode(
            epoch=-1,
            value=0.0,
            task_payload={
                "construction": [],
                "code": "",
            },
        )

    def render_prompt(self, state: ArchiveNode) -> str:
        code = str(state.task_payload.get("code") or "")
        state_ctx = build_circle_packing_state_context(
            state,
            code=code,
            num_circles=self.num_circles,
        )
        return render_circle_packing_prompt(
            state_ctx=state_ctx,
            num_circles=self.num_circles,
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
            num_circles=self.num_circles,
            timeout_s=CIRCLE_PACKING_EVAL_TIMEOUT_SECONDS,
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
        return float(raw_score)

    def make_next_state(
        self,
        *,
        parent_state: ArchiveNode,
        parsed_code: str,
        eval_output: dict[str, object],
        epoch: int,
    ) -> ArchiveNode | None:
        _ = parent_state
        raw_score = eval_output.get("raw_score")
        if raw_score is None:
            return None
        correctness = float(eval_output.get("correctness", 0.0) or 0.0)
        if correctness <= 0:
            return None
        return ArchiveNode(
            epoch=epoch,
            value=float(raw_score),
            task_payload={
                "construction": [],
                "code": format_fenced_python(parsed_code),
                "stdout": str(eval_output.get("stdout", "")),
            },
        )

    def dedupe_key(self, state: ArchiveNode):
        return state.task_payload.get("code")

    def is_state_valid(self, state: ArchiveNode) -> bool:
        return state.value is not None


def build_task() -> CirclePackingTask:
    return CirclePackingTask()
