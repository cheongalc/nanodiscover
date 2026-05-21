from __future__ import annotations

from collections import deque
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.archive import ArchiveNode, Archive, state_sort_value


@dataclass
class SamplerConfig:
    """Configuration for PUCT-based parent sampling."""

    puct_c: float = 1.0
    batch_size: int = 8


@dataclass
class PUCTScore:
    """Annotated PUCT score for one archive state."""

    state: ArchiveNode
    score: float
    value: float
    n: int
    q: float
    prior: float
    bonus: float


def rank_prior(values: np.ndarray) -> np.ndarray:
    """Convert values into rank-based sampling priors."""

    if values.size == 0:
        return np.array([], dtype=np.float64)
    ranks = np.argsort(np.argsort(-values))
    weights = (len(values) - ranks).astype(np.float64)
    return weights / weights.sum()


def compute_scale(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Return the score scale used for the PUCT exploration bonus."""

    if values.size == 0:
        return 1.0
    subset = values[mask] if mask is not None else values
    if subset.size == 0:
        subset = values
    return float(max(np.max(subset) - np.min(subset), 1e-6))


def score_states(
    states: list[ArchiveNode],
    *,
    n_map: dict[str, int],
    m_map: dict[str, float],
    T: int,
    puct_c: float,
    initial_ids: set[str],
) -> tuple[list[PUCTScore], float]:
    """Score archive states with the current PUCT statistics."""

    if not states:
        return [], 1.0
    values = np.array([state_sort_value(state) for state in states])
    non_initial_mask = np.array([state.id not in initial_ids for state in states])
    scale = compute_scale(values, non_initial_mask if non_initial_mask.any() else None)
    priors = rank_prior(values)
    sqrt_t = np.sqrt(1.0 + T)

    scored: list[PUCTScore] = []
    for idx, state in enumerate(states):
        visit_count = int(n_map.get(state.id, 0))
        q_value = float(m_map.get(state.id, values[idx])) if visit_count > 0 else float(values[idx])
        bonus = float(puct_c) * scale * float(priors[idx]) * sqrt_t / (1.0 + visit_count)
        scored.append(
            PUCTScore(
                state=state,
                score=q_value + bonus,
                value=float(values[idx]),
                n=visit_count,
                q=q_value,
                prior=float(priors[idx]),
                bonus=float(bonus),
            )
        )
    scored.sort(key=lambda item: (item.score, item.value), reverse=True)
    return scored, scale


def children_map(states: list[ArchiveNode]) -> dict[str, set[str]]:
    """Map each parent id to the ids of its current children."""

    children: dict[str, set[str]] = {}
    for state in states:
        for parent in state.parents:
            parent_id = parent.get("id")
            if parent_id:
                children.setdefault(str(parent_id), set()).add(state.id)
    return children


def ancestor_ids(state: ArchiveNode) -> list[str]:
    """Return the state id followed by any known ancestor ids."""

    return [state.id] + [str(parent["id"]) for parent in state.parents if parent.get("id")]


def blocked_lineage(state: ArchiveNode, children_by_parent: dict[str, set[str]]) -> set[str]:
    """Return the parent/child lineage blocked after selecting a state."""

    blocked = set(ancestor_ids(state))
    queue = deque([state.id])
    seen = {state.id}
    while queue:
        current_id = queue.popleft()
        for child_id in children_by_parent.get(current_id, set()):
            if child_id in seen:
                continue
            seen.add(child_id)
            blocked.add(child_id)
            queue.append(child_id)
    return blocked


def pick_scored_states(scored: list[PUCTScore], *, num_states: int, all_states: list[ArchiveNode]) -> list[PUCTScore]:
    """Pick top scored states while avoiding overlapping lineages."""

    if num_states <= 0:
        return []
    if num_states == 1:
        return scored[:1]

    children_by_parent = children_map(all_states)
    picked: list[PUCTScore] = []
    blocked_ids: set[str] = set()
    for item in scored:
        if item.state.id in blocked_ids:
            continue
        picked.append(item)
        blocked_ids.update(blocked_lineage(item.state, children_by_parent))
        if len(picked) >= num_states:
            break
    return picked


class Sampler:
    """Track PUCT statistics and sample parent states from the archive."""

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.n_map: dict[str, int] = {}
        self.m_map: dict[str, float] = {}
        self.T: int = 0
        self.current_epoch: int = 0
        self.last_scale: float = 1.0
        self.last_scores: list[PUCTScore] = []

    def checkpoint(self, *, path: str | Path, epoch: int | None = None) -> Path:
        """Write the sampler checkpoint to disk."""

        if epoch is not None:
            self.current_epoch = int(epoch)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": self.current_epoch,
            "puct_n": self.n_map,
            "puct_m": self.m_map,
            "puct_T": self.T,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load(self, *, path: str | Path, epoch: int | None = None) -> None:
        """Load sampler state from a checkpoint file."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot load sampler state for epoch {epoch}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.current_epoch = int(payload["epoch"] if epoch is None else payload.get("epoch", epoch))
        self.n_map = {str(key): int(value) for key, value in (payload.get("puct_n") or {}).items()}
        self.m_map = {str(key): float(value) for key, value in (payload.get("puct_m") or {}).items()}
        self.T = int(payload.get("puct_T", 0) or 0)

    def record_rollout_visit(self, parent: ArchiveNode) -> None:
        """Record one rollout visit for a sampled parent lineage."""

        for ancestor_id in ancestor_ids(parent):
            self.n_map[ancestor_id] = self.n_map.get(ancestor_id, 0) + 1
        self.T += 1

    def update(
        self,
        parents: list[ArchiveNode],
        children: list[ArchiveNode | None],
        *,
        epoch: int,
        checkpoint: bool = True,
        path: str | Path | None = None,
    ) -> None:
        """Update PUCT statistics after one epoch of evaluated rollouts."""

        if len(parents) != len(children):
            raise ValueError("parents and children must have the same length")
        for parent, child in zip(parents, children, strict=True):
            # Parity: the original has two code paths in dataset_builder.py:
            #   - record_failed_rollout(parent): increments n/T for failed evals
            #   - update_states([child], [parent]): increments n/T AND updates m,
            #     but SKIPS entirely if child.value is None
            # So: child=None (failed eval) → increment n/T only.
            #     child with value → increment n/T AND update m.
            #     child with value=None → skip entirely (no n/T, no m).
            if child is not None and child.value is None:
                continue
            self.record_rollout_visit(parent)
            if child is None:
                continue
            self.m_map[parent.id] = max(self.m_map.get(parent.id, float(child.value)), float(child.value))
        if checkpoint:
            if path is None:
                raise ValueError("path is required when checkpoint=True")
            self.checkpoint(path=path, epoch=epoch)

    def sample(self, archive: Archive, *, num_states: int | None = None) -> list[PUCTScore]:
        """Sample parent states from the archive using the current PUCT scores."""

        batch_size = int(num_states or self.config.batch_size)
        scored, scale = score_states(
            archive.states,
            n_map=self.n_map,
            m_map=self.m_map,
            T=self.T,
            puct_c=self.config.puct_c,
            initial_ids={state.id for state in archive.initial_states},
        )
        picked = pick_scored_states(scored, num_states=batch_size, all_states=archive.states)
        initial_ids = {state.id for state in archive.initial_states}
        for item in picked:
            if item.state.id in initial_ids:
                archive.refresh_initial_state(item.state)
        self.last_scale = scale
        self.last_scores = picked
        return picked
