from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable


def new_id() -> str:
    """Return a fresh archive node identifier."""

    return uuid.uuid4().hex


def _json_default_for_numpy(obj: Any) -> Any:
    """Coerce numpy scalars/arrays to native Python types for json.dumps.

    Construction lists in ``task_payload`` may carry numpy scalars when user
    candidate code returns ``list(np.array(...))``. ``np.int64`` is not a
    subclass of ``int`` on every platform, so the default encoder fails.
    """

    try:
        import numpy as np
    except ImportError:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def state_sort_value(state: "ArchiveNode") -> float:
    """Return the sortable numeric value used for archive ranking."""

    return float(state.value) if state.value is not None else float("-inf")


@dataclass
class ArchiveNode:
    """Serializable archive state tracked across epochs."""

    id: str = field(default_factory=new_id)
    epoch: int = -1
    value: float | None = None
    task_payload: dict[str, Any] = field(default_factory=dict)
    parents: list[dict[str, Any]] = field(default_factory=list)
    parent_values: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this archive node into a JSON-friendly dict."""

        return {
            "id": self.id,
            "epoch": self.epoch,
            "value": self.value,
            "task_payload": self.task_payload,
            "parents": self.parents,
            "parent_values": self.parent_values,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchiveNode":
        """Deserialize an archive node from a JSON payload."""

        return cls(
            id=str(payload.get("id") or new_id()),
            epoch=int(payload.get("epoch", -1)),
            value=(float(payload["value"]) if payload.get("value") is not None else None),
            task_payload=dict(payload.get("task_payload") or {}),
            parents=list(payload.get("parents") or []),
            parent_values=[float(x) for x in (payload.get("parent_values") or [])],
            metadata=dict(payload.get("metadata") or {}),
        )


def copy_with_parent(child: ArchiveNode, parent: ArchiveNode) -> ArchiveNode:
    """Return a child node with lineage fields extended from its parent."""

    return ArchiveNode(
        id=child.id,
        epoch=child.epoch,
        value=child.value,
        task_payload=dict(child.task_payload),
        parents=[{"id": parent.id, "epoch": parent.epoch}] + list(parent.parents),
        parent_values=(
            [float(parent.value)] + list(parent.parent_values)
            if parent.value is not None
            else []
        ),
        metadata=dict(child.metadata),
    )


@dataclass
class ArchiveConfig:
    """Archive retention and pruning settings."""

    max_archive_size: int = 1000
    topk_children: int = 2


class Archive:
    """In-memory archive of states and their lineage metadata."""

    def __init__(
        self,
        config: ArchiveConfig,
        *,
        initial_state_factory: Callable[[], ArchiveNode],
        refresh_initial_state_fn: Callable[[ArchiveNode], None] | None = None,
        dedupe_key_fn: Callable[[ArchiveNode], Hashable | None],
        is_state_valid_fn: Callable[[ArchiveNode], bool],
    ) -> None:
        self.config = config
        self.initial_state_factory = initial_state_factory
        self.refresh_initial_state_fn = refresh_initial_state_fn
        self.dedupe_key_fn = dedupe_key_fn
        self.is_state_valid_fn = is_state_valid_fn

        self.states: list[ArchiveNode] = []
        self.initial_states: list[ArchiveNode] = []
        self.current_epoch: int = 0

    def initialize(self, num_seeds: int) -> None:
        """Seed the archive with initial states on a fresh run."""

        if self.states:
            return
        self.initial_states = [self.initial_state_factory() for _ in range(num_seeds)]
        self.states = list(self.initial_states)
        self.current_epoch = 0

    def refresh_initial_state(self, state: ArchiveNode) -> None:
        """Refresh an initial seed state when the task exposes that hook."""

        if self.refresh_initial_state_fn is None:
            return
        self.refresh_initial_state_fn(state)

    def checkpoint(self, *, path: str | Path, epoch: int | None = None) -> Path:
        """Write the archive checkpoint to disk."""

        if epoch is not None:
            self.current_epoch = int(epoch)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": self.current_epoch,
            "states": [state.to_dict() for state in self.states],
            "initial_states": [state.to_dict() for state in self.initial_states],
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default_for_numpy),
            encoding="utf-8",
        )
        return path

    def load(self, *, path: str | Path, epoch: int | None = None) -> None:
        """Load archive state from a checkpoint file."""

        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.current_epoch = int(payload["epoch"] if epoch is None else payload.get("epoch", epoch))
        self.states = [ArchiveNode.from_dict(item) for item in payload.get("states", [])]
        self.initial_states = [ArchiveNode.from_dict(item) for item in payload.get("initial_states", [])]

    def update(
        self,
        parents: list[ArchiveNode],
        children: list[ArchiveNode | None],
        *,
        epoch: int,
        checkpoint: bool = True,
        path: str | Path | None = None,
    ) -> None:
        """Insert new children, prune the archive, and optionally checkpoint."""

        if len(parents) != len(children):
            raise ValueError("parents and children must have the same length")

        existing_keys = {self.dedupe_key_fn(state) for state in self.states}
        existing_keys.discard(None)

        for parent, child in zip(parents, children, strict=True):
            if child is None or child.value is None:
                continue

            candidate = copy_with_parent(child, parent)
            if not self.is_state_valid_fn(candidate):
                continue
            dedupe_key = self.dedupe_key_fn(candidate)
            if dedupe_key is not None and dedupe_key in existing_keys:
                continue
            self.states.append(candidate)
            if dedupe_key is not None:
                existing_keys.add(dedupe_key)

        self.flush(epoch=epoch)
        if checkpoint:
            if path is None:
                raise ValueError("path is required when checkpoint=True")
            self.checkpoint(path=path, epoch=epoch)

    def flush(self, *, epoch: int | None = None) -> None:
        """Apply retention policies and update the tracked epoch."""

        if self.config.topk_children > 0:
            root_states: list[ArchiveNode] = []
            grouped: dict[str, list[ArchiveNode]] = {}
            for state in self.states:
                parent_id = state.parents[0]["id"] if state.parents else None
                if parent_id is None:
                    root_states.append(state)
                else:
                    grouped.setdefault(str(parent_id), []).append(state)
            retained_children: list[ArchiveNode] = []
            for siblings in grouped.values():
                siblings.sort(key=state_sort_value, reverse=True)
                retained_children.extend(siblings[: self.config.topk_children])
            self.states = root_states + retained_children

        if len(self.states) > self.config.max_archive_size:
            initial_ids = {state.id for state in self.initial_states}
            keep_ids = set(initial_ids)
            ranked = sorted(self.states, key=state_sort_value, reverse=True)
            for state in ranked:
                if len(keep_ids) >= self.config.max_archive_size:
                    break
                keep_ids.add(state.id)
            self.states = [state for state in self.states if state.id in keep_ids]

        if epoch is not None:
            self.current_epoch = int(epoch)

    def state_values(self) -> list[float]:
        """Return the numeric values of all scored archive states."""

        return [float(state.value) for state in self.states if state.value is not None]

    def stats(self) -> dict[str, float]:
        """Return summary statistics for archive monitoring."""

        values = self.state_values()
        stats: dict[str, float] = {
            "archive_size": float(len(self.states)),
        }
        if values:
            stats.update(
                {
                    "archive_value_min": float(min(values)),
                    "archive_value_mean": float(sum(values) / len(values)),
                    "archive_value_max": float(max(values)),
                }
            )
        return stats
