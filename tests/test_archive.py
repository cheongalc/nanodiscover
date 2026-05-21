from core.archive import ArchiveConfig, ArchiveNode, Archive


def _make_state() -> ArchiveNode:
    # index parameter removed; just return a generic state.
    return ArchiveNode(epoch=-1, value=1.0, task_payload={"construction": [1]})


def test_archive_dedupes_and_keeps_top2_per_parent(tmp_path):
    archive = Archive(
        ArchiveConfig(topk_children=2, max_archive_size=100),
        initial_state_factory=_make_state,
        dedupe_key_fn=lambda state: tuple(state.task_payload["construction"]),
        is_state_valid_fn=lambda state: True,
    )
    archive.initialize(1)
    parent = archive.states[0]
    children = [
        ArchiveNode(epoch=0, value=0.2, task_payload={"construction": [1, 2]}),
        ArchiveNode(epoch=0, value=0.5, task_payload={"construction": [1, 3]}),
        ArchiveNode(epoch=0, value=0.4, task_payload={"construction": [1, 4]}),
        ArchiveNode(epoch=0, value=0.9, task_payload={"construction": [1, 3]}),
    ]
    archive.update(
        [parent, parent, parent, parent],
        [children[0], children[1], children[2], children[3]],
        epoch=1,
        checkpoint=False,
    )
    kept_children = [state for state in archive.states if state.parents]
    assert len(kept_children) == 2
    kept_values = sorted(state.value for state in kept_children)
    assert kept_values == [0.4, 0.5]


def test_archive_invalid_children_are_not_inserted(tmp_path):
    archive = Archive(
        ArchiveConfig(),
        initial_state_factory=_make_state,
        dedupe_key_fn=lambda state: tuple(state.task_payload["construction"]),
        is_state_valid_fn=lambda state: False,
    )
    archive.initialize(1)
    parent = archive.states[0]
    archive.update(
        [parent],
        [ArchiveNode(epoch=0, value=1.0, task_payload={"construction": [9]})],
        epoch=1,
        checkpoint=False,
    )
    assert len([state for state in archive.states if state.parents]) == 0


def test_copy_with_parent_drops_parent_values_when_parent_value_missing(tmp_path):
    archive = Archive(
        ArchiveConfig(),
        initial_state_factory=lambda: ArchiveNode(
            epoch=-1,
            value=None,
            task_payload={"construction": [1]},
            parent_values=[7.0, 6.0],
            parents=[{"id": "ancestor", "epoch": -2}],
        ),
        dedupe_key_fn=lambda state: tuple(state.task_payload["construction"]),
        is_state_valid_fn=lambda state: True,
    )
    archive.initialize(1)
    parent = archive.states[0]

    archive.update(
        [parent],
        [ArchiveNode(epoch=0, value=1.0, task_payload={"construction": [2]})],
        epoch=1,
        checkpoint=False,
    )

    child = [state for state in archive.states if state.epoch == 0][0]
    assert child.parent_values == []


def test_archive_capacity_keeps_all_initial_states(tmp_path):
    initial_states = [
        ArchiveNode(id="seed-a", epoch=-1, value=10.0, task_payload={"construction": [1]}),
        ArchiveNode(id="seed-b", epoch=-1, value=-100.0, task_payload={"construction": [2]}),
    ]

    def make_state_factory():
        queue = list(initial_states)

        def _make_state() -> ArchiveNode:
            return queue.pop(0)

        return _make_state

    archive = Archive(
        ArchiveConfig(max_archive_size=2, topk_children=0),
        initial_state_factory=make_state_factory(),
        dedupe_key_fn=lambda state: tuple(state.task_payload["construction"]),
        is_state_valid_fn=lambda state: True,
    )
    archive.initialize(2)

    archive.update(
        [archive.states[0]],
        [ArchiveNode(epoch=0, value=999.0, task_payload={"construction": [3]})],
        epoch=1,
        checkpoint=False,
    )

    kept_ids = {state.id for state in archive.states}
    assert kept_ids == {"seed-a", "seed-b"}