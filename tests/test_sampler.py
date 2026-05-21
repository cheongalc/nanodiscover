import json

from core.archive import Archive, ArchiveConfig, ArchiveNode
from core.sampler import Sampler, SamplerConfig, compute_scale, pick_scored_states, rank_prior, score_states


def test_rank_prior_prefers_higher_values():
    priors = rank_prior(__import__("numpy").array([3.0, 2.0, 1.0]))
    assert priors[0] > priors[1] > priors[2]


def test_compute_scale_uses_masked_non_initial_values():
    np = __import__("numpy")
    values = np.array([10.0, 11.0, 30.0])
    mask = np.array([False, True, True])
    assert compute_scale(values, mask) == 19.0


def test_diversity_pick_blocks_ancestor_and_descendant():
    root = ArchiveNode(id="root", epoch=-1, value=1.0, task_payload={})
    child = ArchiveNode(id="child", epoch=0, value=2.0, task_payload={}, parents=[{"id": "root", "epoch": -1}])
    sibling = ArchiveNode(id="sib", epoch=0, value=1.5, task_payload={})
    scored, _ = score_states(
        [child, sibling, root],
        n_map={},
        m_map={},
        T=0,
        puct_c=1.0,
        initial_ids={"root", "sib"},
    )
    picked = pick_scored_states(scored, num_states=2, all_states=[root, child, sibling])
    picked_ids = {item.state.id for item in picked}
    assert "child" in picked_ids
    assert "root" not in picked_ids


def test_diversity_pick_does_not_top_up_with_blocked_states():
    root = ArchiveNode(id="root", epoch=-1, value=3.0, task_payload={})
    child = ArchiveNode(id="child", epoch=0, value=2.0, task_payload={}, parents=[{"id": "root", "epoch": -1}])

    scored, _ = score_states(
        [root, child],
        n_map={},
        m_map={},
        T=0,
        puct_c=1.0,
        initial_ids={"root"},
    )

    picked = pick_scored_states(scored, num_states=2, all_states=[root, child])

    assert len(picked) == 1


def test_sampler_refreshes_sampled_initial_states(tmp_path):
    initial_state = ArchiveNode(id="seed", epoch=-1, value=1.0, task_payload={"construction": [1], "code": "x"})

    def make_state() -> ArchiveNode:
        return initial_state

    def refresh_state(state: ArchiveNode) -> None:
        state.value = 2.0
        state.task_payload["construction"] = [2]

    archive = Archive(
        ArchiveConfig(),
        initial_state_factory=make_state,
        refresh_initial_state_fn=refresh_state,
        dedupe_key_fn=lambda state: tuple(state.task_payload.get("construction") or []),
        is_state_valid_fn=lambda state: True,
    )
    archive.initialize(1)

    sampler = Sampler(SamplerConfig(batch_size=1))
    picked = sampler.sample(archive)

    assert picked[0].state.id == "seed"
    assert picked[0].state.value == 2.0
    assert archive.initial_states[0].task_payload["construction"] == [2]


def test_sampler_tracks_visits_and_T_for_failed_rollouts(tmp_path):
    parent = ArchiveNode(id="seed", epoch=-1, value=1.0, task_payload={"construction": [1]})
    sampler = Sampler(SamplerConfig())

    sampler.update([parent], [None], epoch=1, checkpoint=False)

    assert sampler.n_map[parent.id] == 1
    assert sampler.T == 1
    assert parent.id not in sampler.m_map


def test_sampler_skips_visit_when_child_has_none_value():
    """Parity: original update_states skips entirely when child.value is None.

    This is distinct from the child=None case (failed eval), where n/T still
    increment via record_failed_rollout.
    """
    parent = ArchiveNode(id="seed", epoch=-1, value=1.0, task_payload={"construction": [1]})
    child_with_none_value = ArchiveNode(id="bad", epoch=0, value=None, task_payload={})
    sampler = Sampler(SamplerConfig())

    sampler.update([parent], [child_with_none_value], epoch=1, checkpoint=False)

    assert sampler.n_map.get(parent.id, 0) == 0
    assert sampler.T == 0
    assert parent.id not in sampler.m_map


def test_sampler_updates_immediate_parent_q_only_not_ancestors(tmp_path):
    root = ArchiveNode(id="root", epoch=-1, value=1.0, task_payload={"construction": [1]})
    child = ArchiveNode(
        id="child",
        epoch=0,
        value=2.0,
        task_payload={"construction": [2]},
        parents=[{"id": "root", "epoch": -1}],
    )
    grandchild = ArchiveNode(id="grandchild", epoch=1, value=5.0, task_payload={"construction": [3]})
    sampler = Sampler(SamplerConfig())

    sampler.update([child], [grandchild], epoch=1, checkpoint=False)

    assert sampler.m_map[child.id] == 5.0
    assert root.id not in sampler.m_map
    assert sampler.n_map[child.id] == 1
    assert sampler.n_map[root.id] == 1
    assert sampler.T == 1
def test_sampler_loads_epoch_checkpoint(tmp_path):
    checkpoint = tmp_path / "sampler_epoch_003.json"
    checkpoint.write_text(
        json.dumps(
            {
                "epoch": 3,
                "puct_n": {"a": 5},
                "puct_m": {"a": 1.25},
                "puct_T": 9,
            }
        ),
        encoding="utf-8",
    )

    sampler = Sampler(SamplerConfig())

    sampler.load(path=checkpoint)

    assert sampler.current_epoch == 3
    assert sampler.n_map == {"a": 5}
    assert sampler.m_map == {"a": 1.25}
    assert sampler.T == 9