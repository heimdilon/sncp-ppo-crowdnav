"""Phase 0/1: collect ORCA-expert demos. Only successful episodes are kept (we
never clone a collision or timeout). Demos are grouped per density because the
spatial-edge tensor width = num_humans differs across N (ragged, can't stack)."""
import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.il.collect_demos import run_expert_episode, collect_dataset


def test_run_expert_episode_records_transitions_and_outcome():
    env = CrowdSimEnv(num_humans=3, scenario='hard', randomize_layout=False,
                      robot_vpref=1.0, human_vpref_override=1.0, max_time=15.0)
    ep = run_expert_episode(env, seed=0)
    assert ep['steps'] == len(ep['robot_node']) == len(ep['actions'])
    assert np.asarray(ep['robot_node'][0]).shape == (7,)
    assert np.asarray(ep['spatial_edges'][0]).shape == (3, 6)
    assert np.asarray(ep['actions'][0]).shape == (2,)
    assert isinstance(bool(ep['success']), bool)
    assert isinstance(bool(ep['collision']), bool)


def test_collect_keeps_only_successful_episodes():
    shards, stats = collect_dataset(
        densities=[1], n_per_density=12, seed=0, max_time=15.0,
    )
    s = stats[1]
    assert s['episodes'] == 12
    assert s['kept'] == s['success']
    assert s['success'] + s['collision'] + s['timeout'] == 12
    if s['kept'] > 0:
        shard = shards[1]
        # flattened arrays line up with episode boundaries
        assert sum(shard['episode_lengths']) == shard['robot_node'].shape[0]
        assert shard['robot_node'].shape[1] == 7
        assert shard['spatial_edges'].shape[1:] == (1, 6)
        assert shard['actions'].shape[1] == 2
        assert len(shard['episode_lengths']) == s['kept']


def test_collect_is_deterministic_under_seed():
    _, stats_a = collect_dataset(densities=[3], n_per_density=8, seed=7, max_time=15.0)
    _, stats_b = collect_dataset(densities=[3], n_per_density=8, seed=7, max_time=15.0)
    assert stats_a[3]['kept'] == stats_b[3]['kept']
    assert stats_a[3]['success'] == stats_b[3]['success']
