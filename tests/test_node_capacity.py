"""High-N collision, experiment #2: widen the node-fusion NCP (the 640->128 decision
bottleneck) from AutoNCP(128,48) to AutoNCP(256,96), built on v30 (mean+max). node_units/
node_output are constructor args (defaults 128/48); build_policy_for_checkpoint infers them
from node_ltc.rnn_cell.gleak (units) and output_w (output_size), so v14..v30 checkpoints still
load. The unit tests prove the size is wired + auto-detected; the high-N efficacy is validated
empirically by the Colab eval, not asserted here.
"""
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch, humans):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_default_node_size_unchanged_and_compatible():
    default = SNCPPolicy()
    assert default.node_units == 128 and default.node_output == 48
    assert default.node_wiring.units == 128
    import os, pytest
    if not os.path.exists('checkpoints/sncp_ppo_v18.pt'):
        pytest.skip('milestone checkpoint checkpoints/sncp_ppo_v18.pt is git-ignored; present only locally')
    state = torch.load('checkpoints/sncp_ppo_v18.pt', map_location='cpu')
    policy = build_policy_for_checkpoint(state)
    policy.load_state_dict(state)
    assert policy.node_units == 128


def test_widened_node_builds():
    policy = SNCPPolicy(node_units=256, node_output=96)
    assert policy.node_units == 256 and policy.node_output == 96
    assert policy.node_wiring.units == 256
    assert tuple(policy.node_proj.weight.shape) == (256, 96)


def test_forward_runs_and_action_bounded_widened():
    policy = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, node_units=256, node_output=96)
    h = policy.init_hidden(2, 10, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 10), h)
    assert mu.shape == (2, 2)
    assert torch.isfinite(mu).all() and torch.isfinite(std).all() and torch.isfinite(value).all()
    assert float(mu[:, 0].min()) >= 0.0 and float(mu[:, 0].max()) <= 1.0


def test_widened_node_is_autodetected(tmp_path):
    policy = SNCPPolicy(node_units=256, node_output=96)
    path = tmp_path / 'node256.pt'
    torch.save(policy.state_dict(), path)
    state = torch.load(path, map_location='cpu')
    rebuilt = build_policy_for_checkpoint(state)
    assert rebuilt.node_units == 256 and rebuilt.node_output == 96
    rebuilt.load_state_dict(state)  # must not raise


def test_default_node_state_dict_infers_128(tmp_path):
    policy = SNCPPolicy()  # node 128/48
    path = tmp_path / 'node128.pt'
    torch.save(policy.state_dict(), path)
    state = torch.load(path, map_location='cpu')
    rebuilt = build_policy_for_checkpoint(state)
    assert rebuilt.node_units == 128 and rebuilt.node_output == 48
    rebuilt.load_state_dict(state)  # must not raise


def test_node_capacity_coexists_with_premlp_and_meanmax():
    policy = SNCPPolicy(robot_vpref=1.0, pre_mlp=True, meanmax_pool=True,
                        node_units=256, node_output=96)
    assert policy.pre_mlp and policy.meanmax_pool and policy.node_units == 256
    h = policy.init_hidden(2, 8, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 8), h)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()


def test_train_cli_and_build_thread_node_size():
    import argparse
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.train import build_or_load_policy, build_parser

    a = build_parser().parse_args(['--node_units', '256', '--node_output', '96'])
    assert a.node_units == 256 and a.node_output == 96
    d = build_parser().parse_args([])
    assert d.node_units == 128 and d.node_output == 48

    env = CrowdSimEnv(num_humans=3, scenario='hard', robot_vpref=1.0)
    args = argparse.Namespace(init_checkpoint=None, pre_mlp=False, attn_count_scaling=False,
                              meanmax_pool=False, node_units=256, node_output=96)
    policy = build_or_load_policy(args, env, torch.device('cpu'))
    assert policy.node_units == 256 and policy.node_output == 96
