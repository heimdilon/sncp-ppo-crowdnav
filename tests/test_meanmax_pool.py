"""High-N collision fix: the attention pool is a convex combination of per-human
features (value = M_rh, no W_v), so at high N the pooled vector regresses toward
the mean and the most-threatening agent's signal is diluted. meanmax_pool concats
the attention-weighted mean with an element-wise MAX over humans (cardinality-robust,
PointNet/DeepSet) through pool_merge = Linear(512->256). Default False keeps every
v14..v29 checkpoint byte-identical; build_policy_for_checkpoint auto-detects from the
pool_merge key (same pattern as pre_mlp / attn_count_scaling). The unit tests prove
the mechanism is WIRED and live; the high-N efficacy itself is validated empirically
by the Colab eval, not asserted here.
"""
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch, humans):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_default_off_no_pool_merge_and_checkpoint_compatible():
    default = SNCPPolicy()
    assert default.meanmax_pool is False
    assert not any(k.startswith('pool_merge') for k in default.state_dict())
    import os, pytest
    if not os.path.exists('checkpoints/sncp_ppo_v18.pt'):
        pytest.skip('milestone checkpoint checkpoints/sncp_ppo_v18.pt is git-ignored; present only locally')
    state = torch.load('checkpoints/sncp_ppo_v18.pt', map_location='cpu', weights_only=True)
    policy = build_policy_for_checkpoint(state)
    policy.load_state_dict(state)
    assert policy.meanmax_pool is False


def test_meanmax_builds_pool_merge_and_is_autodetected(tmp_path):
    policy = SNCPPolicy(meanmax_pool=True)
    assert policy.meanmax_pool is True
    assert any(k.startswith('pool_merge') for k in policy.state_dict())

    path = tmp_path / 'meanmax.pt'
    torch.save(policy.state_dict(), path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    rebuilt = build_policy_for_checkpoint(state)
    assert rebuilt.meanmax_pool is True
    rebuilt.load_state_dict(state)  # must not raise


def test_forward_runs_and_action_bounded_with_meanmax():
    policy = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, meanmax_pool=True)
    h = policy.init_hidden(2, 10, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 10), h)
    assert mu.shape == (2, 2)
    assert torch.isfinite(mu).all() and torch.isfinite(std).all() and torch.isfinite(value).all()
    assert float(mu[:, 0].min()) >= 0.0 and float(mu[:, 0].max()) <= 1.0


def test_meanmax_changes_pooled_representation_vs_mean_only():
    """With shared encoder/attention weights, the ONLY difference is the max branch
    + merge. The pooled vector must differ from the mean-only pool — proving the new
    operation is live (not a no-op), tested directly on _attention_pool."""
    mean_only = SNCPPolicy(robot_vpref=1.0, meanmax_pool=False)
    meanmax = SNCPPolicy(robot_vpref=1.0, meanmax_pool=True)
    meanmax.load_state_dict(mean_only.state_dict(), strict=False)  # share W_q/W_k; pool_merge stays

    torch.manual_seed(0)
    M_rh = torch.randn(1, 12, 256)
    m_rr = torch.randn(1, 256)
    with torch.no_grad():
        u_mean = mean_only._attention_pool(M_rh, m_rr, num_humans=12)
        u_mm = meanmax._attention_pool(M_rh, m_rr, num_humans=12)
    assert u_mean.shape == u_mm.shape == (1, 256)
    assert not torch.allclose(u_mean, u_mm, atol=1e-4), "max branch had no effect on the pool"


def test_pre_mlp_and_meanmax_coexist():
    policy = SNCPPolicy(robot_vpref=1.0, pre_mlp=True, meanmax_pool=True)
    assert policy.pre_mlp is True and policy.meanmax_pool is True
    h = policy.init_hidden(2, 8, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 8), h)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()


def test_train_cli_and_build_thread_the_flag():
    import argparse
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.train import build_or_load_policy, build_parser

    assert build_parser().parse_args(['--meanmax_pool']).meanmax_pool is True
    assert build_parser().parse_args([]).meanmax_pool is False

    env = CrowdSimEnv(num_humans=3, scenario='hard', robot_vpref=1.0)
    args = argparse.Namespace(init_checkpoint=None, pre_mlp=False,
                              attn_count_scaling=False, meanmax_pool=True)
    policy = build_or_load_policy(args, env, torch.device('cpu'))
    assert policy.meanmax_pool is True
