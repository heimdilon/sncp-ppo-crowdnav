"""High-N hypothesis: softmax attention pooling is a weighted AVERAGE over humans,
so duplicating similar pedestrians barely changes the pooled vector — count/density
info is lost exactly where it matters (high N). The paper's Eq 13 scales scores by
n/sqrt(d_k) (n = #humans); that n factor feeds count into the softmax temperature.

attn_count_scaling=True adds the n factor. Default False keeps every v14..v23
checkpoint byte-identical (no extra buffer), and build_policy_for_checkpoint
auto-detects the variant from the checkpoint — same pattern as pre_mlp.
"""
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch, humans):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_default_has_no_scaling_buffer_and_is_checkpoint_compatible():
    """Default policy must not add any buffer, so existing checkpoints load."""
    default = SNCPPolicy()
    assert default.attn_count_scaling is False
    assert '_attn_count_scaling' not in default.state_dict()
    # a v18 checkpoint (no scaling buffer) loads into a default policy unchanged
    import os, pytest
    if not os.path.exists('checkpoints/sncp_ppo_v18.pt'):
        pytest.skip('milestone checkpoint checkpoints/sncp_ppo_v18.pt is git-ignored; present only locally')
    state = torch.load('checkpoints/sncp_ppo_v18.pt', map_location='cpu', weights_only=True)
    policy = build_policy_for_checkpoint(state)
    policy.load_state_dict(state)
    assert policy.attn_count_scaling is False


def test_scaling_policy_persists_a_buffer_and_is_autodetected(tmp_path):
    policy = SNCPPolicy(attn_count_scaling=True)
    assert policy.attn_count_scaling is True
    assert '_attn_count_scaling' in policy.state_dict()

    path = tmp_path / 'scaled.pt'
    torch.save(policy.state_dict(), path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    rebuilt = build_policy_for_checkpoint(state)
    assert rebuilt.attn_count_scaling is True
    rebuilt.load_state_dict(state)  # must not raise


def test_forward_runs_and_action_bounded_with_scaling():
    policy = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, attn_count_scaling=True)
    h = policy.init_hidden(2, 10, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 10), h)
    assert mu.shape == (2, 2)
    assert torch.isfinite(mu).all() and torch.isfinite(std).all() and torch.isfinite(value).all()
    assert float(mu[:, 0].min()) >= 0.0 and float(mu[:, 0].max()) <= 1.0


def test_scaling_changes_attention_pool_vs_default():
    """The n factor must actually change the pooled vector (not a no-op). Tested
    directly on _attention_pool with fixed, non-trivial features so the effect
    isn't buried by the downstream LTC/head saturation."""
    base = SNCPPolicy(robot_vpref=1.0, attn_count_scaling=False)
    scaled = SNCPPolicy(robot_vpref=1.0, attn_count_scaling=True)
    scaled.load_state_dict(base.state_dict(), strict=False)  # share weights, only flag differs

    torch.manual_seed(0)
    M_rh = torch.randn(1, 8, 256)
    m_rr = torch.randn(1, 256)
    with torch.no_grad():
        u_base = base._attention_pool(M_rh, m_rr, num_humans=8)
        u_scaled = scaled._attention_pool(M_rh, m_rr, num_humans=8)
    assert not torch.allclose(u_base, u_scaled, atol=1e-4), "n-scaling had no effect on pooling"


def test_train_cli_and_build_or_load_thread_the_flag():
    import argparse
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.train import build_or_load_policy, build_parser

    assert build_parser().parse_args(['--attn_count_scaling']).attn_count_scaling is True
    assert build_parser().parse_args([]).attn_count_scaling is False

    env = CrowdSimEnv(num_humans=3, scenario='hard', robot_vpref=1.0)
    args = argparse.Namespace(init_checkpoint=None, pre_mlp=False, attn_count_scaling=True)
    policy = build_or_load_policy(args, env, torch.device('cpu'))
    assert policy.attn_count_scaling is True


def test_scaling_is_count_sensitive():
    """Same features, different n -> different softmax temperature -> different
    pooled vector. This is the count-sensitivity the paper's Eq 13 introduces."""
    policy = SNCPPolicy(robot_vpref=1.0, attn_count_scaling=True)
    torch.manual_seed(1)
    M_rh = torch.randn(1, 6, 256)
    m_rr = torch.randn(1, 256)
    with torch.no_grad():
        u_small_n = policy._attention_pool(M_rh, m_rr, num_humans=2)
        u_large_n = policy._attention_pool(M_rh, m_rr, num_humans=20)
    assert not torch.allclose(u_small_n, u_large_n, atol=1e-4), "pooling not count-sensitive"
