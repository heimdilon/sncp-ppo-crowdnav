"""Regression tests for the vectorized Beta action path.

Bug (caught by code review): the vectorized rollout (train.py) and
PPOAgent.update_vectorized treated the policy outputs as Normal(mu, std) in ALL
cases, so a Beta policy was actually trained as Normal(alpha, beta) -- never Beta.
Only the single-env path branched on action_dist. These tests lock the shared
distribution builder + the Beta rollout contract (sample in [0,1] -> scale to
physical, in-bounds -> unscale recovers the pre-image for log_prob).
"""
import torch

from sncp_ppo.models import SNCPPolicy


def _obs(batch=4, humans=4):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_make_action_dist_returns_correct_type():
    pb = SNCPPolicy(meanmax_pool=True, action_dist='beta')
    pg = SNCPPolicy(meanmax_pool=True)  # gaussian (default)
    a = torch.rand(3, 2) + 1.0
    b = torch.rand(3, 2) + 1.0
    assert isinstance(pb.make_action_dist(a, b), torch.distributions.Beta)
    mu = torch.zeros(3, 2)
    std = torch.ones(3, 2)
    assert isinstance(pg.make_action_dist(mu, std), torch.distributions.Normal)


def test_beta_rollout_contract_in_bounds():
    """Beta rollout: sample x~Beta in [0,1], scale to physical (in-bounds),
    and unscaling the stored physical action recovers x for log_prob."""
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta')
    h = p.init_hidden(4, 4, torch.device('cpu'))
    out1, out2, _, _ = p(_obs(4, 4), h)
    dist = p.make_action_dist(out1, out2)
    assert isinstance(dist, torch.distributions.Beta)
    x = dist.sample()
    action = p._scale_action(x)                       # stored physical
    assert torch.all(action >= p.action_low - 1e-5)
    assert torch.all(action <= p.action_high + 1e-5)
    # update path: unscale recovers the [0,1] pre-image -> finite Beta log_prob
    x_rec = p._unscale_action(action)
    assert torch.allclose(x, x_rec, atol=1e-4)
    lp = dist.log_prob(x_rec).sum(-1)
    assert torch.isfinite(lp).all()


def test_gaussian_rollout_contract_unchanged():
    p = SNCPPolicy(meanmax_pool=True)  # gaussian
    h = p.init_hidden(4, 4, torch.device('cpu'))
    mu, std, _, _ = p(_obs(4, 4), h)
    dist = p.make_action_dist(mu, std)
    assert isinstance(dist, torch.distributions.Normal)
    action = dist.sample()
    lp = dist.log_prob(action).sum(-1)
    assert torch.isfinite(lp).all()
