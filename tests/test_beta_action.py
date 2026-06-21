import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch=2, humans=5):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_beta_build_head_and_no_logstd():
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta')
    finals = [m for m in p.actor_mu if isinstance(m, torch.nn.Linear)]
    assert finals[-1].out_features == 4          # alpha,beta for 2 action dims
    assert not hasattr(p, 'actor_logstd')
    assert p.action_dist == 'beta'


def test_gaussian_default_unchanged_surface():
    p = SNCPPolicy(meanmax_pool=True)            # action_dist defaults to gaussian
    finals = [m for m in p.actor_mu if isinstance(m, torch.nn.Linear)]
    assert finals[-1].out_features == 2
    assert hasattr(p, 'actor_logstd')
    assert p.action_dist == 'gaussian'


def test_beta_forward_returns_valid_alpha_beta():
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta')
    h = p.init_hidden(2, 5, torch.device('cpu'))
    alpha, beta, value, new_h = p(_obs(2, 5), h)
    assert alpha.shape == (2, 2) and beta.shape == (2, 2)
    assert torch.all(alpha >= 1.0) and torch.all(beta >= 1.0)   # softplus+1
    assert value.shape == (2, 1)
    assert set(new_h) == {'temporal_edge', 'spatial_edge', 'node'}


def test_gaussian_forward_still_returns_mu_std():
    p = SNCPPolicy(meanmax_pool=True)
    h = p.init_hidden(2, 5, torch.device('cpu'))
    mu, std, value, _ = p(_obs(2, 5), h)
    assert mu.shape == (2, 2) and std.shape == (2, 2)
    # std is the broadcast exp(logstd) — gaussian path byte-identical
    assert torch.allclose(std[0], torch.exp(p.actor_logstd).squeeze(0))
    assert torch.all(mu[:, 0:1] >= 0.0) and torch.all(mu[:, 0:1] <= p.robot_vpref)


def test_scale_unscale_roundtrip_and_deterministic_mean():
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta', robot_vpref=1.0, robot_wmax=1.8)
    a = torch.tensor([[0.4, -0.9], [1.0, 1.8]])
    x = p._unscale_action(a)
    assert torch.allclose(p._scale_action(x), a, atol=1e-4)
    alpha = torch.tensor([[3.0, 2.0]]); beta = torch.tensor([[1.0, 2.0]])
    det = p.deterministic_action(alpha, beta)        # scaled alpha/(alpha+beta)
    expected = p._scale_action(alpha / (alpha + beta))
    assert torch.allclose(det, expected)
    assert det[0, 0] >= 0.0 and det[0, 0] <= 1.0     # v within [0,vpref]
    assert det[0, 1] >= -1.8 and det[0, 1] <= 1.8    # w within [-wmax,wmax]


def test_beta_autodetect_roundtrip():
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta')
    sd = p.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert rebuilt.action_dist == 'beta'
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_gaussian_checkpoint_autodetects_gaussian():
    p = SNCPPolicy(meanmax_pool=True)                # gaussian (has actor_logstd)
    rebuilt = build_policy_for_checkpoint(p.state_dict())
    assert rebuilt.action_dist == 'gaussian'
    assert hasattr(rebuilt, 'actor_logstd')


import numpy as np


def _np_obs(humans=10):
    return {
        'robot_node': np.random.randn(7).astype(np.float32),
        'spatial_edges': np.random.randn(humans, 6).astype(np.float32),
        'temporal_edges': np.random.randn(2).astype(np.float32),
    }


def test_select_action_beta_is_bounded():
    from sncp_ppo.ppo import PPOAgent
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta', robot_vpref=1.0, robot_wmax=1.8)
    agent = PPOAgent(p)
    dev = torch.device('cpu')
    h = p.init_hidden(1, 10, dev)
    for det in (False, True):
        a, lp, v, _ = agent.select_action(_np_obs(10), h, dev, deterministic=det)
        assert 0.0 <= a[0] <= 1.0          # v in [0, vpref]
        assert -1.8 <= a[1] <= 1.8         # w in [-wmax, wmax]


def test_beta_update_math_is_finite_and_differentiable():
    # The core of the PPO beta branch: forward -> Beta -> unscale stored action ->
    # log_prob/entropy -> backward. Mirrors ppo.update without the full rollout.
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta', robot_vpref=1.0, robot_wmax=1.8)
    h = p.init_hidden(3, 6, torch.device('cpu'))
    alpha, beta, _, _ = p(_obs(3, 6), h)
    actions = torch.stack([p._scale_action(torch.rand(2)) for _ in range(3)])  # physical
    x = p._unscale_action(actions)
    dist = torch.distributions.Beta(alpha, beta)
    logp = dist.log_prob(x).sum(-1)
    ent = dist.entropy().sum(-1)
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all()
    logp.sum().backward()
    assert p.actor_mu[-1].weight.grad is not None


def test_build_or_load_policy_respects_action_dist():
    from types import SimpleNamespace
    from sncp_ppo.train import build_or_load_policy

    class FakeEnv:
        robot_vpref = 1.0
        robot_wmax = 1.8

    args = SimpleNamespace(init_checkpoint=None, pre_mlp=True, attn_count_scaling=False,
                           meanmax_pool=True, node_units=128, node_output=48,
                           attn_heads=1, action_dist='beta')
    policy = build_or_load_policy(args, FakeEnv(), torch.device('cpu'))
    assert policy.action_dist == 'beta'
    assert not hasattr(policy, 'actor_logstd')
