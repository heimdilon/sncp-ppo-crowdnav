import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch=2, humans=4, offset=0.0):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6) + offset,
        'temporal_edges': torch.randn(batch, 2),
    }


def test_sense_range_build_registers_buffer():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    assert float(p._sense_range.item()) == 6.0
    assert p.sense_range == 6.0


def test_default_has_no_sense_buffer():
    p = SNCPPolicy(meanmax_pool=True)  # sense_range defaults to 0.0
    assert '_sense_range' not in dict(p.named_buffers())
    assert p.sense_range == 0.0


def test_attention_pool_mask_excludes_hidden_humans():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    B, N = 1, 4
    M_rh = torch.randn(B, N, 256)
    m_rr = torch.randn(B, 256)
    mask_first = torch.tensor([[True, False, False, False]])
    first_only = p._attention_pool(M_rh, m_rr, N, mask_first)
    # Pooling with only human 0 visible must equal pooling a 1-human input.
    only = p._attention_pool(M_rh[:, :1, :], m_rr, 1, torch.ones(B, 1, dtype=torch.bool))
    assert torch.allclose(first_only, only, atol=1e-5)
    # And it must differ from pooling all four humans.
    full = p._attention_pool(M_rh, m_rr, N, torch.ones(B, N, dtype=torch.bool))
    assert not torch.allclose(first_only, full)


def test_all_masked_is_finite_zero_pool():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    B, N = 2, 3
    M_rh = torch.randn(B, N, 256)
    m_rr = torch.randn(B, 256)
    out = p._attention_pool(M_rh, m_rr, N, torch.zeros(B, N, dtype=torch.bool))
    assert torch.isfinite(out).all()
    expected = p.pool_merge(torch.zeros(B, 512))
    assert torch.allclose(out, expected, atol=1e-5)


def test_forward_all_far_humans_is_finite():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    h = p.init_hidden(2, 4, torch.device('cpu'))
    obs = _obs(2, 4, offset=100.0)  # all humans ~100 m away -> all masked
    mu, std, value, _ = p(obs, h)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()


def test_sense_range_autodetect_roundtrip():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    sd = p.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert float(rebuilt._sense_range.item()) == 6.0
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_v30_checkpoint_autodetects_no_masking():
    p = SNCPPolicy(meanmax_pool=True)  # sense_range 0
    rebuilt = build_policy_for_checkpoint(p.state_dict())
    assert rebuilt.sense_range == 0.0
    assert '_sense_range' not in dict(rebuilt.named_buffers())


def test_build_or_load_policy_respects_sense_range():
    from types import SimpleNamespace
    from sncp_ppo.train import build_or_load_policy

    class FakeEnv:
        robot_vpref = 1.0
        robot_wmax = 1.8

    args = SimpleNamespace(init_checkpoint=None, pre_mlp=True, attn_count_scaling=False,
                           meanmax_pool=True, node_units=128, node_output=48,
                           attn_heads=1, action_dist='gaussian', sense_range=6.0)
    policy = build_or_load_policy(args, FakeEnv(), torch.device('cpu'))
    assert policy.sense_range == 6.0
    assert float(policy._sense_range.item()) == 6.0
