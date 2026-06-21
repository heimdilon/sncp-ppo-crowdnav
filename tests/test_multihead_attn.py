import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch=2, humans=5):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_multihead_build_has_mha_layers_and_buffer():
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)
    assert hasattr(p, 'W_v') and hasattr(p, 'W_o')
    assert p.W_q.out_features == 256 and p.W_k.out_features == 256
    assert p.W_v.out_features == 256 and p.W_o.out_features == 256
    assert int(p._attn_heads.item()) == 4


def test_single_head_default_is_byte_compatible_surface():
    p = SNCPPolicy(meanmax_pool=True)  # attn_heads defaults to 1
    assert not hasattr(p, 'W_v')
    assert not hasattr(p, 'W_o')
    assert '_attn_heads' not in dict(p.named_buffers())
    assert p.W_q.out_features == 64  # legacy single-head projection unchanged


def test_multihead_forward_shapes():
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)
    h = p.init_hidden(2, 5, torch.device('cpu'))
    mu, std, value, new_h = p(_obs(2, 5), h)
    assert mu.shape == (2, 2)
    assert std.shape == (2, 2)
    assert value.shape == (2, 1)
    assert set(new_h) == {'temporal_edge', 'spatial_edge', 'node'}


def test_multihead_autodetect_roundtrip():
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)
    sd = p.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert int(rebuilt._attn_heads.item()) == 4
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_v30_meanmax_checkpoint_loads_as_single_head():
    v30 = SNCPPolicy(meanmax_pool=True)  # single-head v30 architecture
    sd = v30.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert '_attn_heads' not in dict(rebuilt.named_buffers())
    assert not hasattr(rebuilt, 'W_v')
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_heads_differentiate():
    """With several humans, the per-head attention distributions are not all identical."""
    torch.manual_seed(0)
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)
    M_rh = torch.randn(1, 6, 256)
    m_rr = torch.randn(1, 256)
    _, alpha = p._multihead_attention(M_rh, m_rr)  # alpha: [1, 4, 1, 6]
    a = alpha[0, :, 0, :]  # [4 heads, 6 humans]
    pair_diffs = (a.unsqueeze(0) - a.unsqueeze(1)).abs().sum(-1)  # [4, 4]
    assert pair_diffs.max().item() > 1e-3


def test_build_or_load_policy_respects_attn_heads():
    from types import SimpleNamespace
    from sncp_ppo.train import build_or_load_policy

    class FakeEnv:
        robot_vpref = 0.26
        robot_wmax = 1.8

    args = SimpleNamespace(init_checkpoint=None, pre_mlp=True, attn_count_scaling=False,
                           meanmax_pool=True, node_units=128, node_output=48, attn_heads=4)
    policy = build_or_load_policy(args, FakeEnv(), torch.device('cpu'))
    assert int(policy._attn_heads.item()) == 4
    assert hasattr(policy, 'W_v')
