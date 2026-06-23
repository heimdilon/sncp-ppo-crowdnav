import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
from sncp_ppo.ppo import PPOAgent


# ---------------- Task 2: --ent_coef -> PPOAgent.c2 ----------------

def test_ppo_agent_respects_ent_coef():
    """c2 (entropy coefficient) is the knob --ent_coef wires to. Default 0.01
    (gaussian, backward-compatible); v36 lowers it for beta (0.001). The CLI
    flag->c2 wiring is covered end-to-end by the v36 smoke run."""
    p = SNCPPolicy(meanmax_pool=True)
    assert PPOAgent(policy=p).c2 == 0.01
    assert PPOAgent(policy=p, c2=0.001).c2 == 0.001


# ---------------- Task 1: count-scaling inside multi-head ----------------

def test_count_scaling_affects_multihead_output():
    """With attn_heads>1, attn_count_scaling must scale the attention scores
    (paper Eq 13 n-factor) -> toggling it changes the multi-head output."""
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4, attn_count_scaling=True)
    M_rh = torch.randn(2, 5, 256)
    m_rr = torch.randn(2, 256)
    p.attn_count_scaling = True
    out_on, _ = p._multihead_attention(M_rh, m_rr)
    p.attn_count_scaling = False
    out_off, _ = p._multihead_attention(M_rh, m_rr)
    assert torch.isfinite(out_on).all()
    assert not torch.allclose(out_on, out_off), "count-scaling must change multi-head scores"


def _obs(batch=2, humans=6, offset=0.0):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6) + offset,
        'temporal_edges': torch.randn(batch, 2),
    }


# ---------------- Task 4: combined auto-detect round-trip ----------------

def test_combined_v36_autodetect_roundtrip():
    """v36 stacks every lever at once: build it, then build_policy_for_checkpoint
    must recover ALL variants from the state_dict and load with no missing/unexpected."""
    p = SNCPPolicy(pre_mlp=True, meanmax_pool=True, node_units=256, node_output=96,
                   attn_heads=4, attn_count_scaling=True, action_dist='beta', sense_range=6.0)
    sd = p.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert rebuilt.pre_mlp is True
    assert rebuilt.meanmax_pool is True
    assert rebuilt.attn_heads == 4
    assert rebuilt.attn_count_scaling is True
    assert rebuilt.action_dist == 'beta'
    assert float(rebuilt._sense_range.item()) == 6.0
    assert rebuilt.node_units == 256 and rebuilt.node_output == 96
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_combined_v36_forward_is_finite():
    """The fully-combined policy produces finite beta params + value in a forward pass."""
    p = SNCPPolicy(pre_mlp=True, meanmax_pool=True, node_units=256, node_output=96,
                   attn_heads=4, attn_count_scaling=True, action_dist='beta', sense_range=6.0)
    h = p.init_hidden(2, 6, torch.device('cpu'))
    out1, out2, value, _ = p(_obs(2, 6), h)
    assert torch.isfinite(out1).all() and torch.isfinite(out2).all()
    assert torch.isfinite(value).all()


def test_multihead_no_count_scaling_is_unaffected():
    """attn_count_scaling=False leaves the multi-head path on its plain softmax."""
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)  # count-scaling off (default)
    assert p.attn_count_scaling is False
    M_rh = torch.randn(3, 6, 256)
    m_rr = torch.randn(3, 256)
    out, alpha = p._multihead_attention(M_rh, m_rr)
    assert torch.isfinite(out).all()
    # softmax weights per head sum to 1 over visible humans
    assert torch.allclose(alpha.sum(dim=-1), torch.ones_like(alpha.sum(dim=-1)), atol=1e-5)
