import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


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
