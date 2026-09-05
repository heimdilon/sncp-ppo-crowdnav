"""M2' SparseHIG: optional top-k human-human graph on CfC + existing RH attention.

Dense v37 HH is full H×H MultiheadAttention. SparseHIG keeps the residual / hh_gate
/ CV-intent surface but attends only to k≤4 neighbors. Default LTC + SparseHIG-off
must stay load-compatible with existing checkpoints. No HEIGHT / transformer, no
runtime action shield, no obs/reward change.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sncp_ppo.models import (
    SNCPPolicy,
    SparseHIGMismatchError,
    assert_sparse_hig_compatible,
    build_policy_for_checkpoint,
    detect_hh_topk,
    detect_sparse_hig,
    load_policy_state_dict,
)
from sncp_ppo.ppo import PPOAgent
from sncp_ppo.train import build_or_load_policy, build_parser, build_upgraded_policy
from sncp_ppo.vec_buffer import VectorizedRolloutBuffer


def _obs(batch=2, humans=5, *, far=False, positions=None):
    spatial = torch.linspace(-1.5, 1.5, batch * humans * 6).reshape(batch, humans, 6)
    if positions is not None:
        spatial = spatial.clone()
        spatial[..., 0:2] = torch.as_tensor(positions, dtype=spatial.dtype)
    if far:
        spatial = spatial.clone()
        spatial[..., 0:2] += 100.0
    return {
        "robot_node": torch.linspace(-1.0, 1.0, batch * 7).reshape(batch, 7),
        "spatial_edges": spatial,
        "temporal_edges": torch.linspace(-0.5, 0.5, batch * 2).reshape(batch, 2),
    }


def test_default_policy_has_no_sparsehig_surface():
    policy = SNCPPolicy()
    keys = set(policy.state_dict())
    assert policy.sparse_hig is False
    assert policy.hh_topk == 0
    assert not hasattr(policy, "hh_sparse_attn")
    assert "_hh_sparse_k" not in keys
    assert not any(key.startswith("hh_sparse_attn.") for key in keys)
    assert detect_sparse_hig(policy.state_dict()) is False
    assert detect_hh_topk(policy.state_dict()) == 0


def test_sparse_hig_implies_hh_intent_graph_and_uses_distinct_module():
    policy = SNCPPolicy(sparse_hig=True, hh_topk=3)
    keys = set(policy.state_dict())
    assert policy.sparse_hig is True
    assert policy.hh_intent_graph is True
    assert policy.hh_topk == 3
    assert hasattr(policy, "hh_sparse_attn")
    assert not hasattr(policy, "hh_attn")
    assert isinstance(policy.hh_sparse_attn, nn.MultiheadAttention)
    assert float(policy.state_dict()["_hh_sparse_k"].item()) == 3.0
    assert any(key.startswith("hh_sparse_attn.") for key in keys)
    assert not any(key.startswith("hh_attn.") for key in keys)
    assert detect_sparse_hig(policy.state_dict()) is True
    assert detect_hh_topk(policy.state_dict()) == 3


@pytest.mark.parametrize("k", [0, 1, 3, 4])
@pytest.mark.parametrize("humans", [1, 2, 5, 8])
def test_sparse_hig_forward_finite_for_k_and_variable_h(k, humans):
    policy = SNCPPolicy(sparse_hig=True, hh_topk=k)
    hidden = policy.init_hidden(2, humans, torch.device("cpu"))
    out1, out2, value, new_hidden = policy(_obs(2, humans), hidden)
    assert out1.shape == (2, 2)
    assert value.shape == (2, 1)
    for tensor in (out1, out2, value, *new_hidden.values()):
        assert torch.isfinite(tensor).all()


def test_nonzero_gate_changes_output_and_gets_gradient():
    policy = SNCPPolicy(sparse_hig=True, hh_topk=3)
    zero = SNCPPolicy(sparse_hig=True, hh_topk=3)
    zero.load_state_dict(policy.state_dict())
    policy.hh_gate.data.fill_(0.8)
    obs = _obs(2, 6)
    hidden = policy.init_hidden(2, 6, torch.device("cpu"))
    z_out = zero(obs, zero.init_hidden(2, 6, torch.device("cpu")))
    p_out = policy(obs, hidden)
    assert any(not torch.allclose(a, b) for a, b in zip(z_out[:3], p_out[:3]))
    loss = sum(tensor.sum() for tensor in p_out[:3])
    loss.backward()
    assert policy.hh_gate.grad is not None
    assert torch.isfinite(policy.hh_gate.grad)
    assert float(policy.hh_gate.grad.abs()) > 0.0


def test_k_zero_is_identity_even_with_nonzero_gate():
    sparse = SNCPPolicy(sparse_hig=True, hh_topk=0)
    sparse.hh_gate.data.fill_(0.8)
    m_rh = torch.linspace(-1.0, 1.0, 2 * 6 * 256).reshape(2, 6, 256)
    edges = _obs(2, 6)["spatial_edges"]
    out = sparse._human_intent_graph(m_rh, edges)
    assert torch.allclose(out, m_rh, atol=1e-6)


def test_h_less_than_k_pads_safely():
    policy = SNCPPolicy(sparse_hig=True, hh_topk=4)
    hidden = policy.init_hidden(3, 2, torch.device("cpu"))
    out1, out2, value, new_hidden = policy(_obs(3, 2), hidden)
    for tensor in (out1, out2, value, *new_hidden.values()):
        assert torch.isfinite(tensor).all()

    idx, valid = policy._topk_neighbor_index(_obs(1, 2)["spatial_edges"])
    assert idx.shape == (1, 2, 4)
    assert valid.shape == (1, 2, 4)
    # H=2 ⇒ exactly one real (non-self) neighbor per query.
    assert int(valid[0].sum().item()) == 2
    assert bool(valid[0, 0, 0]) and not bool(valid[0, 0, 1:].any())


def test_topk_selects_nearest_other_humans():
    policy = SNCPPolicy(sparse_hig=True, hh_topk=1)
    # Robot-local positions: 0 near 1; 2 is far.
    positions = [[[0.0, 0.0], [0.4, 0.0], [8.0, 0.0]]]
    edges = _obs(1, 3, positions=positions)["spatial_edges"]
    idx, valid = policy._topk_neighbor_index(edges)
    assert valid.tolist() == [[[True], [True], [True]]]
    assert idx[0, 0, 0].item() == 1
    assert idx[0, 1, 0].item() == 0
    assert idx[0, 2, 0].item() == 1


def test_sense_range_excludes_far_humans_from_topk():
    policy = SNCPPolicy(sparse_hig=True, hh_topk=2, sense_range=1.0)
    positions = [[[0.2, 0.0], [0.3, 0.0], [50.0, 0.0]]]
    edges = _obs(1, 3, positions=positions)["spatial_edges"]
    dist = torch.hypot(edges[..., 0], edges[..., 1])
    mask = dist <= 1.0
    assert mask.tolist() == [[True, True, False]]

    idx, valid = policy._topk_neighbor_index(edges, mask)
    # Far human is never a neighbor; each near human has only the other near one.
    assert not bool(((idx[0] == 2) & valid[0]).any())
    assert idx[0, 0, 0].item() == 1
    assert idx[0, 1, 0].item() == 0
    assert int(valid[0, 0].sum().item()) == 1
    assert int(valid[0, 2].sum().item()) == 0

    hidden = policy.init_hidden(1, 3, torch.device("cpu"))
    outputs = policy(_obs(1, 3, positions=positions), hidden)
    for tensor in (*outputs[:3], *outputs[3].values()):
        assert torch.isfinite(tensor).all()


def test_all_masked_sense_range_stays_finite():
    policy = SNCPPolicy(sparse_hig=True, hh_topk=3, sense_range=0.1)
    hidden = policy.init_hidden(2, 5, torch.device("cpu"))
    outputs = policy(_obs(2, 5, far=True), hidden)
    for tensor in (*outputs[:3], *outputs[3].values()):
        assert torch.isfinite(tensor).all()


def test_sparse_hig_is_not_a_transformer_or_full_graph_module():
    policy = SNCPPolicy(sparse_hig=True, hh_topk=4, hh_attn_heads=4)
    assert policy.hh_topk <= 4
    assert not any(isinstance(mod, (nn.Transformer, nn.TransformerEncoder))
                   for mod in policy.modules())
    n_hh = sum(p.numel() for p in policy.hh_sparse_attn.parameters())
    assert n_hh < 400_000
    assert policy.hh_sparse_attn.embed_dim == 256
    assert policy.hh_sparse_attn.num_heads == 4


def test_k_above_four_is_rejected():
    with pytest.raises(ValueError, match="hh_topk"):
        SNCPPolicy(sparse_hig=True, hh_topk=5)


def test_cfc_sparsehig_risk_head_checkpoint_roundtrip():
    policy = SNCPPolicy(
        robot_vpref=1.0, robot_wmax=1.8,
        pre_mlp=True, meanmax_pool=True, action_dist="beta",
        cell_type="cfc", risk_head=True,
        sparse_hig=True, hh_topk=3,
    )
    state = policy.state_dict()
    assert detect_sparse_hig(state) is True
    rebuilt = build_policy_for_checkpoint(state, robot_vpref=1.0, robot_wmax=1.8)
    assert rebuilt.cell_type == "cfc"
    assert rebuilt.risk_head is True
    assert rebuilt.sparse_hig is True
    assert rebuilt.hh_topk == 3
    assert rebuilt.hh_intent_graph is True
    missing, unexpected = rebuilt.load_state_dict(state, strict=False)
    assert not missing and not unexpected
    load_policy_state_dict(rebuilt, state)


def test_dense_hh_and_sparsehig_weights_are_not_silently_interchangeable():
    dense = SNCPPolicy(hh_intent_graph=True)
    sparse = SNCPPolicy(sparse_hig=True, hh_topk=3)
    dense_state = dense.state_dict()
    sparse_state = sparse.state_dict()

    with pytest.raises(SparseHIGMismatchError, match="SparseHIG|dense"):
        assert_sparse_hig_compatible(dense, sparse_state)
    with pytest.raises(SparseHIGMismatchError, match="SparseHIG|dense"):
        load_policy_state_dict(sparse, dense_state)

    with pytest.raises(RuntimeError):
        dense.load_state_dict(sparse_state)
    with pytest.raises(RuntimeError):
        sparse.load_state_dict(dense_state)


def test_mixed_dense_and_sparse_hh_checkpoint_is_rejected():
    dense_state = SNCPPolicy(hh_intent_graph=True).state_dict()
    sparse_state = SNCPPolicy(sparse_hig=True, hh_topk=3).state_dict()
    mixed = dict(dense_state)
    mixed["hh_sparse_attn.in_proj_weight"] = sparse_state["hh_sparse_attn.in_proj_weight"]
    mixed["_hh_sparse_k"] = sparse_state["_hh_sparse_k"]
    with pytest.raises(SparseHIGMismatchError, match="both"):
        detect_sparse_hig(mixed)


def test_k_mismatch_is_a_hard_error():
    k3 = SNCPPolicy(sparse_hig=True, hh_topk=3)
    k4 = SNCPPolicy(sparse_hig=True, hh_topk=4)
    with pytest.raises(SparseHIGMismatchError, match="k"):
        assert_sparse_hig_compatible(k4, k3.state_dict())


def test_load_state_dict_syncs_k_from_buffer():
    k3 = SNCPPolicy(sparse_hig=True, hh_topk=3)
    k4 = SNCPPolicy(sparse_hig=True, hh_topk=4)
    k3.load_state_dict(k4.state_dict())
    assert k3.hh_topk == 4
    assert int(k3._hh_sparse_k.item()) == 4
    assert detect_hh_topk(k3.state_dict()) == 4


def test_sparsehig_checkpoint_without_k_buffer_is_rejected():
    state = SNCPPolicy(sparse_hig=True, hh_topk=3).state_dict()
    del state["_hh_sparse_k"]
    with pytest.raises(SparseHIGMismatchError, match="_hh_sparse_k"):
        detect_hh_topk(state)


def test_upgrade_rejects_dense_v37_as_already_hh():
    dense = SNCPPolicy(hh_intent_graph=True, robot_vpref=1.0, robot_wmax=1.8)
    with pytest.raises(RuntimeError, match="already v37"):
        build_upgraded_policy(
            dense.state_dict(), robot_vpref=1.0, robot_wmax=1.8,
            device=torch.device("cpu"), sparse_hig=True, hh_topk=3,
        )


def test_cli_exposes_sparse_hig_off_by_default():
    parser = build_parser()
    help_text = parser.format_help()
    defaults = parser.parse_args([])
    assert defaults.sparse_hig is False
    assert defaults.hh_topk == 3
    assert "--sparse_hig" in help_text
    assert "--hh_topk" in help_text
    tuned = parser.parse_args(["--sparse_hig", "--hh_topk", "4", "--temporal_cell", "cfc"])
    assert tuned.sparse_hig is True
    assert tuned.hh_topk == 4
    assert tuned.temporal_cell == "cfc"


def test_build_or_load_policy_sparse_hig_with_cfc_and_risk_head():
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    args = build_parser().parse_args([
        "--temporal_cell", "cfc", "--risk_head", "--sparse_hig", "--hh_topk", "3",
    ])
    policy = build_or_load_policy(args, env, torch.device("cpu"))
    assert policy.cell_type == "cfc"
    assert policy.risk_head is True
    assert policy.sparse_hig is True
    assert policy.hh_topk == 3
    assert policy.hh_intent_graph is True


def test_init_checkpoint_rejects_dense_hh_when_sparse_hig_requested(tmp_path):
    checkpoint = tmp_path / "dense_hh.pt"
    torch.save(SNCPPolicy(hh_intent_graph=True, robot_vpref=1.0, robot_wmax=1.8).state_dict(),
               checkpoint)
    args = build_parser().parse_args([
        "--init_checkpoint", str(checkpoint), "--sparse_hig",
    ])
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    with pytest.raises(ValueError, match="SparseHIG|dense"):
        build_or_load_policy(args, env, torch.device("cpu"))


def test_init_checkpoint_autodetects_sparse_hig(tmp_path):
    src = SNCPPolicy(
        robot_vpref=1.0, robot_wmax=1.8,
        cell_type="cfc", risk_head=True, sparse_hig=True, hh_topk=4,
    )
    checkpoint = tmp_path / "sparse.pt"
    torch.save(src.state_dict(), checkpoint)
    args = build_parser().parse_args([
        "--init_checkpoint", str(checkpoint), "--temporal_cell", "cfc",
    ])
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    policy = build_or_load_policy(args, env, torch.device("cpu"))
    assert policy.sparse_hig is True
    assert policy.hh_topk == 4
    assert policy.cell_type == "cfc"


def test_upgrade_checkpoint_can_attach_zero_gated_sparse_hig():
    base = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, action_dist="beta", pre_mlp=True)
    upgraded = build_upgraded_policy(
        base.state_dict(), robot_vpref=1.0, robot_wmax=1.8,
        device=torch.device("cpu"), sparse_hig=True, hh_topk=3,
    )
    assert upgraded.sparse_hig is True
    assert upgraded.hh_topk == 3
    assert float(upgraded.hh_gate.item()) == 0.0


def test_cfc_sparsehig_vectorized_ppo_lagrange_smoke():
    torch.manual_seed(0)
    num_envs, horizon, humans = 2, 8, 4
    policy = SNCPPolicy(
        cell_type="cfc", risk_head=True, sparse_hig=True, hh_topk=3,
    )
    agent = PPOAgent(
        policy, epochs=1, batch_size=4, seq_len=horizon,
        normalize_returns=False, use_lagrange=True,
        lagrange_lambda_init=0.1, lagrange_cost_limit=0.0,
    )
    buf = VectorizedRolloutBuffer(num_envs=num_envs, horizon=horizon)
    hidden = policy.init_hidden(num_envs, humans, torch.device("cpu"))
    for _ in range(horizon):
        obs = {
            "robot_node": torch.randn(num_envs, 7),
            "spatial_edges": torch.randn(num_envs, humans, 6),
            "temporal_edges": torch.randn(num_envs, 2),
        }
        with torch.no_grad():
            mu, std, value, new_h = policy(obs, hidden)
            dist = policy.make_action_dist(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        buf.store(
            obs=obs, hidden=hidden, actions=action, log_probs=log_prob,
            rewards=torch.randn(num_envs) * 0.1, values=value.squeeze(-1),
            dones=torch.zeros(num_envs), masks=torch.ones(num_envs),
            coll_labels=torch.tensor([1.0, 0.0]),
            clearance_labels=torch.tensor([0.0, 1.5]),
            cost_values=policy.last_cost_value.squeeze(-1).detach(),
        )
        hidden = new_h
    buf.finish(last_values=torch.zeros(num_envs), last_dones=torch.zeros(num_envs),
               last_cost_values=torch.zeros(num_envs))
    before = [p.clone() for p in policy.parameters() if p.requires_grad]
    agent.update_vectorized(buf, torch.device("cpu"))
    after = [p for p in policy.parameters() if p.requires_grad]
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert policy.last_p_coll is not None
    assert agent.lagrange_lambda >= 0.0
    assert policy.sparse_hig is True
