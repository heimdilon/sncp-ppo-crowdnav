"""CfC-NCP side branch: optional Closed-form Continuous-time cells.

Default remains LTC so every existing checkpoint stays load-compatible.
CfC is a drop-in AutoNCP cell swap (temporal + spatial + node). Mixing LTC
and CfC weights must fail loudly — never silently.
"""
from types import SimpleNamespace

import pytest
import torch
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP

from sncp_ppo.models import (
    SNCPPolicy,
    assert_cell_type_compatible,
    build_policy_for_checkpoint,
    detect_cell_type,
    load_policy_state_dict,
)
from sncp_ppo.ppo import PPOAgent
from sncp_ppo.train import build_or_load_policy, build_parser
from sncp_ppo.vec_buffer import VectorizedRolloutBuffer


def _obs(batch=2, humans=5):
    return {
        "robot_node": torch.linspace(-1.0, 1.0, batch * 7).reshape(batch, 7),
        "spatial_edges": torch.linspace(-1.5, 1.5, batch * humans * 6).reshape(batch, humans, 6),
        "temporal_edges": torch.linspace(-0.5, 0.5, batch * 2).reshape(batch, 2),
    }


def test_default_policy_is_ltc_and_has_no_cfc_surface():
    policy = SNCPPolicy()
    keys = set(policy.state_dict())
    assert policy.cell_type == "ltc"
    assert isinstance(policy.temporal_ltc, LTC)
    assert isinstance(policy.spatial_ltc, LTC)
    assert isinstance(policy.node_ltc, LTC)
    assert not any(key.startswith(("temporal_cfc.", "spatial_cfc.", "node_cfc.")) for key in keys)
    assert "_cell_type_cfc" not in keys
    assert detect_cell_type(policy.state_dict()) == "ltc"


def test_invalid_cell_type_is_rejected():
    with pytest.raises(ValueError, match="cell_type"):
        SNCPPolicy(cell_type="gru")


def test_cfc_policy_uses_autoncp_and_cfc_cells():
    policy = SNCPPolicy(cell_type="cfc")
    assert policy.cell_type == "cfc"
    for name, wiring, cell in [
        ("temporal", policy.temporal_wiring, policy.temporal_cfc),
        ("spatial", policy.spatial_wiring, policy.spatial_cfc),
        ("node", policy.node_wiring, policy.node_cfc),
    ]:
        assert isinstance(wiring, AutoNCP), name
        assert isinstance(cell, CfC), name
        assert wiring.output_dim < wiring.units
    assert not hasattr(policy, "temporal_ltc")
    keys = set(policy.state_dict())
    assert any(key.startswith("temporal_cfc.") for key in keys)
    assert any(key.startswith("spatial_cfc.") for key in keys)
    assert any(key.startswith("node_cfc.") for key in keys)
    assert "_cell_type_cfc" in keys
    assert "temporal_ltc.rnn_cell.gleak" not in keys


def test_cfc_forward_contract_and_hidden_shapes():
    policy = SNCPPolicy(robot_vpref=0.26, robot_wmax=1.8, cell_type="cfc")
    batch, humans = 3, 4
    hidden = policy.init_hidden(batch, humans, torch.device("cpu"))
    assert hidden["temporal_edge"].shape == (batch, policy.temporal_wiring.units)
    assert hidden["spatial_edge"].shape == (batch * humans, policy.spatial_wiring.units)
    assert hidden["node"].shape == (batch, policy.node_wiring.units)

    out1, out2, value, new_hidden = policy(_obs(batch, humans), hidden)
    assert len((out1, out2, value, new_hidden)) == 4
    assert out1.shape == (batch, 2)
    assert out2.shape == (batch, 2)
    assert value.shape == (batch, 1)
    assert torch.all((out1[:, 0] >= 0.0) & (out1[:, 0] <= 0.26))
    assert torch.all((out1[:, 1] >= -1.8) & (out1[:, 1] <= 1.8))
    for name in ("temporal_edge", "spatial_edge", "node"):
        assert new_hidden[name].shape == hidden[name].shape
        assert torch.isfinite(new_hidden[name]).all()
    assert torch.isfinite(out1).all() and torch.isfinite(out2).all() and torch.isfinite(value).all()


def test_cfc_proj_reads_motor_outputs():
    policy = SNCPPolicy(cell_type="cfc")
    assert policy.temporal_proj.in_features == policy.temporal_wiring.output_dim
    assert policy.spatial_proj.in_features == policy.spatial_wiring.output_dim
    assert policy.node_proj.in_features == policy.node_wiring.output_dim


def test_cfc_risk_head_and_cost_critic_still_work():
    policy = SNCPPolicy(cell_type="cfc", risk_head=True)
    hidden = policy.init_hidden(2, 5, torch.device("cpu"))
    out1, out2, value, new_hidden = policy(_obs(2, 5), hidden)
    assert out1.shape == (2, 2)
    assert value.shape == (2, 1)
    assert policy.last_p_coll.shape == (2, 1)
    assert policy.last_min_clearance.shape == (2, 1)
    assert policy.last_cost_value.shape == (2, 1)
    assert torch.all((policy.last_p_coll >= 0.0) & (policy.last_p_coll <= 1.0))
    assert torch.all(policy.last_min_clearance >= 0.0)
    assert "node" in new_hidden


def test_cfc_checkpoint_autodetect_roundtrip():
    policy = SNCPPolicy(
        robot_vpref=1.0, robot_wmax=1.8,
        pre_mlp=True, meanmax_pool=True, action_dist="beta",
        cell_type="cfc", risk_head=True, node_units=256, node_output=96,
    )
    state = policy.state_dict()
    assert detect_cell_type(state) == "cfc"
    rebuilt = build_policy_for_checkpoint(state, robot_vpref=1.0, robot_wmax=1.8)
    assert rebuilt.cell_type == "cfc"
    assert rebuilt.risk_head is True
    assert rebuilt.node_units == 256 and rebuilt.node_output == 96
    assert rebuilt.pre_mlp and rebuilt.meanmax_pool
    assert rebuilt.action_dist == "beta"
    missing, unexpected = rebuilt.load_state_dict(state, strict=False)
    assert not missing and not unexpected


def test_ltc_checkpoint_still_autodetects_as_ltc():
    state = SNCPPolicy(action_dist="beta", pre_mlp=True, meanmax_pool=True).state_dict()
    assert detect_cell_type(state) == "ltc"
    rebuilt = build_policy_for_checkpoint(state, robot_vpref=1.0, robot_wmax=1.8)
    assert rebuilt.cell_type == "ltc"
    rebuilt.load_state_dict(state)
    assert not hasattr(rebuilt, "temporal_cfc")


def test_cfc_and_ltc_weights_are_not_silently_interchangeable():
    ltc_state = SNCPPolicy().state_dict()
    cfc_state = SNCPPolicy(cell_type="cfc").state_dict()
    ltc_policy = SNCPPolicy()
    cfc_policy = SNCPPolicy(cell_type="cfc")

    with pytest.raises(ValueError, match="CfC checkpoint into a LTC|LTC checkpoint into a CfC"):
        assert_cell_type_compatible(ltc_policy, cfc_state)
    with pytest.raises(ValueError, match="CfC checkpoint into a LTC|LTC checkpoint into a CfC"):
        load_policy_state_dict(cfc_policy, ltc_state)

    with pytest.raises(RuntimeError):
        ltc_policy.load_state_dict(cfc_state)
    with pytest.raises(RuntimeError):
        cfc_policy.load_state_dict(ltc_state)


def test_mixed_encoder_checkpoint_is_rejected():
    state = SNCPPolicy().state_dict()
    cfc_state = SNCPPolicy(cell_type="cfc").state_dict()
    mixed = dict(state)
    mixed["temporal_cfc.rnn_cell.layer_0.ff1.weight"] = cfc_state[
        "temporal_cfc.rnn_cell.layer_0.ff1.weight"
    ]
    with pytest.raises(ValueError, match="both LTC and CfC"):
        detect_cell_type(mixed)


def test_cli_exposes_temporal_cell_default_ltc():
    parser = build_parser()
    help_text = parser.format_help()
    defaults = parser.parse_args([])
    assert defaults.temporal_cell == "ltc"
    assert "--temporal_cell" in help_text
    assert "--cell_type" in help_text
    tuned = parser.parse_args(["--temporal_cell", "cfc"])
    assert tuned.temporal_cell == "cfc"
    alias = parser.parse_args(["--cell_type", "cfc"])
    assert alias.temporal_cell == "cfc"
    collapsed = " ".join(help_text.lower().split())
    assert "closed-form" in collapsed or "cfc" in collapsed


def test_build_or_load_policy_respects_temporal_cell():
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    ltc_args = build_parser().parse_args([])
    cfc_args = build_parser().parse_args(["--temporal_cell", "cfc", "--risk_head"])
    ltc_policy = build_or_load_policy(ltc_args, env, torch.device("cpu"))
    cfc_policy = build_or_load_policy(cfc_args, env, torch.device("cpu"))
    assert ltc_policy.cell_type == "ltc"
    assert cfc_policy.cell_type == "cfc"
    assert cfc_policy.risk_head is True


def test_init_checkpoint_rejects_explicit_cell_mismatch(tmp_path):
    checkpoint = tmp_path / "ltc.pt"
    torch.save(SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8).state_dict(), checkpoint)
    args = build_parser().parse_args([
        "--init_checkpoint", str(checkpoint), "--temporal_cell", "cfc",
    ])
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    with pytest.raises(ValueError, match="LTC"):
        build_or_load_policy(args, env, torch.device("cpu"))


def test_init_checkpoint_rejects_cfc_file_when_cli_defaults_to_ltc(tmp_path):
    checkpoint = tmp_path / "cfc.pt"
    torch.save(SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, cell_type="cfc").state_dict(),
               checkpoint)
    args = build_parser().parse_args(["--init_checkpoint", str(checkpoint)])
    assert args.temporal_cell == "ltc"
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    with pytest.raises(ValueError, match="CfC"):
        build_or_load_policy(args, env, torch.device("cpu"))


def test_init_checkpoint_autodetects_cfc(tmp_path):
    src = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, cell_type="cfc", risk_head=True)
    checkpoint = tmp_path / "cfc.pt"
    torch.save(src.state_dict(), checkpoint)
    args = build_parser().parse_args([
        "--init_checkpoint", str(checkpoint), "--temporal_cell", "cfc",
    ])
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    policy = build_or_load_policy(args, env, torch.device("cpu"))
    assert policy.cell_type == "cfc"
    assert policy.risk_head is True
    for key, val in src.state_dict().items():
        assert torch.allclose(policy.state_dict()[key], val), key


def test_cfc_vectorized_ppo_smoke_with_risk_head():
    torch.manual_seed(0)
    num_envs, horizon, humans = 2, 8, 3
    policy = SNCPPolicy(cell_type="cfc", risk_head=True)
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
