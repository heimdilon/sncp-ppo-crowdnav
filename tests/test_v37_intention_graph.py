from types import SimpleNamespace

import pytest
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
from sncp_ppo.train import (
    build_or_load_policy,
    build_parser,
    build_upgraded_policy,
    update_diagnostic_row,
)


def _obs(batch: int = 2, humans: int = 5, *, far: bool = False):
    spatial = torch.linspace(-1.5, 1.5, batch * humans * 6).reshape(batch, humans, 6)
    if far:
        spatial[..., 0:2] += 100.0
    return {
        "robot_node": torch.linspace(-1.0, 1.0, batch * 7).reshape(batch, 7),
        "spatial_edges": spatial,
        "temporal_edges": torch.linspace(-0.5, 0.5, batch * 2).reshape(batch, 2),
    }


def _assert_tree_close(left, right, atol=1e-6):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_close(left[key], right[key], atol=atol)
        return
    if isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_tree_close(left_item, right_item, atol=atol)
        return
    assert torch.allclose(left, right, atol=atol, rtol=0.0)


def _base_beta_policy():
    return SNCPPolicy(
        robot_vpref=1.0,
        robot_wmax=1.8,
        pre_mlp=True,
        meanmax_pool=True,
        action_dist="beta",
    )


def test_default_policy_has_no_v37_state_surface():
    policy = SNCPPolicy()
    keys = set(policy.state_dict())
    assert policy.hh_intent_graph is False
    assert not any(key.startswith(("cv_encoder.", "hh_norm.", "hh_attn.")) for key in keys)
    assert "hh_gate" not in keys
    assert not any(key.startswith("_hh_") or key.startswith("_cv_") for key in keys)


def test_v37_build_registers_modules_and_persistent_config():
    policy = SNCPPolicy(hh_intent_graph=True, hh_attn_heads=4,
                        cv_horizons=(1, 2, 3, 4), cv_dt=0.25)
    state = policy.state_dict()
    assert policy.hh_intent_graph is True
    assert policy.hh_attn.num_heads == 4
    assert policy.cv_encoder[0].in_features == 8
    assert float(state["_hh_intent_graph"].item()) == 1.0
    assert int(state["_hh_attn_heads"].item()) == 4
    assert state["_cv_horizons"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert float(state["_cv_dt"].item()) == pytest.approx(0.25)
    assert float(policy.hh_gate.item()) == 0.0


def test_constant_velocity_features_are_exact():
    policy = SNCPPolicy(hh_intent_graph=True, cv_horizons=(1, 3), cv_dt=0.5)
    edges = torch.tensor([[[1.0, 2.0, 0.5, -1.0, 9.0, 9.0]]])
    actual = policy._constant_velocity_features(edges)
    expected = torch.tensor([[[1.25, 1.5, 1.75, 0.5]]])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("humans", [1, 10, 25])
def test_v37_forward_is_finite_for_variable_human_counts(humans):
    policy = SNCPPolicy(pre_mlp=True, meanmax_pool=True, action_dist="beta",
                        hh_intent_graph=True)
    hidden = policy.init_hidden(1, humans, torch.device("cpu"))
    out1, out2, value, new_hidden = policy(_obs(1, humans), hidden)
    for tensor in (out1, out2, value, *new_hidden.values()):
        assert torch.isfinite(tensor).all()


def test_v37_all_masked_humans_stays_finite():
    policy = SNCPPolicy(pre_mlp=True, meanmax_pool=True, action_dist="beta",
                        sense_range=0.1, hh_intent_graph=True)
    hidden = policy.init_hidden(2, 5, torch.device("cpu"))
    outputs = policy(_obs(2, 5, far=True), hidden)
    for tensor in (*outputs[:3], *outputs[3].values()):
        assert torch.isfinite(tensor).all()


def test_gate_zero_upgrade_is_base_equivalent_including_hidden_state():
    base = _base_beta_policy()
    state = base.state_dict()
    upgraded = build_upgraded_policy(
        state,
        robot_vpref=1.0,
        robot_wmax=1.8,
        device=torch.device("cpu"),
        hh_attn_heads=4,
        cv_horizons=(1, 2, 3, 4),
        cv_dt=0.25,
    )
    obs = _obs(2, 7)
    base_out = base(obs, base.init_hidden(2, 7, torch.device("cpu")))
    upgraded_out = upgraded(obs, upgraded.init_hidden(2, 7, torch.device("cpu")))
    _assert_tree_close(base_out[:3], upgraded_out[:3])
    _assert_tree_close(base_out[3], upgraded_out[3])


def test_nonzero_gate_changes_policy_output_and_gets_gradient():
    base = _base_beta_policy()
    upgraded = build_upgraded_policy(
        base.state_dict(), robot_vpref=1.0, robot_wmax=1.8,
        device=torch.device("cpu"), hh_attn_heads=4,
        cv_horizons=(1, 2, 3, 4), cv_dt=0.25,
    )
    upgraded.hh_gate.data.fill_(0.2)
    obs = _obs(2, 8)
    base_out = base(obs, base.init_hidden(2, 8, torch.device("cpu")))
    upgraded_out = upgraded(obs, upgraded.init_hidden(2, 8, torch.device("cpu")))
    assert any(not torch.allclose(a, b) for a, b in zip(base_out[:3], upgraded_out[:3]))

    loss = sum(tensor.sum() for tensor in upgraded_out[:3])
    loss.backward()
    assert upgraded.hh_gate.grad is not None
    assert torch.isfinite(upgraded.hh_gate.grad)
    assert float(upgraded.hh_gate.grad.abs()) > 0.0


def test_v37_checkpoint_autodetect_roundtrip():
    policy = SNCPPolicy(pre_mlp=True, meanmax_pool=True, action_dist="beta",
                        hh_intent_graph=True, hh_attn_heads=4,
                        cv_horizons=(1, 2, 3, 4), cv_dt=0.25)
    state = policy.state_dict()
    rebuilt = build_policy_for_checkpoint(state, robot_vpref=1.0, robot_wmax=1.8)
    assert rebuilt.hh_intent_graph is True
    assert rebuilt.hh_attn_heads == 4
    assert rebuilt.cv_horizons == (1.0, 2.0, 3.0, 4.0)
    assert rebuilt.cv_dt == pytest.approx(0.25)
    missing, unexpected = rebuilt.load_state_dict(state, strict=False)
    assert not missing and not unexpected


def test_combined_v36_checkpoint_can_be_safely_upgraded():
    v36 = SNCPPolicy(
        robot_vpref=1.0, robot_wmax=1.8,
        pre_mlp=True, meanmax_pool=True,
        node_units=256, node_output=96,
        attn_heads=4, attn_count_scaling=True,
        action_dist="beta", sense_range=6.0,
    )
    upgraded = build_upgraded_policy(
        v36.state_dict(), robot_vpref=1.0, robot_wmax=1.8,
        device=torch.device("cpu"), hh_attn_heads=4,
        cv_horizons=(1, 2, 3, 4), cv_dt=0.25,
    )
    assert upgraded.hh_intent_graph is True
    assert upgraded.node_units == 256 and upgraded.node_output == 96
    assert upgraded.attn_heads == 4 and upgraded.attn_count_scaling is True
    assert upgraded.action_dist == "beta" and upgraded.sense_range == pytest.approx(6.0)


def test_upgrade_rejects_missing_base_weight():
    state = _base_beta_policy().state_dict()
    del state["critic.0.weight"]
    with pytest.raises(RuntimeError, match="unsafe checkpoint upgrade"):
        build_upgraded_policy(
            state, robot_vpref=1.0, robot_wmax=1.8, device=torch.device("cpu"),
            hh_attn_heads=4, cv_horizons=(1, 2, 3, 4), cv_dt=0.25,
        )


def test_cli_and_build_or_load_support_v37_upgrade(tmp_path):
    checkpoint = tmp_path / "v34.pt"
    torch.save(_base_beta_policy().state_dict(), checkpoint)
    parsed = build_parser().parse_args([
        "--upgrade_checkpoint", str(checkpoint), "--hh_intent_graph",
        "--hh_attn_heads", "4", "--cv_horizons", "1", "2", "3", "4",
        "--cv_dt", "0.25",
    ])
    assert parsed.hh_intent_graph is True
    assert parsed.cv_horizons == [1, 2, 3, 4]

    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    policy = build_or_load_policy(parsed, env, torch.device("cpu"))
    assert policy.hh_intent_graph is True
    assert policy.action_dist == "beta"


def test_init_and_upgrade_checkpoint_are_mutually_exclusive():
    args = SimpleNamespace(init_checkpoint="a.pt", upgrade_checkpoint="b.pt")
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_or_load_policy(args, env, torch.device("cpu"))


def test_diagnostic_row_logs_gate_only_for_v37():
    class _Rms:
        std = 1.0

    class _Agent:
        last_entropy = -0.5
        last_approx_kl = 0.01
        return_rms = _Rms()

    base_row = update_diagnostic_row(SNCPPolicy(), _Agent())
    v37 = SNCPPolicy(hh_intent_graph=True)
    v37.hh_gate.data.fill_(-0.125)
    v37_row = update_diagnostic_row(v37, _Agent())
    assert len(base_row) == 10 and base_row[5] == ""
    assert len(v37_row) == 10 and float(v37_row[5]) == pytest.approx(-0.125)
