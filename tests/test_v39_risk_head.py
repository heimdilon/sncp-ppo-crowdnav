"""v39: Pi-friendly risk head + privileged short-horizon labels + Lagrangian PPO.

Runtime inference must stay a single policy forward with the action shield OFF.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
from sncp_ppo.ppo import PPOAgent
from sncp_ppo.risk_labeler import (
    RiskLabel,
    label_short_horizon_risk,
    label_vectorized_envs,
)
from sncp_ppo.risk_losses import dual_ascent_update, risk_supervision_loss
from sncp_ppo.train import build_or_load_policy, build_parser
from sncp_ppo.vec_buffer import VectorizedRolloutBuffer


def _obs(batch=2, humans=5):
    return {
        "robot_node": torch.linspace(-1.0, 1.0, batch * 7).reshape(batch, 7),
        "spatial_edges": torch.linspace(-1.5, 1.5, batch * humans * 6).reshape(batch, humans, 6),
        "temporal_edges": torch.linspace(-0.5, 0.5, batch * 2).reshape(batch, 2),
    }


class DummyEnv:
    def __init__(self):
        self.robot_px = 0.0
        self.robot_py = 0.0
        self.robot_theta = 0.0
        self.robot_vpref = 1.0
        self.robot_wmax = 1.8
        self.robot_gx = 5.0
        self.robot_gy = 0.0
        self.time_step = 0.25
        self.collision_threshold = 0.3
        self.humans_px = np.array([0.55], dtype=float)
        self.humans_py = np.array([0.0], dtype=float)
        self.humans_vx = np.array([0.0], dtype=float)
        self.humans_vy = np.array([0.0], dtype=float)


def test_default_policy_has_no_risk_head_surface():
    policy = SNCPPolicy()
    keys = set(policy.state_dict())
    assert policy.risk_head is False
    assert not any(key.startswith("risk_mlp.") for key in keys)
    assert "_risk_head" not in keys
    assert not any(key.startswith("cost_critic.") for key in keys)


def test_risk_head_shapes_and_bounds():
    policy = SNCPPolicy(risk_head=True)
    hidden = policy.init_hidden(3, 4, torch.device("cpu"))
    out1, out2, value, new_hidden = policy(_obs(3, 4), hidden)

    assert out1.shape == (3, 2)
    assert out2.shape == (3, 2)
    assert value.shape == (3, 1)
    assert policy.last_p_coll.shape == (3, 1)
    assert policy.last_min_clearance.shape == (3, 1)
    assert policy.last_cost_value.shape == (3, 1)
    assert torch.all((policy.last_p_coll >= 0.0) & (policy.last_p_coll <= 1.0))
    assert torch.all(policy.last_min_clearance >= 0.0)
    for tensor in (out1, out2, value, policy.last_p_coll, policy.last_min_clearance,
                   policy.last_cost_value, *new_hidden.values()):
        assert torch.isfinite(tensor).all()


def test_risk_head_is_tiny():
    policy = SNCPPolicy(risk_head=True)
    n_params = sum(p.numel() for p in policy.risk_mlp.parameters())
    assert n_params < 10_000


def test_old_checkpoint_still_loads_without_risk_head():
    state = SNCPPolicy(action_dist="beta", pre_mlp=True, meanmax_pool=True).state_dict()
    rebuilt = build_policy_for_checkpoint(state, robot_vpref=1.0, robot_wmax=1.8)
    rebuilt.load_state_dict(state)
    assert rebuilt.risk_head is False
    missing, unexpected = rebuilt.load_state_dict(state, strict=False)
    assert not missing and not unexpected


def test_risk_head_checkpoint_autodetect_roundtrip():
    policy = SNCPPolicy(
        robot_vpref=1.0, robot_wmax=1.8,
        pre_mlp=True, meanmax_pool=True, action_dist="beta", risk_head=True,
    )
    state = policy.state_dict()
    rebuilt = build_policy_for_checkpoint(state, robot_vpref=1.0, robot_wmax=1.8)
    assert rebuilt.risk_head is True
    missing, unexpected = rebuilt.load_state_dict(state, strict=False)
    assert not missing and not unexpected


def test_labeler_is_deterministic():
    env = DummyEnv()
    action = np.array([1.0, 0.0], dtype=np.float32)
    first = label_short_horizon_risk(env, action, horizon_steps=6)
    second = label_short_horizon_risk(env, action, horizon_steps=6)
    assert first == second
    assert isinstance(first, RiskLabel)
    assert first.collision in (0.0, 1.0)
    assert first.min_clearance >= 0.0


def test_labeler_flags_head_on_collision_and_empty_scene_is_safe():
    colliding = DummyEnv()
    colliding_label = label_short_horizon_risk(
        colliding, np.array([1.0, 0.0], dtype=np.float32), horizon_steps=4,
    )
    assert colliding_label.collision == 1.0
    assert colliding_label.min_clearance == pytest.approx(0.0)

    empty = DummyEnv()
    empty.humans_px = np.array([], dtype=float)
    empty.humans_py = np.array([], dtype=float)
    empty.humans_vx = np.array([], dtype=float)
    empty.humans_vy = np.array([], dtype=float)
    empty_label = label_short_horizon_risk(
        empty, np.array([1.0, 0.0], dtype=np.float32), horizon_steps=4,
    )
    assert empty_label.collision == 0.0
    assert empty_label.min_clearance > 1.0


def test_label_vectorized_envs_matches_single_env():
    class _Vec:
        def __init__(self, envs):
            self.envs = envs

    env = DummyEnv()
    wrapped = SimpleNamespace(unwrapped=env)
    actions = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    coll, clearance = label_vectorized_envs(_Vec([wrapped, wrapped]), actions, horizon_steps=4)
    single_a = label_short_horizon_risk(env, actions[0], horizon_steps=4)
    single_b = label_short_horizon_risk(env, actions[1], horizon_steps=4)
    assert coll[0] == pytest.approx(single_a.collision)
    assert clearance[0] == pytest.approx(single_a.min_clearance)
    assert coll[1] == pytest.approx(single_b.collision)
    assert clearance[1] == pytest.approx(single_b.min_clearance)


def test_risk_losses_decrease_on_trivial_synthetic_batch():
    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(head.parameters(), lr=5e-2)
    features = torch.randn(64, 8)
    coll = (features[:, 0] > 0).float()
    clearance = torch.relu(features[:, 1])

    def _split(raw):
        return torch.sigmoid(raw[:, 0]), F.softplus(raw[:, 1])

    p0, c0 = _split(head(features))
    bce0, huber0, _ = risk_supervision_loss(p0, c0, coll, clearance)
    for _ in range(80):
        p_coll, clr = _split(head(features))
        _, _, total = risk_supervision_loss(p_coll, clr, coll, clearance)
        opt.zero_grad()
        total.backward()
        opt.step()
    p1, c1 = _split(head(features))
    bce1, huber1, _ = risk_supervision_loss(p1, c1, coll, clearance)
    assert bce1.item() < bce0.item()
    assert huber1.item() < huber0.item()


def test_dual_ascent_increases_lambda_when_constraint_violated():
    lam = dual_ascent_update(lam=0.0, mean_cost=0.20, cost_limit=0.05, lr=0.5, lam_max=10.0)
    assert lam == pytest.approx(0.075)
    lam = dual_ascent_update(lam=0.5, mean_cost=0.01, cost_limit=0.05, lr=1.0, lam_max=10.0)
    assert lam == pytest.approx(0.46)
    lam = dual_ascent_update(lam=9.9, mean_cost=1.0, cost_limit=0.0, lr=1.0, lam_max=10.0)
    assert lam == pytest.approx(10.0)


def test_inference_forward_does_not_require_action_shield():
    import inspect

    from sncp_ppo import models as models_mod

    source = inspect.getsource(models_mod)
    assert "action_shield" not in source
    assert "shield_action" not in source
    assert "risk_labeler" not in source

    policy = SNCPPolicy(risk_head=True, action_dist="beta")
    hidden = policy.init_hidden(1, 3, torch.device("cpu"))
    mu, std, value, new_hidden = policy(_obs(1, 3), hidden)
    action = policy.deterministic_action(mu, std)
    assert action.shape == (1, 2)
    assert value.shape == (1, 1)
    assert new_hidden["node"].shape[0] == 1


def test_select_action_stays_four_tuple_with_risk_head():
    policy = SNCPPolicy(risk_head=True)
    agent = PPOAgent(policy)
    obs = {
        "robot_node": np.zeros(7, dtype=np.float32),
        "spatial_edges": np.zeros((2, 6), dtype=np.float32),
        "temporal_edges": np.zeros(2, dtype=np.float32),
    }
    hidden = policy.init_hidden(1, 2, torch.device("cpu"))
    action, log_prob, value, new_hidden = agent.select_action(
        obs, hidden, torch.device("cpu"), deterministic=True,
    )
    assert action.shape == (2,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert "node" in new_hidden


def test_cli_exposes_v39_flags_with_documented_defaults():
    parser = build_parser()
    help_text = parser.format_help()
    defaults = parser.parse_args([])
    assert defaults.risk_head is False
    assert defaults.lagrange_ppo is False
    assert defaults.risk_horizon == 6
    assert defaults.risk_bce_coef == pytest.approx(1.0)
    assert defaults.risk_clearance_coef == pytest.approx(0.1)
    assert defaults.lagrange_cost_limit == pytest.approx(0.05)
    assert defaults.lagrange_lr == pytest.approx(0.01)
    collapsed = " ".join(help_text.lower().split())
    assert "runtime action shield" in collapsed or "not a runtime shield" in collapsed

    enabled = parser.parse_args(["--risk_head", "--lagrange_ppo", "--risk_horizon", "6"])
    assert enabled.risk_head is True
    assert enabled.lagrange_ppo is True


def test_init_checkpoint_can_attach_fresh_risk_head(tmp_path):
    checkpoint = tmp_path / "v34.pt"
    base = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, action_dist="beta",
                      pre_mlp=True, meanmax_pool=True)
    torch.save(base.state_dict(), checkpoint)
    args = build_parser().parse_args([
        "--init_checkpoint", str(checkpoint), "--risk_head", "--lagrange_ppo",
    ])
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    policy = build_or_load_policy(args, env, torch.device("cpu"))
    assert policy.risk_head is True
    assert policy.action_dist == "beta"
    hidden = policy.init_hidden(1, 4, torch.device("cpu"))
    policy(_obs(1, 4), hidden)
    assert policy.last_p_coll.shape == (1, 1)


def test_lagrange_flag_implies_risk_head_on_fresh_build():
    args = build_parser().parse_args(["--lagrange_ppo", "--action_dist", "beta"])
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    policy = build_or_load_policy(args, env, torch.device("cpu"))
    assert policy.risk_head is True


def test_update_vectorized_with_risk_and_lagrange_runs():
    torch.manual_seed(0)
    N, T, H = 2, 8, 3
    policy = SNCPPolicy(risk_head=True)
    agent = PPOAgent(
        policy, epochs=1, batch_size=4, seq_len=T,
        normalize_returns=False, use_lagrange=True,
        lagrange_lambda_init=0.1, lagrange_cost_limit=0.0,
    )
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    hidden = policy.init_hidden(N, H, torch.device("cpu"))
    for _ in range(T):
        obs = {
            "robot_node": torch.randn(N, 7),
            "spatial_edges": torch.randn(N, H, 6),
            "temporal_edges": torch.randn(N, 2),
        }
        with torch.no_grad():
            mu, std, value, new_h = policy(obs, hidden)
            dist = policy.make_action_dist(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        buf.store(
            obs=obs, hidden=hidden, actions=action, log_probs=log_prob,
            rewards=torch.randn(N) * 0.1, values=value.squeeze(-1),
            dones=torch.zeros(N), masks=torch.ones(N),
            coll_labels=torch.tensor([1.0, 0.0]),
            clearance_labels=torch.tensor([0.0, 1.5]),
            cost_values=policy.last_cost_value.squeeze(-1).detach(),
        )
        hidden = new_h
    buf.finish(last_values=torch.zeros(N), last_dones=torch.zeros(N),
               last_cost_values=torch.zeros(N))
    before = [p.clone() for p in policy.risk_mlp.parameters()]
    agent.update_vectorized(buf, torch.device("cpu"))
    after = list(policy.risk_mlp.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert agent.lagrange_lambda >= 0.0
    assert isinstance(agent.last_risk_bce, float)


def test_cost_critic_is_independent_of_p_coll():
    torch.manual_seed(0)
    policy = SNCPPolicy(risk_head=True)
    with torch.no_grad():
        policy.cost_critic.bias.fill_(2.5)
        policy.risk_mlp[-1].bias[0] = -8.0
    hidden = policy.init_hidden(2, 3, torch.device("cpu"))
    policy(_obs(2, 3), hidden)
    assert policy.last_cost_value.shape == (2, 1)
    assert not torch.allclose(policy.last_cost_value, policy.last_p_coll)


def test_overlay_truncation_obs_prefers_final_observation():
    from sncp_ppo.train import overlay_truncation_obs

    next_obs = {
        "robot_node": np.zeros((2, 7), dtype=np.float32),
        "spatial_edges": np.zeros((2, 3, 6), dtype=np.float32),
        "temporal_edges": np.zeros((2, 2), dtype=np.float32),
    }
    final = {
        "robot_node": np.ones(7, dtype=np.float32),
        "spatial_edges": np.ones((3, 6), dtype=np.float32),
        "temporal_edges": np.ones(2, dtype=np.float32),
    }
    info = {"final_observation": [final, None]}
    out = overlay_truncation_obs(next_obs, info, [0])
    assert np.allclose(out["robot_node"][0], 1.0)
    assert np.allclose(out["robot_node"][1], 0.0)
    # NEXT_STEP autoreset: no final_observation, next_obs already is s_final.
    copied = overlay_truncation_obs(next_obs, {}, [0])
    assert np.allclose(copied["robot_node"], next_obs["robot_node"])


def test_upgrade_checkpoint_with_lagrange_attaches_risk_and_cost_heads(tmp_path):
    checkpoint = tmp_path / "v34.pt"
    base = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, action_dist="beta",
                      pre_mlp=True, meanmax_pool=True)
    torch.save(base.state_dict(), checkpoint)
    args = build_parser().parse_args([
        "--upgrade_checkpoint", str(checkpoint), "--lagrange_ppo",
    ])
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    policy = build_or_load_policy(args, env, torch.device("cpu"))
    assert policy.hh_intent_graph is True
    assert policy.risk_head is True
    hidden = policy.init_hidden(1, 4, torch.device("cpu"))
    policy(_obs(1, 4), hidden)
    assert policy.last_p_coll.shape == (1, 1)
    assert policy.last_cost_value.shape == (1, 1)


def test_update_vectorized_trains_cost_critic_not_as_p_coll():
    torch.manual_seed(1)
    N, T, H = 2, 8, 3
    policy = SNCPPolicy(risk_head=True)
    agent = PPOAgent(
        policy, epochs=1, batch_size=4, seq_len=T,
        normalize_returns=False, use_lagrange=True,
        lagrange_lambda_init=0.1, lagrange_cost_limit=0.0,
    )
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    hidden = policy.init_hidden(N, H, torch.device("cpu"))
    for _ in range(T):
        obs = {
            "robot_node": torch.randn(N, 7),
            "spatial_edges": torch.randn(N, H, 6),
            "temporal_edges": torch.randn(N, 2),
        }
        with torch.no_grad():
            mu, std, value, new_h = policy(obs, hidden)
            dist = policy.make_action_dist(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        buf.store(
            obs=obs, hidden=hidden, actions=action, log_probs=log_prob,
            rewards=torch.randn(N) * 0.1, values=value.squeeze(-1),
            dones=torch.zeros(N), masks=torch.ones(N),
            coll_labels=torch.tensor([1.0, 0.0]),
            clearance_labels=torch.tensor([0.0, 1.5]),
            cost_values=policy.last_cost_value.squeeze(-1).detach(),
        )
        hidden = new_h
    buf.finish(last_values=torch.zeros(N), last_dones=torch.zeros(N),
               last_cost_values=torch.zeros(N))
    before = [p.clone() for p in policy.cost_critic.parameters()]
    agent.update_vectorized(buf, torch.device("cpu"))
    after = list(policy.cost_critic.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_diagnostic_row_includes_lagrange_and_risk_fields():
    from sncp_ppo.train import update_diagnostic_row

    class _Rms:
        std = 1.0

    class _Agent:
        last_entropy = 0.1
        last_approx_kl = 0.01
        return_rms = _Rms()
        lagrange_lambda = 0.25
        last_risk_bce = 0.4
        last_risk_huber = 0.05
        last_mean_cost = 0.12

    row = update_diagnostic_row(SNCPPolicy(), _Agent())
    assert row[-4:] == ["0.250000", "0.400000", "0.050000", "0.120000"]
