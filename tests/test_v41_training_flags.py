"""v41 CLI knobs: weighted P(N), cost density adapt, KNN tokens, resume, denser holdout."""
import random as _random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
from sncp_ppo.ppo import PPOAgent
from sncp_ppo.train import (
    build_or_load_policy,
    build_parser,
    evaluate_holdout,
    normalize_train_args,
    select_vectorized_phase,
)
from sncp_ppo.v41_flags import (
    adapt_lagrange_cost,
    apply_v41_runtime_flags,
    count_n_near,
    expand_holdout_scenarios,
    pack_crowd_summary,
    parse_human_count_weights,
    resolve_holdout_arm,
    resolve_init_checkpoint_path,
    sample_human_count,
)
from sncp_ppo.vec_buffer import VectorizedRolloutBuffer


def test_parse_human_count_weights_and_reject_bad_spec():
    parsed = parse_human_count_weights("5:0.05,10:0.1,15:0.25,18:0.15,20:0.35")
    assert parsed == {5: 0.05, 10: 0.1, 15: 0.25, 18: 0.15, 20: 0.35}
    assert parse_human_count_weights(None) is None
    assert parse_human_count_weights("  ") is None
    with pytest.raises(ValueError, match="invalid"):
        parse_human_count_weights("5=0.1")
    with pytest.raises(ValueError, match="positive total"):
        parse_human_count_weights("5:0,10:0")


def test_sample_human_count_uniform_is_inclusive_randint():
    rng = _random.Random(0)
    seen = {sample_human_count(rng, (5, 20), None) for _ in range(400)}
    assert min(seen) == 5
    assert max(seen) == 20
    assert seen == set(range(5, 21))


def test_sample_human_count_respects_weights():
    rng = _random.Random(0)
    weights = {5: 0.0, 10: 0.1, 20: 0.9}
    counts = {5: 0, 10: 0, 20: 0}
    for _ in range(2000):
        n = sample_human_count(rng, (5, 20), weights)
        counts[n] += 1
    assert counts[5] == 0
    assert counts[20] > counts[10]
    assert counts[20] > 1500


def test_sample_human_count_errors_when_weights_miss_range():
    with pytest.raises(ValueError, match="no mass"):
        sample_human_count(_random.Random(0), (5, 8), {15: 1.0, 20: 1.0})


def test_select_vectorized_phase_weighted_vs_uniform():
    rng_u = _random.Random(1)
    uniform = {
        select_vectorized_phase(
            300000, 2_500_000, 10, rng=rng_u,
            fixed_scenario="paper_challenging", bootstrap_easy_steps=0,
            num_humans_range=(5, 20),
        )[0][1]
        for _ in range(80)
    }
    assert 5 in uniform and 20 in uniform
    assert len(uniform) >= 8

    rng_w = _random.Random(1)
    weighted = [
        select_vectorized_phase(
            300000, 2_500_000, 10, rng=rng_w,
            fixed_scenario="paper_challenging", bootstrap_easy_steps=0,
            num_humans_range=(5, 20),
            human_count_weights={5: 0.05, 10: 0.1, 15: 0.25, 18: 0.15, 20: 0.35},
        )[0][1]
        for _ in range(80)
    ]
    assert set(weighted) <= {5, 10, 15, 18, 20}
    assert weighted.count(20) > weighted.count(5)


def test_adapt_lagrange_cost_n_near_and_sqrt_n():
    cost = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    n_near = np.array([0.0, 4.0, 4.0], dtype=np.float32)
    scaled = adapt_lagrange_cost(cost, n_near=n_near, alpha=0.25, mode="n_near")
    np.testing.assert_allclose(scaled, [1.0, 1.0 / 2.0, 0.0])
    sqrt = adapt_lagrange_cost(cost, n_humans=16.0, mode="sqrt_n")
    np.testing.assert_allclose(sqrt, [0.25, 0.25, 0.0])


def test_count_n_near_from_spatial_edges():
    spatial = np.zeros((2, 4, 6), dtype=np.float32)
    spatial[0, 0, :2] = (0.5, 0.0)
    spatial[0, 1, :2] = (3.0, 0.0)
    spatial[1, :, 0] = 0.1
    near = count_n_near(spatial, radius=2.0)
    np.testing.assert_array_equal(near, [1.0, 4.0])


def test_parser_v41_defaults_preserve_old_cli():
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.human_count_weights is None
    assert defaults.cost_density_adapt is False
    assert defaults.cost_adapt_alpha == pytest.approx(0.15)
    assert defaults.cost_adapt_mode == "n_near"
    assert defaults.human_token_mode == "all"
    assert defaults.knn_k == 10
    assert defaults.holdout_densities is None
    assert defaults.resume is None
    assert defaults.load_path is None
    help_text = parser.format_help()
    assert "--human_count_weights" in help_text
    assert "--cost_density_adapt" in help_text
    assert "knn_sorted" in help_text
    assert "--resume" in help_text
    assert "SparseHIG" not in help_text or "NOT SparseHIG" in help_text


def test_parser_accepts_v41_flag_bundle():
    args = build_parser().parse_args([
        "--num_humans_range", "5", "20",
        "--human_count_weights", "5:0.05,10:0.1,15:0.25,18:0.15,20:0.35",
        "--cost_density_adapt", "--cost_adapt_alpha", "0.2",
        "--human_token_mode", "knn_sorted", "--knn_k", "10",
        "--resume", "checkpoints/sncp_ppo_v40.pt",
        "--holdout_scenarios", "paper_challenging",
        "--holdout_densities", "15", "20",
        "--lagrange_ppo",
    ])
    assert parse_human_count_weights(args.human_count_weights)[20] == pytest.approx(0.35)
    assert args.cost_density_adapt is True
    assert args.human_token_mode == "knn_sorted"
    assert args.resume.endswith("sncp_ppo_v40.pt")
    assert args.holdout_densities == [15, 20]


def test_resume_aliases_are_synonyms_and_conflict():
    a = SimpleNamespace(init_checkpoint="a.pt", resume=None, load_path=None)
    assert resolve_init_checkpoint_path(a) == "a.pt"
    b = SimpleNamespace(init_checkpoint=None, resume="b.pt", load_path=None)
    assert resolve_init_checkpoint_path(b) == "b.pt"
    c = SimpleNamespace(init_checkpoint="a.pt", resume="a.pt", load_path=None)
    assert resolve_init_checkpoint_path(c) == "a.pt"
    d = SimpleNamespace(init_checkpoint="a.pt", resume="b.pt", load_path=None)
    with pytest.raises(ValueError, match="conflicting"):
        resolve_init_checkpoint_path(d)


def test_holdout_density_expand_and_resolve():
    from sncp_ppo.train import SCENARIO_HOLDOUT_CONFIG

    expanded = expand_holdout_scenarios(["paper_challenging"], [15, 20])
    assert expanded == ["paper_challenging_n15", "paper_challenging_n20"]
    assert expand_holdout_scenarios(["easy", "hard"], [15, 20]) == ["easy", "hard"]
    assert expand_holdout_scenarios(["paper_challenging"], None) == ["paper_challenging"]

    base, n, vpref = resolve_holdout_arm("paper_challenging_n20", SCENARIO_HOLDOUT_CONFIG)
    assert (base, n, vpref) == ("paper_challenging", 20, 1.0)
    base10, n10, _ = resolve_holdout_arm("paper_challenging", SCENARIO_HOLDOUT_CONFIG)
    assert (base10, n10) == ("paper_challenging", 10)


def test_parser_accepts_named_n15_n20_holdout_arms():
    args = build_parser().parse_args([
        "--holdout_scenarios", "paper_challenging_n15", "paper_challenging_n20",
    ])
    assert args.holdout_scenarios == ["paper_challenging_n15", "paper_challenging_n20"]


def test_normalize_train_args_expands_densities_and_requires_range():
    args = build_parser().parse_args([
        "--holdout_scenarios", "paper_challenging",
        "--holdout_densities", "15", "20",
        "--resume", "ckpt.pt",
    ])
    normalize_train_args(args)
    assert args.init_checkpoint == "ckpt.pt"
    assert args.holdout_scenarios == ["paper_challenging_n15", "paper_challenging_n20"]

    bad = build_parser().parse_args(["--human_count_weights", "20:1.0"])
    with pytest.raises(ValueError, match="num_humans_range"):
        normalize_train_args(bad)


def test_evaluate_holdout_uses_alias_density_not_canonical_ten(monkeypatch):
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.train import SCENARIO_HOLDOUT_CONFIG

    captured = {}

    class DummyPolicy:
        def init_hidden(self, batch_size, num_humans, device):
            captured["num_humans"] = num_humans
            return {}

    class DummyAgent:
        @staticmethod
        def clip_action_for_env(action, vpref, wmax):
            return np.zeros(2, dtype=np.float32)

        def select_action(self, obs, h_states, device, deterministic=True):
            return np.zeros(2, dtype=np.float32), 0.0, 0.0, h_states

    env = CrowdSimEnv(num_humans=10, scenario="paper_challenging", max_time=0.25)
    env.reset = lambda seed=None: (
        {
            "robot_node": np.zeros(7, dtype=np.float32),
            "spatial_edges": np.zeros((env.num_humans, 6), dtype=np.float32),
            "temporal_edges": np.zeros(2, dtype=np.float32),
        },
        {},
    )

    def fake_step(_action):
        return (
            {
                "robot_node": np.zeros(7, dtype=np.float32),
                "spatial_edges": np.zeros((env.num_humans, 6), dtype=np.float32),
                "temporal_edges": np.zeros(2, dtype=np.float32),
            },
            0.0,
            True,
            False,
            {"success": True, "collision": False, "timeout": False, "I_sp": 0.0, "d_min": 1.0},
        )

    env.step = fake_step
    evaluate_holdout(
        env, DummyPolicy(), DummyAgent(), torch.device("cpu"),
        n_episodes=1, scenario="paper_challenging_n20", base_seed=0,
    )
    assert captured["num_humans"] == 20
    assert env.num_humans == 10  # restored
    assert env.scenario == "paper_challenging"
    # Canonical mapping still exists for the unsuffixed name.
    assert SCENARIO_HOLDOUT_CONFIG["paper_challenging"][0] == 10


def test_knn_sorted_does_not_change_state_dict_keys():
    base = SNCPPolicy(robot_vpref=1.0, action_dist="beta", pre_mlp=True, meanmax_pool=True, risk_head=True)
    knn = SNCPPolicy(
        robot_vpref=1.0, action_dist="beta", pre_mlp=True, meanmax_pool=True, risk_head=True,
        human_token_mode="knn_sorted", knn_k=10,
    )
    assert set(base.state_dict()) == set(knn.state_dict())


def test_knn_cap_keeps_full_hidden_and_prefers_nearest():
    torch.manual_seed(0)
    policy = SNCPPolicy(human_token_mode="knn_sorted", knn_k=2, crowd_summary=True)
    B, H = 1, 5
    spatial = torch.tensor([[
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]], dtype=torch.float32)
    obs = {
        "robot_node": torch.zeros(B, 7),
        "spatial_edges": spatial,
        "temporal_edges": torch.zeros(B, 2),
    }
    hidden = policy.init_hidden(B, H, torch.device("cpu"))
    hidden["spatial_edge"] = torch.arange(H * policy.spatial_wiring.units, dtype=torch.float32).reshape(H, -1)
    out1, out2, value, new_h = policy(obs, hidden)
    assert out1.shape == (1, 2)
    assert new_h["spatial_edge"].shape == hidden["spatial_edge"].shape
    assert policy.last_knn_idx.tolist() == [[3, 1]]
    old = hidden["spatial_edge"].reshape(H, -1)
    new = new_h["spatial_edge"].reshape(H, -1)
    for far in (0, 2, 4):
        assert torch.equal(old[far], new[far])


def test_pack_crowd_summary_encodes_count():
    spatial = np.zeros((3, 6), dtype=np.float32)
    spatial[0, :2] = (0.5, 0.0)
    token = pack_crowd_summary(spatial, radius=2.0, n_ref=20.0)
    assert token.shape == (6,)
    assert token[0] == pytest.approx(3.0 / 20.0)
    assert token[-1] == pytest.approx(1.0)


def test_v40_like_checkpoint_loads_with_resume_and_knn_flags(tmp_path):
    src = SNCPPolicy(
        robot_vpref=1.0, robot_wmax=1.8, pre_mlp=True, meanmax_pool=True,
        action_dist="beta", hh_intent_graph=True, risk_head=True,
    )
    ckpt = tmp_path / "v40.pt"
    torch.save(src.state_dict(), ckpt)
    args = build_parser().parse_args([
        "--resume", str(ckpt),
        "--lagrange_ppo",
        "--human_token_mode", "knn_sorted",
        "--knn_k", "10",
    ])
    env = SimpleNamespace(robot_vpref=1.0, robot_wmax=1.8)
    policy = build_or_load_policy(args, env, torch.device("cpu"))
    assert policy.risk_head is True
    assert policy.hh_intent_graph is True
    assert policy.human_token_mode == "knn_sorted"
    assert policy.knn_k == 10
    rebuilt = build_policy_for_checkpoint(src.state_dict(), robot_vpref=1.0, robot_wmax=1.8)
    missing, unexpected = rebuilt.load_state_dict(src.state_dict(), strict=False)
    assert not missing and not unexpected


def test_vectorized_lagrange_uses_adapted_cost_not_bce_label():
    torch.manual_seed(0)
    N, T, H = 2, 8, 3
    policy = SNCPPolicy(risk_head=True)
    agent = PPOAgent(
        policy, epochs=1, batch_size=4, seq_len=T,
        normalize_returns=False, use_lagrange=True,
        lagrange_lambda_init=0.0, lagrange_cost_limit=0.05, lagrange_lr=1.0,
        lagrange_lambda_max=10.0,
    )
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    hidden = policy.init_hidden(N, H, torch.device("cpu"))
    coll = torch.ones(N)
    adapted = torch.zeros(N)  # scaled all the way to 0 -> dual should not climb like raw 1.0
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
            rewards=torch.zeros(N), values=value.squeeze(-1),
            dones=torch.zeros(N), masks=torch.ones(N),
            coll_labels=coll, clearance_labels=torch.ones(N),
            cost_values=policy.last_cost_value.squeeze(-1).detach(),
            lagrange_costs=adapted,
        )
        hidden = new_h
    buf.finish(last_values=torch.zeros(N), last_dones=torch.zeros(N),
               last_cost_values=torch.zeros(N))
    agent.update_vectorized(buf, torch.device("cpu"))
    # mean adapted cost 0 < limit 0.05, so λ stays 0 (would rise if raw coll=1 were used).
    assert agent.last_mean_cost == pytest.approx(0.0)
    assert agent.lagrange_lambda == pytest.approx(0.0)


def test_apply_v41_flags_default_summary_only_for_knn():
    policy = SNCPPolicy()
    apply_v41_runtime_flags(policy, SimpleNamespace(human_token_mode="all"))
    assert policy.human_token_mode == "all"
    assert policy.crowd_summary is False
    apply_v41_runtime_flags(policy, SimpleNamespace(human_token_mode="knn_sorted", knn_k=8))
    assert policy.crowd_summary is True
    assert policy.knn_k == 8
    apply_v41_runtime_flags(
        policy, SimpleNamespace(human_token_mode="knn_sorted", knn_k=8, no_crowd_summary=True),
    )
    assert policy.crowd_summary is False
