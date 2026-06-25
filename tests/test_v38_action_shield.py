import numpy as np
import pytest

from sncp_ppo.action_shield import ActionShieldConfig, min_predicted_clearance, shield_action
from scripts.run_v38_shield_probe import analyze_rows


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
        self.humans_px = np.array([], dtype=float)
        self.humans_py = np.array([], dtype=float)
        self.humans_vx = np.array([], dtype=float)
        self.humans_vy = np.array([], dtype=float)


def test_shield_leaves_safe_action_unchanged():
    env = DummyEnv()
    env.humans_px = np.array([0.0])
    env.humans_py = np.array([4.0])
    env.humans_vx = np.array([0.0])
    env.humans_vy = np.array([0.0])
    action = np.array([0.8, 0.1], dtype=np.float32)

    guarded = shield_action(env, action, ActionShieldConfig(horizon_steps=4, safety_margin=0.1))

    np.testing.assert_allclose(guarded, action, atol=1e-6)


def test_shield_replaces_imminent_collision_action_with_safer_candidate():
    env = DummyEnv()
    env.humans_px = np.array([0.55])
    env.humans_py = np.array([0.0])
    env.humans_vx = np.array([0.0])
    env.humans_vy = np.array([0.0])
    action = np.array([1.0, 0.0], dtype=np.float32)
    cfg = ActionShieldConfig(horizon_steps=4, safety_margin=0.1)

    guarded = shield_action(env, action, cfg)

    assert not np.allclose(guarded, action)
    assert 0.0 <= guarded[0] <= env.robot_vpref
    assert -env.robot_wmax <= guarded[1] <= env.robot_wmax
    assert min_predicted_clearance(env, guarded, cfg) > min_predicted_clearance(env, action, cfg)


def test_shield_clips_out_of_bounds_policy_action():
    env = DummyEnv()
    action = np.array([2.0, 5.0], dtype=np.float32)

    guarded = shield_action(env, action, ActionShieldConfig())

    assert guarded[0] == pytest.approx(env.robot_vpref)
    assert guarded[1] == pytest.approx(env.robot_wmax)


def test_v38_probe_go_requires_collision_drop_without_success_timeout_regression():
    rows = [
        {"arm": "c0", "num_humans": 15, "success_rate": 0.90, "collision_rate": 0.10, "timeout_rate": 0.00},
        {"arm": "c1", "num_humans": 15, "success_rate": 0.89, "collision_rate": 0.05, "timeout_rate": 0.01},
        {"arm": "c0", "num_humans": 20, "success_rate": 0.80, "collision_rate": 0.20, "timeout_rate": 0.01},
        {"arm": "c1", "num_humans": 20, "success_rate": 0.79, "collision_rate": 0.16, "timeout_rate": 0.02},
    ]

    verdict = analyze_rows(rows)

    assert verdict["verdict"] == "GO"
    assert verdict["high_n_collision_delta"] == pytest.approx(-0.045)


def test_v38_probe_blocks_success_regression_even_if_collision_drops():
    rows = [
        {"arm": "c0", "num_humans": 15, "success_rate": 0.90, "collision_rate": 0.10, "timeout_rate": 0.00},
        {"arm": "c1", "num_humans": 15, "success_rate": 0.82, "collision_rate": 0.02, "timeout_rate": 0.00},
        {"arm": "c0", "num_humans": 20, "success_rate": 0.80, "collision_rate": 0.20, "timeout_rate": 0.00},
        {"arm": "c1", "num_humans": 20, "success_rate": 0.78, "collision_rate": 0.10, "timeout_rate": 0.00},
    ]

    assert analyze_rows(rows)["verdict"] == "NO-GO"
