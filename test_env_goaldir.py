import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv


def test_spatial_edges_has_6_dims():
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    obs, _ = env.reset(seed=1)
    assert obs['spatial_edges'].shape == (5, 6)


def test_goal_dir_is_unit_vector():
    """Columns 4-5 (goal direction) must be a unit vector per pedestrian."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    obs, _ = env.reset(seed=1)
    gdir = obs['spatial_edges'][:, 4:6]
    norms = np.hypot(gdir[:, 0], gdir[:, 1])
    assert np.allclose(norms, 1.0, atol=1e-4), f"goal-dir norms not unit: {norms}"


def test_goal_dir_rotation_correct():
    """Robot facing east (theta=0), a pedestrian whose goal is due north must
    have local goal-dir (0, 1); robot facing north must give (1, 0)."""
    env = CrowdSimEnv(num_humans=1, scenario='hard')
    env.reset(seed=1)
    # Force a known geometry: pedestrian at origin, goal due north.
    env.humans_px[0], env.humans_py[0] = 0.0, 0.0
    env.humans_gx[0], env.humans_gy[0] = 0.0, 5.0
    env.robot_px, env.robot_py = 2.0, 0.0
    env.robot_theta = 0.0  # facing east
    obs = env._get_obs()
    gx, gy = obs['spatial_edges'][0, 4], obs['spatial_edges'][0, 5]
    assert np.allclose([gx, gy], [0.0, 1.0], atol=1e-4), f"east-facing: got ({gx},{gy})"
    env.robot_theta = np.pi / 2  # facing north
    obs = env._get_obs()
    gx, gy = obs['spatial_edges'][0, 4], obs['spatial_edges'][0, 5]
    assert np.allclose([gx, gy], [1.0, 0.0], atol=1e-4), f"north-facing: got ({gx},{gy})"


def test_first_four_cols_unchanged():
    """The position + relative-velocity columns (0-3) must keep the v8 layout."""
    env = CrowdSimEnv(num_humans=3, scenario='medium')
    env.reset(seed=2)
    obs = env._get_obs()
    se = obs['spatial_edges']
    assert se.shape == (3, 6)
    # Recompute expected pos cols directly from state, robot-local frame.
    cos_t, sin_t = np.cos(env.robot_theta), np.sin(env.robot_theta)
    for i in range(3):
        dx = env.humans_px[i] - env.robot_px
        dy = env.humans_py[i] - env.robot_py
        exp_x = dx * cos_t + dy * sin_t
        exp_y = -dx * sin_t + dy * cos_t
        assert np.allclose(se[i, 0], exp_x, atol=1e-4)
        assert np.allclose(se[i, 1], exp_y, atol=1e-4)
