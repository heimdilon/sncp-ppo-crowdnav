"""Tests for pedestrian velocity in the observation (Yol B / v8).

spatial_edges gains 2 extra dims per pedestrian: the RELATIVE velocity
(pedestrian - robot) rotated into the robot's local frame, appended to the
existing local position. So each pedestrian row is
[dx_local, dy_local, rel_vx_local, rel_vy_local].
"""
import numpy as np
import torch
from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.models import SNCPPolicy


# --- environment: observation shape & content -------------------------------

def test_spatial_edges_has_4_dims():
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    obs, _ = env.reset(seed=1)
    assert obs['spatial_edges'].shape == (5, 4)


def test_position_part_unchanged():
    """First two columns are still the local position (backward-compatible)."""
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=False)
    obs, _ = env.reset(seed=1)
    pos = obs['spatial_edges'][:, :2]
    # On the fixed circle layout the pedestrians sit at radius ~4 from origin,
    # robot at (0,-4); positions must be finite and non-trivial.
    assert pos.shape == (5, 2)
    assert np.all(np.isfinite(pos))


def test_initial_velocity_is_zero():
    """At reset both robot and pedestrians are stationary -> rel velocity ~ 0."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    obs, _ = env.reset(seed=1)
    vel = obs['spatial_edges'][:, 2:]
    assert np.allclose(vel, 0.0, atol=1e-6)


def test_velocity_nonzero_after_motion():
    """Once the robot moves and SFM advances pedestrians, the relative
    velocity columns must become non-zero."""
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=False)
    env.reset(seed=1)
    obs, *_ = env.step(np.array([0.26, 0.0], dtype=np.float32))  # full forward
    vel = obs['spatial_edges'][:, 2:]
    assert np.any(np.abs(vel) > 1e-3)


# --- model: accepts the wider spatial input ---------------------------------

def test_model_accepts_4dim_spatial():
    policy = SNCPPolicy(robot_vpref=0.26, robot_wmax=1.8)
    h = policy.init_hidden(batch_size=2, num_humans=5, device=torch.device('cpu'))
    obs = {
        'robot_node': torch.randn(2, 7),
        'spatial_edges': torch.randn(2, 5, 4),   # 4-dim per pedestrian
        'temporal_edges': torch.randn(2, 2),
    }
    mu, std, value, _ = policy(obs, h)
    assert mu.shape == (2, 2)
    assert std.shape == (2, 2)
    assert value.shape == (2, 1)
