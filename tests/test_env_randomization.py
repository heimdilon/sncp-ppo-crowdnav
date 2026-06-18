"""Tests for episode-level layout randomization in CrowdSimEnv.

Covers the fix for the overfitting root cause: the robot start/goal were
hard-coded to (0,-4)->(0,4) every reset, so the policy memorised one scene.
With randomize_layout=True the robot and pedestrians are placed on a circle of
radius R at random antipodal positions, and reset(seed) is now deterministic
(uses self.np_random, not the global np.random).
"""
import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv

R = 4.0  # circle radius used by the env's circle scenarios


# --- backward compatibility -------------------------------------------------

def test_fixed_layout_unchanged():
    """randomize_layout=False must reproduce the old fixed geometry exactly."""
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=False)
    env.reset(seed=1)
    assert (env.robot_px, env.robot_py) == (0.0, -4.0)
    assert (env.robot_gx, env.robot_gy) == (0.0, 4.0)
    assert env.robot_theta == np.pi / 2.0


# --- seed determinism (the self.np_random fix) ------------------------------

def test_same_seed_same_layout():
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=True)
    env.reset(seed=123)
    r1 = (env.robot_px, env.robot_py, env.robot_theta)
    h1x, h1y = env.humans_px.copy(), env.humans_py.copy()
    env.reset(seed=123)
    r2 = (env.robot_px, env.robot_py, env.robot_theta)
    assert r1 == r2
    assert np.allclose(h1x, env.humans_px) and np.allclose(h1y, env.humans_py)


# --- variety ----------------------------------------------------------------

def test_different_seed_different_start():
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=True)
    env.reset(seed=1)
    a = (env.robot_px, env.robot_py)
    env.reset(seed=2)
    b = (env.robot_px, env.robot_py)
    assert a != b


def test_robot_heading_randomized():
    """Robot should face a random direction, not always pi/2."""
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=True)
    thetas = []
    for s in range(10):
        env.reset(seed=s)
        thetas.append(round(float(env.robot_theta), 4))
    assert len(set(thetas)) > 1                      # randomized
    assert any(abs(t - np.pi / 2.0) > 0.1 for t in thetas)  # not all north


# --- antipodal-on-circle invariants -----------------------------------------

def test_robot_antipodal_on_circle():
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=True)
    env.reset(seed=7)
    assert np.isclose(np.hypot(env.robot_px, env.robot_py), R)
    assert np.isclose(env.robot_gx, -env.robot_px)
    assert np.isclose(env.robot_gy, -env.robot_py)


def test_humans_antipodal_on_circle():
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=True)
    env.reset(seed=7)
    for i in range(env.num_humans):
        assert np.isclose(np.hypot(env.humans_px[i], env.humans_py[i]), R)
        assert np.isclose(env.humans_gx[i], -env.humans_px[i])
        assert np.isclose(env.humans_gy[i], -env.humans_py[i])
