import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.models import SNCPPolicy


def test_defaults_are_turtlebot_regime():
    """Bare env preserves the real-robot defaults (0.26 m/s, scenario speeds)."""
    env = CrowdSimEnv(num_humans=3, scenario='easy')
    assert env.robot_vpref == 0.26
    assert env.human_vpref_override is None
    env.reset(seed=0)
    assert np.isclose(env.human_vpref, 0.13)  # easy scenario default


def test_robot_vpref_param_scales_action_space_and_policy():
    """The paper regime sets robot_vpref=1.0; the action space and the policy's
    velocity head must scale to [0, 1.0]."""
    env = CrowdSimEnv(num_humans=3, scenario='hard', robot_vpref=1.0)
    assert env.robot_vpref == 1.0
    assert np.isclose(env.action_space.high[0], 1.0)
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax)
    assert np.isclose(policy.robot_vpref, 1.0)


def test_goal_noise_breaks_antipodal_funneling():
    """With human_goal_noise>0, circle-crossing goals are perturbed off the exact
    antipodal point, so pedestrian paths no longer all funnel through (0,0) —
    matching the paper's spread human trajectories. Default (0) stays antipodal."""
    env = CrowdSimEnv(num_humans=10, scenario='circle', human_goal_noise=2.0)
    env.reset(seed=5)
    offset = np.hypot(env.humans_gx + env.humans_px, env.humans_gy + env.humans_py)
    assert offset.mean() > 0.3, f"goals still ~antipodal: mean offset {offset.mean():.2f}"

    env0 = CrowdSimEnv(num_humans=10, scenario='circle')  # default goal noise 0.0
    env0.reset(seed=5)
    off0 = np.hypot(env0.humans_gx + env0.humans_px, env0.humans_gy + env0.humans_py)
    assert np.allclose(off0, 0.0), "default should keep exact antipodal goals"


def test_human_vpref_override_forces_parity_speed():
    """human_vpref_override forces a flat pedestrian speed across scenarios AND
    lands in humans_vpref (what the motion model reads), so it is genuinely
    applied — bypassing the latent `env.human_vpref=...`-after-reset gotcha."""
    env = CrowdSimEnv(num_humans=5, scenario='easy', human_vpref_override=1.0)
    env.reset(seed=1)
    assert np.allclose(env.humans_vpref, 1.0), "override did not reach humans_vpref"
    # Pedestrians actually move near 1.0 m/s (sparse so ORCA mostly unconstrained).
    p0 = np.array([env.humans_px[0], env.humans_py[0]])
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    moved = np.hypot(env.humans_px[0] - p0[0], env.humans_py[0] - p0[1])
    assert moved > 1.0 * env.time_step * 0.5, f"pedestrian too slow: {moved/env.time_step:.2f} m/s"
