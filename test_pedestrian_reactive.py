import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv


def _closest_approach(env, steps=12):
    """Deterministic head-on scenario: a stationary robot at the origin and a
    single pedestrian walking straight across it. Returns the minimum
    robot-pedestrian distance over the rollout — the lower, the more the
    pedestrian plowed into the robot."""
    env.reset(seed=0)
    env.robot_px, env.robot_py = 0.0, 0.0
    env.robot_theta = 0.0
    env.humans_px[0], env.humans_py[0] = 1.6, 0.15
    env.humans_vx[0], env.humans_vy[0] = 0.0, 0.0
    # Goal on the far side, so the pedestrian's straight path crosses the robot.
    env.humans_gx[0], env.humans_gy[0] = -3.0, 0.15
    min_d = np.inf
    for _ in range(steps):
        env.step(np.array([0.0, 0.0], dtype=np.float32))  # robot holds position
        d = np.hypot(env.humans_px[0] - env.robot_px, env.humans_py[0] - env.robot_py)
        min_d = min(min_d, d)
    return min_d


def test_pedestrians_ignore_robot_by_default():
    """v15: the default is NON-reactive ('invisible robot', the paper's CrowdNav
    regime) so the robot must actively avoid. Training (make_env), the holdout
    eval, and test_eval all inherit this default. Reactivity stays available via
    the flag for the cooperative-crowd experiments (v14)."""
    assert CrowdSimEnv(num_humans=1, scenario='hard').human_dodge_robot is False
    assert CrowdSimEnv(num_humans=1, scenario='hard', human_dodge_robot=True).human_dodge_robot is True


def test_reactive_pedestrians_keep_more_clearance():
    """With reactivity ON (explicit flag), a pedestrian on a collision course
    keeps a larger closest-approach distance than with it OFF — proof the
    avoidance force is actually applied, not just a flag toggled."""
    reactive = CrowdSimEnv(num_humans=1, scenario='hard', human_dodge_robot=True)
    nonreactive = CrowdSimEnv(num_humans=1, scenario='hard', human_dodge_robot=False)
    d_reactive = _closest_approach(reactive)
    d_nonreactive = _closest_approach(nonreactive)
    assert d_reactive > d_nonreactive + 0.05, (
        f"reactive clearance {d_reactive:.3f} not greater than non-reactive "
        f"{d_nonreactive:.3f} — pedestrians are not avoiding the robot")
