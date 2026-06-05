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


def test_pedestrians_react_to_robot_by_default():
    """The env default makes pedestrians avoid the robot (a COOPERATIVE-crowd
    assumption). Training (make_env), the holdout eval, and test_eval all
    construct the env WITHOUT passing this flag, so they inherit the default;
    this test locks it to True for the project's chosen setting.

    NOTE: this DEVIATES from the source paper, which uses CrowdNav's
    invisible-robot ORCA (pedestrians ignore the robot). We chose the cooperative
    setting because the target TurtleBot3 (0.26 m/s) is too slow for the
    invisible-robot setting — see train.py. Results are therefore NOT directly
    comparable to the paper's invisible-robot numbers."""
    env = CrowdSimEnv(num_humans=1, scenario='hard')
    assert env.human_dodge_robot is True


def test_reactive_pedestrians_keep_more_clearance():
    """With reactivity ON (default), a pedestrian on a collision course keeps a
    larger closest-approach distance than with it OFF — proof the avoidance
    force is actually applied, not just a flag toggled."""
    reactive = CrowdSimEnv(num_humans=1, scenario='hard')  # default -> reactive
    nonreactive = CrowdSimEnv(num_humans=1, scenario='hard', human_dodge_robot=False)
    d_reactive = _closest_approach(reactive)
    d_nonreactive = _closest_approach(nonreactive)
    assert d_reactive > d_nonreactive + 0.05, (
        f"reactive clearance {d_reactive:.3f} not greater than non-reactive "
        f"{d_nonreactive:.3f} — pedestrians are not avoiding the robot")
