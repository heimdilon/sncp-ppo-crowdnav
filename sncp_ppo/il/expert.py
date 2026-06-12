"""ORCA robot-expert for behavior-cloning demos.

The robot runs ORCA against the pedestrians (as neighbors) with FULL avoidance
responsibility — pedestrians are invisible to the robot in this env, so it must
own the whole avoidance burden. ORCA returns a holonomic velocity; the robot is
differential-drive, so velocity_to_action projects it onto [v, w].

This mirrors the controller in _oracle_feasibility.py (oracle_action), generalized
from "head to the goal" to "head along the ORCA-chosen velocity".
"""
import numpy as np

from crowd_sim.orca import orca_new_velocity

_EPS = 1e-9


def _wrap(angle):
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def velocity_to_action(vx, vy, robot_theta, vpref, wmax, time_step):
    """Project a holonomic velocity (vx, vy) onto the env's differential [v, w].

    w turns the heading toward the velocity's direction within one step (clamped
    to wmax); v is the velocity's magnitude (capped at vpref) gated by heading
    alignment so the robot does not drive sideways/backward — when the desired
    direction is perpendicular or behind, forward speed collapses to ~0 and the
    robot turns in place first.
    """
    speed = float(np.hypot(vx, vy))
    if speed < _EPS:
        return 0.0, 0.0
    desired_heading = float(np.arctan2(vy, vx))
    err = _wrap(desired_heading - robot_theta)
    w = float(np.clip(err / time_step, -wmax, wmax))
    align = max(0.0, float(np.cos(err)))  # 1 aligned, 0 perpendicular, 0 behind
    v = float(np.clip(speed, 0.0, vpref)) * align
    return v, w


def expert_pref_velocity(env):
    """Preferred velocity: straight at the goal, at the robot's preferred speed."""
    to_goal = np.array([env.robot_gx - env.robot_px, env.robot_gy - env.robot_py])
    dist = float(np.hypot(to_goal[0], to_goal[1]))
    if dist < _EPS:
        return np.zeros(2)
    return to_goal / dist * env.robot_vpref


def expert_action(env, responsibility=1.0, time_horizon=3.0):
    """The action the ORCA expert would take in env's current state.

    Robot avoids the pedestrians (neighbors) with full responsibility; the
    resulting collision-free velocity is projected onto [v, w].
    """
    pos = np.array([env.robot_px, env.robot_py])
    vel = np.array([env.robot_vx, env.robot_vy])
    pref = expert_pref_velocity(env)
    neighbors = [
        (
            np.array([env.humans_px[i], env.humans_py[i]]),
            np.array([env.humans_vx[i], env.humans_vy[i]]),
            float(env.human_radius),
        )
        for i in range(env.num_humans)
    ]
    new_vel = orca_new_velocity(
        pos, vel, float(env.robot_radius), pref, neighbors,
        max_speed=float(env.robot_vpref), time_horizon=time_horizon,
        time_step=float(env.time_step), responsibility=responsibility,
    )
    return velocity_to_action(
        new_vel[0], new_vel[1], float(env.robot_theta),
        float(env.robot_vpref), float(env.robot_wmax), float(env.time_step),
    )
