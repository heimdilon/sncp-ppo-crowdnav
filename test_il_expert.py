"""Phase 0: ORCA robot-expert that the BC pretrain will clone.

velocity_to_action converts a holonomic ORCA velocity to the env's differential
[v, w]; expert_action wires the env state into ORCA (pedestrians as neighbors,
full responsibility) and returns the action the expert would take."""
import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.il.expert import velocity_to_action, expert_action


def test_velocity_to_action_aligned_drives_full_speed_straight():
    v, w = velocity_to_action(1.0, 0.0, robot_theta=0.0, vpref=1.0, wmax=1.8, time_step=0.25)
    assert np.isclose(v, 1.0, atol=1e-6)
    assert np.isclose(w, 0.0, atol=1e-6)


def test_velocity_to_action_caps_speed_at_vpref():
    v, _ = velocity_to_action(5.0, 0.0, robot_theta=0.0, vpref=1.0, wmax=1.8, time_step=0.25)
    assert v <= 1.0 + 1e-6


def test_velocity_to_action_perpendicular_turns_left_and_slows():
    # desired heading +90deg vs robot facing 0 -> turn left (w>0), don't drive sideways
    v, w = velocity_to_action(0.0, 1.0, robot_theta=0.0, vpref=1.0, wmax=1.8, time_step=0.25)
    assert w > 0.5
    assert v < 0.2


def test_velocity_to_action_right_turn_has_negative_w():
    _, w = velocity_to_action(0.0, -1.0, robot_theta=0.0, vpref=1.0, wmax=1.8, time_step=0.25)
    assert w < -0.5


def test_velocity_to_action_respects_wmax():
    _, w = velocity_to_action(-1.0, 0.0, robot_theta=0.0, vpref=1.0, wmax=1.8, time_step=0.25)
    assert abs(w) <= 1.8 + 1e-6


def test_expert_action_in_bounds():
    env = CrowdSimEnv(num_humans=5, scenario='hard', randomize_layout=False, robot_vpref=1.0)
    env.reset(seed=0)
    v, w = expert_action(env)
    assert 0.0 <= v <= env.robot_vpref + 1e-6
    assert -env.robot_wmax - 1e-6 <= w <= env.robot_wmax + 1e-6


def test_expert_goes_nearly_straight_with_clear_path():
    """Robot at (0,-4) facing north, goal (0,4), pedestrian parked far off-path:
    the expert should head straight for the goal (low |w|, near-full speed)."""
    env = CrowdSimEnv(num_humans=1, scenario='hard', randomize_layout=False, robot_vpref=1.0)
    env.reset(seed=0)
    env.humans_px[0], env.humans_py[0] = 5.0, 5.0
    env.humans_vx[0], env.humans_vy[0] = 0.0, 0.0
    v, w = expert_action(env)
    assert v > 0.5
    assert abs(w) < 0.2


def test_expert_avoids_pedestrian_in_path():
    """Pedestrian directly ahead on the goal line -> the expert must deviate
    (nonzero turn or reduced forward speed), not drive straight through."""
    env = CrowdSimEnv(num_humans=1, scenario='hard', randomize_layout=False, robot_vpref=1.0)
    env.reset(seed=0)
    env.humans_px[0], env.humans_py[0] = 0.0, -2.0
    env.humans_vx[0], env.humans_vy[0] = 0.0, 0.0
    v, w = expert_action(env)
    assert abs(w) > 0.05 or v < 0.95
