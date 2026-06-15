import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv


def test_collision_threshold_defaults_to_radii_sum():
    env = CrowdSimEnv(num_humans=3, scenario='hard', human_motion_model='orca')
    assert env.collision_threshold == env.robot_radius + env.human_radius  # 0.6


def test_collision_threshold_is_configurable():
    env = CrowdSimEnv(num_humans=3, scenario='hard', human_motion_model='orca',
                      collision_threshold=0.3)
    assert env.collision_threshold == 0.3
