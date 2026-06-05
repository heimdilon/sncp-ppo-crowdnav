from crowd_sim.crowd_env import CrowdSimEnv


def test_pedestrian_speed_never_exceeds_robot():
    """v15 parity: at the hardware robot speed (0.26 m/s), no scenario sets
    pedestrians faster than the robot, so a slow robot can feasibly avoid a
    non-reactive crowd."""
    for scenario in ['easy', 'easy_plus', 'medium', 'hard', 'extreme', 'circle']:
        env = CrowdSimEnv(num_humans=3, scenario=scenario)
        env.reset(seed=0)
        assert env.human_vpref <= env.robot_vpref + 1e-9, (
            f"{scenario}: human_vpref {env.human_vpref} > robot {env.robot_vpref}")
