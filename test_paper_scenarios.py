import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.train import SCENARIO_HOLDOUT_CONFIG, make_env, build_parser


def test_collision_threshold_defaults_to_radii_sum():
    env = CrowdSimEnv(num_humans=3, scenario='hard', human_motion_model='orca')
    assert env.collision_threshold == env.robot_radius + env.human_radius  # 0.6


def test_collision_threshold_is_configurable():
    env = CrowdSimEnv(num_humans=3, scenario='hard', human_motion_model='orca',
                      collision_threshold=0.3)
    assert env.collision_threshold == 0.3


def test_paper_standard_layout():
    env = CrowdSimEnv(num_humans=5, scenario='paper_standard', human_motion_model='orca')
    env.reset(seed=0)
    # robot fixed bottom -> top
    assert (env.robot_px, env.robot_py) == (0.0, -4.0)
    assert (env.robot_gx, env.robot_gy) == (0.0, 4.0)
    half = 5.0  # 10x10 arena
    assert np.all(np.abs(env.humans_px) <= half) and np.all(np.abs(env.humans_py) <= half)
    # scattered, NOT all on the radius-4 circle (the antipodal regime)
    radii = np.hypot(env.humans_px, env.humans_py)
    assert np.std(radii) > 0.3, f"humans look circle-placed: radii={radii}"
    # no human starts inside the collision threshold of the robot start
    d_start = np.hypot(env.humans_px - env.robot_px, env.humans_py - env.robot_py)
    assert np.all(d_start >= env.collision_threshold)
    assert env.human_vpref == 1.0  # parity


def test_paper_challenging_scales_arena():
    for n in (10, 15, 20):
        env = CrowdSimEnv(num_humans=n, scenario='paper_challenging', human_motion_model='orca')
        env.reset(seed=1)
        assert (env.robot_px, env.robot_py) == (0.0, -6.0)
        assert (env.robot_gx, env.robot_gy) == (0.0, 6.0)
        half = 7.5  # 15x15 arena
        assert np.all(np.abs(env.humans_px) <= half) and np.all(np.abs(env.humans_py) <= half)
        assert env.humans_px.shape == (n,)


def test_existing_hard_scenario_unchanged():
    # Default preservation: 'hard' still uses radius-4 circle-crossing placement.
    env = CrowdSimEnv(num_humans=8, scenario='hard', human_motion_model='orca',
                      randomize_layout=True)
    env.reset(seed=0)
    radii = np.hypot(env.humans_px, env.humans_py)
    assert np.allclose(radii, 4.0, atol=1e-9), f"hard placement changed: radii={radii}"


def test_paper_scenarios_in_holdout_config():
    assert SCENARIO_HOLDOUT_CONFIG['paper_standard'][1] == 1.0      # parity vpref
    assert SCENARIO_HOLDOUT_CONFIG['paper_challenging'][1] == 1.0


def test_train_parser_has_collision_threshold():
    args = build_parser().parse_args(['--collision_threshold', '0.3'])
    assert args.collision_threshold == 0.3
    assert build_parser().parse_args([]).collision_threshold is None


def test_make_env_builds_paper_scenario_with_threshold():
    env = make_env(num_humans=10, scenario='paper_challenging', seed=0,
                   collision_threshold=0.3)()
    assert env.scenario == 'paper_challenging'
    assert env.collision_threshold == 0.3


def test_train_parser_accepts_paper_scenarios():
    # Regression: argparse choices were hardcoded to the legacy scenario list,
    # rejecting --fixed_scenario paper_challenging even though env/config supported it.
    args = build_parser().parse_args([
        '--fixed_scenario', 'paper_challenging',
        '--holdout_scenarios', 'paper_standard', 'paper_challenging',
    ])
    assert args.fixed_scenario == 'paper_challenging'
    assert args.holdout_scenarios == ['paper_standard', 'paper_challenging']
