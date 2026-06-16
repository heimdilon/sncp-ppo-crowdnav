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
    # robot fixed bottom -> top, 10 m crossing (v25: was +/-4)
    assert (env.robot_px, env.robot_py) == (0.0, -5.0)
    assert (env.robot_gx, env.robot_gy) == (0.0, 5.0)
    assert env.sense_range == 4.0
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
        assert (env.robot_px, env.robot_py) == (0.0, -5.0)   # 10 m crossing (v25: was +/-6)
        assert (env.robot_gx, env.robot_gy) == (0.0, 5.0)
        assert env.sense_range == 6.0
        half = 7.5  # 15x15 arena
        assert np.all(np.abs(env.humans_px) <= half) and np.all(np.abs(env.humans_py) <= half)
        assert env.humans_px.shape == (n,)


def test_paper_regime_easy_keeps_circle_geometry():
    # paper_regime forces the BUDGET but must NOT impose the paper crossing on 'easy'.
    env = CrowdSimEnv(num_humans=5, scenario='easy', human_motion_model='orca',
                      paper_regime=True)
    env.reset(seed=0)
    radii = np.hypot(env.humans_px, env.humans_py)
    assert np.allclose(radii, 4.0, atol=1e-9), f"easy geometry changed: radii={radii}"


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


# --- v25: paper-faithful time budget / comfort / d_col resolution ---

def test_paper_scenario_resolves_paper_regime_params():
    # Constructed WITH a paper scenario -> paper budget/comfort/d_col, no flags needed.
    env = CrowdSimEnv(num_humans=10, scenario='paper_challenging', human_motion_model='orca')
    assert env.max_time == 12.5
    assert env.comfort_coeff == 2.0
    assert env.collision_threshold == 0.3


def test_paper_regime_flag_forces_budget_on_nonpaper_scenario():
    # The easy-bootstrap case: scenario is NOT paper, but paper_regime forces the budget.
    env = CrowdSimEnv(num_humans=3, scenario='easy', human_motion_model='orca',
                      paper_regime=True)
    assert env.max_time == 12.5
    assert env.comfort_coeff == 2.0
    assert env.collision_threshold == 0.3


def test_nonpaper_env_regime_unchanged():
    env = CrowdSimEnv(num_humans=5, scenario='hard', human_motion_model='orca')
    assert env.max_time == 50.0
    assert env.comfort_coeff == 6.0
    assert env.collision_threshold == env.robot_radius + env.human_radius  # 0.6


def test_explicit_regime_args_override_paper():
    env = CrowdSimEnv(num_humans=10, scenario='paper_challenging', human_motion_model='orca',
                      max_time=50.0, comfort_coeff=6.0, collision_threshold=0.6)
    assert (env.max_time, env.comfort_coeff, env.collision_threshold) == (50.0, 6.0, 0.6)


def test_make_env_paper_regime_forces_budget():
    env = make_env(num_humans=3, scenario='easy', seed=0, paper_regime=True)()
    assert env.max_time == 12.5
    assert env.comfort_coeff == 2.0
    assert env.collision_threshold == 0.3


def test_make_env_nonpaper_defaults_unchanged():
    env = make_env(num_humans=5, scenario='hard', seed=0)()
    assert env.max_time == 50.0
    assert env.comfort_coeff == 6.0
    assert env.collision_threshold == env.robot_radius + env.human_radius


def test_train_parser_budget_defaults_are_none():
    args = build_parser().parse_args([])
    assert args.max_time is None
    assert args.comfort_coeff is None


def test_eval_max_time_defaults_to_none():
    # The eval entry points default max_time to None so a paper-scenario eval resolves
    # to the paper's 12.5s in the env (test_paper_scenario_resolves_paper_regime_params
    # covers None -> 12.5; evaluate_density passes max_time straight to CrowdSimEnv).
    import inspect
    from sncp_ppo.eval_report import evaluate_density
    from sncp_ppo.post_run_pipeline import run_v16_post_eval
    from run_post_eval import build_parser as eval_build_parser
    assert inspect.signature(evaluate_density).parameters['max_time'].default is None
    assert inspect.signature(run_v16_post_eval).parameters['max_time'].default is None
    assert eval_build_parser().parse_args(['--version', '25']).max_time is None
