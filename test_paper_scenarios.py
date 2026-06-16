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
    # robot fixed bottom -> top, 8 m crossing (v26: paper S5.3.2 "(0,-4)->(0,4)")
    assert (env.robot_px, env.robot_py) == (0.0, -4.0)
    assert (env.robot_gx, env.robot_gy) == (0.0, 4.0)
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
        assert (env.robot_px, env.robot_py) == (0.0, -4.0)   # 8 m crossing (v26: paper S5.3.2)
        assert (env.robot_gx, env.robot_gy) == (0.0, 4.0)
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


# --- v25/v26: paper-faithful time budget / comfort / d_col resolution ---
# v26: budget is per-scenario (standard 12.5s, challenging 50.0s from Table 3 nav-times).

def test_paper_scenario_resolves_paper_regime_params():
    # Constructed WITH a paper scenario -> paper budget/comfort/d_col, no flags needed.
    chal = CrowdSimEnv(num_humans=10, scenario='paper_challenging', human_motion_model='orca')
    assert chal.max_time == 50.0   # v26: challenging budget (Table 3 nav-time 15.92s)
    assert chal.comfort_coeff == 2.0
    assert chal.collision_threshold == 0.3
    std = CrowdSimEnv(num_humans=5, scenario='paper_standard', human_motion_model='orca')
    assert std.max_time == 12.5     # standard budget stays 12.5 (Table 1)


def test_paper_regime_flag_forces_budget_on_nonpaper_scenario():
    # The easy-bootstrap / holdout-eval_env case: scenario is NOT paper, but paper_regime
    # forces the binding (challenging) paper budget so the whole run is consistent.
    env = CrowdSimEnv(num_humans=3, scenario='easy', human_motion_model='orca',
                      paper_regime=True)
    assert env.max_time == 50.0
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
    assert env.max_time == 50.0
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


# --- v26: comfort I_sp = paper Eq 7 normalized weighted average (gradient preserved) ---

def test_isp_normalized_gradient_not_clipped():
    # Eq 7: I_sp is a normalized weighted average -> it must vary smoothly with distance.
    # The old code summed the (un-normalized) numerator and clipped to 1.0, which pegged
    # the penalty flat in close range and destroyed the back-off gradient.
    env = CrowdSimEnv(num_humans=1, scenario='paper_challenging', human_motion_model='orca')
    env.reset(seed=0)
    env.humans_px[:] = 0.0
    env.humans_py[:] = 0.0
    env.humans_theta[:] = 0.0  # human faces +x; robot approaches from the front (a_front=1.5)
    vals = []
    for d in (0.4, 0.7, 1.0, 1.3):
        env.robot_px = float(d)
        env.robot_py = 0.0
        vals.append(env._compute_social_pressure())
    # strictly decreasing with distance = gradient preserved
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), vals
    # the close value must NOT be the saturated 1.0 the old clip produced
    assert vals[0] < 0.9, vals
    assert 0.0 <= min(vals) and max(vals) <= 1.0


def test_isp_normalized_not_clipped_in_close_crowd():
    # 3 humans close to the robot (0.4 m) but spread apart -> the old un-normalized sum
    # (~2.3) would clip to a flat 1.0; the paper's weighted average stays a true mean < 1.
    env = CrowdSimEnv(num_humans=3, scenario='paper_challenging', human_motion_model='orca')
    env.reset(seed=0)
    env.robot_px = 0.0
    env.robot_py = 0.0
    ang = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    env.humans_px[:] = 0.4 * np.cos(ang)
    env.humans_py[:] = 0.4 * np.sin(ang)
    env.humans_theta[:] = ang + np.pi  # face the robot
    isp = env._compute_social_pressure()
    assert 0.0 < isp < 0.9, isp


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
