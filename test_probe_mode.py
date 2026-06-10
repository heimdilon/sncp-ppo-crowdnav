"""Probe-mode training support: fixed curriculum phase + pedestrian motion-model
selection from the CLI. Used by the Faz-0 attribution probes that decompose the
v21 paper-regime failure into its bundled variables (robot speed / ped model /
parity speed / LR) with short fixed-density local runs."""
import numpy as np

from sncp_ppo.train import build_parser, make_env, select_vectorized_phase


def test_cli_accepts_probe_args():
    args = build_parser().parse_args(
        ['--human_motion_model', 'sfm', '--fixed_scenario', 'hard']
    )
    assert args.human_motion_model == 'sfm'
    assert args.fixed_scenario == 'hard'


def test_cli_defaults_preserve_current_regime():
    """No flags -> ORCA pedestrians (v20+ default) and the normal step-budget
    curriculum, so existing notebook/readiness configs are unaffected."""
    args = build_parser().parse_args([])
    assert args.human_motion_model == 'orca'
    assert args.fixed_scenario is None


def test_make_env_threads_motion_model():
    env = make_env(3, 'hard', seed=0, human_motion_model='sfm')()
    assert env.human_motion_model == 'sfm'
    env_default = make_env(3, 'hard', seed=0)()
    assert env_default.human_motion_model == 'orca'


def test_fixed_scenario_pins_phase_and_disables_replay():
    """fixed_scenario bypasses the 10/25/50/75% step-budget curriculum: every
    update window uses the same (scenario, num_humans, canonical speed) phase
    and never counts as a replay update."""
    for steps in [0, 100_000, 400_000, 499_999]:
        (scenario, n_humans, vpref), is_replay = select_vectorized_phase(
            steps, 500_000, 5, replay_ratio=0.5, fixed_scenario='hard'
        )
        assert scenario == 'hard'
        assert n_humans == 5
        assert np.isclose(vpref, 0.26)
        assert is_replay is False


def test_fixed_scenario_none_keeps_curriculum():
    (scenario, n_humans, _), _ = select_vectorized_phase(
        0, 500_000, 5, replay_ratio=0.0, fixed_scenario=None
    )
    assert scenario == 'easy'
    assert n_humans == 1
