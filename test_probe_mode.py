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


def test_bootstrap_easy_steps_warms_up_before_fixed_phase():
    """Probe run 1 lesson: cold-starting at fixed N=5 never bootstraps
    goal-reaching (the curriculum's easy/1 phase IS the bootstrap; even the
    known-good v18-regime control stayed at 0% for 300k). bootstrap_easy_steps
    prepends an easy/1 warmup before the pinned probe phase."""
    (scenario, n_humans, vpref), is_replay = select_vectorized_phase(
        0, 300_000, 5, replay_ratio=0.5,
        fixed_scenario='hard', bootstrap_easy_steps=150_000,
    )
    assert (scenario, n_humans) == ('easy', 1)
    assert np.isclose(vpref, 0.13)
    assert is_replay is False

    (scenario, n_humans, _), _ = select_vectorized_phase(
        150_000, 300_000, 5, replay_ratio=0.5,
        fixed_scenario='hard', bootstrap_easy_steps=150_000,
    )
    assert (scenario, n_humans) == ('hard', 5)


def test_cli_accepts_bootstrap_easy_steps():
    args = build_parser().parse_args(['--bootstrap_easy_steps', '150000'])
    assert args.bootstrap_easy_steps == 150_000
    assert build_parser().parse_args([]).bootstrap_easy_steps == 0


def test_vectorized_rows_log_episode_outcomes(tmp_path):
    """Probe run 1 lesson #2: vectorized CSV rows logged ONLY diagnostics —
    reward/success/collision/timeout stayed empty, leaving holdout (every 20
    updates) as the sole outcome signal. Update rows must now aggregate the
    episodes that FINISH inside the rollout window."""
    import argparse
    import csv

    import torch

    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.models import SNCPPolicy
    from sncp_ppo.ppo import PPOAgent
    from sncp_ppo.train import _train_vectorized
    from test_vec_curriculum import _write_vector_log_header

    args = argparse.Namespace(
        num_envs=2, horizon=16, total_steps=96, eval_freq_updates=0,
        episodes=1, num_humans=2, seed=11,
        holdout_scenarios=['easy'], holdout_episodes=1,
        best_warmup_evals=0, best_min_success_threshold=0.0,
        save_path=str(tmp_path / 'probe_smoke.pt'),
        lr=1e-4, target_kl=0.01, curriculum_replay_ratio=0.0,
        max_time=2.0,  # 8-step episodes -> several finish per 16-step window
        fixed_scenario='hard',
    )
    device = torch.device('cpu')
    env = CrowdSimEnv(num_humans=args.num_humans, scenario='hard', max_time=args.max_time)
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)
    agent = PPOAgent(policy=policy, lr=args.lr, target_kl=args.target_kl,
                     epochs=1, batch_size=2, seq_len=4)

    log_path = tmp_path / 'probe_log.csv'
    with log_path.open('w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        _write_vector_log_header(csv_writer)
        _train_vectorized(args, env, policy, agent, device, str(log_path), csv_writer, csv_file)

    with log_path.open(newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows, 'no update rows logged'
    outcome_rows = [r for r in rows if r['success'] not in ('', 'nan')]
    assert outcome_rows, 'no update row aggregated finished-episode outcomes'
    for row in outcome_rows:
        success = float(row['success'])
        collision = float(row['collision'])
        timeout = float(row['timeout'])
        assert 0.0 <= success <= 1.0
        assert 0.0 <= collision <= 1.0
        assert 0.0 <= timeout <= 1.0
        assert abs(success + collision + timeout - 1.0) < 1e-6
        float(row['reward'])  # mean finished-episode return parses as float
