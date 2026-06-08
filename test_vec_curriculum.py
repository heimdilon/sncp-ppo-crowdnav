import argparse
import csv
import os
import subprocess
import sys

import torch

from sncp_ppo.train import _train_vectorized, evaluate_holdout, step_to_phase, SCENARIO_HOLDOUT_CONFIG


VECTOR_LOG_HEADER = [
    'episode', 'scenario', 'num_humans', 'human_vpref', 'is_replay_update',
    'steps', 'reward', 'success', 'collision', 'timeout', 'comfort',
    'entropy', 'approx_kl', 'std_linear', 'std_angular', 'return_rms_std',
    'is_best_checkpoint', 'best_reason',
    'holdout_easy_success', 'holdout_easy_collision', 'holdout_easy_timeout',
    'holdout_easy_reward', 'holdout_easy_avg_steps', 'holdout_easy_avg_I_sp',
    'holdout_easy_min_d_min',
    'holdout_hard_success', 'holdout_hard_collision',
    'holdout_hard_timeout', 'holdout_hard_reward', 'holdout_hard_avg_steps',
    'holdout_hard_avg_I_sp', 'holdout_hard_min_d_min',
]


def _write_vector_log_header(csv_writer):
    csv_writer.writerow(VECTOR_LOG_HEADER)


def _assert_finite_update_diagnostics(row):
    for key in ['entropy', 'approx_kl', 'std_linear', 'std_angular', 'return_rms_std']:
        assert row[key] not in ('', None), f"{key} missing from CSV row"
        value = float(row[key])
        assert value == value, f"{key} is NaN"
    assert float(row['std_linear']) > 0.0
    assert float(row['std_angular']) > 0.0
    assert float(row['return_rms_std']) > 0.0


def test_step_to_phase_boundaries():
    total = 1000
    final = 10  # v15 final density

    assert step_to_phase(0, total, final) == ('easy', 1, 0.13)
    assert step_to_phase(99, total, final) == ('easy', 1, 0.13)
    assert step_to_phase(100, total, final) == ('easy', 1, 0.13)
    assert step_to_phase(101, total, final) == ('easy_plus', 3, 0.18)
    assert step_to_phase(250, total, final) == ('easy_plus', 3, 0.18)
    assert step_to_phase(251, total, final) == ('medium', 5, 0.22)
    assert step_to_phase(500, total, final) == ('medium', 5, 0.22)
    assert step_to_phase(501, total, final) == ('hard', 8, 0.24)
    assert step_to_phase(750, total, final) == ('hard', 8, 0.24)
    assert step_to_phase(751, total, final) == ('circle', final, 0.26)
    assert step_to_phase(1000, total, final) == ('circle', final, 0.26)
    assert step_to_phase(99999, total, final) == ('circle', final, 0.26)


def test_vectorized_cli_args_are_listed_in_help():
    result = subprocess.run(
        [sys.executable, '-m', 'sncp_ppo.train', '--help'],
        check=True,
        capture_output=True,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        text=True,
    )

    assert '--total_steps' in result.stdout
    assert '--eval_freq_updates' in result.stdout


def test_vectorized_runs_with_curriculum_holdout_and_saves(tmp_path):
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.models import SNCPPolicy
    from sncp_ppo.ppo import PPOAgent

    save_path = tmp_path / 'vc_smoke.pt'
    log_path = tmp_path / 'vc_log.csv'
    args = argparse.Namespace(
        num_envs=2,
        horizon=8,
        total_steps=64,
        eval_freq_updates=4,
        episodes=1,
        num_humans=5,
        seed=42,
        holdout_scenarios=['easy', 'hard'],
        holdout_episodes=1,
        best_warmup_evals=0,
        best_min_success_threshold=0.0,
        save_path=str(save_path),
        lr=1e-4,
        target_kl=0.01,
        curriculum_replay_ratio=0.0,
    )
    device = torch.device('cpu')
    env = CrowdSimEnv(num_humans=args.num_humans, scenario='circle')
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)
    agent = PPOAgent(
        policy=policy,
        lr=args.lr,
        target_kl=args.target_kl,
        epochs=1,
        batch_size=2,
        seq_len=4,
    )

    with log_path.open('w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        _write_vector_log_header(csv_writer)
        _train_vectorized(args, env, policy, agent, device, str(log_path), csv_writer, csv_file)

    with log_path.open(newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))

    num_humans_seen = {int(row['num_humans']) for row in rows}
    assert 1 in num_humans_seen
    assert len(num_humans_seen) > 1
    assert rows[-1]['is_best_checkpoint'] == '1'
    assert rows[-1]['holdout_easy_success'] != 'nan'
    assert rows[-1]['holdout_hard_success'] != 'nan'
    for key in ['holdout_easy_avg_steps', 'holdout_hard_avg_I_sp', 'holdout_hard_min_d_min']:
        assert rows[-1][key] not in ('', None, 'nan')
        assert float(rows[-1][key]) == float(rows[-1][key])
    _assert_finite_update_diagnostics(rows[-1])
    assert save_path.exists() or os.path.exists(str(save_path).replace('.pt', '_final.pt'))


def test_evaluate_holdout_returns_behavior_diagnostics():
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.models import SNCPPolicy
    from sncp_ppo.ppo import PPOAgent

    device = torch.device('cpu')
    env = CrowdSimEnv(num_humans=1, scenario='easy', max_time=1.0)
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)
    agent = PPOAgent(policy=policy, epochs=1, batch_size=2, seq_len=4)

    result = evaluate_holdout(
        env,
        policy,
        agent,
        device,
        n_episodes=1,
        scenario='easy',
        base_seed=123,
    )

    assert result['avg_steps'] > 0.0
    assert result['avg_I_sp'] >= 0.0
    assert result['min_d_min'] >= 0.0
    assert result['avg_reward'] == result['avg_reward']


def test_compute_total_updates_vectorized_vs_single():
    """LR scheduler horizon must match the ACTUAL number of PPO updates.

    Single-env: episodes // update_freq (+ phase-boundary flushes).
    Vectorized: total_steps // (num_envs * horizon).
    The v11 bug used the single-env formula in vectorized mode, so the LR
    decayed to its floor at ~1/3 of the run and HARD/CIRCLE trained at ~0 lr.
    """
    from sncp_ppo.train import compute_total_updates

    # single-env: 1500 episodes, update_freq 5 -> 300 (+5 boundary flushes)
    assert compute_total_updates(num_envs=1, episodes=1500, update_freq=5,
                                 total_steps=2_000_000, horizon=128) == 305

    # vectorized: 2,000,000 steps / (16*128=2048) = 976 updates
    assert compute_total_updates(num_envs=16, episodes=1500, update_freq=5,
                                 total_steps=2_000_000, horizon=128) == 976

    # vectorized with different geometry: 8 envs * 64 = 512 -> 12000/512 = 23
    assert compute_total_updates(num_envs=8, episodes=50, update_freq=5,
                                 total_steps=12000, horizon=64) == 23


def test_curriculum_ramps_to_final_humans_with_parity_speed():
    """v15: density is monotonic and reaches final_num_humans; every phase speed
    is <= 0.26 (robot parity), so non-reactive avoidance stays feasible."""
    final = 10
    phases = [step_to_phase(int(f * 1000), 1000, final)
              for f in (0.0, 0.2, 0.4, 0.6, 0.9)]
    humans = [p[1] for p in phases]
    speeds = [p[2] for p in phases]
    assert humans == sorted(humans), f"density not monotonic: {humans}"
    assert humans[-1] == final, f"final phase N != {final}: {humans[-1]}"
    assert max(speeds) <= 0.26 + 1e-9, f"a phase exceeds parity: {speeds}"


def test_holdout_config_is_parity_and_has_highdensity():
    """v15: every holdout speed is <= 0.26, and a high-density (N>=10) holdout
    scenario exists to monitor the real target during training."""
    for name, (n, v) in SCENARIO_HOLDOUT_CONFIG.items():
        assert v <= 0.26 + 1e-9, f"{name} holdout speed {v} > parity"
    assert SCENARIO_HOLDOUT_CONFIG['circle'][0] >= 10, "no high-density holdout"


class _ReplayRng:
    def __init__(self, random_value=0.0, randint_value=0):
        self.random_value = random_value
        self.randint_value = randint_value
        self.randint_bounds = None

    def random(self):
        return self.random_value

    def randint(self, low, high):
        self.randint_bounds = (low, high)
        return self.randint_value


def test_select_vectorized_phase_uses_current_phase_when_replay_disabled():
    from sncp_ppo.train import select_vectorized_phase

    phase, is_replay = select_vectorized_phase(
        steps_seen=800,
        total_steps=1000,
        final_num_humans=10,
        replay_ratio=0.0,
        rng=_ReplayRng(random_value=0.0, randint_value=0),
    )

    assert phase == ('circle', 10, 0.26)
    assert is_replay is False


def test_select_vectorized_phase_samples_only_earlier_phases_for_replay():
    from sncp_ppo.train import select_vectorized_phase

    rng = _ReplayRng(random_value=0.0, randint_value=1)
    phase, is_replay = select_vectorized_phase(
        steps_seen=800,
        total_steps=1000,
        final_num_humans=10,
        replay_ratio=1.0,
        rng=rng,
    )

    assert phase == ('easy_plus', 3, 0.18)
    assert is_replay is True
    assert rng.randint_bounds == (0, 3)


def test_select_vectorized_phase_never_replays_before_second_phase():
    from sncp_ppo.train import select_vectorized_phase

    phase, is_replay = select_vectorized_phase(
        steps_seen=0,
        total_steps=1000,
        final_num_humans=10,
        replay_ratio=1.0,
        rng=_ReplayRng(random_value=0.0, randint_value=0),
    )

    assert phase == ('easy', 1, 0.13)
    assert is_replay is False


def test_vectorized_replay_updates_are_logged(tmp_path):
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.models import SNCPPolicy
    from sncp_ppo.ppo import PPOAgent

    save_path = tmp_path / 'vc_replay_smoke.pt'
    log_path = tmp_path / 'vc_replay_log.csv'
    args = argparse.Namespace(
        num_envs=2,
        horizon=8,
        total_steps=96,
        eval_freq_updates=0,
        episodes=1,
        num_humans=10,
        seed=42,
        holdout_scenarios=['easy', 'hard'],
        holdout_episodes=1,
        best_warmup_evals=0,
        best_min_success_threshold=0.0,
        save_path=str(save_path),
        lr=1e-4,
        target_kl=0.01,
        curriculum_replay_ratio=1.0,
    )
    device = torch.device('cpu')
    env = CrowdSimEnv(num_humans=args.num_humans, scenario='circle')
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)
    agent = PPOAgent(
        policy=policy, lr=args.lr, target_kl=args.target_kl,
        epochs=1, batch_size=2, seq_len=4,
    )

    with log_path.open('w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        _write_vector_log_header(csv_writer)
        _train_vectorized(args, env, policy, agent, device, str(log_path), csv_writer, csv_file)

    with log_path.open(newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert any(row['is_replay_update'] == '1' for row in rows)
    assert any(row['is_replay_update'] == '0' for row in rows)
    for row in rows:
        _assert_finite_update_diagnostics(row)
