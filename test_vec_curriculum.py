import argparse
import csv
import os
import subprocess
import sys

import torch

from sncp_ppo.train import _train_vectorized, step_to_phase, SCENARIO_HOLDOUT_CONFIG


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
        _train_vectorized(args, env, policy, agent, device, str(log_path), csv_writer, csv_file)

    with log_path.open(newline='') as csv_file:
        rows = list(csv.reader(csv_file))

    num_humans_seen = {int(row[2]) for row in rows}
    assert 1 in num_humans_seen
    assert len(num_humans_seen) > 1
    assert rows[-1][10] == '1'
    assert rows[-1][12] != 'nan'
    assert rows[-1][16] != 'nan'
    assert save_path.exists() or os.path.exists(str(save_path).replace('.pt', '_final.pt'))


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
