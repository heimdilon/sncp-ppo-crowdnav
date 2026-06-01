import argparse
import csv
import os
import subprocess
import sys

import torch

from sncp_ppo.train import _train_vectorized, step_to_phase


def test_step_to_phase_boundaries():
    total = 1000

    assert step_to_phase(0, total, 5) == ('easy', 1, 0.15)
    assert step_to_phase(99, total, 5) == ('easy', 1, 0.15)
    assert step_to_phase(100, total, 5) == ('easy', 1, 0.15)
    assert step_to_phase(101, total, 5) == ('easy_plus', 2, 0.20)
    assert step_to_phase(250, total, 5) == ('easy_plus', 2, 0.20)
    assert step_to_phase(251, total, 5) == ('medium', 3, 0.30)
    assert step_to_phase(500, total, 5) == ('medium', 3, 0.30)
    assert step_to_phase(501, total, 5) == ('hard', 4, 0.40)
    assert step_to_phase(750, total, 5) == ('hard', 4, 0.40)
    assert step_to_phase(751, total, 5) == ('circle', 5, 0.50)
    assert step_to_phase(1000, total, 5) == ('circle', 5, 0.50)
    assert step_to_phase(99999, total, 5) == ('circle', 5, 0.50)


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
