import argparse
import csv
import math
import os
import random
from datetime import datetime

import numpy as np
import torch

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.models import SNCPPolicy
from sncp_ppo.ppo import PPOAgent


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch (CPU + CUDA). The env itself reseeds
    on every reset(seed=...) call if the caller passes one."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _obs_to_tensor(obs, device):
    return {
        'robot_node': torch.tensor(obs['robot_node'], dtype=torch.float32, device=device).unsqueeze(0),
        'spatial_edges': torch.tensor(obs['spatial_edges'], dtype=torch.float32, device=device).unsqueeze(0),
        'temporal_edges': torch.tensor(obs['temporal_edges'], dtype=torch.float32, device=device).unsqueeze(0),
    }


#: Canonical (num_humans, human_vpref) per scenario name — what each scenario
#: "really means" independent of the current curriculum phase. Used so that
#: e.g. holdout on 'hard' evaluates 5 fast pedestrians regardless of whether
#: the trainer is still in the N=1 easy phase. Mirrors test_eval.py defaults
#: for hard/extreme (5 humans, vpref=0.50).
SCENARIO_HOLDOUT_CONFIG = {
    'easy':      (1, 0.15),
    'easy_plus': (2, 0.20),
    'medium':    (3, 0.30),
    'hard':      (5, 0.50),
    'extreme':   (5, 0.50),
    'circle':    (5, 0.50),
    'random':    (5, 0.50),
}


def evaluate_holdout(env, policy, agent, device, n_episodes, scenario, base_seed):
    """Deterministic rollouts on a fixed holdout scenario.

    Used to track real generalization instead of the curriculum-window
    training success rate (which inflates when curriculum is easy).

    Sets num_humans + human_vpref to the *canonical* per-scenario values
    (SCENARIO_HOLDOUT_CONFIG), not whatever the curriculum currently has —
    otherwise "holdout on hard" during the easy phase would actually be
    "1 fast human" rather than the canonical 5-human hard scenario.
    """
    prev_scenario = env.scenario
    prev_num_humans = env.num_humans
    prev_vpref = env.human_vpref

    n_h, vpref = SCENARIO_HOLDOUT_CONFIG.get(scenario, (5, 0.50))
    env.scenario = scenario
    env.num_humans = n_h
    env.human_vpref = vpref

    successes = 0
    collisions = 0
    timeouts = 0
    rewards = []
    max_steps = int(env.max_time / env.time_step) + 1

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        h_states = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)
        ep_reward = 0.0
        info = {'success': False, 'collision': False, 'timeout': False}

        for _ in range(max_steps):
            action, _, _, h_states = agent.select_action(obs, h_states, device, deterministic=True)
            env_action = PPOAgent.clip_action_for_env(action, env.robot_vpref, env.robot_wmax)
            obs, r, terminated, truncated, info = env.step(env_action)
            ep_reward += r
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        if info.get('success'):
            successes += 1
        elif info.get('collision'):
            collisions += 1
        else:
            timeouts += 1

    # Restore env config so the curriculum loop isn't perturbed
    env.scenario = prev_scenario
    env.num_humans = prev_num_humans
    env.human_vpref = prev_vpref

    return {
        'success_rate': successes / n_episodes,
        'collision_rate': collisions / n_episodes,
        'timeout_rate': timeouts / n_episodes,
        'avg_reward': float(np.mean(rewards)),
    }


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Seed: {args.seed}")

    # 1. Create environment — start with easy scenario, curriculum will change it
    env = CrowdSimEnv(num_humans=args.num_humans, scenario='easy', human_dodge_robot=False)

    # 2. Create SNCP policy and PPO agent
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)
    # Estimate total scheduled PPO updates:
    #   periodic = episodes // update_freq
    #   plus one forced-flush update at each curriculum-phase boundary (~5)
    # Over-estimating is harmless: LinearLR clamps at end_factor after total_iters.
    total_updates = args.episodes // args.update_freq + 5
    agent = PPOAgent(
        policy=policy,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        total_updates=total_updates,
        lr_end_factor=args.lr_end_factor,
    )

    # Defensive: if `checkpoints` exists as a *file* (e.g. left over from a
    # crashed run or a Colab artifact), os.makedirs would raise FileExistsError
    # even with exist_ok=True. Remove the stray file before recreating.
    ckpt_dir = os.path.dirname(args.save_path)
    if ckpt_dir:
        if os.path.exists(ckpt_dir) and not os.path.isdir(ckpt_dir):
            os.remove(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.exists('logs') and not os.path.isdir('logs'):
        os.remove('logs')
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Dynamic CSV header: one 4-tuple per holdout scenario
    holdout_cols = []
    for sc in args.holdout_scenarios:
        holdout_cols += [f'holdout_{sc}_vpref', f'holdout_{sc}_success', f'holdout_{sc}_collision',
                         f'holdout_{sc}_timeout', f'holdout_{sc}_reward']
    csv_writer.writerow([
        'episode', 'scenario', 'num_humans', 'human_vpref', 'phase_vpref', 'steps', 'reward',
        'success', 'collision', 'timeout', 'comfort',
    ] + holdout_cols)
    print(f"CSV log: {log_path}")

    # Multi-scenario holdout state: dict of per-scenario last results + best generalist
    nan_result = {'success_rate': float('nan'), 'collision_rate': float('nan'),
                  'timeout_rate': float('nan'), 'avg_reward': float('nan')}
    last_holdout_per_scenario = {sc: dict(nan_result) for sc in args.holdout_scenarios}
    best_holdout_min_success = -1.0  # generalist metric: min across scenarios

    success_history = []
    collision_history = []
    comfort_history = []
    reward_history = []

    # Curriculum schedule (4 phases on the 'circle' pattern). The previous
    # 5-phase schedule ended with an EXTREME phase that switched to 'random'
    # human spawn; this created a distribution shift the policy could not
    # adapt to in the time remaining, and reward actively deteriorated
    # (-26 → -90 over 360 episodes) — classic catastrophic forgetting on
    # out-of-distribution data. Five humans at vpref 0.50 on the (well-tested)
    # circle pattern is already a hard generalist target.
    curriculum = [
        (args.curriculum_easy_until,       'easy',      0.15, 1),
        (args.curriculum_easy_plus_until,  'easy_plus', 0.20, 2),
        (args.curriculum_medium_until,     'medium',    0.30, 3),
        (args.curriculum_hard_until,       'hard',      args.hard_vpref_train, 4),
        (args.episodes,                    'circle',    0.50, args.num_humans),
    ]

    print("\nStarting SNCP-PPO training with curriculum learning...")
    print(f"Episodes: {args.episodes} | Humans (final): {args.num_humans} | Seq len: {args.seq_len}")
    print(f"LR: {args.lr:.1e} -> {args.lr * args.lr_end_factor:.1e} over ~{total_updates} updates")
    print(f"Curriculum: " + " | ".join(
        f"{sc}<={thr} (N={n})" for thr, sc, _, n in curriculum))
    print(f"Replay ratio: {args.curriculum_replay_ratio:.0%} of update windows "
          f"sample an earlier phase (anti-forgetting)")
    print(f"Holdout: {args.holdout_episodes} eps × {args.holdout_scenarios} every "
          f"{args.eval_freq} eps (best ckpt = min(success))")
    hard_holdout_vpref = SCENARIO_HOLDOUT_CONFIG['hard'][1]
    print(f"Hard vpref (train vs holdout): {args.hard_vpref_train:.2f} vs {hard_holdout_vpref:.2f}")
    if not math.isclose(args.hard_vpref_train, hard_holdout_vpref, rel_tol=0.0, abs_tol=1e-9):
        print("[warning] hard vpref mismatch: training hard phase differs from canonical holdout hard vpref.")
    print("-" * 90)

    # Align env to first curriculum phase
    env.scenario, env.human_vpref, env.num_humans = (
        curriculum[0][1], curriculum[0][2], curriculum[0][3],
    )

    total_steps = 0
    # Persist the chosen phase across an entire PPO update window so a single
    # rollout buffer stays single-N (avoids shape mismatches in _extract_subsequences).
    window_phase = curriculum[0]
    window_is_replay = False

    for episode in range(1, args.episodes + 1):
        # At the start of every update window, pick this window's phase.
        # With prob (1 - replay_ratio) use the current curriculum phase;
        # otherwise sample a uniformly-random *earlier* phase as replay.
        # This prevents catastrophic forgetting of low-density scenarios: in
        # the v3 run the policy trained ~1350 episodes on N=3..5 and lost
        # the N=1 'easy' skill (test_eval scored 6% on 1-human after a
        # successful 81% on 5-human hard).
        if episode == 1 or ((episode - 1) % args.update_freq == 0):
            current_phase_idx = len(curriculum) - 1
            for idx, (threshold, _, _, _) in enumerate(curriculum):
                if episode <= threshold:
                    current_phase_idx = idx
                    break
            if (args.curriculum_replay_ratio > 0
                    and current_phase_idx > 0
                    and random.random() < args.curriculum_replay_ratio):
                replay_idx = random.randint(0, current_phase_idx - 1)
                window_phase = curriculum[replay_idx]
                window_is_replay = True
            else:
                window_phase = curriculum[current_phase_idx]
                window_is_replay = False

        target_scenario, target_vpref, target_num_humans = window_phase[1:]

        # If phase changed, flush memory (only num_humans matters for obs shape).
        # vpref is excluded from the equality check because env.reset() will
        # overwrite it from the scenario-default mapping; we re-set it AFTER
        # reset below so the curriculum value actually takes effect.
        if (env.scenario != target_scenario
                or env.num_humans != target_num_humans):
            if len(agent.memory.actions) > 0:
                print(f"\n  [Curriculum shift @ Ep {episode}] "
                      f"{env.scenario}/{env.num_humans}h -> {target_scenario}/{target_num_humans}h. "
                      f"Flushing memory and updating model.")
                agent.update(device)
            env.scenario = target_scenario
            env.num_humans = target_num_humans

        obs, info = env.reset(seed=args.seed + episode)
        # env.reset() resets human_vpref to the scenario default (e.g. 'hard' -> 0.50).
        # Re-apply the curriculum value so the hard phase uses --hard_vpref_train (default 0.50)
        # and pedestrians move at the intended monotone-ramp speed.
        env.human_vpref = target_vpref
        h_states = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)

        episode_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        next_obs = obs
        h_states_next = h_states

        while not (terminated or truncated):
            # select_action returns the un-clipped sample (with its true log_prob)
            action, log_prob, value, h_states_next = agent.select_action(obs, h_states, device)
            env_action = PPOAgent.clip_action_for_env(action, env.robot_vpref, env.robot_wmax)

            next_obs, reward, terminated, truncated, info = env.step(env_action)

            # mask = 1 means "the world keeps going from here": true unless terminated
            # (truncation is a soft boundary handled via bootstrap value below).
            mask = 0.0 if terminated else 1.0
            agent.memory.store(obs, h_states, action, log_prob, reward, value, mask)

            obs = next_obs
            h_states = h_states_next
            episode_reward += reward
            step_count += 1
            total_steps += 1

        # Bootstrap V(s_final) for truncated rollouts so GAE doesn't assume
        # the world ends at timeout.
        if truncated and not terminated:
            with torch.no_grad():
                _, _, next_value_tensor, _ = policy(_obs_to_tensor(next_obs, device), h_states_next)
                bootstrap_value = next_value_tensor.item()
        else:
            bootstrap_value = 0.0
        agent.memory.end_episode(bootstrap_value=bootstrap_value)

        success_history.append(float(info['success']))
        collision_history.append(float(info['collision']))
        comfort_history.append(float(info['comfort']))
        reward_history.append(episode_reward)

        # Periodic PPO update
        if episode % args.update_freq == 0:
            agent.update(device)

        # Multi-scenario holdout evaluation
        ran_eval = False
        if episode % args.eval_freq == 0:
            for sc in args.holdout_scenarios:
                last_holdout_per_scenario[sc] = evaluate_holdout(
                    env, policy, agent, device,
                    n_episodes=args.holdout_episodes,
                    scenario=sc,
                    base_seed=args.seed + 10_000 + episode,
                )
            ran_eval = True

            # Generalist metric: min success across all holdout scenarios.
            # Refuses to crown "100% on easy, 0% on hard" as a 50% best.
            min_success = min(r['success_rate'] for r in last_holdout_per_scenario.values())
            if min_success > best_holdout_min_success:
                best_holdout_min_success = min_success
                torch.save(policy.state_dict(), args.save_path)
                per_sc = {sc: f"{r['success_rate']:.0%}"
                          for sc, r in last_holdout_per_scenario.items()}
                print(f"  --> New best generalist min={min_success:.1%} {per_sc}, "
                      f"saved to {args.save_path}")

        # Per-episode CSV row (dynamic per-scenario holdout tail)
        ho_row = []
        for sc in args.holdout_scenarios:
            r = last_holdout_per_scenario[sc]
            canonical_vpref = SCENARIO_HOLDOUT_CONFIG.get(sc, (5, 0.50))[1]
            ho_row += [f"{canonical_vpref:.2f}", f"{r['success_rate']:.4f}", f"{r['collision_rate']:.4f}",
                       f"{r['timeout_rate']:.4f}", f"{r['avg_reward']:.4f}"]
        csv_writer.writerow([
            episode, env.scenario, env.num_humans, env.human_vpref, target_vpref, step_count, f"{episode_reward:.4f}",
            int(info['success']), int(info['collision']), int(info['timeout']), f"{info['comfort']:.4f}",
        ] + ho_row)
        csv_file.flush()

        # Stdout summary on the same cadence as before
        if episode % args.log_freq == 0:
            window = min(args.log_freq, len(reward_history))
            avg_reward = np.mean(reward_history[-window:])
            avg_success = np.mean(success_history[-window:])
            avg_collision = np.mean(collision_history[-window:])
            avg_comfort = np.mean(comfort_history[-window:])

            replay_mark = "R" if window_is_replay else " "
            line = (f"Ep {episode:4d}/{args.episodes} "
                    f"[{replay_mark} {env.scenario.upper():9s} {env.num_humans}h] | "
                    f"Steps: {total_steps:7d} | Reward: {avg_reward:7.2f} | "
                    f"Success: {avg_success:5.1%} | Collision: {avg_collision:5.1%} | "
                    f"Comfort: {avg_comfort:6.2f}")
            ho_summary = " ".join(
                f"{sc[:3]}:{r['success_rate']:.0%}"
                for sc, r in last_holdout_per_scenario.items()
                if r['success_rate'] == r['success_rate']  # not NaN
            )
            if ho_summary:
                line += f" | Hold[{ho_summary}]"
            # PPO diagnostics from the most recent update — reveals policy
            # collapse (ent→0), exploding KL, or stuck std before the holdout
            # metrics catch up.
            with torch.no_grad():
                std = policy.actor_logstd.exp().squeeze().cpu().numpy()
            line += (f" | ent={agent.last_entropy:+.3f}"
                     f" kl={agent.last_approx_kl:.5f}"
                     f" std=[{std[0]:.3f},{std[1]:.3f}]"
                     f" rms={agent.return_rms.std:.2f}")
            print(line)

        # Periodic checkpoints
        if episode % 200 == 0:
            periodic_path = args.save_path.replace('.pt', f'_ep{episode}.pt')
            torch.save(policy.state_dict(), periodic_path)

    # Final save
    torch.save(policy.state_dict(), args.save_path.replace('.pt', '_final.pt'))
    csv_file.close()
    print("\nTraining completed!")
    print(f"Best generalist (min across {args.holdout_scenarios}): {best_holdout_min_success:.1%}")
    print(f"CSV log saved to: {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SNCP-PPO with curriculum + holdout eval.')

    # Episodes / data sizing
    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--num_humans', type=int, default=5,
                        help='Humans in final curriculum phase (and used by env eval).')
    parser.add_argument('--scenario', type=str, default='easy',
                        help='Initial env scenario (curriculum overrides).')
    parser.add_argument('--seed', type=int, default=42)

    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate. Lowered from 3e-4 to stabilize PPO updates '
                             'after easy-phase convergence; decayed by --lr_end_factor.')
    parser.add_argument('--lr_end_factor', type=float, default=0.1,
                        help='Final lr = base lr * this factor (linear decay over training).')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=4,
                        help='PPO optimization epochs per update (standard 4-10).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Mini-batch size in number of BPTT subsequences.')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Subsequence length for BPTT through the LTC cells.')
    parser.add_argument('--update_freq', type=int, default=5,
                        help='Episodes between PPO updates.')
    parser.add_argument('--curriculum_replay_ratio', type=float, default=0.0,
                        help='[EXPERIMENTAL — default off after v5 regression] '
                             'Fraction of PPO update windows that re-sample a '
                             'uniformly-random earlier curriculum phase instead '
                             'of training on the current one. The intent was to '
                             'prevent forgetting of low-density (N=1,2) scenarios. '
                             'Empirically at 0.2 this stole ~20%% of phase-specific '
                             'sample budget AND contaminated the return-RMS '
                             'normalizer with mixed-distribution returns, killing '
                             'HARD-phase learning (rolling success 85%% → 10%%). '
                             'Kept opt-in for experiments; set to 0 by default.')

    # Curriculum thresholds (inclusive) — 5-phase: 10%/25%/50%/75%/100%
    parser.add_argument('--curriculum_easy_until', type=int, default=None,
                        help='Episodes <= this run easy (N=1). Default: 10%% of total.')
    parser.add_argument('--curriculum_easy_plus_until', type=int, default=None,
                        help='Episodes <= this run easy_plus (N=2). Default: 25%% of total.')
    parser.add_argument('--curriculum_medium_until', type=int, default=None,
                        help='Episodes <= this run medium (N=3). Default: 50%% of total.')
    parser.add_argument('--curriculum_hard_until', type=int, default=None,
                        help='Episodes <= this run hard (N=4). Default: 75%% of total. Rest run extreme (N=5).')

    parser.add_argument('--hard_vpref_train', type=float, default=0.50,
                        help='Human vpref used during hard curriculum phase (canonical default: 0.50).')

    # Holdout evaluation
    parser.add_argument('--eval_freq', type=int, default=50,
                        help='Episodes between holdout evaluations.')
    parser.add_argument('--holdout_episodes', type=int, default=30,
                        help='Episodes per holdout evaluation per scenario (higher = lower variance).')
    parser.add_argument('--holdout_scenarios', type=str, nargs='+',
                        default=['easy', 'hard'],
                        choices=['easy', 'easy_plus', 'medium', 'hard', 'extreme', 'circle', 'random'],
                        help='Scenarios for periodic holdout eval. Best checkpoint is saved '
                             'when min(success across these) improves — rewards generalists, '
                             'not "100% on one, 0% on the other" specialists.')
    parser.add_argument('--holdout_scenario', type=str, default=None,
                        help='[Deprecated] Single-scenario alias for --holdout_scenarios. '
                             'If set, overrides --holdout_scenarios with a one-element list.')

    # Logging / checkpointing
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='checkpoints/sncp_ppo.pt')

    args = parser.parse_args()

    # Deprecated alias: --holdout_scenario hard  ->  --holdout_scenarios [hard]
    if args.holdout_scenario is not None:
        print(f"[deprecated] --holdout_scenario is deprecated, prefer --holdout_scenarios. "
              f"Promoting '{args.holdout_scenario}' to a single-element list.")
        args.holdout_scenarios = [args.holdout_scenario]

    # Default curriculum thresholds derived from total episodes
    # 5-phase split: 10% / 25% / 50% / 75% / 100%
    if args.curriculum_easy_until is None:
        args.curriculum_easy_until = int(args.episodes * 0.10)
    if args.curriculum_easy_plus_until is None:
        args.curriculum_easy_plus_until = int(args.episodes * 0.25)
    if args.curriculum_medium_until is None:
        args.curriculum_medium_until = int(args.episodes * 0.50)
    if args.curriculum_hard_until is None:
        args.curriculum_hard_until = int(args.episodes * 0.75)

    train(args)
