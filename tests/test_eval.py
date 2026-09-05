import argparse
import os
import random

import numpy as np
import torch

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.models import SNCPPolicy
from sncp_ppo.ppo import PPOAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode(env, policy, agent, device, seed, max_steps, verbose=False):
    obs, _ = env.reset(seed=seed)
    h_states = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)

    total_reward = 0.0
    total_comfort = 0.0
    step = 0
    info = {'success': False, 'collision': False, 'timeout': False, 'comfort': 0.0, 'I_sp': 0.0}

    while step < max_steps:
        action, _, _, h_states_next = agent.select_action(obs, h_states, device, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_comfort += info.get('I_sp', 0.0)
        step += 1
        if verbose:
            print(f"  Step {step:4d} | Pos ({env.robot_px:6.2f}, {env.robot_py:6.2f}) | "
                  f"Theta {env.robot_theta:7.4f} | Action ({action[0]:5.3f}, {action[1]:6.3f}) | Reward {reward:6.3f}")
        h_states = h_states_next
        if terminated or truncated:
            break

    return {
        'success': bool(info.get('success', False)),
        'collision': bool(info.get('collision', False)),
        'timeout': bool(info.get('timeout', False)),
        'steps': step,
        'total_reward': total_reward,
        'avg_I_sp': total_comfort / max(step, 1),
    }


def run_evaluation(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Seed: {args.seed} | Scenario: {args.scenario} | num_humans: {args.num_humans}")

    env = CrowdSimEnv(num_humans=args.num_humans, scenario=args.scenario)
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)

    if os.path.exists(args.checkpoint):
        policy.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded policy from {args.checkpoint}")
    else:
        print(f"No checkpoint at {args.checkpoint} - using untrained policy.")

    policy.train(False)  # equivalent to policy.eval(); switches to inference mode
    agent = PPOAgent(policy=policy)

    results = []
    max_steps = int(env.max_time / env.time_step) + 1
    print(f"\nRunning {args.n_episodes} episodes (max_steps={max_steps})...")
    print("-" * 70)

    for ep in range(args.n_episodes):
        ep_seed = args.seed + ep
        r = run_episode(env, policy, agent, device, ep_seed, max_steps, verbose=args.verbose)
        results.append(r)
        outcome = 'SUCCESS' if r['success'] else ('COLLISION' if r['collision'] else 'TIMEOUT')
        print(f"Ep {ep+1:3d} (seed {ep_seed}) | {outcome:9s} | steps {r['steps']:4d} | "
              f"reward {r['total_reward']:7.2f} | avg I_sp {r['avg_I_sp']:6.3f}")

    n = len(results)
    success_rate = sum(r['success'] for r in results) / n
    collision_rate = sum(r['collision'] for r in results) / n
    timeout_rate = sum(r['timeout'] for r in results) / n
    avg_steps_success = np.mean([r['steps'] for r in results if r['success']]) if any(r['success'] for r in results) else float('nan')
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_I_sp = np.mean([r['avg_I_sp'] for r in results])

    print("\n" + "=" * 70)
    print(f"SUMMARY over {n} episodes")
    print("=" * 70)
    print(f"  Success rate:           {success_rate:6.1%}")
    print(f"  Collision rate:         {collision_rate:6.1%}")
    print(f"  Timeout rate:           {timeout_rate:6.1%}")
    print(f"  Avg steps (successes):  {avg_steps_success:6.1f}")
    print(f"  Avg total reward:       {avg_reward:7.2f}")
    print(f"  Avg per-step I_sp:      {avg_I_sp:6.3f}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SNCP-PPO policy.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sncp_ppo.pt')
    parser.add_argument('--num_humans', type=int, default=5,
                        help='Must match training num_humans for fair comparison.')
    parser.add_argument('--scenario', type=str, default='circle',
                        choices=['easy', 'easy_plus', 'medium', 'hard', 'extreme', 'circle', 'random'])
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true', help='Print per-step trace.')
    run_evaluation(parser.parse_args())
