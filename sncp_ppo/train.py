import argparse
import csv
import os
import random
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import torch

from crowd_sim.crowd_env import CrowdSimEnv, PAPER_SCENARIO_CONFIG
from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
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


def _fmt_duration(seconds):
    """Human-readable wall-clock duration.

    Drops seconds once we're at the hour scale (where they're just noise):
    '38s', '1m 15s', '45m 12s', '1h 2m', '5h 12m'.
    """
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _eta_seconds(recent_ep_times, remaining_episodes):
    """Estimate remaining wall-clock time from a moving window of episode
    durations. Uses the mean of `recent_ep_times` (the last N episodes) so the
    estimate tracks the *current* curriculum phase's speed rather than the
    whole-run average — important here because per-episode cost grows as the
    curriculum ramps N=1 -> N=5. Returns 0.0 when there are no samples yet."""
    if not recent_ep_times:
        return 0.0
    avg = sum(recent_ep_times) / len(recent_ep_times)
    return avg * remaining_episodes


#: Canonical (num_humans, human_vpref) per scenario name — what each scenario
#: "really means" independent of the current curriculum phase. Used so that
#: e.g. holdout on 'hard' evaluates 5 fast pedestrians regardless of whether
#: the trainer is still in the N=1 easy phase. Mirrors test_eval.py defaults
#: for hard/extreme (5 humans, vpref=0.50).
SCENARIO_HOLDOUT_CONFIG = {
    'easy':      (1, 0.13),
    'easy_plus': (3, 0.18),
    'medium':    (5, 0.22),
    'hard':      (5, 0.26),
    'extreme':   (10, 0.26),
    'circle':    (10, 0.26),
    'random':    (10, 0.26),
    'paper_standard': (5, 1.0),
    'paper_challenging': (10, 1.0),
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
    step_counts = []
    episode_avg_i_sp = []
    episode_min_d_min = []
    max_steps = int(env.max_time / env.time_step) + 1

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        h_states = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)
        ep_reward = 0.0
        info = {'success': False, 'collision': False, 'timeout': False}
        step_count = 0
        i_sp_values = []
        d_min_values = []

        for _ in range(max_steps):
            action, _, _, h_states = agent.select_action(obs, h_states, device, deterministic=True)
            env_action = PPOAgent.clip_action_for_env(action, env.robot_vpref, env.robot_wmax)
            obs, r, terminated, truncated, info = env.step(env_action)
            ep_reward += r
            step_count += 1
            i_sp = float(info.get('I_sp', float('nan')))
            d_min = float(info.get('d_min', float('nan')))
            if i_sp == i_sp:
                i_sp_values.append(i_sp)
            if d_min == d_min and np.isfinite(d_min):
                d_min_values.append(d_min)
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        step_counts.append(step_count)
        if i_sp_values:
            episode_avg_i_sp.append(float(np.mean(i_sp_values)))
        if d_min_values:
            episode_min_d_min.append(float(np.min(d_min_values)))
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
        'avg_steps': float(np.mean(step_counts)),
        'avg_I_sp': float(np.mean(episode_avg_i_sp)) if episode_avg_i_sp else float('nan'),
        'min_d_min': float(np.min(episode_min_d_min)) if episode_min_d_min else float('nan'),
    }


HOLDOUT_CSV_FIELDS = [
    ('success', 'success_rate'),
    ('collision', 'collision_rate'),
    ('timeout', 'timeout_rate'),
    ('reward', 'avg_reward'),
    ('avg_steps', 'avg_steps'),
    ('avg_I_sp', 'avg_I_sp'),
    ('min_d_min', 'min_d_min'),
]


def holdout_csv_columns(scenarios):
    columns = []
    for sc in scenarios:
        columns.extend(f'holdout_{sc}_{csv_name}' for csv_name, _ in HOLDOUT_CSV_FIELDS)
    return columns


def empty_holdout_result():
    return {result_key: float('nan') for _, result_key in HOLDOUT_CSV_FIELDS}


def holdout_csv_row(holdout_per_scenario, scenarios):
    row = []
    for sc in scenarios:
        result = holdout_per_scenario[sc]
        for _, result_key in HOLDOUT_CSV_FIELDS:
            row.append(f"{result[result_key]:.4f}")
    return row


UPDATE_DIAGNOSTIC_COLUMNS = [
    'entropy',
    'approx_kl',
    'std_linear',
    'std_angular',
    'return_rms_std',
]


def _policy_std_pair(policy):
    """(std_v, std_w) for the Gaussian global logstd, or (nan, nan) for Beta
    (the Beta action dist is state-dependent and has no single global std)."""
    if hasattr(policy, 'actor_logstd'):
        with torch.no_grad():
            s = policy.actor_logstd.exp().squeeze().detach().cpu().numpy()
        return float(s[0]), float(s[1])
    return float('nan'), float('nan')


def update_diagnostic_row(policy, agent):
    """Return PPO stability diagnostics for CSV logging."""
    std0, std1 = _policy_std_pair(policy)
    return [
        f"{float(agent.last_entropy):.6f}",
        f"{float(agent.last_approx_kl):.8f}",
        f"{std0:.6f}",
        f"{std1:.6f}",
        f"{float(agent.return_rms.std):.6f}",
    ]


def make_env(num_humans, scenario, seed, comfort_coeff=None, max_time=None,
             robot_vpref=0.26, human_vpref_override=None, human_goal_noise=0.0,
             human_motion_model='orca', collision_threshold=None, paper_regime=False):
    """Factory for a single CrowdSimEnv, used by SyncVectorEnv.

    comfort_coeff/max_time default to None so the env resolves them per regime:
    paper_regime (or a paper scenario) -> 12.5s / comfort 2.0 / d_col 0.3; otherwise
    the legacy 50s / 6.0 / 0.6. paper_regime carries the paper budget into non-paper
    bootstrap phases (the easy curriculum) so the whole run trains under time pressure.
    """
    def _thunk():
        env = CrowdSimEnv(
            num_humans=num_humans,
            scenario=scenario,
            comfort_coeff=comfort_coeff,
            max_time=max_time,
            robot_vpref=robot_vpref,
            human_vpref_override=human_vpref_override,
            human_goal_noise=human_goal_noise,
            human_motion_model=human_motion_model,
            collision_threshold=collision_threshold,
            paper_regime=paper_regime,
        )
        env.reset(seed=seed)
        return env
    return _thunk


def build_or_load_policy(args, env, device):
    """Build the SNCP policy, optionally initializing it from a checkpoint.

    With --init_checkpoint (v23 IL warm-start), the policy is loaded from the BC
    checkpoint and its architecture is auto-detected from the saved keys
    (build_policy_for_checkpoint), so PPO fine-tunes from the cloned weights
    instead of from scratch. Without it, a fresh policy is built per --pre_mlp.
    """
    init_ckpt = getattr(args, 'init_checkpoint', None)
    if init_ckpt:
        state = torch.load(init_ckpt, map_location=device)
        policy = build_policy_for_checkpoint(
            state, robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax
        ).to(device)
        policy.load_state_dict(state)
        print(f"Initialized policy from {init_ckpt} (IL warm-start)")
        return policy
    return SNCPPolicy(
        robot_vpref=env.robot_vpref,
        robot_wmax=env.robot_wmax,
        pre_mlp=getattr(args, 'pre_mlp', False),
        attn_count_scaling=getattr(args, 'attn_count_scaling', False),
        meanmax_pool=getattr(args, 'meanmax_pool', False),
        node_units=getattr(args, 'node_units', 128),
        node_output=getattr(args, 'node_output', 48),
        attn_heads=getattr(args, 'attn_heads', 1),
        action_dist=getattr(args, 'action_dist', 'gaussian'),
    ).to(device)


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Seed: {args.seed}")

    # Paper regime is derived from the fixed scenario so the 12.5s/comfort-2/d_col-0.3
    # budget applies to EVERY env (incl. the non-paper easy bootstrap and the holdout
    # eval_env), not just envs literally constructed with a paper scenario. This is the
    # fix for the v24 failure where the CLI silently trained at the 50s env default.
    paper_regime = getattr(args, 'fixed_scenario', None) in PAPER_SCENARIO_CONFIG

    # 1. Create environment — start with easy scenario, curriculum will change it.
    # human_dodge_robot inherits the env default (False, v15): pedestrians ignore
    # the robot ("invisible robot", the paper's CrowdNav regime), so the robot
    # must ACTIVELY avoid them. This is feasible because v15 caps pedestrian speed
    # to the robot's (parity, <=0.26 m/s) — see step_to_phase + the scenario speed
    # block. (v14 used a reactive cooperative crowd, which let the robot beeline;
    # v15 reverts to non-reactive + strong comfort to force genuine avoidance.)
    env = CrowdSimEnv(
        num_humans=args.num_humans,
        scenario='easy',
        comfort_coeff=args.comfort_coeff,
        max_time=args.max_time,
        robot_vpref=args.robot_vpref,
        human_vpref_override=args.human_vpref_override,
        human_goal_noise=args.human_goal_noise,
        human_motion_model=getattr(args, 'human_motion_model', 'orca'),
        collision_threshold=args.collision_threshold,
        paper_regime=paper_regime,
    )

    # 2. Create SNCP policy and PPO agent
    policy = build_or_load_policy(args, env, device)
    # Scheduled PPO updates for the LR scheduler. Vectorized mode does one
    # update per fixed-horizon rollout (total_steps // (num_envs*horizon)),
    # which is very different from the single-env episodes//update_freq. Using
    # the wrong one (the v11 bug) collapses the LR to its floor partway through.
    total_updates = compute_total_updates(
        num_envs=args.num_envs, episodes=args.episodes,
        update_freq=args.update_freq, total_steps=args.total_steps,
        horizon=args.horizon,
    )
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
        target_kl=args.target_kl,
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
    csv_writer.writerow([
        'episode', 'scenario', 'num_humans', 'human_vpref', 'is_replay_update',
        'steps', 'reward', 'success', 'collision', 'timeout', 'comfort',
        *UPDATE_DIAGNOSTIC_COLUMNS,
        'is_best_checkpoint', 'best_reason',
    ] + holdout_csv_columns(args.holdout_scenarios))
    print(f"CSV log: {log_path}")

    # Multi-scenario holdout state: dict of per-scenario last results + best generalist
    last_holdout_per_scenario = {sc: empty_holdout_result() for sc in args.holdout_scenarios}
    best_holdout_min_success = -1.0  # generalist metric: min across scenarios
    best_holdout_score = (-1.0, -float('inf'), -float('inf'))  # (min_success, avg_reward, -collision_rate)
    holdout_eval_count = 0

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
        (args.curriculum_hard_until,       'hard',      0.40, 4),
        (args.episodes,                    'circle',    0.50, args.num_humans),
    ]

    print("\nStarting SNCP-PPO training with curriculum learning...")
    print(f"Episodes: {args.episodes} | Humans (final): {args.num_humans} | Seq len: {args.seq_len}")
    print(f"LR: {args.lr:.1e} -> {args.lr * args.lr_end_factor:.1e} over ~{total_updates} updates")
    print("Curriculum: " + " | ".join(
        f"{sc}<={thr} (N={n})" for thr, sc, _, n in curriculum))
    print(f"Replay ratio: {args.curriculum_replay_ratio:.0%} of update windows "
          f"sample an earlier phase (anti-forgetting)")
    print(f"Holdout: {args.holdout_episodes} eps × {args.holdout_scenarios} every "
          f"{args.eval_freq} eps (best ckpt = min(success))")
    print("-" * 90)

    # Align env to first curriculum phase
    env.scenario, env.human_vpref, env.num_humans = (
        curriculum[0][1], curriculum[0][2], curriculum[0][3],
    )

    total_steps = 0
    train_start = time.perf_counter()
    recent_ep_times = deque(maxlen=50)  # last-N episode wall-clock times for ETA
    # Persist the chosen phase across an entire PPO update window so a single
    # rollout buffer stays single-N (avoids shape mismatches in _extract_subsequences).
    window_phase = curriculum[0]
    window_is_replay = False

    if args.num_envs > 1:
        _train_vectorized(args, env, policy, agent, device, log_path, csv_writer, csv_file)
        return

    for episode in range(1, args.episodes + 1):
        iter_start = time.perf_counter()
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
        # Re-apply the curriculum value so e.g. 'hard' phase actually uses 0.40
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
        is_best_checkpoint = 0
        best_reason = ''
        if episode % args.eval_freq == 0:
            for sc in args.holdout_scenarios:
                last_holdout_per_scenario[sc] = evaluate_holdout(
                    env, policy, agent, device,
                    n_episodes=args.holdout_episodes,
                    scenario=sc,
                    base_seed=args.seed + 10_000 + episode,
                )
            holdout_eval_count += 1

            # Generalist metric: min success across all holdout scenarios.
            # Refuses to crown "100% on easy, 0% on hard" as a 50% best.
            min_success = min(r['success_rate'] for r in last_holdout_per_scenario.values())
            avg_reward = float(np.mean([r['avg_reward'] for r in last_holdout_per_scenario.values()]))
            avg_collision = float(np.mean([r['collision_rate'] for r in last_holdout_per_scenario.values()]))
            current_score = (min_success, avg_reward, -avg_collision)

            if holdout_eval_count <= args.best_warmup_evals:
                best_reason = (f"best skipped due to warmup "
                               f"(eval {holdout_eval_count}/{args.best_warmup_evals})")
                print(f"  --> {best_reason}: min={min_success:.1%}, "
                      f"avg_reward={avg_reward:.3f}, collision={avg_collision:.1%}")
            elif min_success < args.best_min_success_threshold:
                best_reason = (f"best skipped due to threshold "
                               f"(min_success={min_success:.1%} < {args.best_min_success_threshold:.1%})")
                print(f"  --> {best_reason}")
            elif current_score > best_holdout_score:
                best_holdout_min_success = min_success
                best_holdout_score = current_score
                # I/O-robust best save (Drive FUSE can die mid-run -> Errno 107).
                try:
                    torch.save(policy.state_dict(), args.save_path)
                    is_best_checkpoint = 1
                except OSError as e:
                    fallback = os.path.join('/content', os.path.basename(args.save_path))
                    try:
                        torch.save(policy.state_dict(), fallback)
                        is_best_checkpoint = 1
                        print(f"  [warning] best save to {args.save_path} failed ({e}); "
                              f"saved to {fallback} instead.")
                    except OSError as e2:
                        print(f"  [error] best save failed on both paths ({e2}); skipping.")
                best_reason = ('best updated (priority: min_success, tie-break: avg_reward, '
                               'then lower collision_rate)')
                per_sc = {sc: f"{r['success_rate']:.0%}"
                          for sc, r in last_holdout_per_scenario.items()}
                print(f"  --> New best generalist min={min_success:.1%}, "
                      f"avg_reward={avg_reward:.3f}, collision={avg_collision:.1%} {per_sc}, "
                      f"saved to {args.save_path}")
            else:
                best_reason = ('best not updated: score did not improve '
                               '(priority: min_success, tie-break: avg_reward, lower collision_rate)')
                print(f"  --> {best_reason}")

        # Per-episode CSV row (dynamic per-scenario holdout tail)
        ho_row = holdout_csv_row(last_holdout_per_scenario, args.holdout_scenarios)
        # Best-effort CSV write. On Colab the log dir sometimes lives behind a
        # FUSE mount (Drive) that can disconnect mid-run; if the underlying
        # write/flush fails, drop this row rather than tearing down the whole
        # training process. The model is still being saved to args.save_path
        # via torch.save() on holdout-best, so the run is recoverable from
        # checkpoints even without the CSV.
        try:
            csv_writer.writerow([
                episode, env.scenario, env.num_humans, env.human_vpref, int(window_is_replay),
                step_count, f"{episode_reward:.4f}",
                int(info['success']), int(info['collision']), int(info['timeout']), f"{info['comfort']:.4f}",
                *update_diagnostic_row(policy, agent),
                is_best_checkpoint, best_reason,
            ] + ho_row)
            csv_file.flush()
        except OSError as e:
            # Only warn once per session to avoid flooding the terminal.
            if not getattr(train, '_csv_io_warned', False):
                print(f"  [warning] CSV log write failed ({e}); continuing without per-episode logging.")
                train._csv_io_warned = True

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
            std0, std1 = _policy_std_pair(policy)
            line += (f" | ent={agent.last_entropy:+.3f}"
                     f" kl={agent.last_approx_kl:.5f}"
                     f" std=[{std0:.3f},{std1:.3f}]"
                     f" rms={agent.return_rms.std:.2f}")
            # Live progress: elapsed wall-clock + moving-average ETA + clock
            # time of the projected finish (uses the last <=50 episodes, so the
            # estimate tracks the current curriculum phase's pace).
            elapsed = time.perf_counter() - train_start
            eta = _eta_seconds(list(recent_ep_times), args.episodes - episode)
            line += f" | elapsed {_fmt_duration(elapsed)}"
            if eta > 0:
                finish = (datetime.now() + timedelta(seconds=eta)).strftime('%H:%M')
                line += f" | eta {_fmt_duration(eta)} | ~{finish} biter"
            print(line)

        # Periodic checkpoints
        if episode % 200 == 0:
            periodic_path = args.save_path.replace('.pt', f'_ep{episode}.pt')
            torch.save(policy.state_dict(), periodic_path)

        # Record this iteration's wall-clock time (rollout + update + any
        # holdout eval) for the moving-average ETA shown on log lines.
        recent_ep_times.append(time.perf_counter() - iter_start)

    # Final save
    torch.save(policy.state_dict(), args.save_path.replace('.pt', '_final.pt'))
    csv_file.close()
    print("\nTraining completed!")
    print(f"Total time: {_fmt_duration(time.perf_counter() - train_start)}")
    print(f"Best generalist (min across {args.holdout_scenarios}): {best_holdout_min_success:.1%}")
    print(f"CSV log saved to: {log_path}")


def compute_total_updates(num_envs, episodes, update_freq, total_steps, horizon):
    """Number of PPO updates the LR scheduler should decay over.

    Single-env (num_envs == 1): one update every `update_freq` episodes, plus a
    forced-flush update at each of the ~5 curriculum-phase boundaries.
    Vectorized (num_envs > 1): one update per fixed-horizon rollout, i.e.
    total_steps // (num_envs * horizon). The v11 bug used the single-env formula
    in vectorized mode, so LR hit its floor at ~1/3 of the run and HARD/CIRCLE
    trained at the floor lr.
    """
    if num_envs > 1:
        return total_steps // (num_envs * horizon)
    return episodes // update_freq + 5


def curriculum_phases(final_num_humans):
    """Return the v15/v16 density curriculum phases.

    Each phase is (scenario, num_humans, human_vpref). Speeds stay at or below
    robot parity so non-reactive avoidance remains physically feasible.
    """
    return [
        ('easy', 1, 0.13),
        ('easy_plus', 3, 0.18),
        ('medium', 5, 0.22),
        ('hard', 8, 0.24),
        ('circle', final_num_humans, 0.26),
    ]


def phase_index_for_steps(steps_seen, total_steps):
    """Map an env-step count to a curriculum phase index.

    Boundaries are inclusive fractions of total_steps: 10/25/50/75%, matching
    the single-env curriculum.
    """
    frac = steps_seen / max(1, total_steps)
    if frac <= 0.10:
        return 0
    if frac <= 0.25:
        return 1
    if frac <= 0.50:
        return 2
    if frac <= 0.75:
        return 3
    return 4


def step_to_phase(steps_seen, total_steps, final_num_humans):
    """Map an env-step count to a curriculum phase."""
    return curriculum_phases(final_num_humans)[
        phase_index_for_steps(steps_seen, total_steps)
    ]


def select_vectorized_phase(steps_seen, total_steps, final_num_humans,
                            replay_ratio=0.0, rng=random, fixed_scenario=None,
                            bootstrap_easy_steps=0, num_humans_range=None):
    """Select the next vectorized PPO update phase.

    With replay enabled, a fraction of update windows re-samples a uniformly
    random earlier phase. The whole update still uses one phase, preserving the
    fixed human-count tensor shape expected by the vectorized rollout buffer.

    fixed_scenario pins every update window to a single phase
    (scenario, final_num_humans, canonical scenario speed) and disables replay —
    probe mode for short fixed-density attribution runs. bootstrap_easy_steps
    prepends an easy/1 warmup before the pinned phase: probe run 1 showed that
    cold-starting at fixed N=5 never bootstraps goal-reaching in ANY regime
    (the curriculum's easy phase is the bootstrap), so probes without a warmup
    measure exploration failure, not regime difficulty.
    """
    if fixed_scenario is not None:
        if steps_seen < bootstrap_easy_steps:
            _, easy_vpref = SCENARIO_HOLDOUT_CONFIG['easy']
            return ('easy', 1, easy_vpref), False
        _, vpref = SCENARIO_HOLDOUT_CONFIG.get(fixed_scenario, (5, 0.26))
        n_humans = final_num_humans
        if num_humans_range is not None:
            n_humans = rng.randint(int(num_humans_range[0]), int(num_humans_range[1]))
        return (fixed_scenario, n_humans, vpref), False
    phases = curriculum_phases(final_num_humans)
    current_idx = phase_index_for_steps(steps_seen, total_steps)
    if replay_ratio > 0.0 and current_idx > 0 and rng.random() < replay_ratio:
        replay_idx = rng.randint(0, current_idx - 1)
        return phases[replay_idx], True
    return phases[current_idx], False


def _vec_episode_flags(info, i):
    """(success, collision) flags for env i's just-finished episode.

    gymnasium >=1.0 NEXT_STEP autoreset returns the final step's info directly
    as stacked arrays; older SAME_STEP autoreset tucks it into `final_info`.
    """
    final_infos = info.get('final_info')
    if final_infos is not None and final_infos[i] is not None:
        fi = final_infos[i]
        return bool(fi.get('success')), bool(fi.get('collision'))

    def flag(key):
        arr = info.get(key)
        return bool(arr[i]) if arr is not None else False

    return flag('success'), flag('collision')


def _train_vectorized(args, env, policy, agent, device, log_path, csv_writer, csv_file):
    """Vectorized rollout with step-budgeted curriculum and holdout eval.

    All parallel envs share num_humans so observations batch cleanly. At phase
    boundaries, envs are recreated between PPO updates and recurrent state is
    reinitialized for the new human count.
    """
    import gymnasium as gym
    from sncp_ppo.vec_buffer import VectorizedRolloutBuffer, reset_hidden_where_done

    N, T = args.num_envs, args.horizon
    comfort_coeff = getattr(args, 'comfort_coeff', 6.0)
    max_time = getattr(args, 'max_time', 50.0)
    robot_vpref = getattr(args, 'robot_vpref', 0.26)
    human_vpref_override = getattr(args, 'human_vpref_override', None)
    human_goal_noise = getattr(args, 'human_goal_noise', 0.0)
    human_motion_model = getattr(args, 'human_motion_model', 'orca')
    collision_threshold = getattr(args, 'collision_threshold', None)
    fixed_scenario = getattr(args, 'fixed_scenario', None)
    bootstrap_easy_steps = getattr(args, 'bootstrap_easy_steps', 0)
    num_humans_range = getattr(args, 'num_humans_range', None)
    # Forces the paper budget on every env (incl. the easy bootstrap + holdout eval_env);
    # comfort_coeff/max_time above may be None, which the env resolves per this flag.
    paper_regime = fixed_scenario in PAPER_SCENARIO_CONFIG

    def set_envs_vpref(envs, vpref):
        for sub_env in envs.envs:
            sub_env.unwrapped.human_vpref = vpref

    def build_envs(num_humans, scenario, vpref):
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    num_humans,
                    scenario,
                    args.seed + i,
                    comfort_coeff=comfort_coeff,
                    max_time=max_time,
                    robot_vpref=robot_vpref,
                    human_vpref_override=human_vpref_override,
                    human_goal_noise=human_goal_noise,
                    human_motion_model=human_motion_model,
                    collision_threshold=collision_threshold,
                    paper_regime=paper_regime,
                )
                for i in range(N)
            ]
        )
        obs, _ = envs.reset(seed=args.seed)
        set_envs_vpref(envs, vpref)
        return envs, obs

    def to_tensor(o):
        return {
            'robot_node': torch.tensor(o['robot_node'], dtype=torch.float32, device=device),
            'spatial_edges': torch.tensor(o['spatial_edges'], dtype=torch.float32, device=device),
            'temporal_edges': torch.tensor(o['temporal_edges'], dtype=torch.float32, device=device),
        }

    last_holdout_per_scenario = {sc: empty_holdout_result() for sc in args.holdout_scenarios}
    best_holdout_min_success = -1.0
    best_holdout_score = (-1.0, -float('inf'), -float('inf'))
    holdout_eval_count = 0
    eval_env = CrowdSimEnv(
        num_humans=args.num_humans,
        scenario='circle',
        comfort_coeff=comfort_coeff,
        max_time=max_time,
        robot_vpref=robot_vpref,
        human_vpref_override=human_vpref_override,
        human_goal_noise=human_goal_noise,
        human_motion_model=human_motion_model,
        collision_threshold=collision_threshold,
        paper_regime=paper_regime,
    )

    (scenario, H, vpref), _ = select_vectorized_phase(
        0, args.total_steps, args.num_humans, fixed_scenario=fixed_scenario,
        bootstrap_easy_steps=bootstrap_easy_steps, num_humans_range=num_humans_range,
    )
    envs, obs_np = build_envs(H, scenario, vpref)
    h = policy.init_hidden(batch_size=N, num_humans=H, device=device)
    ep_return = np.zeros(N, dtype=np.float64)

    total_steps = 0
    update_idx = 0
    replay_ratio = getattr(args, 'curriculum_replay_ratio', 0.0)
    print(f"\nVectorized training: {N} envs x {T} steps = {N * T} transitions/update")
    print(f"Curriculum by step budget: total={args.total_steps}, phases 10/25/50/75%")
    print(f"Replay ratio: {replay_ratio:.0%} of vectorized update windows sample earlier phases")
    print(f"Holdout every {args.eval_freq_updates} updates on {args.holdout_scenarios}")
    print("Note: --episodes is ignored in vectorized mode; --total_steps controls run length.")
    print("-" * 90)

    while total_steps < args.total_steps:
        (next_scenario, next_H, next_vpref), is_replay_update = select_vectorized_phase(
            total_steps,
            args.total_steps,
            args.num_humans,
            replay_ratio=replay_ratio,
            rng=random,
            fixed_scenario=fixed_scenario,
            bootstrap_easy_steps=bootstrap_easy_steps,
            num_humans_range=num_humans_range,
        )
        if next_H != H or next_scenario != scenario:
            replay_mark = " replay" if is_replay_update else ""
            print(f"\n  [Curriculum shift @ step {total_steps}{replay_mark}] "
                  f"{scenario}/{H}h -> {next_scenario}/{next_H}h")
            envs.close()
            scenario, H, vpref = next_scenario, next_H, next_vpref
            envs, obs_np = build_envs(H, scenario, vpref)
            h = policy.init_hidden(batch_size=N, num_humans=H, device=device)
            ep_return = np.zeros(N, dtype=np.float64)  # in-flight episodes discarded

        buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
        win_counts = {'success': 0, 'collision': 0, 'timeout': 0}
        win_returns = []
        for t in range(T):
            obs_t = to_tensor(obs_np)
            with torch.no_grad():
                mu, std, value, h_next = policy(obs_t, h)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            act_np = action.cpu().numpy()
            act_np[:, 0] = np.clip(act_np[:, 0], 0.0, env.robot_vpref)
            act_np[:, 1] = np.clip(act_np[:, 1], -env.robot_wmax, env.robot_wmax)
            next_obs, reward, term, trunc, info = envs.step(act_np)
            done = np.logical_or(term, trunc)
            set_envs_vpref(envs, vpref)
            ep_return += reward
            for i in np.nonzero(done)[0]:
                win_returns.append(float(ep_return[i]))
                ep_return[i] = 0.0
                success, collision = _vec_episode_flags(info, i)
                if success:
                    win_counts['success'] += 1
                elif collision:
                    win_counts['collision'] += 1
                else:
                    win_counts['timeout'] += 1
            done_t = torch.tensor(done, dtype=torch.float32, device=device)
            reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
            mask_t = torch.tensor(1.0 - term.astype('float32'), device=device)
            buf.store(obs=obs_t, hidden=h, actions=action, log_probs=log_prob,
                      rewards=reward_t, values=value.squeeze(-1),
                      dones=done_t, masks=mask_t)
            obs_np = next_obs
            h = reset_hidden_where_done(h_next, done_t, H)
            total_steps += N

        with torch.no_grad():
            last_v = policy(to_tensor(obs_np), h)[2].squeeze(-1)
        # last_dones = whether each env *terminated* on the final rollout step
        # (term from the last loop iteration). Passing zeros (the old bug) gave a
        # terminated-at-horizon env a nonzero bootstrap V(auto-reset_s0), biasing
        # its return upward — and with timeout=0% in this env, episodes terminate
        # frequently right at the horizon. last_dones=term zeros that bootstrap.
        last_dones = torch.tensor(term.astype('float32'), device=device)
        buf.finish(last_values=last_v, last_dones=last_dones)
        agent.update_vectorized(buf, device)
        update_idx += 1

        is_best_checkpoint = 0
        best_reason = ''
        if args.eval_freq_updates > 0 and update_idx % args.eval_freq_updates == 0:
            for sc in args.holdout_scenarios:
                last_holdout_per_scenario[sc] = evaluate_holdout(
                    eval_env, policy, agent, device,
                    n_episodes=args.holdout_episodes,
                    scenario=sc,
                    base_seed=args.seed + 10_000 + total_steps,
                )
            holdout_eval_count += 1

            min_success = min(r['success_rate'] for r in last_holdout_per_scenario.values())
            avg_reward = float(np.mean([r['avg_reward'] for r in last_holdout_per_scenario.values()]))
            avg_collision = float(np.mean([r['collision_rate'] for r in last_holdout_per_scenario.values()]))
            current_score = (min_success, avg_reward, -avg_collision)

            if holdout_eval_count <= args.best_warmup_evals:
                best_reason = (f"best skipped due to warmup "
                               f"(eval {holdout_eval_count}/{args.best_warmup_evals})")
                print(f"  --> {best_reason}: min={min_success:.1%}, "
                      f"avg_reward={avg_reward:.3f}, collision={avg_collision:.1%}")
            elif min_success < args.best_min_success_threshold:
                best_reason = (f"best skipped due to threshold "
                               f"(min_success={min_success:.1%} < {args.best_min_success_threshold:.1%})")
                print(f"  --> {best_reason}")
            elif current_score > best_holdout_score:
                best_holdout_min_success = min_success
                best_holdout_score = current_score
                # I/O-robust best save (Drive FUSE can die mid-run -> Errno 107).
                try:
                    torch.save(policy.state_dict(), args.save_path)
                    is_best_checkpoint = 1
                except OSError as e:
                    fallback = os.path.join('/content', os.path.basename(args.save_path))
                    try:
                        torch.save(policy.state_dict(), fallback)
                        is_best_checkpoint = 1
                        print(f"  [warning] best save to {args.save_path} failed ({e}); "
                              f"saved to {fallback} instead.")
                    except OSError as e2:
                        print(f"  [error] best save failed on both paths ({e2}); skipping.")
                best_reason = ('best updated (priority: min_success, tie-break: avg_reward, '
                               'then lower collision_rate)')
                per_sc = {sc: f"{r['success_rate']:.0%}"
                          for sc, r in last_holdout_per_scenario.items()}
                print(f"  --> New best generalist min={min_success:.1%}, "
                      f"avg_reward={avg_reward:.3f}, collision={avg_collision:.1%} {per_sc}, "
                      f"saved to {args.save_path}")
            else:
                best_reason = ('best not updated: score did not improve '
                               '(priority: min_success, tie-break: avg_reward, lower collision_rate)')
                print(f"  --> {best_reason}")

        ho_row = holdout_csv_row(last_holdout_per_scenario, args.holdout_scenarios)
        # Outcomes of episodes that FINISHED inside this rollout window. Before
        # this, update rows logged only PPO diagnostics and the holdout (every
        # eval_freq_updates) was the sole outcome signal — too coarse to read
        # short probe runs (probe run 1 lesson).
        n_finished = len(win_returns)
        if n_finished:
            win_reward = f"{float(np.mean(win_returns)):.4f}"
            win_success = f"{win_counts['success'] / n_finished:.4f}"
            win_collision = f"{win_counts['collision'] / n_finished:.4f}"
            win_timeout = f"{win_counts['timeout'] / n_finished:.4f}"
        else:
            win_reward = win_success = win_collision = win_timeout = ''
        try:
            csv_writer.writerow([
                total_steps, scenario, H, vpref, int(is_replay_update), T,
                win_reward, win_success, win_collision, win_timeout, '',
                *update_diagnostic_row(policy, agent),
                is_best_checkpoint, best_reason,
            ] + ho_row)
            csv_file.flush()
        except OSError as e:
            if not getattr(_train_vectorized, '_csv_io_warned', False):
                print(f"  [warning] CSV log write failed ({e}); continuing without per-update logging.")
                _train_vectorized._csv_io_warned = True

        if update_idx % 10 == 0:
            std0, std1 = _policy_std_pair(policy)
            replay_mark = "R" if is_replay_update else " "
            print(f"Update {update_idx} | step {total_steps}/{args.total_steps} [{scenario} {H}h] | "
                  f"[{replay_mark}] ent={agent.last_entropy:+.3f} kl={agent.last_approx_kl:.5f} "
                  f"std=[{std0:.3f},{std1:.3f}] rms={agent.return_rms.std:.2f}")

    envs.close()
    # I/O-robust final save: on Colab the save dir may sit behind a Drive FUSE
    # mount that can die mid-run (Errno 107). A bare torch.save would then crash
    # and lose the final checkpoint. Try the configured path; on OSError fall
    # back to a local /content path so hours of compute aren't lost.
    final_path = args.save_path.replace('.pt', '_final.pt')
    try:
        torch.save(policy.state_dict(), final_path)
    except OSError as e:
        fallback = os.path.join('/content', os.path.basename(final_path))
        try:
            torch.save(policy.state_dict(), fallback)
            print(f"  [warning] final save to {final_path} failed ({e}); "
                  f"saved to {fallback} instead.")
        except OSError as e2:
            print(f"  [error] final save failed on both paths ({e2}); "
                  f"best checkpoint (if any) may still be on disk.")
    try:
        csv_file.close()
    except OSError:
        pass
    print("\nVectorized training completed!")
    print(f"Best generalist (min across {args.holdout_scenarios}): {best_holdout_min_success:.1%}")
    print(f"CSV log saved to: {log_path}")


def build_parser():
    parser = argparse.ArgumentParser(description='Train SNCP-PPO with curriculum + holdout eval.')

    # Episodes / data sizing
    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--num_humans', type=int, default=5,
                        help='Humans in final curriculum phase (and used by env eval).')
    parser.add_argument('--num_humans_range', type=int, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Density curriculum: sample N~U[MIN,MAX] per update window '
                             '(paper trains 10-20). Default None = fixed --num_humans.')
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
    parser.add_argument('--target_kl', type=float, default=0.015,
                        help='Approx-KL early-stop threshold for PPO update epochs '
                             '(an epoch breaks once approx_kl > 1.5x this). Lower = '
                             'more conservative updates / steadier convergence. v9 uses 0.01.')
    parser.add_argument('--epochs', type=int, default=4,
                        help='PPO optimization epochs per update (standard 4-10).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Mini-batch size in BPTT subsequences. Default 16 preserves '
                             'v22/v23 training behavior; --batch_size 64 is an opt-in throughput '
                             'mode that improves GPU utilization but changes PPO minibatch statistics.')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Subsequence length for BPTT through the LTC cells.')
    parser.add_argument('--update_freq', type=int, default=5,
                        help='Episodes between PPO updates.')
    parser.add_argument('--curriculum_replay_ratio', type=float, default=0.0,
                        help='Fraction of PPO update windows that re-sample a '
                             'uniformly-random earlier curriculum phase instead '
                             'of training on the current one. v16 uses this in '
                             'the vectorized path to reduce catastrophic '
                             'forgetting after the v15 N=5 peak. Default stays '
                             '0 so replay is an explicit experiment variable.')
    parser.add_argument('--comfort_coeff', type=float, default=None,
                        help='Social-pressure comfort penalty coefficient. None lets '
                             'the env resolve it: paper regime 2.0, else legacy 6.0.')
    parser.add_argument('--max_time', type=float, default=None,
                        help='Episode time limit in seconds. None lets the env resolve '
                             'it: paper regime 12.5, else legacy 50.0.')
    parser.add_argument('--collision_threshold', type=float, default=None,
                        help='Robot-human collision distance. Default = robot_radius '
                             '+ human_radius (0.6). The paper uses 0.3 (Table 1).')
    parser.add_argument('--robot_vpref', type=float, default=0.26,
                        help='Robot max speed (m/s). TurtleBot3 Waffle hardware = 0.26; '
                             'the paper-reproduction run passes 1.0 to match the paper.')
    parser.add_argument('--human_vpref_override', type=float, default=None,
                        help='If set, force a flat pedestrian speed across all scenarios '
                             '(parity regime, e.g. 1.0 for paper reproduction). Default None '
                             'keeps the per-scenario speeds.')
    parser.add_argument('--human_goal_noise', type=float, default=0.0,
                        help='Per-axis uniform noise on circle-crossing pedestrian goals so '
                             'they do not all funnel through the exact center (paper-like '
                             'spread). 0 = exact antipodal (legacy). Paper regime uses ~2.0.')
    parser.add_argument('--human_motion_model', type=str, default='orca',
                        choices=['sfm', 'orca', 'linear'],
                        help='Pedestrian motion model. Default orca (v20+, the paper\'s '
                             'CrowdSim regime); sfm restores the v18 Social-Force crowd '
                             'for ablation probes.')
    parser.add_argument('--fixed_scenario', type=str, default=None,
                        choices=list(SCENARIO_HOLDOUT_CONFIG),
                        help='Probe mode: pin EVERY vectorized update window to this single '
                             'phase (scenario, --num_humans, canonical speed), bypassing the '
                             '10/25/50/75%% curriculum and replay. For short fixed-density '
                             'attribution runs; leave unset for real training.')
    parser.add_argument('--pre_mlp', action='store_true',
                        help='Paper Eq 11 fidelity: expand edge inputs to the 256-dim encoding '
                             'with an MLP BEFORE the NCP encoders (v22 candidate). Default off '
                             'preserves the v14..v21 architecture and checkpoint compatibility.')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Initialize the policy from this checkpoint instead of fresh weights '
                             '(v23 IL warm-start: PPO fine-tunes from the BC checkpoint). The '
                             'architecture is auto-detected from the saved keys.')
    parser.add_argument('--attn_count_scaling', action='store_true',
                        help='Scale attention scores by n/sqrt(d_k) (paper Eq 13, n=#humans) so the '
                             'pedestrian count enters the softmax temperature — high-N candidate. '
                             'Default off preserves v14..v23 behavior. Ignored when --init_checkpoint '
                             'is set (the variant is taken from the checkpoint).')
    parser.add_argument('--meanmax_pool', action='store_true',
                        help='Mean+max attention pooling (v30): concat the attention-weighted '
                             'mean with an element-wise max over humans, merged by Linear(512->256). '
                             'Cardinality-robust fix for the high-N convex-combination washout. '
                             'Default off preserves v14..v29 architecture and checkpoint compatibility.')
    parser.add_argument('--node_units', type=int, default=128,
                        help='Node-fusion NCP total neuron count (v31 capacity experiment; default '
                             '128 preserves v14..v30). Auto-detected from the checkpoint on load.')
    parser.add_argument('--node_output', type=int, default=48,
                        help='Node-fusion NCP motor (output) neuron count; must be < --node_units '
                             '(default 48; v31 uses 96 for units=256). Auto-detected on load.')
    parser.add_argument('--attn_heads', type=int, default=1,
                        help='Attention heads for crowd pooling. 1 (default) = legacy single-head; '
                             '>1 = canonical multi-head cross-attention (v33). Auto-detected on load.')
    parser.add_argument('--action_dist', type=str, default='gaussian',
                        choices=['gaussian', 'beta'],
                        help='Policy action distribution. gaussian (default) = Normal+clip; '
                             'beta = bounded state-dependent Beta head (v34). Auto-detected on load.')
    parser.add_argument('--bootstrap_easy_steps', type=int, default=0,
                        help='Probe mode only: run an easy/1 warmup for this many env steps '
                             'before the --fixed_scenario phase. Cold-starting at fixed N=5 '
                             'never bootstraps goal-reaching (probe run 1), so attribution '
                             'probes need an in-regime warmup.')

    # Curriculum thresholds (inclusive) — 5-phase: 10%/25%/50%/75%/100%
    parser.add_argument('--curriculum_easy_until', type=int, default=None,
                        help='Episodes <= this run easy (N=1). Default: 10%% of total.')
    parser.add_argument('--curriculum_easy_plus_until', type=int, default=None,
                        help='Episodes <= this run easy_plus (N=2). Default: 25%% of total.')
    parser.add_argument('--curriculum_medium_until', type=int, default=None,
                        help='Episodes <= this run medium (N=3). Default: 50%% of total.')
    parser.add_argument('--curriculum_hard_until', type=int, default=None,
                        help='Episodes <= this run hard (N=4). Default: 75%% of total. Rest run extreme (N=5).')

    # Holdout evaluation
    parser.add_argument('--eval_freq', type=int, default=50,
                        help='Episodes between holdout evaluations.')
    parser.add_argument('--holdout_episodes', type=int, default=50,
                        help='Episodes per holdout evaluation per scenario (higher = lower variance). '
                             'Raised 30->50: at 30 the best-checkpoint metric was noisy '
                             '(v7 "50%%" holdout was really 38%% on 100-ep eval).')
    parser.add_argument('--holdout_scenarios', type=str, nargs='+',
                        default=['easy', 'hard'],
                        choices=list(SCENARIO_HOLDOUT_CONFIG),
                        help='Scenarios for periodic holdout eval. Best checkpoint is saved '
                             'when min(success across these) improves — rewards generalists, '
                             'not "100%% on one, 0%% on the other" specialists.')
    parser.add_argument('--holdout_scenario', type=str, default=None,
                        help='[Deprecated] Single-scenario alias for --holdout_scenarios. '
                             'If set, overrides --holdout_scenarios with a one-element list.')
    parser.add_argument('--best_min_success_threshold', type=float, default=0.05,
                        help='Minimum holdout min_success required before considering a new best checkpoint.')
    parser.add_argument('--best_warmup_evals', type=int, default=3,
                        help='Number of initial holdout evaluations used only for metric collection (no best save).')

    # Logging / checkpointing
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='checkpoints/sncp_ppo.pt')

    parser.add_argument('--num_envs', type=int, default=1,
                        help='Parallel envs. 1 = legacy single-env path; '
                             '>1 = vectorized fixed-horizon rollout.')
    parser.add_argument('--horizon', type=int, default=128,
                        help='Steps per env per PPO update in vectorized mode.')
    parser.add_argument('--total_steps', type=int, default=2_000_000,
                        help='Env-step budget (vectorized mode): drives curriculum '
                             'phase boundaries (10/25/50/75%%) and total run length.')
    parser.add_argument('--eval_freq_updates', type=int, default=20,
                        help='Holdout evaluation cadence in PPO updates (vectorized mode).')

    return parser


if __name__ == '__main__':
    parser = build_parser()
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
