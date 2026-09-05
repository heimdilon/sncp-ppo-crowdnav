"""Phase 0 (validation) + Phase 1 (collection) for IL warm-start.

Runs the ORCA robot-expert (il.expert) in the paper regime and either:
  --validate : print a success/collision/timeout table per density (the gate:
               is the expert good enough at high N to be worth cloning?), or
  (default)  : collect successful-episode demos and save them to an .npz.

Demos are grouped per density (the spatial-edge width = num_humans, so different
N cannot be stacked into one ragged array). Each density shard holds flattened
per-step arrays plus episode_lengths so the BC pretrain can cut BPTT windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.il.expert import expert_action

PAPER_REGIME = dict(
    scenario="hard",
    robot_vpref=1.0,
    human_vpref_override=1.0,
    human_goal_noise=2.0,
    max_time=15.0,
    human_motion_model="orca",
)


def run_expert_episode(env, seed, responsibility=1.0, time_horizon=3.0):
    """Run one expert-controlled episode; return its transitions and outcome."""
    obs, _ = env.reset(seed=seed)
    rn, se, te, acts = [], [], [], []
    max_steps = int(env.max_time / env.time_step) + 1
    info = {"success": False, "collision": False, "timeout": False}
    steps = 0
    while steps < max_steps:
        v, w = expert_action(
            env, responsibility=responsibility, time_horizon=time_horizon
        )
        action = np.array([v, w], dtype=np.float32)
        rn.append(np.asarray(obs["robot_node"], dtype=np.float32))
        se.append(np.asarray(obs["spatial_edges"], dtype=np.float32))
        te.append(np.asarray(obs["temporal_edges"], dtype=np.float32))
        acts.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    return {
        "robot_node": rn,
        "spatial_edges": se,
        "temporal_edges": te,
        "actions": acts,
        "success": bool(info["success"]),
        "collision": bool(info["collision"]),
        "timeout": bool(info.get("timeout", False)),
        "steps": steps,
    }


def collect_dataset(
    densities, n_per_density, seed, responsibility=1.0, time_horizon=3.0, **regime
):
    """Collect successful-episode demos per density. Returns (shards, stats).

    shards[N] = {robot_node (T,7), spatial_edges (T,N,6), temporal_edges (T,2),
                 actions (T,2), episode_lengths [list]} or absent if 0 kept.
    stats[N]  = {episodes, success, collision, timeout, kept, mean_len}.
    """
    cfg = {**PAPER_REGIME, **regime}
    shards, stats = {}, {}
    seed_cursor = seed
    for n_humans in densities:
        env = CrowdSimEnv(num_humans=n_humans, **cfg)
        kept_rn, kept_se, kept_te, kept_act, ep_lens = [], [], [], [], []
        n_succ = n_coll = n_tout = 0
        for _ in range(n_per_density):
            ep = run_expert_episode(
                env,
                seed=seed_cursor,
                responsibility=responsibility,
                time_horizon=time_horizon,
            )
            seed_cursor += 1
            n_succ += ep["success"]
            n_coll += ep["collision"]
            n_tout += ep["timeout"] and not ep["success"]
            if ep["success"]:
                kept_rn.extend(ep["robot_node"])
                kept_se.extend(ep["spatial_edges"])
                kept_te.extend(ep["temporal_edges"])
                kept_act.extend(ep["actions"])
                ep_lens.append(ep["steps"])
        stats[n_humans] = {
            "episodes": n_per_density,
            "success": n_succ,
            "collision": n_coll,
            "timeout": n_tout,
            "kept": len(ep_lens),
            "mean_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
        }
        if ep_lens:
            shards[n_humans] = {
                "robot_node": np.stack(kept_rn),
                "spatial_edges": np.stack(kept_se),
                "temporal_edges": np.stack(kept_te),
                "actions": np.stack(kept_act),
                "episode_lengths": ep_lens,
            }
    return shards, stats


def save_dataset(shards, path):
    """Flatten per-density shards into one .npz with n{N}_<field> keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for n_humans, shard in shards.items():
        flat[f"n{n_humans}_robot_node"] = shard["robot_node"]
        flat[f"n{n_humans}_spatial_edges"] = shard["spatial_edges"]
        flat[f"n{n_humans}_temporal_edges"] = shard["temporal_edges"]
        flat[f"n{n_humans}_actions"] = shard["actions"]
        flat[f"n{n_humans}_episode_lengths"] = np.asarray(
            shard["episode_lengths"], dtype=np.int64
        )
    flat["densities"] = np.asarray(sorted(shards), dtype=np.int64)
    np.savez_compressed(path, **flat)
    return path


def load_dataset(path):
    """Inverse of save_dataset: returns shards keyed by density."""
    data = np.load(path, allow_pickle=False)
    shards = {}
    for n_humans in data["densities"].tolist():
        shards[n_humans] = {
            "robot_node": data[f"n{n_humans}_robot_node"],
            "spatial_edges": data[f"n{n_humans}_spatial_edges"],
            "temporal_edges": data[f"n{n_humans}_temporal_edges"],
            "actions": data[f"n{n_humans}_actions"],
            "episode_lengths": data[f"n{n_humans}_episode_lengths"].tolist(),
        }
    return shards


def _print_stats(stats):
    print(
        f"{'N':>3} | {'succ':>5} {'coll':>5} {'tout':>5} | {'kept':>5} {'mean_len':>8}"
    )
    print("-" * 44)
    for n_humans in sorted(stats):
        s = stats[n_humans]
        ep = s["episodes"]
        print(
            f"{n_humans:>3} | {s['success'] / ep:>5.0%} {s['collision'] / ep:>5.0%} "
            f"{s['timeout'] / ep:>5.0%} | {s['kept']:>5} {s['mean_len']:>8.1f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--densities", type=int, nargs="+", default=[1, 3, 5, 8, 10])
    parser.add_argument(
        "--n_per_density",
        type=int,
        default=50,
        help="Validation: 50 is a sweep. Collection: use a large value (e.g. 400).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--responsibility", type=float, default=1.0)
    parser.add_argument(
        "--time_horizon",
        type=float,
        default=3.0,
        help="ORCA planning horizon (s). Longer = earlier, more cautious avoidance.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Phase-0 gate: only print the expert sweep, do not save demos.",
    )
    parser.add_argument("--out", type=str, default="data/il_demos.npz")
    args = parser.parse_args(argv)

    shards, stats = collect_dataset(
        args.densities,
        args.n_per_density,
        args.seed,
        responsibility=args.responsibility,
        time_horizon=args.time_horizon,
    )
    _print_stats(stats)
    if not args.validate:
        path = save_dataset(shards, args.out)
        total = sum(len(s["episode_lengths"]) for s in shards.values())
        print(
            f"\nSaved {total} successful episodes across {len(shards)} densities -> {path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
