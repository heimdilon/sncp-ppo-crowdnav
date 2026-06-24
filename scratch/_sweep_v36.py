"""Honest local multi-seed density sweep for v36 combined levers.

Protocol: 5 seeds (100..500) x 50 episodes at N=5/10/15/20,
paper_challenging, robot 1.0, human 1.0, max_time None (the environment
resolves the paper CHALLENGING 50 s budget), and goal_noise 0.

Run from the repo root after placing sncp_ppo_v36.pt there:
    python scratch/_sweep_v36.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sncp_ppo.eval_report import evaluate_density  # noqa: E402


CKPT = 'sncp_ppo_v36.pt'
SEEDS = [100, 200, 300, 400, 500]
DENSITIES = [5, 10, 15, 20]
N_EP = 50
OUT = 'v36_multiseed_result.json'


def pooled_se(p: float, n: int) -> float:
    return math.sqrt(p * (1.0 - p) / n) if n else float("nan")


def main() -> None:
    results: dict = {}
    t0 = time.time()
    for num_humans in DENSITIES:
        block_means = []
        all_success, all_collision, all_timeout = [], [], []
        success_steps, social_pressure = [], []
        for seed in SEEDS:
            episodes = evaluate_density(
                checkpoint_path=CKPT,
                num_humans=num_humans,
                scenario="paper_challenging",
                n_episodes=N_EP,
                seed=seed,
                robot_vpref=1.0,
                human_vpref_override=1.0,
                max_time=None,
                human_goal_noise=0.0,
            )
            seed_success = [episode.success for episode in episodes]
            block_means.append(sum(seed_success) / len(seed_success))
            all_success.extend(seed_success)
            all_collision.extend(episode.collision for episode in episodes)
            all_timeout.extend(episode.timeout for episode in episodes)
            success_steps.extend(episode.steps for episode in episodes if episode.success)
            social_pressure.extend(episode.avg_i_sp for episode in episodes)
            print(
                f"N={num_humans} seed={seed} success={block_means[-1] * 100:.1f}% "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

        n = len(all_success)
        success = sum(all_success) / n
        results[str(num_humans)] = {
            "block_means": block_means,
            "pooled_success": success,
            "pooled_se": pooled_se(success, n),
            "pooled_collision": sum(all_collision) / n,
            "pooled_timeout": sum(all_timeout) / n,
            "avg_success_steps": (
                sum(success_steps) / len(success_steps) if success_steps else float("nan")
            ),
            "avg_i_sp": sum(social_pressure) / len(social_pressure),
            "n": n,
        }
        print(
            f"== N={num_humans} POOLED success={success * 100:.1f}% "
            f"coll={results[str(num_humans)]['pooled_collision'] * 100:.1f}% "
            f"to={results[str(num_humans)]['pooled_timeout'] * 100:.1f}% ==",
            flush=True,
        )
        results["_elapsed_s"] = time.time() - t0
        Path(OUT).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"DONE {OUT} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
