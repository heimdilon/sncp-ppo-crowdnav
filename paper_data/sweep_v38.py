"""Honest local multi-seed density sweep for v38 (locked v34 Beta policy + training-free action shield).

IDENTICAL protocol/accounting to scratch/_sweep_v34.py (5 seeds x 50 ep = 250/density,
paper_challenging, robot 1.0, human 1.0, max_time None, goal_noise 0) but with
action_shield=True (horizon 6, safety_margin 0.0) so the numbers are directly comparable
to v34_multiseed_result.json -- same evaluate_density path, shield on vs off.

Run from repo root:  C:/ProgramData/miniconda3/python.exe scratch/_sweep_v38.py
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sncp_ppo.eval_report import evaluate_density  # noqa: E402

CKPT = "sncp_ppo_v34.pt"          # v38 = locked v34 Beta policy + runtime shield
SEEDS = [100, 200, 300, 400, 500]
DENSITIES = [5, 10, 15, 20]
N_EP = 50
OUT = "v38_multiseed_result.json"


def pooled_se(p: float, n: int) -> float:
    return math.sqrt(p * (1.0 - p) / n) if n else float("nan")


def main() -> None:
    results: dict = {}
    t0 = time.time()
    for N in DENSITIES:
        block_means = []
        all_succ, all_coll, all_to = [], [], []
        succ_steps, i_sps = [], []
        for s in SEEDS:
            eps = evaluate_density(
                checkpoint_path=CKPT,
                num_humans=N,
                scenario="paper_challenging",
                n_episodes=N_EP,
                seed=s,
                robot_vpref=1.0,
                human_vpref_override=1.0,
                max_time=None,
                human_goal_noise=0.0,
                action_shield=True,
                shield_horizon_steps=6,
                shield_safety_margin=0.0,
            )
            b_succ = [e.success for e in eps]
            block_means.append(sum(b_succ) / len(b_succ))
            all_succ += b_succ
            all_coll += [e.collision for e in eps]
            all_to += [e.timeout for e in eps]
            succ_steps += [e.steps for e in eps if e.success]
            i_sps += [e.avg_i_sp for e in eps]
            print(f"N={N} seed={s} success={block_means[-1] * 100:.1f}% ({time.time() - t0:.0f}s)", flush=True)
        n = len(all_succ)
        p = sum(all_succ) / n
        results[str(N)] = {
            "block_means": block_means,
            "pooled_success": p,
            "pooled_se": pooled_se(p, n),
            "pooled_collision": sum(all_coll) / n,
            "pooled_timeout": sum(all_to) / n,
            "avg_success_steps": (sum(succ_steps) / len(succ_steps)) if succ_steps else float("nan"),
            "avg_i_sp": sum(i_sps) / len(i_sps),
            "n": n,
        }
        print(
            f"== N={N} POOLED success={p * 100:.1f}% "
            f"coll={results[str(N)]['pooled_collision'] * 100:.1f}% "
            f"to={results[str(N)]['pooled_timeout'] * 100:.1f}% ==",
            flush=True,
        )
        results["_elapsed_s"] = time.time() - t0
        json.dump(results, open(OUT, "w"), indent=2)

    print(f"DONE {OUT} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
