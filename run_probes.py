"""Faz-0 attribution probes for the v21 paper-regime failure.

v21 bundled five variable changes (robot speed, SFM->ORCA, pedestrian parity
speed, max_time, goal noise) and failed; this ladder decomposes the difficulty
with SHORT fixed-density local runs (probe mode: --fixed_scenario hard, N=5,
500k steps) so the next A100 run targets the variable that actually matters.

Ladder (each probe differs from its predecessor by ~one concept):
  P1_v18_regime   robot 0.26 + SFM @ scenario speed + max_time 50   (known-good anchor)
  P2_speed_only   robot 1.0  + SFM @ scenario speed + max_time 15   (speed effect)
  P3_orca         robot 1.0  + ORCA @ scenario speed + max_time 15  (ped-model effect)
  P4_v21_core     robot 1.0  + ORCA @ parity 1.0 + goal noise 2.0   (parity effect; v21 anchor)
  P5_paper_lr     P4 with LR 1e-4 (the paper's Table-1 value)       (LR effect)

Run-1 lesson: cold-starting at fixed N=5 never bootstraps goal-reaching in ANY
regime (even the v18-regime control stayed at 0% for 300k) — the curriculum's
easy phase IS the bootstrap. Probes therefore warm up on easy/1 for the first
--bootstrap_easy_steps before the pinned N=5 phase, and the trainer now logs
per-update window outcomes (success/collision/timeout/reward) so short runs
are readable between holdouts.

Usage:
  python run_probes.py                 # run all five sequentially
  python run_probes.py --probes P4 P5  # subset
  python run_probes.py --total_steps 300000
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

LOG_DIR = Path("logs")
CKPT_DIR = Path("checkpoints")

COMMON = [
    "--num_envs", "16",
    "--horizon", "128",
    "--fixed_scenario", "hard",
    "--num_humans", "5",
    "--bootstrap_easy_steps", "150000",
    "--comfort_coeff", "6.0",
    "--curriculum_replay_ratio", "0.0",
    "--holdout_scenarios", "hard", "circle",
    "--holdout_episodes", "20",
    "--eval_freq_updates", "20",
    "--seed", "7",
]

PROBES = {
    "P1_v18_regime": [
        "--robot_vpref", "0.26", "--human_motion_model", "sfm",
        "--max_time", "50.0", "--lr", "5e-5",
    ],
    "P2_speed_only": [
        "--robot_vpref", "1.0", "--human_motion_model", "sfm",
        "--max_time", "15.0", "--lr", "5e-5",
    ],
    "P3_orca": [
        "--robot_vpref", "1.0", "--human_motion_model", "orca",
        "--max_time", "15.0", "--lr", "5e-5",
    ],
    "P4_v21_core": [
        "--robot_vpref", "1.0", "--human_motion_model", "orca",
        "--human_vpref_override", "1.0", "--human_goal_noise", "2.0",
        "--max_time", "15.0", "--lr", "5e-5",
    ],
    "P5_paper_lr": [
        "--robot_vpref", "1.0", "--human_motion_model", "orca",
        "--human_vpref_override", "1.0", "--human_goal_noise", "2.0",
        "--max_time", "15.0", "--lr", "1e-4",
    ],
}


def newest_training_csv() -> Path | None:
    matches = sorted(LOG_DIR.glob("training_*.csv"))
    return matches[-1] if matches else None


def summarize(csv_path: Path) -> str:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "empty csv"

    def fvals(key, subset):
        out = []
        for r in subset:
            try:
                v = float(r.get(key, "") or "nan")
            except ValueError:
                continue
            if v == v:  # not NaN
                out.append(v)
        return out

    tail = rows[max(0, len(rows) - max(1, len(rows) // 5)):]  # last 20%
    parts = [f"updates={len(rows)}"]
    for key in ("success", "collision", "timeout"):
        vals = fvals(key, tail)
        if vals:
            parts.append(f"train_{key}(son%20)={sum(vals)/len(vals):.0%}")
    for key in ("holdout_hard_success", "holdout_circle_success"):
        vals = fvals(key, rows)
        if vals:
            parts.append(f"{key}: best={max(vals):.0%} final={vals[-1]:.0%}")
    return " | ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES))
    parser.add_argument("--total_steps", type=int, default=500_000)
    args = parser.parse_args(argv)

    results = {}
    for name in args.probes:
        save_path = CKPT_DIR / f"probe_{name}.pt"
        cmd = [
            sys.executable, "-m", "sncp_ppo.train",
            *COMMON, *PROBES[name],
            "--total_steps", str(args.total_steps),
            "--save_path", str(save_path),
        ]
        print(f"\n{'=' * 80}\n[{name}] starting: {' '.join(cmd)}\n{'=' * 80}", flush=True)
        before = newest_training_csv()
        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = (time.time() - t0) / 60.0
        if proc.returncode != 0:
            print(f"[{name}] FAILED (exit {proc.returncode}) after {elapsed:.1f} min", flush=True)
            results[name] = f"FAILED exit {proc.returncode}"
            continue
        produced = newest_training_csv()
        if produced is not None and produced != before:
            target = LOG_DIR / f"probe_{name}.csv"
            produced.replace(target)
            results[name] = summarize(target)
        else:
            results[name] = "no training csv produced"
        print(f"[{name}] done in {elapsed:.1f} min -> {results[name]}", flush=True)

    print(f"\n{'=' * 80}\nPROBE SUMMARY\n{'=' * 80}")
    for name, line in results.items():
        print(f"{name}: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
