"""Create a density-sweep evaluation report for a trained SNCP-PPO checkpoint."""

from __future__ import annotations

import os as _os, sys as _sys  # repo-root path bootstrap (run standalone: python scripts/X.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.eval_report import run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic density sweep and write CSV/JSON/PNG/Markdown "
            "artifacts for real-avoidance evaluation."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("eval_v16"))
    parser.add_argument("--densities", type=int, nargs="+", default=[1, 3, 5, 8, 10])
    parser.add_argument("--scenario", type=str, default="hard")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--trajectory_densities", type=int, nargs="*", default=[5, 10])
    parser.add_argument("--robot_vpref", type=float, default=0.26,
                        help="Robot max speed; paper-regime eval uses 1.0.")
    parser.add_argument("--human_vpref_override", type=float, default=None,
                        help="If set, force a flat pedestrian speed such as 1.0.")
    parser.add_argument("--max_time", type=float, default=None,
                        help="Episode time cap; None lets the env resolve scenario defaults.")
    parser.add_argument("--human_goal_noise", type=float, default=0.0,
                        help="Pedestrian goal noise; match the evaluated regime.")
    parser.add_argument(
        "--baseline_nav_steps",
        type=float,
        default=121.5,
        help="v14 straight-line beeline baseline used in the nav-time plot.",
    )
    parser.add_argument("--action_shield", action="store_true",
                        help="Apply the v38 training-free action safety shield during eval.")
    parser.add_argument("--shield_horizon_steps", type=int, default=6)
    parser.add_argument("--shield_safety_margin", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = run_report(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        densities=args.densities,
        scenario=args.scenario,
        n_episodes=args.n_episodes,
        seed=args.seed,
        trajectory_densities=args.trajectory_densities,
        baseline_nav_steps=args.baseline_nav_steps,
        robot_vpref=args.robot_vpref,
        human_vpref_override=args.human_vpref_override,
        max_time=args.max_time,
        human_goal_noise=args.human_goal_noise,
        action_shield=args.action_shield,
        shield_horizon_steps=args.shield_horizon_steps,
        shield_safety_margin=args.shield_safety_margin,
    )

    print("Wrote evaluation artifacts:")
    for key, value in artifacts.items():
        if isinstance(value, list):
            for path in value:
                print(f"  {key}: {path}")
        else:
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
