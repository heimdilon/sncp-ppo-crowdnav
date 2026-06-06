"""Create a density-sweep evaluation report for a trained SNCP-PPO checkpoint."""

from __future__ import annotations

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
    parser.add_argument(
        "--baseline_nav_steps",
        type=float,
        default=121.5,
        help="v14 straight-line beeline baseline used in the nav-time plot.",
    )
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
