"""Run a version-aware post-training evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.post_run_pipeline import (
    PostRunResult,
    find_latest_training_csv,
    run_v16_post_eval as run_post_eval,
)


def _checkpoint_for_version(version: int) -> Path:
    return Path(f"checkpoints/sncp_ppo_v{version}.pt")


def _output_dir_for_version(version: int) -> Path:
    return Path(f"eval_v{version}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evaluation, v15 comparison, training diagnostics, and artifact "
            "verification for a numbered SNCP-PPO experiment."
        )
    )
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training_csv", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--baseline_json", type=Path, default=Path("eval_v15/density_sweep.json"))
    parser.add_argument("--densities", type=int, nargs="+", default=[1, 3, 5, 8, 10])
    parser.add_argument("--scenario", type=str, default="hard")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--trajectory_densities", type=int, nargs="*", default=[5, 10])
    parser.add_argument("--robot_vpref", type=float, default=0.26,
                        help="Robot max speed; the paper-reproduction run uses 1.0 to match the paper.")
    parser.add_argument("--human_vpref_override", type=float, default=None,
                        help="If set, force a flat pedestrian speed (parity regime, e.g. 1.0).")
    parser.add_argument("--max_time", type=float, default=None,
                        help="Episode time cap for eval; None lets the env resolve it "
                             "(paper scenarios -> 12.5s, else 50.0). Match the regime.")
    parser.add_argument("--human_goal_noise", type=float, default=0.0,
                        help="Pedestrian goal noise; match the training regime (paper run uses ~2.0).")
    parser.add_argument("--baseline_nav_steps", type=float, default=121.5,
                        help="Beeline reference (successful nav steps) for the no-beeline gate. "
                             "121.5 fits the 0.26 m/s robot; the 1.0 m/s paper regime uses ~32 "
                             "(v21 lesson: the gate is regime-dependent).")
    parser.add_argument("--nav_margin_steps", type=float, default=30.0,
                        help="Required margin above the beeline reference. Scale together with "
                             "--baseline_nav_steps (paper regime: ~8 with a 60-step episode cap).")
    parser.add_argument("--expected_replay_ratio", type=float, default=0.20)
    parser.add_argument("--replay_tolerance", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint or _checkpoint_for_version(args.version)
    output_dir = args.output_dir or _output_dir_for_version(args.version)
    training_csv = args.training_csv or find_latest_training_csv(args.log_dir)

    result = run_post_eval(
        checkpoint_path=checkpoint,
        training_csv=training_csv,
        output_dir=output_dir,
        baseline_json=args.baseline_json,
        densities=args.densities,
        scenario=args.scenario,
        n_episodes=args.n_episodes,
        seed=args.seed,
        trajectory_densities=args.trajectory_densities,
        robot_vpref=args.robot_vpref,
        human_vpref_override=args.human_vpref_override,
        max_time=args.max_time,
        human_goal_noise=args.human_goal_noise,
        baseline_nav_steps=args.baseline_nav_steps,
        nav_margin_steps=args.nav_margin_steps,
        expected_replay_ratio=args.expected_replay_ratio,
        replay_tolerance=args.replay_tolerance,
    )

    print(f"Overall status: {result.status}")
    print(f"Output dir: {result.output_dir}")
    print(f"Comparison report: {result.comparison_report}")
    print(f"Training report: {result.training_report}")
    print(f"Artifact report: {result.artifact_report}")
    return 1 if result.status == "fail" else 0


__all__ = [
    "PostRunResult",
    "build_parser",
    "find_latest_training_csv",
    "main",
    "run_post_eval",
]


if __name__ == "__main__":
    raise SystemExit(main())
