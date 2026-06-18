"""Verify that completed v17 artifacts are ready for v18 experiment review."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.v18_gate import V18GateSummary, verify_v18_decision_ready, write_v18_gate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check v17 artifacts and v18 decision files before editing or launching v18."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sncp_ppo_v17.pt"))
    parser.add_argument("--eval_dir", type=Path, default=Path("eval_v17"))
    parser.add_argument("--output", type=Path, default=Path("eval_v17/v18_ready.md"))
    parser.add_argument("--min_episodes", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = verify_v18_decision_ready(
        checkpoint_path=args.checkpoint,
        eval_dir=args.eval_dir,
        min_episodes=args.min_episodes,
    )
    write_v18_gate_report(summary, args.output)

    print(f"Overall status: {summary.status}")
    print(f"Branch: {summary.branch_id or 'n/a'}")
    print(f"Single variable: {summary.single_variable or 'n/a'}")
    print(f"Report: {args.output}")
    return 0 if summary.status == "pass" else 1


__all__ = ["V18GateSummary", "build_parser", "main", "verify_v18_decision_ready"]


if __name__ == "__main__":
    raise SystemExit(main())
