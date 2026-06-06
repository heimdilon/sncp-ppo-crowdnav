"""Verify that a completed v16 run produced the required artifact set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.artifact_verifier import (
    verify_v16_artifacts,
    write_artifact_verification_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check v16 checkpoint/evaluation/training diagnostic artifacts."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sncp_ppo_v16.pt"))
    parser.add_argument("--eval_dir", type=Path, default=Path("eval_v16"))
    parser.add_argument("--output", type=Path, default=Path("eval_v16/artifact_verification.md"))
    parser.add_argument("--min_episodes", type=int, default=50)
    parser.add_argument("--expected_replay_ratio", type=float, default=0.20)
    parser.add_argument("--replay_tolerance", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = verify_v16_artifacts(
        checkpoint_path=args.checkpoint,
        eval_dir=args.eval_dir,
        min_episodes=args.min_episodes,
        expected_replay_ratio=args.expected_replay_ratio,
        replay_tolerance=args.replay_tolerance,
    )
    write_artifact_verification_report(summary, args.output)
    print(f"Overall status: {summary.status}")
    print(f"Wrote artifact_verification.md: {args.output}")
    return 1 if summary.status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
