"""Run the complete v17 post-run review and pre-v18 evidence gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.v17_review_pipeline import V17ReviewResult, run_v17_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage_colab", action="store_true")
    parser.add_argument("--staging_dir", type=Path, default=Path("colabout"))
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--baseline_json", type=Path, default=Path("eval_v15/density_sweep.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_v17_review(
        staging_dir=args.staging_dir,
        repo_root=args.repo_root,
        stage_artifacts=args.stage_colab,
        overwrite=args.overwrite,
        checkpoint_path=args.checkpoint,
        training_csv=args.training_csv,
        output_dir=args.output_dir,
        baseline_json=args.baseline_json,
    )

    print(f"Overall status: {result.status}")
    print(f"Post-eval artifact status: {result.post_eval_status}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Training CSV: {result.training_csv}")
    print(f"Output dir: {result.output_dir}")
    print(f"Branch: {result.branch_id or 'n/a'}")
    print(f"Single variable: {result.single_variable or 'n/a'}")
    print(f"Decision report: {result.decision_report}")
    print(f"Gate report: {result.gate_report}")
    return 0 if result.status == "pass" else 1


__all__ = ["V17ReviewResult", "build_parser", "main", "run_v17_review"]


if __name__ == "__main__":
    raise SystemExit(main())
