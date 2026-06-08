"""Stage Colab-downloaded SNCP-PPO artifacts into repo paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.colab_artifacts import stage_colab_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True, help="Run version, e.g. 17 for v17.")
    parser.add_argument("--staging_dir", type=Path, default=Path("colabout"))
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_extract_eval", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    staged = stage_colab_artifacts(
        staging_dir=args.staging_dir,
        repo_root=args.repo_root,
        version=args.version,
        overwrite=args.overwrite,
        extract_eval_artifacts=not args.no_extract_eval,
    )
    print(f"Checkpoint: {staged.checkpoint_source} -> {staged.checkpoint_path}")
    print(f"Training CSV: {staged.training_csv_source} -> {staged.training_csv_path}")
    if staged.eval_source is not None and staged.eval_dir is not None:
        print(f"Eval artifacts: {staged.eval_source} -> {staged.eval_dir}")
    else:
        print("Eval artifacts: not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
