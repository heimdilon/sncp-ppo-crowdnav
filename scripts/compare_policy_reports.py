"""Compare a candidate SNCP-PPO density sweep against a baseline report."""

from __future__ import annotations

import os as _os, sys as _sys  # repo-root path bootstrap (run standalone: python scripts/X.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.eval_report import (
    compare_density_sweeps,
    load_summary_json,
    write_comparison_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two density_sweep.json files and write a gate report."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("comparison_vs_baseline.md"))
    parser.add_argument("--baseline_nav_steps", type=float, default=121.5)
    parser.add_argument("--nav_margin_steps", type=float, default=30.0)
    parser.add_argument("--success_tolerance", type=float, default=0.05)
    parser.add_argument("--collision_tolerance", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = compare_density_sweeps(
        load_summary_json(args.baseline),
        load_summary_json(args.candidate),
        baseline_nav_steps=args.baseline_nav_steps,
        nav_margin_steps=args.nav_margin_steps,
        success_tolerance=args.success_tolerance,
        collision_tolerance=args.collision_tolerance,
    )
    write_comparison_report(
        comparison,
        args.output,
        baseline_path=args.baseline,
        candidate_path=args.candidate,
    )
    print(f"Overall verdict: {comparison.overall_status}")
    print(f"Wrote comparison report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
