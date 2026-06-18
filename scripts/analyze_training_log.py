"""Create holdout-collapse diagnostics from an SNCP-PPO training CSV."""

from __future__ import annotations

import os as _os, sys as _sys  # repo-root path bootstrap (run standalone: python scripts/X.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.training_log_report import (
    analyze_training_log,
    write_training_diagnostic_json,
    write_training_diagnostic_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize best/final holdout and replay diagnostics from a training CSV."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("training_diagnostics"))
    parser.add_argument("--collapse_threshold", type=float, default=0.20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze_training_log(args.csv, collapse_threshold=args.collapse_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "training_diagnostics.json"
    report_path = args.output_dir / "training_diagnostics.md"
    write_training_diagnostic_json(summary, json_path)
    write_training_diagnostic_report(summary, report_path)

    print(f"Best min success: {summary.best_min_success:.1%} @ step {summary.best_step}")
    print(f"Final min success: {summary.final_min_success:.1%} @ step {summary.final_step}")
    print(f"Collapse detected: {'yes' if summary.collapse_detected else 'no'}")
    print(f"Wrote training_diagnostics.json: {json_path}")
    print(f"Wrote training_diagnostics.md: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
