"""Select the next v18 candidate from completed v17 evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.v18_decision import (
    V18Decision,
    select_v18_candidate,
    write_v18_decision_json,
    write_v18_decision_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a v18 single-variable decision report from density and training diagnostics."
    )
    parser.add_argument("--version", type=int, default=17)
    parser.add_argument("--eval_dir", type=Path, default=None)
    parser.add_argument("--density_json", type=Path, default=None)
    parser.add_argument("--training_json", type=Path, default=None)
    parser.add_argument("--baseline_json", type=Path, default=Path("eval_v15/density_sweep.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json_output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_dir = args.eval_dir or Path(f"eval_v{args.version}")
    density_json = args.density_json or eval_dir / "density_sweep.json"
    training_json = args.training_json or eval_dir / "training_diagnostics.json"
    output = args.output or eval_dir / "v18_decision.md"
    json_output = args.json_output or eval_dir / "v18_decision.json"

    decision = select_v18_candidate(
        density_json,
        baseline_json=args.baseline_json,
        training_json=training_json,
    )
    write_v18_decision_report(decision, output)
    write_v18_decision_json(decision, json_output)

    print(f"Status: {decision.status}")
    print(f"Branch: {decision.branch_id}")
    print(f"Single variable: {decision.single_variable}")
    print(f"Report: {output}")
    print(f"JSON: {json_output}")
    return 0 if decision.status in {"ready_for_review", "needs_manual_review"} else 1


__all__ = ["V18Decision", "build_parser", "main", "select_v18_candidate"]


if __name__ == "__main__":
    raise SystemExit(main())
