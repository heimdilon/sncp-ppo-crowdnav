"""Check that the repo is ready for the current Colab experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sncp_ppo.run_readiness import verify_v16_run_ready, write_readiness_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight-check current Colab run configuration.")
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("eval_v21/run_readiness.md"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = verify_v16_run_ready(args.repo_root)
    write_readiness_report(summary, args.output)
    print(f"Overall status: {summary.status}")
    print(f"Wrote run_readiness.md: {args.output}")
    return 1 if summary.status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
