"""Analyze paired episode-level v37 C0/C1 probe artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sncp_ppo.v37_probe import DENSITIES, analyze_probe_runs  # noqa: E402


def _write_report(result: dict, path: Path) -> None:
    lines = [
        "# v37 Paired Probe Decision",
        "",
        f"Verdict: **{result['verdict']}**",
        "",
        "| N | C0 success | C1 success | Δ success | C0 collision | C1 collision | Δ collision | C1 timeout |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for density in DENSITIES:
        key = str(density)
        c0 = result["rates"].get("c0", {}).get(key, {})
        c1 = result["rates"].get("c1", {}).get(key, {})
        if not c0 or not c1:
            continue
        lines.append(
            f"| {density} | {c0['success']:.1%} | {c1['success']:.1%} | "
            f"{c1['success'] - c0['success']:+.1%} | {c0['collision']:.1%} | "
            f"{c1['collision']:.1%} | {c1['collision'] - c0['collision']:+.1%} | "
            f"{c1['timeout']:.1%} |"
        )
    lines.extend([
        "",
        "## Gate checks",
        "",
        f"- Complete paired bank: {result['complete']}",
        f"- High-N success delta: {result['high_n_success_delta']}",
        f"- High-N collision delta: {result['high_n_collision_delta']}",
        f"- Direction-matching seeds: {result['direction_seed_count']}/3",
        f"- No low-N regression: {not result['low_n_regression']}",
        f"- Timeout zero: {result['timeout_zero']}",
        f"- HH gate active: {result['gate_active']}",
        f"- Diagnostics healthy: {result['diagnostics_healthy']}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("eval_v37_probe"))
    args = parser.parse_args(argv)
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_dir.glob("*_episodes.json"))
    ]
    result = analyze_probe_runs(runs)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    (args.input_dir / "verdict.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(result, args.input_dir / "report.md")
    print(json.dumps({key: result[key] for key in (
        "verdict", "high_n_success_delta", "high_n_collision_delta",
        "direction_seed_count", "low_n_regression", "timeout_zero",
        "gate_active", "diagnostics_healthy",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
