"""Run the v38 training-free action-shield probe.

This evaluates the same checkpoint twice on the same deterministic episode bank:

* C0: raw deterministic policy action.
* C1: policy action post-processed by the v38 safety shield.

No PPO training is performed.
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from sncp_ppo.eval_report import evaluate_density, summarize_density


def _row(summary, arm: str) -> dict:
    data = asdict(summary)
    data["arm"] = arm
    return data


def _by_density(rows: list[dict], arm: str) -> dict[int, dict]:
    return {int(row["num_humans"]): row for row in rows if row["arm"] == arm}


def analyze_rows(rows: list[dict], high_densities: tuple[int, ...] = (15, 20)) -> dict:
    base = _by_density(rows, "c0")
    shield = _by_density(rows, "c1")
    complete = all(n in base and n in shield for n in high_densities)
    if not complete:
        return {
            "verdict": "NO-GO",
            "complete": False,
            "reason": "missing high-density paired rows",
        }

    success_delta = mean(shield[n]["success_rate"] - base[n]["success_rate"] for n in high_densities)
    collision_delta = mean(shield[n]["collision_rate"] - base[n]["collision_rate"] for n in high_densities)
    timeout_delta = mean(shield[n]["timeout_rate"] - base[n]["timeout_rate"] for n in high_densities)
    low_regression = any(
        (
            shield[n]["success_rate"] - base[n]["success_rate"] < -0.02
            or shield[n]["collision_rate"] - base[n]["collision_rate"] > 0.02
        )
        for n in base
        if n < min(high_densities) and n in shield
    )
    verdict = "GO" if (
        collision_delta <= -0.03
        and success_delta >= -0.02
        and timeout_delta <= 0.02
        and not low_regression
    ) else "NO-GO"
    return {
        "verdict": verdict,
        "complete": True,
        "high_densities": list(high_densities),
        "high_n_success_delta": round(success_delta, 10),
        "high_n_collision_delta": round(collision_delta, 10),
        "high_n_timeout_delta": round(timeout_delta, 10),
        "low_n_regression": low_regression,
    }


def write_report(rows: list[dict], verdict: dict, path: Path) -> None:
    base = _by_density(rows, "c0")
    shield = _by_density(rows, "c1")
    lines = [
        "# v38 Action Shield Probe",
        "",
        f"Verdict: **{verdict['verdict']}**",
        "",
        "| N | C0 success | C1 success | Δ success | C0 collision | C1 collision | Δ collision | C0 timeout | C1 timeout |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(base):
        if n not in shield:
            continue
        c0 = base[n]
        c1 = shield[n]
        lines.append(
            f"| {n} | {c0['success_rate']:.1%} | {c1['success_rate']:.1%} | "
            f"{c1['success_rate'] - c0['success_rate']:+.1%} | "
            f"{c0['collision_rate']:.1%} | {c1['collision_rate']:.1%} | "
            f"{c1['collision_rate'] - c0['collision_rate']:+.1%} | "
            f"{c0['timeout_rate']:.1%} | {c1['timeout_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Gate checks",
        "",
        f"- Complete paired high-N rows: {verdict.get('complete')}",
        f"- High-N success delta: {verdict.get('high_n_success_delta')}",
        f"- High-N collision delta: {verdict.get('high_n_collision_delta')}",
        f"- High-N timeout delta: {verdict.get('high_n_timeout_delta')}",
        f"- Low-N regression: {verdict.get('low_n_regression')}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("sncp_ppo_v34.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("eval_v38_shield_probe"))
    parser.add_argument("--densities", type=int, nargs="+", default=[15, 20])
    parser.add_argument("--scenario", default="paper_challenging")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--robot_vpref", type=float, default=1.0)
    parser.add_argument("--human_vpref_override", type=float, default=1.0)
    parser.add_argument("--shield_horizon_steps", type=int, default=6)
    parser.add_argument("--shield_safety_margin", type=float, default=0.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for density in args.densities:
        for arm, enabled in (("c0", False), ("c1", True)):
            episodes = evaluate_density(
                checkpoint_path=args.checkpoint,
                num_humans=density,
                scenario=args.scenario,
                n_episodes=args.n_episodes,
                seed=args.seed,
                robot_vpref=args.robot_vpref,
                human_vpref_override=args.human_vpref_override,
                max_time=None,
                action_shield=enabled,
                shield_horizon_steps=args.shield_horizon_steps,
                shield_safety_margin=args.shield_safety_margin,
            )
            summary = summarize_density(density, args.scenario, episodes)
            rows.append(_row(summary, arm))
            print(
                f"{arm} N={density}: success={summary.success_rate:.1%} "
                f"collision={summary.collision_rate:.1%} timeout={summary.timeout_rate:.1%}",
                flush=True,
            )

    high_densities = tuple(n for n in (15, 20) if n in args.densities) or tuple(args.densities)
    verdict = analyze_rows(rows, high_densities=high_densities)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(rows, verdict, args.output_dir / "report.md")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
