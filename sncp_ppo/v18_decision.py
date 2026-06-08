"""Evidence-based v18 candidate selection after a completed v17 evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


BEELINE_NAV_STEPS = 121.5


@dataclass(frozen=True)
class V18Decision:
    branch_id: str
    title: str
    status: str
    single_variable: str
    reasons: tuple[str, ...]
    manual_checks: tuple[str, ...]
    metrics: dict[str, Any]


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _density_rows(path: str | Path) -> dict[int, dict[str, Any]]:
    data = _load_json(path)
    rows = data.get("density_sweep", [])
    return {int(row["num_humans"]): row for row in rows}


def _rate(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return 0.0 if value is None else float(value)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _pp(value: float) -> str:
    return f"{value * 100:.1f} pp"


def _nav_margin(rows: Mapping[int, Mapping[str, Any]], beeline_steps: float) -> float | None:
    margins = []
    for row in rows.values():
        steps = row.get("avg_success_steps")
        if steps is not None:
            margins.append(float(steps) - beeline_steps)
    return min(margins) if margins else None


def _scenario_drops(training: Mapping[str, Any]) -> dict[str, float]:
    best = training.get("best_success_by_scenario") or {}
    final = training.get("final_success_by_scenario") or {}
    drops = {}
    for scenario in ("easy", "hard"):
        if scenario in best and scenario in final:
            drops[scenario] = float(best[scenario]) - float(final[scenario])
    return drops


def _std_stable(training: Mapping[str, Any]) -> bool:
    max_linear = training.get("max_std_linear")
    max_angular = training.get("max_std_angular")
    if max_linear is None or max_angular is None:
        return True
    return float(max_linear) < 0.35 and float(max_angular) < 0.45


def _base_metrics(
    candidate: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[int, Mapping[str, Any]],
    training: Mapping[str, Any],
    beeline_steps: float,
) -> dict[str, Any]:
    nav_margin = _nav_margin(candidate, beeline_steps)
    return {
        "densities": sorted(candidate),
        "beeline_nav_steps": beeline_steps,
        "min_nav_margin": nav_margin,
        "observed_replay_ratio": training.get("observed_replay_ratio"),
        "collapse_detected": training.get("collapse_detected"),
        "candidate": {
            str(n): {
                "success": _rate(row, "success_rate"),
                "collision": _rate(row, "collision_rate"),
                "timeout": _rate(row, "timeout_rate"),
                "avg_success_steps": row.get("avg_success_steps"),
                "avg_i_sp": row.get("avg_i_sp"),
            }
            for n, row in sorted(candidate.items())
        },
        "baseline_success": {
            str(n): _rate(row, "success_rate") for n, row in sorted(baseline.items())
        },
    }


def _decision(
    *,
    branch_id: str,
    title: str,
    single_variable: str,
    reasons: list[str],
    metrics: dict[str, Any],
    status: str = "ready_for_review",
) -> V18Decision:
    return V18Decision(
        branch_id=branch_id,
        title=title,
        status=status,
        single_variable=single_variable,
        reasons=tuple(reasons),
        manual_checks=(
            "Inspect N=5/N=10 trajectory plots before launching A100.",
            "Confirm nav-time stays above the v14 beeline reference and I_sp stays low.",
        ),
        metrics=metrics,
    )


def select_v18_candidate(
    candidate_json: str | Path,
    *,
    baseline_json: str | Path = "eval_v15/density_sweep.json",
    training_json: str | Path | None = None,
    beeline_steps: float = BEELINE_NAV_STEPS,
) -> V18Decision:
    candidate = _density_rows(candidate_json)
    baseline = _density_rows(baseline_json)
    training = _load_json(training_json)
    metrics = _base_metrics(candidate, baseline, training, beeline_steps)

    required = {1, 3, 5, 8, 10}
    missing = sorted(required - set(candidate))
    if missing:
        return _decision(
            branch_id="WAIT_FOR_ARTIFACTS",
            title="Missing v17 Density Sweep Evidence",
            single_variable="none",
            reasons=[f"Missing density rows for N={missing}; run post-evaluation first."],
            metrics=metrics,
            status="wait_for_artifacts",
        )

    nav_margin = metrics["min_nav_margin"]
    high_i_sp = any(_rate(row, "avg_i_sp") >= 0.04 for row in candidate.values())
    if nav_margin is not None and (nav_margin <= 15.0 or high_i_sp):
        reasons = []
        if nav_margin <= 15.0:
            reasons.append(
                f"Minimum nav margin is only {nav_margin:.1f} steps above the {beeline_steps:.1f}-step beeline reference."
            )
        if high_i_sp:
            reasons.append("At least one density has avg I_sp >= 0.040, indicating damaged social-distance behavior.")
        return _decision(
            branch_id="D_STOP_COMFORT_RELAXATION",
            title="Comfort Relaxation Damaged Avoidance",
            single_variable="Do not run comfort_coeff 4.0",
            reasons=reasons,
            metrics=metrics,
        )

    drops = _scenario_drops(training)
    forgetting = {name: drop for name, drop in drops.items() if drop >= 0.20}
    if forgetting and _std_stable(training):
        reasons = [
            f"{name} dropped by {_pp(drop)} from best to final holdout while policy std stayed controlled."
            for name, drop in sorted(forgetting.items())
        ]
        replay = training.get("observed_replay_ratio")
        if replay is not None:
            reasons.append(f"Observed replay ratio was {_pct(float(replay))}.")
        return _decision(
            branch_id="C_REPLAY30",
            title="Easy/Hard Forgetting Dominant",
            single_variable="curriculum_replay_ratio 0.20 -> 0.30",
            reasons=reasons,
            metrics=metrics,
        )

    sparse_timeout_reasons = []
    for n in (1, 3):
        row = candidate[n]
        timeout = _rate(row, "timeout_rate")
        collision = _rate(row, "collision_rate")
        if timeout >= 0.35 and collision <= 0.15 and timeout >= collision + 0.20:
            sparse_timeout_reasons.append(
                f"N={n} timeout {_pct(timeout)} dominates collision {_pct(collision)} with low I_sp {_rate(row, 'avg_i_sp'):.4f}."
            )
    if sparse_timeout_reasons:
        return _decision(
            branch_id="A_TIMEOUT_MAX_TIME60",
            title="Timeout / Slow-Detour Dominant",
            single_variable="max_time 50 -> 60",
            reasons=sparse_timeout_reasons,
            metrics=metrics,
        )

    high_density_reasons = []
    low_density_ok = all(_rate(candidate[n], "success_rate") >= 0.50 for n in (1, 3))
    for n in (8, 10):
        row = candidate[n]
        collision = _rate(row, "collision_rate")
        timeout = _rate(row, "timeout_rate")
        if collision >= 0.35 and collision >= timeout + 0.15:
            high_density_reasons.append(
                f"N={n} collision {_pct(collision)} dominates timeout {_pct(timeout)}."
            )
    if low_density_ok and high_density_reasons:
        return _decision(
            branch_id="B_HIGH_DENSITY_EXPOSURE",
            title="High-Density Collision Dominant",
            single_variable="increase high-density training exposure",
            reasons=high_density_reasons,
            metrics=metrics,
        )

    return _decision(
        branch_id="NEEDS_MANUAL_REVIEW",
        title="Mixed Failure Profile",
        single_variable="none until reviewed",
        reasons=["The density sweep does not cleanly match one v18 decision branch."],
        metrics=metrics,
        status="needs_manual_review",
    )


def write_v18_decision_json(decision: V18Decision, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(decision), indent=2) + "\n", encoding="utf-8")


def write_v18_decision_report(decision: V18Decision, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v18 Candidate Decision",
        "",
        f"Status: {decision.status}",
        f"Branch: {decision.branch_id}",
        f"Title: {decision.title}",
        f"Single variable: {decision.single_variable}",
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision.reasons)
    lines.extend(
        [
            "",
            "## Manual Checks",
            "",
        ]
    )
    lines.extend(f"- {check}" for check in decision.manual_checks)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "BEELINE_NAV_STEPS",
    "V18Decision",
    "select_v18_candidate",
    "write_v18_decision_json",
    "write_v18_decision_report",
]
