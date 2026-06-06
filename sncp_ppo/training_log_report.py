"""Training-log diagnostics for SNCP-PPO curriculum runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TrainingLogSummary:
    csv_path: str
    total_rows: int
    evaluated_rows: int
    holdout_scenarios: tuple[str, ...]
    observed_replay_ratio: float | None
    best_step: int
    best_min_success: float
    best_success_by_scenario: dict[str, float]
    best_reason: str
    final_step: int
    final_min_success: float
    final_success_by_scenario: dict[str, float]
    collapse_delta: float
    collapse_detected: bool


def _parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _holdout_scenarios(fieldnames: Sequence[str]) -> tuple[str, ...]:
    prefix = "holdout_"
    suffix = "_success"
    scenarios = []
    for name in fieldnames:
        if name.startswith(prefix) and name.endswith(suffix):
            scenarios.append(name[len(prefix) : -len(suffix)])
    return tuple(scenarios)


def _success_by_scenario(row: dict[str, str], scenarios: Sequence[str]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for scenario in scenarios:
        value = _parse_float(row.get(f"holdout_{scenario}_success"))
        if math.isnan(value):
            return None
        values[scenario] = value
    return values


def _row_step(row: dict[str, str]) -> int:
    return int(float(row.get("episode") or row.get("step") or 0))


def _row_is_replay(row: dict[str, str]) -> bool | None:
    if "is_replay_update" not in row:
        return None
    value = row.get("is_replay_update", "")
    if value == "":
        return None
    return bool(int(float(value)))


def analyze_training_log(
    csv_path: str | Path,
    *,
    collapse_threshold: float = 0.20,
) -> TrainingLogSummary:
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    scenarios = _holdout_scenarios(fieldnames)
    if not scenarios:
        raise ValueError("training log has no holdout_*_success columns")

    evaluated = []
    replay_values = []
    for row in rows:
        replay = _row_is_replay(row)
        if replay is not None:
            replay_values.append(replay)

        successes = _success_by_scenario(row, scenarios)
        if successes is None:
            continue
        min_success = min(successes.values())
        evaluated.append((row, successes, min_success))

    if not evaluated:
        raise ValueError("training log has no evaluated holdout rows")

    best_row, best_successes, best_min_success = max(evaluated, key=lambda item: item[2])
    final_row, final_successes, final_min_success = evaluated[-1]
    collapse_delta = final_min_success - best_min_success

    observed_replay_ratio = None
    if replay_values:
        observed_replay_ratio = sum(replay_values) / len(replay_values)

    return TrainingLogSummary(
        csv_path=str(csv_path),
        total_rows=len(rows),
        evaluated_rows=len(evaluated),
        holdout_scenarios=tuple(scenarios),
        observed_replay_ratio=observed_replay_ratio,
        best_step=_row_step(best_row),
        best_min_success=best_min_success,
        best_success_by_scenario=best_successes,
        best_reason=best_row.get("best_reason", ""),
        final_step=_row_step(final_row),
        final_min_success=final_min_success,
        final_success_by_scenario=final_successes,
        collapse_delta=collapse_delta,
        collapse_detected=collapse_delta <= -collapse_threshold,
    )


def write_training_diagnostic_json(summary: TrainingLogSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")


def _format_rate(value: float) -> str:
    return f"{value:.1%}"


def write_training_diagnostic_report(summary: TrainingLogSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    replay = (
        "not logged"
        if summary.observed_replay_ratio is None
        else _format_rate(summary.observed_replay_ratio)
    )
    lines = [
        "# SNCP-PPO Training Diagnostics",
        "",
        f"CSV: `{summary.csv_path}`",
        f"Rows: {summary.total_rows}",
        f"Evaluated holdout rows: {summary.evaluated_rows}",
        f"Observed replay ratio: {replay}",
        "",
        "## Best Holdout",
        "",
        f"Best step: {summary.best_step}",
        f"Best min success: {_format_rate(summary.best_min_success)}",
        f"Best reason: {summary.best_reason or 'n/a'}",
        "",
        "## Final Holdout",
        "",
        f"Final step: {summary.final_step}",
        f"Final min success: {_format_rate(summary.final_min_success)}",
        f"Collapse delta: {summary.collapse_delta:+.1%}",
        f"Collapse detected: {'yes' if summary.collapse_detected else 'no'}",
        "",
        "## Per-Scenario Success",
        "",
        "| Scenario | Best | Final | Delta |",
        "|---|---:|---:|---:|",
    ]
    for scenario in summary.holdout_scenarios:
        best = summary.best_success_by_scenario[scenario]
        final = summary.final_success_by_scenario[scenario]
        lines.append(
            f"| {scenario} | {_format_rate(best)} | {_format_rate(final)} | {final - best:+.1%} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
