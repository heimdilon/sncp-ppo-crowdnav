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
    best_collision_by_scenario: dict[str, float | None]
    best_timeout_by_scenario: dict[str, float | None]
    best_reward_by_scenario: dict[str, float | None]
    best_avg_steps_by_scenario: dict[str, float | None]
    best_avg_I_sp_by_scenario: dict[str, float | None]
    best_min_d_min_by_scenario: dict[str, float | None]
    best_reason: str
    final_step: int
    final_min_success: float
    final_success_by_scenario: dict[str, float]
    final_collision_by_scenario: dict[str, float | None]
    final_timeout_by_scenario: dict[str, float | None]
    final_reward_by_scenario: dict[str, float | None]
    final_avg_steps_by_scenario: dict[str, float | None]
    final_avg_I_sp_by_scenario: dict[str, float | None]
    final_min_d_min_by_scenario: dict[str, float | None]
    collapse_delta: float
    collapse_detected: bool
    final_std_linear: float | None
    final_std_angular: float | None
    max_std_linear: float | None
    max_std_angular: float | None
    std_linear_delta: float | None
    std_angular_delta: float | None


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


def _metric_by_scenario(
    row: dict[str, str],
    scenarios: Sequence[str],
    suffix: str,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for scenario in scenarios:
        value = _parse_float(row.get(f"holdout_{scenario}_{suffix}"))
        values[scenario] = None if math.isnan(value) else value
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


def _std_diagnostics(rows: Sequence[dict[str, str]]) -> tuple[float | None, ...]:
    values = []
    for row in rows:
        std_linear = _parse_float(row.get("std_linear"))
        std_angular = _parse_float(row.get("std_angular"))
        if math.isnan(std_linear) or math.isnan(std_angular):
            continue
        values.append((std_linear, std_angular))

    if not values:
        return (None, None, None, None, None, None)

    first_linear, first_angular = values[0]
    final_linear, final_angular = values[-1]
    max_linear = max(item[0] for item in values)
    max_angular = max(item[1] for item in values)
    return (
        final_linear,
        final_angular,
        max_linear,
        max_angular,
        final_linear - first_linear,
        final_angular - first_angular,
    )


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
    (
        final_std_linear,
        final_std_angular,
        max_std_linear,
        max_std_angular,
        std_linear_delta,
        std_angular_delta,
    ) = _std_diagnostics(rows)

    return TrainingLogSummary(
        csv_path=str(csv_path),
        total_rows=len(rows),
        evaluated_rows=len(evaluated),
        holdout_scenarios=tuple(scenarios),
        observed_replay_ratio=observed_replay_ratio,
        best_step=_row_step(best_row),
        best_min_success=best_min_success,
        best_success_by_scenario=best_successes,
        best_collision_by_scenario=_metric_by_scenario(best_row, scenarios, "collision"),
        best_timeout_by_scenario=_metric_by_scenario(best_row, scenarios, "timeout"),
        best_reward_by_scenario=_metric_by_scenario(best_row, scenarios, "reward"),
        best_avg_steps_by_scenario=_metric_by_scenario(best_row, scenarios, "avg_steps"),
        best_avg_I_sp_by_scenario=_metric_by_scenario(best_row, scenarios, "avg_I_sp"),
        best_min_d_min_by_scenario=_metric_by_scenario(best_row, scenarios, "min_d_min"),
        best_reason=best_row.get("best_reason", ""),
        final_step=_row_step(final_row),
        final_min_success=final_min_success,
        final_success_by_scenario=final_successes,
        final_collision_by_scenario=_metric_by_scenario(final_row, scenarios, "collision"),
        final_timeout_by_scenario=_metric_by_scenario(final_row, scenarios, "timeout"),
        final_reward_by_scenario=_metric_by_scenario(final_row, scenarios, "reward"),
        final_avg_steps_by_scenario=_metric_by_scenario(final_row, scenarios, "avg_steps"),
        final_avg_I_sp_by_scenario=_metric_by_scenario(final_row, scenarios, "avg_I_sp"),
        final_min_d_min_by_scenario=_metric_by_scenario(final_row, scenarios, "min_d_min"),
        collapse_delta=collapse_delta,
        collapse_detected=collapse_delta <= -collapse_threshold,
        final_std_linear=final_std_linear,
        final_std_angular=final_std_angular,
        max_std_linear=max_std_linear,
        max_std_angular=max_std_angular,
        std_linear_delta=std_linear_delta,
        std_angular_delta=std_angular_delta,
    )


def write_training_diagnostic_json(summary: TrainingLogSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")


def _format_rate(value: float) -> str:
    return f"{value:.1%}"


def _format_optional_rate(value: float | None) -> str:
    return "n/a" if value is None else _format_rate(value)


def _format_optional_float(value: float | None) -> str:
    return "not logged" if value is None else f"{value:.3f}"


def _format_table_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


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
        "## PPO Stability",
        "",
        f"Final std linear: {_format_optional_float(summary.final_std_linear)}",
        f"Final std angular: {_format_optional_float(summary.final_std_angular)}",
        f"Max std linear: {_format_optional_float(summary.max_std_linear)}",
        f"Max std angular: {_format_optional_float(summary.max_std_angular)}",
        f"Std linear delta: {_format_optional_float(summary.std_linear_delta)}",
        f"Std angular delta: {_format_optional_float(summary.std_angular_delta)}",
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

    lines.extend(
        [
            "",
            "## Per-Scenario Failure Profile",
            "",
            "| Scenario | Final success | Final collision | Final timeout | Final avg steps | Final avg I_sp | Final min d_min |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in summary.holdout_scenarios:
        lines.append(
            "| "
            f"{scenario} | "
            f"{_format_rate(summary.final_success_by_scenario[scenario])} | "
            f"{_format_optional_rate(summary.final_collision_by_scenario[scenario])} | "
            f"{_format_optional_rate(summary.final_timeout_by_scenario[scenario])} | "
            f"{_format_table_float(summary.final_avg_steps_by_scenario[scenario], digits=1)} | "
            f"{_format_table_float(summary.final_avg_I_sp_by_scenario[scenario])} | "
            f"{_format_table_float(summary.final_min_d_min_by_scenario[scenario])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
