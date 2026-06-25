"""Shared command and decision logic for the v37 paired probe."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ARMS = ("c0", "c1")
TRAIN_SEEDS = (40, 41, 42)
DENSITIES = (5, 10, 15, 20)
LOW_DENSITIES = (5, 10)
HIGH_DENSITIES = (15, 20)


def build_training_command(
    arm: str,
    seed: int,
    *,
    base_checkpoint: Path,
    save_path: Path,
    python_executable: str,
    total_steps: int = 300_000,
) -> list[str]:
    """Return one controlled C0/C1 fine-tune command.

    C0 and C1 differ only in exact continuation vs the zero-gated v37 upgrade.
    Failed v36 levers are intentionally absent.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown probe arm: {arm}")
    base_cli = base_checkpoint.as_posix()
    save_cli = save_path.as_posix()
    command = [
        python_executable, "-u", "-m", "sncp_ppo.train",
        "--num_envs", "16",
        "--horizon", "128",
        "--total_steps", str(total_steps),
        "--eval_freq_updates", "20",
        "--fixed_scenario", "paper_challenging",
        "--num_humans", "10",
        "--num_humans_range", "10", "25",
        "--bootstrap_easy_steps", "0",
        "--seed", str(seed),
        "--lr", str(5e-5),
        "--lr_end_factor", "0.5",
        "--target_kl", "0.01",
        "--robot_vpref", "1.0",
        "--holdout_scenarios", "paper_standard", "paper_challenging",
        "--holdout_episodes", "50",
        "--ent_coef", "0.001",
        "--save_path", save_cli,
    ]
    if arm == "c0":
        command.extend(["--init_checkpoint", base_cli])
    else:
        command.extend([
            "--upgrade_checkpoint", base_cli,
            "--hh_intent_graph",
            "--hh_attn_heads", "4",
            "--cv_horizons", "1", "2", "3", "4",
            "--cv_dt", "0.25",
        ])
    return command


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def _exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2.0 ** discordant))


def _rate(rows: Iterable[dict], key: str) -> float:
    rows = list(rows)
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else math.nan


def _paired_metric(c0_rows, c1_rows, metric):
    c0 = {(row["train_seed"], row["case_id"]): bool(row[metric]) for row in c0_rows}
    c1 = {(row["train_seed"], row["case_id"]): bool(row[metric]) for row in c1_rows}
    if c0.keys() != c1.keys():
        raise ValueError(f"paired case bank mismatch for metric {metric}")
    left_only = sum(c0[key] and not c1[key] for key in c0)
    right_only = sum(c1[key] and not c0[key] for key in c0)
    return {
        "c0_only": left_only,
        "c1_only": right_only,
        "discordant": left_only + right_only,
        "mcnemar_exact_p": _exact_mcnemar_p(left_only, right_only),
    }


def analyze_probe_runs(runs: Sequence[dict]) -> dict:
    """Apply the preregistered v37 probe GO/NO-GO rule to episode-level runs."""
    by_run = {(run["arm"], int(run["train_seed"])): run for run in runs}
    complete = all((arm, seed) in by_run for arm in ARMS for seed in TRAIN_SEEDS)

    flattened = defaultdict(list)
    per_seed = defaultdict(list)
    for run in runs:
        arm = run["arm"]
        train_seed = int(run["train_seed"])
        for episode in run.get("episodes", []):
            row = dict(episode, arm=arm, train_seed=train_seed)
            density = int(row["density"])
            flattened[(arm, density)].append(row)
            per_seed[(arm, train_seed, density)].append(row)

    for arm in ARMS:
        for density in DENSITIES:
            if not flattened[(arm, density)]:
                complete = False

    rates = {arm: {} for arm in ARMS}
    paired = {}
    if complete:
        for arm in ARMS:
            for density in DENSITIES:
                rows = flattened[(arm, density)]
                success_count = sum(bool(row["success"]) for row in rows)
                low, high = wilson_interval(success_count, len(rows))
                rates[arm][str(density)] = {
                    "n": len(rows),
                    "success": _rate(rows, "success"),
                    "collision": _rate(rows, "collision"),
                    "timeout": _rate(rows, "timeout"),
                    "success_wilson_95": [low, high],
                }
                paired[str(density)] = {
                    "success": _paired_metric(
                        flattened[("c0", density)], flattened[("c1", density)], "success"
                    ),
                    "collision": _paired_metric(
                        flattened[("c0", density)], flattened[("c1", density)], "collision"
                    ),
                }

    def delta(metric, density):
        return rates["c1"][str(density)][metric] - rates["c0"][str(density)][metric]

    if complete:
        high_n_success_delta = sum(delta("success", n) for n in HIGH_DENSITIES) / 2.0
        high_n_collision_delta = sum(delta("collision", n) for n in HIGH_DENSITIES) / 2.0
        success_gate = high_n_success_delta >= 0.03 - 1e-12
        collision_gate = high_n_collision_delta <= -0.03 + 1e-12
        success_direction = 0
        collision_direction = 0
        for seed in TRAIN_SEEDS:
            seed_success_delta = sum(
                _rate(per_seed[("c1", seed, n)], "success")
                - _rate(per_seed[("c0", seed, n)], "success")
                for n in HIGH_DENSITIES
            ) / 2.0
            seed_collision_delta = sum(
                _rate(per_seed[("c1", seed, n)], "collision")
                - _rate(per_seed[("c0", seed, n)], "collision")
                for n in HIGH_DENSITIES
            ) / 2.0
            success_direction += seed_success_delta > 0
            collision_direction += seed_collision_delta < 0
        eligible_directions = []
        if success_gate:
            eligible_directions.append(success_direction)
        if collision_gate:
            eligible_directions.append(collision_direction)
        direction_seed_count = max(eligible_directions, default=0)
        low_n_regression = any(
            delta("success", n) < -0.02 - 1e-12
            or delta("collision", n) > 0.02 + 1e-12
            for n in LOW_DENSITIES
        )
        timeout_zero = all(rates["c1"][str(n)]["timeout"] == 0.0 for n in DENSITIES)
    else:
        high_n_success_delta = math.nan
        high_n_collision_delta = math.nan
        success_gate = collision_gate = False
        direction_seed_count = 0
        low_n_regression = True
        timeout_zero = False

    c1_runs = [run for run in runs if run.get("arm") == "c1"]
    gate_active = (
        len(c1_runs) == len(TRAIN_SEEDS)
        and all(abs(float(run.get("hh_gate") or 0.0)) >= 0.01 for run in c1_runs)
    )
    diagnostics_healthy = (
        len(c1_runs) == len(TRAIN_SEEDS)
        and all(
            bool(run.get("diagnostics", {}).get("finite"))
            and float(run.get("diagnostics", {}).get("collapse_delta", -math.inf)) >= -0.05
            for run in c1_runs
        )
    )
    high_n_gain = success_gate or collision_gate
    verdict = "GO" if all([
        complete,
        high_n_gain,
        direction_seed_count >= 2,
        not low_n_regression,
        timeout_zero,
        gate_active,
        diagnostics_healthy,
    ]) else "NO-GO"
    return {
        "complete": complete,
        "rates": rates,
        "paired": paired,
        "high_n_success_delta": round(high_n_success_delta, 10) if complete else None,
        "high_n_collision_delta": round(high_n_collision_delta, 10) if complete else None,
        "high_n_gain": high_n_gain,
        "direction_seed_count": direction_seed_count,
        "low_n_regression": low_n_regression,
        "timeout_zero": timeout_zero,
        "gate_active": gate_active,
        "diagnostics_healthy": diagnostics_healthy,
        "verdict": verdict,
    }
