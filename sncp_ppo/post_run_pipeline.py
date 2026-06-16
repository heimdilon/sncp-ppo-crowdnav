"""One-command post-run pipeline for v16 evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from sncp_ppo.artifact_verifier import (
    verify_v16_artifacts,
    write_artifact_verification_report,
)
from sncp_ppo.eval_report import (
    compare_density_sweeps,
    load_summary_json,
    run_report,
    write_comparison_report,
)
from sncp_ppo.run_readiness import verify_v16_run_ready, write_readiness_report
from sncp_ppo.training_log_report import (
    analyze_training_log,
    write_training_diagnostic_json,
    write_training_diagnostic_report,
)


@dataclass(frozen=True)
class PostRunResult:
    status: str
    output_dir: Path
    comparison_report: Path
    training_report: Path
    artifact_report: Path


ReportRunner = Callable[..., object]


def find_latest_training_csv(log_dir: str | Path) -> Path:
    log_dir = Path(log_dir)
    matches = sorted(log_dir.glob("training_*.csv"))
    if not matches:
        raise FileNotFoundError(f"no training_*.csv files in {log_dir}")
    return matches[-1]


def run_v16_post_eval(
    *,
    checkpoint_path: str | Path,
    training_csv: str | Path,
    output_dir: str | Path = "eval_v16",
    baseline_json: str | Path = "eval_v15/density_sweep.json",
    densities: Sequence[int] = (1, 3, 5, 8, 10),
    scenario: str = "hard",
    n_episodes: int = 50,
    seed: int = 100,
    trajectory_densities: Sequence[int] = (5, 10),
    baseline_nav_steps: float = 121.5,
    nav_margin_steps: float = 30.0,
    robot_vpref: float = 0.26,
    human_vpref_override: float | None = None,
    max_time: float | None = None,
    human_goal_noise: float = 0.0,
    expected_replay_ratio: float = 0.20,
    replay_tolerance: float = 0.10,
    report_runner: ReportRunner = run_report,
    readiness_root: str | Path = ".",
    readiness_checker: Callable = verify_v16_run_ready,
) -> PostRunResult:
    checkpoint_path = Path(checkpoint_path)
    training_csv = Path(training_csv)
    output_dir = Path(output_dir)
    baseline_json = Path(baseline_json)

    report_runner(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        densities=list(densities),
        scenario=scenario,
        n_episodes=n_episodes,
        seed=seed,
        trajectory_densities=list(trajectory_densities),
        baseline_nav_steps=baseline_nav_steps,
        robot_vpref=robot_vpref,
        human_vpref_override=human_vpref_override,
        max_time=max_time,
        human_goal_noise=human_goal_noise,
    )

    # Ensure the bundle carries a run-readiness report even if the operator
    # skipped the preflight cell (v22 lesson: a skipped preflight should not
    # fail an otherwise-good eval — the artifact verifier requires this file).
    readiness_path = output_dir / "run_readiness.md"
    if not readiness_path.exists():
        write_readiness_report(readiness_checker(readiness_root), readiness_path)

    comparison = compare_density_sweeps(
        load_summary_json(baseline_json),
        load_summary_json(output_dir / "density_sweep.json"),
        baseline_nav_steps=baseline_nav_steps,
        nav_margin_steps=nav_margin_steps,
    )
    comparison_report = output_dir / "comparison_vs_v15.md"
    write_comparison_report(
        comparison,
        comparison_report,
        baseline_path=baseline_json,
        candidate_path=output_dir / "density_sweep.json",
    )

    training_summary = analyze_training_log(training_csv)
    training_json = output_dir / "training_diagnostics.json"
    training_report = output_dir / "training_diagnostics.md"
    write_training_diagnostic_json(training_summary, training_json)
    write_training_diagnostic_report(training_summary, training_report)

    artifact_summary = verify_v16_artifacts(
        checkpoint_path=checkpoint_path,
        eval_dir=output_dir,
        min_episodes=n_episodes,
        expected_replay_ratio=expected_replay_ratio,
        replay_tolerance=replay_tolerance,
    )
    artifact_report = output_dir / "artifact_verification.md"
    write_artifact_verification_report(artifact_summary, artifact_report)

    return PostRunResult(
        status=artifact_summary.status,
        output_dir=output_dir,
        comparison_report=comparison_report,
        training_report=training_report,
        artifact_report=artifact_report,
    )
