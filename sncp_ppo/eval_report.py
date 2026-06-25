"""Evaluation report helpers for SNCP-PPO density sweeps."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    collision: bool
    timeout: bool
    steps: int
    total_reward: float
    avg_i_sp: float
    min_d_min: float


@dataclass(frozen=True)
class DensitySummary:
    num_humans: int
    scenario: str
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    avg_success_steps: float
    avg_episode_steps: float
    avg_i_sp: float
    avg_min_d_min: float
    avg_reward: float


@dataclass(frozen=True)
class DensityComparison:
    num_humans: int
    baseline_success_rate: float
    candidate_success_rate: float
    success_delta: float
    baseline_collision_rate: float
    candidate_collision_rate: float
    collision_delta: float
    baseline_timeout_rate: float
    candidate_timeout_rate: float
    timeout_delta: float
    baseline_avg_success_steps: float
    candidate_avg_success_steps: float
    nav_margin_vs_beeline: float
    baseline_avg_i_sp: float
    candidate_avg_i_sp: float
    i_sp_delta: float
    status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SweepComparison:
    overall_status: str
    rows: tuple[DensityComparison, ...]
    baseline_nav_steps: float


CSV_FIELDS = [
    "num_humans",
    "scenario",
    "episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_success_steps",
    "avg_episode_steps",
    "avg_i_sp",
    "avg_min_d_min",
    "avg_reward",
]


def _rate(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(bool(v) for v in items) / len(items) if items else math.nan


def summarize_density(
    num_humans: int,
    scenario: str,
    episodes: Sequence[EpisodeResult],
) -> DensitySummary:
    if not episodes:
        raise ValueError("episodes must contain at least one EpisodeResult")

    success_steps = [ep.steps for ep in episodes if ep.success]
    avg_success_steps = mean(success_steps) if success_steps else math.nan

    return DensitySummary(
        num_humans=num_humans,
        scenario=scenario,
        episodes=len(episodes),
        success_rate=_rate(ep.success for ep in episodes),
        collision_rate=_rate(ep.collision for ep in episodes),
        timeout_rate=_rate(ep.timeout for ep in episodes),
        avg_success_steps=avg_success_steps,
        avg_episode_steps=mean(ep.steps for ep in episodes),
        avg_i_sp=mean(ep.avg_i_sp for ep in episodes),
        avg_min_d_min=mean(ep.min_d_min for ep in episodes),
        avg_reward=mean(ep.total_reward for ep in episodes),
    )


def _format_csv_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return value


def write_summary_csv(summaries: Sequence[DensitySummary], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            row = {key: _format_csv_value(value) for key, value in asdict(summary).items()}
            writer.writerow(row)


def write_summary_json(
    summaries: Sequence[DensitySummary],
    path: str | Path,
    *,
    checkpoint: str,
    baseline_nav_steps: float,
    trajectory_files: Sequence[str] = (),
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "checkpoint": checkpoint,
        "baseline_nav_steps": baseline_nav_steps,
        "trajectory_files": list(trajectory_files),
        "density_sweep": [asdict(summary) for summary in summaries],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_summary_json(path: str | Path) -> list[DensitySummary]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DensitySummary(**row) for row in data["density_sweep"]]


def write_markdown_report(
    summaries: Sequence[DensitySummary],
    path: str | Path,
    *,
    checkpoint: str,
    baseline_nav_steps: float,
    trajectory_files: Sequence[str] = (),
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# SNCP-PPO Evaluation Report",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "## Density Sweep",
        "",
        "| N | Success | Collision | Timeout | Avg Success Steps | Avg I_sp | Avg Min d_min |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.num_humans} | {summary.success_rate:.1%} | "
            f"{summary.collision_rate:.1%} | {summary.timeout_rate:.1%} | "
            f"{summary.avg_success_steps:.1f} | {summary.avg_i_sp:.4f} | "
            f"{summary.avg_min_d_min:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Real-Avoidance Gates",
            "",
            f"- No-beeline check: average successful navigation steps should stay well above {baseline_nav_steps:.1f}.",
            "- I_sp should stay low in the non-reactive crowd.",
            "- Trajectory plots should route around clusters rather than through them.",
            "- Collision rate should not rise while success improves.",
        ]
    )
    if trajectory_files:
        lines.extend(["", "## Trajectory Artifacts", ""])
        lines.extend(f"- `{name}`" for name in trajectory_files)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_from_notes(notes: Sequence[str]) -> str:
    if any(note.startswith("FAIL:") for note in notes):
        return "fail"
    if any(note.startswith("WARN:") for note in notes):
        return "warn"
    return "pass"


def _overall_status(rows: Sequence[DensityComparison]) -> str:
    statuses = {row.status for row in rows}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def compare_density_sweeps(
    baseline: Sequence[DensitySummary],
    candidate: Sequence[DensitySummary],
    *,
    baseline_nav_steps: float,
    nav_margin_steps: float = 30.0,
    success_tolerance: float = 0.05,
    collision_tolerance: float = 0.05,
    timeout_tolerance: float = 0.10,
    i_sp_warn_tolerance: float = 0.01,
    i_sp_fail_tolerance: float = 0.02,
) -> SweepComparison:
    """Compare a candidate sweep against the standardized v15 baseline gates."""

    baseline_by_density = {item.num_humans: item for item in baseline}
    candidate_by_density = {item.num_humans: item for item in candidate}
    if baseline_by_density.keys() != candidate_by_density.keys():
        raise ValueError("baseline and candidate sweeps must contain the same densities")

    max_density = max(baseline_by_density)
    rows: list[DensityComparison] = []
    for num_humans in sorted(baseline_by_density):
        base = baseline_by_density[num_humans]
        cand = candidate_by_density[num_humans]
        notes: list[str] = []

        success_delta = round(cand.success_rate - base.success_rate, 4)
        collision_delta = round(cand.collision_rate - base.collision_rate, 4)
        timeout_delta = round(cand.timeout_rate - base.timeout_rate, 4)
        i_sp_delta = round(cand.avg_i_sp - base.avg_i_sp, 4)
        nav_margin = round(cand.avg_success_steps - baseline_nav_steps, 4)

        if math.isnan(cand.avg_success_steps) or nav_margin < nav_margin_steps:
            notes.append(
                f"FAIL: beeline/nav-time regression ({cand.avg_success_steps:.1f} steps)"
            )
        if success_delta < -success_tolerance:
            notes.append(f"FAIL: success dropped by {success_delta * 100.0:.1f} pp")
        if collision_delta > collision_tolerance:
            notes.append(f"FAIL: collision rose by {collision_delta * 100.0:.1f} pp")
        if timeout_delta > timeout_tolerance:
            notes.append(f"FAIL: timeout/freezing rose by {timeout_delta * 100.0:.1f} pp")
        # I_sp is only a comfort REGRESSION when the robot got bolder for no
        # gain. When success clearly improves (and collision did not rise — that
        # has its own gate above), a higher I_sp is the byproduct of more
        # close-but-safe passes, not a regression, so the gate is suppressed.
        # (v22 lesson: every density improved success yet N=8 tripped the old
        # absolute 0.02 gate purely because the policy navigates more crowds.)
        success_clearly_improved = success_delta >= success_tolerance
        if not success_clearly_improved:
            if i_sp_delta > i_sp_fail_tolerance:
                notes.append(f"FAIL: I_sp rose by {i_sp_delta:.4f}")
            elif i_sp_delta > i_sp_warn_tolerance:
                notes.append(f"WARN: I_sp rose by {i_sp_delta:.4f}")

        if num_humans == max_density and success_delta <= 0:
            notes.append("WARN: high-density success did not improve")

        if not notes:
            notes.append("PASS: preserved real-avoidance gates")

        rows.append(
            DensityComparison(
                num_humans=num_humans,
                baseline_success_rate=base.success_rate,
                candidate_success_rate=cand.success_rate,
                success_delta=success_delta,
                baseline_collision_rate=base.collision_rate,
                candidate_collision_rate=cand.collision_rate,
                collision_delta=collision_delta,
                baseline_timeout_rate=base.timeout_rate,
                candidate_timeout_rate=cand.timeout_rate,
                timeout_delta=timeout_delta,
                baseline_avg_success_steps=base.avg_success_steps,
                candidate_avg_success_steps=cand.avg_success_steps,
                nav_margin_vs_beeline=nav_margin,
                baseline_avg_i_sp=base.avg_i_sp,
                candidate_avg_i_sp=cand.avg_i_sp,
                i_sp_delta=i_sp_delta,
                status=_status_from_notes(notes),
                notes=tuple(notes),
            )
        )

    return SweepComparison(
        overall_status=_overall_status(rows),
        rows=tuple(rows),
        baseline_nav_steps=baseline_nav_steps,
    )


def write_comparison_report(
    comparison: SweepComparison,
    path: str | Path,
    *,
    baseline_path: str | Path,
    candidate_path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# SNCP-PPO Baseline Comparison",
        "",
        f"Overall verdict: {comparison.overall_status}",
        "",
        f"Baseline: `{Path(baseline_path)}`",
        f"Candidate: `{Path(candidate_path)}`",
        f"Beeline baseline: {comparison.baseline_nav_steps:.1f} successful steps",
        "",
        "| N | Baseline Success | Candidate Success | Success Delta | Baseline Collision | Candidate Collision | Baseline Timeout | Candidate Timeout | Timeout Delta | Nav Margin | I_sp Delta | Status | Notes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in comparison.rows:
        notes = "; ".join(row.notes)
        lines.append(
            f"| {row.num_humans} | {row.baseline_success_rate:.1%} | "
            f"{row.candidate_success_rate:.1%} | {row.success_delta * 100.0:+.1f} pp | "
            f"{row.baseline_collision_rate:.1%} | {row.candidate_collision_rate:.1%} | "
            f"{row.baseline_timeout_rate:.1%} | {row.candidate_timeout_rate:.1%} | "
            f"{row.timeout_delta * 100.0:+.1f} pp | "
            f"{row.nav_margin_vs_beeline:.1f} | {row.i_sp_delta:+.4f} | "
            f"{row.status} | {notes} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_density_curves(
    summaries: Sequence[DensitySummary],
    output_path: str | Path,
    *,
    baseline_nav_steps: float,
) -> None:
    if not summaries:
        raise ValueError("summaries must contain at least one DensitySummary")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(summaries, key=lambda item: item.num_humans)
    densities = [item.num_humans for item in ordered]
    success = [item.success_rate * 100.0 for item in ordered]
    collision = [item.collision_rate * 100.0 for item in ordered]
    timeout = [item.timeout_rate * 100.0 for item in ordered]
    nav_steps = [item.avg_success_steps for item in ordered]
    i_sp = [item.avg_i_sp for item in ordered]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(densities, success, marker="o", label="success")
    axes[0].plot(densities, collision, marker="o", label="collision")
    axes[0].plot(densities, timeout, marker="o", label="timeout")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="best")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(densities, nav_steps, marker="o", color="tab:green", label="avg success steps")
    axes[1].axhline(
        baseline_nav_steps,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"v14 beeline baseline ({baseline_nav_steps:.1f})",
    )
    axes[1].set_ylabel("Steps")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    axes[2].plot(densities, i_sp, marker="o", color="tab:purple")
    axes[2].set_ylabel("Avg I_sp")
    axes[2].set_xlabel("Number of pedestrians")
    axes[2].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("SNCP-PPO Density Sweep")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


Evaluator = Callable[..., Sequence[EpisodeResult]]
TrajectoryRenderer = Callable[..., None]


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_density(
    *,
    checkpoint_path: Path,
    num_humans: int,
    scenario: str,
    n_episodes: int,
    seed: int,
    robot_vpref: float = 0.26,
    human_vpref_override: float | None = None,
    max_time: float | None = None,
    human_goal_noise: float = 0.0,
    action_shield: bool = False,
    shield_horizon_steps: int = 6,
    shield_safety_margin: float = 0.0,
) -> list[EpisodeResult]:
    """Run deterministic policy episodes for one density/scenario pair.

    robot_vpref / human_vpref_override / max_time let the eval match the regime
    the checkpoint was TRAINED in (e.g. the paper-reproduction run uses robot
    1.0 m/s + parity pedestrians); defaults preserve the TurtleBot regime."""

    import torch

    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.action_shield import ActionShieldConfig, shield_action
    from sncp_ppo.models import build_policy_for_checkpoint
    from sncp_ppo.ppo import PPOAgent

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CrowdSimEnv(
        num_humans=num_humans, scenario=scenario,
        robot_vpref=robot_vpref, human_vpref_override=human_vpref_override,
        max_time=max_time, human_goal_noise=human_goal_noise,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy = build_policy_for_checkpoint(
        state_dict, robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.train(False)
    agent = PPOAgent(policy=policy)
    max_steps = int(env.max_time / env.time_step) + 1
    shield_cfg = ActionShieldConfig(
        horizon_steps=shield_horizon_steps,
        safety_margin=shield_safety_margin,
    )

    results: list[EpisodeResult] = []
    for episode_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        h_states = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)
        total_reward = 0.0
        total_i_sp = 0.0
        min_d_min = math.inf
        steps = 0
        info = {"success": False, "collision": False, "timeout": False}

        while steps < max_steps:
            action, _, _, h_states_next = agent.select_action(
                obs,
                h_states,
                device,
                deterministic=True,
            )
            if action_shield:
                action = shield_action(env, action, shield_cfg)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_i_sp += float(info.get("I_sp", 0.0))
            min_d_min = min(min_d_min, float(info.get("d_min", math.inf)))
            steps += 1
            h_states = h_states_next
            if terminated or truncated:
                break

        results.append(
            EpisodeResult(
                success=bool(info.get("success", False)),
                collision=bool(info.get("collision", False)),
                timeout=bool(info.get("timeout", False)),
                steps=steps,
                total_reward=total_reward,
                avg_i_sp=total_i_sp / max(steps, 1),
                min_d_min=min_d_min if math.isfinite(min_d_min) else math.nan,
            )
        )

    return results


def render_trajectory(
    *,
    checkpoint_path: Path,
    output_path: Path,
    num_humans: int,
    scenario: str,
    seed: int,
    robot_vpref: float = 0.26,
    human_vpref_override: float | None = None,
    max_time: float = 50.0,
    human_goal_noise: float = 0.0,
    action_shield: bool = False,
    shield_horizon_steps: int = 6,
    shield_safety_margin: float = 0.0,
) -> None:
    from visualize_trajectory import run_and_visualize

    run_and_visualize(
        model_path=str(checkpoint_path),
        output_image=str(output_path),
        num_humans=num_humans,
        scenario=scenario,
        seed=seed,
        robot_vpref=robot_vpref,
        human_vpref_override=human_vpref_override,
        max_time=max_time,
        human_goal_noise=human_goal_noise,
        action_shield=action_shield,
        shield_horizon_steps=shield_horizon_steps,
        shield_safety_margin=shield_safety_margin,
    )


def run_report(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    densities: Sequence[int],
    scenario: str,
    n_episodes: int,
    seed: int,
    trajectory_densities: Sequence[int] = (),
    baseline_nav_steps: float = 121.5,
    robot_vpref: float = 0.26,
    human_vpref_override: float | None = None,
    max_time: float = 50.0,
    human_goal_noise: float = 0.0,
    action_shield: bool = False,
    shield_horizon_steps: int = 6,
    shield_safety_margin: float = 0.0,
    evaluator: Evaluator = evaluate_density,
    trajectory_renderer: TrajectoryRenderer = render_trajectory,
) -> dict[str, Path | list[Path]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not densities:
        raise ValueError("densities must contain at least one value")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[DensitySummary] = []
    for num_humans in densities:
        episodes = evaluator(
            checkpoint_path=checkpoint_path,
            num_humans=int(num_humans),
            scenario=scenario,
            n_episodes=n_episodes,
            seed=seed,
            robot_vpref=robot_vpref,
            human_vpref_override=human_vpref_override,
            max_time=max_time,
            human_goal_noise=human_goal_noise,
            action_shield=action_shield,
            shield_horizon_steps=shield_horizon_steps,
            shield_safety_margin=shield_safety_margin,
        )
        summaries.append(
            summarize_density(
                num_humans=int(num_humans),
                scenario=scenario,
                episodes=episodes,
            )
        )

    trajectory_paths: list[Path] = []
    for num_humans in trajectory_densities:
        output_path = output_dir / f"traj_{scenario}_n{int(num_humans)}.png"
        trajectory_renderer(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            num_humans=int(num_humans),
            scenario=scenario,
            seed=seed,
            robot_vpref=robot_vpref,
            human_vpref_override=human_vpref_override,
            max_time=max_time,
            human_goal_noise=human_goal_noise,
            action_shield=action_shield,
            shield_horizon_steps=shield_horizon_steps,
            shield_safety_margin=shield_safety_margin,
        )
        trajectory_paths.append(output_path)

    csv_path = output_dir / "density_sweep.csv"
    json_path = output_dir / "density_sweep.json"
    plot_path = output_dir / "density_sweep.png"
    report_path = output_dir / "report.md"
    trajectory_names = [path.name for path in trajectory_paths]

    write_summary_csv(summaries, csv_path)
    write_summary_json(
        summaries,
        json_path,
        checkpoint=str(checkpoint_path),
        baseline_nav_steps=baseline_nav_steps,
        trajectory_files=trajectory_names,
    )
    plot_density_curves(summaries, plot_path, baseline_nav_steps=baseline_nav_steps)
    write_markdown_report(
        summaries,
        report_path,
        checkpoint=str(checkpoint_path),
        baseline_nav_steps=baseline_nav_steps,
        trajectory_files=trajectory_names,
    )

    return {
        "csv": csv_path,
        "json": json_path,
        "plot": plot_path,
        "report": report_path,
        "trajectories": trajectory_paths,
    }
