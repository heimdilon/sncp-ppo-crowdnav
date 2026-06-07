"""Artifact verification for completed v16 SNCP-PPO runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REQUIRED_EVAL_FILES = (
    "run_readiness.md",
    "density_sweep.csv",
    "density_sweep.json",
    "density_sweep.png",
    "report.md",
    "comparison_vs_v15.md",
    "training_diagnostics.json",
    "training_diagnostics.md",
    "traj_hard_n5.png",
    "traj_hard_n10.png",
)


@dataclass(frozen=True)
class ArtifactVerificationSummary:
    status: str
    checkpoint_path: str
    eval_dir: str
    missing_files: tuple[str, ...]
    densities: tuple[int, ...]
    readiness_status: str | None
    comparison_verdict: str | None
    collapse_detected: bool | None
    replay_ratio: float | None
    notes: tuple[str, ...]


def _overall_status(notes: Sequence[str]) -> str:
    if any(note.startswith("FAIL:") for note in notes):
        return "fail"
    if any(note.startswith("WARN:") for note in notes):
        return "warn"
    return "pass"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_verdict(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Overall verdict:\s*(pass|warn|fail)", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _readiness_status(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Overall status:\s*(pass|warn|fail)", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def verify_v16_artifacts(
    *,
    checkpoint_path: str | Path,
    eval_dir: str | Path,
    required_densities: Sequence[int] = (1, 3, 5, 8, 10),
    min_episodes: int = 50,
    expected_replay_ratio: float = 0.20,
    replay_tolerance: float = 0.10,
) -> ArtifactVerificationSummary:
    checkpoint_path = Path(checkpoint_path)
    eval_dir = Path(eval_dir)
    notes: list[str] = []
    missing: list[str] = []
    empty: list[str] = []

    if not checkpoint_path.exists():
        missing.append("checkpoint")
    elif checkpoint_path.stat().st_size == 0:
        empty.append("checkpoint")
    if not eval_dir.exists():
        missing.append(str(eval_dir))
    for file_name in REQUIRED_EVAL_FILES:
        artifact_path = eval_dir / file_name
        if not artifact_path.exists():
            missing.append(file_name)
        elif artifact_path.stat().st_size == 0:
            empty.append(file_name)
    if missing:
        notes.append(f"FAIL: missing required artifacts: {', '.join(missing)}")
    if empty:
        notes.append(f"FAIL: empty required artifacts: {', '.join(empty)}")

    readiness = _readiness_status(eval_dir / "run_readiness.md")
    if readiness is None and (eval_dir / "run_readiness.md").exists():
        notes.append("FAIL: run readiness report does not contain an overall status")
    elif readiness != "pass" and readiness is not None:
        notes.append(f"FAIL: run readiness status is {readiness}")

    densities: tuple[int, ...] = ()
    sweep = _load_json(eval_dir / "density_sweep.json")
    if sweep is not None:
        rows = sweep.get("density_sweep", [])
        densities = tuple(sorted(int(row["num_humans"]) for row in rows))
        expected = tuple(required_densities)
        if densities != expected:
            notes.append(f"FAIL: density sweep has {densities}, expected {expected}")
        too_small = [
            int(row["num_humans"])
            for row in rows
            if int(row.get("episodes", 0)) < min_episodes
        ]
        if too_small:
            notes.append(f"FAIL: densities below {min_episodes} episodes: {too_small}")

    verdict = _comparison_verdict(eval_dir / "comparison_vs_v15.md")
    if verdict is None and (eval_dir / "comparison_vs_v15.md").exists():
        notes.append("FAIL: comparison report does not contain an overall verdict")
    elif verdict == "fail":
        notes.append("FAIL: comparison verdict is fail")
    elif verdict == "warn":
        notes.append("WARN: comparison verdict is warn")

    diagnostics = _load_json(eval_dir / "training_diagnostics.json")
    collapse_detected = None
    replay_ratio = None
    if diagnostics is not None:
        collapse_detected = diagnostics.get("collapse_detected")
        replay_ratio = diagnostics.get("observed_replay_ratio")
        if collapse_detected is True:
            notes.append("FAIL: training collapse detected")
        if replay_ratio is None:
            notes.append("WARN: replay ratio not logged")
        elif abs(float(replay_ratio) - expected_replay_ratio) > replay_tolerance:
            notes.append(
                f"WARN: replay ratio {float(replay_ratio):.1%} differs from expected "
                f"{expected_replay_ratio:.1%}"
            )

    if not notes:
        notes.append("PASS: required v16 artifacts are present and gates passed")

    return ArtifactVerificationSummary(
        status=_overall_status(notes),
        checkpoint_path=str(checkpoint_path),
        eval_dir=str(eval_dir),
        missing_files=tuple(missing),
        densities=densities,
        readiness_status=readiness,
        comparison_verdict=verdict,
        collapse_detected=collapse_detected,
        replay_ratio=replay_ratio,
        notes=tuple(notes),
    )


def write_artifact_verification_report(
    summary: ArtifactVerificationSummary,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    replay = "not logged" if summary.replay_ratio is None else f"{summary.replay_ratio:.1%}"
    lines = [
        "# SNCP-PPO v16 Artifact Verification",
        "",
        f"Overall status: {summary.status}",
        f"Checkpoint: `{summary.checkpoint_path}`",
        f"Eval dir: `{summary.eval_dir}`",
        f"Densities: {', '.join(str(item) for item in summary.densities) or 'n/a'}",
        f"Run readiness: {summary.readiness_status or 'n/a'}",
        f"Comparison verdict: {summary.comparison_verdict or 'n/a'}",
        f"Training collapse detected: {summary.collapse_detected}",
        f"Replay ratio: {replay}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in summary.notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
