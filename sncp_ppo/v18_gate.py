"""Pre-v18 artifact gate for evidence-based SNCP-PPO experiment selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REQUIRED_FILES = (
    "run_readiness.md",
    "density_sweep.csv",
    "density_sweep.json",
    "density_sweep.png",
    "report.md",
    "comparison_vs_v15.md",
    "training_diagnostics.json",
    "training_diagnostics.md",
    "artifact_verification.md",
    "traj_hard_n5.png",
    "traj_hard_n10.png",
    "v18_decision.md",
    "v18_decision.json",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VALID_BRANCHES = {
    "A_TIMEOUT_MAX_TIME60",
    "B_HIGH_DENSITY_EXPOSURE",
    "C_REPLAY30",
    "D_STOP_COMFORT_RELAXATION",
}


@dataclass(frozen=True)
class V18GateSummary:
    status: str
    checkpoint_path: str
    eval_dir: str
    missing_files: tuple[str, ...]
    densities: tuple[int, ...]
    branch_id: str | None
    decision_status: str | None
    single_variable: str | None
    notes: tuple[str, ...]


def _overall_status(notes: Sequence[str]) -> str:
    return "fail" if any(note.startswith("FAIL:") for note in notes) else "pass"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_v18_decision_ready(
    *,
    checkpoint_path: str | Path = "checkpoints/sncp_ppo_v17.pt",
    eval_dir: str | Path = "eval_v17",
    required_densities: Sequence[int] = (1, 3, 5, 8, 10),
    min_episodes: int = 50,
) -> V18GateSummary:
    checkpoint_path = Path(checkpoint_path)
    eval_dir = Path(eval_dir)
    missing: list[str] = []
    empty: list[str] = []
    invalid_png: list[str] = []
    notes: list[str] = []

    if not checkpoint_path.exists():
        missing.append("checkpoint")
    elif checkpoint_path.stat().st_size == 0:
        empty.append("checkpoint")
    if not eval_dir.exists():
        missing.append(str(eval_dir))

    for file_name in REQUIRED_FILES:
        path = eval_dir / file_name
        if not path.exists():
            missing.append(file_name)
        elif path.stat().st_size == 0:
            empty.append(file_name)
        elif path.suffix.lower() == ".png":
            with path.open("rb") as f:
                if f.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                    invalid_png.append(file_name)

    if missing:
        notes.append(f"FAIL: missing required artifacts: {', '.join(missing)}")
    if empty:
        notes.append(f"FAIL: empty required artifacts: {', '.join(empty)}")
    if invalid_png:
        notes.append(f"FAIL: invalid PNG artifacts: {', '.join(invalid_png)}")

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

    decision = _load_json(eval_dir / "v18_decision.json") or {}
    branch_id = decision.get("branch_id")
    decision_status = decision.get("status")
    single_variable = decision.get("single_variable")
    if decision:
        if decision_status != "ready_for_review":
            notes.append(f"FAIL: v18 decision status is {decision_status}")
        if branch_id not in VALID_BRANCHES:
            notes.append(f"FAIL: unsupported v18 decision branch {branch_id}")

    if not notes:
        notes.append(
            f"PASS: v17 artifacts are ready for v18 review with branch {branch_id} "
            f"({single_variable})"
        )

    return V18GateSummary(
        status=_overall_status(notes),
        checkpoint_path=str(checkpoint_path),
        eval_dir=str(eval_dir),
        missing_files=tuple(missing),
        densities=densities,
        branch_id=branch_id,
        decision_status=decision_status,
        single_variable=single_variable,
        notes=tuple(notes),
    )


def write_v18_gate_report(summary: V18GateSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    densities = ", ".join(str(item) for item in summary.densities) or "n/a"
    lines = [
        "# v18 Artifact Gate",
        "",
        f"Overall status: {summary.status}",
        f"Checkpoint: `{summary.checkpoint_path}`",
        f"Eval dir: `{summary.eval_dir}`",
        f"Densities: {densities}",
        f"Decision status: {summary.decision_status or 'n/a'}",
        f"Branch: {summary.branch_id or 'n/a'}",
        f"Single variable: {summary.single_variable or 'n/a'}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in summary.notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "REQUIRED_FILES",
    "V18GateSummary",
    "verify_v18_decision_ready",
    "write_v18_gate_report",
]
