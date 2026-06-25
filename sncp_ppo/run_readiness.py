"""Preflight checks for the current Colab experiment run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# v37 = probe-first experiment. The notebook must not launch the failed v36
# combined-levers run; it should run the paired C0/C1 probe from the locked
# v34-fixed-beta checkpoint and then analyze the GO/NO-GO gate.
TRAINING_TOKENS = (
    "BASE_CHECKPOINT = 'sncp_ppo_v34.pt'",
    "OUTPUT_DIR = 'eval_v37_probe'",
    "TOTAL_STEPS = 300_000",
    "EVAL_EPISODES = 100",
    "scripts/run_v37_probes.py",
    "'--mode', 'run'",
    "'--base_checkpoint', BASE_CHECKPOINT",
    "'--output_dir', OUTPUT_DIR",
    "'--python', sys.executable",
    "'--eval_episodes', str(EVAL_EPISODES)",
    "'--total_steps', str(TOTAL_STEPS)",
    "if not os.path.exists(BASE_CHECKPOINT):",
    "raise FileNotFoundError(",
    "if p.returncode != 0:",
    "raise SystemExit(p.returncode)",
)

EVALUATION_TOKENS = (
    "EVAL_OUT = 'eval_v37_probe'",
    "scratch/_analyze_v37_probe.py",
    "'--input_dir', EVAL_OUT",
    "report = os.path.join(EVAL_OUT, 'report.md')",
    "verdict = os.path.join(EVAL_OUT, 'verdict.json')",
    "Verdict:",
)


@dataclass(frozen=True)
class V16RunReadinessSummary:
    status: str
    repo_root: str
    training_cell_found: bool
    evaluation_cell_found: bool
    baseline_densities: tuple[int, ...]
    notes: tuple[str, ...]


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _load_notebook(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        _source_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _find_unique_cell(cells: Sequence[str], marker: str, notes: list[str], name: str) -> str | None:
    matches = [source for source in cells if marker in source]
    if len(matches) != 1:
        notes.append(f"FAIL: expected exactly one {name} cell with marker `{marker}`, found {len(matches)}")
        return None
    return matches[0]


def _check_tokens(source: str | None, tokens: Sequence[str], notes: list[str], name: str) -> None:
    if source is None:
        return
    for token in tokens:
        if token not in source:
            notes.append(f"FAIL: {name} cell missing `{token}`")


def _baseline_densities(path: Path, notes: list[str]) -> tuple[int, ...]:
    if not path.exists():
        notes.append(f"FAIL: missing baseline density sweep `{path}`")
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("density_sweep", [])
    densities = tuple(sorted(int(row["num_humans"]) for row in rows))
    expected = (1, 3, 5, 8, 10)
    if densities != expected:
        notes.append(f"FAIL: baseline densities are {densities}, expected {expected}")
    return densities


def _status(notes: Sequence[str]) -> str:
    return "fail" if any(note.startswith("FAIL:") for note in notes) else "pass"


def verify_v16_run_ready(repo_root: str | Path = ".") -> V16RunReadinessSummary:
    repo_root = Path(repo_root)
    notes: list[str] = []

    required_files = (
        repo_root / "sncp_ppo_colab.ipynb",
        repo_root / "scripts" / "run_v37_probes.py",
        repo_root / "scratch" / "_analyze_v37_probe.py",
    )
    for path in required_files:
        if not path.exists():
            notes.append(f"FAIL: missing required file `{path}`")

    cells = _load_notebook(repo_root / "sncp_ppo_colab.ipynb") or []
    training_cell = _find_unique_cell(
        cells,
        "BASE_CHECKPOINT = 'sncp_ppo_v34.pt'",
        notes,
        "v37 probe training",
    )
    evaluation_cell = _find_unique_cell(
        cells,
        "scratch/_analyze_v37_probe.py",
        notes,
        "v37 probe analysis",
    )
    _check_tokens(training_cell, TRAINING_TOKENS, notes, "v37 probe training")
    _check_tokens(evaluation_cell, EVALUATION_TOKENS, notes, "v37 probe analysis")

    densities = ()
    if not notes:
        notes.append("PASS: v37 Colab paired-probe configuration is ready")

    return V16RunReadinessSummary(
        status=_status(notes),
        repo_root=str(repo_root),
        training_cell_found=training_cell is not None,
        evaluation_cell_found=evaluation_cell is not None,
        baseline_densities=densities,
        notes=tuple(notes),
    )


def write_readiness_report(summary: V16RunReadinessSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    densities = ", ".join(str(item) for item in summary.baseline_densities) or "n/a"
    lines = [
        "# SNCP-PPO Run Readiness",
        "",
        f"Overall status: {summary.status}",
        f"Repo root: `{summary.repo_root}`",
        f"Training cell found: {summary.training_cell_found}",
        f"Evaluation cell found: {summary.evaluation_cell_found}",
        f"Baseline densities: {densities}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in summary.notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
