"""Preflight checks for the final end-to-end Colab notebook.

The final notebook is a single self-contained pipeline (no per-version cells):
  1. train the v34 Beta policy from scratch,
  2. honest 5-seed sweep of the raw policy (v34) and the shielded system (v38)
     through one evaluate_density path,
  3. statistical analysis (Wilson / two-proportion z / Bonferroni),
  4. artifact download bundle.

This module verifies those four cells exist with the expected tokens, so an
expensive Colab run is not launched against a stale notebook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# 2. v34 final policy: v30 base (pre-MLP + mean+max + density curriculum) + Beta action
#    distribution with tuned entropy, 2.5M steps, holdout-best.
TRAINING_TOKENS = (
    "python -m sncp_ppo.train",
    "--total_steps 2500000",
    "--fixed_scenario paper_challenging",
    "--num_humans_range 10 20",
    "--bootstrap_easy_steps 200000",
    "--robot_vpref 1.0",
    "--holdout_scenarios paper_standard paper_challenging",
    "--pre_mlp",
    "--meanmax_pool",
    "--action_dist beta",
    "--ent_coef 0.001",
    "--save_path checkpoints/sncp_ppo_v34.pt",
)

# 3. Honest 5-seed sweep: same evaluate_density path, shield off (v34) and on (v38).
SWEEP_TOKENS = (
    "from sncp_ppo.eval_report import evaluate_density",
    "def honest_sweep(action_shield, out_path, label):",
    "action_shield=action_shield",
    "shield_horizon_steps=6",
    "SEEDS = [100, 200, 300, 400, 500]",
    "DENSITIES = [5, 10, 15, 20]",
    "honest_sweep(action_shield=False",
    "honest_sweep(action_shield=True",
    "v34_multiseed_result.json",
    "v38_multiseed_result.json",
)

# 4. Statistical analysis: Wilson CI, two-proportion z, Bonferroni alpha=0.0125.
ANALYSIS_TOKENS = (
    "ALPHA = 0.0125",
    "def wilson(",
    "def ztest(",
    "Bonferroni-anlaml",
)

# 5. Artifact download bundle (Colab).
DOWNLOAD_TOKENS = (
    "DOWNLOAD = True",
    "zipfile.ZipFile",
    "sncp_ppo_v38_final_artifacts.zip",
    "files.download(bundle)",
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


def _status(notes: Sequence[str]) -> str:
    return "fail" if any(note.startswith("FAIL:") for note in notes) else "pass"


def verify_v16_run_ready(repo_root: str | Path = ".") -> V16RunReadinessSummary:
    repo_root = Path(repo_root)
    notes: list[str] = []

    required_files = (
        repo_root / "sncp_ppo_colab.ipynb",
        repo_root / "sncp_ppo" / "action_shield.py",
        repo_root / "sncp_ppo" / "train.py",
        repo_root / "sncp_ppo" / "eval_report.py",
    )
    for path in required_files:
        if not path.exists():
            notes.append(f"FAIL: missing required file `{path}`")

    cells = _load_notebook(repo_root / "sncp_ppo_colab.ipynb") or []
    training_cell = _find_unique_cell(
        cells, "--save_path checkpoints/sncp_ppo_v34.pt", notes, "v34 training")
    sweep_cell = _find_unique_cell(
        cells, "def honest_sweep(", notes, "honest multi-seed sweep")
    analysis_cell = _find_unique_cell(
        cells, "ALPHA = 0.0125", notes, "statistical analysis")
    download_cell = _find_unique_cell(
        cells, "DOWNLOAD = True", notes, "artifact download")

    _check_tokens(training_cell, TRAINING_TOKENS, notes, "v34 training")
    _check_tokens(sweep_cell, SWEEP_TOKENS, notes, "honest multi-seed sweep")
    _check_tokens(analysis_cell, ANALYSIS_TOKENS, notes, "statistical analysis")
    _check_tokens(download_cell, DOWNLOAD_TOKENS, notes, "artifact download")

    if not notes:
        notes.append("PASS: final v34-train + v38-shield Colab pipeline is ready")

    return V16RunReadinessSummary(
        status=_status(notes),
        repo_root=str(repo_root),
        training_cell_found=training_cell is not None,
        evaluation_cell_found=sweep_cell is not None and analysis_cell is not None,
        baseline_densities=(),
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
