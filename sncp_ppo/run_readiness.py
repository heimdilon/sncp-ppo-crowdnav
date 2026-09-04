"""Preflight checks for the final end-to-end Colab notebook.

The final notebook is a self-contained v39 pipeline:
  1. optional v34 Beta warm-start (if the base checkpoint is missing),
  2. v39 smoke + full train (risk head + Lagrangian PPO, shield not used),
  3. honest 5-seed sweep of v39 with action_shield=False (recommended deploy),
     plus an optional C0/C1 oracle (v34 raw / v34+v38 shield),
  4. statistical analysis (Wilson / two-proportion z / Bonferroni),
  5. artifact download bundle.

This module verifies those cells exist with the expected tokens, so an
expensive Colab run is not launched against a stale notebook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# 2. Optional v34 warm-start (kept so a machine without sncp_ppo_v34.pt can still train).
V34_TRAINING_TOKENS = (
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

# 2b. v39 learned system: risk head + Lagrangian PPO, v34 init, no runtime shield.
TRAINING_TOKENS = (
    "python -m sncp_ppo.train",
    "--risk_head --lagrange_ppo",
    "--risk_horizon 6",
    "--risk_bce_coef 1.0 --risk_clearance_coef 0.1",
    "--lagrange_cost_limit 0.05 --lagrange_lr 0.01",
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
    "--init_checkpoint checkpoints/sncp_ppo_v34.pt",
    "--save_path checkpoints/sncp_ppo_v39.pt",
)

SMOKE_TOKENS = (
    "python -m sncp_ppo.train",
    "--risk_head --lagrange_ppo",
    "--total_steps 64",
    "--save_path checkpoints/sncp_ppo_v39_smoke.pt",
)

# 3. Honest 5-seed sweep: primary C2 = v39 shield-off; optional C0/C1 oracle.
SWEEP_TOKENS = (
    "from sncp_ppo.eval_report import evaluate_density",
    "def honest_sweep(",
    "action_shield=action_shield",
    "shield_horizon_steps=6",
    "SEEDS = [100, 200, 300, 400, 500]",
    "DENSITIES = [5, 10, 15, 20]",
    "action_shield=False",
    "checkpoints/sncp_ppo_v39.pt",
    "v39_multiseed_result.json",
    "RUN_ORACLE_MATRIX",
    "action_shield=True",
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
    "sncp_ppo_v39_final_artifacts.zip",
    "checkpoints/sncp_ppo_v39.pt",
    "v39_multiseed_result.json",
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


def _find_unique_cell(
    cells: Sequence[str],
    marker: str,
    notes: list[str],
    name: str,
    *,
    exclude: str | None = None,
) -> str | None:
    matches = [source for source in cells if marker in source]
    if exclude is not None:
        matches = [source for source in matches if exclude not in source]
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
        repo_root / "sncp_ppo" / "risk_labeler.py",
        repo_root / "sncp_ppo" / "train.py",
        repo_root / "sncp_ppo" / "eval_report.py",
    )
    for path in required_files:
        if not path.exists():
            notes.append(f"FAIL: missing required file `{path}`")

    cells = _load_notebook(repo_root / "sncp_ppo_colab.ipynb") or []
    # Smoke save path is a prefix of this marker (`...v39.pt` ⊂ `...v39_smoke.pt`).
    training_cell = _find_unique_cell(
        cells, "--save_path checkpoints/sncp_ppo_v39.pt", notes, "v39 training",
        exclude="sncp_ppo_v39_smoke.pt")
    v34_cell = _find_unique_cell(
        cells, "--save_path checkpoints/sncp_ppo_v34.pt", notes, "optional v34 training")
    smoke_cell = _find_unique_cell(
        cells, "--save_path checkpoints/sncp_ppo_v39_smoke.pt", notes, "v39 smoke")
    sweep_cell = _find_unique_cell(
        cells, "def honest_sweep(", notes, "honest multi-seed sweep")
    analysis_cell = _find_unique_cell(
        cells, "ALPHA = 0.0125", notes, "statistical analysis")
    download_cell = _find_unique_cell(
        cells, "DOWNLOAD = True", notes, "artifact download")

    _check_tokens(training_cell, TRAINING_TOKENS, notes, "v39 training")
    _check_tokens(v34_cell, V34_TRAINING_TOKENS, notes, "optional v34 training")
    _check_tokens(smoke_cell, SMOKE_TOKENS, notes, "v39 smoke")
    _check_tokens(sweep_cell, SWEEP_TOKENS, notes, "honest multi-seed sweep")
    _check_tokens(analysis_cell, ANALYSIS_TOKENS, notes, "statistical analysis")
    _check_tokens(download_cell, DOWNLOAD_TOKENS, notes, "artifact download")

    setup_hits = [source for source in cells if "git clone" in source]
    if len(setup_hits) != 1:
        notes.append(f"FAIL: expected exactly one setup/clone cell, found {len(setup_hits)}")
    else:
        setup = setup_hits[0]
        if "cursor/v39-risk-head-lagrangian-6377" not in setup:
            notes.append("FAIL: setup cell missing v39 branch checkout")
        if "requirements.txt" not in setup:
            notes.append("FAIL: setup cell missing `requirements.txt`")

    if not notes:
        notes.append("PASS: final v39 Colab pipeline is ready (shield-off deploy)")

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
