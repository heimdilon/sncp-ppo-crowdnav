import json
from pathlib import Path

from sncp_ppo.run_readiness import verify_v16_run_ready, write_readiness_report


def _source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def test_final_pipeline_run_readiness_passes_current_repo():
    summary = verify_v16_run_ready(Path("."))

    assert summary.status == "pass"
    assert summary.training_cell_found is True
    assert summary.evaluation_cell_found is True
    assert summary.baseline_densities == ()


def test_final_pipeline_run_readiness_flags_stale_notebook(tmp_path):
    # A stale notebook (old per-version shield-probe cells, no full v34 training and no
    # inline honest sweep) must be flagged: the final entry point trains v34 then sweeps
    # the raw policy and the shielded system.
    notebook = {
        "cells": [
            {"cell_type": "code", "source": "BASE_CHECKPOINT = 'sncp_ppo_v34.pt'\n"},
            {"cell_type": "code", "source": "CHECKPOINT = 'sncp_ppo_v34.pt'\nreport = 'x'\n"},
        ]
    }
    (tmp_path / "sncp_ppo_colab.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    package_dir = tmp_path / "sncp_ppo"
    package_dir.mkdir()
    for name in ("action_shield.py", "train.py", "eval_report.py"):
        (package_dir / name).write_text("stub\n", encoding="utf-8")

    summary = verify_v16_run_ready(tmp_path)

    assert summary.status == "fail"
    assert any("v34 training" in note for note in summary.notes)
    assert any("honest multi-seed sweep" in note for note in summary.notes)
    assert any("statistical analysis" in note for note in summary.notes)


def test_write_readiness_report(tmp_path):
    summary = verify_v16_run_ready(Path("."))
    output = tmp_path / "readiness.md"

    write_readiness_report(summary, output)

    report = output.read_text(encoding="utf-8")
    assert "Overall status: pass" in report
    assert "Baseline densities: n/a" in report


def test_colab_download_cell_bundles_final_artifacts():
    notebook = json.loads(Path("sncp_ppo_colab.ipynb").read_text(encoding="utf-8"))
    code_sources = [
        _source_text(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    persist_cells = [source for source in code_sources if "DOWNLOAD = True" in source]

    assert len(persist_cells) == 1
    persist_cell = persist_cells[0]
    assert "zipfile.ZipFile" in persist_cell
    assert "sncp_ppo_v38_final_artifacts.zip" in persist_cell
    assert "checkpoints/sncp_ppo_v34.pt" in persist_cell
    assert "v38_multiseed_result.json" in persist_cell
    assert "files.download(bundle)" in persist_cell
