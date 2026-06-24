import csv
import json
from pathlib import Path

from sncp_ppo.eval_report import DensitySummary, write_summary_json
from sncp_ppo.post_run_pipeline import find_latest_training_csv, run_v16_post_eval


def _source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _colab_code_sources():
    notebook = json.loads(Path("sncp_ppo_colab.ipynb").read_text(encoding="utf-8"))
    return [
        _source_text(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def _write_training_csv(path):
    header = [
        "episode",
        "scenario",
        "num_humans",
        "human_vpref",
        "is_replay_update",
        "std_linear",
        "std_angular",
        "is_best_checkpoint",
        "best_reason",
        "holdout_easy_success",
        "holdout_hard_success",
        "holdout_circle_success",
    ]
    rows = []
    for idx, replay in enumerate([0, 1, 0, 0, 0], start=1):
        rows.append(
            {
                "episode": idx * 100,
                "scenario": "circle",
                "num_humans": 10,
                "human_vpref": 0.26,
                "is_replay_update": replay,
                "std_linear": 0.13,
                "std_angular": 0.22,
                "is_best_checkpoint": 1 if idx == 4 else 0,
                "best_reason": "best updated" if idx == 4 else "",
                "holdout_easy_success": 0.50,
                "holdout_hard_success": 0.46,
                "holdout_circle_success": 0.42,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_fake_report_outputs(output_dir, *, checkpoint_path, densities, scenario, **_kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_readiness.md").write_text(
        "# readiness\n\nOverall status: pass\n",
        encoding="utf-8",
    )
    summaries = [
        DensitySummary(
            num_humans=n,
            scenario=scenario,
            episodes=50,
            success_rate=0.70 if n == 10 else 0.60,
            collision_rate=0.20,
            timeout_rate=0.10,
            avg_success_steps=188.0,
            avg_episode_steps=170.0,
            avg_i_sp=0.015,
            avg_min_d_min=0.90,
            avg_reward=5.0,
        )
        for n in densities
    ]
    write_summary_json(
        summaries,
        output_dir / "density_sweep.json",
        checkpoint=str(checkpoint_path),
        baseline_nav_steps=121.5,
        trajectory_files=["traj_hard_n5.png", "traj_hard_n10.png"],
    )
    (output_dir / "density_sweep.csv").write_text("stub\n", encoding="utf-8")
    (output_dir / "density_sweep.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 128)
    (output_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (output_dir / "traj_hard_n5.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"5" * 128)
    (output_dir / "traj_hard_n10.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"10" * 128)
    return {}


def test_find_latest_training_csv_picks_newest_by_name(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    older = log_dir / "training_20260606_010101.csv"
    newer = log_dir / "training_20260607_020202.csv"
    older.write_text("old\n", encoding="utf-8")
    newer.write_text("new\n", encoding="utf-8")

    assert find_latest_training_csv(log_dir) == newer


def _write_fake_report_outputs_without_readiness(output_dir, *, checkpoint_path, densities, scenario, **_kwargs):
    """Like the full fake, but does NOT write run_readiness.md — models the real
    run_report (which never wrote it) plus an operator who skipped the preflight
    cell. The pipeline must then generate readiness itself so the bundle stays
    complete (v22 lesson: a skipped preflight should not fail an otherwise-good eval)."""
    _write_fake_report_outputs(
        output_dir, checkpoint_path=checkpoint_path, densities=densities, scenario=scenario
    )
    (output_dir / "run_readiness.md").unlink()


def test_run_v16_post_eval_generates_readiness_when_report_runner_skips_it(tmp_path):
    from sncp_ppo.run_readiness import V16RunReadinessSummary

    checkpoint = tmp_path / "checkpoints" / "sncp_ppo_v22.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "eval_v22"
    training_csv = tmp_path / "logs" / "training_20260612_073945.csv"
    training_csv.parent.mkdir()
    _write_training_csv(training_csv)
    baseline_path = tmp_path / "eval_v21" / "density_sweep.json"
    baseline_path.parent.mkdir()
    baseline = [
        DensitySummary(n, "hard", 50, 0.50, 0.30, 0.20, 50.0, 45.0, 0.015, 0.8, 1.0)
        for n in [1, 3, 5, 8, 10]
    ]
    write_summary_json(baseline, baseline_path, checkpoint="v21.pt", baseline_nav_steps=32.0)

    def fake_checker(root):
        return V16RunReadinessSummary(
            status="pass", repo_root=str(root), training_cell_found=True,
            evaluation_cell_found=True, baseline_densities=(1, 3, 5, 8, 10),
            notes=("PASS: generated by pipeline",),
        )

    result = run_v16_post_eval(
        checkpoint_path=checkpoint,
        training_csv=training_csv,
        output_dir=output_dir,
        baseline_json=baseline_path,
        baseline_nav_steps=32.0,
        nav_margin_steps=8.0,
        report_runner=_write_fake_report_outputs_without_readiness,
        readiness_checker=fake_checker,
    )

    assert (output_dir / "run_readiness.md").exists()
    assert "Overall status: pass" in (output_dir / "run_readiness.md").read_text(encoding="utf-8")
    assert result.status == "pass"


def test_run_v16_post_eval_writes_all_reports_and_verifies(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sncp_ppo_v16.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "eval_v16"
    training_csv = tmp_path / "logs" / "training_20260607_010101.csv"
    training_csv.parent.mkdir()
    _write_training_csv(training_csv)
    baseline_path = tmp_path / "eval_v15" / "density_sweep.json"
    baseline_path.parent.mkdir()
    baseline = [
        DensitySummary(n, "hard", 50, 0.50, 0.30, 0.20, 187.0, 160.0, 0.015, 0.8, 1.0)
        for n in [1, 3, 5, 8, 10]
    ]
    write_summary_json(baseline, baseline_path, checkpoint="v15.pt", baseline_nav_steps=121.5)

    result = run_v16_post_eval(
        checkpoint_path=checkpoint,
        training_csv=training_csv,
        output_dir=output_dir,
        baseline_json=baseline_path,
        report_runner=_write_fake_report_outputs,
    )

    assert result.status == "pass"
    assert (output_dir / "comparison_vs_v15.md").exists()
    assert (output_dir / "training_diagnostics.md").exists()
    assert (output_dir / "artifact_verification.md").exists()
    verification = (output_dir / "artifact_verification.md").read_text(encoding="utf-8")
    assert "Overall status: pass" in verification

    diagnostics = json.loads((output_dir / "training_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["observed_replay_ratio"] == 0.2


def test_post_eval_cli_uses_latest_training_csv_and_returns_status(tmp_path, monkeypatch, capsys):
    import run_v16_post_eval

    checkpoint = tmp_path / "sncp_ppo_v16.pt"
    checkpoint.write_bytes(b"checkpoint")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    latest = log_dir / "training_20260607_030303.csv"
    latest.write_text("header\n", encoding="utf-8")
    captured = {}

    def fake_run_v16_post_eval(**kwargs):
        captured.update(kwargs)
        return run_v16_post_eval.PostRunResult(
            status="warn",
            output_dir=kwargs["output_dir"],
            comparison_report=kwargs["output_dir"] / "comparison_vs_v15.md",
            training_report=kwargs["output_dir"] / "training_diagnostics.md",
            artifact_report=kwargs["output_dir"] / "artifact_verification.md",
        )

    monkeypatch.setattr(run_v16_post_eval, "run_v16_post_eval", fake_run_v16_post_eval)

    exit_code = run_v16_post_eval.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--log_dir",
            str(log_dir),
            "--output_dir",
            str(tmp_path / "eval_v16"),
        ]
    )

    assert exit_code == 0
    assert captured["training_csv"] == latest
    assert "Overall status: warn" in capsys.readouterr().out


def test_versioned_post_eval_cli_derives_paths_from_version(tmp_path, monkeypatch, capsys):
    import run_post_eval

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    latest = log_dir / "training_20260608_070945.csv"
    latest.write_text("header\n", encoding="utf-8")
    captured = {}

    def fake_run_post_eval(**kwargs):
        captured.update(kwargs)
        return run_post_eval.PostRunResult(
            status="pass",
            output_dir=kwargs["output_dir"],
            comparison_report=kwargs["output_dir"] / "comparison_vs_v15.md",
            training_report=kwargs["output_dir"] / "training_diagnostics.md",
            artifact_report=kwargs["output_dir"] / "artifact_verification.md",
        )

    monkeypatch.setattr(run_post_eval, "run_post_eval", fake_run_post_eval)

    exit_code = run_post_eval.main(["--version", "17", "--log_dir", str(log_dir)])

    assert exit_code == 0
    assert captured["checkpoint_path"] == Path("checkpoints/sncp_ppo_v17.pt")
    assert captured["output_dir"] == Path("eval_v17")
    assert captured["training_csv"] == latest
    assert "Overall status: pass" in capsys.readouterr().out


def test_notebook_is_v36_combined():
    # v36 = v30 base + ALL levers combined (deliberately multi-variable):
    # node-cap (v31) + reach/budget (v32) + multi-head (v33) + count-scaling (v29)
    # + Beta with tuned entropy (v34 + --ent_coef) + sense-range (v35).
    code = _colab_code_sources()
    train_cells = [s for s in code if "sncp_ppo.train" in s and "--fixed_scenario" in s]
    eval_cells = [s for s in code if "run_post_eval.py" in s]
    assert len(train_cells) == 1 and len(eval_cells) == 1
    train, ev = train_cells[0], eval_cells[0]
    assert "paper_challenging" in train
    assert "checkpoints/sncp_ppo_v36.pt" in train
    assert "'--pre_mlp'" in train                          # v27 carried forward
    assert "'--meanmax_pool'" in train                    # v30 carried forward
    assert "'--sense_range', '6.0'" in train               # v35
    assert "'--num_humans_range', '10', '25'" in train    # v32 reach
    assert "TOTAL_STEPS = 4_000_000" in train             # v32 budget
    assert "'--node_units', '256'" in train                # v31 capacity
    assert "'--node_output', '96'" in train
    assert "'--attn_heads', '4'" in train                  # v33 multi-head
    assert "'--attn_count_scaling'" in train               # v29 count-scaling
    assert "'--action_dist', 'beta'" in train              # v34 Beta
    assert "'--ent_coef', '0.001'" in train                # tuned entropy
    for tok in ("'--robot_vpref', '1.0'", "'--holdout_episodes', '50'"):
        assert tok in train, tok
    assert "'--version', '36'" in ev
    assert "'--baseline_nav_steps', '32'" in ev
    assert "'--max_time'" not in ev


def test_post_eval_cli_threads_regime_scaled_beeline_gate(tmp_path, monkeypatch):
    import run_post_eval

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "training_20260611_000000.csv").write_text("header\n", encoding="utf-8")
    captured = {}

    def fake_post_run(**kwargs):
        captured.update(kwargs)
        return run_post_eval.PostRunResult(
            status="pass",
            output_dir=kwargs["output_dir"],
            comparison_report=kwargs["output_dir"] / "comparison_vs_v15.md",
            training_report=kwargs["output_dir"] / "training_diagnostics.md",
            artifact_report=kwargs["output_dir"] / "artifact_verification.md",
        )

    monkeypatch.setattr(run_post_eval, "run_post_eval", fake_post_run)
    exit_code = run_post_eval.main(
        ["--version", "22", "--log_dir", str(log_dir),
         "--baseline_nav_steps", "32", "--nav_margin_steps", "8"]
    )

    assert exit_code == 0
    assert captured["baseline_nav_steps"] == 32.0
    assert captured["nav_margin_steps"] == 8.0
