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


def test_colab_v19_eval_cell_uses_post_run_pipeline():
    code_sources = _colab_code_sources()
    eval_cells = [source for source in code_sources if "CHECKPOINT = 'checkpoints/sncp_ppo_v19.pt'" in source]

    assert len(eval_cells) == 1
    eval_cell = eval_cells[0]
    assert "EVAL_OUT = 'eval_v19'" in eval_cell
    assert "run_post_eval.py" in eval_cell
    assert "'--version', '19'" in eval_cell
    assert "run_v16_post_eval.py" not in eval_cell
    assert "evaluate_policy_report.py" not in eval_cell
    assert "compare_policy_reports.py" not in eval_cell


def test_colab_v19_training_cell_fails_fast_and_preserves_single_variable_config():
    code_sources = _colab_code_sources()
    train_cells = [source for source in code_sources if "SAVE_PATH = 'checkpoints/sncp_ppo_v19.pt'" in source]

    assert len(train_cells) == 1
    train_cell = train_cells[0]
    assert "TOTAL_STEPS = 2_500_000" in train_cell
    assert "NUM_ENVS = 16" in train_cell
    assert "HORIZON = 128" in train_cell
    assert "REPLAY_RATIO = 0.20" in train_cell
    assert "COMFORT_COEFF = 6.0" in train_cell
    assert "MAX_TIME = 50.0" in train_cell
    assert "'--curriculum_replay_ratio', str(REPLAY_RATIO)" in train_cell
    assert "'--comfort_coeff', str(COMFORT_COEFF)" in train_cell
    assert "'--max_time', str(MAX_TIME)" in train_cell
    assert "'--num_humans', '10'" in train_cell
    assert "'--holdout_scenarios', 'easy', 'hard', 'circle'" in train_cell
    assert "'--holdout_episodes', '50'" in train_cell
    assert "'--save_path', SAVE_PATH" in train_cell
    assert "if p.returncode != 0:" in train_cell
    assert "raise SystemExit(p.returncode)" in train_cell
