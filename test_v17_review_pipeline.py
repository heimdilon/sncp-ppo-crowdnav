import json
from pathlib import Path

from sncp_ppo.v17_review_pipeline import run_v17_review


def test_v17_review_pipeline_stages_runs_decides_and_gates(tmp_path):
    staging_dir = tmp_path / "colabout"
    repo_root = tmp_path / "repo"
    staging_dir.mkdir()
    (staging_dir / "sncp_ppo_v17.pt").write_bytes(b"checkpoint")
    (staging_dir / "training_20260608_070945.csv").write_text("csv\n", encoding="utf-8")
    calls = []

    def fake_post_eval(**kwargs):
        calls.append(("post_eval", kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "density_sweep.json").write_text("{}", encoding="utf-8")
        (output_dir / "training_diagnostics.json").write_text("{}", encoding="utf-8")
        return type(
            "PostResult",
            (),
            {
                "status": "fail",
                "output_dir": output_dir,
                "comparison_report": output_dir / "comparison_vs_v15.md",
                "training_report": output_dir / "training_diagnostics.md",
                "artifact_report": output_dir / "artifact_verification.md",
            },
        )()

    def fake_select(candidate_json, *, baseline_json, training_json):
        calls.append(("select", candidate_json, baseline_json, training_json))
        return type(
            "Decision",
            (),
            {
                "branch_id": "A_TIMEOUT_MAX_TIME60",
                "status": "ready_for_review",
                "single_variable": "max_time 50 -> 60",
                "reasons": ("N=1 timeout dominates collision.",),
                "manual_checks": (),
                "metrics": {},
            },
        )()

    def fake_write_decision_report(decision, path):
        Path(path).write_text(f"Branch: {decision.branch_id}\n", encoding="utf-8")

    def fake_write_decision_json(decision, path):
        Path(path).write_text(
            json.dumps(
                {
                    "status": decision.status,
                    "branch_id": decision.branch_id,
                    "single_variable": decision.single_variable,
                }
            ),
            encoding="utf-8",
        )

    def fake_verify(**kwargs):
        calls.append(("gate", kwargs))
        return type(
            "Gate",
            (),
            {
                "status": "pass",
                "branch_id": "A_TIMEOUT_MAX_TIME60",
                "single_variable": "max_time 50 -> 60",
            },
        )()

    def fake_write_gate_report(summary, path):
        Path(path).write_text(f"Overall status: {summary.status}\n", encoding="utf-8")

    result = run_v17_review(
        staging_dir=staging_dir,
        repo_root=repo_root,
        stage_artifacts=True,
        post_eval_runner=fake_post_eval,
        decision_selector=fake_select,
        decision_report_writer=fake_write_decision_report,
        decision_json_writer=fake_write_decision_json,
        gate_verifier=fake_verify,
        gate_report_writer=fake_write_gate_report,
    )

    assert result.status == "pass"
    assert result.post_eval_status == "fail"
    assert result.branch_id == "A_TIMEOUT_MAX_TIME60"
    assert result.training_csv == repo_root / "logs" / "training_20260608_070945.csv"
    assert result.checkpoint_path == repo_root / "checkpoints" / "sncp_ppo_v17.pt"
    assert result.output_dir == repo_root / "eval_v17"
    assert (repo_root / "eval_v17" / "v18_decision.md").exists()
    assert (repo_root / "eval_v17" / "v18_ready.md").exists()
    assert calls[0][0] == "post_eval"
    assert calls[1][0] == "select"
    assert calls[2][0] == "gate"


def test_v17_review_cli_returns_gate_status(tmp_path, monkeypatch, capsys):
    import run_v17_review

    captured = {}

    def fake_run_v17_review(**kwargs):
        captured.update(kwargs)
        return run_v17_review.V17ReviewResult(
            status="fail",
            checkpoint_path=Path("checkpoints/sncp_ppo_v17.pt"),
            training_csv=Path("logs/training_20260608_070945.csv"),
            output_dir=Path("eval_v17"),
            post_eval_status="fail",
            branch_id="WAIT_FOR_ARTIFACTS",
            single_variable="none",
            decision_report=Path("eval_v17/v18_decision.md"),
            gate_report=Path("eval_v17/v18_ready.md"),
        )

    monkeypatch.setattr(run_v17_review, "run_v17_review", fake_run_v17_review)

    exit_code = run_v17_review.main(
        [
            "--stage_colab",
            "--staging_dir",
            str(tmp_path / "colabout"),
            "--repo_root",
            str(tmp_path / "repo"),
            "--overwrite",
        ]
    )

    assert exit_code == 1
    assert captured["stage_artifacts"] is True
    assert captured["overwrite"] is True
    assert "Branch: WAIT_FOR_ARTIFACTS" in capsys.readouterr().out
