import json

from sncp_ppo.v18_gate import verify_v18_decision_ready


PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def _write_density(path):
    path.write_text(
        json.dumps(
            {
                "density_sweep": [
                    {
                        "num_humans": n,
                        "episodes": 50,
                        "success_rate": 0.5,
                        "collision_rate": 0.2,
                        "timeout_rate": 0.3,
                        "avg_success_steps": 170.0,
                        "avg_i_sp": 0.01,
                    }
                    for n in [1, 3, 5, 8, 10]
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_complete_eval(eval_dir, *, decision_status="ready_for_review", branch_id="A_TIMEOUT_MAX_TIME60"):
    eval_dir.mkdir(parents=True)
    (eval_dir / "run_readiness.md").write_text("Overall status: pass\n", encoding="utf-8")
    (eval_dir / "density_sweep.csv").write_text("stub\n", encoding="utf-8")
    _write_density(eval_dir / "density_sweep.json")
    (eval_dir / "density_sweep.png").write_bytes(PNG)
    (eval_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (eval_dir / "comparison_vs_v15.md").write_text("Overall verdict: fail\n", encoding="utf-8")
    (eval_dir / "training_diagnostics.md").write_text("# diagnostics\n", encoding="utf-8")
    (eval_dir / "training_diagnostics.json").write_text(
        json.dumps({"collapse_detected": False, "observed_replay_ratio": 0.18}),
        encoding="utf-8",
    )
    (eval_dir / "artifact_verification.md").write_text("Overall status: fail\n", encoding="utf-8")
    (eval_dir / "traj_hard_n5.png").write_bytes(PNG)
    (eval_dir / "traj_hard_n10.png").write_bytes(PNG)
    (eval_dir / "v18_decision.md").write_text("# decision\n", encoding="utf-8")
    (eval_dir / "v18_decision.json").write_text(
        json.dumps(
            {
                "status": decision_status,
                "branch_id": branch_id,
                "single_variable": "max_time 50 -> 60",
                "reasons": ["N=1 timeout dominates collision."],
            }
        ),
        encoding="utf-8",
    )


def test_v18_gate_passes_complete_decision_artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sncp_ppo_v17.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v17"
    _write_complete_eval(eval_dir)

    summary = verify_v18_decision_ready(checkpoint_path=checkpoint, eval_dir=eval_dir)

    assert summary.status == "pass"
    assert summary.branch_id == "A_TIMEOUT_MAX_TIME60"
    assert summary.single_variable == "max_time 50 -> 60"
    assert summary.densities == (1, 3, 5, 8, 10)
    assert any("ready for v18 review" in note for note in summary.notes)


def test_v18_gate_fails_missing_artifacts(tmp_path):
    summary = verify_v18_decision_ready(
        checkpoint_path=tmp_path / "checkpoints" / "sncp_ppo_v17.pt",
        eval_dir=tmp_path / "eval_v17",
    )

    assert summary.status == "fail"
    assert "checkpoint" in summary.missing_files
    assert "v18_decision.json" in summary.missing_files
    assert any("missing required artifacts" in note for note in summary.notes)


def test_v18_gate_fails_wait_for_artifacts_decision(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sncp_ppo_v17.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v17"
    _write_complete_eval(eval_dir, decision_status="wait_for_artifacts", branch_id="WAIT_FOR_ARTIFACTS")

    summary = verify_v18_decision_ready(checkpoint_path=checkpoint, eval_dir=eval_dir)

    assert summary.status == "fail"
    assert summary.branch_id == "WAIT_FOR_ARTIFACTS"
    assert any("decision status is wait_for_artifacts" in note for note in summary.notes)


def test_v18_gate_fails_invalid_trajectory_png(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sncp_ppo_v17.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v17"
    _write_complete_eval(eval_dir)
    (eval_dir / "traj_hard_n10.png").write_bytes(b"not png")

    summary = verify_v18_decision_ready(checkpoint_path=checkpoint, eval_dir=eval_dir)

    assert summary.status == "fail"
    assert any("invalid PNG artifacts" in note for note in summary.notes)
    assert any("traj_hard_n10.png" in note for note in summary.notes)


def test_v18_gate_cli_writes_report_and_returns_status(tmp_path, capsys):
    import verify_v18_ready

    checkpoint = tmp_path / "checkpoints" / "sncp_ppo_v17.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v17"
    _write_complete_eval(eval_dir)
    output = tmp_path / "v18_ready.md"

    exit_code = verify_v18_ready.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--eval_dir",
            str(eval_dir),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert "Overall status: pass" in output.read_text(encoding="utf-8")
    assert "A_TIMEOUT_MAX_TIME60" in capsys.readouterr().out
