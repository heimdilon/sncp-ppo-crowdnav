import json

from sncp_ppo.artifact_verifier import verify_v16_artifacts, write_artifact_verification_report


def _write_complete_eval_dir(
    eval_dir,
    *,
    verdict="pass",
    collapse=False,
    replay_ratio=0.18,
    readiness="pass",
):
    eval_dir.mkdir(parents=True)
    if readiness is not None:
        (eval_dir / "run_readiness.md").write_text(
            f"# readiness\n\nOverall status: {readiness}\n",
            encoding="utf-8",
        )
    (eval_dir / "density_sweep.csv").write_text("stub\n", encoding="utf-8")
    (eval_dir / "density_sweep.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 128)
    (eval_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (eval_dir / "comparison_vs_v15.md").write_text(
        f"# comparison\n\nOverall verdict: {verdict}\n",
        encoding="utf-8",
    )
    (eval_dir / "training_diagnostics.md").write_text("# diagnostics\n", encoding="utf-8")
    (eval_dir / "traj_hard_n5.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"5" * 128)
    (eval_dir / "traj_hard_n10.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"10" * 128)
    (eval_dir / "density_sweep.json").write_text(
        json.dumps(
            {
                "checkpoint": "checkpoints/sncp_ppo_v16.pt",
                "baseline_nav_steps": 121.5,
                "trajectory_files": ["traj_hard_n5.png", "traj_hard_n10.png"],
                "density_sweep": [
                    {
                        "num_humans": n,
                        "scenario": "hard",
                        "episodes": 50,
                        "success_rate": 0.5,
                        "collision_rate": 0.2,
                        "timeout_rate": 0.3,
                        "avg_success_steps": 188.0,
                        "avg_episode_steps": 160.0,
                        "avg_i_sp": 0.02,
                        "avg_min_d_min": 0.8,
                        "avg_reward": 1.0,
                    }
                    for n in [1, 3, 5, 8, 10]
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "training_diagnostics.json").write_text(
        json.dumps(
            {
                "csv_path": "logs/training_v16.csv",
                "observed_replay_ratio": replay_ratio,
                "best_min_success": 0.5,
                "final_min_success": 0.45,
                "collapse_detected": collapse,
                "final_std_linear": 0.13,
                "final_std_angular": 0.22,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_verify_v16_artifacts_passes_complete_noncollapsed_run(tmp_path):
    checkpoint = tmp_path / "sncp_ppo_v16.pt"
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v16"
    _write_complete_eval_dir(eval_dir)

    summary = verify_v16_artifacts(checkpoint_path=checkpoint, eval_dir=eval_dir)

    assert summary.status == "pass"
    assert summary.missing_files == ()
    assert summary.densities == (1, 3, 5, 8, 10)
    assert summary.readiness_status == "pass"
    assert summary.comparison_verdict == "pass"
    assert summary.collapse_detected is False


def test_verify_v16_artifacts_fails_missing_checkpoint_and_required_files(tmp_path):
    eval_dir = tmp_path / "eval_v16"
    eval_dir.mkdir()

    summary = verify_v16_artifacts(
        checkpoint_path=tmp_path / "missing.pt",
        eval_dir=eval_dir,
    )

    assert summary.status == "fail"
    assert "checkpoint" in summary.missing_files
    assert "density_sweep.json" in summary.missing_files
    assert any("missing" in note for note in summary.notes)


def test_verify_v16_artifacts_requires_successful_run_readiness_report(tmp_path):
    checkpoint = tmp_path / "sncp_ppo_v16.pt"
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v16"
    _write_complete_eval_dir(eval_dir, readiness=None)

    summary = verify_v16_artifacts(checkpoint_path=checkpoint, eval_dir=eval_dir)

    assert summary.status == "fail"
    assert "run_readiness.md" in summary.missing_files


def test_verify_v16_artifacts_fails_nonpassing_run_readiness_report(tmp_path):
    checkpoint = tmp_path / "sncp_ppo_v16.pt"
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v16"
    _write_complete_eval_dir(eval_dir, readiness="fail")

    summary = verify_v16_artifacts(checkpoint_path=checkpoint, eval_dir=eval_dir)

    assert summary.status == "fail"
    assert summary.readiness_status == "fail"
    assert any("run readiness status is fail" in note for note in summary.notes)


def test_verify_v16_artifacts_fails_failed_comparison_or_training_collapse(tmp_path):
    checkpoint = tmp_path / "sncp_ppo_v16.pt"
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v16"
    _write_complete_eval_dir(eval_dir, verdict="fail", collapse=True)

    summary = verify_v16_artifacts(checkpoint_path=checkpoint, eval_dir=eval_dir)

    assert summary.status == "fail"
    assert summary.comparison_verdict == "fail"
    assert summary.collapse_detected is True
    assert any("comparison verdict is fail" in note for note in summary.notes)
    assert any("training collapse detected" in note for note in summary.notes)


def test_artifact_verification_report_summarizes_status(tmp_path):
    checkpoint = tmp_path / "sncp_ppo_v16.pt"
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = tmp_path / "eval_v16"
    _write_complete_eval_dir(eval_dir, verdict="warn")
    summary = verify_v16_artifacts(checkpoint_path=checkpoint, eval_dir=eval_dir)
    output = tmp_path / "artifact_check.md"

    write_artifact_verification_report(summary, output)

    report = output.read_text(encoding="utf-8")
    assert "Overall status: warn" in report
    assert "Run readiness: pass" in report
    assert "Comparison verdict: warn" in report
    assert "Replay ratio: 18.0%" in report


def test_verify_cli_writes_report_and_returns_nonzero_on_failure(tmp_path, capsys):
    import verify_v16_artifacts

    output = tmp_path / "artifact_verification.md"

    exit_code = verify_v16_artifacts.main(
        [
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--eval_dir",
            str(tmp_path / "missing_eval"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert output.exists()
    assert "Overall status: fail" in output.read_text(encoding="utf-8")
    assert "artifact_verification.md" in capsys.readouterr().out
