import json

from sncp_ppo.training_log_report import (
    analyze_training_log,
    write_training_diagnostic_report,
    write_training_diagnostic_json,
)


def _write_csv(path, rows):
    header = [
        "episode",
        "scenario",
        "num_humans",
        "human_vpref",
        "is_replay_update",
        "steps",
        "reward",
        "success",
        "collision",
        "timeout",
        "comfort",
        "entropy",
        "approx_kl",
        "std_linear",
        "std_angular",
        "return_rms_std",
        "is_best_checkpoint",
        "best_reason",
        "holdout_easy_success",
        "holdout_easy_collision",
        "holdout_easy_timeout",
        "holdout_easy_reward",
        "holdout_easy_avg_steps",
        "holdout_easy_avg_I_sp",
        "holdout_easy_min_d_min",
        "holdout_hard_success",
        "holdout_hard_collision",
        "holdout_hard_timeout",
        "holdout_hard_reward",
        "holdout_hard_avg_steps",
        "holdout_hard_avg_I_sp",
        "holdout_hard_min_d_min",
        "holdout_circle_success",
        "holdout_circle_collision",
        "holdout_circle_timeout",
        "holdout_circle_reward",
        "holdout_circle_avg_steps",
        "holdout_circle_avg_I_sp",
        "holdout_circle_min_d_min",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analyze_training_log_detects_holdout_collapse_and_replay_fraction(tmp_path):
    csv_path = tmp_path / "training.csv"
    _write_csv(
        csv_path,
        [
            {
                "episode": 100,
                "scenario": "easy",
                "num_humans": 1,
                "human_vpref": 0.13,
                "is_replay_update": 0,
                "std_linear": 0.10,
                "std_angular": 0.20,
                "holdout_easy_success": "nan",
                "holdout_hard_success": "nan",
                "holdout_circle_success": "nan",
            },
            {
                "episode": 200,
                "scenario": "medium",
                "num_humans": 5,
                "human_vpref": 0.22,
                "is_replay_update": 0,
                "std_linear": 0.11,
                "std_angular": 0.21,
                "is_best_checkpoint": 1,
                "best_reason": "best updated",
                "holdout_easy_success": 0.40,
                "holdout_hard_success": 0.30,
                "holdout_circle_success": 0.20,
            },
            {
                "episode": 300,
                "scenario": "hard",
                "num_humans": 8,
                "human_vpref": 0.24,
                "is_replay_update": 1,
                "std_linear": 0.12,
                "std_angular": 0.35,
                "is_best_checkpoint": 1,
                "best_reason": "best updated",
                "holdout_easy_success": 0.60,
                "holdout_easy_collision": 0.10,
                "holdout_easy_timeout": 0.30,
                "holdout_easy_reward": -1.0,
                "holdout_easy_avg_steps": 150.0,
                "holdout_easy_avg_I_sp": 0.010,
                "holdout_easy_min_d_min": 0.90,
                "holdout_hard_success": 0.50,
                "holdout_hard_collision": 0.20,
                "holdout_hard_timeout": 0.30,
                "holdout_hard_reward": -2.0,
                "holdout_hard_avg_steps": 170.0,
                "holdout_hard_avg_I_sp": 0.020,
                "holdout_hard_min_d_min": 0.70,
                "holdout_circle_success": 0.40,
                "holdout_circle_collision": 0.30,
                "holdout_circle_timeout": 0.30,
                "holdout_circle_reward": -3.0,
                "holdout_circle_avg_steps": 190.0,
                "holdout_circle_avg_I_sp": 0.030,
                "holdout_circle_min_d_min": 0.60,
            },
            {
                "episode": 400,
                "scenario": "circle",
                "num_humans": 10,
                "human_vpref": 0.26,
                "is_replay_update": 0,
                "std_linear": 0.16,
                "std_angular": 0.50,
                "is_best_checkpoint": 0,
                "holdout_easy_success": 0.10,
                "holdout_easy_collision": 0.05,
                "holdout_easy_timeout": 0.85,
                "holdout_easy_reward": -4.0,
                "holdout_easy_avg_steps": 199.0,
                "holdout_easy_avg_I_sp": 0.005,
                "holdout_easy_min_d_min": 1.10,
                "holdout_hard_success": 0.00,
                "holdout_hard_collision": 0.75,
                "holdout_hard_timeout": 0.25,
                "holdout_hard_reward": -18.0,
                "holdout_hard_avg_steps": 80.0,
                "holdout_hard_avg_I_sp": 0.040,
                "holdout_hard_min_d_min": 0.40,
                "holdout_circle_success": 0.00,
                "holdout_circle_collision": 0.90,
                "holdout_circle_timeout": 0.10,
                "holdout_circle_reward": -21.0,
                "holdout_circle_avg_steps": 65.0,
                "holdout_circle_avg_I_sp": 0.050,
                "holdout_circle_min_d_min": 0.35,
            },
        ],
    )

    summary = analyze_training_log(csv_path)

    assert summary.holdout_scenarios == ("easy", "hard", "circle")
    assert summary.total_rows == 4
    assert summary.evaluated_rows == 3
    assert summary.observed_replay_ratio == 0.25
    assert summary.best_step == 300
    assert summary.best_min_success == 0.40
    assert summary.best_success_by_scenario == {
        "easy": 0.60,
        "hard": 0.50,
        "circle": 0.40,
    }
    assert summary.final_step == 400
    assert summary.final_min_success == 0.0
    assert summary.collapse_delta == -0.40
    assert summary.collapse_detected is True
    assert summary.final_std_linear == 0.16
    assert summary.final_std_angular == 0.50
    assert summary.max_std_angular == 0.50
    assert summary.std_angular_delta == 0.30


def test_analyze_training_log_handles_old_logs_without_replay_column(tmp_path):
    csv_path = tmp_path / "training_old.csv"
    csv_path.write_text(
        "\n".join(
            [
                "episode,scenario,num_humans,human_vpref,steps,is_best_checkpoint,best_reason,holdout_easy_success,holdout_hard_success",
                "100,easy,1,0.13,128,0,,nan,nan",
                "200,hard,5,0.26,128,1,best updated,0.6,0.4",
                "300,circle,10,0.26,128,0,,0.2,0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyze_training_log(csv_path)

    assert summary.observed_replay_ratio is None
    assert summary.final_std_linear is None
    assert summary.final_std_angular is None
    assert summary.best_min_success == 0.4
    assert summary.final_min_success == 0.1
    assert summary.collapse_detected is True


def test_training_diagnostic_writers_include_best_final_and_collapse(tmp_path):
    csv_path = tmp_path / "training.csv"
    _write_csv(
        csv_path,
        [
            {
                "episode": 100,
                "scenario": "hard",
                "num_humans": 8,
                "human_vpref": 0.24,
                "is_replay_update": 1,
                "std_linear": 0.10,
                "std_angular": 0.20,
                "is_best_checkpoint": 1,
                "best_reason": "best updated",
                "holdout_easy_success": 0.50,
                "holdout_easy_collision": 0.10,
                "holdout_easy_timeout": 0.40,
                "holdout_easy_reward": -1.0,
                "holdout_easy_avg_steps": 150.0,
                "holdout_easy_avg_I_sp": 0.010,
                "holdout_easy_min_d_min": 0.90,
                "holdout_hard_success": 0.40,
                "holdout_hard_collision": 0.20,
                "holdout_hard_timeout": 0.40,
                "holdout_hard_reward": -2.0,
                "holdout_hard_avg_steps": 170.0,
                "holdout_hard_avg_I_sp": 0.020,
                "holdout_hard_min_d_min": 0.70,
                "holdout_circle_success": 0.30,
                "holdout_circle_collision": 0.30,
                "holdout_circle_timeout": 0.40,
                "holdout_circle_reward": -3.0,
                "holdout_circle_avg_steps": 190.0,
                "holdout_circle_avg_I_sp": 0.030,
                "holdout_circle_min_d_min": 0.60,
            },
            {
                "episode": 200,
                "scenario": "circle",
                "num_humans": 10,
                "human_vpref": 0.26,
                "is_replay_update": 0,
                "std_linear": 0.15,
                "std_angular": 0.45,
                "is_best_checkpoint": 0,
                "holdout_easy_success": 0.10,
                "holdout_easy_collision": 0.05,
                "holdout_easy_timeout": 0.85,
                "holdout_easy_reward": -4.0,
                "holdout_easy_avg_steps": 199.0,
                "holdout_easy_avg_I_sp": 0.005,
                "holdout_easy_min_d_min": 1.10,
                "holdout_hard_success": 0.00,
                "holdout_hard_collision": 0.75,
                "holdout_hard_timeout": 0.25,
                "holdout_hard_reward": -18.0,
                "holdout_hard_avg_steps": 80.0,
                "holdout_hard_avg_I_sp": 0.040,
                "holdout_hard_min_d_min": 0.40,
                "holdout_circle_success": 0.00,
                "holdout_circle_collision": 0.90,
                "holdout_circle_timeout": 0.10,
                "holdout_circle_reward": -21.0,
                "holdout_circle_avg_steps": 65.0,
                "holdout_circle_avg_I_sp": 0.050,
                "holdout_circle_min_d_min": 0.35,
            },
        ],
    )
    summary = analyze_training_log(csv_path)
    json_path = tmp_path / "training_diagnostics.json"
    report_path = tmp_path / "training_diagnostics.md"

    write_training_diagnostic_json(summary, json_path)
    write_training_diagnostic_report(summary, report_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["best_step"] == 100
    assert data["collapse_detected"] is True
    assert data["final_std_angular"] == 0.45
    assert data["final_collision_by_scenario"]["circle"] == 0.90
    assert data["final_timeout_by_scenario"]["easy"] == 0.85
    assert data["final_avg_steps_by_scenario"]["easy"] == 199.0
    assert data["final_avg_I_sp_by_scenario"]["circle"] == 0.050

    report = report_path.read_text(encoding="utf-8")
    assert "Collapse detected: yes" in report
    assert "Best min success: 30.0%" in report
    assert "Final min success: 0.0%" in report
    assert "Final std angular: 0.450" in report
    assert "## Per-Scenario Failure Profile" in report
    assert "| circle | 0.0% | 90.0% | 10.0% | 65.0 | 0.050 | 0.350 |" in report


def test_training_log_cli_writes_json_and_markdown(tmp_path, capsys):
    import analyze_training_log

    csv_path = tmp_path / "training.csv"
    _write_csv(
        csv_path,
        [
            {
                "episode": 100,
                "scenario": "hard",
                "num_humans": 8,
                "human_vpref": 0.24,
                "is_replay_update": 1,
                "std_linear": 0.11,
                "std_angular": 0.22,
                "is_best_checkpoint": 1,
                "best_reason": "best updated",
                "holdout_easy_success": 0.60,
                "holdout_hard_success": 0.50,
                "holdout_circle_success": 0.40,
            },
            {
                "episode": 200,
                "scenario": "circle",
                "num_humans": 10,
                "human_vpref": 0.26,
                "is_replay_update": 0,
                "std_linear": 0.16,
                "std_angular": 0.47,
                "is_best_checkpoint": 0,
                "holdout_easy_success": 0.10,
                "holdout_hard_success": 0.00,
                "holdout_circle_success": 0.00,
            },
        ],
    )
    output_dir = tmp_path / "diag"

    exit_code = analyze_training_log.main(
        [
            "--csv",
            str(csv_path),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "training_diagnostics.json").exists()
    assert (output_dir / "training_diagnostics.md").exists()
    assert "Collapse detected: yes" in (output_dir / "training_diagnostics.md").read_text(
        encoding="utf-8"
    )
    assert "training_diagnostics.md" in capsys.readouterr().out
