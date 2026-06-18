import math
import json

from sncp_ppo.eval_report import (
    compare_density_sweeps,
    DensitySummary,
    EpisodeResult,
    load_summary_json,
    plot_density_curves,
    run_report,
    summarize_density,
    write_comparison_report,
    write_markdown_report,
    write_summary_csv,
    write_summary_json,
)


def test_summarize_density_reports_success_nav_time_and_social_metrics():
    episodes = [
        EpisodeResult(
            success=True,
            collision=False,
            timeout=False,
            steps=180,
            total_reward=22.0,
            avg_i_sp=0.010,
            min_d_min=0.68,
        ),
        EpisodeResult(
            success=False,
            collision=True,
            timeout=False,
            steps=44,
            total_reward=-18.0,
            avg_i_sp=0.035,
            min_d_min=0.48,
        ),
        EpisodeResult(
            success=True,
            collision=False,
            timeout=False,
            steps=190,
            total_reward=20.0,
            avg_i_sp=0.015,
            min_d_min=0.72,
        ),
        EpisodeResult(
            success=False,
            collision=False,
            timeout=True,
            steps=201,
            total_reward=-2.0,
            avg_i_sp=0.020,
            min_d_min=0.58,
        ),
    ]

    summary = summarize_density(num_humans=5, scenario="hard", episodes=episodes)

    assert summary.num_humans == 5
    assert summary.scenario == "hard"
    assert summary.episodes == 4
    assert summary.success_rate == 0.5
    assert summary.collision_rate == 0.25
    assert summary.timeout_rate == 0.25
    assert summary.avg_success_steps == 185.0
    assert summary.avg_episode_steps == 153.75
    assert math.isclose(summary.avg_i_sp, 0.020)
    assert math.isclose(summary.avg_min_d_min, 0.615)
    assert math.isclose(summary.avg_reward, 5.5)


def test_report_files_include_real_avoidance_gates(tmp_path):
    summaries = [
        DensitySummary(
            num_humans=1,
            scenario="hard",
            episodes=50,
            success_rate=0.44,
            collision_rate=0.30,
            timeout_rate=0.26,
            avg_success_steps=188.2,
            avg_episode_steps=173.4,
            avg_i_sp=0.009,
            avg_min_d_min=0.71,
            avg_reward=4.2,
        ),
        DensitySummary(
            num_humans=10,
            scenario="hard",
            episodes=50,
            success_rate=0.46,
            collision_rate=0.46,
            timeout_rate=0.08,
            avg_success_steps=187.6,
            avg_episode_steps=143.2,
            avg_i_sp=0.025,
            avg_min_d_min=0.55,
            avg_reward=3.8,
        ),
    ]

    csv_path = tmp_path / "density_sweep.csv"
    report_path = tmp_path / "report.md"
    write_summary_csv(summaries, csv_path)
    write_markdown_report(
        summaries,
        report_path,
        checkpoint="checkpoints/sncp_ppo_v16.pt",
        baseline_nav_steps=121.5,
        trajectory_files=["traj_v16_hard_n5.png", "traj_v16_hard_n10.png"],
    )

    csv_text = csv_path.read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == (
        "num_humans,scenario,episodes,success_rate,collision_rate,timeout_rate,"
        "avg_success_steps,avg_episode_steps,avg_i_sp,avg_min_d_min,avg_reward"
    )
    assert "10,hard,50,0.4600,0.4600,0.0800,187.6000,143.2000,0.0250,0.5500,3.8000" in csv_text

    report_text = report_path.read_text(encoding="utf-8")
    assert "checkpoints/sncp_ppo_v16.pt" in report_text
    assert "No-beeline check" in report_text
    assert "121.5" in report_text
    assert "traj_v16_hard_n10.png" in report_text


def test_plot_density_curves_creates_png_with_nav_time_baseline(tmp_path):
    summaries = [
        DensitySummary(
            num_humans=1,
            scenario="hard",
            episodes=10,
            success_rate=0.4,
            collision_rate=0.2,
            timeout_rate=0.4,
            avg_success_steps=180.0,
            avg_episode_steps=165.0,
            avg_i_sp=0.010,
            avg_min_d_min=0.70,
            avg_reward=2.0,
        ),
        DensitySummary(
            num_humans=10,
            scenario="hard",
            episodes=10,
            success_rate=0.5,
            collision_rate=0.4,
            timeout_rate=0.1,
            avg_success_steps=190.0,
            avg_episode_steps=150.0,
            avg_i_sp=0.030,
            avg_min_d_min=0.56,
            avg_reward=4.0,
        ),
    ]

    output = tmp_path / "density_sweep.png"
    plot_density_curves(summaries, output, baseline_nav_steps=121.5)

    assert output.exists()
    assert output.stat().st_size > 1000


def test_run_report_writes_sweep_artifacts_and_trajectory_manifest(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    calls = []

    def fake_evaluator(*, checkpoint_path, num_humans, scenario, n_episodes, seed, **_kwargs):
        calls.append((checkpoint_path, num_humans, scenario, n_episodes, seed))
        return [
            EpisodeResult(
                success=True,
                collision=False,
                timeout=False,
                steps=180 + num_humans,
                total_reward=20.0,
                avg_i_sp=0.01 * num_humans,
                min_d_min=0.70,
            ),
            EpisodeResult(
                success=False,
                collision=True,
                timeout=False,
                steps=60,
                total_reward=-20.0,
                avg_i_sp=0.02 * num_humans,
                min_d_min=0.50,
            ),
        ]

    def fake_trajectory_renderer(*, checkpoint_path, output_path, num_humans, scenario, seed, **_kwargs):
        output_path.write_text(
            f"{checkpoint_path.name} {num_humans} {scenario} {seed}",
            encoding="utf-8",
        )

    artifacts = run_report(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "eval_v16",
        densities=[1, 5],
        scenario="hard",
        n_episodes=2,
        seed=100,
        trajectory_densities=[5],
        evaluator=fake_evaluator,
        trajectory_renderer=fake_trajectory_renderer,
    )

    assert calls == [
        (checkpoint, 1, "hard", 2, 100),
        (checkpoint, 5, "hard", 2, 100),
    ]
    assert artifacts["csv"].exists()
    assert artifacts["json"].exists()
    assert artifacts["plot"].exists()
    assert artifacts["report"].exists()
    assert artifacts["trajectories"] == [tmp_path / "eval_v16" / "traj_hard_n5.png"]

    data = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert [row["num_humans"] for row in data["density_sweep"]] == [1, 5]
    assert data["checkpoint"] == str(checkpoint)

    report_text = artifacts["report"].read_text(encoding="utf-8")
    assert "traj_hard_n5.png" in report_text
    assert "No-beeline check" in report_text


def test_cli_main_passes_arguments_to_report_runner(tmp_path, monkeypatch, capsys):
    import evaluate_policy_report

    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    captured = {}

    def fake_run_report(**kwargs):
        captured.update(kwargs)
        return {
            "csv": tmp_path / "density_sweep.csv",
            "json": tmp_path / "density_sweep.json",
            "plot": tmp_path / "density_sweep.png",
            "report": tmp_path / "report.md",
            "trajectories": [tmp_path / "traj_hard_n5.png"],
        }

    monkeypatch.setattr(evaluate_policy_report, "run_report", fake_run_report)

    exit_code = evaluate_policy_report.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(tmp_path / "eval"),
            "--densities",
            "1",
            "5",
            "10",
            "--scenario",
            "hard",
            "--n_episodes",
            "50",
            "--seed",
            "100",
            "--trajectory_densities",
            "5",
            "10",
        ]
    )

    assert exit_code == 0
    assert captured["checkpoint_path"] == checkpoint
    assert captured["output_dir"] == tmp_path / "eval"
    assert captured["densities"] == [1, 5, 10]
    assert captured["scenario"] == "hard"
    assert captured["n_episodes"] == 50
    assert captured["seed"] == 100
    assert captured["trajectory_densities"] == [5, 10]
    assert "density_sweep.csv" in capsys.readouterr().out


def test_compare_density_sweeps_passes_when_v16_preserves_avoidance_and_improves_high_density():
    baseline = [
        DensitySummary(5, "hard", 50, 0.66, 0.18, 0.16, 187.1, 169.6, 0.015, 1.02, 5.3),
        DensitySummary(10, "hard", 50, 0.46, 0.46, 0.08, 188.9, 127.4, 0.025, 0.74, -4.4),
    ]
    candidate = [
        DensitySummary(5, "hard", 50, 0.68, 0.16, 0.16, 186.5, 170.0, 0.016, 1.00, 6.0),
        DensitySummary(10, "hard", 50, 0.58, 0.34, 0.08, 190.0, 150.0, 0.024, 0.78, 2.0),
    ]

    comparison = compare_density_sweeps(baseline, candidate, baseline_nav_steps=121.5)

    assert comparison.overall_status == "pass"
    high_density = comparison.rows[-1]
    assert high_density.num_humans == 10
    assert high_density.success_delta == 0.12
    assert high_density.collision_delta == -0.12
    assert high_density.nav_margin_vs_beeline == 68.5
    assert high_density.status == "pass"


def test_compare_density_sweeps_fails_beeline_regression_even_with_higher_success():
    baseline = [
        DensitySummary(10, "hard", 50, 0.46, 0.46, 0.08, 188.9, 127.4, 0.025, 0.74, -4.4),
    ]
    candidate = [
        DensitySummary(10, "hard", 50, 0.80, 0.10, 0.10, 124.0, 124.0, 0.010, 1.20, 10.0),
    ]

    comparison = compare_density_sweeps(baseline, candidate, baseline_nav_steps=121.5)

    assert comparison.overall_status == "fail"
    assert comparison.rows[0].status == "fail"
    assert any("beeline" in note for note in comparison.rows[0].notes)


def test_compare_density_sweeps_warns_when_high_density_success_does_not_improve():
    baseline = [
        DensitySummary(5, "hard", 50, 0.66, 0.18, 0.16, 187.1, 169.6, 0.015, 1.02, 5.3),
        DensitySummary(10, "hard", 50, 0.46, 0.46, 0.08, 188.9, 127.4, 0.025, 0.74, -4.4),
    ]
    candidate = [
        DensitySummary(5, "hard", 50, 0.66, 0.18, 0.16, 187.0, 169.6, 0.015, 1.02, 5.3),
        DensitySummary(10, "hard", 50, 0.46, 0.40, 0.14, 189.0, 140.0, 0.025, 0.76, -2.0),
    ]

    comparison = compare_density_sweeps(baseline, candidate, baseline_nav_steps=121.5)

    assert comparison.overall_status == "warn"
    assert comparison.rows[-1].status == "warn"
    assert any("high-density success did not improve" in note for note in comparison.rows[-1].notes)


def test_compare_density_sweeps_fails_timeout_freezing_regression():
    baseline = [
        DensitySummary(5, "hard", 50, 0.66, 0.18, 0.16, 187.1, 169.6, 0.015, 1.02, 5.3),
        DensitySummary(10, "hard", 50, 0.46, 0.46, 0.08, 188.9, 127.4, 0.025, 0.74, -4.4),
    ]
    candidate = [
        DensitySummary(5, "hard", 50, 0.66, 0.00, 0.50, 190.0, 190.0, 0.014, 1.10, 3.0),
        DensitySummary(10, "hard", 50, 0.58, 0.34, 0.08, 190.0, 150.0, 0.024, 0.78, 2.0),
    ]

    comparison = compare_density_sweeps(baseline, candidate, baseline_nav_steps=121.5)

    assert comparison.overall_status == "fail"
    sparse = comparison.rows[0]
    assert sparse.timeout_delta == 0.34
    assert sparse.status == "fail"
    assert any("timeout/freezing" in note for note in sparse.notes)


def test_compare_density_sweeps_isp_rise_is_not_a_regression_when_success_clearly_improves():
    """v22 lesson: I_sp rising is a BYPRODUCT of higher success (the robot makes
    more close-but-safe passes), not a comfort regression — as long as collision
    did not rise. When success clearly improves, the I_sp gate must not fail/warn.
    Models v22 N=8 (success +6pp, I_sp +0.0228, collision flat)."""
    baseline = [
        DensitySummary(8, "hard", 50, 0.32, 0.48, 0.20, 50.0, 45.0, 0.015, 0.85, 4.0),
    ]
    candidate = [
        DensitySummary(8, "hard", 50, 0.38, 0.48, 0.16, 52.3, 41.1, 0.0378, 0.85, 6.2),
    ]

    comparison = compare_density_sweeps(
        baseline, candidate, baseline_nav_steps=32.0, nav_margin_steps=8.0
    )

    row = comparison.rows[0]
    assert row.i_sp_delta > 0.02  # would have tripped the old absolute gate
    assert row.status == "pass", row.notes
    assert not any("I_sp" in note for note in row.notes)


def test_compare_density_sweeps_isp_rise_still_fails_when_success_does_not_improve():
    """The I_sp gate stays meaningful: if success did NOT improve but I_sp rose
    materially, that IS a comfort regression (robot got bolder for no gain)."""
    baseline = [
        DensitySummary(5, "hard", 50, 0.66, 0.18, 0.16, 50.0, 45.0, 0.015, 1.02, 5.3),
    ]
    candidate = [
        DensitySummary(5, "hard", 50, 0.66, 0.18, 0.16, 52.0, 46.0, 0.045, 1.00, 5.0),
    ]

    comparison = compare_density_sweeps(
        baseline, candidate, baseline_nav_steps=32.0, nav_margin_steps=8.0
    )

    row = comparison.rows[0]
    assert row.status == "fail"
    assert any("I_sp" in note for note in row.notes)


def test_summary_json_round_trips_and_comparison_report_mentions_verdict(tmp_path):
    baseline = [
        DensitySummary(10, "hard", 50, 0.46, 0.46, 0.08, 188.9, 127.4, 0.025, 0.74, -4.4),
    ]
    candidate = [
        DensitySummary(10, "hard", 50, 0.58, 0.34, 0.08, 190.0, 150.0, 0.024, 0.78, 2.0),
    ]
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.md"
    write_summary_json(baseline, baseline_path, checkpoint="v15.pt", baseline_nav_steps=121.5)
    write_summary_json(candidate, candidate_path, checkpoint="v16.pt", baseline_nav_steps=121.5)

    comparison = compare_density_sweeps(
        load_summary_json(baseline_path),
        load_summary_json(candidate_path),
        baseline_nav_steps=121.5,
    )
    write_comparison_report(
        comparison,
        output_path,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )

    report = output_path.read_text(encoding="utf-8")
    assert "Overall verdict: pass" in report
    assert "v15.pt" not in report
    assert "baseline.json" in report
    assert "| 10 | 46.0% | 58.0% | +12.0 pp |" in report


def test_compare_cli_loads_reports_and_writes_comparison(tmp_path, monkeypatch, capsys):
    import compare_policy_reports

    baseline_path = tmp_path / "eval_v15.json"
    candidate_path = tmp_path / "eval_v16.json"
    output_path = tmp_path / "comparison.md"
    baseline = [
        DensitySummary(10, "hard", 50, 0.46, 0.46, 0.08, 188.9, 127.4, 0.025, 0.74, -4.4),
    ]
    candidate = [
        DensitySummary(10, "hard", 50, 0.58, 0.34, 0.08, 190.0, 150.0, 0.024, 0.78, 2.0),
    ]
    write_summary_json(baseline, baseline_path, checkpoint="v15.pt", baseline_nav_steps=121.5)
    write_summary_json(candidate, candidate_path, checkpoint="v16.pt", baseline_nav_steps=121.5)

    exit_code = compare_policy_reports.main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(output_path),
            "--baseline_nav_steps",
            "121.5",
        ]
    )

    assert exit_code == 0
    assert "Overall verdict: pass" in output_path.read_text(encoding="utf-8")
    assert "comparison.md" in capsys.readouterr().out
