import math
import json

from sncp_ppo.eval_report import (
    DensitySummary,
    EpisodeResult,
    plot_density_curves,
    run_report,
    summarize_density,
    write_markdown_report,
    write_summary_csv,
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

    def fake_evaluator(*, checkpoint_path, num_humans, scenario, n_episodes, seed):
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

    def fake_trajectory_renderer(*, checkpoint_path, output_path, num_humans, scenario, seed):
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
