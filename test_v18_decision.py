import json
from pathlib import Path

from sncp_ppo.v18_decision import select_v18_candidate, write_v18_decision_json, write_v18_decision_report


def _density_row(n, *, success, collision, timeout, steps=170.0, i_sp=0.012):
    return {
        "num_humans": n,
        "scenario": "hard",
        "episodes": 50,
        "success_rate": success,
        "collision_rate": collision,
        "timeout_rate": timeout,
        "avg_success_steps": steps,
        "avg_episode_steps": 160.0,
        "avg_i_sp": i_sp,
        "avg_min_d_min": 0.9,
        "avg_reward": 0.0,
    }


def _write_density(path, rows):
    path.write_text(
        json.dumps(
            {
                "checkpoint": "candidate.pt",
                "baseline_nav_steps": 121.5,
                "trajectory_files": ["traj_hard_n5.png", "traj_hard_n10.png"],
                "density_sweep": rows,
            }
        ),
        encoding="utf-8",
    )


def _write_training(path, *, best=None, final=None, replay=0.20, max_std=0.24):
    best = best or {"easy": 0.60, "hard": 0.55, "circle": 0.45}
    final = final or {"easy": 0.58, "hard": 0.52, "circle": 0.40}
    path.write_text(
        json.dumps(
            {
                "observed_replay_ratio": replay,
                "best_success_by_scenario": best,
                "final_success_by_scenario": final,
                "collapse_detected": False,
                "max_std_linear": 0.16,
                "max_std_angular": max_std,
            }
        ),
        encoding="utf-8",
    )


def _baseline_rows():
    return [
        _density_row(1, success=0.44, collision=0.30, timeout=0.26, steps=187.0, i_sp=0.009),
        _density_row(3, success=0.50, collision=0.30, timeout=0.20, steps=187.0, i_sp=0.021),
        _density_row(5, success=0.66, collision=0.18, timeout=0.16, steps=187.0, i_sp=0.015),
        _density_row(8, success=0.50, collision=0.34, timeout=0.16, steps=187.0, i_sp=0.019),
        _density_row(10, success=0.46, collision=0.46, timeout=0.08, steps=187.0, i_sp=0.025),
    ]


def test_v18_decision_recommends_max_time60_for_sparse_timeout(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "density_sweep.json"
    training = tmp_path / "training_diagnostics.json"
    _write_density(baseline, _baseline_rows())
    _write_density(
        candidate,
        [
            _density_row(1, success=0.36, collision=0.04, timeout=0.60, steps=166.7, i_sp=0.009),
            _density_row(3, success=0.62, collision=0.08, timeout=0.30, steps=167.6, i_sp=0.010),
            _density_row(5, success=0.56, collision=0.18, timeout=0.26, steps=171.8, i_sp=0.006),
            _density_row(8, success=0.40, collision=0.34, timeout=0.26, steps=174.4, i_sp=0.012),
            _density_row(10, success=0.44, collision=0.44, timeout=0.12, steps=181.9, i_sp=0.021),
        ],
    )
    _write_training(training)

    decision = select_v18_candidate(candidate, baseline_json=baseline, training_json=training)

    assert decision.branch_id == "A_TIMEOUT_MAX_TIME60"
    assert decision.single_variable == "max_time 50 -> 60"
    assert any("N=1 timeout 60.0%" in reason for reason in decision.reasons)
    assert "Inspect N=5/N=10 trajectory plots before launching A100." in decision.manual_checks


def test_v18_decision_recommends_high_density_exposure_for_n10_collision(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "density_sweep.json"
    training = tmp_path / "training_diagnostics.json"
    _write_density(baseline, _baseline_rows())
    _write_density(
        candidate,
        [
            _density_row(1, success=0.62, collision=0.08, timeout=0.30, steps=168.0),
            _density_row(3, success=0.64, collision=0.12, timeout=0.24, steps=170.0),
            _density_row(5, success=0.58, collision=0.20, timeout=0.22, steps=176.0),
            _density_row(8, success=0.34, collision=0.52, timeout=0.14, steps=181.0),
            _density_row(10, success=0.30, collision=0.58, timeout=0.12, steps=186.0),
        ],
    )
    _write_training(training)

    decision = select_v18_candidate(candidate, baseline_json=baseline, training_json=training)

    assert decision.branch_id == "B_HIGH_DENSITY_EXPOSURE"
    assert decision.single_variable == "increase high-density training exposure"
    assert any("N=10 collision 58.0%" in reason for reason in decision.reasons)


def test_v18_decision_recommends_replay30_for_easy_hard_forgetting(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "density_sweep.json"
    training = tmp_path / "training_diagnostics.json"
    _write_density(baseline, _baseline_rows())
    _write_density(
        candidate,
        [
            _density_row(1, success=0.50, collision=0.12, timeout=0.38, steps=168.0),
            _density_row(3, success=0.52, collision=0.16, timeout=0.32, steps=172.0),
            _density_row(5, success=0.48, collision=0.24, timeout=0.28, steps=178.0),
            _density_row(8, success=0.42, collision=0.32, timeout=0.26, steps=181.0),
            _density_row(10, success=0.40, collision=0.38, timeout=0.22, steps=184.0),
        ],
    )
    _write_training(
        training,
        best={"easy": 0.78, "hard": 0.70, "circle": 0.48},
        final={"easy": 0.42, "hard": 0.40, "circle": 0.44},
        replay=0.18,
    )

    decision = select_v18_candidate(candidate, baseline_json=baseline, training_json=training)

    assert decision.branch_id == "C_REPLAY30"
    assert decision.single_variable == "curriculum_replay_ratio 0.20 -> 0.30"
    assert any("easy dropped by 36.0 pp" in reason for reason in decision.reasons)


def test_v18_decision_report_writes_json_and_markdown(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "density_sweep.json"
    training = tmp_path / "training_diagnostics.json"
    _write_density(baseline, _baseline_rows())
    _write_density(
        candidate,
        [
            _density_row(1, success=0.30, collision=0.20, timeout=0.50, steps=126.0, i_sp=0.040),
            _density_row(3, success=0.32, collision=0.30, timeout=0.38, steps=128.0, i_sp=0.042),
            _density_row(5, success=0.28, collision=0.46, timeout=0.26, steps=129.0, i_sp=0.050),
            _density_row(8, success=0.24, collision=0.54, timeout=0.22, steps=130.0, i_sp=0.055),
            _density_row(10, success=0.20, collision=0.60, timeout=0.20, steps=131.0, i_sp=0.060),
        ],
    )
    _write_training(training)
    decision = select_v18_candidate(candidate, baseline_json=baseline, training_json=training)
    report = tmp_path / "v18_decision.md"
    json_path = tmp_path / "v18_decision.json"

    write_v18_decision_report(decision, report)
    write_v18_decision_json(decision, json_path)

    assert decision.branch_id == "D_STOP_COMFORT_RELAXATION"
    assert "Do not run comfort_coeff 4.0" in report.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["branch_id"] == "D_STOP_COMFORT_RELAXATION"


def test_v18_decision_cli_writes_default_outputs(tmp_path, capsys):
    import select_v18_candidate

    eval_dir = tmp_path / "eval_v17"
    eval_dir.mkdir()
    baseline = tmp_path / "eval_v15" / "density_sweep.json"
    baseline.parent.mkdir()
    _write_density(baseline, _baseline_rows())
    _write_density(
        eval_dir / "density_sweep.json",
        [
            _density_row(1, success=0.36, collision=0.04, timeout=0.60, steps=166.7, i_sp=0.009),
            _density_row(3, success=0.62, collision=0.08, timeout=0.30, steps=167.6, i_sp=0.010),
            _density_row(5, success=0.56, collision=0.18, timeout=0.26, steps=171.8, i_sp=0.006),
            _density_row(8, success=0.40, collision=0.34, timeout=0.26, steps=174.4, i_sp=0.012),
            _density_row(10, success=0.44, collision=0.44, timeout=0.12, steps=181.9, i_sp=0.021),
        ],
    )
    _write_training(eval_dir / "training_diagnostics.json")

    exit_code = select_v18_candidate.main(
        [
            "--eval_dir",
            str(eval_dir),
            "--baseline_json",
            str(baseline),
        ]
    )

    assert exit_code == 0
    assert (eval_dir / "v18_decision.md").exists()
    assert (eval_dir / "v18_decision.json").exists()
    assert "A_TIMEOUT_MAX_TIME60" in capsys.readouterr().out
