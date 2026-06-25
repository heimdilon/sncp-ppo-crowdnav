from pathlib import Path

from sncp_ppo.v37_probe import analyze_probe_runs, build_training_command


def _fake_runs(*, timeout=False, gate=0.02):
    runs = []
    for arm in ("c0", "c1"):
        for train_seed in (40, 41, 42):
            episodes = []
            for density in (5, 10, 15, 20):
                successes = 9 if density in (5, 10) else (6 if arm == "c0" else 7)
                for case_index in range(10):
                    is_success = case_index < successes
                    is_timeout = bool(timeout and arm == "c1" and density == 20 and case_index == 9)
                    episodes.append(
                        {
                            "case_id": f"n{density}_case{case_index}",
                            "density": density,
                            "episode_seed": 10_000 + density * 100 + case_index,
                            "success": is_success,
                            "collision": not is_success and not is_timeout,
                            "timeout": is_timeout,
                        }
                    )
            runs.append(
                {
                    "arm": arm,
                    "train_seed": train_seed,
                    "checkpoint": f"{arm}_s{train_seed}.pt",
                    "hh_gate": gate if arm == "c1" else None,
                    "diagnostics": {"finite": True, "collapse_delta": -0.02},
                    "episodes": episodes,
                }
            )
    return runs


def test_probe_commands_are_controlled_v34_continuations():
    base = Path("sncp_ppo_v34.pt")
    c0 = build_training_command(
        "c0", 40, base_checkpoint=base,
        save_path=Path("checkpoints/c0.pt"), python_executable="python",
    )
    c1 = build_training_command(
        "c1", 40, base_checkpoint=base,
        save_path=Path("checkpoints/c1.pt"), python_executable="python",
    )
    joined0, joined1 = " ".join(c0), " ".join(c1)
    assert "--init_checkpoint sncp_ppo_v34.pt" in joined0
    assert "--upgrade_checkpoint sncp_ppo_v34.pt" in joined1
    assert "--hh_intent_graph" in c1
    assert "--hh_attn_heads 4" in joined1
    assert "--cv_horizons 1 2 3 4" in joined1
    assert "--cv_dt 0.25" in joined1
    for joined in (joined0, joined1):
        assert "--total_steps 300000" in joined
        assert "--num_humans_range 10 25" in joined
        assert "--lr 5e-05" in joined
        assert "--lr_end_factor 0.5" in joined
        assert "--ent_coef 0.001" in joined
        assert "--attn_count_scaling" not in joined
        assert "--sense_range" not in joined
        assert "--node_units" not in joined


def test_probe_analysis_emits_go_for_paired_high_n_gain():
    result = analyze_probe_runs(_fake_runs())
    assert result["complete"] is True
    assert result["high_n_success_delta"] == 0.10
    assert result["high_n_collision_delta"] == -0.10
    assert result["direction_seed_count"] == 3
    assert result["low_n_regression"] is False
    assert result["timeout_zero"] is True
    assert result["gate_active"] is True
    assert result["diagnostics_healthy"] is True
    assert result["verdict"] == "GO"
    assert result["paired"]["15"]["success"]["discordant"] > 0


def test_probe_analysis_fails_for_timeout_and_inactive_gate():
    result = analyze_probe_runs(_fake_runs(timeout=True, gate=0.005))
    assert result["timeout_zero"] is False
    assert result["gate_active"] is False
    assert result["verdict"] == "NO-GO"
