import json
import math

import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.custom_scenario import create_custom_env, load_custom_scenario


def test_custom_scenario_applies_robot_humans_and_individual_speeds(tmp_path):
    scenario_path = tmp_path / "crossing.json"
    scenario_path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "two-person crossing",
                "time_step": 0.25,
                "max_time": 12.5,
                "human_motion_model": "linear",
                "human_dodge_robot": False,
                "robot": {
                    "position": {"x": -3.0, "y": -1.0},
                    "goal": {"x": 3.5, "y": 1.0},
                    "theta_deg": 15.0,
                },
                "humans": [
                    {
                        "id": "h1",
                        "position": {"x": 0.0, "y": -1.0},
                        "theta_deg": 90.0,
                        "speed": 0.20,
                    },
                    {
                        "id": "h2",
                        "position": {"x": 1.0, "y": 1.0},
                        "theta": math.pi,
                        "speed": 0.13,
                        "goal": {"x": -3.0, "y": 1.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    scenario = load_custom_scenario(scenario_path)
    env = create_custom_env(scenario, seed=7)
    obs = env._get_obs()

    assert env.num_humans == 2
    assert env.time_step == 0.25
    assert env.max_time == 12.5
    assert env.human_motion_model == "linear"
    assert env.human_dodge_robot is False

    np.testing.assert_allclose([env.robot_px, env.robot_py], [-3.0, -1.0])
    np.testing.assert_allclose([env.robot_gx, env.robot_gy], [3.5, 1.0])
    assert env.robot_theta == np.float64(math.radians(15.0))

    np.testing.assert_allclose(env.humans_px, [0.0, 1.0])
    np.testing.assert_allclose(env.humans_py, [-1.0, 1.0])
    np.testing.assert_allclose(env.humans_theta, [math.pi / 2.0, math.pi])
    np.testing.assert_allclose(env.humans_vpref, [0.20, 0.13])
    np.testing.assert_allclose(env.humans_vx, [0.0, -0.13], atol=1e-7)
    np.testing.assert_allclose(env.humans_vy, [0.20, 0.0], atol=1e-7)

    # Human 1 omitted an explicit goal; the loader should extend its heading
    # into a goal so the policy still observes pedestrian intent.
    assert env.humans_gx[0] == np.float64(0.0)
    assert env.humans_gy[0] > env.humans_py[0]
    np.testing.assert_allclose([env.humans_gx[1], env.humans_gy[1]], [-3.0, 1.0])
    assert obs["spatial_edges"].shape == (2, 6)


def test_linear_human_motion_preserves_custom_direction_and_speed():
    env = CrowdSimEnv(
        num_humans=1,
        scenario="circle",
        randomize_layout=False,
        human_motion_model="linear",
    )
    env.reset(seed=1)
    env.robot_px = -4.0
    env.robot_py = -4.0
    env.robot_gx = 4.0
    env.robot_gy = 4.0
    env.robot_theta = 0.0
    env.humans_px = np.array([0.0], dtype=float)
    env.humans_py = np.array([0.5], dtype=float)
    env.humans_theta = np.array([0.0], dtype=float)
    env.humans_vpref = np.array([0.20], dtype=float)
    env.humans_vx = np.array([0.20], dtype=float)
    env.humans_vy = np.array([0.0], dtype=float)
    env.humans_gx = np.array([5.0], dtype=float)
    env.humans_gy = np.array([0.5], dtype=float)

    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert not terminated
    assert not truncated
    np.testing.assert_allclose(env.humans_px, [0.05], atol=1e-8)
    np.testing.assert_allclose(env.humans_py, [0.5], atol=1e-8)
    np.testing.assert_allclose(env.humans_vx, [0.20], atol=1e-8)
    np.testing.assert_allclose(env.humans_vy, [0.0], atol=1e-8)
    assert obs["spatial_edges"].shape == (1, 6)


def test_custom_episode_runner_records_metrics_and_paths():
    from evaluate_custom_scenario import run_episode_with_action_provider

    scenario = load_custom_scenario(
        {
            "version": 1,
            "name": "stationary smoke",
            "time_step": 0.25,
            "max_time": 5.0,
            "human_motion_model": "linear",
            "robot": {
                "position": {"x": -2.0, "y": 0.0},
                "goal": {"x": 2.0, "y": 0.0},
                "theta": 0.0,
            },
            "humans": [
                {
                    "id": "h1",
                    "position": {"x": 0.0, "y": 2.5},
                    "theta": 0.0,
                    "speed": 0.0,
                }
            ],
        }
    )
    env = create_custom_env(scenario, seed=3)

    result = run_episode_with_action_provider(
        env,
        action_provider=lambda obs, step_index: np.array([0.0, 0.0], dtype=np.float32),
        max_steps=3,
    )

    assert result["steps"] == 3
    assert result["done_reason"] == "max_steps"
    assert result["success"] is False
    assert result["collision"] is False
    assert result["timeout"] is False
    assert len(result["robot_path"]) == 4
    assert len(result["human_paths"]) == 1
    assert len(result["human_paths"][0]) == 4
    assert "avg_I_sp" in result


def test_custom_episode_runner_records_action_trace_and_braking_metrics():
    from evaluate_custom_scenario import run_episode_with_action_provider

    scenario = load_custom_scenario(
        {
            "version": 1,
            "name": "action trace smoke",
            "time_step": 0.25,
            "max_time": 2.0,
            "human_motion_model": "linear",
            "robot": {
                "position": {"x": -1.0, "y": 0.0},
                "goal": {"x": 1.0, "y": 0.0},
                "theta": 0.0,
            },
            "humans": [],
        }
    )
    env = create_custom_env(scenario, seed=5)

    result = run_episode_with_action_provider(
        env,
        action_provider=lambda obs, step_index: np.array([0.26, (-1) ** step_index], dtype=np.float32),
        max_steps=3,
    )

    assert result["raw_actions"] == [[0.26, 1.0], [0.26, -1.0], [0.26, 1.0]]
    assert result["env_actions"] == [[0.26, 1.0], [0.26, -1.0], [0.26, 1.0]]
    assert result["linear_speeds"] == [0.26, 0.26, 0.26]
    assert result["angular_speeds"] == [1.0, -1.0, 1.0]
    assert result["min_linear_speed"] == 0.26
    assert result["final_1s_avg_linear_speed"] == 0.26


def test_custom_gif_renderer_writes_animated_gif(tmp_path):
    from evaluate_custom_scenario import render_custom_gif

    scenario = load_custom_scenario(
        {
            "version": 1,
            "name": "gif smoke",
            "time_step": 0.25,
            "max_time": 1.0,
            "human_motion_model": "linear",
            "robot": {
                "position": {"x": -1.0, "y": 0.0},
                "goal": {"x": 1.0, "y": 0.0},
                "theta": 0.0,
            },
            "humans": [
                {
                    "id": "h1",
                    "position": {"x": 0.0, "y": 1.0},
                    "theta": 0.0,
                    "speed": 0.0,
                }
            ],
        }
    )
    result = {
        "steps": 3,
        "done_reason": "max_steps",
        "avg_I_sp": 0.0,
        "robot_path": [[-1.0, 0.0], [-0.8, 0.0], [-0.6, 0.0], [-0.4, 0.0]],
        "robot_headings": [0.0, 0.0, 0.0, 0.0],
        "human_paths": [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]],
        "human_headings": [[0.0, 0.0, 0.0, 0.0]],
    }
    gif_path = tmp_path / "custom.gif"

    render_custom_gif(result, scenario, gif_path, fps=8, step_skip=1)

    assert gif_path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    assert gif_path.stat().st_size > 1000
