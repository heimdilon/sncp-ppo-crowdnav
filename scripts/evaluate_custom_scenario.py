"""Run a trained SNCP-PPO checkpoint on a hand-authored custom scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import torch

from sncp_ppo.custom_scenario import (
    CustomScenario,
    create_custom_env,
    load_custom_scenario,
)
from sncp_ppo.models import build_policy_for_checkpoint
from sncp_ppo.ppo import PPOAgent


ActionProvider = Callable[[dict[str, np.ndarray], int], np.ndarray]


def _rounded_float(value: float) -> float:
    return float(round(float(value), 6))


def run_episode_with_action_provider(
    env,
    action_provider: ActionProvider,
    max_steps: int | None = None,
) -> dict:
    limit = max_steps or int(np.ceil(env.max_time / env.time_step)) + 1
    obs = env._get_obs()
    robot_path = [[float(env.robot_px), float(env.robot_py)]]
    robot_headings = [float(env.robot_theta)]
    human_paths = [
        [[float(env.humans_px[i]), float(env.humans_py[i])]]
        for i in range(env.num_humans)
    ]
    human_headings = [[float(env.humans_theta[i])] for i in range(env.num_humans)]
    rewards: list[float] = []
    min_distances: list[float] = []
    social_pressures: list[float] = []
    raw_actions: list[list[float]] = []
    env_actions: list[list[float]] = []
    linear_speeds: list[float] = []
    angular_speeds: list[float] = []
    info = {
        "success": False,
        "collision": False,
        "timeout": False,
        "d_min": float("inf"),
        "I_sp": 0.0,
    }
    done_reason = "max_steps"

    for step_index in range(limit):
        raw_action = np.asarray(action_provider(obs, step_index), dtype=np.float32)
        env_action = PPOAgent.clip_action_for_env(raw_action, env.robot_vpref, env.robot_wmax)
        obs, reward, terminated, truncated, info = env.step(env_action)

        robot_path.append([float(env.robot_px), float(env.robot_py)])
        robot_headings.append(float(env.robot_theta))
        for i in range(env.num_humans):
            human_paths[i].append([float(env.humans_px[i]), float(env.humans_py[i])])
            human_headings[i].append(float(env.humans_theta[i]))
        rewards.append(float(reward))
        min_distances.append(float(info["d_min"]))
        social_pressures.append(float(info["I_sp"]))
        raw_actions.append([_rounded_float(raw_action[0]), _rounded_float(raw_action[1])])
        env_actions.append([_rounded_float(env_action[0]), _rounded_float(env_action[1])])
        linear_speeds.append(_rounded_float(env_action[0]))
        angular_speeds.append(_rounded_float(env_action[1]))

        if terminated or truncated:
            if bool(info["success"]):
                done_reason = "success"
            elif bool(info["collision"]):
                done_reason = "collision"
            elif bool(info["timeout"]):
                done_reason = "timeout"
            else:
                done_reason = "terminated"
            break

    finite_distances = [d for d in min_distances if np.isfinite(d)]
    one_second_window = max(1, int(np.ceil(1.0 / env.time_step)))
    final_linear_window = linear_speeds[-one_second_window:]
    return {
        "steps": len(rewards),
        "time_sec": float(len(rewards) * env.time_step),
        "done_reason": done_reason,
        "success": bool(info["success"]),
        "collision": bool(info["collision"]),
        "timeout": bool(info["timeout"]),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "avg_I_sp": float(np.mean(social_pressures)) if social_pressures else 0.0,
        "min_d_min": float(np.min(finite_distances)) if finite_distances else None,
        "raw_actions": raw_actions,
        "env_actions": env_actions,
        "linear_speeds": linear_speeds,
        "angular_speeds": angular_speeds,
        "min_linear_speed": float(np.min(linear_speeds)) if linear_speeds else None,
        "final_1s_avg_linear_speed": (
            float(np.mean(final_linear_window)) if final_linear_window else None
        ),
        "robot_path": robot_path,
        "robot_headings": robot_headings,
        "human_paths": human_paths,
        "human_headings": human_headings,
    }


def make_policy_action_provider(
    checkpoint_path: str | Path,
    env,
    device_name: str = "auto",
    deterministic: bool = True,
) -> ActionProvider:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    state_dict = torch.load(checkpoint_path, map_location=device)
    policy = build_policy_for_checkpoint(
        state_dict, robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.train(False)
    agent = PPOAgent(policy=policy)
    h_states = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)

    def provide(obs: dict[str, np.ndarray], step_index: int) -> np.ndarray:
        nonlocal h_states
        action, _, _, h_states_next = agent.select_action(
            obs,
            h_states,
            device,
            deterministic=deterministic,
        )
        h_states = h_states_next
        return np.asarray(action, dtype=np.float32)

    return provide


def render_custom_trajectory(
    result: dict,
    scenario: CustomScenario,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)

    robot_path = np.asarray(result["robot_path"], dtype=float)
    ax.plot(robot_path[:, 0], robot_path[:, 1], color="#1f9d55", linewidth=2.5, label="Robot")
    ax.scatter(robot_path[0, 0], robot_path[0, 1], color="#1f9d55", s=55, zorder=4)
    ax.scatter(scenario.robot.gx, scenario.robot.gy, color="#d22f27", marker="*", s=170, label="Goal")
    ax.add_patch(
        patches.Circle(
            (robot_path[-1, 0], robot_path[-1, 1]),
            radius=0.3,
            color="#1f9d55",
            alpha=0.25,
        )
    )

    palette = ["#2457c5", "#b73a8c", "#d18410", "#2b8c7f", "#6f4dbd", "#687076"]
    for i, human in enumerate(scenario.humans):
        path = np.asarray(result["human_paths"][i], dtype=float)
        color = palette[i % len(palette)]
        ax.plot(path[:, 0], path[:, 1], linestyle="--", color=color, alpha=0.8, linewidth=1.4)
        ax.scatter(path[0, 0], path[0, 1], color=color, s=38)
        ax.add_patch(patches.Circle((path[-1, 0], path[-1, 1]), radius=0.3, fill=False, color=color))
        ax.arrow(
            path[0, 0],
            path[0, 1],
            0.45 * np.cos(human.theta),
            0.45 * np.sin(human.theta),
            head_width=0.08,
            head_length=0.12,
            color=color,
            length_includes_head=True,
        )

    ax.set_title(
        f"{scenario.name} | {result['done_reason']} | steps={result['steps']} | I_sp={result['avg_I_sp']:.3f}"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def render_custom_gif(
    result: dict,
    scenario: CustomScenario,
    output_path: str | Path,
    fps: int = 12,
    step_skip: int = 2,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(fps))
    step_skip = max(1, int(step_skip))

    robot_path = np.asarray(result["robot_path"], dtype=float)
    robot_headings = np.asarray(result["robot_headings"], dtype=float)
    human_paths = [np.asarray(path, dtype=float) for path in result["human_paths"]]
    human_headings = [np.asarray(headings, dtype=float) for headings in result["human_headings"]]
    frame_count = len(robot_path)
    animated_indices = list(range(0, frame_count, step_skip))
    if animated_indices[-1] != frame_count - 1:
        animated_indices.append(frame_count - 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(f"{scenario.name} | {result['done_reason']} | I_sp={result['avg_I_sp']:.3f}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.scatter(scenario.robot.gx, scenario.robot.gy, color="#d22f27", marker="*", s=170, label="Goal")
    ax.scatter(robot_path[0, 0], robot_path[0, 1], color="#1f9d55", s=55, label="Robot start")

    robot_line, = ax.plot([], [], color="#1f9d55", linewidth=2.5, label="Robot")
    robot_body = patches.Circle((robot_path[0, 0], robot_path[0, 1]), radius=0.3, color="#1f9d55", alpha=0.45)
    ax.add_patch(robot_body)
    robot_dir, = ax.plot([], [], color="#1f9d55", linewidth=2.0)

    palette = ["#2457c5", "#b73a8c", "#d18410", "#2b8c7f", "#6f4dbd", "#687076"]
    human_lines = []
    human_bodies = []
    comfort_ellipses = []
    human_dirs = []
    for i, path in enumerate(human_paths):
        color = palette[i % len(palette)]
        line, = ax.plot([], [], linestyle="--", color=color, alpha=0.8, linewidth=1.3)
        body = patches.Circle((path[0, 0], path[0, 1]), radius=0.3, color=color, alpha=0.55)
        ellipse = patches.Ellipse(
            (path[0, 0], path[0, 1]),
            width=2.0,
            height=1.5,
            angle=0.0,
            color=color,
            alpha=0.10,
            fill=True,
        )
        direction, = ax.plot([], [], color=color, linewidth=1.5)
        human_lines.append(line)
        human_bodies.append(body)
        comfort_ellipses.append(ellipse)
        human_dirs.append(direction)
        ax.add_patch(ellipse)
        ax.add_patch(body)

    ax.legend(loc="upper left")

    def animate(frame_index: int):
        idx = animated_indices[frame_index]
        robot_line.set_data(robot_path[: idx + 1, 0], robot_path[: idx + 1, 1])
        rx, ry = robot_path[idx]
        robot_body.set_center((rx, ry))
        theta = robot_headings[idx]
        robot_dir.set_data([rx, rx + 0.45 * np.cos(theta)], [ry, ry + 0.45 * np.sin(theta)])

        for i, path in enumerate(human_paths):
            human_lines[i].set_data(path[: idx + 1, 0], path[: idx + 1, 1])
            hx, hy = path[idx]
            human_bodies[i].set_center((hx, hy))
            h_theta = human_headings[i][idx]
            comfort_ellipses[i].center = (hx + 0.5 * np.cos(h_theta), hy + 0.5 * np.sin(h_theta))
            comfort_ellipses[i].angle = np.degrees(h_theta)
            human_dirs[i].set_data(
                [hx, hx + 0.45 * np.cos(h_theta)],
                [hy, hy + 0.45 * np.sin(h_theta)],
            )

        return [robot_line, robot_body, robot_dir] + human_lines + human_bodies + comfort_ellipses + human_dirs

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(animated_indices),
        interval=int(1000 / fps),
        blit=True,
    )
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)


def run_custom_checkpoint(
    scenario_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    gif_path: str | Path | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    device: str = "auto",
    deterministic: bool = True,
    gif_fps: int = 12,
    gif_step_skip: int = 2,
) -> dict:
    scenario = load_custom_scenario(scenario_path)
    env = create_custom_env(scenario, seed=seed)
    action_provider = make_policy_action_provider(
        checkpoint_path,
        env,
        device_name=device,
        deterministic=deterministic,
    )
    result = run_episode_with_action_provider(env, action_provider, max_steps=max_steps)
    render_custom_trajectory(result, scenario, output_path)
    if gif_path is not None:
        render_custom_gif(result, scenario, gif_path, fps=gif_fps, step_skip=gif_step_skip)
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("custom_eval/trajectory.png"))
    parser.add_argument("--summary", type=Path, default=Path("custom_eval/summary.json"))
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using policy mean")
    parser.add_argument("--gif_fps", type=int, default=12)
    parser.add_argument("--gif_step_skip", type=int, default=2)
    args = parser.parse_args()

    result = run_custom_checkpoint(
        scenario_path=args.scenario,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        summary_path=args.summary,
        gif_path=args.gif,
        seed=args.seed,
        max_steps=args.max_steps,
        device=args.device,
        deterministic=not args.stochastic,
        gif_fps=args.gif_fps,
        gif_step_skip=args.gif_step_skip,
    )
    print(
        "Custom eval | "
        f"reason={result['done_reason']} steps={result['steps']} "
        f"success={result['success']} collision={result['collision']} "
        f"avg_I_sp={result['avg_I_sp']:.4f}"
    )
    print(f"Trajectory: {args.output}")
    print(f"Summary: {args.summary}")
    if args.gif is not None:
        print(f"GIF: {args.gif}")


if __name__ == "__main__":
    main()
