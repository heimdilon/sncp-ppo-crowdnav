import os as _os, sys as _sys  # repo-root path bootstrap (run standalone: python scripts/X.py)

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import random
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.action_shield import ActionShieldConfig, shield_action
from sncp_ppo.models import build_policy_for_checkpoint
from sncp_ppo.ppo import PPOAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_and_animate(
    model_path="checkpoints/sncp_ppo.pt",
    output_gif="trajectory_animation.gif",
    num_humans=5,
    scenario="circle",
    seed=42,
    robot_vpref=0.26,
    human_vpref_override=None,
    max_time=None,
    human_goal_noise=0.0,
    action_shield=False,
    shield_horizon_steps=6,
    shield_safety_margin=0.0,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed={seed} | num_humans={num_humans}")

    env = CrowdSimEnv(
        num_humans=num_humans,
        scenario=scenario,
        robot_vpref=robot_vpref,
        human_vpref_override=human_vpref_override,
        max_time=max_time,
        human_goal_noise=human_goal_noise,
    )

    # Initialize policy and agent (architecture auto-detected from the checkpoint)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        policy = build_policy_for_checkpoint(
            state_dict, robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax
        ).to(device)
        policy.load_state_dict(state_dict)
        print(f"Loaded policy from {model_path}")
    else:
        print("Checkpoint not found. Running with uninitialized policy weights.")
        return

    policy.train(False)  # inference mode
    agent = PPOAgent(policy=policy)
    shield_cfg = ActionShieldConfig(
        horizon_steps=shield_horizon_steps,
        safety_margin=shield_safety_margin,
    )

    max_search_episodes = 20
    success_found = False

    for ep in range(max_search_episodes):
        obs, info = env.reset(seed=seed + ep)
        h_states = policy.init_hidden(
            batch_size=1, num_humans=env.num_humans, device=device
        )

        robot_path = []
        human_paths = [[] for _ in range(env.num_humans)]
        human_headings = [[] for _ in range(env.num_humans)]
        robot_headings = []

        done = False
        step_count = 0
        while not done and step_count < 240:
            robot_path.append((env.robot_px, env.robot_py))
            robot_headings.append(env.robot_theta)
            for i in range(env.num_humans):
                human_paths[i].append((env.humans_px[i], env.humans_py[i]))
                human_headings[i].append(env.humans_theta[i])

            action, _, _, h_states_next = agent.select_action(
                obs, h_states, device, deterministic=True
            )
            if action_shield:
                action = shield_action(env, action, shield_cfg)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            h_states = h_states_next
            step_count += 1

        print(
            f"Eval Episode {ep+1} | Steps: {step_count} | Goal Reached: {info['success']} | Collision: {info['collision']}"
        )
        if info["success"]:
            success_found = True
            break

    if not success_found:
        print(
            "Could not find a successful episode in 20 evaluation runs. Cannot animate."
        )
        return

    print("Generating animation frames...")

    # Setup plotting figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    suffix = " + V38 Shield" if action_shield else ""
    ax.set_title(f"SNCP-PPO Crowd Navigation{suffix}", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")

    # Draw static goal
    ax.plot(env.robot_gx, env.robot_gy, "r*", markersize=14, label="Robot Goal")
    ax.plot(robot_path[0][0], robot_path[0][1], "go", markersize=8, label="Robot Start")

    # Initialize animated elements
    (robot_line,) = ax.plot([], [], "g-", linewidth=2.0, label="Robot Path")

    # Human paths, bodies and comfort space ellipses
    human_lines = []
    human_bodies = []
    comfort_ellipses = []

    colors = ["blue", "cyan", "magenta", "orange", "purple"]
    for i in range(env.num_humans):
        (h_line,) = ax.plot(
            [],
            [],
            linestyle="--",
            color=colors[i % len(colors)],
            linewidth=1.5,
            alpha=0.7,
            label=f"Pedestrian {i+1} Path" if i == 0 else "",
        )
        h_body = patches.Circle(
            (0, 0),
            env.human_radius,
            color=colors[i % len(colors)],
            fill=True,
            alpha=0.6,
            label="Pedestrians" if i == 0 else "",
        )
        c_ellipse = patches.Ellipse(
            (0, 0),
            width=env.a_front + env.a_back,
            height=env.b_left + env.b_right,
            angle=0,
            color=colors[i % len(colors)],
            alpha=0.10,
            fill=True,
            label="Comfort Space" if i == 0 else "",
        )

        human_lines.append(h_line)
        human_bodies.append(h_body)
        comfort_ellipses.append(c_ellipse)

        ax.add_patch(h_body)
        ax.add_patch(c_ellipse)

    robot_body = patches.Circle(
        (0, 0), env.robot_radius, color="green", fill=True, alpha=0.6, label="Robot"
    )
    ax.add_patch(robot_body)

    # Robot heading indicator
    (robot_dir,) = ax.plot([], [], "g-", linewidth=2.0)

    ax.legend(loc="upper left")

    # To reduce the size of the GIF, we can animate every 2nd step
    step_skip = 2
    animated_indices = list(range(0, len(robot_path), step_skip))
    if animated_indices[-1] != len(robot_path) - 1:
        animated_indices.append(len(robot_path) - 1)

    def init():
        robot_line.set_data([], [])
        for h_line in human_lines:
            h_line.set_data([], [])
        robot_body.set_center((robot_path[0][0], robot_path[0][1]))
        for i in range(env.num_humans):
            human_bodies[i].set_center((human_paths[i][0][0], human_paths[i][0][1]))
        return (
            [robot_line, robot_body, robot_dir]
            + human_lines
            + human_bodies
            + comfort_ellipses
        )

    def animate(frame_idx):
        idx = animated_indices[frame_idx]

        # Update trails
        r_coords = np.array(robot_path[: idx + 1])
        robot_line.set_data(r_coords[:, 0], r_coords[:, 1])

        # Update robot current position
        rx, ry = robot_path[idx]
        robot_body.set_center((rx, ry))

        # Update robot heading indicator
        theta = robot_headings[idx]
        arrow_len = 0.4
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        robot_dir.set_data([rx, rx + dx], [ry, ry + dy])

        # Update humans
        for i in range(env.num_humans):
            h_coords = np.array(human_paths[i][: idx + 1])
            human_lines[i].set_data(h_coords[:, 0], h_coords[:, 1])

            hx, hy = human_paths[i][idx]
            human_bodies[i].set_center((hx, hy))

            h_theta = human_headings[i][idx]
            comfort_ellipses[i].angle = np.degrees(h_theta)

            # Adjust comfort space center slightly due to front/back asymmetry
            shift_dist = (env.a_front - env.a_back) / 2.0
            ellipse_cx = hx + shift_dist * np.cos(h_theta)
            ellipse_cy = hy + shift_dist * np.sin(h_theta)
            comfort_ellipses[i].center = (ellipse_cx, ellipse_cy)

        return (
            [robot_line, robot_body, robot_dir]
            + human_lines
            + human_bodies
            + comfort_ellipses
        )

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(animated_indices),
        interval=80,
        blit=True,
    )

    # Save the animation
    print(f"Saving GIF to {output_gif} (this may take a moment)...")
    ani.save(output_gif, writer="pillow")
    print("GIF saved successfully!")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sncp_ppo.pt")
    parser.add_argument("--output", type=str, default="trajectory_animation.gif")
    parser.add_argument("--num_humans", type=int, default=5)
    parser.add_argument("--scenario", type=str, default="circle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robot_vpref", type=float, default=0.26)
    parser.add_argument("--human_vpref_override", type=float, default=None)
    parser.add_argument("--human_goal_noise", type=float, default=0.0)
    parser.add_argument("--max_time", type=float, default=None)
    parser.add_argument("--action_shield", action="store_true")
    parser.add_argument("--shield_horizon_steps", type=int, default=6)
    parser.add_argument("--shield_safety_margin", type=float, default=0.0)
    args = parser.parse_args()
    run_and_animate(
        args.checkpoint,
        args.output,
        args.num_humans,
        args.scenario,
        args.seed,
        robot_vpref=args.robot_vpref,
        human_vpref_override=args.human_vpref_override,
        max_time=args.max_time,
        human_goal_noise=args.human_goal_noise,
        action_shield=args.action_shield,
        shield_horizon_steps=args.shield_horizon_steps,
        shield_safety_margin=args.shield_safety_margin,
    )
