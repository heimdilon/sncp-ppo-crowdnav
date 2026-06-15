import os
import random
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.models import build_policy_for_checkpoint
from sncp_ppo.ppo import PPOAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_and_animate_scenario(scenario_name, model_path='checkpoints/sncp_ppo.pt',
                             num_humans=5, seed=42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n==================================================")
    print(f"Processing Scenario: {scenario_name.upper()}")
    print("==================================================")
    
    env = CrowdSimEnv(num_humans=num_humans, scenario=scenario_name)
    print(f"Scenario: {scenario_name} | Pedestrian count: {env.num_humans} | Config: {env.scenario}")
    
    # Initialize policy and agent (architecture auto-detected from the checkpoint)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        policy = build_policy_for_checkpoint(
            state_dict, robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax
        ).to(device)
        policy.load_state_dict(state_dict)
        print(f"Loaded policy from {model_path}")
    else:
        print(f"Checkpoint not found at {model_path}. Exiting.")
        return
        
    policy.train(False)  # inference mode
    agent = PPOAgent(policy=policy)

    max_search_episodes = 50
    success_found = False

    robot_path = []
    human_paths = []
    human_headings = []
    robot_headings = []
    step_count = 0

    for ep in range(max_search_episodes):
        obs, info = env.reset(seed=seed + ep)
        h_states = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)
        
        ep_robot_path = []
        ep_human_paths = [[] for _ in range(env.num_humans)]
        ep_human_headings = [[] for _ in range(env.num_humans)]
        ep_robot_headings = []
        
        done = False
        ep_step_count = 0
        while not done and ep_step_count < 240:
            ep_robot_path.append((env.robot_px, env.robot_py))
            ep_robot_headings.append(env.robot_theta)
            for i in range(env.num_humans):
                ep_human_paths[i].append((env.humans_px[i], env.humans_py[i]))
                ep_human_headings[i].append(env.humans_theta[i])
                
            action, _, _, h_states_next = agent.select_action(obs, h_states, device, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            h_states = h_states_next
            ep_step_count += 1
            
        if info['success']:
            success_found = True
            robot_path = ep_robot_path
            human_paths = ep_human_paths
            human_headings = ep_human_headings
            robot_headings = ep_robot_headings
            step_count = ep_step_count
            print(f"-> SUCCESS found on trial {ep+1}! Steps: {step_count}")
            break
        else:
            print(f"   Trial {ep+1}: Reached: {info['success']} | Collision: {info['collision']}")
            
    if not success_found:
        print(f"Could not find a successful episode in {max_search_episodes} evaluation runs for {scenario_name}.")
        return
        
    print("Generating animation frames...")
    
    # Setup plotting figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"SNCP-PPO Navigation - {scenario_name.upper()} Map", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")
    
    # Draw static goal
    ax.plot(env.robot_gx, env.robot_gy, 'r*', markersize=14, label='Robot Goal')
    ax.plot(robot_path[0][0], robot_path[0][1], 'go', markersize=8, label='Robot Start')
    
    # Initialize animated elements
    robot_line, = ax.plot([], [], 'g-', linewidth=2.0, label='Robot Path')
    
    human_lines = []
    human_bodies = []
    comfort_ellipses = []
    
    colors = ['blue', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']
    for i in range(env.num_humans):
        h_line, = ax.plot([], [], linestyle='--', color=colors[i % len(colors)], linewidth=1.5, alpha=0.7, label='Pedestrian Path' if i == 0 else "")
        h_body = patches.Circle((0, 0), env.human_radius, color=colors[i % len(colors)], fill=True, alpha=0.6, label='Pedestrians' if i == 0 else "")
        c_ellipse = patches.Ellipse(
            (0, 0), 
            width=env.a_front + env.a_back, 
            height=env.b_left + env.b_right, 
            angle=0, 
            color=colors[i % len(colors)], alpha=0.08, fill=True, label='Comfort Space' if i == 0 else ""
        )
        
        human_lines.append(h_line)
        human_bodies.append(h_body)
        comfort_ellipses.append(c_ellipse)
        
        ax.add_patch(h_body)
        ax.add_patch(c_ellipse)
        
    robot_body = patches.Circle((0, 0), env.robot_radius, color='green', fill=True, alpha=0.6, label='Robot')
    ax.add_patch(robot_body)
    
    # Robot heading indicator
    robot_dir, = ax.plot([], [], 'g-', linewidth=2.0)
    
    ax.legend(loc='upper left')
    
    # To reduce the size of the GIF, we animate every 2nd step
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
        return [robot_line, robot_body, robot_dir] + human_lines + human_bodies + comfort_ellipses
        
    def animate(frame_idx):
        idx = animated_indices[frame_idx]
        
        # Update trails
        r_coords = np.array(robot_path[:idx+1])
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
            h_coords = np.array(human_paths[i][:idx+1])
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
            
        return [robot_line, robot_body, robot_dir] + human_lines + human_bodies + comfort_ellipses
        
    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, 
        frames=len(animated_indices), interval=80, blit=True
    )
    
    # Save the animation
    output_gif = f"{scenario_name}_trajectory.gif"
    ani.save(output_gif, writer='pillow')
    print(f"Saved animation to {output_gif}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sncp_ppo.pt')
    parser.add_argument('--num_humans', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=['easy', 'medium', 'hard', 'extreme'])
    args = parser.parse_args()
    for scen in args.scenarios:
        run_and_animate_scenario(scen, args.checkpoint, args.num_humans, args.seed)
