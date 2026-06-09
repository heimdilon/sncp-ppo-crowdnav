import os
import random
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.models import SNCPPolicy
from sncp_ppo.ppo import PPOAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_and_visualize(model_path='checkpoints/sncp_ppo.pt', output_image='trajectory_plot.png',
                      num_humans=5, scenario='circle', seed=42,
                      robot_vpref=0.26, human_vpref_override=None, max_time=50.0):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | seed={seed} | num_humans={num_humans}")

    env = CrowdSimEnv(num_humans=num_humans, scenario=scenario,
                      robot_vpref=robot_vpref, human_vpref_override=human_vpref_override,
                      max_time=max_time)
    
    # Initialize policy and agent
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded policy from {model_path}")
    else:
        print("Checkpoint not found. Running with uninitialized policy weights.")
        
    policy.train(False)  # inference mode
    agent = PPOAgent(policy=policy)

    max_search_episodes = 20
    success_found = False

    for ep in range(max_search_episodes):
        obs, info = env.reset(seed=seed + ep)
        h_states = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)
        
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
                
            action, _, _, h_states_next = agent.select_action(obs, h_states, device, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            h_states = h_states_next
            step_count += 1
            
        print(f"Eval Episode {ep+1} | Steps: {step_count} | Goal Reached: {info['success']} | Collision: {info['collision']}")
        if info['success']:
            success_found = True
            break
            
    if not success_found:
        print("Could not find a successful episode in 20 evaluation runs. Plotting the last run.")
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    
    # Plot robot path
    robot_path = np.array(robot_path)
    ax.plot(robot_path[:, 0], robot_path[:, 1], 'g-', linewidth=2.5, label='Robot Trajectory')
    ax.plot(env.robot_gx, env.robot_gy, 'r*', markersize=14, label='Robot Goal')
    ax.plot(robot_path[0, 0], robot_path[0, 1], 'go', markersize=8, label='Robot Start')
    
    # Plot human paths
    colors = ['b', 'c', 'm', 'y', 'orange']
    for i in range(env.num_humans):
        h_path = np.array(human_paths[i])
        ax.plot(h_path[:, 0], h_path[:, 1], color=colors[i % len(colors)], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot(h_path[0, 0], h_path[0, 1], color=colors[i % len(colors)], marker='o', markersize=6, alpha=0.7)
        
        # Draw final human circle and orientation
        circle = patches.Circle((h_path[-1, 0], h_path[-1, 1]), env.human_radius, color=colors[i % len(colors)], fill=False, alpha=0.8)
        ax.add_patch(circle)
        
        # Draw comfort space ellipse at the final step
        deg = np.degrees(human_headings[i][-1])
        ellipse = patches.Ellipse(
            (h_path[-1, 0], h_path[-1, 1]), 
            width=env.a_front + env.a_back, 
            height=env.b_left + env.b_right, 
            angle=deg, 
            color=colors[i % len(colors)], alpha=0.1, fill=True
        )
        ax.add_patch(ellipse)
        
    # Draw robot final circle
    r_circle = patches.Circle((env.robot_px, env.robot_py), env.robot_radius, color='g', fill=True, alpha=0.5, label='Robot End')
    ax.add_patch(r_circle)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Robot & Pedestrian Trajectories — {scenario.capitalize()} (N={num_humans})",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")
    ax.legend(loc='upper left')
    
    # Save image
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Saved trajectory plot to {output_image}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sncp_ppo.pt')
    parser.add_argument('--output', type=str, default='trajectory_plot.png')
    parser.add_argument('--num_humans', type=int, default=5)
    parser.add_argument('--scenario', type=str, default='circle')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_and_visualize(args.checkpoint, args.output, args.num_humans, args.scenario, args.seed)
