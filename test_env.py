import sys
import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv

def test_environment():
    print("Initializing environment...")
    env = CrowdSimEnv(num_humans=5, scenario='circle')
    
    obs, info = env.reset()
    print("Reset successful!")
    print(f"Robot Node Obs Shape: {obs['robot_node'].shape}")
    print(f"Spatial Edges Obs Shape: {obs['spatial_edges'].shape}")
    print(f"Temporal Edges Obs Shape: {obs['temporal_edges'].shape}")
    
    # Check shape values (updated for local-frame observations)
    assert obs['robot_node'].shape == (7,), f"Expected (7,), got {obs['robot_node'].shape}"
    assert obs['spatial_edges'].shape == (5, 2)
    assert obs['temporal_edges'].shape == (2,)
    
    print("\nRunning test steps...")
    done = False
    step_count = 0
    while not done and step_count < 10:
        # Sample random action: linear speed, angular speed
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        print(f"Step {step_count}: Reward = {reward:.4f}, Collision = {info['collision']}, Goal Reached = {info['success']}, Social Pressure (I_sp) = {info['I_sp']:.4f}")
        
    print("\nEnvironment test passed successfully!")

if __name__ == '__main__':
    test_environment()
