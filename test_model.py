import torch
from sncp_ppo.models import SNCPPolicy

def test_model():
    print("Initializing SNCP Policy model...")
    policy = SNCPPolicy(robot_vpref=0.26, robot_wmax=1.8)
    
    batch_size = 4
    num_humans = 5
    device = torch.device('cpu')
    
    # Create fake observations (updated for local-frame, dim=7)
    obs = {
        'robot_node': torch.randn(batch_size, 7),
        'spatial_edges': torch.randn(batch_size, num_humans, 2),
        'temporal_edges': torch.randn(batch_size, 2)
    }
    
    # Initialize hidden states
    h_states = policy.init_hidden(batch_size, num_humans, device)
    print("Hidden states initialized successfully!")
    print(f"Temporal edge hidden state shape: {h_states['temporal_edge'].shape}")
    print(f"Spatial edge hidden state shape: {h_states['spatial_edge'].shape}")
    print(f"Node hidden state shape: {h_states['node'].shape}")
    
    # Run forward pass
    mu, std, value, h_states_new = policy(obs, h_states)
    print("\nForward pass successful!")
    print(f"Mu shape: {mu.shape} (Expected: [4, 2])")
    print(f"Std shape: {std.shape} (Expected: [4, 2])")
    print(f"Value shape: {value.shape} (Expected: [4, 1])")
    
    # Assertions
    assert mu.shape == (batch_size, 2)
    assert std.shape == (batch_size, 2)
    assert value.shape == (batch_size, 1)
    assert h_states_new['temporal_edge'].shape == h_states['temporal_edge'].shape
    assert h_states_new['spatial_edge'].shape == h_states['spatial_edge'].shape
    assert h_states_new['node'].shape == h_states['node'].shape
    
    # Verify values are in limits
    # Linear speed is in [0, 0.26]
    assert torch.all(mu[:, 0] >= 0.0) and torch.all(mu[:, 0] <= 0.26)
    # Angular speed is in [-1.8, 1.8]
    assert torch.all(mu[:, 1] >= -1.8) and torch.all(mu[:, 1] <= 1.8)
    
    print("\nModel test passed successfully!")

if __name__ == '__main__':
    test_model()
