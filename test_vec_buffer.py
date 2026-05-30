import torch
from sncp_ppo.vec_buffer import VectorizedRolloutBuffer


def _hidden(N, H, units=32):
    return {
        'temporal_edge': torch.zeros(N, units),
        'spatial_edge': torch.zeros(N * H, units),
        'node': torch.zeros(N, units),
    }


def test_buffer_accumulates_NT_shapes():
    N, T, H = 4, 8, 5
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    for t in range(T):
        obs = {
            'robot_node': torch.zeros(N, 7),
            'spatial_edges': torch.zeros(N, H, 4),
            'temporal_edges': torch.zeros(N, 2),
        }
        buf.store(
            obs=obs, hidden=_hidden(N, H),
            actions=torch.zeros(N, 2), log_probs=torch.zeros(N),
            rewards=torch.zeros(N), values=torch.zeros(N),
            dones=torch.zeros(N), masks=torch.ones(N),
        )
    data = buf.get_tensors(torch.device('cpu'))
    assert data['rewards'].shape == (N, T)
    assert data['actions'].shape == (N, T, 2)
    assert data['obs']['spatial_edges'].shape == (N, T, H, 4)
    assert data['obs']['robot_node'].shape == (N, T, 7)
    assert data['dones'].shape == (N, T)
