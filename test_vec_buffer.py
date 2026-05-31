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
    assert data['h_temporal'].shape == (N, T, 32)
    assert data['h_node'].shape == (N, T, 32)
    assert data['h_spatial'].shape == (N, T, H * 32)


def test_finish_sets_bootstrap_and_horizon_done():
    N, T, H = 2, 3, 5
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    # env 0: terminates (collision/goal) at t=1; env 1: never terminates
    done_seq = [torch.tensor([0., 0.]), torch.tensor([1., 0.]), torch.tensor([0., 0.])]
    mask_seq = [torch.tensor([1., 1.]), torch.tensor([0., 1.]), torch.tensor([1., 1.])]
    for t in range(T):
        obs = {'robot_node': torch.zeros(N, 7),
               'spatial_edges': torch.zeros(N, H, 4),
               'temporal_edges': torch.zeros(N, 2)}
        buf.store(obs=obs,
                  hidden={'temporal_edge': torch.zeros(N, 32),
                          'spatial_edge': torch.zeros(N * H, 32),
                          'node': torch.zeros(N, 32)},
                  actions=torch.zeros(N, 2), log_probs=torch.zeros(N),
                  rewards=torch.zeros(N), values=torch.zeros(N),
                  dones=done_seq[t], masks=mask_seq[t])
    # last_values = V(s_next) at horizon end for each env
    buf.finish(last_values=torch.tensor([9.0, 7.0]),
               last_dones=torch.tensor([0., 0.]))
    data = buf.get_tensors(torch.device('cpu'))
    # Horizon end (t=T-1) is forced done for BOTH envs
    assert data['dones'][0, T - 1] == 1.0
    assert data['dones'][1, T - 1] == 1.0
    # horizon-end bootstrap = each env's last_value (9.0, 7.0)
    assert torch.allclose(data['bootstrap_values'][1, T - 1], torch.tensor(7.0))
    assert torch.allclose(data['bootstrap_values'][0, T - 1], torch.tensor(9.0))
    # env0 terminated at t=1 -> bootstrap there is 0
    assert torch.allclose(data['bootstrap_values'][0, 1], torch.tensor(0.0))


def test_finish_bootstrap_on_last_values_device():
    N, T, H = 2, 3, 5
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    for t in range(T):
        obs = {'robot_node': torch.zeros(N, 7),
               'spatial_edges': torch.zeros(N, H, 4),
               'temporal_edges': torch.zeros(N, 2)}
        buf.store(obs=obs,
                  hidden={'temporal_edge': torch.zeros(N, 32),
                          'spatial_edge': torch.zeros(N * H, 32),
                          'node': torch.zeros(N, 32)},
                  actions=torch.zeros(N, 2), log_probs=torch.zeros(N),
                  rewards=torch.zeros(N), values=torch.zeros(N),
                  dones=torch.zeros(N), masks=torch.ones(N))
    last_v = torch.tensor([5.0, 6.0])
    buf.finish(last_values=last_v, last_dones=torch.zeros(N))
    assert buf.bootstrap_values.device == last_v.device
    assert torch.allclose(buf.bootstrap_values[:, T - 1], last_v)
