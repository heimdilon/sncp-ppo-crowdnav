"""Paper Eq 11 fidelity: `embedding = MLP(input)` -> `NCP(embedding)`.

Our encoders historically fed RAW low-dim inputs (temporal 2-dim, spatial
6-dim) straight into the LTC and projected to 256 only AFTER it; the paper
expands to the encoding dimension (256 for the time edge) BEFORE the NCP.
pre_mlp=True restores the paper's ordering. Default stays False so every
existing checkpoint (v14..v21) keeps loading byte-for-byte."""
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _dummy_obs(batch, humans):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_default_policy_still_loads_v18_checkpoint():
    state = torch.load('checkpoints/sncp_ppo_v18.pt', map_location='cpu')
    policy = build_policy_for_checkpoint(state)
    policy.load_state_dict(state)  # must not raise
    assert policy.pre_mlp is False


def test_pre_mlp_expands_encoder_inputs_to_256():
    policy = SNCPPolicy(pre_mlp=True)
    assert policy.temporal_wiring.input_dim == 256
    assert policy.spatial_wiring.input_dim == 256
    default = SNCPPolicy()
    assert default.temporal_wiring.input_dim == 2
    assert default.spatial_wiring.input_dim == 6


def test_pre_mlp_forward_shapes_and_action_scaling():
    policy = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, pre_mlp=True)
    batch, humans = 3, 5
    h = policy.init_hidden(batch_size=batch, num_humans=humans, device=torch.device('cpu'))
    mu, std, value, h_new = policy(_dummy_obs(batch, humans), h)
    assert mu.shape == (batch, 2)
    assert std.shape == (batch, 2)
    assert value.shape == (batch, 1)
    assert float(mu[:, 0].min()) >= 0.0 and float(mu[:, 0].max()) <= 1.0
    assert h_new['temporal_edge'].shape == h['temporal_edge'].shape
    assert h_new['spatial_edge'].shape == h['spatial_edge'].shape


def test_checkpoint_roundtrip_detects_pre_mlp(tmp_path):
    saved = SNCPPolicy(pre_mlp=True)
    path = tmp_path / 'premlp.pt'
    torch.save(saved.state_dict(), path)
    state = torch.load(path, map_location='cpu')
    policy = build_policy_for_checkpoint(state, robot_vpref=1.0, robot_wmax=1.8)
    policy.load_state_dict(state)  # must not raise
    assert policy.pre_mlp is True
