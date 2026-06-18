"""Fidelity guard: the three SNCP encoders must use TRUE sparse Neural Circuit
Policy (NCP) wiring, as the source paper claims (Ao et al. 2026, "Structured
Neural Circuit Policies"), not a dense FullyConnected LTC.

The paper specifies the LTC neuron/synapse dynamics (Eq 8-10) and shows sparse
synaptic connections (Fig 4), but does NOT give the wiring parameters (neuron
counts, sparsity). Our job is to choose those parameters for our problem — not
to substitute a different architecture. These tests lock in the sparse NCP.
"""
import torch
from ncps.wirings import AutoNCP

from sncp_ppo.models import SNCPPolicy


def test_three_encoders_use_sparse_ncp_wiring():
    """temporal/spatial/node encoders must be sparse AutoNCP, not dense."""
    policy = SNCPPolicy()
    for name, w in [
        ('temporal', policy.temporal_wiring),
        ('spatial', policy.spatial_wiring),
        ('node', policy.node_wiring),
    ]:
        assert isinstance(w, AutoNCP), f"{name} wiring is not AutoNCP (true NCP): {type(w)}"
        # Sparse: internal connectivity must be well below fully-connected (100%).
        adj = w.adjacency_matrix
        density = float((adj != 0).sum()) / adj.size
        assert density < 0.5, f"{name} wiring not sparse (density {density:.2f}) — looks dense"
        # An NCP routes outputs through a MOTOR-neuron subset, so the output
        # dimension is strictly smaller than the total neuron count. A dense
        # FullyConnected LTC has output_dim == units.
        assert w.output_dim < w.units, (
            f"{name}: output_dim {w.output_dim} == units {w.units} (dense, not NCP)")


def test_proj_inputs_match_ncp_motor_outputs():
    """Each projection must read from its NCP's motor-neuron output (output_dim),
    not the full hidden state — otherwise the forward pass dimensions break."""
    policy = SNCPPolicy()
    assert policy.temporal_proj.in_features == policy.temporal_wiring.output_dim
    assert policy.spatial_proj.in_features == policy.spatial_wiring.output_dim
    assert policy.node_proj.in_features == policy.node_wiring.output_dim


def test_node_encoder_sized_for_640_dim_fusion_input():
    """The node encoder fuses robot(128)+temporal(256)+attention(256)=640 dims.
    Its inter-neuron layer (units - output - command) is where those 640 inputs
    first land, so it must not be tighter than the dense baseline it replaces
    (32). We size the node NCP up to keep adequate fusion capacity."""
    policy = SNCPPolicy()
    w = policy.node_wiring
    import math
    command = math.ceil(0.4 * w.output_dim)
    inter = w.units - w.output_dim - command
    assert inter >= 32, f"node inter-neuron capacity {inter} < dense baseline 32"


def test_forward_pass_still_works_with_ncp():
    """End-to-end forward with the sparse NCP encoders: shapes + action limits."""
    policy = SNCPPolicy(robot_vpref=0.26, robot_wmax=1.8)
    B, H = 2, 3
    obs = {
        'robot_node': torch.randn(B, 7),
        'spatial_edges': torch.randn(B, H, 6),
        'temporal_edges': torch.randn(B, 2),
    }
    h = policy.init_hidden(B, H, torch.device('cpu'))
    mu, std, value, h_new = policy(obs, h)
    assert mu.shape == (B, 2) and std.shape == (B, 2) and value.shape == (B, 1)
    assert torch.all(mu[:, 0] >= 0.0) and torch.all(mu[:, 0] <= 0.26)
    assert torch.all(mu[:, 1] >= -1.8) and torch.all(mu[:, 1] <= 1.8)
