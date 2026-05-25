import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC
from ncps.wirings import FullyConnected


def _orthogonal_linear(layer, gain):
    """Apply orthogonal init to nn.Linear weights and zero out biases.
    Standard PPO best practice (Engstrom et al., "Implementation Matters")."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class SNCPPolicy(nn.Module):
    def __init__(self, robot_vpref=0.26, robot_wmax=1.8):
        super(SNCPPolicy, self).__init__()
        
        self.robot_vpref = robot_vpref
        self.robot_wmax = robot_wmax
        
        # 1. Robot Node Encoder (MLP)
        # Input robot_node: [dg_local_x, dg_local_y, v_linear, dist_to_goal, vpref, radius, w_angular] (dim 7)
        self.robot_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 2. Temporal Edge Encoder (LTC size 32 -> project to 256)
        self.temporal_wiring = FullyConnected(units=32)
        self.temporal_ltc = LTC(input_size=2, units=self.temporal_wiring)
        self.temporal_proj = nn.Linear(32, 256)
        
        # 3. Spatial Edge Encoder (LTC size 32 -> project to 256)
        self.spatial_wiring = FullyConnected(units=32)
        self.spatial_ltc = LTC(input_size=2, units=self.spatial_wiring)
        self.spatial_proj = nn.Linear(32, 256)
        
        # 4. Attention Pooling weights
        self.W_q = nn.Linear(256, 64)
        self.W_k = nn.Linear(256, 64)
        
        # 5. Node NCP Encoder (LTC size 32 -> project to 256)
        self.node_wiring = FullyConnected(units=32)
        self.node_ltc = LTC(input_size=640, units=self.node_wiring)
        self.node_proj = nn.Linear(32, 256)
        
        # 6. Actor & Critic Heads
        self.actor_mu = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        # Initial std scaled per action dimension to avoid wasted clipping.
        # Linear v in [0, 0.26]: exp(-2.0) ≈ 0.135 → ~half-range exploration.
        # Angular w in [-1.8, 1.8]: exp(-1.5) ≈ 0.22 — kept small so that
        # σθ per step = 0.22 · 0.25 ≈ 3.2° and the 240-step heading random
        # walk stays under ~50°. With the old exp(-0.5) ≈ 0.607 the heading
        # drift was ~135° per episode and the robot could not maintain
        # orientation toward the goal.
        self.actor_logstd = nn.Parameter(torch.tensor([[-2.0, -1.5]]), requires_grad=True)
        
        self.critic = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self._init_linear_weights()

    def _init_linear_weights(self):
        sqrt2 = math.sqrt(2.0)
        for m in self.robot_mlp:
            if isinstance(m, nn.Linear):
                _orthogonal_linear(m, gain=sqrt2)
        _orthogonal_linear(self.temporal_proj, gain=sqrt2)
        _orthogonal_linear(self.spatial_proj, gain=sqrt2)
        _orthogonal_linear(self.W_q, gain=sqrt2)
        _orthogonal_linear(self.W_k, gain=sqrt2)
        _orthogonal_linear(self.node_proj, gain=sqrt2)
        
        linears = [m for m in self.actor_mu if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _orthogonal_linear(m, gain=sqrt2)
        _orthogonal_linear(linears[-1], gain=0.01)
        # Bias the linear-velocity pre-activation so sigmoid(2.0)·vpref ≈ 0.88·vpref
        # at init (~0.23 m/s vs the old 0.13 m/s = sigmoid(0)·vpref). The robot
        # needs ≥0.133 m/s to cover the 8 m goal distance within max_time = 60 s;
        # the old default was right on the timeout cliff so the agent never
        # received a goal-reward signal to bootstrap learning.
        with torch.no_grad():
            linears[-1].bias.data.copy_(torch.tensor([2.0, 0.0]))

        linears = [m for m in self.critic if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _orthogonal_linear(m, gain=sqrt2)
        _orthogonal_linear(linears[-1], gain=1.0)

    def init_hidden(self, batch_size, num_humans, device):
        h_temp = torch.zeros(batch_size, self.temporal_wiring.units, device=device)
        h_spat = torch.zeros(batch_size * num_humans, self.spatial_wiring.units, device=device)
        h_node = torch.zeros(batch_size, self.node_wiring.units, device=device)
        return {
            'temporal_edge': h_temp,
            'spatial_edge': h_spat,
            'node': h_node
        }

    def forward(self, obs, hidden_states):
        """
        Forward pass of the SNCP model.
        Args:
            obs: dict containing 'robot_node' (B,7), 'spatial_edges' (B,H,2),
                 'temporal_edges' (B,2). All in robot-local frame.
            hidden_states: dict with 'temporal_edge', 'spatial_edge', 'node' LTC states
        Returns:
            mu: actor mean [batch_size, 2] — scaled to (v in [0, vpref], w in [-wmax, wmax])
            std: actor std [batch_size, 2] — broadcast from per-dim logstd parameter
            value: critic value [batch_size, 1]
            new_hidden_states: updated LTC hidden states
        """
        robot_node = obs['robot_node']        # [batch_size, 7]
        spatial_edges = obs['spatial_edges']  # [batch_size, num_humans, 2]
        temporal_edges = obs['temporal_edges']# [batch_size, 2]
        
        batch_size = robot_node.shape[0]
        num_humans = spatial_edges.shape[1]
        
        # 1. Robot Node Encoding
        v_m = self.robot_mlp(robot_node)  # [batch_size, 128]
        
        # 2. Temporal Edge Encoding (LTC)
        temporal_input = temporal_edges.unsqueeze(1)
        h_temp = hidden_states['temporal_edge']
        m_rr_seq, h_temp_new = self.temporal_ltc(temporal_input, h_temp)
        m_rr = self.temporal_proj(m_rr_seq.squeeze(1))
        
        # 3. Spatial Edge Encoding (LTC)
        spatial_input = spatial_edges.reshape(batch_size * num_humans, 1, 2)
        h_spat = hidden_states['spatial_edge']
        if h_spat.dim() == 3:
            h_spat_flat = h_spat.reshape(batch_size * num_humans, -1)
        else:
            h_spat_flat = h_spat
            
        M_rh_seq, h_spat_new_flat = self.spatial_ltc(spatial_input, h_spat_flat)
        M_rh_proj = self.spatial_proj(M_rh_seq.squeeze(1))
        M_rh = M_rh_proj.reshape(batch_size, num_humans, 256)
        
        # 4. Attention Pooling
        Q = self.W_q(M_rh)
        K = self.W_k(m_rr).unsqueeze(1)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / 8.0
        alpha = F.softmax(attn_scores, dim=1)
        u_att = torch.bmm(M_rh.transpose(1, 2), alpha).squeeze(2)
        
        # 5. Node LTC Encoder
        node_input = torch.cat([v_m, m_rr, u_att], dim=-1).unsqueeze(1)
        h_node = hidden_states['node']
        sf_seq, h_node_new = self.node_ltc(node_input, h_node)
        sf = self.node_proj(sf_seq.squeeze(1))
        
        # 6. Actor & Critic Outputs
        mu_raw = self.actor_mu(sf)  # [batch_size, 2]
        
        # Scale actor outputs to physical robot limits
        # Linear velocity: [0, robot_vpref]
        v_mu = torch.sigmoid(mu_raw[:, 0:1]) * self.robot_vpref
        # Angular velocity: [-robot_wmax, robot_wmax]
        w_mu = torch.tanh(mu_raw[:, 1:2]) * self.robot_wmax
        mu = torch.cat([v_mu, w_mu], dim=-1)
        
        # Standard deviation for PPO exploration
        std = torch.exp(self.actor_logstd).expand_as(mu)
        
        # State value
        value = self.critic(sf)  # [batch_size, 1]
        
        if h_spat.dim() == 3:
            h_spat_new = h_spat_new_flat.reshape(batch_size, num_humans, -1)
        else:
            h_spat_new = h_spat_new_flat
            
        new_hidden_states = {
            'temporal_edge': h_temp_new,
            'spatial_edge': h_spat_new,
            'node': h_node_new
        }
        
        return mu, std, value, new_hidden_states
