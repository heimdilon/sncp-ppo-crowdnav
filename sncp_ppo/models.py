import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC
from ncps.wirings import AutoNCP


def _orthogonal_linear(layer, gain):
    """Apply orthogonal init to nn.Linear weights and zero out biases.
    Standard PPO best practice (Engstrom et al., "Implementation Matters")."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class SNCPPolicy(nn.Module):
    def __init__(self, robot_vpref=0.26, robot_wmax=1.8, pre_mlp=False,
                 attn_count_scaling=False, meanmax_pool=False, node_units=128,
                 node_output=48, attn_heads=1, action_dist='gaussian'):
        super(SNCPPolicy, self).__init__()

        self.robot_vpref = robot_vpref
        self.robot_wmax = robot_wmax
        self.pre_mlp = pre_mlp
        # Paper Eq 13 scales attention scores by n/sqrt(d_k) (n = #humans); we
        # historically used 1/sqrt(d_k), making the pooled vector a pure
        # weighted average that loses count/density info at high N. With this
        # flag on, the n factor feeds the pedestrian count into the softmax
        # temperature. A buffer is registered ONLY when on, so default
        # checkpoints stay byte-identical and build_policy_for_checkpoint can
        # auto-detect the variant (same pattern as pre_mlp).
        self.attn_count_scaling = attn_count_scaling
        if attn_count_scaling:
            self.register_buffer('_attn_count_scaling', torch.tensor(1.0))

        # Mean+max attention pooling (v30): the default pool is a convex combination
        # Sum_h alpha_h * M_rh,h, which regresses to the mean at high N and dilutes the
        # most-threatening agent. meanmax_pool concats that mean with an element-wise
        # max over humans (cardinality-robust) and merges them with pool_merge. The
        # layer exists ONLY when on, so default checkpoints stay byte-identical and
        # build_policy_for_checkpoint can auto-detect the variant (pre_mlp pattern).
        self.meanmax_pool = meanmax_pool
        self.action_dist = action_dist

        # 1. Robot Node Encoder (MLP)
        # Input robot_node: [dg_local_x, dg_local_y, v_linear, dist_to_goal, vpref, radius, w_angular] (dim 7)
        self.robot_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # 2. Temporal Edge Encoder — TRUE sparse NCP (AutoNCP), not a dense LTC.
        # The paper (Ao et al. 2026) claims Neural Circuit Policies but omits the
        # wiring parameters; we choose them for our problem. AutoNCP(units, motor)
        # builds the ~90%-sparse sensory->inter->command->motor C. elegans circuit.
        # The LTC output is the MOTOR-neuron subset (output_dim), then projected to
        # 256. Seeded so the random topology is reproducible (the sparsity masks
        # are persisted inside the checkpoint's state_dict).
        #
        # pre_mlp=True restores the paper's Eq 11 ordering: the raw edge input is
        # first expanded to the paper's encoding dimension (time edge = 256) by an
        # MLP and the NCP consumes that embedding, instead of eating the raw 2-dim
        # signal. Default False keeps every existing checkpoint loadable.
        self.temporal_wiring = AutoNCP(units=32, output_size=16, seed=48201)
        if pre_mlp:
            self.temporal_pre_mlp = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
            )
            self.temporal_ltc = LTC(input_size=256, units=self.temporal_wiring)
        else:
            self.temporal_ltc = LTC(input_size=2, units=self.temporal_wiring)
        self.temporal_proj = nn.Linear(self.temporal_wiring.output_dim, 256)

        # 3. Spatial Edge Encoder — sparse NCP. Raw input dim 6:
        # [dx, dy, rel_vx, rel_vy, goal_dir_x, goal_dir_y] per human. Sized a bit
        # larger (48 units / 24 motor) since this encoder carries the crowd signal.
        # The paper does not give the spatial-edge embedding size; with pre_mlp we
        # use 256 symmetric to the time edge.
        self.spatial_wiring = AutoNCP(units=48, output_size=24, seed=48202)
        if pre_mlp:
            self.spatial_pre_mlp = nn.Sequential(
                nn.Linear(6, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
            )
            self.spatial_ltc = LTC(input_size=256, units=self.spatial_wiring)
        else:
            self.spatial_ltc = LTC(input_size=6, units=self.spatial_wiring)
        self.spatial_proj = nn.Linear(self.spatial_wiring.output_dim, 256)
        
        # 4. Attention Pooling weights
        # Single-head (default, attn_heads=1): legacy projection — robot key m_rr
        # scores each human, Value = raw M_rh. Byte-identical to v14-v32.
        # Multi-head (attn_heads>1, v33): canonical cross-attention — robot is the
        # query token, humans are key/value tokens, d_model=256 split across heads
        # so each head specializes on a different simultaneous threat (the high-N
        # failure mode). A buffer persists the head count (not recoverable from any
        # weight shape) so build_policy_for_checkpoint can auto-detect the variant.
        self.attn_heads = attn_heads
        if attn_heads > 1:
            assert 256 % attn_heads == 0, "attn_heads must divide d_model=256"
            self.W_q = nn.Linear(256, 256)
            self.W_k = nn.Linear(256, 256)
            self.W_v = nn.Linear(256, 256)
            self.W_o = nn.Linear(256, 256)
            self.register_buffer('_attn_heads', torch.tensor(float(attn_heads)))
        else:
            self.W_q = nn.Linear(256, 64)
            self.W_k = nn.Linear(256, 64)
        if meanmax_pool:
            self.pool_merge = nn.Linear(512, 256)
        
        # 5. Node NCP Encoder — sparse NCP fusing robot(128)+temporal(256)+
        # attention(256)=640 dims. Sized up to 128 units / 48 motor so the
        # inter-neuron layer (=60) that first absorbs the 640 fused inputs stays
        # WIDER than the old dense-32 bottleneck (the 640->32 squeeze the user
        # flagged as the capacity ceiling).
        self.node_units, self.node_output = node_units, node_output
        self.node_wiring = AutoNCP(units=node_units, output_size=node_output, seed=48203)
        self.node_ltc = LTC(input_size=640, units=self.node_wiring)
        self.node_proj = nn.Linear(self.node_wiring.output_dim, 256)
        
        # 6. Actor & Critic Heads
        # action_dist='gaussian' (default): mean head (2) + global logstd; mean is
        # scaled by sigmoid/tanh, sampled as Normal, then clipped by the env.
        # action_dist='beta' (v34): head emits 4 raw values -> alpha,beta (softplus+1
        # => unimodal) for a Beta on [0,1]^2, scaled to the physical action box by the
        # PPO layer. No logstd. Naturally bounded (no clip bias), state-dependent.
        if action_dist == 'beta':
            self.actor_mu = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 4)
            )
            self.register_buffer('action_low', torch.tensor([0.0, -robot_wmax]))
            self.register_buffer('action_high', torch.tensor([robot_vpref, robot_wmax]))
        else:
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
        if self.pre_mlp:
            for mlp in (self.temporal_pre_mlp, self.spatial_pre_mlp):
                for m in mlp:
                    if isinstance(m, nn.Linear):
                        _orthogonal_linear(m, gain=sqrt2)
        _orthogonal_linear(self.temporal_proj, gain=sqrt2)
        _orthogonal_linear(self.spatial_proj, gain=sqrt2)
        _orthogonal_linear(self.W_q, gain=sqrt2)
        _orthogonal_linear(self.W_k, gain=sqrt2)
        if self.attn_heads > 1:
            _orthogonal_linear(self.W_v, gain=sqrt2)
            _orthogonal_linear(self.W_o, gain=sqrt2)
        _orthogonal_linear(self.node_proj, gain=sqrt2)
        if self.meanmax_pool:
            _orthogonal_linear(self.pool_merge, gain=sqrt2)
        
        linears = [m for m in self.actor_mu if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _orthogonal_linear(m, gain=sqrt2)
        _orthogonal_linear(linears[-1], gain=0.01)
        # Gaussian only: bias the linear-velocity pre-activation so sigmoid(2.0)·vpref
        # ≈ 0.88·vpref at init. The robot needs ≥0.133 m/s to cover the 8 m goal within
        # the budget; the old default sat on the timeout cliff. Beta keeps bias 0 ->
        # alpha=beta=softplus(0)+1≈1.69 (symmetric, mean≈0.5·vpref, ample at vpref=1.0).
        if self.action_dist == 'gaussian':
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

    def _multihead_attention(self, M_rh, m_rr):
        """Canonical multi-head cross-attention: robot m_rr is the single query
        token, humans M_rh are the key/value tokens. Returns (a_attn [B,256],
        alpha [B, heads, 1, H]); each head has d_head = 256 // heads dims."""
        B, H, _ = M_rh.shape
        nh = self.attn_heads
        dh = 256 // nh
        Q = self.W_q(m_rr).view(B, nh, 1, dh)                        # [B, nh, 1, dh]
        K = self.W_k(M_rh).view(B, H, nh, dh).permute(0, 2, 1, 3)    # [B, nh, H, dh]
        V = self.W_v(M_rh).view(B, H, nh, dh).permute(0, 2, 1, 3)    # [B, nh, H, dh]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dh)  # [B, nh, 1, H]
        alpha = F.softmax(scores, dim=-1)
        ctx = torch.matmul(alpha, V).reshape(B, 256)                 # [B, 256]
        return self.W_o(ctx), alpha

    def _attention_pool(self, M_rh, m_rr, num_humans):
        """Attention-weighted pooling of per-human spatial features M_rh against
        the robot/temporal key m_rr. attn_heads>1 uses multi-head cross-attention;
        otherwise the legacy single-head weighted average. attn_count_scaling
        (single-head only) scales scores by n (paper Eq 13)."""
        if self.attn_heads > 1:
            a_attn, _ = self._multihead_attention(M_rh, m_rr)
            if not self.meanmax_pool:
                return a_attn
            a_max = M_rh.max(dim=1).values
            return self.pool_merge(torch.cat([a_attn, a_max], dim=1))
        Q = self.W_q(M_rh)                      # [B, H, 64]
        K = self.W_k(m_rr).unsqueeze(1)         # [B, 1, 64]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / 8.0   # /sqrt(d_k)
        if self.attn_count_scaling:
            attn_scores = attn_scores * num_humans
        alpha = F.softmax(attn_scores, dim=1)   # [B, H, 1]
        a_mean = torch.bmm(M_rh.transpose(1, 2), alpha).squeeze(2)  # [B, 256]
        if not self.meanmax_pool:
            return a_mean
        a_max = M_rh.max(dim=1).values          # [B, 256] cardinality-robust
        return self.pool_merge(torch.cat([a_mean, a_max], dim=1))  # [B, 256]

    def _scale_action(self, x):
        """Map x in [0,1]^2 to the physical action box [0,vpref]x[-wmax,wmax]."""
        return self.action_low + (self.action_high - self.action_low) * x

    def _unscale_action(self, a):
        """Inverse of _scale_action; clamp to (eps,1-eps) for Beta.log_prob safety."""
        x = (a - self.action_low) / (self.action_high - self.action_low)
        return x.clamp(1e-6, 1.0 - 1e-6)

    def deterministic_action(self, out1, out2):
        """Greedy action: Gaussian mean (already physical) or scaled Beta mean."""
        if self.action_dist == 'beta':
            return self._scale_action(out1 / (out1 + out2))   # alpha/(alpha+beta)
        return out1

    def forward(self, obs, hidden_states):
        """
        Forward pass of the SNCP model.
        Args:
            obs: dict containing 'robot_node' (B,7), 'spatial_edges' (B,H,4)
                 = [pos_local, rel_vel_local], 'temporal_edges' (B,2). All in
                 robot-local frame.
            hidden_states: dict with 'temporal_edge', 'spatial_edge', 'node' LTC states
        Returns:
            mu: actor mean [batch_size, 2] — scaled to (v in [0, vpref], w in [-wmax, wmax])
            std: actor std [batch_size, 2] — broadcast from per-dim logstd parameter
            value: critic value [batch_size, 1]
            new_hidden_states: updated LTC hidden states
        """
        robot_node = obs['robot_node']        # [batch_size, 7]
        spatial_edges = obs['spatial_edges']  # [batch_size, num_humans, 4]
        temporal_edges = obs['temporal_edges']# [batch_size, 2]
        
        batch_size = robot_node.shape[0]
        num_humans = spatial_edges.shape[1]
        
        # 1. Robot Node Encoding
        v_m = self.robot_mlp(robot_node)  # [batch_size, 128]
        
        # 2. Temporal Edge Encoding (LTC). With pre_mlp, the paper's Eq 11
        # ordering: expand to the 256-dim encoding first, then the NCP.
        temporal_features = self.temporal_pre_mlp(temporal_edges) if self.pre_mlp else temporal_edges
        temporal_input = temporal_features.unsqueeze(1)
        h_temp = hidden_states['temporal_edge']
        m_rr_seq, h_temp_new = self.temporal_ltc(temporal_input, h_temp)
        m_rr = self.temporal_proj(m_rr_seq.squeeze(1))

        # 3. Spatial Edge Encoding (LTC)
        spatial_flat = spatial_edges.reshape(batch_size * num_humans, 6)
        if self.pre_mlp:
            spatial_flat = self.spatial_pre_mlp(spatial_flat)
        spatial_input = spatial_flat.unsqueeze(1)
        h_spat = hidden_states['spatial_edge']
        if h_spat.dim() == 3:
            h_spat_flat = h_spat.reshape(batch_size * num_humans, -1)
        else:
            h_spat_flat = h_spat
            
        M_rh_seq, h_spat_new_flat = self.spatial_ltc(spatial_input, h_spat_flat)
        M_rh_proj = self.spatial_proj(M_rh_seq.squeeze(1))
        M_rh = M_rh_proj.reshape(batch_size, num_humans, 256)
        
        # 4. Attention Pooling
        u_att = self._attention_pool(M_rh, m_rr, num_humans)
        
        # 5. Node LTC Encoder
        node_input = torch.cat([v_m, m_rr, u_att], dim=-1).unsqueeze(1)
        h_node = hidden_states['node']
        sf_seq, h_node_new = self.node_ltc(node_input, h_node)
        sf = self.node_proj(sf_seq.squeeze(1))
        
        # 6. Actor & Critic Outputs
        actor_raw = self.actor_mu(sf)
        if self.action_dist == 'beta':
            # alpha,beta for a Beta on [0,1]^2; +1 => unimodal. Scaling to the
            # physical action box happens in the PPO layer (_scale_action).
            out1 = F.softplus(actor_raw[:, :2]) + 1.0   # alpha [B,2]
            out2 = F.softplus(actor_raw[:, 2:]) + 1.0   # beta  [B,2]
        else:
            # Gaussian: scale mean to physical limits, std from the global logstd.
            v_mu = torch.sigmoid(actor_raw[:, 0:1]) * self.robot_vpref
            w_mu = torch.tanh(actor_raw[:, 1:2]) * self.robot_wmax
            out1 = torch.cat([v_mu, w_mu], dim=-1)               # mu  [B,2]
            out2 = torch.exp(self.actor_logstd).expand_as(out1)  # std [B,2]
        
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

        return out1, out2, value, new_hidden_states


def build_policy_for_checkpoint(state_dict, robot_vpref=0.26, robot_wmax=1.8):
    """Construct an SNCPPolicy whose architecture matches a saved state dict.

    Checkpoints are plain `policy.state_dict()` files; the only architecture
    variant is the paper-Eq-11 pre-MLP (v22+), detectable from its keys. Old
    checkpoints (v14..v21) have no `*_pre_mlp` keys and get the legacy layout.
    """
    pre_mlp = any(key.startswith('temporal_pre_mlp') for key in state_dict)
    attn_count_scaling = '_attn_count_scaling' in state_dict
    meanmax_pool = any(key.startswith('pool_merge') for key in state_dict)
    gleak = state_dict.get('node_ltc.rnn_cell.gleak')
    node_units = int(gleak.shape[0]) if gleak is not None else 128
    out_w = state_dict.get('node_ltc.rnn_cell.output_w')
    node_output = int(out_w.shape[0]) if out_w is not None else 48
    ah = state_dict.get('_attn_heads')
    attn_heads = int(ah.item()) if ah is not None else 1
    action_dist = 'gaussian' if 'actor_logstd' in state_dict else 'beta'
    return SNCPPolicy(robot_vpref=robot_vpref, robot_wmax=robot_wmax,
                      pre_mlp=pre_mlp, attn_count_scaling=attn_count_scaling,
                      meanmax_pool=meanmax_pool, node_units=node_units,
                      node_output=node_output, attn_heads=attn_heads,
                      action_dist=action_dist)
