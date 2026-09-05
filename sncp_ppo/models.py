import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP


VALID_CELL_TYPES = ('ltc', 'cfc')
SPARSE_HIG_MAX_K = 4
_LTC_MODULE_PREFIXES = ('temporal_ltc.', 'spatial_ltc.', 'node_ltc.')
_CFC_MODULE_PREFIXES = ('temporal_cfc.', 'spatial_cfc.', 'node_cfc.')
_DENSE_HH_PREFIXES = ('hh_attn.',)
_SPARSE_HH_PREFIXES = ('hh_sparse_attn.',)


class CellTypeMismatchError(ValueError):
    """LTC and CfC encoder weights are not interchangeable."""


class SparseHIGMismatchError(ValueError):
    """Dense v37 HH (full MHA) and SparseHIG weights are not interchangeable."""


def _orthogonal_linear(layer, gain):
    """Apply orthogonal init to nn.Linear weights and zero out biases.
    Standard PPO best practice (Engstrom et al., "Implementation Matters")."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class SNCPPolicy(nn.Module):
    def __init__(self, robot_vpref=0.26, robot_wmax=1.8, pre_mlp=False,
                 attn_count_scaling=False, meanmax_pool=False, node_units=128,
                 node_output=48, attn_heads=1, action_dist='gaussian', sense_range=0.0,
                 hh_intent_graph=False, hh_attn_heads=4,
                 cv_horizons=(1, 2, 3, 4), cv_dt=0.25, risk_head=False,
                 cell_type='ltc', sparse_hig=False, hh_topk=3):
        super(SNCPPolicy, self).__init__()

        self.robot_vpref = robot_vpref
        self.robot_wmax = robot_wmax
        self.cell_type = _normalize_cell_type(cell_type)
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
        # Sense-range masking (v35): when >0, humans beyond this radius (metres) are
        # excluded from the crowd attention pool — the paper's limited perception
        # (challenging = 6 m). A buffer persists it for auto-detect; default 0 = sense
        # all humans (v14-v34 byte-identical).
        self.sense_range = sense_range
        if sense_range > 0:
            self.register_buffer('_sense_range', torch.tensor(float(sense_range)))

        # v37: gated human-human intention graph. This branch is conditional so
        # every pre-v37/default checkpoint keeps exactly the same state-dict
        # surface. Constant-velocity geometry is derived inside the policy from
        # the existing [dx,dy,rel_vx,rel_vy] spatial edge fields; no observation
        # or environment schema change is required. The scalar gate starts at
        # zero, making an upgraded checkpoint behaviorally identical to its base.
        #
        # M2' SparseHIG: optional top-k (k≤4) neighbor HH instead of the full
        # H×H MHA. Distinct module name (hh_sparse_attn) so dense v37 weights
        # cannot silently load. sparse_hig implies the v37 CV+gate surface.
        self.sparse_hig = bool(sparse_hig)
        self.hh_intent_graph = bool(hh_intent_graph) or self.sparse_hig
        self.hh_attn_heads = int(hh_attn_heads)
        self.hh_topk = int(hh_topk) if self.sparse_hig else 0
        self.cv_horizons = tuple(float(h) for h in cv_horizons)
        self.cv_dt = float(cv_dt)
        if self.sparse_hig and not (0 <= self.hh_topk <= SPARSE_HIG_MAX_K):
            raise ValueError(
                f"hh_topk must be in 0..{SPARSE_HIG_MAX_K} for SparseHIG, "
                f"got {hh_topk!r}"
            )
        if self.hh_intent_graph:
            if not self.cv_horizons or any(h <= 0 for h in self.cv_horizons):
                raise ValueError("cv_horizons must contain positive values")
            if self.hh_attn_heads <= 0 or 256 % self.hh_attn_heads != 0:
                raise ValueError("hh_attn_heads must be positive and divide 256")
            self.register_buffer('_hh_intent_graph', torch.tensor(1.0))
            self.register_buffer('_hh_attn_heads', torch.tensor(float(self.hh_attn_heads)))
            self.register_buffer(
                '_cv_horizons', torch.tensor(self.cv_horizons, dtype=torch.float32)
            )
            self.register_buffer('_cv_dt', torch.tensor(self.cv_dt, dtype=torch.float32))
            self.cv_encoder = nn.Sequential(
                nn.Linear(2 * len(self.cv_horizons), 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
            )
            self.hh_norm = nn.LayerNorm(256)
            hh_mha = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=self.hh_attn_heads,
                batch_first=True,
            )
            if self.sparse_hig:
                self.register_buffer('_hh_sparse_k', torch.tensor(float(self.hh_topk)))
                self.hh_sparse_attn = hh_mha
            else:
                self.hh_attn = hh_mha
            self.hh_gate = nn.Parameter(torch.tensor(0.0))

        # v39: tiny fusion-level risk head. Conditional so every pre-v39
        # checkpoint keeps the same state-dict surface. Predicts a short-horizon
        # collision probability and a non-negative min-clearance from the node
        # fusion features; trained with privileged CV labels, then discarded as
        # a runtime controller — inference stays one forward pass, no shield.
        self.risk_head = bool(risk_head)
        self.last_p_coll = None
        self.last_min_clearance = None
        self.last_cost_value = None
        if self.risk_head:
            self.register_buffer('_risk_head', torch.tensor(1.0))
            self.risk_mlp = nn.Sequential(
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
            # Discounted cost-to-go critic. Separate from p_coll (short-horizon
            # collision classifier used only in L_risk BCE).
            self.cost_critic = nn.Linear(256, 1)

        # Side-research CfC marker. Registered ONLY when cell_type='cfc' so
        # default LTC checkpoints stay byte-identical (same pattern as pre_mlp
        # / risk_head). build_policy_for_checkpoint auto-detects from this
        # buffer or from temporal_cfc.* keys.
        if self.cell_type == 'cfc':
            self.register_buffer('_cell_type_cfc', torch.tensor(1.0))

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
            self._set_ncp_cell('temporal', input_size=256, wiring=self.temporal_wiring)
        else:
            self._set_ncp_cell('temporal', input_size=2, wiring=self.temporal_wiring)
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
            self._set_ncp_cell('spatial', input_size=256, wiring=self.spatial_wiring)
        else:
            self._set_ncp_cell('spatial', input_size=6, wiring=self.spatial_wiring)
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
        self._set_ncp_cell('node', input_size=640, wiring=self.node_wiring)
        self.node_proj = nn.Linear(self.node_wiring.output_dim, 256)
        if self.cell_type == 'cfc':
            self.register_buffer('_cfc_node_units', torch.tensor(float(self.node_units)))
            self.register_buffer('_cfc_node_output', torch.tensor(float(self.node_output)))
        
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

    def _set_ncp_cell(self, name, input_size, wiring):
        """Attach temporal/spatial/node as *_ltc (default) or *_cfc.

        Module names stay distinct so LTC and CfC checkpoints cannot load into
        each other without a missing/unexpected-key error.
        """
        cell = (CfC(input_size, wiring) if self.cell_type == 'cfc'
                else LTC(input_size, wiring))
        setattr(self, f'{name}_{self.cell_type}', cell)

    def _ncp(self, name):
        return getattr(self, f'{name}_{self.cell_type}')

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
        if self.hh_intent_graph:
            for m in self.cv_encoder:
                if isinstance(m, nn.Linear):
                    _orthogonal_linear(m, gain=sqrt2)
            hh_mha = self._hh_mha()
            nn.init.xavier_uniform_(hh_mha.in_proj_weight)
            if hh_mha.in_proj_bias is not None:
                nn.init.zeros_(hh_mha.in_proj_bias)
            _orthogonal_linear(hh_mha.out_proj, gain=sqrt2)
        
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

        if self.risk_head:
            risk_linears = [m for m in self.risk_mlp if isinstance(m, nn.Linear)]
            for m in risk_linears[:-1]:
                _orthogonal_linear(m, gain=sqrt2)
            _orthogonal_linear(risk_linears[-1], gain=1.0)
            _orthogonal_linear(self.cost_critic, gain=1.0)

    def init_hidden(self, batch_size, num_humans, device):
        h_temp = torch.zeros(batch_size, self.temporal_wiring.units, device=device)
        h_spat = torch.zeros(batch_size * num_humans, self.spatial_wiring.units, device=device)
        h_node = torch.zeros(batch_size, self.node_wiring.units, device=device)
        return {
            'temporal_edge': h_temp,
            'spatial_edge': h_spat,
            'node': h_node
        }

    @staticmethod
    def _masked_max(M_rh, mask, none_visible):
        """Element-wise max over visible humans (mask True = visible); rows with no
        visible human return a zero vector instead of -inf."""
        masked = M_rh.masked_fill((~mask).unsqueeze(-1), float('-inf'))
        a_max = masked.max(dim=1).values
        return torch.where(none_visible.unsqueeze(-1), torch.zeros_like(a_max), a_max)

    def _constant_velocity_features(self, spatial_edges):
        """Future robot-relative human positions for each configured horizon.

        spatial_edges[..., :2] is current relative position and [..., 2:4]
        relative velocity. The returned shape is [B,H,2*len(cv_horizons)].
        """
        if not self.hh_intent_graph:
            raise RuntimeError("constant-velocity features require hh_intent_graph=True")
        rel_pos = spatial_edges[..., 0:2].unsqueeze(-2)
        rel_vel = spatial_edges[..., 2:4].unsqueeze(-2)
        times = (self._cv_horizons * self._cv_dt).to(
            device=spatial_edges.device, dtype=spatial_edges.dtype
        ).view(1, 1, -1, 1)
        future = rel_pos + rel_vel * times
        return future.flatten(start_dim=-2)

    def _hh_mha(self):
        """Dense v37 `hh_attn` or SparseHIG `hh_sparse_attn`."""
        return self.hh_sparse_attn if self.sparse_hig else self.hh_attn

    def _topk_neighbor_index(self, spatial_edges, mask=None):
        """Nearest-other-human indices [B,H,k] and validity [B,H,k].

        Distance is pairwise Euclidean on robot-local (dx, dy). Self is never a
        neighbor. Invisible humans (mask False) cannot be selected. When H-1 < k
        the leftover slots are invalid pads (index 0, valid=False).
        """
        k = self.hh_topk
        pos = spatial_edges[..., :2]
        batch, num_humans, _ = pos.shape
        if k <= 0 or num_humans <= 0:
            empty_idx = pos.new_zeros(batch, num_humans, max(k, 0), dtype=torch.long)
            empty_valid = torch.zeros(batch, num_humans, max(k, 0), dtype=torch.bool,
                                      device=pos.device)
            return empty_idx, empty_valid

        dist = torch.cdist(pos, pos)
        self_mask = torch.eye(num_humans, dtype=torch.bool, device=pos.device)
        dist = dist.masked_fill(self_mask.unsqueeze(0), float('inf'))
        if mask is not None:
            dist = dist.masked_fill(~mask.unsqueeze(1), float('inf'))
            dist = dist.masked_fill(~mask.unsqueeze(2), float('inf'))
        if k > num_humans:
            pad = dist.new_full((batch, num_humans, k - num_humans), float('inf'))
            dist = torch.cat([dist, pad], dim=-1)
        values, idx = dist.topk(k, dim=-1, largest=False)
        valid = torch.isfinite(values)
        idx = idx.clamp(min=0, max=max(num_humans - 1, 0))
        return idx, valid

    def _sparse_hh_attention(self, tokens, spatial_edges, mask=None):
        """Query each human against its k nearest others; pad-safe when H < k."""
        if self.hh_topk <= 0:
            return torch.zeros_like(tokens)
        idx, valid = self._topk_neighbor_index(spatial_edges, mask)
        batch, num_humans, dim = tokens.shape
        k = self.hh_topk
        gather_idx = idx.reshape(batch, num_humans * k).unsqueeze(-1).expand(
            batch, num_humans * k, dim
        )
        gathered = torch.gather(tokens, 1, gather_idx).reshape(batch, num_humans, k, dim)
        query = tokens.reshape(batch * num_humans, 1, dim)
        key_value = gathered.reshape(batch * num_humans, k, dim)
        key_padding = ~valid.reshape(batch * num_humans, k)
        none_valid = ~valid.any(dim=-1)
        none_valid_flat = none_valid.reshape(batch * num_humans)
        if bool(none_valid_flat.any()):
            key_padding = key_padding.clone()
            key_padding[none_valid_flat, 0] = False
        hh_flat, _ = self.hh_sparse_attn(
            query, key_value, key_value,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        hh = hh_flat.reshape(batch, num_humans, dim)
        hh = hh.masked_fill(none_valid.unsqueeze(-1), 0.0)
        if mask is not None:
            hh = hh.masked_fill((~mask).unsqueeze(-1), 0.0)
        return hh

    def _human_intent_graph(self, M_rh, spatial_edges, mask=None):
        """Apply gated human-human self-attention over CV-enriched tokens.

        Dense path: full H×H MHA (v37). SparseHIG: each human attends to its
        k≤4 nearest others. mask uses the policy convention True=visible, while
        PyTorch MHA expects True=hidden in key_padding_mask. Rows with no
        visible humans temporarily expose one safe key to avoid all-masked
        softmax NaNs, then zero the whole residual. Hidden query rows are also
        zeroed.
        """
        if not self.hh_intent_graph:
            return M_rh
        cv_embed = self.cv_encoder(self._constant_velocity_features(spatial_edges))
        tokens = self.hh_norm(M_rh + cv_embed)
        if self.sparse_hig:
            hh = self._sparse_hh_attention(tokens, spatial_edges, mask)
            return M_rh + torch.tanh(self.hh_gate) * hh
        key_padding_mask = None
        all_masked = None
        safe_mask = mask
        if mask is not None:
            safe_mask = mask.clone()
            all_masked = ~safe_mask.any(dim=1)
            if bool(all_masked.any()):
                safe_mask[all_masked, 0] = True
            key_padding_mask = ~safe_mask
        hh, _ = self.hh_attn(
            tokens, tokens, tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if mask is not None:
            hh = hh.masked_fill((~mask).unsqueeze(-1), 0.0)
            if bool(all_masked.any()):
                hh = torch.where(all_masked.view(-1, 1, 1), torch.zeros_like(hh), hh)
        return M_rh + torch.tanh(self.hh_gate) * hh

    def _multihead_attention(self, M_rh, m_rr, mask=None):
        """Canonical multi-head cross-attention: robot m_rr is the single query
        token, humans M_rh are the key/value tokens. Returns (a_attn [B,256],
        alpha [B, heads, 1, H]); each head has d_head = 256 // heads dims. mask
        ([B,H] bool, True = visible) zeroes hidden humans' attention weights."""
        B, H, _ = M_rh.shape
        nh = self.attn_heads
        dh = 256 // nh
        Q = self.W_q(m_rr).view(B, nh, 1, dh)                        # [B, nh, 1, dh]
        K = self.W_k(M_rh).view(B, H, nh, dh).permute(0, 2, 1, 3)    # [B, nh, H, dh]
        V = self.W_v(M_rh).view(B, H, nh, dh).permute(0, 2, 1, 3)    # [B, nh, H, dh]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dh)  # [B, nh, 1, H]
        if self.attn_count_scaling:
            scores = scores * H   # paper Eq 13 n-factor; matches single-head num_humans
        if mask is not None:
            scores = scores.masked_fill((~mask).view(B, 1, 1, H), float('-inf'))
        alpha = F.softmax(scores, dim=-1)
        if mask is not None:
            alpha = torch.nan_to_num(alpha, nan=0.0)                 # all-hidden rows -> 0
        ctx = torch.matmul(alpha, V).reshape(B, 256)                 # [B, 256]
        return self.W_o(ctx), alpha

    def _attention_pool(self, M_rh, m_rr, num_humans, mask=None):
        """Attention-weighted pooling of per-human spatial features M_rh against
        the robot/temporal key m_rr. attn_heads>1 uses multi-head cross-attention;
        otherwise the legacy single-head weighted average. attn_count_scaling
        scales attention scores by n (paper Eq 13) in BOTH paths. mask ([B,N] bool, True
        = visible) restricts pooling to humans within the sensing radius: hidden
        humans get zero attention weight and are excluded from the max; rows with no
        visible human return a zero crowd vector."""
        none_visible = (~mask).all(dim=1) if mask is not None else None
        if self.attn_heads > 1:
            a_attn, _ = self._multihead_attention(M_rh, m_rr, mask)
            if not self.meanmax_pool:
                return a_attn
            a_max = (self._masked_max(M_rh, mask, none_visible) if mask is not None
                     else M_rh.max(dim=1).values)
            return self.pool_merge(torch.cat([a_attn, a_max], dim=1))
        Q = self.W_q(M_rh)                      # [B, H, 64]
        K = self.W_k(m_rr).unsqueeze(1)         # [B, 1, 64]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / 8.0   # /sqrt(d_k)
        if self.attn_count_scaling:
            attn_scores = attn_scores * num_humans
        if mask is not None:
            attn_scores = attn_scores.masked_fill((~mask).unsqueeze(-1), float('-inf'))
        alpha = F.softmax(attn_scores, dim=1)   # [B, H, 1]
        if mask is not None:
            alpha = torch.nan_to_num(alpha, nan=0.0)   # all-hidden rows -> 0 weights
        a_mean = torch.bmm(M_rh.transpose(1, 2), alpha).squeeze(2)  # [B, 256]
        if not self.meanmax_pool:
            return a_mean
        a_max = (self._masked_max(M_rh, mask, none_visible) if mask is not None
                 else M_rh.max(dim=1).values)          # [B, 256] cardinality-robust
        return self.pool_merge(torch.cat([a_mean, a_max], dim=1))  # [B, 256]

    def _scale_action(self, x):
        """Map x in [0,1]^2 to the physical action box [0,vpref]x[-wmax,wmax]."""
        return self.action_low + (self.action_high - self.action_low) * x

    def _unscale_action(self, a):
        """Inverse of _scale_action; clamp to (eps,1-eps) for Beta.log_prob safety."""
        x = (a - self.action_low) / (self.action_high - self.action_low)
        return x.clamp(1e-6, 1.0 - 1e-6)

    def make_action_dist(self, p1, p2):
        """Single source of truth for the action distribution: Beta(alpha,beta)
        when action_dist=='beta', else Normal(mu,std). Used by BOTH the single-env
        and the vectorized PPO paths so the Beta branch can never be silently
        skipped (the bug this method fixes)."""
        if self.action_dist == 'beta':
            return torch.distributions.Beta(p1, p2)
        return torch.distributions.Normal(p1, p2)

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
            hidden_states: dict with 'temporal_edge', 'spatial_edge', 'node' NCP states
        Returns:
            mu: actor mean [batch_size, 2] — scaled to (v in [0, vpref], w in [-wmax, wmax])
            std: actor std [batch_size, 2] — broadcast from per-dim logstd parameter
            value: critic value [batch_size, 1]
            new_hidden_states: updated NCP hidden states (LTC or CfC)
        """
        robot_node = obs['robot_node']        # [batch_size, 7]
        spatial_edges = obs['spatial_edges']  # [batch_size, num_humans, 4]
        temporal_edges = obs['temporal_edges']# [batch_size, 2]
        
        batch_size = robot_node.shape[0]
        num_humans = spatial_edges.shape[1]
        
        # 1. Robot Node Encoding
        v_m = self.robot_mlp(robot_node)  # [batch_size, 128]
        
        # 2. Temporal Edge Encoding (LTC or CfC). With pre_mlp, the paper's Eq 11
        # ordering: expand to the 256-dim encoding first, then the NCP.
        temporal_features = self.temporal_pre_mlp(temporal_edges) if self.pre_mlp else temporal_edges
        temporal_input = temporal_features.unsqueeze(1)
        h_temp = hidden_states['temporal_edge']
        m_rr_seq, h_temp_new = self._ncp('temporal')(temporal_input, h_temp)
        m_rr = self.temporal_proj(m_rr_seq.squeeze(1))

        # 3. Spatial Edge Encoding (LTC or CfC)
        spatial_flat = spatial_edges.reshape(batch_size * num_humans, 6)
        if self.pre_mlp:
            spatial_flat = self.spatial_pre_mlp(spatial_flat)
        spatial_input = spatial_flat.unsqueeze(1)
        h_spat = hidden_states['spatial_edge']
        if h_spat.dim() == 3:
            h_spat_flat = h_spat.reshape(batch_size * num_humans, -1)
        else:
            h_spat_flat = h_spat
            
        M_rh_seq, h_spat_new_flat = self._ncp('spatial')(spatial_input, h_spat_flat)
        M_rh_proj = self.spatial_proj(M_rh_seq.squeeze(1))
        M_rh = M_rh_proj.reshape(batch_size, num_humans, 256)

        # 4. Attention Pooling — optionally mask humans beyond the sensing radius
        # (paper's limited perception: only nearby humans influence the action).
        sense_mask = None
        if self.sense_range > 0:
            dist = torch.hypot(spatial_edges[:, :, 0], spatial_edges[:, :, 1])  # [B, N] m
            sense_mask = dist <= self.sense_range                                # [B, N] bool
        if self.hh_intent_graph:
            M_rh = self._human_intent_graph(M_rh, spatial_edges, sense_mask)
        u_att = self._attention_pool(M_rh, m_rr, num_humans, sense_mask)
        
        # 5. Node NCP Encoder (LTC or CfC)
        node_input = torch.cat([v_m, m_rr, u_att], dim=-1).unsqueeze(1)
        h_node = hidden_states['node']
        sf_seq, h_node_new = self._ncp('node')(node_input, h_node)
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

        # v39 risk head (optional): p_coll in (0,1), min_clearance >= 0.
        # Stored on the module so the 4-tuple actor/critic contract is unchanged
        # and old callers (eval, waffle_ros, PPO unroll) keep unpacking four values.
        if self.risk_head:
            risk_raw = self.risk_mlp(sf)
            self.last_p_coll = torch.sigmoid(risk_raw[:, 0:1])
            self.last_min_clearance = F.softplus(risk_raw[:, 1:2])
            self.last_cost_value = self.cost_critic(sf)
        else:
            self.last_p_coll = None
            self.last_min_clearance = None
            self.last_cost_value = None
        
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


def _normalize_cell_type(cell_type):
    name = str(cell_type).lower()
    if name not in VALID_CELL_TYPES:
        raise ValueError(
            f"cell_type must be one of {VALID_CELL_TYPES}, got {cell_type!r}"
        )
    return name


def detect_cell_type(state_dict):
    """Return 'ltc' or 'cfc' from a policy state dict.

    Default / historical checkpoints have `*_ltc.*` keys and no CfC marker.
    CfC checkpoints have `*_cfc.*` keys and `_cell_type_cfc`. Mixing both is
    an error rather than a silent guess.
    """
    keys = list(state_dict)
    has_cfc = any(key.startswith(_CFC_MODULE_PREFIXES) for key in keys)
    has_ltc = any(key.startswith(_LTC_MODULE_PREFIXES) for key in keys)
    marked_cfc = '_cell_type_cfc' in state_dict
    if has_cfc and has_ltc:
        raise CellTypeMismatchError(
            "checkpoint contains both LTC and CfC encoder keys; refusing to guess"
        )
    if marked_cfc and has_ltc:
        raise CellTypeMismatchError(
            "checkpoint is marked CfC but contains LTC encoder keys"
        )
    if has_cfc or marked_cfc:
        return 'cfc'
    return 'ltc'


def assert_cell_type_compatible(policy, state_dict):
    """Raise if `state_dict` belongs to the other NCP cell family."""
    detected = detect_cell_type(state_dict)
    requested = getattr(policy, 'cell_type', 'ltc')
    if detected != requested:
        pretty = {'ltc': 'LTC', 'cfc': 'CfC'}
        raise CellTypeMismatchError(
            f"Cannot load a {pretty[detected]} checkpoint into a "
            f"{pretty[requested]} SNCPPolicy. Reconstruct with "
            f"build_policy_for_checkpoint(...) or SNCPPolicy(cell_type="
            f"'{detected}') / --temporal_cell {detected}."
        )
    return detected


def detect_sparse_hig(state_dict):
    """Return True if `state_dict` is a SparseHIG checkpoint.

    Marker is `_hh_sparse_k` and/or `hh_sparse_attn.*`. Mixing those with dense
    v37 `hh_attn.*` is a hard error — never a silent guess.
    """
    keys = list(state_dict)
    has_sparse = (
        '_hh_sparse_k' in state_dict
        or any(key.startswith(_SPARSE_HH_PREFIXES) for key in keys)
    )
    has_dense = any(key.startswith(_DENSE_HH_PREFIXES) for key in keys)
    if has_sparse and has_dense:
        raise SparseHIGMismatchError(
            "checkpoint contains both dense HH (hh_attn.*) and SparseHIG "
            "(hh_sparse_attn.* / _hh_sparse_k) keys; refusing to guess"
        )
    return has_sparse


def detect_hh_topk(state_dict):
    """SparseHIG k from `_hh_sparse_k`, or 0 when the branch is off."""
    if detect_sparse_hig(state_dict):
        buf = state_dict.get('_hh_sparse_k')
        return int(buf.item()) if buf is not None else 3
    return 0


def assert_sparse_hig_compatible(policy, state_dict):
    """Raise if `state_dict` is the other HH graph family (dense vs SparseHIG)."""
    detected = detect_sparse_hig(state_dict)
    requested = bool(getattr(policy, 'sparse_hig', False))
    if detected != requested:
        have = "SparseHIG" if detected else "dense HH"
        want = "SparseHIG" if requested else "dense HH"
        raise SparseHIGMismatchError(
            f"Cannot load a {have} checkpoint into a {want} SNCPPolicy. "
            "Reconstruct with build_policy_for_checkpoint(...) or pass "
            "--sparse_hig / omit it to match the file."
        )
    if detected:
        ckpt_k = detect_hh_topk(state_dict)
        pol_k = int(getattr(policy, 'hh_topk', 0))
        if ckpt_k != pol_k:
            raise SparseHIGMismatchError(
                f"SparseHIG k mismatch: checkpoint k={ckpt_k} vs policy k={pol_k}"
            )
    return detected


def load_policy_state_dict(policy, state_dict, strict=True):
    """load_state_dict with explicit LTC/CfC and dense-HH/SparseHIG checks."""
    assert_cell_type_compatible(policy, state_dict)
    assert_sparse_hig_compatible(policy, state_dict)
    return policy.load_state_dict(state_dict, strict=strict)


def _infer_cfc_node_sizes(state_dict):
    units_buf = state_dict.get('_cfc_node_units')
    output_buf = state_dict.get('_cfc_node_output')
    if units_buf is not None and output_buf is not None:
        return int(units_buf.item()), int(output_buf.item())
    node_output = int(state_dict['node_proj.weight'].shape[1]) if 'node_proj.weight' in state_dict else 48
    units = 0
    layer = 0
    while f'node_cfc.rnn_cell.layer_{layer}.ff1.bias' in state_dict:
        units += int(state_dict[f'node_cfc.rnn_cell.layer_{layer}.ff1.bias'].shape[0])
        layer += 1
    return (units or 128), node_output


def checkpoint_has_risk_head(state_dict):
    return '_risk_head' in state_dict or any(
        key.startswith('risk_mlp.') for key in state_dict
    )


def _is_risk_head_key(key):
    return (
        key == '_risk_head'
        or key.startswith('risk_mlp.')
        or key.startswith('cost_critic.')
    )


def build_policy_for_checkpoint(state_dict, robot_vpref=0.26, robot_wmax=1.8,
                                risk_head=None):
    """Construct an SNCPPolicy whose architecture matches a saved state dict.

    Checkpoints are plain `policy.state_dict()` files; architecture variants
    (pre-MLP, pooling, cell_type, …) are detected from keys. Old checkpoints
    (v14..v21) have no `*_pre_mlp` keys and get the legacy LTC layout.
    Pass risk_head=True to attach a fresh v39 head on top of a pre-v39
    checkpoint (training-only); eval leaves this None so auto-detect preserves
    old weights byte-for-byte.
    """
    cell_type = detect_cell_type(state_dict)
    pre_mlp = any(key.startswith('temporal_pre_mlp') for key in state_dict)
    attn_count_scaling = '_attn_count_scaling' in state_dict
    meanmax_pool = any(key.startswith('pool_merge') for key in state_dict)
    if cell_type == 'cfc':
        node_units, node_output = _infer_cfc_node_sizes(state_dict)
    else:
        gleak = state_dict.get('node_ltc.rnn_cell.gleak')
        node_units = int(gleak.shape[0]) if gleak is not None else 128
        out_w = state_dict.get('node_ltc.rnn_cell.output_w')
        node_output = int(out_w.shape[0]) if out_w is not None else 48
    ah = state_dict.get('_attn_heads')
    attn_heads = int(ah.item()) if ah is not None else 1
    action_dist = 'gaussian' if 'actor_logstd' in state_dict else 'beta'
    sr = state_dict.get('_sense_range')
    sense_range = float(sr.item()) if sr is not None else 0.0
    hh_intent_graph = '_hh_intent_graph' in state_dict
    hh = state_dict.get('_hh_attn_heads')
    hh_attn_heads = int(hh.item()) if hh is not None else 4
    cvh = state_dict.get('_cv_horizons')
    cv_horizons = tuple(float(value) for value in cvh.tolist()) if cvh is not None else (1, 2, 3, 4)
    cvdt = state_dict.get('_cv_dt')
    cv_dt = float(cvdt.item()) if cvdt is not None else 0.25
    sparse_hig = detect_sparse_hig(state_dict)
    hh_topk = detect_hh_topk(state_dict) if sparse_hig else 3
    detected_risk = checkpoint_has_risk_head(state_dict)
    if risk_head is None:
        risk_head = detected_risk
    return SNCPPolicy(robot_vpref=robot_vpref, robot_wmax=robot_wmax,
                      pre_mlp=pre_mlp, attn_count_scaling=attn_count_scaling,
                      meanmax_pool=meanmax_pool, node_units=node_units,
                      node_output=node_output, attn_heads=attn_heads,
                      action_dist=action_dist, sense_range=sense_range,
                      hh_intent_graph=hh_intent_graph or sparse_hig,
                      hh_attn_heads=hh_attn_heads,
                      cv_horizons=cv_horizons, cv_dt=cv_dt, risk_head=risk_head,
                      cell_type=cell_type, sparse_hig=sparse_hig, hh_topk=hh_topk)
