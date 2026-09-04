import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sncp_ppo.vec_buffer import compute_gae_vectorized


class RunningMeanStd:
    """Welford's online mean/variance estimator. Used to normalize the GAE
    return targets so the critic MSE loss isn't dominated by reward outliers
    (e.g. the +50 goal terminal vs the +5·Δd dense shaping vs the -25
    collision). PPO best practice — see Engstrom et al. 2020
    "Implementation Matters in Deep Policy Gradients"."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).flatten()
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = float(x.size)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self):
        return float(max(np.sqrt(self.var), 1e-6))


class PPOMemory:
    """
    Stores transitions from multiple episodes, tracking episode boundaries
    so that sub-sequence BPTT can be applied during PPO updates.

    Per-episode bootstrap values are recorded so GAE can compute
    V(s_final) for truncated episodes without bleeding across boundaries.
    """
    def __init__(self):
        # Per-step data stored as flat lists
        self.obs_robot_node = []
        self.obs_spatial_edges = []
        self.obs_temporal_edges = []

        self.h_temporal_edge = []
        self.h_spatial_edge = []
        self.h_node = []

        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.coll_labels = []
        self.clearance_labels = []
        self.cost_values = []

        # Episode boundary tracking
        self.episode_lengths = []
        self.episode_bootstrap_values = []  # V(s_final): nonzero only for truncated episodes
        self.episode_bootstrap_costs = []
        self._current_ep_len = 0

    def store(self, obs, h_states, action, log_prob, reward, value, mask,
              coll_label=0.0, clearance_label=0.0, cost_value=0.0):
        self.obs_robot_node.append(obs['robot_node'])
        self.obs_spatial_edges.append(obs['spatial_edges'])
        self.obs_temporal_edges.append(obs['temporal_edges'])

        # Clone and detach hidden states to save them in memory
        self.h_temporal_edge.append(h_states['temporal_edge'].clone().detach())
        self.h_spatial_edge.append(h_states['spatial_edge'].clone().detach())
        self.h_node.append(h_states['node'].clone().detach())

        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.masks.append(mask)
        self.coll_labels.append(float(coll_label))
        self.clearance_labels.append(float(clearance_label))
        self.cost_values.append(float(cost_value))
        self._current_ep_len += 1

    def end_episode(self, bootstrap_value=0.0, bootstrap_cost=0.0):
        """Record the episode boundary.

        bootstrap_value: V(s_final) for truncated episodes (timeout); 0.0
        for terminated episodes (collision/goal-reached). Without this,
        GAE assumes the world ends at every episode boundary, which biases
        the value function for truncated rollouts.
        """
        if self._current_ep_len > 0:
            self.episode_lengths.append(self._current_ep_len)
            self.episode_bootstrap_values.append(float(bootstrap_value))
            self.episode_bootstrap_costs.append(float(bootstrap_cost))
            self._current_ep_len = 0

    def clear(self):
        self.obs_robot_node.clear()
        self.obs_spatial_edges.clear()
        self.obs_temporal_edges.clear()

        self.h_temporal_edge.clear()
        self.h_spatial_edge.clear()
        self.h_node.clear()

        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
        self.coll_labels.clear()
        self.clearance_labels.clear()
        self.cost_values.clear()
        self.episode_lengths.clear()
        self.episode_bootstrap_values.clear()
        self.episode_bootstrap_costs.clear()
        self._current_ep_len = 0

    def get_tensors(self, device):
        # Stack observations
        robot_nodes = torch.tensor(np.array(self.obs_robot_node), dtype=torch.float32, device=device)
        spatial_edges = torch.tensor(np.array(self.obs_spatial_edges), dtype=torch.float32, device=device)
        temporal_edges = torch.tensor(np.array(self.obs_temporal_edges), dtype=torch.float32, device=device)
        obs = {
            'robot_node': robot_nodes,
            'spatial_edges': spatial_edges,
            'temporal_edges': temporal_edges
        }
        
        # Stack hidden states
        h_states = {
            'temporal_edge': torch.cat(self.h_temporal_edge, dim=0).to(device),
            'spatial_edge': torch.stack(self.h_spatial_edge).to(device),
            'node': torch.cat(self.h_node, dim=0).to(device)
        }
        
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        values = torch.tensor(np.array(self.values), dtype=torch.float32, device=device)
        masks = torch.tensor(np.array(self.masks), dtype=torch.float32, device=device)
        coll_labels = torch.tensor(np.array(self.coll_labels), dtype=torch.float32, device=device)
        clearance_labels = torch.tensor(np.array(self.clearance_labels), dtype=torch.float32, device=device)
        cost_values = torch.tensor(np.array(self.cost_values), dtype=torch.float32, device=device)
        
        return (obs, h_states, actions, log_probs, rewards, values, masks,
                coll_labels, clearance_labels, cost_values)

class PPOAgent:
    def __init__(self, policy, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 c1=0.5, c2=0.01, epochs=4, batch_size=64, seq_len=16,
                 total_updates=None, lr_end_factor=0.1, target_kl=0.015,
                 normalize_returns=True, risk_bce_coef=1.0, risk_clearance_coef=0.1,
                 use_lagrange=False, lagrange_cost_limit=0.05, lagrange_lr=0.01,
                 lagrange_lambda_init=0.0, lagrange_lambda_max=10.0):
        """
        total_updates: if given, attaches a LinearLR scheduler that decays the
            learning rate from `lr` to `lr * lr_end_factor` linearly across
            `total_updates` calls to update(). Smooths value-function transitions
            across curriculum shifts.
        target_kl: approximate KL above which a PPO update epoch early-stops.
            Set to None to disable. 0.015 is the CleanRL / SB3 default.
        normalize_returns: if True, scale GAE returns by a running std before
            computing the value loss. Stabilizes the critic when reward magnitudes
            change across curriculum phases (e.g. +50 goal vs -25 collision).
        """
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.target_kl = target_kl
        self.normalize_returns = normalize_returns
        self.return_rms = RunningMeanStd()
        self.memory = PPOMemory()
        # Last-update diagnostics for logging from train.py
        self.last_entropy = 0.0
        self.last_approx_kl = 0.0
        self.last_clip_frac = 0.0
        self.last_epochs_ran = 0
        self.risk_bce_coef = float(risk_bce_coef)
        self.risk_clearance_coef = float(risk_clearance_coef)
        self.use_lagrange = bool(use_lagrange)
        self.lagrange_cost_limit = float(lagrange_cost_limit)
        self.lagrange_lr = float(lagrange_lr)
        self.lagrange_lambda = float(lagrange_lambda_init)
        self.lagrange_lambda_max = float(lagrange_lambda_max)
        self.last_risk_bce = 0.0
        self.last_risk_huber = 0.0
        self.last_mean_cost = 0.0

        if total_updates is not None and total_updates > 0:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=lr_end_factor,
                total_iters=total_updates,
            )
        else:
            self.scheduler = None

    def select_action(self, obs, h_states, device, deterministic=False):
        """Sample an action from the policy.

        Returns the *un-clipped* sample plus its log-prob under Normal(mu, std).
        The caller is responsible for clipping the action to the env action_space
        before stepping. Storing the un-clipped sample preserves the PPO ratio
        identity exp(new_logp - old_logp) — otherwise log_prob would refer to
        a different (clipped) distribution and the importance weight is biased.

        In deterministic mode action = mu, which is already in-bounds due to
        sigmoid/tanh in the policy head, so no clipping is needed downstream.
        """
        obs_tensor = {
            'robot_node': torch.tensor(obs['robot_node'], dtype=torch.float32, device=device).unsqueeze(0),
            'spatial_edges': torch.tensor(obs['spatial_edges'], dtype=torch.float32, device=device).unsqueeze(0),
            'temporal_edges': torch.tensor(obs['temporal_edges'], dtype=torch.float32, device=device).unsqueeze(0)
        }

        with torch.no_grad():
            out1, out2, value, h_states_new = self.policy(obs_tensor, h_states)

        if self.policy.action_dist == 'beta':
            if deterministic:
                action = self.policy.deterministic_action(out1, out2)
                log_prob_value = 0.0
            else:
                dist = torch.distributions.Beta(out1, out2)
                x = dist.sample()
                action = self.policy._scale_action(x)        # store physical, in-bounds
                log_prob_value = dist.log_prob(x).sum(-1).item()
        else:
            mu, std = out1, out2
            if deterministic:
                action = mu
                log_prob_value = 0.0
            else:
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                log_prob_value = dist.log_prob(action).sum(-1).item()

        action_np = action.cpu().numpy()[0]

        return action_np, log_prob_value, value.item(), h_states_new

    @staticmethod
    def clip_action_for_env(action, vpref, wmax):
        """Clip an un-clipped policy sample to the env's action_space bounds.

        Kept separate from select_action so that the action stored in memory
        is the same one whose log-prob was recorded.
        """
        return np.array([
            np.clip(action[0], 0.0, vpref),
            np.clip(action[1], -wmax, wmax),
        ], dtype=np.float32)

    def _risk_enabled(self):
        return bool(getattr(self.policy, 'risk_head', False))

    def _normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _clipped_value_loss(self, pred, old, target, valid):
        clipped = old + torch.clamp(pred - old, -self.clip_eps, self.clip_eps)
        unclipped = (pred - target).pow(2)
        clipped_l = (clipped - target).pow(2)
        return (torch.max(unclipped, clipped_l) * valid).sum() / valid.sum()

    def _cost_gae(self, costs, cost_values, masks_or_dones, bootstrap, *, vectorized=False):
        if not self._risk_enabled():
            zeros = torch.zeros_like(costs)
            return zeros, zeros
        if vectorized:
            cost_adv, cost_ret = compute_gae_vectorized(
                costs, cost_values, masks_or_dones, bootstrap,
                self.gamma, self.gae_lambda,
            )
        else:
            cost_adv, cost_ret = self.compute_gae(
                costs, cost_values, masks_or_dones,
                self.memory.episode_lengths, bootstrap,
            )
        return cost_adv, cost_ret

    def _risk_aux_loss(self, p_coll_steps, clearance_steps, coll_labels, clearance_labels, valid):
        if not self._risk_enabled() or not p_coll_steps:
            return 0.0, 0.0, 0.0
        from sncp_ppo.risk_losses import risk_supervision_loss
        p_coll = torch.stack(p_coll_steps, dim=1).squeeze(-1)
        clearance = torch.stack(clearance_steps, dim=1).squeeze(-1)
        bce, huber, _ = risk_supervision_loss(
            p_coll, clearance, coll_labels, clearance_labels, valid,
        )
        loss = self.risk_bce_coef * bce + self.risk_clearance_coef * huber
        return loss, float(bce.detach()), float(huber.detach())

    def _step_lagrange_dual(self, mean_cost):
        from sncp_ppo.risk_losses import dual_ascent_update
        self.last_mean_cost = float(mean_cost)
        if not self.use_lagrange:
            return
        self.lagrange_lambda = dual_ascent_update(
            self.lagrange_lambda, mean_cost, self.lagrange_cost_limit,
            self.lagrange_lr, self.lagrange_lambda_max,
        )

    def compute_gae(self, rewards, values, masks, episode_lengths, bootstrap_values):
        """Episode-aware GAE.

        Previously this function used `values[t+1]` for the next-state value
        regardless of episode boundaries, which bled the first state of the
        next episode into the previous one's advantage. It also hard-coded
        `next_value=0` for the buffer's last step, biasing truncated rollouts.

        Now each episode is processed independently using its own bootstrap
        value (0 for terminated, V(s_final) for truncated).
        """
        advantages = torch.zeros_like(rewards)
        offset = 0
        for ep_idx, ep_len in enumerate(episode_lengths):
            ep_end = offset + ep_len
            ep_bootstrap = bootstrap_values[ep_idx]

            gae = 0.0
            for t in reversed(range(offset, ep_end)):
                if t == ep_end - 1:
                    nv = ep_bootstrap
                else:
                    nv = values[t + 1].item()
                delta = rewards[t].item() + self.gamma * nv * masks[t].item() - values[t].item()
                gae = delta + self.gamma * self.gae_lambda * masks[t].item() * gae
                advantages[t] = gae

            offset = ep_end

        returns = advantages + values
        return advantages, returns

    def _extract_subsequences(self, obs, h_states, actions, old_log_probs, advantages, returns, values, device,
                              coll_labels=None, clearance_labels=None, cost_advantages=None,
                              cost_returns=None, old_cost_values=None):
        """
        Extract contiguous subsequences from episodes for BPTT.
        Each subsequence uses the hidden state from the START of the subsequence
        and rolls forward through time, allowing gradient flow through the LTC cells.

        `values` are the per-step value estimates from the *behavior* policy
        (recorded at rollout); they are needed by the clipped value loss in update()
        to bound how far the critic can move per PPO step.
        """
        seq_obs_rn = []      # [num_seqs, seq_len, robot_node_dim]
        seq_obs_se = []      # [num_seqs, seq_len, num_humans, 2]
        seq_obs_te = []      # [num_seqs, seq_len, 2]
        seq_h_temp = []      # [num_seqs, h_dim]
        seq_h_spat = []      # [num_seqs, num_humans*h_dim] or flat
        seq_h_node = []      # [num_seqs, h_dim]
        seq_actions = []     # [num_seqs, seq_len, 2]
        seq_old_lp = []      # [num_seqs, seq_len]
        seq_advantages = []  # [num_seqs, seq_len]
        seq_returns = []     # [num_seqs, seq_len]
        seq_old_values = []  # [num_seqs, seq_len]
        seq_coll = []
        seq_clearance = []
        seq_cost_adv = []
        seq_cost_ret = []
        seq_old_cost = []
        
        offset = 0
        for ep_len in self.memory.episode_lengths:
            # Slice this episode's data
            ep_end = offset + ep_len
            
            # Create subsequences within this episode
            for start in range(offset, ep_end, self.seq_len):
                end = min(start + self.seq_len, ep_end)
                actual_len = end - start
                
                if actual_len < 4:  # Skip very short fragments
                    continue
                
                # Pad to seq_len if needed
                rn = obs['robot_node'][start:end]
                se = obs['spatial_edges'][start:end]
                te = obs['temporal_edges'][start:end]
                act = actions[start:end]
                olp = old_log_probs[start:end]
                adv = advantages[start:end]
                ret = returns[start:end]
                ov = values[start:end]
                coll = coll_labels[start:end] if coll_labels is not None else torch.zeros(actual_len, device=device)
                clr = clearance_labels[start:end] if clearance_labels is not None else torch.zeros(actual_len, device=device)
                cadv = cost_advantages[start:end] if cost_advantages is not None else torch.zeros(actual_len, device=device)
                cret = cost_returns[start:end] if cost_returns is not None else torch.zeros(actual_len, device=device)
                ocv = old_cost_values[start:end] if old_cost_values is not None else torch.zeros(actual_len, device=device)

                if actual_len < self.seq_len:
                    pad_len = self.seq_len - actual_len
                    rn = F.pad(rn, (0, 0, 0, pad_len))
                    se = F.pad(se, (0, 0, 0, 0, 0, pad_len))
                    te = F.pad(te, (0, 0, 0, pad_len))
                    act = F.pad(act, (0, 0, 0, pad_len))
                    olp = F.pad(olp, (0, pad_len))
                    adv = F.pad(adv, (0, pad_len))
                    ret = F.pad(ret, (0, pad_len))
                    ov  = F.pad(ov,  (0, pad_len))
                    coll = F.pad(coll, (0, pad_len))
                    clr = F.pad(clr, (0, pad_len))
                    cadv = F.pad(cadv, (0, pad_len))
                    cret = F.pad(cret, (0, pad_len))
                    ocv = F.pad(ocv, (0, pad_len))

                seq_obs_rn.append(rn)
                seq_obs_se.append(se)
                seq_obs_te.append(te)

                # Hidden state at the start of this subsequence
                seq_h_temp.append(h_states['temporal_edge'][start])
                seq_h_spat.append(h_states['spatial_edge'][start])
                seq_h_node.append(h_states['node'][start])

                seq_actions.append(act)
                seq_old_lp.append(olp)
                seq_advantages.append(adv)
                seq_returns.append(ret)
                seq_old_values.append(ov)
                seq_coll.append(coll)
                seq_clearance.append(clr)
                seq_cost_adv.append(cadv)
                seq_cost_ret.append(cret)
                seq_old_cost.append(ocv)
            
            offset = ep_end
        
        if len(seq_obs_rn) == 0:
            return None
        
        result = {
            'obs_rn': torch.stack(seq_obs_rn),         # [N, S, 7]
            'obs_se': torch.stack(seq_obs_se),         # [N, S, H, 2]
            'obs_te': torch.stack(seq_obs_te),         # [N, S, 2]
            'h_temp': torch.stack(seq_h_temp),         # [N, h_dim]
            'h_spat': torch.stack(seq_h_spat),         # [N, h_dim] or [N, num_humans, h_dim]
            'h_node': torch.stack(seq_h_node),         # [N, h_dim]
            'actions': torch.stack(seq_actions),       # [N, S, 2]
            'old_lp': torch.stack(seq_old_lp),         # [N, S]
            'advantages': torch.stack(seq_advantages), # [N, S]
            'returns': torch.stack(seq_returns),       # [N, S]
            'old_values': torch.stack(seq_old_values), # [N, S]
            'coll_labels': torch.stack(seq_coll),
            'clearance_labels': torch.stack(seq_clearance),
            'cost_advantages': torch.stack(seq_cost_adv),
            'cost_returns': torch.stack(seq_cost_ret),
            'old_cost_values': torch.stack(seq_old_cost),
            'seq_lengths': [],                         # actual lengths before padding
        }
        
        # Record actual lengths for masking
        offset = 0
        for ep_len in self.memory.episode_lengths:
            ep_end = offset + ep_len
            for start in range(offset, ep_end, self.seq_len):
                end = min(start + self.seq_len, ep_end)
                actual_len = end - start
                if actual_len >= 4:
                    result['seq_lengths'].append(actual_len)
            offset = ep_end
        
        return result

    def update(self, device):
        # 1. Retrieve all trajectories
        obs, h_states, actions, old_log_probs, rewards, values, masks, \
            coll_labels, clearance_labels, cost_values = self.memory.get_tensors(device)

        # 2. Compute advantages and returns with episode-aware GAE
        advantages, returns = self.compute_gae(
            rewards, values, masks,
            self.memory.episode_lengths,
            self.memory.episode_bootstrap_values,
        )

        # Return normalization: scale returns by a running std so the critic
        # MSE loss stays in a stable magnitude across curriculum phases. The
        # behavior critic targets `values` are also rescaled by the *same*
        # divisor — they were produced under the previous normalizer state
        # but the clipped value loss compares them on the same scale, so we
        # apply the current divisor to both.
        if self.normalize_returns:
            self.return_rms.update(returns.detach().cpu().numpy())
            ret_std = self.return_rms.std
            returns = returns / ret_std
            values = values / ret_std

        # Normalize advantages
        advantages = self._normalize_advantages(advantages)
        cost_adv_raw, cost_returns = self._cost_gae(
            coll_labels, cost_values, masks, self.memory.episode_bootstrap_costs,
        )
        cost_adv = (
            self._normalize_advantages(cost_adv_raw)
            if self.use_lagrange else torch.zeros_like(cost_adv_raw)
        )

        # 2. Extract subsequences for BPTT (values plumbed for clipped value loss)
        seqs = self._extract_subsequences(obs, h_states, actions, old_log_probs,
                                          advantages, returns, values, device,
                                          coll_labels=coll_labels,
                                          clearance_labels=clearance_labels,
                                          cost_advantages=cost_adv,
                                          cost_returns=cost_returns,
                                          old_cost_values=cost_values)

        if seqs is None:
            self.memory.clear()
            return

        num_seqs = seqs['obs_rn'].shape[0]
        S = self.seq_len
        num_humans = seqs['obs_se'].shape[2]

        # Per-epoch diagnostics — populated from the last batch of each epoch
        # so we can early-stop on KL and surface entropy/clip-fraction to train.
        epoch_kl = 0.0
        epoch_entropy = 0.0
        epoch_clip_frac = 0.0
        epochs_ran = 0

        # 3. PPO Update Epochs
        for epoch in range(self.epochs):
            perm = torch.randperm(num_seqs)
            batch_kls = []
            batch_entropies = []
            batch_clip_fracs = []
            
            for batch_start in range(0, num_seqs, self.batch_size):
                batch_idx = perm[batch_start:batch_start + self.batch_size]
                B = len(batch_idx)
                
                b_rn = seqs['obs_rn'][batch_idx]      # [B, S, 7]
                b_se = seqs['obs_se'][batch_idx]      # [B, S, H, 2]
                b_te = seqs['obs_te'][batch_idx]      # [B, S, 2]
                b_h_temp = seqs['h_temp'][batch_idx]  # [B, h_dim]
                b_h_spat = seqs['h_spat'][batch_idx]  # [B, h_dim] or [B, H, h_dim]
                b_h_node = seqs['h_node'][batch_idx]  # [B, h_dim]
                b_actions = seqs['actions'][batch_idx]    # [B, S, 2]
                b_old_lp = seqs['old_lp'][batch_idx]      # [B, S]
                b_adv = seqs['advantages'][batch_idx]     # [B, S]
                b_ret = seqs['returns'][batch_idx]        # [B, S]
                b_old_v = seqs['old_values'][batch_idx]   # [B, S] (behavior critic)
                b_coll = seqs['coll_labels'][batch_idx]
                b_clr = seqs['clearance_labels'][batch_idx]
                if self.use_lagrange:
                    b_adv = b_adv - self.lagrange_lambda * seqs['cost_advantages'][batch_idx]
                
                # Get actual lengths for this batch
                b_lengths = [seqs['seq_lengths'][idx.item()] for idx in batch_idx]
                
                # Create valid mask [B, S]
                b_len_t = torch.tensor(b_lengths, device=device)
                valid_mask = (torch.arange(S, device=device)[None, :] < b_len_t[:, None]).float()
                
                # Unroll through the sequence with BPTT
                all_p1 = []
                all_p2 = []
                all_values = []
                all_pcoll = []
                all_clr_pred = []
                all_cost_v = []
                
                h_temp = b_h_temp.clone()
                h_node = b_h_node.clone()
                
                # Handle spatial hidden state shape
                if b_h_spat.dim() == 3:
                    # [B, num_humans, h_dim] -> [B*num_humans, h_dim]
                    h_spat = b_h_spat.reshape(B * num_humans, -1)
                else:
                    h_spat = b_h_spat.clone()
                
                for t in range(S):
                    step_obs = {
                        'robot_node': b_rn[:, t],       # [B, 7]
                        'spatial_edges': b_se[:, t],    # [B, H, 2]
                        'temporal_edges': b_te[:, t]    # [B, 2]
                    }
                    step_h = {
                        'temporal_edge': h_temp,
                        'spatial_edge': h_spat,
                        'node': h_node
                    }
                    
                    out1, out2, value, new_h = self.policy(step_obs, step_h)
                    all_p1.append(out1)
                    all_p2.append(out2)
                    all_values.append(value)
                    if self._risk_enabled():
                        all_pcoll.append(self.policy.last_p_coll)
                        all_clr_pred.append(self.policy.last_min_clearance)
                        all_cost_v.append(self.policy.last_cost_value)
                    
                    # Update hidden states for next step (BPTT: keep graph!)
                    h_temp = new_h['temporal_edge']
                    h_node = new_h['node']
                    h_spat_out = new_h['spatial_edge']
                    if h_spat_out.dim() == 3:
                        h_spat = h_spat_out.reshape(B * num_humans, -1)
                    else:
                        h_spat = h_spat_out
                
                # Stack: [B, S, ...]
                all_p1 = torch.stack(all_p1, dim=1)      # [B, S, 2]  mu or alpha
                all_p2 = torch.stack(all_p2, dim=1)      # [B, S, 2]  std or beta
                all_values = torch.stack(all_values, dim=1).squeeze(-1)  # [B, S]

                # Compute log probs and entropy under the policy's distribution.
                # Beta: log_prob on the [0,1] pre-image of the stored physical action;
                # the affine _scale Jacobian is constant so it cancels in the ratio.
                if self.policy.action_dist == 'beta':
                    x = self.policy._unscale_action(b_actions)
                    dist = torch.distributions.Beta(all_p1, all_p2)
                    new_log_probs = dist.log_prob(x).sum(-1)        # [B, S]
                    entropy = dist.entropy().sum(-1)                 # [B, S]
                else:
                    dist = torch.distributions.Normal(all_p1, all_p2)
                    new_log_probs = dist.log_prob(b_actions).sum(-1)  # [B, S]
                    entropy = dist.entropy().sum(-1)                  # [B, S]
                
                # Apply valid mask
                ratio = torch.exp(new_log_probs - b_old_lp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv

                # Masked losses
                actor_loss = -(torch.min(surr1, surr2) * valid_mask).sum() / valid_mask.sum()

                # Diagnostics — approximate KL (Schulman 2020 "Approximating KL
                # Divergence"), entropy, and the fraction of ratios that hit the
                # clip. Computed under no_grad to avoid memory bloat.
                with torch.no_grad():
                    log_ratio = new_log_probs - b_old_lp
                    approx_kl = (((torch.exp(log_ratio) - 1) - log_ratio) * valid_mask).sum() / valid_mask.sum()
                    ent_mean = (entropy * valid_mask).sum() / valid_mask.sum()
                    clip_frac = (((ratio - 1.0).abs() > self.clip_eps).float() * valid_mask).sum() / valid_mask.sum()
                    batch_kls.append(approx_kl.item())
                    batch_entropies.append(ent_mean.item())
                    batch_clip_fracs.append(clip_frac.item())

                # Clipped value loss (OpenAI / CleanRL standard): prevents the
                # critic from moving more than clip_eps per step away from the
                # behavior critic. Bounds value-target shock during curriculum
                # shifts and stops MSE from exploding on outlier returns.
                critic_loss = self._clipped_value_loss(all_values, b_old_v, b_ret, valid_mask)

                entropy_loss = -(entropy * valid_mask).sum() / valid_mask.sum()
                
                loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
                if self._risk_enabled() and all_cost_v:
                    pred_cv = torch.stack(all_cost_v, dim=1).squeeze(-1)
                    loss = loss + self.c1 * self._clipped_value_loss(
                        pred_cv, seqs['old_cost_values'][batch_idx],
                        seqs['cost_returns'][batch_idx], valid_mask,
                    )
                risk_loss, bce_val, huber_val = self._risk_aux_loss(
                    all_pcoll, all_clr_pred, b_coll, b_clr, valid_mask,
                )
                if torch.is_tensor(risk_loss):
                    loss = loss + risk_loss
                    self.last_risk_bce = bce_val
                    self.last_risk_huber = huber_val
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

            epochs_ran = epoch + 1
            if batch_kls:
                epoch_kl = float(np.mean(batch_kls))
                epoch_entropy = float(np.mean(batch_entropies))
                epoch_clip_frac = float(np.mean(batch_clip_fracs))
            # KL early stopping — prevent the policy from drifting too far in
            # a single update. Threshold 1.5 × target_kl matches the OpenAI
            # spinning-up / CleanRL convention.
            if self.target_kl is not None and epoch_kl > 1.5 * self.target_kl:
                break

        # Persist diagnostics for the training loop to log.
        self.last_entropy = epoch_entropy
        self.last_approx_kl = epoch_kl
        self.last_clip_frac = epoch_clip_frac
        self.last_epochs_ran = epochs_ran
        self._step_lagrange_dual(float(coll_labels.mean().item()) if coll_labels.numel() else 0.0)

        # Decay learning rate per update if a scheduler is attached
        if self.scheduler is not None:
            self.scheduler.step()

        # Clear memory buffer after updating
        self.memory.clear()


    def update_vectorized(self, buffer, device):
        """PPO update from a VectorizedRolloutBuffer (N envs x T steps).

        Reuses the same surrogate/value/KL/RMS machinery as update(), but the
        advantages come from the done-masked vectorized GAE and BPTT windows are
        cut so they never span an env or an episode boundary.
        """
        data = buffer.get_tensors(device)
        N, T = data['rewards'].shape
        num_humans = data['obs']['spatial_edges'].shape[2]

        advantages, returns = compute_gae_vectorized(
            data['rewards'], data['values'], data['dones'],
            data['bootstrap_values'], self.gamma, self.gae_lambda,
        )

        values = data['values']
        if self.normalize_returns:
            self.return_rms.update(returns.detach().cpu().numpy())
            ret_std = self.return_rms.std
            returns = returns / ret_std
            values = values / ret_std

        advantages = self._normalize_advantages(advantages)
        cost_adv_raw, cost_returns = self._cost_gae(
            data['coll_labels'], data['cost_values'], data['dones'],
            data['bootstrap_costs'], vectorized=True,
        )
        cost_adv = (
            self._normalize_advantages(cost_adv_raw)
            if self.use_lagrange else torch.zeros_like(cost_adv_raw)
        )

        windows = []  # (n, start, length) -- never spans an env or episode boundary

        boundaries = data['dones'] > 0.5
        boundaries[:, -1] = True

        # nonzero() returns indices in row-major (C) order (sorted by env, then
        # timestep), which is what the per-env segment walk below requires. Sort
        # the (env, timestep) pairs explicitly so iteration is deterministic
        # regardless of backend (CPU/CUDA) or torch version.
        envs, timesteps = boundaries.nonzero(as_tuple=True)
        boundary_pairs = sorted(zip(envs.tolist(), timesteps.tolist()))

        current_env = -1
        seg_start = 0

        for n, t in boundary_pairs:
            if n != current_env:
                seg_start = 0
                current_env = n

            seg_end = t + 1
            s = seg_start
            while s < seg_end:
                e = min(s + self.seq_len, seg_end)
                if e - s >= 4:
                    windows.append((n, s, e - s))
                s = e
            seg_start = seg_end
        if not windows:
            return

        S = self.seq_len
        num_win = len(windows)

        rn = torch.zeros(num_win, S, 7, device=device)
        # spatial dim = 6 (pos + rel_vel + goal_dir); must match crowd_env
        # _get_obs spatial_edges width and models.SNCPPolicy spatial_ltc input.
        se = torch.zeros(num_win, S, num_humans, 6, device=device)
        te = torch.zeros(num_win, S, 2, device=device)
        act = torch.zeros(num_win, S, 2, device=device)
        olp = torch.zeros(num_win, S, device=device)
        adv = torch.zeros(num_win, S, device=device)
        ret = torch.zeros(num_win, S, device=device)
        ov = torch.zeros(num_win, S, device=device)
        coll = torch.zeros(num_win, S, device=device)
        clr = torch.zeros(num_win, S, device=device)
        cadv = torch.zeros(num_win, S, device=device)
        cret = torch.zeros(num_win, S, device=device)
        ocv = torch.zeros(num_win, S, device=device)
        h_te = torch.zeros(num_win, data['h_temporal'].shape[-1], device=device)
        h_no = torch.zeros(num_win, data['h_node'].shape[-1], device=device)
        h_sp = torch.zeros(num_win, num_humans, data['h_spatial'].shape[-1] // num_humans, device=device)
        lengths = []
        for i, (n, st, L) in enumerate(windows):
            rn[i, :L] = data['obs']['robot_node'][n, st:st + L]
            se[i, :L] = data['obs']['spatial_edges'][n, st:st + L]
            te[i, :L] = data['obs']['temporal_edges'][n, st:st + L]
            act[i, :L] = data['actions'][n, st:st + L]
            olp[i, :L] = data['log_probs'][n, st:st + L]
            adv[i, :L] = advantages[n, st:st + L]
            ret[i, :L] = returns[n, st:st + L]
            ov[i, :L] = values[n, st:st + L]
            coll[i, :L] = data['coll_labels'][n, st:st + L]
            clr[i, :L] = data['clearance_labels'][n, st:st + L]
            cadv[i, :L] = cost_adv[n, st:st + L]
            cret[i, :L] = cost_returns[n, st:st + L]
            ocv[i, :L] = data['cost_values'][n, st:st + L]
            h_te[i] = data['h_temporal'][n, st]
            h_no[i] = data['h_node'][n, st]
            h_sp[i] = data['h_spatial'][n, st].reshape(num_humans, -1)
            lengths.append(L)

        epoch_kl = epoch_ent = epoch_clip = 0.0
        epochs_ran = 0
        for epoch in range(self.epochs):
            perm = torch.randperm(num_win)
            batch_kls, batch_ents, batch_clips = [], [], []
            for bs in range(0, num_win, self.batch_size):
                bi = perm[bs:bs + self.batch_size]
                B = len(bi)
                b_rn, b_se, b_te = rn[bi], se[bi], te[bi]
                b_act, b_olp = act[bi], olp[bi]
                b_adv, b_ret, b_ov = adv[bi], ret[bi], ov[bi]
                b_coll, b_clr = coll[bi], clr[bi]
                if self.use_lagrange:
                    b_adv = b_adv - self.lagrange_lambda * cadv[bi]
                b_len = [lengths[j] for j in bi.tolist()]
                b_len_t = torch.tensor(b_len, device=device)
                valid = (torch.arange(S, device=device)[None, :] < b_len_t[:, None]).float()

                h_temp = h_te[bi].clone()
                h_node = h_no[bi].clone()
                h_spat = h_sp[bi].reshape(B * num_humans, -1).clone()

                mus, stds, vals = [], [], []
                risk_p, risk_c, cost_vs = [], [], []
                for t in range(S):
                    step_obs = {'robot_node': b_rn[:, t],
                                'spatial_edges': b_se[:, t],
                                'temporal_edges': b_te[:, t]}
                    step_h = {'temporal_edge': h_temp, 'spatial_edge': h_spat, 'node': h_node}
                    mu, std, val, nh = self.policy(step_obs, step_h)
                    mus.append(mu)
                    stds.append(std)
                    vals.append(val)
                    if self._risk_enabled():
                        risk_p.append(self.policy.last_p_coll)
                        risk_c.append(self.policy.last_min_clearance)
                        cost_vs.append(self.policy.last_cost_value)
                    h_temp = nh['temporal_edge']
                    h_node = nh['node']
                    hs = nh['spatial_edge']
                    h_spat = hs.reshape(B * num_humans, -1) if hs.dim() == 3 else hs

                all_mu = torch.stack(mus, dim=1)
                all_std = torch.stack(stds, dim=1)
                all_val = torch.stack(vals, dim=1).squeeze(-1)

                # Beta vs Gaussian: shared builder so the Beta branch is never
                # skipped here (the vectorized-path bug). Beta log_prob is on the
                # [0,1] pre-image of the stored physical action (affine Jacobian
                # is constant -> cancels in the ratio).
                dist = self.policy.make_action_dist(all_mu, all_std)
                if self.policy.action_dist == 'beta':
                    new_lp = dist.log_prob(self.policy._unscale_action(b_act)).sum(-1)
                else:
                    new_lp = dist.log_prob(b_act).sum(-1)
                entropy = dist.entropy().sum(-1)
                ratio = torch.exp(new_lp - b_olp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                actor_loss = -(torch.min(surr1, surr2) * valid).sum() / valid.sum()

                with torch.no_grad():
                    lr_ = new_lp - b_olp
                    approx_kl = (((torch.exp(lr_) - 1) - lr_) * valid).sum() / valid.sum()
                    ent_mean = (entropy * valid).sum() / valid.sum()
                    clip_frac = (((ratio - 1).abs() > self.clip_eps).float() * valid).sum() / valid.sum()
                    batch_kls.append(approx_kl.item())
                    batch_ents.append(ent_mean.item())
                    batch_clips.append(clip_frac.item())

                critic_loss = self._clipped_value_loss(all_val, b_ov, b_ret, valid)

                entropy_loss = -(entropy * valid).sum() / valid.sum()
                loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
                if self._risk_enabled() and cost_vs:
                    pred_cv = torch.stack(cost_vs, dim=1).squeeze(-1)
                    loss = loss + self.c1 * self._clipped_value_loss(
                        pred_cv, ocv[bi], cret[bi], valid,
                    )
                risk_loss, bce_val, huber_val = self._risk_aux_loss(
                    risk_p, risk_c, b_coll, b_clr, valid,
                )
                if torch.is_tensor(risk_loss):
                    loss = loss + risk_loss
                    self.last_risk_bce = bce_val
                    self.last_risk_huber = huber_val

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

            epochs_ran = epoch + 1
            if batch_kls:
                epoch_kl = float(np.mean(batch_kls))
                epoch_ent = float(np.mean(batch_ents))
                epoch_clip = float(np.mean(batch_clips))
            if self.target_kl is not None and epoch_kl > 1.5 * self.target_kl:
                break

        self.last_entropy = epoch_ent
        self.last_approx_kl = epoch_kl
        self.last_clip_frac = epoch_clip
        self.last_epochs_ran = epochs_ran
        self._step_lagrange_dual(float(data['coll_labels'].mean().item()))
        if self.scheduler is not None:
            self.scheduler.step()
