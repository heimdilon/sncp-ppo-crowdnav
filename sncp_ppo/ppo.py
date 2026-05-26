import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


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

        # Episode boundary tracking
        self.episode_lengths = []
        self.episode_bootstrap_values = []  # V(s_final): nonzero only for truncated episodes
        self._current_ep_len = 0

    def store(self, obs, h_states, action, log_prob, reward, value, mask):
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
        self._current_ep_len += 1

    def end_episode(self, bootstrap_value=0.0):
        """Record the episode boundary.

        bootstrap_value: V(s_final) for truncated episodes (timeout); 0.0
        for terminated episodes (collision/goal-reached). Without this,
        GAE assumes the world ends at every episode boundary, which biases
        the value function for truncated rollouts.
        """
        if self._current_ep_len > 0:
            self.episode_lengths.append(self._current_ep_len)
            self.episode_bootstrap_values.append(float(bootstrap_value))
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
        self.episode_lengths.clear()
        self.episode_bootstrap_values.clear()
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
        
        return obs, h_states, actions, log_probs, rewards, values, masks

class PPOAgent:
    def __init__(self, policy, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 c1=0.5, c2=0.01, epochs=4, batch_size=64, seq_len=16,
                 total_updates=None, lr_end_factor=0.1, target_kl=0.015,
                 normalize_returns=True):
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
            mu, std, value, h_states_new = self.policy(obs_tensor, h_states)

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

    def _extract_subsequences(self, obs, h_states, actions, old_log_probs, advantages, returns, values, device):
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
        obs, h_states, actions, old_log_probs, rewards, values, masks = self.memory.get_tensors(device)

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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Extract subsequences for BPTT (values plumbed for clipped value loss)
        seqs = self._extract_subsequences(obs, h_states, actions, old_log_probs,
                                          advantages, returns, values, device)

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
        kl_break = False

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
                
                # Get actual lengths for this batch
                b_lengths = [seqs['seq_lengths'][idx.item()] for idx in batch_idx]
                
                # Create valid mask [B, S]
                valid_mask = torch.zeros(B, S, device=device)
                for i, L in enumerate(b_lengths):
                    valid_mask[i, :L] = 1.0
                
                # Unroll through the sequence with BPTT
                all_mu = []
                all_std = []
                all_values = []
                
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
                    
                    mu, std, value, new_h = self.policy(step_obs, step_h)
                    all_mu.append(mu)
                    all_std.append(std)
                    all_values.append(value)
                    
                    # Update hidden states for next step (BPTT: keep graph!)
                    h_temp = new_h['temporal_edge']
                    h_node = new_h['node']
                    h_spat_out = new_h['spatial_edge']
                    if h_spat_out.dim() == 3:
                        h_spat = h_spat_out.reshape(B * num_humans, -1)
                    else:
                        h_spat = h_spat_out
                
                # Stack: [B, S, ...]
                all_mu = torch.stack(all_mu, dim=1)      # [B, S, 2]
                all_std = torch.stack(all_std, dim=1)     # [B, S, 2]
                all_values = torch.stack(all_values, dim=1).squeeze(-1)  # [B, S]
                
                # Compute log probs and entropy
                dist = torch.distributions.Normal(all_mu, all_std)
                new_log_probs = dist.log_prob(b_actions).sum(-1)  # [B, S]
                entropy = dist.entropy().sum(-1)                    # [B, S]
                
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
                v_clipped = b_old_v + torch.clamp(all_values - b_old_v,
                                                  -self.clip_eps, self.clip_eps)
                vloss_unclipped = (all_values - b_ret).pow(2)
                vloss_clipped   = (v_clipped  - b_ret).pow(2)
                vloss_per_step  = torch.max(vloss_unclipped, vloss_clipped)
                critic_loss = (vloss_per_step * valid_mask).sum() / valid_mask.sum()

                entropy_loss = -(entropy * valid_mask).sum() / valid_mask.sum()
                
                loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
                
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
                kl_break = True
                break

        # Persist diagnostics for the training loop to log.
        self.last_entropy = epoch_entropy
        self.last_approx_kl = epoch_kl
        self.last_clip_frac = epoch_clip_frac
        self.last_epochs_ran = epochs_ran

        # Decay learning rate per update if a scheduler is attached
        if self.scheduler is not None:
            self.scheduler.step()

        # Clear memory buffer after updating
        self.memory.clear()
