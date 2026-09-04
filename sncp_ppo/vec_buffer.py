import torch


def compute_gae_vectorized(rewards, values, dones, bootstrap_values,
                           gamma=0.99, gae_lambda=0.95):
    """Done-masked GAE over (N, T) tensors.

    Args:
        rewards:          (N, T) per-step rewards.
        values:           (N, T) per-step value estimates V(s_t).
        dones:            (N, T) 1.0 if step t is the last step of an episode
                          for env n (terminated OR truncated OR horizon-cut).
        bootstrap_values: (N, T) V(s_{t+1}) to use at episode-boundary steps;
                          0.0 for terminated, V(s_final) for truncated/horizon.
                          Ignored where dones==0.
        gamma, gae_lambda: GAE hyperparameters.

    Returns:
        advantages (N, T), returns (N, T).

    Each env is processed independently along the time axis. At a done step the
    next-state value is `bootstrap_values[n, t]` (not values[n, t+1]) and the
    GAE accumulator resets afterwards, so advantages never bleed across episode
    boundaries within an env's rollout. Mirrors PPOAgent.compute_gae but driven
    by a done mask instead of an episode-length list.
    """
    N, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    for n in range(N):
        gae = 0.0
        for t in reversed(range(T)):
            if dones[n, t] > 0.5:
                next_value = bootstrap_values[n, t]
                next_nonterminal = 0.0  # accumulator resets after a boundary
            else:
                next_value = values[n, t + 1] if t + 1 < T else 0.0
                next_nonterminal = 1.0
            delta = rewards[n, t] + gamma * next_value - values[n, t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[n, t] = gae
    returns = advantages + values
    return advantages, returns


def reset_hidden_where_done(hidden, dones, num_humans):
    """Zero the LTC hidden state of envs that just finished an episode.

    SyncVectorEnv auto-resets a done env, so its next observation is already the
    new episode's first step. The recurrent hidden must be cleared for those env
    rows or the new episode inherits stale memory (a silent correctness bug).

    hidden: dict with 'temporal_edge' (N,U), 'spatial_edge' (N*H,U), 'node' (N,U).
    dones:  (N,) 1.0 where the env finished this step.
    """
    N = dones.shape[0]
    keep = (dones < 0.5).float()                  # (N,) 1.0 to keep, 0.0 to zero
    keep_col = keep.view(N, 1)
    new = {
        'temporal_edge': hidden['temporal_edge'] * keep_col,
        'node': hidden['node'] * keep_col,
    }
    # spatial is (N*H, U): expand keep per env across its H rows
    keep_spat = keep.repeat_interleave(num_humans).view(N * num_humans, 1)
    new['spatial_edge'] = hidden['spatial_edge'] * keep_spat
    return new


class VectorizedRolloutBuffer:
    """Fixed (num_envs, horizon) rollout storage for vectorized PPO.

    Unlike the list-based, episode-bounded PPOMemory, this stores a dense N x T
    block per step. Hidden states fed INTO each step are recorded so BPTT can
    re-feed them during the update. Episode boundaries are tracked via the
    `dones` mask, which drives both GAE reset and BPTT hidden-reset.
    """

    def __init__(self, num_envs, horizon):
        self.N = num_envs
        self.T = horizon
        self._reset_lists()

    def _reset_lists(self):
        self.obs_robot_node = []      # each: (N, 7)
        self.obs_spatial_edges = []   # each: (N, H, 4)
        self.obs_temporal_edges = []  # each: (N, 2)
        self.h_temporal = []          # each: (N, units)
        self.h_spatial = []           # each: (N, H*units)  (reshaped in store)
        self.h_node = []              # each: (N, units)
        self.actions = []             # each: (N, 2)
        self.log_probs = []           # each: (N,)
        self.rewards = []             # each: (N,)
        self.values = []              # each: (N,)
        self.dones = []               # each: (N,)
        self.masks = []               # each: (N,)  (1 - terminated)
        self.coll_labels = []         # each: (N,) privileged short-horizon collision
        self.clearance_labels = []    # each: (N,) privileged min clearance (>=0)
        self.cost_values = []         # each: (N,) cost-critic V_cost(s_t)
        self.bootstraps = []          # each: (N,) V(s_final) at truncation; else 0
        self.cost_bootstraps = []     # each: (N,) V_cost(s_final) at truncation
        self.bootstrap_values = None  # (N, T), filled by finish()
        self.bootstrap_costs = None   # (N, T), cost-critic bootstrap

    def store(self, obs, hidden, actions, log_probs, rewards, values, dones, masks,
              coll_labels=None, clearance_labels=None, cost_values=None,
              bootstrap=None, cost_bootstrap=None):
        self.obs_robot_node.append(obs['robot_node'].detach().clone())
        self.obs_spatial_edges.append(obs['spatial_edges'].detach().clone())
        self.obs_temporal_edges.append(obs['temporal_edges'].detach().clone())
        self.h_temporal.append(hidden['temporal_edge'].detach().clone())
        self.h_node.append(hidden['node'].detach().clone())
        # spatial hidden is (N*H, units) -> store as (N, H*units) so the stacked
        # tensor is (N, T, H*units) and indexes cleanly per-env in later tasks.
        h_sp = hidden['spatial_edge'].detach().clone()
        self.h_spatial.append(h_sp.reshape(self.N, -1))
        self.actions.append(torch.as_tensor(actions, dtype=torch.float32).detach().clone())
        self.log_probs.append(torch.as_tensor(log_probs, dtype=torch.float32).detach().clone())
        self.rewards.append(torch.as_tensor(rewards, dtype=torch.float32).detach().clone())
        self.values.append(torch.as_tensor(values, dtype=torch.float32).detach().clone())
        self.dones.append(torch.as_tensor(dones, dtype=torch.float32).detach().clone())
        self.masks.append(torch.as_tensor(masks, dtype=torch.float32).detach().clone())
        if coll_labels is None:
            coll_labels = torch.zeros(self.N, dtype=torch.float32)
        if clearance_labels is None:
            clearance_labels = torch.zeros(self.N, dtype=torch.float32)
        if cost_values is None:
            cost_values = torch.zeros(self.N, dtype=torch.float32)
        if bootstrap is None:
            bootstrap = torch.zeros(self.N, dtype=torch.float32)
        if cost_bootstrap is None:
            cost_bootstrap = torch.zeros(self.N, dtype=torch.float32)
        self.coll_labels.append(torch.as_tensor(coll_labels, dtype=torch.float32).detach().clone())
        self.clearance_labels.append(torch.as_tensor(clearance_labels, dtype=torch.float32).detach().clone())
        self.cost_values.append(torch.as_tensor(cost_values, dtype=torch.float32).detach().clone())
        self.bootstraps.append(torch.as_tensor(bootstrap, dtype=torch.float32).detach().clone())
        self.cost_bootstraps.append(torch.as_tensor(cost_bootstrap, dtype=torch.float32).detach().clone())

    def finish(self, last_values, last_dones, last_cost_values=None):
        """Finalize the rollout: merge per-step truncation bootstraps with the
        horizon-cut successor value and mark the last column done for GAE.

        last_values: (N,) V(s_next) after the final stored step, per env.
        last_dones:  kept for call-site compatibility (terminated-at-horizon).
                     Horizon-end fill now keys off the stored dones mask:
                     continuing envs get last_values; term/trunc keep the
                     per-step bootstrap (0 for termination, V(s_final) for
                     timeout). last_dones is unused for the fill itself.

        Mid-rollout timeouts must pass V(s_final) via store(..., bootstrap=);
        otherwise GAE would treat them as terminated (implicit next-value 0).
        """
        N, T = self.N, self.T
        last_values = torch.as_tensor(last_values, dtype=torch.float32)
        device = last_values.device
        if self.bootstraps:
            boot = torch.stack(self.bootstraps, dim=1).to(device)
        else:
            boot = torch.zeros(N, T, device=device)
        already_done = self.dones[-1].to(device).clone() if self.dones else torch.zeros(N, device=device)
        boot[:, T - 1] = torch.where(already_done > 0.5, boot[:, T - 1], last_values)
        done_device = self.dones[-1].device if self.dones else device
        self.dones[T - 1] = torch.ones(N, device=done_device)
        self.bootstrap_values = boot
        if self.cost_bootstraps:
            cost_boot = torch.stack(self.cost_bootstraps, dim=1).to(device)
        else:
            cost_boot = torch.zeros(N, T, device=device)
        if last_cost_values is not None:
            last_cost_values = torch.as_tensor(
                last_cost_values, dtype=torch.float32, device=device,
            )
            cost_boot[:, T - 1] = torch.where(
                already_done > 0.5, cost_boot[:, T - 1], last_cost_values,
            )
        self.bootstrap_costs = cost_boot

    def get_tensors(self, device):
        def stack(lst):
            return torch.stack(lst, dim=1).to(device)  # (N, T, ...)
        bootstrap = self.bootstrap_values if self.bootstrap_values is not None \
            else torch.zeros(self.N, self.T, device=device)
        cost_bootstrap = self.bootstrap_costs if self.bootstrap_costs is not None \
            else torch.zeros(self.N, self.T, device=device)
        return {
            'obs': {
                'robot_node': stack(self.obs_robot_node),
                'spatial_edges': stack(self.obs_spatial_edges),
                'temporal_edges': stack(self.obs_temporal_edges),
            },
            'h_temporal': stack(self.h_temporal),
            'h_spatial': stack(self.h_spatial),
            'h_node': stack(self.h_node),
            'actions': stack(self.actions),
            'log_probs': stack(self.log_probs),
            'rewards': stack(self.rewards),
            'values': stack(self.values),
            'dones': stack(self.dones),
            'masks': stack(self.masks),
            'bootstrap_values': bootstrap.to(device),
            'coll_labels': stack(self.coll_labels) if self.coll_labels else torch.zeros(self.N, self.T, device=device),
            'clearance_labels': stack(self.clearance_labels) if self.clearance_labels else torch.zeros(self.N, self.T, device=device),
            'cost_values': stack(self.cost_values) if self.cost_values else torch.zeros(self.N, self.T, device=device),
            'bootstrap_costs': cost_bootstrap.to(device),
        }
