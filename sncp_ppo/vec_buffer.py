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
        self.bootstrap_values = None  # (N, T), filled by finish() in a later task

    def store(self, obs, hidden, actions, log_probs, rewards, values, dones, masks):
        self.obs_robot_node.append(obs['robot_node'].detach().clone())
        self.obs_spatial_edges.append(obs['spatial_edges'].detach().clone())
        self.obs_temporal_edges.append(obs['temporal_edges'].detach().clone())
        self.h_temporal.append(hidden['temporal_edge'].detach().clone())
        self.h_node.append(hidden['node'].detach().clone())
        # spatial hidden is (N*H, units) -> store as (N, H*units) so the stacked
        # tensor is (N, T, H*units) and indexes cleanly per-env in later tasks.
        h_sp = hidden['spatial_edge'].detach().clone()
        self.h_spatial.append(h_sp.reshape(self.N, -1))
        self.actions.append(torch.as_tensor(actions).detach().clone())
        self.log_probs.append(torch.as_tensor(log_probs).detach().clone())
        self.rewards.append(torch.as_tensor(rewards, dtype=torch.float32).clone())
        self.values.append(torch.as_tensor(values, dtype=torch.float32).clone())
        self.dones.append(torch.as_tensor(dones, dtype=torch.float32).clone())
        self.masks.append(torch.as_tensor(masks, dtype=torch.float32).clone())

    def get_tensors(self, device):
        def stack(lst):
            return torch.stack(lst, dim=1).to(device)  # (N, T, ...)
        bootstrap = self.bootstrap_values if self.bootstrap_values is not None \
            else torch.zeros(self.N, self.T)
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
        }
