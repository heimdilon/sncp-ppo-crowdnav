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
