import torch
from sncp_ppo.vec_buffer import compute_gae_vectorized
from sncp_ppo.ppo import PPOAgent
from sncp_ppo.models import SNCPPolicy


def test_vectorized_gae_matches_single_env():
    """The new (N,T) done-masked GAE must equal the legacy episode-aware GAE
    on the same transition sequence (N=1, one truncated episode)."""
    gamma, lam = 0.99, 0.95
    T = 6
    rewards = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    masks = torch.ones(T)
    dones = torch.zeros(T)
    bootstrap = 0.7

    agent = PPOAgent(policy=SNCPPolicy())
    ref_adv, ref_ret = agent.compute_gae(
        rewards, values, masks, episode_lengths=[T], bootstrap_values=[bootstrap]
    )

    dones_NT = dones.view(1, T)
    dones_NT[0, T - 1] = 1.0
    adv, ret = compute_gae_vectorized(
        rewards.view(1, T), values.view(1, T), dones_NT,
        bootstrap_values=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, bootstrap]]),
        gamma=gamma, gae_lambda=lam,
    )
    assert torch.allclose(adv.view(-1), ref_adv, atol=1e-5), f"{adv.view(-1)} vs {ref_adv}"
    assert torch.allclose(ret.view(-1), ref_ret, atol=1e-5)


def test_vectorized_gae_resets_at_done():
    """A terminated episode mid-rollout must not bleed advantage into the next
    episode in the same env row."""
    gamma, lam = 0.99, 0.95
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])   # N=1, T=4
    values = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    dones = torch.tensor([[0.0, 1.0, 0.0, 1.0]])     # episode ends at t=1 and t=3
    bootstrap = torch.tensor([[0.0, 0.0, 0.0, 0.0]]) # terminated: no bootstrap
    adv, ret = compute_gae_vectorized(rewards, values, dones, bootstrap, gamma, lam)
    expected_first = 1.0 + gamma * lam * 1.0
    assert torch.allclose(adv[0, 1], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(adv[0, 0], torch.tensor(expected_first), atol=1e-5)
    assert torch.allclose(adv[0, 3], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(adv[0, 2], torch.tensor(expected_first), atol=1e-5)
