"""Phase 2: behavior-cloning pretrain. A fresh SNCPPolicy is trained to match the
ORCA expert's actions (MSE in normalized [v/vpref, w/wmax] space) over BPTT
windows = whole episodes (zero hidden at episode start, unroll to the end)."""
import numpy as np
import torch

from sncp_ppo.il.pretrain_bc import pretrain_bc, episodes_from_shard
from sncp_ppo.models import build_policy_for_checkpoint


def _tiny_shard(n_humans=3, n_eps=4, ep_len=8, seed=0):
    rng = np.random.default_rng(seed)
    T = n_eps * ep_len
    actions = np.stack([
        rng.uniform(0.0, 1.0, T),          # v in [0, vpref]
        rng.uniform(-1.8, 1.8, T),         # w in [-wmax, wmax]
    ], axis=1).astype(np.float32)
    return {
        'robot_node': rng.standard_normal((T, 7)).astype(np.float32),
        'spatial_edges': rng.standard_normal((T, n_humans, 6)).astype(np.float32),
        'temporal_edges': rng.standard_normal((T, 2)).astype(np.float32),
        'actions': actions,
        'episode_lengths': [ep_len] * n_eps,
    }


def test_episodes_from_shard_splits_on_boundaries():
    shard = _tiny_shard(n_humans=3, n_eps=4, ep_len=8)
    eps = episodes_from_shard(shard)
    assert len(eps) == 4
    assert all(ep['robot_node'].shape[0] == 8 for ep in eps)
    assert eps[0]['spatial_edges'].shape == (8, 3, 6)


def test_bc_loss_decreases_on_overfit():
    shards = {3: _tiny_shard(n_humans=3, n_eps=4, ep_len=8, seed=1)}
    _, history = pretrain_bc(
        shards, epochs=80, lr=1e-3, batch_size=4,
        robot_vpref=1.0, robot_wmax=1.8, device=torch.device('cpu'), seed=0,
    )
    # Random obs->action overfit is hard (no input-target structure, only
    # positional memorization), so require a clear drop rather than a tiny one.
    assert history[-1] < 0.6 * history[0], f"loss did not drop: {history[0]:.3f} -> {history[-1]:.3f}"


def test_bc_checkpoint_loads_via_builder(tmp_path):
    shards = {3: _tiny_shard(n_humans=3, n_eps=2, ep_len=8, seed=2)}
    policy, _ = pretrain_bc(
        shards, epochs=3, lr=1e-3, batch_size=2,
        robot_vpref=1.0, robot_wmax=1.8, device=torch.device('cpu'), seed=0,
    )
    path = tmp_path / 'v23_bc.pt'
    torch.save(policy.state_dict(), path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    rebuilt = build_policy_for_checkpoint(state, robot_vpref=1.0, robot_wmax=1.8)
    rebuilt.load_state_dict(state)  # must not raise
    assert rebuilt.pre_mlp is False


def test_bc_moves_mu_toward_expert_actions():
    """After overfitting a constant-action episode, the policy's mean action is
    closer to that target than a fresh policy's."""
    n_humans, ep_len = 3, 10
    target = np.array([0.7, -0.9], dtype=np.float32)
    shard = {
        'robot_node': np.ones((ep_len, 7), np.float32),
        'spatial_edges': np.ones((ep_len, n_humans, 6), np.float32),
        'temporal_edges': np.ones((ep_len, 2), np.float32),
        'actions': np.tile(target, (ep_len, 1)),
        'episode_lengths': [ep_len],
    }
    device = torch.device('cpu')
    trained, _ = pretrain_bc(
        {3: shard}, epochs=120, lr=2e-3, batch_size=1,
        robot_vpref=1.0, robot_wmax=1.8, device=device, seed=0,
    )

    obs = {
        'robot_node': torch.ones(1, 7),
        'spatial_edges': torch.ones(1, n_humans, 6),
        'temporal_edges': torch.ones(1, 2),
    }
    h = trained.init_hidden(1, n_humans, device)
    with torch.no_grad():
        mu_trained, _, _, _ = trained(obs, h)
    err = float(torch.abs(mu_trained.squeeze(0) - torch.tensor(target)).sum())
    assert err < 0.4, f"trained mu {mu_trained.squeeze(0).tolist()} far from target {target.tolist()}"
