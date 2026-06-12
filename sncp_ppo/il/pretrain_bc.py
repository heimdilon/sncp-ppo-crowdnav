"""Phase 2: behavior-cloning pretrain of a fresh SNCPPolicy from ORCA-expert demos.

BPTT window = a whole episode (zero hidden at the start, unroll to the end), since
the demos carry no policy hidden state. Batches are per-density (the spatial-edge
width = num_humans differs across N). Loss is MSE between the policy mean action
and the expert action in NORMALIZED [v/vpref, w/wmax] space, so the two action
dimensions (different scales) contribute evenly.

Produces checkpoints/sncp_ppo_v23_bc.pt, a plain policy.state_dict() that
build_policy_for_checkpoint loads — the same checkpoint format PPO fine-tune and
the eval pipeline already consume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sncp_ppo.il.collect_demos import load_dataset
from sncp_ppo.models import SNCPPolicy


def episodes_from_shard(shard):
    """Split a flattened per-density shard back into per-episode arrays."""
    eps = []
    offset = 0
    for length in shard['episode_lengths']:
        length = int(length)
        eps.append({
            'robot_node': np.asarray(shard['robot_node'][offset:offset + length]),
            'spatial_edges': np.asarray(shard['spatial_edges'][offset:offset + length]),
            'temporal_edges': np.asarray(shard['temporal_edges'][offset:offset + length]),
            'actions': np.asarray(shard['actions'][offset:offset + length]),
            'length': length,
        })
        offset += length
    return eps


def _batch_loss(policy, batch, n_humans, vpref, wmax, device):
    """MSE(policy mean, expert action) over a batch of same-density episodes,
    unrolled with BPTT from a zero hidden state, masked to real (unpadded) steps."""
    B = len(batch)
    max_len = max(ep['length'] for ep in batch)
    rn = torch.zeros(B, max_len, 7, device=device)
    se = torch.zeros(B, max_len, n_humans, 6, device=device)
    te = torch.zeros(B, max_len, 2, device=device)
    act = torch.zeros(B, max_len, 2, device=device)
    valid = torch.zeros(B, max_len, device=device)
    for i, ep in enumerate(batch):
        L = ep['length']
        rn[i, :L] = torch.from_numpy(ep['robot_node']).float()
        se[i, :L] = torch.from_numpy(ep['spatial_edges']).float()
        te[i, :L] = torch.from_numpy(ep['temporal_edges']).float()
        act[i, :L] = torch.from_numpy(ep['actions']).float()
        valid[i, :L] = 1.0

    h = policy.init_hidden(B, n_humans, device)
    mus = []
    for t in range(max_len):
        step_obs = {'robot_node': rn[:, t], 'spatial_edges': se[:, t], 'temporal_edges': te[:, t]}
        mu, _, _, h = policy(step_obs, h)
        mus.append(mu)
    mu_seq = torch.stack(mus, dim=1)  # [B, T, 2]

    scale = torch.tensor([vpref, wmax], device=device)
    pred = mu_seq / scale
    tgt = act / scale
    sq = ((pred - tgt) ** 2).sum(-1)  # [B, T]
    return (sq * valid).sum() / valid.sum().clamp(min=1.0)


def pretrain_bc(shards, epochs=10, lr=1e-3, batch_size=32,
                robot_vpref=1.0, robot_wmax=1.8, device=None, seed=0,
                pre_mlp=False, log_every=0):
    """Train a fresh SNCPPolicy to clone the expert. Returns (policy, history)."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = SNCPPolicy(robot_vpref=robot_vpref, robot_wmax=robot_wmax, pre_mlp=pre_mlp).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    per_density = {int(n): episodes_from_shard(shard) for n, shard in shards.items()}
    history = []
    for epoch in range(epochs):
        # Build a flat list of (density, batch) so densities interleave each epoch.
        batches = []
        for n_humans, eps in per_density.items():
            order = rng.permutation(len(eps))
            for start in range(0, len(eps), batch_size):
                idx = order[start:start + batch_size]
                batches.append((n_humans, [eps[j] for j in idx]))
        rng.shuffle(batches)

        epoch_losses = []
        for n_humans, batch in batches:
            loss = _batch_loss(policy, batch, n_humans, robot_vpref, robot_wmax, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        history.append(avg)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch + 1}/{epochs}  bc_loss={avg:.5f}")
    return policy, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--demos', type=str, default='data/il_demos.npz')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--robot_vpref', type=float, default=1.0)
    parser.add_argument('--robot_wmax', type=float, default=1.8)
    parser.add_argument('--pre_mlp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='checkpoints/sncp_ppo_v23_bc.pt')
    args = parser.parse_args(argv)

    shards = load_dataset(args.demos)
    counts = {n: len(s['episode_lengths']) for n, s in shards.items()}
    print(f"Loaded demos: {counts} (total {sum(counts.values())} episodes)")

    policy, history = pretrain_bc(
        shards, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        robot_vpref=args.robot_vpref, robot_wmax=args.robot_wmax,
        pre_mlp=args.pre_mlp, seed=args.seed, log_every=max(1, args.epochs // 10),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out)
    print(f"\nBC loss {history[0]:.5f} -> {history[-1]:.5f}; saved -> {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
