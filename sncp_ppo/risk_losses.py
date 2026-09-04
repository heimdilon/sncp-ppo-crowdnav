"""Supervised risk-head losses and Lagrangian dual ascent for v39 PPO."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(per_step: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return per_step.mean()
    weights = valid_mask.reshape_as(per_step)
    denom = weights.sum().clamp_min(1.0)
    return (per_step * weights).sum() / denom


def risk_supervision_loss(
    p_coll: torch.Tensor,
    clearance_pred: torch.Tensor,
    coll_label: torch.Tensor,
    clearance_label: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BCE on collision probability + Huber/SmoothL1 on non-negative clearance.

    Returns (bce, huber, bce + huber). Tensors may be any broadcastable shape;
    they are flattened against ``valid_mask`` when one is provided.
    """
    p_coll = p_coll.reshape(-1)
    clearance_pred = clearance_pred.reshape(-1)
    coll_label = coll_label.reshape(-1).to(dtype=p_coll.dtype)
    clearance_label = clearance_label.reshape(-1).to(dtype=clearance_pred.dtype)
    if valid_mask is not None:
        valid_mask = valid_mask.reshape(-1).to(dtype=p_coll.dtype)

    bce_each = F.binary_cross_entropy(
        p_coll.clamp(1e-6, 1.0 - 1e-6), coll_label, reduction="none",
    )
    huber_each = F.smooth_l1_loss(clearance_pred, clearance_label, reduction="none")
    bce = _masked_mean(bce_each, valid_mask)
    huber = _masked_mean(huber_each, valid_mask)
    return bce, huber, bce + huber


def dual_ascent_update(
    lam: float, mean_cost: float, cost_limit: float, lr: float, lam_max: float,
) -> float:
    """Projected dual ascent: λ ← clip(λ + α (J_cost − d), 0, λ_max)."""
    updated = float(lam) + float(lr) * (float(mean_cost) - float(cost_limit))
    return float(max(0.0, min(float(lam_max), updated)))
