"""Training-free action safety shield for v38 evaluations.

The shield is deliberately outside PPO/training. It post-processes a deterministic
policy action at evaluation time by checking a short constant-velocity rollout of
the robot and pedestrians. If the policy action is already predicted safe, it is
returned unchanged; otherwise a small action lattice is scored for clearance,
goal progress, and deviation from the policy action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ActionShieldConfig:
    horizon_steps: int = 6
    safety_margin: float = 0.0
    collision_penalty: float = 1000.0
    clearance_weight: float = 20.0
    progress_weight: float = 1.0
    deviation_weight: float = 0.05
    speed_weight: float = 0.02

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")


def _clip_action(env, action: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.clip(float(action[0]), 0.0, float(env.robot_vpref)),
            np.clip(float(action[1]), -float(env.robot_wmax), float(env.robot_wmax)),
        ],
        dtype=np.float32,
    )


def _candidate_actions(env, action: np.ndarray) -> Iterable[np.ndarray]:
    action = _clip_action(env, action)
    vpref = float(env.robot_vpref)
    wmax = float(env.robot_wmax)
    original_v = float(action[0])
    original_w = float(action[1])

    candidates = [action]
    linear_values = {
        0.0,
        0.25 * vpref,
        0.50 * vpref,
        0.75 * vpref,
        vpref,
        0.50 * original_v,
        0.75 * original_v,
        original_v,
    }
    angular_values = {
        -wmax,
        -0.50 * wmax,
        0.0,
        0.50 * wmax,
        wmax,
        original_w,
    }
    for v in sorted(linear_values):
        for w in sorted(angular_values):
            candidates.append(np.array([v, w], dtype=np.float32))

    seen = set()
    for candidate in candidates:
        clipped = _clip_action(env, candidate)
        key = (round(float(clipped[0]), 6), round(float(clipped[1]), 6))
        if key not in seen:
            seen.add(key)
            yield clipped


def _rollout_stats(env, action: np.ndarray, cfg: ActionShieldConfig) -> tuple[float, float, float]:
    """Return (min_distance, progress, final_speed) for a constant-velocity rollout."""
    action = _clip_action(env, action)
    dt = float(env.time_step)
    px = float(env.robot_px)
    py = float(env.robot_py)
    theta = float(env.robot_theta)
    v = float(action[0])
    w = float(action[1])
    humans_px = np.asarray(env.humans_px, dtype=float).copy()
    humans_py = np.asarray(env.humans_py, dtype=float).copy()
    humans_vx = np.asarray(env.humans_vx, dtype=float)
    humans_vy = np.asarray(env.humans_vy, dtype=float)

    start_goal_dist = math.hypot(float(env.robot_gx) - px, float(env.robot_gy) - py)
    min_dist = math.inf
    for _ in range(cfg.horizon_steps):
        theta = (theta + w * dt + math.pi) % (2.0 * math.pi) - math.pi
        px += v * math.cos(theta) * dt
        py += v * math.sin(theta) * dt
        humans_px = humans_px + humans_vx * dt
        humans_py = humans_py + humans_vy * dt
        if len(humans_px):
            min_dist = min(min_dist, float(np.min(np.hypot(humans_px - px, humans_py - py))))

    final_goal_dist = math.hypot(float(env.robot_gx) - px, float(env.robot_gy) - py)
    progress = start_goal_dist - final_goal_dist
    return min_dist, progress, v


def min_predicted_clearance(env, action: np.ndarray, cfg: ActionShieldConfig) -> float:
    min_dist, _, _ = _rollout_stats(env, action, cfg)
    if not math.isfinite(min_dist):
        return math.inf
    return min_dist - float(env.collision_threshold)


def _score(env, candidate: np.ndarray, original: np.ndarray, cfg: ActionShieldConfig) -> float:
    min_dist, progress, speed = _rollout_stats(env, candidate, cfg)
    clearance = min_dist - float(env.collision_threshold) if math.isfinite(min_dist) else math.inf
    required = cfg.safety_margin
    shortfall = max(0.0, required - clearance)
    collision = clearance < 0.0

    vpref = max(float(env.robot_vpref), 1e-6)
    wmax = max(float(env.robot_wmax), 1e-6)
    deviation = abs(float(candidate[0] - original[0])) / vpref
    deviation += abs(float(candidate[1] - original[1])) / wmax
    speed_loss = max(0.0, 1.0 - speed / vpref)

    return (
        (cfg.collision_penalty if collision else 0.0)
        + cfg.clearance_weight * shortfall * shortfall
        - cfg.progress_weight * progress
        + cfg.deviation_weight * deviation
        + cfg.speed_weight * speed_loss
    )


def shield_action(env, action: np.ndarray, cfg: ActionShieldConfig | None = None) -> np.ndarray:
    """Return a safe replacement action if the policy action is predicted risky."""
    cfg = cfg or ActionShieldConfig()
    original = _clip_action(env, np.asarray(action, dtype=np.float32))
    if min_predicted_clearance(env, original, cfg) >= cfg.safety_margin:
        return original

    best = min(_candidate_actions(env, original), key=lambda candidate: _score(env, candidate, original, cfg))
    return _clip_action(env, best)
