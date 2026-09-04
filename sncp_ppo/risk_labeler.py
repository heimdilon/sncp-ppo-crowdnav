"""Privileged short-horizon collision / clearance labels for v39 training.

This module is an OFFLINE LABELER only. It reuses the constant-velocity
rollout geometry from the v38 action shield so training targets match the
oracle the shield would have used, but it must not be imported by the
deployed policy path (Raspberry Pi / waffle_ros / eval with shield OFF).

Runtime inference is a single SNCPPolicy forward. Do not call these helpers
inside models.SNCPPolicy or the ROS control loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


CLEARANCE_LABEL_CAP = 10.0
DEFAULT_HORIZON_STEPS = 6


def clip_action(env: Any, action: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.clip(float(action[0]), 0.0, float(env.robot_vpref)),
            np.clip(float(action[1]), -float(env.robot_wmax), float(env.robot_wmax)),
        ],
        dtype=np.float32,
    )


def constant_velocity_rollout(
    env: Any, action: np.ndarray, horizon_steps: int = DEFAULT_HORIZON_STEPS,
) -> tuple[float, float, float]:
    """Return (min_distance, progress, final_speed) for a CV rollout.

    Shared by the v39 labeler and the v38 eval-only shield. Pedestrians keep
    their current velocity; the robot applies a constant (v, w) for
    ``horizon_steps`` (default 6 * 0.25s = 1.5s).
    """
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    action = clip_action(env, action)
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
    for _ in range(horizon_steps):
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


def raw_min_clearance(
    env: Any, action: np.ndarray, horizon_steps: int = DEFAULT_HORIZON_STEPS,
) -> float:
    min_dist, _, _ = constant_velocity_rollout(env, action, horizon_steps)
    if not math.isfinite(min_dist):
        return math.inf
    return min_dist - float(env.collision_threshold)


@dataclass(frozen=True)
class RiskLabel:
    collision: float
    min_clearance: float


def label_short_horizon_risk(
    env: Any,
    action: np.ndarray,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    max_clearance: float = CLEARANCE_LABEL_CAP,
) -> RiskLabel:
    """Privileged (collision indicator, non-negative clearance) for one action.

    Collision is 1 if the CV rollout enters the collision threshold. Clearance
    is clipped to ``[0, max_clearance]`` so it matches the softplus risk head.
    """
    raw = raw_min_clearance(env, action, horizon_steps)
    if not math.isfinite(raw):
        raw = float(max_clearance)
    collision = 1.0 if raw < 0.0 else 0.0
    clearance = float(min(max(raw, 0.0), max_clearance))
    return RiskLabel(collision=collision, min_clearance=clearance)


def _unwrap(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def label_vectorized_envs(
    envs: Any,
    actions: np.ndarray,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    max_clearance: float = CLEARANCE_LABEL_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Label each parallel env's current state + clipped action.

    ``envs`` is a Gymnasium VectorEnv (or a stub with ``.envs``). Returns
    ``(coll_labels, clearance_labels)`` as float32 arrays of shape (N,).
    """
    actions = np.asarray(actions, dtype=np.float32)
    n = actions.shape[0]
    coll = np.zeros(n, dtype=np.float32)
    clearance = np.zeros(n, dtype=np.float32)
    for i, env in enumerate(envs.envs):
        label = label_short_horizon_risk(
            _unwrap(env), actions[i],
            horizon_steps=horizon_steps, max_clearance=max_clearance,
        )
        coll[i] = label.collision
        clearance[i] = label.min_clearance
    return coll, clearance
