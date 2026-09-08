"""v41 training knobs: weighted P(N), Lagrange cost density scaling, denser holdout.

Defaults preserve pre-v41 behaviour. None of these flags change the SNCP
architecture (LTC + soft HH + risk head); they only change sampling, cost
scaling, optional KNN token selection, and holdout arms.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_NEAR_RADIUS = 2.0
PAPER_HOLDOUT_DENSITY_NS = (5, 10, 15, 20)
_PAPER_HOLD_BASES = ("paper_standard", "paper_challenging")
_HOLD_N_RE = re.compile(r"^(paper_standard|paper_challenging)_n(\d+)$")
_WEIGHT_PAIR_RE = re.compile(r"^\s*(\d+)\s*:\s*([0-9]*\.?[0-9]+)\s*$")


def parse_human_count_weights(spec: str | None) -> dict[int, float] | None:
    """Parse ``5:0.05,10:0.1,20:0.35`` into ``{N: weight}``.

    Empty / None -> None (caller should use uniform ``randint``).
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    weights: dict[int, float] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _WEIGHT_PAIR_RE.match(chunk)
        if match is None:
            raise ValueError(
                "invalid --human_count_weights entry "
                f"{chunk!r}; expected N:weight (e.g. 5:0.05,20:0.35)"
            )
        n = int(match.group(1))
        w = float(match.group(2))
        if n <= 0:
            raise ValueError(f"human count must be positive, got {n}")
        if w < 0.0:
            raise ValueError(f"human-count weight must be >= 0, got {w} for N={n}")
        weights[n] = w
    if not weights:
        return None
    if sum(weights.values()) <= 0.0:
        raise ValueError("--human_count_weights must sum to a positive total")
    return weights


def sample_human_count(
    rng: Any,
    num_humans_range: Sequence[int],
    weights: Mapping[int, float] | None = None,
) -> int:
    """Sample N for one vectorized update window.

    No weights -> inclusive uniform ``randint(MIN, MAX)`` (pre-v41).
    With weights -> discrete sample over keys that fall inside ``[MIN, MAX]``.
    """
    lo = int(num_humans_range[0])
    hi = int(num_humans_range[1])
    if lo > hi:
        raise ValueError(f"num_humans_range MIN={lo} exceeds MAX={hi}")
    if not weights:
        return int(rng.randint(lo, hi))
    items = [(int(n), float(w)) for n, w in weights.items() if lo <= int(n) <= hi and float(w) > 0.0]
    if not items:
        raise ValueError(
            f"--human_count_weights {dict(weights)} has no mass inside "
            f"num_humans_range [{lo}, {hi}]"
        )
    ns, ws = zip(*items)
    return int(_weighted_choice(rng, ns, ws))


def _weighted_choice(rng: Any, items: Sequence[int], weights: Sequence[float]) -> int:
    total = float(sum(weights))
    draw = float(rng.random()) * total
    acc = 0.0
    chosen = items[-1]
    for item, weight in zip(items, weights):
        acc += float(weight)
        if draw <= acc:
            chosen = item
            break
    return int(chosen)


def count_n_near(spatial_edges: np.ndarray, radius: float = DEFAULT_NEAR_RADIUS) -> np.ndarray:
    """Count humans with Euclidean robot-relative distance <= radius.

    ``spatial_edges`` is ``(..., H, 6)`` with ``[..., 0:2] = (dx, dy)``.
    Returns an array of shape ``spatial_edges.shape[:-2]``.
    """
    dist = np.hypot(spatial_edges[..., 0], spatial_edges[..., 1])
    return (dist <= float(radius)).sum(axis=-1).astype(np.float32)


def adapt_lagrange_cost(
    cost: np.ndarray | float,
    *,
    n_near: np.ndarray | float | None = None,
    n_humans: np.ndarray | float | None = None,
    alpha: float = 0.15,
    mode: str = "n_near",
) -> np.ndarray:
    """Scale privileged collision cost for the Lagrangian, not the BCE label.

    ``n_near``: ``C' = C / (1 + alpha * n_near)``
    ``sqrt_n``: ``C' = C / max(sqrt(N), 1)``
    ``n_humans``: ``C' = C / (1 + alpha * N)``
    """
    cost_arr = np.asarray(cost, dtype=np.float32)
    mode = str(mode)
    alpha = float(alpha)
    if mode == "n_near":
        if n_near is None:
            raise ValueError("adapt_lagrange_cost(mode='n_near') requires n_near")
        denom = 1.0 + alpha * np.asarray(n_near, dtype=np.float32)
    elif mode == "sqrt_n":
        if n_humans is None:
            raise ValueError("adapt_lagrange_cost(mode='sqrt_n') requires n_humans")
        denom = np.maximum(np.sqrt(np.asarray(n_humans, dtype=np.float32)), 1.0)
    elif mode == "n_humans":
        if n_humans is None:
            raise ValueError("adapt_lagrange_cost(mode='n_humans') requires n_humans")
        denom = 1.0 + alpha * np.asarray(n_humans, dtype=np.float32)
    else:
        raise ValueError(f"unknown cost-adapt mode {mode!r}; use n_near, sqrt_n, or n_humans")
    return (cost_arr / np.maximum(denom, 1e-6)).astype(np.float32)


def paper_holdout_alias(base: str, n_humans: int) -> str:
    return f"{base}_n{int(n_humans)}"


def holdout_density_aliases() -> dict[str, tuple[str, int, float]]:
    """Named arms so train-time best-ckpt can track min(N=15, N=20)."""
    aliases: dict[str, tuple[str, int, float]] = {}
    for base in _PAPER_HOLD_BASES:
        for n in PAPER_HOLDOUT_DENSITY_NS:
            aliases[paper_holdout_alias(base, n)] = (base, int(n), 1.0)
    return aliases


def holdout_scenario_choices(base_config: Mapping[str, tuple[int, float]]) -> list[str]:
    names = list(base_config)
    for alias in holdout_density_aliases():
        if alias not in names:
            names.append(alias)
    return names


def resolve_holdout_arm(
    name: str,
    base_config: Mapping[str, tuple[int, float]],
) -> tuple[str, int, float]:
    """Return ``(env_scenario, num_humans, human_vpref)`` for a holdout arm."""
    aliases = holdout_density_aliases()
    if name in aliases:
        base, n_h, vpref = aliases[name]
        if name in aliases and base in base_config:
            vpref = float(base_config[base][1])
        return base, int(n_h), float(vpref)
    match = _HOLD_N_RE.fullmatch(str(name))
    if match is not None:
        base = match.group(1)
        vpref = float(base_config[base][1]) if base in base_config else 1.0
        return base, int(match.group(2)), vpref
    n_h, vpref = base_config.get(name, (5, 0.50))
    return str(name), int(n_h), float(vpref)


def expand_holdout_scenarios(
    scenarios: Sequence[str],
    densities: Sequence[int] | None,
) -> list[str]:
    """Expand paper holdout scenarios across ``--holdout_densities``.

    Example: ``['paper_challenging']`` + ``[15, 20]`` ->
    ``['paper_challenging_n15', 'paper_challenging_n20']``.
    Non-paper names are kept as-is. ``densities is None`` is a no-op.
    """
    if not densities:
        return list(scenarios)
    expanded: list[str] = []
    for sc in scenarios:
        match = _HOLD_N_RE.fullmatch(str(sc))
        if match is not None:
            expanded.append(sc)
            continue
        if sc in _PAPER_HOLD_BASES:
            for n in densities:
                expanded.append(paper_holdout_alias(sc, int(n)))
        else:
            expanded.append(sc)
    return expanded


def resolve_init_checkpoint_path(args: Any) -> str | None:
    """``--init_checkpoint``, ``--resume``, and ``--load_path`` are synonyms."""
    paths = []
    for name in ("init_checkpoint", "resume", "load_path"):
        value = getattr(args, name, None)
        if value:
            paths.append(str(value))
    unique = list(dict.fromkeys(paths))
    if len(unique) > 1:
        raise ValueError(
            "conflicting checkpoint paths: "
            f"--init_checkpoint/--resume/--load_path got {unique}"
        )
    return unique[0] if unique else None


def apply_v41_runtime_flags(policy: Any, args: Any) -> None:
    """Attach KNN token-mode flags to a loaded or freshly built policy.

    These are observation-side processing knobs, not extra weights, so a v40
    LTC+soft-HH+risk checkpoint still ``load_state_dict``s strictly.
    """
    mode = getattr(args, "human_token_mode", "all") or "all"
    policy.human_token_mode = str(mode)
    policy.knn_k = int(getattr(args, "knn_k", 10) or 10)
    no_summary = bool(getattr(args, "no_crowd_summary", False))
    explicit = bool(getattr(args, "crowd_summary", False))
    policy.crowd_summary = bool(
        explicit or (policy.human_token_mode == "knn_sorted" and not no_summary)
    )
    radius = getattr(args, "knn_summary_radius", None)
    if radius is None:
        radius = getattr(args, "cost_adapt_radius", DEFAULT_NEAR_RADIUS)
    policy.knn_summary_radius = float(radius if radius is not None else DEFAULT_NEAR_RADIUS)
    if hasattr(policy, '_sync_knn_buffers'):
        policy._sync_knn_buffers()


def pack_crowd_summary(
    spatial_edges: np.ndarray,
    radius: float = DEFAULT_NEAR_RADIUS,
    n_ref: float = 20.0,
) -> np.ndarray:
    """One 6-dim crowd-summary token: count / mean relative pos / occupancy / min dist.

    Layout: ``[N/n_ref, mean_dx, mean_dy, n_near/n_ref, min_dist, 1.0]``.
    """
    n_humans = spatial_edges.shape[-2]
    n_near = count_n_near(spatial_edges, radius)
    mean_xy = spatial_edges[..., :2].mean(axis=-2)
    min_dist = np.hypot(spatial_edges[..., 0], spatial_edges[..., 1]).min(axis=-1)
    ones = np.ones_like(min_dist, dtype=np.float32)
    n_all = np.full_like(min_dist, float(n_humans), dtype=np.float32)
    token = np.stack(
        [
            n_all / float(n_ref),
            mean_xy[..., 0],
            mean_xy[..., 1],
            n_near / float(n_ref),
            min_dist.astype(np.float32),
            ones,
        ],
        axis=-1,
    )
    return token.astype(np.float32)
