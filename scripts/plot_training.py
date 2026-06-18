"""Plot training & holdout metrics from a sncp_ppo training CSV.

Usage:
    python plot_training.py --csv logs/training_YYYYMMDD_HHMMSS.csv
                            --output training_curves.png
                            --window 50

The CSV is produced by sncp_ppo/train.py and contains one row per training
episode plus the most-recent holdout result snapshotted on that row.

Supports both:
  - Legacy single-scenario format: holdout_{success,collision,timeout,reward}
  - New multi-scenario format:     holdout_<scenario>_{success,collision,...}
"""
import os as _os, sys as _sys  # repo-root path bootstrap (run standalone: python scripts/X.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import csv
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_csv(path):
    """Read training CSV into a dict-of-lists (one list per column)."""
    if not os.path.exists(path):
        sys.exit(f"CSV not found: {path}")
    cols = None
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for row in reader:
            rows.append(row)
    data = {k: [] for k in cols}
    for row in rows:
        for k in cols:
            data[k].append(row[k])
    return data


def to_float_or_nan(values):
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        try:
            out[i] = float(v)
        except (ValueError, TypeError):
            out[i] = np.nan
    return out


def rolling_mean(x, window):
    """Right-aligned rolling mean. First (window-1) entries are NaN."""
    out = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        out[i] = np.nanmean(x[i - window + 1:i + 1])
    return out


def holdout_eval_points(episodes, holdout_values):
    """Return only the episodes where the holdout value *changed* from the
    previous row — i.e., the actual evaluation points (not the snapshots
    repeated on every row)."""
    eps = []
    vals = []
    prev = None
    for ep, v in zip(episodes, holdout_values):
        if np.isnan(v):
            continue
        if prev is None or v != prev:
            eps.append(ep)
            vals.append(v)
            prev = v
    return np.array(eps), np.array(vals)


def discover_holdout_scenarios(data_keys):
    """Find holdout scenarios by scanning column names.

    Returns:
      list of (scenario_name, {metric: col_name}) tuples.
      Empty list if no holdout columns.

    Handles both new format (holdout_<sc>_success) and legacy
    (holdout_success → scenario name 'holdout').
    """
    metrics = ('success', 'collision', 'timeout', 'reward')
    # Look for "holdout_<scenario>_<metric>" pattern
    pat = re.compile(r'^holdout_(.+)_(' + '|'.join(metrics) + r')$')
    found = {}  # scenario -> {metric: col_name}
    for key in data_keys:
        m = pat.match(key)
        if m:
            sc, metric = m.group(1), m.group(2)
            found.setdefault(sc, {})[metric] = key

    # Legacy fallback: holdout_success etc. (no scenario in name)
    if not found:
        legacy_present = all(f'holdout_{m}' in data_keys for m in metrics)
        if legacy_present:
            found['holdout'] = {m: f'holdout_{m}' for m in metrics}

    # Keep only scenarios with all 4 metrics
    scenarios = []
    for sc, metric_map in found.items():
        if all(m in metric_map for m in metrics):
            scenarios.append((sc, metric_map))
    return scenarios


def scenario_boundaries(episodes, scenarios):
    """Find the first episode of each scenario phase (in order)."""
    boundaries = []
    prev = None
    for ep, sc in zip(episodes, scenarios):
        if sc != prev:
            boundaries.append((ep, sc))
            prev = sc
    return boundaries


SCENARIO_COLORS = {
    'easy':      '#2ca02c',  # green — easiest
    'easy_plus': '#9ecae1',  # light blue
    'medium':    '#ff7f0e',  # orange
    'hard':      '#d62728',  # red — primary holdout
    'extreme':   '#8c564b',  # brown
    'circle':    '#d62728',
    'random':    '#8c564b',
    'holdout':   '#d62728',  # legacy
}

PHASE_BG = {
    'easy':      '#d4edda',
    'easy_plus': '#cce5e8',
    'medium':    '#fff3cd',
    'hard':      '#f8d7da',
    'extreme':   '#d1ecf1',
    'circle':    '#e2e3e5',
    'random':    '#e2e3e5',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Training CSV path.')
    parser.add_argument('--output', type=str, default='training_curves.png')
    parser.add_argument('--window', type=int, default=50, help='Rolling-mean window for training metrics.')
    args = parser.parse_args()

    data = read_csv(args.csv)
    episodes = to_float_or_nan(data['episode']).astype(int)
    scenarios = data['scenario']

    train_success = to_float_or_nan(data['success'])
    train_collision = to_float_or_nan(data['collision'])
    train_reward = to_float_or_nan(data['reward'])

    # Auto-discover holdout columns (handles legacy + multi-scenario)
    holdout_scs = discover_holdout_scenarios(list(data.keys()))
    print(f"Detected holdout scenarios: {[sc for sc, _ in holdout_scs]}")

    # Rolling means for training (noisy per-episode → smooth window)
    w = min(args.window, max(1, len(episodes) // 4))
    train_success_smooth = rolling_mean(train_success, w)
    train_collision_smooth = rolling_mean(train_collision, w)
    train_reward_smooth = rolling_mean(train_reward, w)

    # Per-scenario holdout extracted as event points (changed values only)
    holdout_data = {}  # sc -> {metric: (eps_arr, vals_arr)}
    for sc, metric_map in holdout_scs:
        d = {}
        for metric in ('success', 'collision', 'reward'):
            vals = to_float_or_nan(data[metric_map[metric]])
            d[metric] = holdout_eval_points(episodes, vals)
        holdout_data[sc] = d

    # Generalist min line: for each episode where ALL holdouts have a value
    if len(holdout_scs) >= 2:
        per_sc_success_at_eps = {}
        for sc, metric_map in holdout_scs:
            vals = to_float_or_nan(data[metric_map['success']])
            per_sc_success_at_eps[sc] = vals
        min_success = np.nanmin(
            np.stack([per_sc_success_at_eps[sc] for sc, _ in holdout_scs]),
            axis=0,
        )
        min_eps, min_vals = holdout_eval_points(episodes, min_success)
    else:
        min_eps, min_vals = np.array([]), np.array([])

    # Curriculum boundaries
    boundaries = scenario_boundaries(episodes, scenarios)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # Shade curriculum phases
    bounds_with_end = boundaries + [(int(episodes[-1]) + 1, None)]
    for ax in axes:
        for i in range(len(bounds_with_end) - 1):
            start_ep, sc = bounds_with_end[i]
            end_ep, _ = bounds_with_end[i + 1]
            ax.axvspan(start_ep, end_ep, color=PHASE_BG.get(sc, '#f0f0f0'),
                       alpha=0.4, zorder=0)

    # --- Plot 1: success rate
    ax = axes[0]
    ax.plot(episodes, train_success_smooth,
            label=f'Train success ({w}-ep rolling)', color='#1f77b4', linewidth=1.6)
    for sc, _ in holdout_scs:
        eps_arr, vals_arr = holdout_data[sc]['success']
        if len(eps_arr) > 0:
            ax.plot(eps_arr, vals_arr, 'o-',
                    label=f'Holdout {sc}',
                    color=SCENARIO_COLORS.get(sc, '#777'),
                    markersize=5, linewidth=1.4, alpha=0.9)
    if len(min_eps) > 0 and len(holdout_scs) >= 2:
        ax.plot(min_eps, min_vals, 'k--',
                label='min(holdout) — best-ckpt metric',
                linewidth=1.8, alpha=0.85)
    ax.set_ylabel('Success Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('SNCP-PPO Training Progress')

    # --- Plot 2: collision rate
    ax = axes[1]
    ax.plot(episodes, train_collision_smooth,
            label=f'Train collision ({w}-ep rolling)', color='#ff7f0e', linewidth=1.6)
    for sc, _ in holdout_scs:
        eps_arr, vals_arr = holdout_data[sc]['collision']
        if len(eps_arr) > 0:
            ax.plot(eps_arr, vals_arr, 's-',
                    label=f'Holdout {sc} collision',
                    color=SCENARIO_COLORS.get(sc, '#777'),
                    markersize=5, linewidth=1.4, alpha=0.9)
    ax.set_ylabel('Collision Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # --- Plot 3: reward
    ax = axes[2]
    ax.plot(episodes, train_reward_smooth,
            label=f'Train reward ({w}-ep rolling)', color='#2ca02c', linewidth=1.6)
    for sc, _ in holdout_scs:
        eps_arr, vals_arr = holdout_data[sc]['reward']
        if len(eps_arr) > 0:
            ax.plot(eps_arr, vals_arr, '^-',
                    label=f'Holdout {sc} reward',
                    color=SCENARIO_COLORS.get(sc, '#777'),
                    markersize=5, linewidth=1.4, alpha=0.9)
    ax.set_ylabel('Episode Reward')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    # Curriculum legend strip
    phase_patches = []
    seen = set()
    for ep, sc in boundaries:
        if sc in seen:
            continue
        seen.add(sc)
        phase_patches.append(mpatches.Patch(color=PHASE_BG.get(sc, '#f0f0f0'),
                                            alpha=0.5, label=f'{sc} (from ep {ep})'))
    if phase_patches:
        # Add to a second legend on the first axis
        from matplotlib.legend import Legend
        leg = Legend(axes[0], handles=phase_patches, labels=[p.get_label() for p in phase_patches],
                     loc='upper right', fontsize=8, title='Curriculum')
        axes[0].add_artist(leg)

    plt.tight_layout()
    plt.savefig(args.output, dpi=140, bbox_inches='tight')
    print(f"Saved plot to {args.output}")

    # ---- Summary ----
    print("\nSummary:")
    final_train_succ = np.nanmean(train_success[-min(100, len(train_success)):])
    print(f"  Final 100-ep training success:   {final_train_succ:.1%}")

    for sc, _ in holdout_scs:
        eps_arr, vals_arr = holdout_data[sc]['success']
        if len(vals_arr) > 0:
            best_idx = int(np.nanargmax(vals_arr))
            print(f"  Holdout {sc}: best={np.nanmax(vals_arr):.1%} (ep {int(eps_arr[best_idx])}), "
                  f"last={vals_arr[-1]:.1%} (ep {int(eps_arr[-1])})")

    if len(min_vals) > 0:
        best_min_idx = int(np.nanargmax(min_vals))
        print(f"  Best generalist min:             {np.nanmax(min_vals):.1%} "
              f"(at ep {int(min_eps[best_min_idx])})")

    print(f"  Total episodes:                  {len(episodes)}")
    print(f"  Curriculum phases visited:       {[sc for _, sc in boundaries]}")


if __name__ == '__main__':
    main()
