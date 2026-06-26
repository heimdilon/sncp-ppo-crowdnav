"""Generate V38-only figures for the final paper.

This script deliberately avoids the older v30/v34 ablation figures.  The paper
should read as the final V38 system report: locked v34 Beta policy plus a
training-free runtime action shield.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DENSITIES = [5, 10, 15, 20]
PAPER_TARGET = 94.0

V30_JSON = ROOT / "paper_data" / "v30_multiseed_result.json"
V34_JSON = ROOT / "paper_data" / "v34_multiseed_result.json"
V38_JSON = ROOT / "paper_data" / "v38_multiseed_result.json"
TRAJ_N10 = OUT / "v38_traj_n10.png"
TRAJ_N20 = OUT / "v38_traj_n20.png"

INK = "#263238"
MUTED = "#607D8B"
GRID = "#D7DEE2"
V30 = "#90A4AE"
V34 = "#5E81AC"
V38 = "#E76F51"
SUCCESS = "#2A9D8F"
COLLISION = "#E76F51"
TIMEOUT = "#7B8794"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def series(data: dict, key: str) -> np.ndarray:
    return np.array([100.0 * float(data[str(n)][key]) for n in DENSITIES])


def steps(data: dict) -> np.ndarray:
    return np.array([float(data[str(n)]["avg_success_steps"]) for n in DENSITIES])


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "-",
        }
    )


def save(fig: plt.Figure, name: str, *, pdf: bool = True) -> None:
    fig.savefig(OUT / f"{name}.png")
    if pdf:
        fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(6.75, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.035, 0.28, 0.20, 0.46, "Locked v34\nBeta policy", "SNCP--PPO weights\nunchanged", "#E8F1F8"),
        (0.30, 0.28, 0.17, 0.46, "Deterministic\naction", r"proposal $(v,\omega)$", "#F7F7F5"),
        (0.535, 0.28, 0.22, 0.46, "V38 action\nshield", "6-step CV risk check\nedit only unsafe actions", "#FDECEC"),
        (0.82, 0.28, 0.15, 0.46, "CrowdNav\nenvironment", "ORCA pedestrians\npaper_challenging", "#EAF6F0"),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.016,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#D0D6DA",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + 0.63 * h, title, ha="center", va="center", weight="bold", color=INK)
        ax.text(x + w / 2, y + 0.30 * h, subtitle, ha="center", va="center", fontsize=8.5, color=MUTED)

    arrow_kw = dict(arrowstyle="-|>", lw=1.7, color=INK, shrinkA=8, shrinkB=8, mutation_scale=12)
    ax.annotate("", xy=(0.30, 0.51), xytext=(0.235, 0.51), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.535, 0.51), xytext=(0.47, 0.51), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.82, 0.51), xytext=(0.755, 0.51), arrowprops=arrow_kw)
    ax.annotate(
        "observation",
        xy=(0.12, 0.24),
        xytext=(0.89, 0.24),
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color=MUTED, connectionstyle="arc3,rad=-0.22"),
    )
    ax.text(0.645, 0.80, "training-free safety layer", ha="center", va="center", color=V38, weight="bold")
    save(fig, "fig_v38_pipeline")


def fig_effect() -> None:
    v34 = load(V34_JSON)
    v38 = load(V38_JSON)
    x = np.arange(len(DENSITIES))
    width = 0.34
    before_success = series(v34, "pooled_success")
    after_success = series(v38, "pooled_success")
    before_collision = series(v34, "pooled_collision")
    after_collision = series(v38, "pooled_collision")

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75), sharex=True)
    for ax, before, after, title, ylabel, ymax in [
        (axes[0], before_success, after_success, "Success: V38 raises high-N reliability", "Success rate (%)", 104),
        (axes[1], before_collision, after_collision, "Collision: shield removes most failures", "Collision rate (%)", 16),
    ]:
        ax.bar(x - width / 2, before, width, label="Raw v34", color=V34, edgecolor="white", linewidth=0.8)
        ax.bar(x + width / 2, after, width, label="V38 shield", color=V38, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Number of humans")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in DENSITIES])
        ax.set_ylim(0, ymax)
        for xi, b, a in zip(x, before, after):
            delta = a - b
            ax.text(xi, max(b, a) + ymax * 0.035, f"{delta:+.1f}pp", ha="center", va="bottom", fontsize=8)
    axes[0].legend(loc="lower left")
    save(fig, "fig_v38_shield_effect")


def fig_final_sweep() -> None:
    v38 = load(V38_JSON)
    x = np.array(DENSITIES)
    succ = series(v38, "pooled_success")
    coll = series(v38, "pooled_collision")
    to = series(v38, "pooled_timeout")
    nav = steps(v38)

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75))
    ax = axes[0]
    ax.plot(x, succ, marker="o", color=SUCCESS, lw=2.2, label="Success")
    ax.plot(x, coll, marker="s", color=COLLISION, lw=2.0, label="Collision")
    ax.plot(x, to, marker="^", color=TIMEOUT, lw=1.8, label="Timeout")
    ax.axhline(PAPER_TARGET, color="#8C8C8C", lw=1.2, ls="--", label="Paper challenging target")
    ax.set_title("Final V38 rates")
    ax.set_xlabel("Number of humans")
    ax.set_ylabel("Episode rate (%)")
    ax.set_xticks(x)
    ax.set_ylim(0, 104)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.45))

    ax = axes[1]
    ax.plot(x, nav, marker="D", color=INK, lw=2.2)
    ax.set_title("Successful path length")
    ax.set_xlabel("Number of humans")
    ax.set_ylabel("Avg success steps")
    ax.set_xticks(x)
    ax.set_ylim(42, 64)
    for n, s in zip(x, nav):
        ax.text(n, s + 0.55, f"{s:.1f}", ha="center", va="bottom", fontsize=8)
    save(fig, "fig_v38_final_sweep")


def fig_context() -> None:
    v30 = load(V30_JSON)
    v34 = load(V34_JSON)
    v38 = load(V38_JSON)
    x = np.arange(len(DENSITIES))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75), sharex=True)
    for ax, key, title, ylabel, ymax in [
        (axes[0], "pooled_success", "Context: from learned policies to shielded system", "Success rate (%)", 104),
        (axes[1], "pooled_collision", "Failure mode removed at runtime", "Collision rate (%)", 24),
    ]:
        vals30 = series(v30, key)
        vals34 = series(v34, key)
        vals38 = series(v38, key)
        ax.bar(x - width, vals30, width, label="v30 learned", color=V30, edgecolor="white", linewidth=0.8)
        ax.bar(x, vals34, width, label="v34 Beta", color=V34, edgecolor="white", linewidth=0.8)
        ax.bar(x + width, vals38, width, label="v38 shield", color=V38, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Number of humans")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in DENSITIES])
        ax.set_ylim(0, ymax)
    axes[0].legend(loc="lower left", ncol=1)
    save(fig, "fig_v38_context")


def fig_trajectories() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.05))
    for ax, n, path in [(axes[0], 10, TRAJ_N10), (axes[1], 20, TRAJ_N20)]:
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(f"V38 shielded trajectory, N={n}")
        ax.axis("off")
    fig.text(
        0.5,
        0.02,
        "paper_challenging, locked v34 Beta checkpoint + V38 action shield",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "fig_v38_trajectories", pdf=False)


def main() -> None:
    style()
    fig_pipeline()
    fig_effect()
    fig_final_sweep()
    fig_context()
    fig_trajectories()
    print(f"Wrote V38 final figures to {OUT}")


if __name__ == "__main__":
    main()
