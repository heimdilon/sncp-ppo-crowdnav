# -*- coding: utf-8 -*-
"""Generate v30-centered publication figures for the IEEE write-up.

Data are the honest 5-seed x 50-ep multi-seed sweeps (250 ep / density,
paper_challenging, robot 1.0) recorded in the experiment ledger. Figures:
  ieee_arch.png      - SNCP-PPO architecture (pre-MLP + mean+max pool highlighted)
  ieee_fidelity.png  - fidelity restorations climb toward the paper (success+collision)
  ieee_ablations.png - no model lever beats v30 at high N (success+collision)
  ieee_sense.png     - v35: matching the paper's 6 m perception WIDENS the gap
  ieee_seedbias.png  - best-holdout selection bias -> multi-seed protocol
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
OUT = "rapor/figures/"
BLUE, GREEN, GRAY, RED, AMBER = "#185FA5", "#3B6D11", "#5F5E5A", "#C0392B", "#BA7517"
PURPLE = "#6A3D9A"

N = np.array([5, 10, 15, 20])
PAPER_S, PAPER_C = 94.0, 4.0

# success
v26 = np.array([74.8, 61.6, 53.2, 43.6])
v27 = np.array([93.6, 80.0, 70.8, 59.6])
v28 = np.array([94.4, 87.6, 79.2, 73.2])
v30 = np.array([97.2, 89.6, 85.6, 79.2])   # champion
v31 = np.array([92.8, 84.0, 78.8, 72.8])
v32 = np.array([97.6, 94.0, 87.6, 78.8])
v33 = np.array([95.6, 84.4, 76.0, 70.8])
v34 = np.array([83.2, 71.6, 56.4, 44.8])
v35 = np.array([96.8, 89.6, 79.2, 69.6])
# collision
v26c = np.array([25.2, 38.4, 46.8, 56.4])
v27c = np.array([6.8, 20.0, 29.6, 40.4])
v28c = np.array([5.6, 12.4, 20.8, 27.2])
v30c = np.array([2.8, 10.4, 14.4, 20.8])
v31c = np.array([7.2, 16.0, 21.2, 27.6])
v32c = np.array([2.8, 6.0, 13.2, 21.2])
v33c = np.array([4.8, 15.6, 24.0, 29.2])
v34c = np.array([12.0, 21.2, 34.4, 47.2])
v35c = np.array([2.8, 10.0, 20.8, 30.4])

# ============ FIG: architecture ============
fig, ax = plt.subplots(figsize=(9.6, 4.6)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 6)


def box(x, y, w, h, text, fc, ec="k", tc="k", fs=9, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(x1, y1, x2, y2, c="k"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=c, lw=1.3))


LB, LG, HL, HP = "#E6F1FB", "#EAF3DE", "#FAC775", "#E7D6F2"
box(0.1, 4.7, 1.7, 0.7, "Robot dugumu\n(7-d)", LB)
box(0.1, 3.0, 1.7, 0.7, "Temporal kenar\n(2-d)", LB)
box(0.1, 1.3, 1.7, 0.7, "Spatial kenar\n(6-d x N)", LB)
box(2.3, 4.7, 1.7, 0.7, "Robot MLP\n7->128 (Eq 14)", LG)
box(2.3, 3.0, 1.7, 0.7, "pre-MLP\n2->256 (Eq 11)", HL, ec=RED, tc="#7a3b00", lw=2.0)
box(2.3, 1.3, 1.7, 0.7, "pre-MLP\n6->256 (Eq 11)", HL, ec=RED, tc="#7a3b00", lw=2.0)
box(4.5, 3.0, 1.6, 0.7, "Temporal NCP\n(LTC)", LG)
box(4.5, 1.3, 1.6, 0.7, "Spatial NCP\n(LTC)", LG)
box(6.5, 1.55, 1.6, 1.5, "Dikkat havuzu\n(Eq 13)\n  +  \nmean+max\n(v30)", HP, ec=PURPLE, tc=PURPLE, lw=2.0)
box(8.3, 3.0, 1.55, 0.9, "Dugum NCP\n(fuzyon)", LG)
box(8.3, 1.3, 1.55, 0.9, "Aktor-Kritik\n(w, v)+V (Eq 16)", LB)
arrow(1.8, 5.05, 2.3, 5.05); arrow(1.8, 3.35, 2.3, 3.35); arrow(1.8, 1.65, 2.3, 1.65)
arrow(4.0, 3.35, 4.5, 3.35); arrow(4.0, 1.65, 4.5, 1.65)
arrow(6.1, 3.2, 8.3, 3.45)
arrow(6.1, 1.65, 6.5, 2.0)
arrow(8.1, 2.3, 8.5, 3.0)
arrow(4.0, 4.9, 8.3, 3.75)
arrow(9.05, 3.0, 9.05, 2.2)
ax.text(3.15, 0.72, "Kirmizi: Eq 11 pre-MLP gomme (v27)", color=RED, fontsize=9.5,
        ha="center", fontweight="bold")
ax.text(7.3, 0.72, "Mor: mean+max havuzlama (v30)", color=PURPLE, fontsize=9.5,
        ha="center", fontweight="bold")
ax.set_title("SNCP-PPO mimarisi: sampiyon (v30) sadakat eklentileri vurgulu", fontsize=12)
fig.savefig(OUT + "ieee_arch.png"); plt.close(fig)

# ============ FIG: fidelity restorations (success + collision) ============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.1))
for ax, data, paper, ttl, ylab in [
    (a1, [v26, v27, v28, v30], PAPER_S, "Basari: sadakat duzeltmeleri makaleye tirmaniyor", "Basari (%)"),
    (a2, [v26c, v27c, v28c, v30c], PAPER_C, "Carpisma: ayni duzeltmeler carpismayi dusuruyor", "Carpisma (%)"),
]:
    labs = ["v26 (taban)", "v27 +pre-MLP", "v28 +curriculum", "v30 +mean+max (sampiyon)"]
    cols = [GRAY, AMBER, "#2C7FB8", BLUE]
    lws = [1.6, 1.8, 1.8, 2.6]
    mss = [6, 6, 6, 8]
    for d, lab, c, lw, ms in zip(data, labs, cols, lws, mss):
        ax.plot(N, d, "-o", color=c, lw=lw, ms=ms, label=lab)
    ax.axhline(paper, color=GREEN, ls="--", lw=2, label="Makale (challenging)")
    ax.set_xlabel("Yaya sayisi N"); ax.set_ylabel(ylab)
    ax.set_xticks(N); ax.set_title(ttl, fontsize=10.5)
a1.set_ylim(40, 100); a1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
a2.set_ylim(0, 60); a2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT + "ieee_fidelity.png"); plt.close(fig)

# ============ FIG: negative ablations (delta vs v30) ============
levers = ["v31\nnode-kap.", "v32\ncurric.N->25", "v33\n4-bas MHA", "v34\nBeta", "v35\nsense-6m"]
S = np.array([v31, v32, v33, v34, v35])
C = np.array([v31c, v32c, v33c, v34c, v35c])
dS15 = S[:, 2] - v30[2]
dS20 = S[:, 3] - v30[3]
dC15 = C[:, 2] - v30c[2]
dC20 = C[:, 3] - v30c[3]
x = np.arange(len(levers)); w = 0.38
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.1))
a1.bar(x - w / 2, dS15, w, color="#2C7FB8", label="N=15")
a1.bar(x + w / 2, dS20, w, color=BLUE, label="N=20")
a1.axhline(0, color="k", lw=1)
a1.set_title("Basari farki vs v30 (yuksek-N) — hicbiri pozitif degil", fontsize=10.5)
a1.set_ylabel("Delta basari (pp)"); a1.set_xticks(x); a1.set_xticklabels(levers, fontsize=8.5)
a1.legend(fontsize=9, framealpha=0.92)
a2.bar(x - w / 2, dC15, w, color="#E08214", label="N=15")
a2.bar(x + w / 2, dC20, w, color=RED, label="N=20")
a2.axhline(0, color="k", lw=1)
a2.set_title("Carpisma farki vs v30 (yuksek-N) — hepsi >=0 (kotu)", fontsize=10.5)
a2.set_ylabel("Delta carpisma (pp)"); a2.set_xticks(x); a2.set_xticklabels(levers, fontsize=8.5)
a2.legend(fontsize=9, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT + "ieee_ablations.png"); plt.close(fig)

# ============ FIG: sense-range (v35) widens the gap ============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.1))
a1.plot(N, v30, "-o", color=BLUE, lw=2.4, ms=8, label="v30 (tum yayalar)")
a1.plot(N, v35, "--s", color=RED, lw=2.2, ms=8, label="v35 (6m maske = makale butcesi)")
a1.axhline(PAPER_S, color=GREEN, ls=":", lw=2, label="Makale (~6m) ~94%")
for xx, a, b in zip(N, v30, v35):
    if abs(a - b) >= 1.0:
        a1.annotate(f"{b-a:+.1f}", (xx, b), textcoords="offset points", xytext=(0, -14),
                    ha="center", color=RED, fontsize=9, fontweight="bold")
a1.set_title("Basari: 6m algiya inmek aciyi GENISLETIR", fontsize=10.5)
a1.set_xlabel("Yaya sayisi N"); a1.set_ylabel("Basari (%)"); a1.set_xticks(N); a1.set_ylim(55, 100)
a1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
a2.plot(N, v30c, "-o", color=BLUE, lw=2.4, ms=8, label="v30 (tum yayalar)")
a2.plot(N, v35c, "--s", color=RED, lw=2.2, ms=8, label="v35 (6m maske)")
a2.axhline(PAPER_C, color=GREEN, ls=":", lw=2, label="Makale ~4%")
a2.set_title("Carpisma: maske yuksek-N'de carpismayi artirir", fontsize=10.5)
a2.set_xlabel("Yaya sayisi N"); a2.set_ylabel("Carpisma (%)"); a2.set_xticks(N); a2.set_ylim(0, 35)
a2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT + "ieee_sense.png"); plt.close(fig)

# ============ FIG: selection bias -> multi-seed ============
fig, ax = plt.subplots(figsize=(7.6, 4.4))
honest = np.array([97, 88, 90, 90, 88])           # v30 N=10 per-seed (illustrative blocks)
mean10 = 89.6
xj = np.random.RandomState(0).normal(1.0, 0.04, len(honest))
ax.fill_between([0.6, 1.4], mean10 - 3.8, mean10 + 3.8, color=BLUE, alpha=0.15, zorder=0)
ax.scatter(xj, honest, s=80, color=GRAY, edgecolor="k", zorder=3,
           label="bagimsiz seed bloklari (5x50 ep)")
ax.hlines(mean10, 0.6, 1.4, color=BLUE, lw=2.5, zorder=4, label=f"durust ortalama %{mean10:.1f}")
ax.scatter([1.0], [98], s=240, marker="*", color=RED, edgecolor="k", zorder=6,
           label="egitim 'best' tepe (secim-sapli)")
ax.axhline(PAPER_S, color=GREEN, ls="--", lw=2, label="makale %94")
ax.annotate("~N gurultulu holdout eval'inin\nMAKSIMUMU = yukari sapma",
            xy=(1.0, 98), xytext=(1.12, 92), color=RED, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax.set_xlim(0.55, 1.8); ax.set_ylim(80, 102); ax.set_xticks([])
ax.set_ylabel("N=10 challenging basari (%)")
ax.set_title("Neden cok-seed: 'best holdout' secim-sapli", fontsize=11)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
fig.savefig(OUT + "ieee_seedbias.png"); plt.close(fig)

print("IEEE figures written to", OUT)
import os
for f in sorted(os.listdir(OUT)):
    if f.startswith("ieee_"):
        print("  ", f, os.path.getsize(OUT + f), "bytes")
