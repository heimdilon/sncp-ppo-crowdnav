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
v34 = np.array([96.8, 92.8, 91.2, 86.0])   # CLEAN Beta (fb0bf07 fix); eski buggy Normal(a,b)=[83.2,71.6,56.4,44.8]
v35 = np.array([96.8, 89.6, 79.2, 69.6])
# collision
v26c = np.array([25.2, 38.4, 46.8, 56.4])
v27c = np.array([6.8, 20.0, 29.6, 40.4])
v28c = np.array([5.6, 12.4, 20.8, 27.2])
v30c = np.array([2.8, 10.4, 14.4, 20.8])
v31c = np.array([7.2, 16.0, 21.2, 27.6])
v32c = np.array([2.8, 6.0, 13.2, 21.2])
v33c = np.array([4.8, 15.6, 24.0, 29.2])
v34c = np.array([2.8, 7.2, 8.8, 13.2])     # CLEAN Beta carpisma; eski buggy=[12.0,21.2,34.4,47.2]
v35c = np.array([2.8, 10.0, 20.8, 30.4])
# v38 = locked v34 Beta policy + training-free action shield (honest 5-seed pool; v38_multiseed_result.json)
v38 = np.array([99.6, 99.6, 99.6, 98.8])
v38c = np.array([0.0, 0.0, 0.4, 0.4])

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
box(0.1, 4.7, 1.7, 0.7, "Robot düğümü\n(7-b)", LB)
box(0.1, 3.0, 1.7, 0.7, "Zamansal kenar\n(2-b)", LB)
box(0.1, 1.3, 1.7, 0.7, "Uzaysal kenar\n(6-b × N)", LB)
box(2.3, 4.7, 1.7, 0.7, "Robot MLP\n7→128 (Dnk 14)", LG)
box(2.3, 3.0, 1.7, 0.7, "ön-MLP\n2→256 (Dnk 11)", HL, ec=RED, tc="#7a3b00", lw=2.0)
box(2.3, 1.3, 1.7, 0.7, "ön-MLP\n6→256 (Dnk 11)", HL, ec=RED, tc="#7a3b00", lw=2.0)
box(4.5, 3.0, 1.6, 0.7, "Zamansal NCP\n(LTC)", LG)
box(4.5, 1.3, 1.6, 0.7, "Uzaysal NCP\n(LTC)", LG)
box(6.5, 1.55, 1.6, 1.5, "Dikkat havuzu\n(Dnk 13)\n  +  \nmean+max\n(v30)", HP, ec=PURPLE, tc=PURPLE, lw=2.0)
box(8.3, 3.0, 1.55, 0.9, "Düğüm NCP\n(füzyon)", LG)
box(8.3, 1.3, 1.55, 0.9, "Aktör-Kritik\n(ω, v)+V (Dnk 16)", LB)
arrow(1.8, 5.05, 2.3, 5.05); arrow(1.8, 3.35, 2.3, 3.35); arrow(1.8, 1.65, 2.3, 1.65)
arrow(4.0, 3.35, 4.5, 3.35); arrow(4.0, 1.65, 4.5, 1.65)
arrow(6.1, 3.2, 8.3, 3.45)
arrow(6.1, 1.65, 6.5, 2.0)
arrow(8.1, 2.3, 8.5, 3.0)
arrow(4.0, 4.9, 8.3, 3.75)
arrow(9.05, 3.0, 9.05, 2.2)
ax.text(3.15, 0.72, "Kırmızı: Dnk 11 ön-MLP gömme (v27)", color=RED, fontsize=9.5,
        ha="center", fontweight="bold")
ax.text(7.3, 0.72, "Mor: mean+max havuzlama (v30, katkımız)", color=PURPLE, fontsize=9.5,
        ha="center", fontweight="bold")
ax.set_title("SNCP-PPO mimarisi: v30 sadakat eklentileri ve mimari katkı vurgulu", fontsize=12)
fig.savefig(OUT + "ieee_arch.png"); plt.close(fig)

# ============ FIG: fidelity restorations (success + collision) ============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.1))
for ax, data, paper, ttl, ylab in [
    (a1, [v26, v27, v28, v30], PAPER_S, "Başarı: sadakat düzeltmeleri makaleye tırmanıyor", "Başarı (%)"),
    (a2, [v26c, v27c, v28c, v30c], PAPER_C, "Çarpışma: aynı düzeltmeler çarpışmayı düşürür", "Çarpışma (%)"),
]:
    labs = ["v26 (taban)", "v27 +ön-MLP", "v28 +müfredat", "v30 +mean+max (katkımız)"]
    cols = [GRAY, AMBER, "#2C7FB8", BLUE]
    lws = [1.6, 1.8, 1.8, 2.6]
    mss = [6, 6, 6, 8]
    for d, lab, c, lw, ms in zip(data, labs, cols, lws, mss):
        ax.plot(N, d, "-o", color=c, lw=lw, ms=ms, label=lab)
    ax.axhline(paper, color=GREEN, ls="--", lw=2, label="Makale (challenging)")
    ax.set_xlabel("Yaya sayısı N"); ax.set_ylabel(ylab)
    ax.set_xticks(N); ax.set_title(ttl, fontsize=10.5)
a1.set_ylim(40, 100); a1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
a2.set_ylim(0, 60); a2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT + "ieee_fidelity.png"); plt.close(fig)

# ============ FIG: ablations (delta vs v30 at high N) — v34 is the LONE positive-direction lever ============
# Five v30-based levers; the sixth (v29) is v28-based and discussed in text, not plotted here.
levers = ["v31\nnode-kap.", "v32\ncurric.\nN→25", "v33\n4-baş\nMHA", "v34\nBeta", "v35\nsense\n-6m"]
S = np.array([v31, v32, v33, v34, v35])
C = np.array([v31c, v32c, v33c, v34c, v35c])
dS15 = S[:, 2] - v30[2]
dS20 = S[:, 3] - v30[3]
dC15 = C[:, 2] - v30c[2]
dC20 = C[:, 3] - v30c[3]
x = np.arange(len(levers)); w = 0.38
V34 = 3  # index of v34 = the only positive-direction lever
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.1))
for ax in (a1, a2):
    ax.axvspan(V34 - 0.5, V34 + 0.5, color=GREEN, alpha=0.09, zorder=0)
a1.bar(x - w / 2, dS15, w, color="#2C7FB8", label="N=15")
a1.bar(x + w / 2, dS20, w, color=BLUE, label="N=20")
a1.axhline(0, color="k", lw=1)
a1.set_title("Başarı farkı vs v30 (yüksek-N): yalnız v34 pozitif", fontsize=10.5)
a1.set_ylabel("Δ başarı (pp)"); a1.set_xticks(x); a1.set_xticklabels(levers, fontsize=8.5)
a1.annotate("yön-doğru\n(Bonferroni-altı)", xy=(V34 + w / 2, dS20[V34]),
            xytext=(V34 - 0.45, dS20[V34] + 4.0), color=GREEN, fontsize=8, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
a1.legend(fontsize=9, framealpha=0.92)
a2.bar(x - w / 2, dC15, w, color="#E08214", label="N=15")
a2.bar(x + w / 2, dC20, w, color=RED, label="N=20")
a2.axhline(0, color="k", lw=1)
a2.set_title("Çarpışma farkı vs v30 (yüksek-N): yalnız v34 negatif (iyi)", fontsize=10.5)
a2.set_ylabel("Δ çarpışma (pp)"); a2.set_xticks(x); a2.set_xticklabels(levers, fontsize=8.5)
a2.legend(fontsize=9, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT + "ieee_ablations.png"); plt.close(fig)

# ============ FIG: sense-range (v35) widens the gap ============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.1))
def se95(arr):
    p = np.asarray(arr) / 100.0
    return 1.96 * np.sqrt(p * (1.0 - p) / 250.0) * 100.0
a1.errorbar(N, v30, yerr=se95(v30), fmt="-o", color=BLUE, lw=2.4, ms=8, capsize=3, label="v30 (tüm yayalar)")
a1.errorbar(N, v35, yerr=se95(v35), fmt="--s", color=RED, lw=2.2, ms=8, capsize=3, label="v35 (6m maske = makale bütçesi)")
a1.axhline(PAPER_S, color=GREEN, ls=":", lw=2, label="Makale (~6m) ~%94")
for xx, a, b in zip(N, v30, v35):
    if abs(a - b) >= 1.0:
        a1.annotate(f"{b-a:+.1f}", (xx, b), textcoords="offset points", xytext=(0, -14),
                    ha="center", color=RED, fontsize=9, fontweight="bold")
a1.set_title("Başarı: 6m algıya inmek açığı GENİŞLETİR", fontsize=10.5)
a1.set_xlabel("Yaya sayısı N"); a1.set_ylabel("Başarı (%)"); a1.set_xticks(N); a1.set_ylim(55, 100)
a1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
a2.errorbar(N, v30c, yerr=se95(v30c), fmt="-o", color=BLUE, lw=2.4, ms=8, capsize=3, label="v30 (tüm yayalar)")
a2.errorbar(N, v35c, yerr=se95(v35c), fmt="--s", color=RED, lw=2.2, ms=8, capsize=3, label="v35 (6m maske)")
a2.axhline(PAPER_C, color=GREEN, ls=":", lw=2, label="Makale ~%4")
a2.set_title("Çarpışma: maske yüksek-N'de çarpışmayı artırır", fontsize=10.5)
a2.set_xlabel("Yaya sayısı N"); a2.set_ylabel("Çarpışma (%)"); a2.set_xticks(N); a2.set_ylim(0, 35)
a2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT + "ieee_sense.png"); plt.close(fig)

# ============ FIG: selection bias -> multi-seed (REAL v30 N=10 seed blocks, no fabricated data) ============
import json as _json
_v30 = _json.load(open("v30_multiseed_result.json"))
blocks10 = np.array(_v30["10"]["block_means"]) * 100.0     # GERCEK per-seed: [88,88,86,94,92]
mean10 = _v30["10"]["pooled_success"] * 100.0              # GERCEK havuz: 89.6
hw10 = 1.96 * _v30["10"]["pooled_se"] * 100.0              # GERCEK +-1.96*SE (~3.8pp)
bestof = float(blocks10.max())                             # secim-sapmasi: bloklarin maksimumu (94)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
xj = np.random.RandomState(0).normal(1.0, 0.045, len(blocks10))
ax.fill_between([0.6, 1.4], mean10 - hw10, mean10 + hw10, color=BLUE, alpha=0.15, zorder=0,
                label="dürüst ortalama ±1.96·SE")
ax.scatter(xj, blocks10, s=85, color=GRAY, edgecolor="k", zorder=3,
           label="bağımsız tohum blokları (5×50 ep)")
ax.hlines(mean10, 0.6, 1.4, color=BLUE, lw=2.5, zorder=4, label=f"dürüst ortalama %{mean10:.1f}")
ax.scatter([1.0], [bestof], s=240, marker="*", color=RED, edgecolor="k", zorder=6,
           label=f"5 değerlendirmenin maksimumu %{bestof:.0f}")
ax.annotate(f"+{bestof - mean10:.1f} pp seçim sapması\n(eğitim onlarca eval yapar →\nkaydedilen 'best' daha da yüksek)",
            xy=(1.0, bestof), xytext=(1.14, mean10 - 3.0), color=RED, fontsize=8.3,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax.set_xlim(0.55, 1.9); ax.set_ylim(82, 98); ax.set_xticks([])
ax.set_ylabel("N=10 challenging başarı (%)")
ax.set_title("Neden çok-tohum: 'best holdout' seçim-saplı", fontsize=11)
ax.legend(loc="lower right", fontsize=8.0, framealpha=0.95)
fig.savefig(OUT + "ieee_seedbias.png"); plt.close(fig)

# ============ FIG: v38 action shield (v30 / v34 raw / v38 shield) ============
def _se95(arr):
    p = np.asarray(arr) / 100.0
    return 1.96 * np.sqrt(p * (1.0 - p) / 250.0) * 100.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.1))
a1.plot(N, v30, "-o", color=GRAY, lw=1.8, ms=6, label="v30 (taban politika)")
a1.plot(N, v34, "--s", color=AMBER, lw=1.8, ms=6, label="v34 (Beta politika)")
a1.errorbar(N, v38, yerr=_se95(v38), fmt="-D", color=GREEN, lw=2.6, ms=8, capsize=3, label="v38 (v34 + eylem kalkanı)")
a1.axhline(PAPER_S, color="#888888", ls=":", lw=1.5, label="Makale ~%94")
a1.set_title("Başarı: eğitimsiz eylem kalkanı yüksek-N'i kurtarır", fontsize=10.5)
a1.set_xlabel("Yaya sayısı N"); a1.set_ylabel("Başarı (%)"); a1.set_xticks(N); a1.set_ylim(65, 101)
a1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
a2.plot(N, v30c, "-o", color=GRAY, lw=1.8, ms=6, label="v30")
a2.plot(N, v34c, "--s", color=AMBER, lw=1.8, ms=6, label="v34")
a2.errorbar(N, v38c, yerr=_se95(v38c), fmt="-D", color=GREEN, lw=2.6, ms=8, capsize=3, label="v38 (kalkan)")
a2.set_title("Çarpışma: kalkan çarpışmayı ≈%0'a indirir", fontsize=10.5)
a2.set_xlabel("Yaya sayısı N"); a2.set_ylabel("Çarpışma (%)"); a2.set_xticks(N); a2.set_ylim(-1, 32)
a2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
fig.tight_layout()
fig.savefig(OUT + "ieee_v38.png"); plt.close(fig)

print("IEEE figures written to", OUT)
import os
for f in sorted(os.listdir(OUT)):
    if f.startswith("ieee_"):
        print("  ", f, os.path.getsize(OUT + f), "bytes")
