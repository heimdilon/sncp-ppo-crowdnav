# -*- coding: utf-8 -*-
"""Generate publication figures for the v27 status report."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
OUT = "rapor/figures/"

# ---- data (honest local multi-seed sweep, 5 seeds x 50 ep) ----
N = np.array([5, 10, 15, 20])
v26_mean = np.array([74.8, 61.6, 53.2, 43.6])
v26_se   = np.array([2.7, 3.1, 3.2, 3.1])          # pooled SE (pp)
v26_ci   = 1.96 * v26_se
v24_1seed = np.array([80, 56, 32, 16])
PAPER = 94.0
BLUE, GREEN, GRAY, RED, AMBER = "#185FA5", "#3B6D11", "#5F5E5A", "#C0392B", "#BA7517"

# ============ FIG 1: honest density curve + CI + v24 + paper ============
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.axhline(PAPER, color=GREEN, ls="--", lw=2, label="Makale (challenging ~%94)")
ax.fill_between(N, v26_mean - v26_ci, v26_mean + v26_ci, color=BLUE, alpha=0.18,
                label="v26 %95 güven aralığı")
ax.plot(N, v26_mean, "-o", color=BLUE, lw=2.4, ms=8, label="v26 dürüst ortalama (5 seed)")
ax.plot(N, v24_1seed, "--s", color=GRAY, lw=1.8, ms=7, label="v24 tek-seed")
for x, y in zip(N, v26_mean):
    ax.annotate(f"%{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", color=BLUE, fontsize=11, fontweight="bold")
# gap (a) at N=10
ax.annotate("", xy=(10, PAPER - 1), xytext=(10, v26_mean[1] + 1),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
ax.text(10.4, (PAPER + v26_mean[1]) / 2, "(a) 32 pp\ntemel açık\n(N=10, eğitilen yoğunluk)",
        color=RED, fontsize=10, va="center")
# gap (b) falloff
ax.annotate("(b) yüksek-N düşüşü\n%62 → %44", xy=(18, 47), xytext=(14.5, 30),
            color=AMBER, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6))
ax.set_xlabel("Yaya sayısı (paper_challenging, 50 s bütçe)")
ax.set_ylabel("Başarı oranı (%)")
ax.set_title("v26 dürüst yoğunluk taraması — makaleye karşı iki açık")
ax.set_xticks(N); ax.set_ylim(0, 100); ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
fig.savefig(OUT + "fig_density.png"); plt.close(fig)

# ============ FIG 2: selection-bias / seed variance at N=10 ============
fig, ax = plt.subplots(figsize=(8.2, 5.0))
honest = np.array([56, 66, 64, 58, 64])      # seeds 100-500
xj = np.random.RandomState(0).normal(1.0, 0.04, len(honest))
ax.fill_between([0.6, 1.4], 61.6 - 6.1, 61.6 + 6.1, color=BLUE, alpha=0.15, zorder=0)
ax.scatter(xj, honest, s=80, color=GRAY, edgecolor="k", zorder=3,
           label="bağımsız seed blokları (5×50 ep)")
ax.hlines(61.6, 0.6, 1.4, color=BLUE, lw=2.5, zorder=4, label="dürüst ortalama %61.6")
ax.scatter([1.0], [56], s=140, marker="v", color=BLUE, edgecolor="k", zorder=5,
           label="rapor edilen sweep (seed=100) %56")
ax.scatter([1.0], [76], s=240, marker="*", color=RED, edgecolor="k", zorder=6,
           label="eğitim 'best' tepe (seçim-saplı) %76")
ax.axhline(PAPER, color=GREEN, ls="--", lw=2, label="makale %94")
ax.annotate("~58 gürültülü holdout eval'inin\nMAKSİMUMU = yukarı sapma",
            xy=(1.0, 76), xytext=(1.15, 84), color=RED, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
ax.set_xlim(0.55, 1.75); ax.set_ylim(45, 100); ax.set_xticks([])
ax.set_ylabel("N=10 challenging başarı (%)")
ax.set_title("Neden çok-seed: 'best holdout' seçim-saplı, gerçek ≈ %62")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
fig.savefig(OUT + "fig_seedbias.png"); plt.close(fig)

# ============ FIG 3: v26 training curve (generalist-min progression) ============
steps, std_s, chal_s, best_flag = [], [], [], []
with open("training_20260616_144223.csv", newline="") as f:
    for r in csv.DictReader(f):
        s = r.get("holdout_paper_standard_success"); c = r.get("holdout_paper_challenging_success")
        try:
            sv = float(s); cv = float(c)
        except (TypeError, ValueError):
            continue
        if sv != sv or cv != cv:
            continue
        steps.append(float(r["episode"]) / 1e6); std_s.append(sv * 100); chal_s.append(cv * 100)
        best_flag.append(r.get("is_best_checkpoint") == "1")
steps = np.array(steps); std_s = np.array(std_s); chal_s = np.array(chal_s)
genmin = np.minimum(std_s, chal_s); best_flag = np.array(best_flag)
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(steps, std_s, color=GRAY, lw=1.3, alpha=0.8, label="standard holdout")
ax.plot(steps, chal_s, color=AMBER, lw=1.3, alpha=0.8, label="challenging holdout")
ax.plot(steps, genmin, color=BLUE, lw=2.2, label="generalist-min (best ölçütü)")
bx = steps[best_flag]; by = genmin[best_flag]
ax.scatter(bx, by, marker="*", s=140, color=GREEN, edgecolor="k", zorder=5,
           label="yeni 'best' checkpoint olayı")
# peak best vs final
pk = int(np.argmax(genmin))
ax.annotate(f"best tepe %{genmin[pk]:.0f}\n@ {steps[pk]:.2f}M",
            xy=(steps[pk], genmin[pk]), xytext=(steps[pk]-0.9, genmin[pk]+8),
            color=GREEN, fontsize=10, arrowprops=dict(arrowstyle="->", color=GREEN))
ax.annotate(f"final %{genmin[-1]:.0f}\n(best→final düşüş)",
            xy=(steps[-1], genmin[-1]), xytext=(steps[-1]-1.4, genmin[-1]-22),
            color=RED, fontsize=10, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("Eğitim adımı (milyon)"); ax.set_ylabel("Holdout başarı (%)")
ax.set_title("v26 eğitimi — sağlıklı tırmanış, sonra best→final degradasyon")
ax.set_ylim(0, 100); ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
fig.savefig(OUT + "fig_training.png"); plt.close(fig)

# ============ FIG 4: SNCP architecture (pre-MLP highlighted) ============
fig, ax = plt.subplots(figsize=(10.0, 5.2)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
def box(x, y, w, h, text, fc, ec="k", tc="k", fs=10, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc)
def arrow(x1, y1, x2, y2, c="k"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=c, lw=1.4))
LB, LG, HL = "#E6F1FB", "#EAF3DE", "#FAC775"
# inputs
box(0.1, 4.7, 1.7, 0.7, "Robot düğümü\n(7-d)", LB, fs=9)
box(0.1, 3.0, 1.7, 0.7, "Temporal kenar\n(2-d)", LB, fs=9)
box(0.1, 1.3, 1.7, 0.7, "Spatial kenar\n(6-d × N yaya)", LB, fs=9)
# robot mlp + pre-mlp (highlight)
box(2.3, 4.7, 1.7, 0.7, "Robot MLP\n7→128 (Eq 14)", LG, fs=9)
box(2.3, 3.0, 1.7, 0.7, "pre-MLP\n2→256 (Eq 11)", HL, ec=RED, tc="#7a3b00", fs=9, lw=2.2)
box(2.3, 1.3, 1.7, 0.7, "pre-MLP\n6→256 (Eq 11)", HL, ec=RED, tc="#7a3b00", fs=9, lw=2.2)
# ncp encoders
box(4.5, 3.0, 1.6, 0.7, "Temporal NCP\n(LTC)", LG, fs=9)
box(4.5, 1.3, 1.6, 0.7, "Spatial NCP\n(LTC)", LG, fs=9)
# attention
box(6.5, 1.95, 1.5, 0.9, "Attention\nhavuzlama\n(Eq 13, d_k=64)", "#F4C0D1", fs=9)
# node ncp
box(8.2, 3.0, 1.6, 0.9, "Düğüm NCP\n(füzyon 640)", LG, fs=9)
# actor critic
box(8.2, 1.3, 1.6, 0.9, "Actor–Critic\n(ω, v) + V (Eq 16)", LB, fs=9)
arrow(1.8, 5.05, 2.3, 5.05); arrow(1.8, 3.35, 2.3, 3.35); arrow(1.8, 1.65, 2.3, 1.65)
arrow(4.0, 3.35, 4.5, 3.35); arrow(4.0, 1.65, 4.5, 1.65)
arrow(6.1, 3.2, 8.2, 3.4)                       # temporal -> node
arrow(6.1, 1.65, 6.5, 2.1)                      # spatial -> attn
arrow(8.0, 2.5, 8.2, 3.0)                       # attn -> node
arrow(4.0, 4.9, 8.2, 3.7)                       # robot mlp -> node
arrow(9.0, 3.0, 9.0, 2.2)                       # node -> actor-critic
ax.text(3.15, 0.7, "v27 = bu pre-MLP blokları açıldı (--pre_mlp)", color=RED,
        fontsize=11, ha="center", fontweight="bold")
ax.set_title("SNCP-PPO mimarisi — v27, makale Eq 11 pre-MLP gömme eklenmesi", fontsize=13)
fig.savefig(OUT + "fig_arch.png"); plt.close(fig)

# ============ FIG 5: version journey timeline ============
fig, ax = plt.subplots(figsize=(10.0, 3.2)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 3)
versions = [
    ("v18", "Eq18 ödülü\ngerçek-robot\nbaseline", GREEN),
    ("v22", "LR 1e-4\nen iyi antipodal", GREEN),
    ("v23", "IL warm-start\nyüksek-N\nbaşarısız", RED),
    ("v24", "paper geometri\n(bütçe kaçtı)", AMBER),
    ("v25", "12.5s aşırı\nsıkı, başarısız", RED),
    ("v26", "50s+8m+comfort\ndürüst %62", GREEN),
    ("v27", "Eq 11 pre-MLP\n(yarın koşacak)", BLUE),
]
xs = np.linspace(0.7, 9.3, len(versions))
ax.plot([0.5, 9.5], [1.7, 1.7], color=GRAY, lw=2, zorder=0)
for (lab, desc, col), x in zip(versions, xs):
    ax.scatter([x], [1.7], s=180, color=col, edgecolor="k", zorder=3)
    ax.text(x, 2.15, lab, ha="center", fontsize=11, fontweight="bold", color=col)
    ax.text(x, 1.15, desc, ha="center", va="top", fontsize=8.0)
ax.set_title("Sürüm yolculuğu (v18 → v27)", fontsize=13)
fig.savefig(OUT + "fig_journey.png"); plt.close(fig)

# ============ FIG 6: paper-vs-impl comparison scorecard ============
cats = ["Ortam /\nsenaryo", "Robot\nkinematiği", "Yaya\nmodeli", "Ödül /\nkonfor",
        "Mimari", "RL /\noptimizasyon", "Değerlen-\ndirme"]
match  = np.array([9, 1, 3, 4, 6, 5, 1])
differ = np.array([1, 0, 0, 0, 1, 0, 2])
unspec = np.array([1, 4, 2, 0, 2, 3, 0])
y = np.arange(len(cats))[::-1]
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.barh(y, match, color=GREEN, label=f"Eşleşiyor ({match.sum()})")
ax.barh(y, differ, left=match, color=RED, label=f"Farklı / kısmî ({differ.sum()})")
ax.barh(y, unspec, left=match + differ, color=AMBER, label=f"Makalede belirtilmemiş ({unspec.sum()})")
for yi, (m, d, u) in zip(y, zip(match, differ, unspec)):
    if m: ax.text(m/2, yi, str(m), va="center", ha="center", color="white", fontsize=10, fontweight="bold")
    if d: ax.text(m + d/2, yi, str(d), va="center", ha="center", color="white", fontsize=10, fontweight="bold")
    if u: ax.text(m + d + u/2, yi, str(u), va="center", ha="center", color="white", fontsize=10, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=10)
ax.set_xlabel("Değişken sayısı"); ax.set_xlim(0, 12); ax.grid(axis="y", alpha=0)
ax.set_title("Makale ile karşılaştırma: değişkenlerin durumu (kategori bazında)")
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
fig.savefig(OUT + "fig_compare.png"); plt.close(fig)

print("figures written:", OUT)
import os
for f in sorted(os.listdir(OUT)):
    print("  ", f, os.path.getsize(OUT + f), "bytes")
