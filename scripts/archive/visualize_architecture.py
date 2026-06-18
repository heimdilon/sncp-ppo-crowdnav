"""Render the SNCP-PPO policy architecture as a data-flow diagram (PNG).

Mirrors exactly what is implemented in
`sncp_ppo/models.py::SNCPPolicy.forward` plus the observation/action spaces in
`crowd_sim/crowd_env.py`. Pure matplotlib — no training deps.

Intermediate tensors (v_m, m_rr, M_rh, u_att, sf) are drawn as arrow labels
rather than boxes to keep the dataflow readable.

Usage:
    python visualize_architecture.py --output architecture.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- palette -----------------------------------------------------------------
C_INPUT  = ("#cfe8ff", "#2b6cb0")
C_MLP    = ("#e2e8f0", "#4a5568")
C_LTC    = ("#ffe0b2", "#dd6b20")   # Liquid Time-Constant (recurrent)
C_ATTN   = ("#e9d8fd", "#6b46c1")
C_FUSE   = ("#fefcbf", "#b7791f")
C_ACTOR  = ("#c6f6d5", "#2f855a")
C_CRITIC = ("#fed7d7", "#c53030")
C_OUT    = ("#edf2f7", "#1a202c")
INK = "#1a202c"


def box(ax, cx, cy, w, h, text, colors, fs=9.5, fw="normal", tc=INK):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.6",
        linewidth=1.7, facecolor=fc, edgecolor=ec, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, zorder=4, linespacing=1.3)
    return (cx, cy, w, h)


def arrow(ax, src, dst, color="#4a5568", lw=1.8, rad=0.0, label=None,
          lfs=7.8, loff=(0, 1.3)):
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    if dx > sx:
        p0, p1 = (sx + sw / 2, sy), (dx - dw / 2, dy)
    elif dx < sx:
        p0, p1 = (sx - sw / 2, sy), (dx + dw / 2, dy)
    else:
        p0, p1 = (sx, sy - sh / 2), (dx, dy + dh / 2)
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        mx = (p0[0] + p1[0]) / 2 + loff[0]
        my = (p0[1] + p1[1]) / 2 + loff[1] + rad * 8
        ax.text(mx, my, label, ha="center", va="center", fontsize=lfs,
                color=color, style="italic", zorder=5,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9))


def selfloop(ax, blk, label):
    cx, cy, w, h = blk
    ax.add_patch(FancyArrowPatch(
        (cx - w / 2 + 2.2, cy - h / 2), (cx + w / 2 - 2.2, cy - h / 2),
        arrowstyle="-|>", mutation_scale=11, lw=1.3, color=C_LTC[1],
        connectionstyle="arc3,rad=0.85", zorder=2, linestyle=(0, (4, 2))))
    ax.text(cx, cy - h / 2 - 3.4, f"↻ {label}", ha="center", va="center",
            fontsize=7, color=C_LTC[1], style="italic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="architecture.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(54, 98, "SNCP-PPO Politika Mimarisi  —  Girdi → Model → Çıktı",
            ha="center", va="center", fontsize=16.5, fontweight="bold", color=INK)

    X_IN, X_ENC, X_ATT, X_FUSE, X_HEAD, X_OUT = 11, 34, 55, 73, 90, 102
    for x, lab in [(X_IN, "GİRDİLER (gözlem)"), (X_ENC, "KODLAYICILAR"),
                   (X_ATT, "DİKKAT"), (X_FUSE, "FÜZYON"), (X_HEAD, "BAŞLIKLAR"),
                   (X_OUT, "ÇIKTI")]:
        ax.text(x, 91.5, lab, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="#718096")

    # --- inputs ---
    in_robot = box(ax, X_IN, 73, 18, 12,
                   "robot_node  (7)\n[dg_x, dg_y, v, dist,\nvpref, radius, w]", C_INPUT, fs=8.3)
    in_spat = box(ax, X_IN, 49, 18, 12,
                  "spatial_edges  (H,2)\nher yaya:\n[dx_local, dy_local]", C_INPUT, fs=8.3)
    in_temp = box(ax, X_IN, 25, 18, 11,
                  "temporal_edges  (2)\n[v_linear, w_angular]", C_INPUT, fs=8.3)

    # --- encoders ---
    enc_mlp = box(ax, X_ENC, 73, 18, 11, "Robot MLP\n7 → 64 → 128\n(ReLU)", C_MLP, fs=8.7)
    enc_spat = box(ax, X_ENC, 49, 18, 12, "Spatial LTC\n2 → LTC(32) → 256\n(her yaya ayrı)", C_LTC, fs=8.5)
    enc_temp = box(ax, X_ENC, 25, 18, 11, "Temporal LTC\n2 → LTC(32) → 256", C_LTC, fs=8.5)

    # --- attention ---
    attn = box(ax, X_ATT, 49, 16, 13,
               "Attention\nQ=Wq(M_rh), K=Wk(m_rr)\nsoftmax(QKᵀ/8) · M_rh", C_ATTN, fs=8.0)

    # --- fusion ---
    node = box(ax, X_FUSE, 49, 16, 14,
               "Node LTC\n(füzyon)\nconcat[v_m, m_rr, u_att]\n640 → LTC(32) → 256", C_FUSE, fs=8.2)

    # --- heads ---
    actor = box(ax, X_HEAD, 66, 13, 12, "Actor μ\n256 → 64 → 2\nσ = exp(logstd)", C_ACTOR, fs=8.6)
    critic = box(ax, X_HEAD, 32, 13, 11, "Critic V\n256 → 64 → 1", C_CRITIC, fs=8.6)

    # --- outputs ---
    out_act = box(ax, X_OUT, 66, 12, 15,
                  "action (2)\nv = σ()·vpref\n∈ [0, 0.26]\nw = tanh()·wmax\n∈ [-1.8, 1.8]", C_OUT, fs=7.6)
    out_val = box(ax, X_OUT, 32, 10, 8, "value (1)\nV(s)", C_OUT, fs=8.4)

    # --- arrows ---
    arrow(ax, in_robot, enc_mlp)
    arrow(ax, in_spat, enc_spat)
    arrow(ax, in_temp, enc_temp)

    arrow(ax, enc_spat, attn, color=C_ATTN[1], label="M_rh [H,256] · Q")
    arrow(ax, enc_temp, attn, color=C_ATTN[1], rad=0.22, label="m_rr · K")

    arrow(ax, enc_mlp, node, color=C_FUSE[1], rad=-0.18, label="v_m [128]")
    arrow(ax, enc_temp, node, color=C_FUSE[1], rad=0.30, label="m_rr [256]")
    arrow(ax, attn, node, color=C_FUSE[1], label="u_att [256]")

    arrow(ax, node, actor, color=C_ACTOR[1], rad=0.12, label="sf [256]")
    arrow(ax, node, critic, color=C_CRITIC[1], rad=-0.12, label="sf [256]")

    arrow(ax, actor, out_act, color=C_ACTOR[1])
    arrow(ax, critic, out_val, color=C_CRITIC[1])

    # --- recurrent self-loops on LTC blocks ---
    selfloop(ax, enc_temp, "h_temp [32]")
    selfloop(ax, enc_spat, "h_spat [H,32]")
    selfloop(ax, node, "h_node [32]")

    # --- legend ---
    legend = [("Gözlem girdisi", C_INPUT), ("MLP / Dense", C_MLP),
              ("LTC (recurrent)", C_LTC), ("Attention", C_ATTN),
              ("Füzyon", C_FUSE), ("Actor", C_ACTOR), ("Critic", C_CRITIC)]
    lx = 7
    for name, (fc, ec) in legend:
        ax.add_patch(FancyBboxPatch((lx, 5), 2.4, 2.4,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     fc=fc, ec=ec, lw=1.4, zorder=3))
        ax.text(lx + 3.0, 6.2, name, ha="left", va="center", fontsize=8.2, color=INK)
        lx += 14.2

    ax.text(54, 1.4,
            "Kaynak: sncp_ppo/models.py::SNCPPolicy.forward  ·  "
            "LTC gizli durumları her adımda taşınır (BPTT ile eğitilir)  ·  "
            "PPO: GAE + clipped value loss",
            ha="center", va="center", fontsize=8, color="#718096", style="italic")

    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved architecture diagram to {args.output}")


if __name__ == "__main__":
    main()
