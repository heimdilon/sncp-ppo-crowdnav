# -*- coding: utf-8 -*-
"""v30 trajectory figure for the IEEE paper: N=10 and N=20 side by side.

Reuses the rollout logic from scripts/visualize_trajectory.py but renders in the
paper's clean style with arena-correct limits (15x15 challenging -> +/-7.5 m). For
each density it searches for a successful avoidance episode (up to N_SEARCH seeds)
so the figure shows the champion actually weaving through the crowd. Same regime as
the honest sweeps: robot 1.0, human 1.0, paper_challenging, env-derived budget.

Run: C:/ProgramData/miniconda3/python.exe rapor/make_traj_figure.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.models import build_policy_for_checkpoint
from sncp_ppo.ppo import PPOAgent

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})
BLUE, GREEN, GRAY, RED = "#185FA5", "#3B6D11", "#9A9A95", "#C0392B"
CKPT = "sncp_ppo_v30.pt"
N_SEARCH = 40


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def rollout(policy, agent, device, num_humans, base_seed):
    """Return (robot_path, human_paths, env, info) for the first successful episode."""
    last = None
    for ep in range(N_SEARCH):
        seed = base_seed + ep
        set_seed(seed)
        env = CrowdSimEnv(num_humans=num_humans, scenario="paper_challenging",
                          robot_vpref=1.0, human_vpref_override=1.0,
                          max_time=None, human_goal_noise=0.0)
        obs, _ = env.reset(seed=seed)
        h = policy.init_hidden(batch_size=1, num_humans=env.num_humans, device=device)
        robot_path, human_paths = [], [[] for _ in range(env.num_humans)]
        done, steps = False, 0
        while not done and steps < 240:
            robot_path.append((env.robot_px, env.robot_py))
            for i in range(env.num_humans):
                human_paths[i].append((env.humans_px[i], env.humans_py[i]))
            action, _, _, h = agent.select_action(obs, h, device, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        last = (np.array(robot_path), [np.array(p) for p in human_paths], env, info, steps)
        if info.get("success"):
            return last
    return last  # fall back to last episode if none succeeded


def draw(ax, robot_path, human_paths, env, info, steps, num_humans):
    lim = 7.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.add_patch(patches.Rectangle((-lim, -lim), 2 * lim, 2 * lim, fill=False, ec="#CCCCCC", lw=1))
    # pedestrians: faint paths + final positions
    for p in human_paths:
        ax.plot(p[:, 0], p[:, 1], color=GRAY, ls="-", lw=0.8, alpha=0.55)
        ax.plot(p[0, 0], p[0, 1], "o", color=GRAY, ms=3, alpha=0.5)
        ax.add_patch(patches.Circle((p[-1, 0], p[-1, 1]), env.human_radius,
                                    color=GRAY, alpha=0.35, fill=True))
    # robot
    ax.plot(robot_path[:, 0], robot_path[:, 1], "-", color=BLUE, lw=2.6, label="Robot yörüngesi", zorder=5)
    ax.plot(robot_path[0, 0], robot_path[0, 1], "o", color=BLUE, ms=9, label="Başlangıç", zorder=6)
    ax.plot(env.robot_gx, env.robot_gy, "*", color=RED, ms=16, label="Hedef", zorder=6)
    ax.add_patch(patches.Circle((robot_path[-1, 0], robot_path[-1, 1]), env.robot_radius,
                                color=GREEN, alpha=0.7, zorder=6))
    res = "başarı" if info.get("success") else ("çarpışma" if info.get("collision") else "zaman aşımı")
    ax.set_title(f"N={num_humans} ({res}, {steps} adım)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, ls=":", alpha=0.4)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(CKPT, map_location=device, weights_only=True)
    policy = build_policy_for_checkpoint(sd, robot_vpref=1.0, robot_wmax=1.8).to(device)
    policy.load_state_dict(sd)
    policy.train(False)
    agent = PPOAgent(policy=policy)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1))
    for ax, N, base in [(axes[0], 10, 100), (axes[1], 20, 100)]:
        rp, hp, env, info, steps = rollout(policy, agent, device, N, base)
        print(f"N={N}: success={info.get('success')} collision={info.get('collision')} steps={steps}", flush=True)
        draw(ax, rp, hp, env, info, steps, N)
    axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    fig.suptitle("Şampiyon (v30) kalabalıkta gerçek yörüngeler — paper_challenging",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "rapor/figures/ieee_traj.png"
    fig.savefig(out); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
