"""Micro-benchmark: original nested-loop SFM repulsion vs the vectorized
CrowdSimEnv._human_repulsion_forces. Asserts bit-equivalence (atol 1e-12) before
timing, then reports us/call and speedup across N. Scratch tool (untracked)."""

import time

import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv

A_REP, B_REP = 2.0, 0.3


def loop_repulsion(px, py, radius, A=A_REP, B=B_REP):
    """The original O(N^2) inner loop, verbatim, as a standalone baseline."""
    N = len(px)
    fx = np.zeros(N)
    fy = np.zeros(N)
    for i in range(N):
        f_rep_x = 0.0
        f_rep_y = 0.0
        for j in range(N):
            if j != i:
                dx = px[i] - px[j]
                dy = py[i] - py[j]
                dist = np.hypot(dx, dy)
                r_sum = 2.0 * radius
                if dist > 0:
                    force = A * np.exp((r_sum - dist) / B)
                    f_rep_x += force * (dx / dist)
                    f_rep_y += force * (dy / dist)
        fx[i] = f_rep_x
        fy[i] = f_rep_y
    return fx, fy


def bench(N, reps, seed=0):
    rng = np.random.default_rng(seed)
    env = CrowdSimEnv(num_humans=N, scenario='hard', human_motion_model='sfm')
    env.reset(seed=seed)
    env.humans_px = rng.uniform(-4.0, 4.0, size=N)
    env.humans_py = rng.uniform(-4.0, 4.0, size=N)
    px, py, r = env.humans_px, env.humans_py, env.human_radius

    lfx, lfy = loop_repulsion(px, py, r)
    vfx, vfy = env._human_repulsion_forces()
    assert np.allclose(lfx, vfx, rtol=0.0, atol=1e-12), "x mismatch"
    assert np.allclose(lfy, vfy, rtol=0.0, atol=1e-12), "y mismatch"

    for _ in range(5):  # warmup
        loop_repulsion(px, py, r)
        env._human_repulsion_forces()

    t0 = time.perf_counter()
    for _ in range(reps):
        loop_repulsion(px, py, r)
    t_loop = (time.perf_counter() - t0) / reps * 1e6

    t0 = time.perf_counter()
    for _ in range(reps):
        env._human_repulsion_forces()
    t_vec = (time.perf_counter() - t0) / reps * 1e6

    return t_loop, t_vec


if __name__ == "__main__":
    print(f"{'N':>4} {'loop us/call':>13} {'vec us/call':>12} {'speedup':>9}")
    print("-" * 42)
    for N, reps in [(2, 20000), (5, 10000), (10, 5000),
                    (20, 2000), (50, 1000), (100, 300)]:
        loop_us, vec_us = bench(N, reps)
        print(f"{N:>4} {loop_us:>13.2f} {vec_us:>12.2f} {loop_us / vec_us:>7.1f}x")
