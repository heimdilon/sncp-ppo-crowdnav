"""Equivalence tests for the vectorized Social Force Model (SFM) pedestrian
repulsion in CrowdSimEnv._move_humans.

The SFM human-human repulsion was a nested O(N^2) Python loop calling np.hypot /
np.exp on scalars (crowd_env.py:534). It is being replaced with a vectorized
numpy implementation. These tests pin the EXACT current numerics so the swap is
provably behaviour-preserving:

  * a frozen reference re-implementation of the original inner loop, and a frozen
    reference of the WHOLE per-agent SFM step, both byte-for-byte from the code
    as it was, drive the comparison;
  * the vectorized output must match the reference to ~floating-point exactness
    (atol 1e-12), not merely "close".
"""

import numpy as np

from crowd_sim.crowd_env import CrowdSimEnv

A_REP = 2.0  # repulsive force magnitude   (matches _move_humans)
B_REP = 0.3  # repulsive force range        (matches _move_humans)


def _reference_human_repulsion(px, py, radius, A=A_REP, B=B_REP):
    """Frozen, literal re-implementation of the original inner N^2 loop
    (crowd_env.py lines 531-544). Operation order is preserved exactly so the
    result is bit-comparable with the vectorized version."""
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


def _reference_move_humans_sfm(env):
    """Frozen, literal re-implementation of the ENTIRE original _move_humans SFM
    step (driving force + human repulsion + robot repulsion + integration +
    speed clamp + orientation). Returns the post-step arrays."""
    N = env.num_humans
    tau = 0.5
    A = A_REP
    B = B_REP

    px_arr = env.humans_px.copy()
    py_arr = env.humans_py.copy()
    vx_arr = env.humans_vx.copy()
    vy_arr = env.humans_vy.copy()
    theta = env.humans_theta.copy()

    new_px = np.zeros(N)
    new_py = np.zeros(N)
    new_vx = np.zeros(N)
    new_vy = np.zeros(N)

    for i in range(N):
        px, py = px_arr[i], py_arr[i]
        vx, vy = vx_arr[i], vy_arr[i]

        gx, gy = env.humans_gx[i], env.humans_gy[i]
        dx_g = gx - px
        dy_g = gy - py
        dist_g = np.hypot(dx_g, dy_g)
        if dist_g < 0.1:
            pref_vx = 0.0
            pref_vy = 0.0
        else:
            vpref_i = env._human_vpref(i)
            pref_vx = (dx_g / dist_g) * vpref_i
            pref_vy = (dy_g / dist_g) * vpref_i
        f_drive_x = (pref_vx - vx) / tau
        f_drive_y = (pref_vy - vy) / tau

        f_rep_x = 0.0
        f_rep_y = 0.0
        for j in range(N):
            if j != i:
                dx = px - px_arr[j]
                dy = py - py_arr[j]
                dist = np.hypot(dx, dy)
                r_sum = 2.0 * env.human_radius
                if dist > 0:
                    force = A * np.exp((r_sum - dist) / B)
                    f_rep_x += force * (dx / dist)
                    f_rep_y += force * (dy / dist)

        if env.human_dodge_robot:
            dx_r = px - env.robot_px
            dy_r = py - env.robot_py
            dist_r = np.hypot(dx_r, dy_r)
            r_sum_r = env.human_radius + env.robot_radius
            if dist_r > 0:
                force_r = A * np.exp((r_sum_r - dist_r) / B)
                f_rep_x += force_r * (dx_r / dist_r)
                f_rep_y += force_r * (dy_r / dist_r)

        ax = f_drive_x + f_rep_x
        ay = f_drive_y + f_rep_y

        nvx = vx + ax * env.time_step
        nvy = vy + ay * env.time_step

        speed = np.hypot(nvx, nvy)
        vpref_i = env._human_vpref(i)
        if speed > vpref_i:
            nvx = (nvx / speed) * vpref_i
            nvy = (nvy / speed) * vpref_i

        new_px[i] = px + nvx * env.time_step
        new_py[i] = py + nvy * env.time_step
        new_vx[i] = nvx
        new_vy[i] = nvy

        if np.hypot(nvx, nvy) > 0.01:
            theta[i] = np.arctan2(nvy, nvx)

    return new_px, new_py, new_vx, new_vy, theta


def _sfm_env(num_humans, seed=0):
    env = CrowdSimEnv(num_humans=num_humans, scenario='hard', human_motion_model='sfm')
    env.reset(seed=seed)
    return env


# --------------------------------------------------------------------------- #
# 1. The extracted unit: vectorized pairwise human-human repulsion.
# --------------------------------------------------------------------------- #

def test_human_repulsion_forces_matches_loop_reference():
    """Vectorized pairwise repulsion is bit-equivalent to the original loop on a
    non-trivial randomized crowd."""
    env = _sfm_env(num_humans=8, seed=3)
    rng = np.random.default_rng(0)
    env.humans_px = rng.uniform(-4.0, 4.0, size=8)
    env.humans_py = rng.uniform(-4.0, 4.0, size=8)

    exp_fx, exp_fy = _reference_human_repulsion(
        env.humans_px, env.humans_py, env.human_radius)
    fx, fy = env._human_repulsion_forces()

    np.testing.assert_allclose(fx, exp_fx, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fy, exp_fy, rtol=0.0, atol=1e-12)


def test_human_repulsion_forces_single_human_is_zero():
    """A lone pedestrian has no one to be repelled by."""
    env = _sfm_env(num_humans=1, seed=0)
    fx, fy = env._human_repulsion_forces()
    assert fx.shape == (1,)
    assert fy.shape == (1,)
    np.testing.assert_array_equal(fx, np.zeros(1))
    np.testing.assert_array_equal(fy, np.zeros(1))


def test_human_repulsion_forces_coincident_humans_no_nan():
    """Two pedestrians at the exact same point: the original loop skips the pair
    (dist == 0 fails `dist > 0`), so the force is zero and finite — the
    vectorized version must not emit nan/inf from a 0/0 division."""
    env = _sfm_env(num_humans=2, seed=0)
    env.humans_px = np.array([1.0, 1.0])
    env.humans_py = np.array([2.0, 2.0])
    fx, fy = env._human_repulsion_forces()
    assert np.all(np.isfinite(fx))
    assert np.all(np.isfinite(fy))
    np.testing.assert_array_equal(fx, np.zeros(2))
    np.testing.assert_array_equal(fy, np.zeros(2))


# --------------------------------------------------------------------------- #
# 2. Characterization guard: the WHOLE _move_humans SFM step is unchanged.
# --------------------------------------------------------------------------- #

def test_move_humans_sfm_step_matches_reference():
    """End-to-end: one full SFM step (with robot dodging on) is byte-identical to
    the frozen reference re-implementation of the original method."""
    env = CrowdSimEnv(num_humans=6, scenario='hard',
                      human_motion_model='sfm', human_dodge_robot=True)
    env.reset(seed=5)
    rng = np.random.default_rng(1)
    env.humans_px = rng.uniform(-3.0, 3.0, size=6)
    env.humans_py = rng.uniform(-3.0, 3.0, size=6)
    env.humans_vx = rng.uniform(-0.5, 0.5, size=6)
    env.humans_vy = rng.uniform(-0.5, 0.5, size=6)

    exp_px, exp_py, exp_vx, exp_vy, exp_theta = _reference_move_humans_sfm(env)
    env._move_humans()

    np.testing.assert_allclose(env.humans_px, exp_px, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(env.humans_py, exp_py, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(env.humans_vx, exp_vx, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(env.humans_vy, exp_vy, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(env.humans_theta, exp_theta, rtol=0.0, atol=1e-12)
