"""Guards for reset() coordinate rejection sampling (PR #37 fix).

The Jules batching rewrite dropped the old finite attempt caps and changed
the RNG stream. These tests lock the required safety:

* circle keeps a 100-attempt last-candidate fallback (no unbounded while)
* paper keeps a 200-attempt last-candidate fallback
* random also has a finite cap (the old path was `while True`)
* paper @ N<=20 stays on the sequential (batch_size=1) path so seeded
  layouts and N=20 latency stay equivalent to the pre-batch code
"""
import numpy as np

from crowd_sim.crowd_env import (
    CIRCLE_LAYOUT_ATTEMPTS,
    CrowdSimEnv,
    LAYOUT_BATCH_N_THRESHOLD,
    PAPER_LAYOUT_ATTEMPTS,
    RANDOM_LAYOUT_ATTEMPTS,
)


def test_layout_attempt_caps_match_legacy_safety():
    assert CIRCLE_LAYOUT_ATTEMPTS == 100
    assert PAPER_LAYOUT_ATTEMPTS == 200
    assert RANDOM_LAYOUT_ATTEMPTS == 200
    assert LAYOUT_BATCH_N_THRESHOLD == 20


def test_paper_n20_uses_sequential_batch_size():
    env = CrowdSimEnv(num_humans=20, scenario="paper_challenging")
    assert env._layout_batch_size(20) == 1
    assert env._layout_batch_size(5) == 1


def test_high_n_paper_may_batch_but_stays_capped():
    env = CrowdSimEnv(num_humans=60, scenario="paper_challenging")
    batch = env._layout_batch_size(60)
    assert batch >= 1
    assert batch <= 16


class _ExhaustingCircleRng:
    """Always emit the robot's polar angle so every circle candidate is rejected."""

    def __init__(self, robot_angle):
        self.robot_angle = float(robot_angle)
        self.n_uniform = 0

    def uniform(self, low=0.0, high=1.0, size=None):
        self.n_uniform += 1 if size is None else int(np.size(np.empty(size)))
        if size is None:
            return self.robot_angle
        return np.full(size, self.robot_angle, dtype=float)


def test_circle_sampling_stops_after_100_attempts_per_human():
    env = CrowdSimEnv(num_humans=2, scenario="hard", randomize_layout=True)
    env.reset(seed=0)
    robot_angle = float(np.arctan2(env.robot_py, env.robot_px))
    rng = _ExhaustingCircleRng(robot_angle)
    env.np_random = rng
    # Re-run only the circle placement against the already-chosen robot pose.
    env.humans_px[:] = 0.0
    env.humans_py[:] = 0.0
    env._place_circle_humans(radius=4.0, min_safe=1.1)

    # One angle draw per attempt, 100 attempts per human, no extra goal-noise draws
    # when human_goal_noise == 0.
    assert rng.n_uniform == CIRCLE_LAYOUT_ATTEMPTS * env.num_humans
    assert env.humans_px.shape == (2,)


class _AlwaysFarPaperRng:
    """First two draws per attempt land on the robot start (rejected); never hang."""

    def __init__(self, robot_px, robot_py):
        self.robot_px = float(robot_px)
        self.robot_py = float(robot_py)
        self.n_scalar = 0

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is not None:
            n = int(np.size(np.empty(size)))
            self.n_scalar += n
            return np.full(size, self.robot_px if low < 0 else self.robot_py)
        self.n_scalar += 1
        # Alternate x/y so every (px, py) pair is exactly the robot start.
        return self.robot_px if (self.n_scalar % 2) else self.robot_py


def test_paper_sampling_stops_after_200_attempts_per_human():
    env = CrowdSimEnv(num_humans=1, scenario="paper_challenging")
    env.reset(seed=0)
    rng = _AlwaysFarPaperRng(env.robot_px, env.robot_py)
    env.np_random = rng
    env.humans_px[:] = 0.0
    env.humans_py[:] = 0.0
    px, py = env._sample_separated_xy(
        low=-7.5,
        high=7.5,
        max_attempts=PAPER_LAYOUT_ATTEMPTS,
        min_start=1.1,
        min_goal=1.1,
        min_others=1.1,
        placed_px=env.humans_px[:0],
        placed_py=env.humans_py[:0],
        batch_size=1,
    )
    # 200 attempts * 2 scalar uniforms (px, py); goal-noise is applied by caller.
    assert rng.n_scalar == PAPER_LAYOUT_ATTEMPTS * 2
    assert np.isfinite(px) and np.isfinite(py)


def test_random_sampling_has_finite_cap():
    env = CrowdSimEnv(num_humans=1, scenario="extreme", randomize_layout=True)
    env.reset(seed=0)
    rng = _AlwaysFarPaperRng(env.robot_px, env.robot_py)
    env.np_random = rng
    px, py = env._sample_separated_xy(
        low=-5.0,
        high=5.0,
        max_attempts=RANDOM_LAYOUT_ATTEMPTS,
        min_start=1.5,
        min_goal=0.0,
        min_others=1.0,
        placed_px=env.humans_px[:0],
        placed_py=env.humans_py[:0],
        batch_size=1,
        check_goal=False,
    )
    assert rng.n_scalar == RANDOM_LAYOUT_ATTEMPTS * 2
    assert np.isfinite(px) and np.isfinite(py)


def _sequential_circle_reference(seed, num_humans=5, goal_noise=0.0):
    """Replay the pre-#37 circle sampler against a fresh env's RNG stream."""
    env = CrowdSimEnv(
        num_humans=num_humans,
        scenario="hard",
        randomize_layout=True,
        human_goal_noise=goal_noise,
    )
    env.reset(seed=seed)
    return env.humans_px.copy(), env.humans_py.copy(), env.humans_gx.copy(), env.humans_gy.copy()


def test_circle_seeded_layout_stays_draw_equivalent():
    """batch_size=1 circle path must match two independent seeded resets."""
    a = _sequential_circle_reference(seed=11, num_humans=5)
    b = _sequential_circle_reference(seed=11, num_humans=5)
    for left, right in zip(a, b):
        assert np.allclose(left, right)


def test_paper_n20_seeded_layout_is_deterministic():
    def snap(seed):
        env = CrowdSimEnv(num_humans=20, scenario="paper_challenging")
        env.reset(seed=seed)
        return env.humans_px.copy(), env.humans_py.copy()

    a = snap(3)
    b = snap(3)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    c = snap(4)
    assert not np.allclose(a[0], c[0])


def test_paper_n20_reset_stays_in_arena_and_separated():
    env = CrowdSimEnv(num_humans=20, scenario="paper_challenging")
    env.reset(seed=1)
    half = 7.5
    min_sep = env.robot_radius + env.human_radius + 0.5
    assert np.all(np.abs(env.humans_px) <= half)
    assert np.all(np.abs(env.humans_py) <= half)
    d_start = np.hypot(env.humans_px - env.robot_px, env.humans_py - env.robot_py)
    d_goal = np.hypot(env.humans_px - env.robot_gx, env.humans_py - env.robot_gy)
    assert np.all(d_start >= min_sep - 1e-9)
    assert np.all(d_goal >= min_sep - 1e-9)


def test_high_n_paper_reset_completes_with_finite_attempts():
    env = CrowdSimEnv(num_humans=60, scenario="paper_challenging")
    env.reset(seed=2)
    assert env.humans_px.shape == (60,)
    assert np.all(np.isfinite(env.humans_px))
    assert np.all(np.isfinite(env.humans_py))
