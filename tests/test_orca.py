import numpy as np

import crowd_sim.orca as orca_module
from crowd_sim.crowd_env import CrowdSimEnv
from crowd_sim.orca import orca_new_velocity, orca_velocities


def test_env_default_motion_model_is_orca():
    """v20: pedestrians move with ORCA by default, so training AND evaluation
    both run in the paper's CrowdSim-style navigable crowd."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    assert env.human_motion_model == 'orca'


def test_env_orca_keeps_pedestrians_separated():
    """In steady state, ORCA keeps pedestrians apart (the SFM crowd knotted at
    the center; ORCA does not). The env spawns pedestrians at random angles
    WITHOUT inter-pedestrian separation, so the first few steps can show spawn
    overlap that ORCA then resolves — we measure after a short settle window."""
    env = CrowdSimEnv(num_humans=8, scenario='circle')
    env.reset(seed=3)
    min_pair = np.inf
    for step in range(120):
        env.step(np.array([0.2, 0.0], dtype=np.float32))
        if step < 30:
            continue  # let ORCA resolve any spawn overlap first
        px, py = env.humans_px, env.humans_py
        for i in range(env.num_humans):
            for j in range(i + 1, env.num_humans):
                min_pair = min(min_pair, float(np.hypot(px[i] - px[j], py[i] - py[j])))
    # Combined pedestrian radius is 0.6; allow discrete-step slack.
    assert min_pair > 0.45, f"ORCA pedestrians overlapped in steady state: {min_pair:.3f}"


def test_no_neighbors_returns_pref_velocity():
    out = orca_new_velocity(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.3,
        np.array([0.8, 0.0]), [], [], [], max_speed=1.0,
    )
    assert np.allclose(out, [0.8, 0.0], atol=1e-6)


def test_max_speed_caps_pref_velocity():
    out = orca_new_velocity(
        np.zeros(2), np.zeros(2), 0.3, np.array([2.0, 0.0]), [], [], [], max_speed=1.0,
    )
    assert np.isclose(np.hypot(*out), 1.0, atol=1e-6)


def test_head_on_agents_avoid():
    """Two agents closing head-on must each pick up a sideways component."""
    pos_a, vel_a = np.array([-2.0, 0.0]), np.array([1.0, 0.0])
    pos_b, vel_b = np.array([2.0, 0.0]), np.array([-1.0, 0.0])
    pref = np.array([1.0, 0.0])
    new_a = orca_new_velocity(pos_a, vel_a, 0.3, pref, [pos_b], [vel_b], [0.3], max_speed=1.0)
    assert abs(new_a[1]) > 1e-3, f"no sideways avoidance: {new_a}"


def test_responsibility_default_is_unchanged():
    """The robot expert (il.expert) needs FULL avoidance responsibility because
    pedestrians are invisible to it. Adding the param must not change the
    reciprocal-0.5 pedestrian behavior: default 0.5 reproduces the old output."""
    pos_b, vel_b = np.array([2.0, 0.0]), np.array([-1.0, 0.0])
    pref = np.array([1.0, 0.0])
    out_default = orca_new_velocity(
        np.array([-2.0, 0.0]), np.array([1.0, 0.0]), 0.3, pref,
        [pos_b], [vel_b], [0.3], max_speed=1.0,
    )
    out_half = orca_new_velocity(
        np.array([-2.0, 0.0]), np.array([1.0, 0.0]), 0.3, pref,
        [pos_b], [vel_b], [0.3], max_speed=1.0, responsibility=0.5,
    )
    assert np.allclose(out_default, out_half, atol=1e-9)


def test_full_responsibility_avoids_more_than_half():
    """responsibility=1.0 makes the agent take the whole avoidance burden, so it
    deviates further sideways than the reciprocal 0.5 case for the same encounter."""
    pos_b, vel_b = np.array([2.0, 0.0]), np.array([-1.0, 0.0])
    pref = np.array([1.0, 0.0])
    half = orca_new_velocity(
        np.array([-2.0, 0.0]), np.array([1.0, 0.0]), 0.3, pref,
        [pos_b], [vel_b], [0.3], max_speed=1.0, responsibility=0.5,
    )
    full = orca_new_velocity(
        np.array([-2.0, 0.0]), np.array([1.0, 0.0]), 0.3, pref,
        [pos_b], [vel_b], [0.3], max_speed=1.0, responsibility=1.0,
    )
    assert abs(full[1]) > abs(half[1]) + 1e-4, f"full {full} not more avoidant than half {half}"


def _assert_public_matches_scalar(positions, velocities, radii, pref_velocities, max_speeds,
                                  time_horizon=3.0, time_step=0.25):
    scalar = getattr(orca_module, '_orca_velocities_scalar', None)
    assert scalar is not None, "freeze the current ORCA wrapper as _orca_velocities_scalar"
    expected = scalar(
        positions, velocities, radii, pref_velocities, max_speeds,
        time_horizon=time_horizon, time_step=time_step,
    )
    actual = orca_velocities(
        positions, velocities, radii, pref_velocities, max_speeds,
        time_horizon=time_horizon, time_step=time_step,
    )
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-9), (
        f"public ORCA diverged from scalar reference; "
        f"max abs error={np.max(np.abs(actual - expected)):.3e}"
    )
    return expected


def test_public_orca_matches_scalar_reference_for_random_agent_configs():
    """The vectorized solver must preserve the scalar port to numerical noise."""
    rng = np.random.default_rng(20260615)
    for case_idx in range(300):
        n = int(rng.integers(1, 13))
        positions = rng.uniform(-4.0, 4.0, size=(n, 2))
        velocities = rng.uniform(-1.2, 1.2, size=(n, 2))
        radii = rng.uniform(0.2, 0.45, size=n)
        pref_velocities = rng.uniform(-1.2, 1.2, size=(n, 2))
        max_speeds = rng.uniform(0.1, 1.2, size=n)
        if n > 1 and case_idx % 10 == 0:
            # Exercise the already-colliding and nearly-coincident branches.
            positions[1] = positions[0] + rng.uniform(-1e-7, 1e-7, size=2)
            velocities[1] = velocities[0] + rng.uniform(-1e-7, 1e-7, size=2)
        _assert_public_matches_scalar(
            positions,
            velocities,
            radii,
            pref_velocities,
            max_speeds,
            time_horizon=float(rng.uniform(0.5, 8.0)),
            time_step=float(rng.uniform(0.05, 0.5)),
        )


def test_public_orca_matches_scalar_reference_for_existing_edge_cases():
    cases = [
        (
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            np.array([0.3]),
            np.array([[0.8, 0.0]]),
            np.array([1.0]),
        ),
        (
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            np.array([0.3]),
            np.array([[2.0, 0.0]]),
            np.array([1.0]),
        ),
        (
            np.array([[-2.0, 0.0], [2.0, 0.0]]),
            np.array([[1.0, 0.0], [-1.0, 0.0]]),
            np.array([0.3, 0.3]),
            np.array([[1.0, 0.0], [-1.0, 0.0]]),
            np.array([1.0, 1.0]),
        ),
        (
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.zeros((2, 2)),
            np.array([0.3, 0.3]),
            np.array([[1.0, 0.0], [-1.0, 0.0]]),
            np.array([1.0, 1.0]),
        ),
    ]
    for case in cases:
        _assert_public_matches_scalar(*case)


def test_public_orca_matches_scalar_reference_during_circle_crossing():
    n, radius, max_speed, dt = 8, 0.3, 1.0, 0.25
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.05, 0.05, n)
    pos = np.stack([4.0 * np.cos(angles), 4.0 * np.sin(angles)], axis=1)
    goals = -pos.copy()
    vel = np.zeros((n, 2))
    radii = np.full(n, radius)
    max_speeds = np.full(n, max_speed)

    for _ in range(400):
        delta = goals - pos
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        pref = np.where(dist > 0.1, delta / np.maximum(dist, 1e-9) * max_speed, 0.0)
        vel = _assert_public_matches_scalar(
            pos, vel, radii, pref, max_speeds, time_horizon=3.0, time_step=dt,
        )
        pos = pos + vel * dt


def test_circle_crossing_no_collision_and_reach():
    """The canonical ORCA validation: agents on a circle heading to antipodal
    goals must reach them WITHOUT ever overlapping (this is exactly the crowd
    scenario, and it is what SFM fails — it knots at the center)."""
    n, radius, max_speed, dt = 8, 0.3, 1.0, 0.25
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.05, 0.05, n)
    pos = np.stack([4.0 * np.cos(angles), 4.0 * np.sin(angles)], axis=1)
    goals = -pos.copy()
    vel = np.zeros((n, 2))
    radii = np.full(n, radius)
    max_speeds = np.full(n, max_speed)

    min_dist = np.inf
    for _ in range(400):
        d = goals - pos
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        pref = np.where(dist > 0.1, d / np.maximum(dist, 1e-9) * max_speed, 0.0)
        vel = orca_velocities(pos, vel, radii, pref, max_speeds, time_horizon=3.0, time_step=dt)
        pos = pos + vel * dt
        for i in range(n):
            for j in range(i + 1, n):
                min_dist = min(min_dist, float(np.hypot(*(pos[i] - pos[j]))))

    reached = np.linalg.norm(pos - goals, axis=1) < 0.5
    # Collision-free: never closer than the combined radius (with small slack).
    assert min_dist > 2 * radius - 0.05, f"agents collided: min pairwise dist {min_dist:.3f}"
    # Nearly all reach (allow one straggler from symmetry).
    assert reached.sum() >= n - 1, f"only {reached.sum()}/{n} reached their goal"
