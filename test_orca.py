import numpy as np

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
        np.array([0.8, 0.0]), [], max_speed=1.0,
    )
    assert np.allclose(out, [0.8, 0.0], atol=1e-6)


def test_max_speed_caps_pref_velocity():
    out = orca_new_velocity(
        np.zeros(2), np.zeros(2), 0.3, np.array([2.0, 0.0]), [], max_speed=1.0,
    )
    assert np.isclose(np.hypot(*out), 1.0, atol=1e-6)


def test_head_on_agents_avoid():
    """Two agents closing head-on must each pick up a sideways component."""
    pos_a, vel_a = np.array([-2.0, 0.0]), np.array([1.0, 0.0])
    pos_b, vel_b = np.array([2.0, 0.0]), np.array([-1.0, 0.0])
    pref = np.array([1.0, 0.0])
    new_a = orca_new_velocity(pos_a, vel_a, 0.3, pref, [(pos_b, vel_b, 0.3)], max_speed=1.0)
    assert abs(new_a[1]) > 1e-3, f"no sideways avoidance: {new_a}"


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
