import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv


def test_goal_reward_is_20():
    """Reaching the goal contributes +20 (paper Eq 18), not +50."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    env.reset(seed=1)
    # Place robot essentially on the goal so the next step reaches it.
    env.robot_px, env.robot_py = env.robot_gx, env.robot_gy
    _, reward, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert bool(info['success']), "robot should have reached the goal"
    # On goal: r_g=20, r_c=0, r_s small negative. Goal term dominates; reward≈20.
    assert reward > 19.0, f"goal reward not ~20: {reward}"
    assert reward < 21.0, f"goal reward too high (still +50?): {reward}"


def test_collision_penalty_is_minus_20(monkeypatch):
    """A collision contributes -20 (paper Eq 19), not -25.

    Comfort is mocked to 0 to isolate the collision term — otherwise a human on
    top of the robot makes I_sp (and thus -2*I_sp) blow up and mask r_c."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    env.reset(seed=1)
    monkeypatch.setattr(env, '_compute_social_pressure', lambda: 0.0)
    # Put a human on top of the robot so the next step collides.
    env.humans_px[0], env.humans_py[0] = env.robot_px, env.robot_py
    _, reward, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert bool(info['collision']), "robot should have collided"
    # With comfort=0: reward = r_c (-20) + r_g(dense, ~0 since not moving toward goal).
    assert np.isclose(reward, -20.0, atol=1.0), f"collision penalty not ~-20: {reward}"


def test_comfort_is_minus_6_times_Isp(monkeypatch):
    """Comfort penalty = -6.0 * I_sp (v15: strengthened from -2 to teach social
    distance in the non-reactive crowd)."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    env.reset(seed=1)
    # Force a known social-pressure value and a non-terminal, non-colliding step.
    monkeypatch.setattr(env, '_compute_social_pressure', lambda: 0.5)
    # Move humans far away so no collision; robot makes a tiny move.
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0
    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.isclose(info['comfort'], -6.0 * 0.5), f"comfort not -6*I_sp: {info['comfort']}"
    # Confirm it is NOT the old -2*I_sp value
    assert not np.isclose(info['comfort'], -2.0 * 0.5)


def test_custom_comfort_coeff_controls_Isp(monkeypatch):
    env = CrowdSimEnv(num_humans=5, scenario='hard', comfort_coeff=5.0)
    env.reset(seed=1)
    monkeypatch.setattr(env, '_compute_social_pressure', lambda: 0.5)
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert np.isclose(info['comfort'], -5.0 * 0.5)


def test_approach_coefficient_is_2():
    """Dense approach shaping = 2.0 * delta-distance (paper Eq 18).

    v18 restores the paper coefficient. v15 had halved it to 1.0 (on the mistaken
    belief that it would "stop over-penalising detours" — but potential-based
    shaping telescopes to k*(d_0 - d_final) regardless of path, so halving just
    uniformly starves the progress signal). The paper introduces this exact term
    to fix the sparse-reward failure where the robot "may only learn to avoid
    humans without making progress toward the target" (Sec 4.2)."""
    env = CrowdSimEnv(num_humans=1, scenario='easy')
    env.reset(seed=2)
    # Push the only human far away so comfort ~0 and isolate the approach term.
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0
    # Place the goal 1 m ahead (+x), robot heading +x (aligned -> no heading
    # term either way), drive at vpref.
    env.robot_gx, env.robot_gy = env.robot_px + 1.0, env.robot_py
    env.robot_theta = 0.0
    prev = np.hypot(env.robot_px - env.robot_gx, env.robot_py - env.robot_gy)
    _, reward, *_ = env.step(np.array([env.robot_vpref, 0.0], dtype=np.float32))
    moved = prev - np.hypot(env.robot_px - env.robot_gx, env.robot_py - env.robot_gy)
    assert moved > 0, "robot did not move toward goal"
    # reward ~= 2.0*moved (paper Eq 18), not ~1x.
    assert np.isclose(reward, 2.0 * moved, atol=1e-4), f"approach not ~2x: {reward} vs 2*{moved}"


def test_no_orientation_penalty():
    """Paper Eq 18 has NO heading/orientation term.

    v18 removes the impl's ad-hoc `-weight*|angle_diff|` shaping. That term
    penalised the very maneuvering avoidance requires and (at coeff 1.0) exceeded
    the max per-step progress reward, making progress-while-turning net-negative.
    With pedestrians far away (comfort ~0), the dense reward must equal exactly
    2*delta-distance even when the robot's heading is misaligned with the goal."""
    env = CrowdSimEnv(num_humans=1, scenario='easy')
    env.reset(seed=3)
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0
    # Robot at origin facing +x; goal at 45 deg (heading misaligned by ~45 deg).
    env.robot_px, env.robot_py = 0.0, 0.0
    env.robot_gx, env.robot_gy = 3.0, 3.0
    env.robot_theta = 0.0
    prev = np.hypot(env.robot_px - env.robot_gx, env.robot_py - env.robot_gy)
    _, reward, *_ = env.step(np.array([env.robot_vpref, 0.0], dtype=np.float32))
    moved = prev - np.hypot(env.robot_px - env.robot_gx, env.robot_py - env.robot_gy)
    assert moved > 0, "robot did not move closer to goal"
    # No orientation penalty: reward is exactly 2*moved despite the misalignment.
    assert np.isclose(reward, 2.0 * moved, atol=1e-4), (
        f"orientation penalty still present: reward {reward} != 2*moved {2.0 * moved}"
    )


def test_isp_bounded_to_unit_interval():
    """I_sp is clamped to [0,1] (paper Sec 3.3: 'Isp ranges from 0 to 1').

    v19: the impl summed an unbounded per-human term (1/d_hr, capped at 10) over
    the crowd, so a close pedestrian could drive I_sp far above 1 and make the
    comfort penalty (-comfort_coeff * I_sp) spike to -48+/step during training,
    drowning the -20 collision signal. Clamping restores the paper's range and
    lets the collision penalty stay the dominant 'do not hit' signal."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    env.reset(seed=1)
    # One pedestrian almost on top of the robot, the rest far away, so the close
    # one dominates omega and the raw (unclamped) I_sp would be ~8 >> 1.
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0
    env.humans_px[0] = env.robot_px + 0.05
    env.humans_py[0] = env.robot_py
    i_sp = env._compute_social_pressure()
    assert 0.0 <= i_sp <= 1.0, f"I_sp not clamped to [0,1]: {i_sp}"
    # The pile-up should saturate the clamp (proves the raw value exceeded 1).
    assert i_sp >= 0.99, f"I_sp not saturated near 1 for a close pedestrian: {i_sp}"


def test_comfort_coeff_default_unchanged_in_v19():
    """v19 changes ONLY the I_sp range, not the comfort coefficient: default
    stays 6.0 (lowering it would reduce the caution v18 relies on and risk more
    high-density collisions)."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    assert env.comfort_coeff == 6.0


def test_max_time_default_is_50():
    # 50s (200 steps, ~13m reach) gives the randomly-oriented robot room to turn
    # toward the goal + traverse ~8m + maneuver around pedestrians. 35s was too
    # tight (timeout-dominant: robot never reached the goal, so it never saw the
    # +20 signal and couldn't learn). Paper's 12.5s is for a 1.0 m/s robot; ours
    # is 0.26 m/s, so the time budget must be larger.
    env = CrowdSimEnv()
    assert env.max_time == 50.0
