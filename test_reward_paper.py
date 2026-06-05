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


def test_approach_coefficient_is_1():
    """Dense approach shaping = 1.0 * delta-distance (v15: halved from 2.0 so
    detours around people are not over-penalized vs the straight line)."""
    env = CrowdSimEnv(num_humans=1, scenario='easy')
    env.reset(seed=2)
    # Push the only human far away so comfort ~0 and isolate the approach term.
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0
    # Place the goal 1 m ahead (+x), robot heading +x, drive at vpref.
    env.robot_gx, env.robot_gy = env.robot_px + 1.0, env.robot_py
    env.robot_theta = 0.0
    prev = np.hypot(env.robot_px - env.robot_gx, env.robot_py - env.robot_gy)
    _, reward, *_ = env.step(np.array([env.robot_vpref, 0.0], dtype=np.float32))
    moved = prev - np.hypot(env.robot_px - env.robot_gx, env.robot_py - env.robot_gy)
    assert moved > 0, "robot did not move toward goal"
    # reward ~= 1.0*moved (minus tiny orientation/comfort). ~1x, not ~2x.
    assert reward < 1.6 * moved, f"approach looks like 2x ({reward} vs moved {moved})"
    assert reward > 0.5 * moved


def test_max_time_default_is_50():
    # 50s (200 steps, ~13m reach) gives the randomly-oriented robot room to turn
    # toward the goal + traverse ~8m + maneuver around pedestrians. 35s was too
    # tight (timeout-dominant: robot never reached the goal, so it never saw the
    # +20 signal and couldn't learn). Paper's 12.5s is for a 1.0 m/s robot; ours
    # is 0.26 m/s, so the time budget must be larger.
    env = CrowdSimEnv()
    assert env.max_time == 50.0
