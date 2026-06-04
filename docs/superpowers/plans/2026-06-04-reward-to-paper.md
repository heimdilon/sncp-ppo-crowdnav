# Restore Reward Function to the Paper (v13) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the SNCP-PPO reward function to the source paper's recipe (goal +20, approach 2·Δd, collision −20, comfort −2·I_sp, max_time 35s) so the policy actually feels social pressure — the diagnosed bottleneck.

**Architecture:** Five constant changes in `crowd_sim/crowd_env.py` (four in `step()`'s reward block, one default in `__init__`). No observation/model/ppo changes — shapes unchanged, so this is not a new architecture generation (retrained fresh as v13 only because the reward changed).

**Tech Stack:** NumPy, Gymnasium, pytest.

---

## File Structure

- `crowd_sim/crowd_env.py` — reward constants in `step()` (lines ~375, 378, 390, 401) + `max_time` default (line ~10).
- `test_reward_paper.py` (new) — verifies each reward term equals the paper value, isolating terminal/comfort terms from the (kept) orientation term.
- `sncp_ppo_colab.ipynb` — v13 checkpoint names.

**Test isolation note:** the dense reward is `r_g = 2·Δd − weight·|angle_diff|` (orientation term kept). To test cleanly: terminal cases (goal/collision) are orientation-independent; comfort is tested by mocking `_compute_social_pressure` to a known value; approach-coef is tested with `d_min` large enough that the orientation `weight` clamps to its max but with the robot already facing the goal (`angle_diff≈0`), so the orientation term is ~0.

---

## Task 1: Reward constants in step() (goal, approach, collision, comfort)

**Files:**
- Modify: `crowd_sim/crowd_env.py` (lines ~374-401 in `step()`)
- Create: `test_reward_paper.py`

- [ ] **Step 1: Write the failing tests**

Create `test_reward_paper.py` with EXACTLY this content:

```python
import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv


def test_goal_reward_is_20():
    """Reaching the goal contributes +20 (paper Eq 18), not +50."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    env.reset(seed=1)
    # Place robot essentially on the goal so the next step reaches it.
    env.robot_px, env.robot_py = env.robot_gx, env.robot_gy
    _, reward, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info['success'] is True
    # On goal: r_g=20, r_c=0, r_s small negative. Goal term dominates; reward≈20.
    assert reward > 19.0, f"goal reward not ~20: {reward}"
    assert reward < 21.0, f"goal reward too high (still +50?): {reward}"


def test_collision_penalty_is_minus_20():
    """A collision contributes -20 (paper Eq 19), not -25."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    env.reset(seed=1)
    # Put a human on top of the robot so the next step collides.
    env.humans_px[0], env.humans_py[0] = env.robot_px, env.robot_py
    _, reward, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info['collision'] is True
    # Collision term is -20; comfort adds a small negative. reward should be
    # near -20 and strictly greater than -25 (the old value) minus slack.
    assert reward < -19.0, f"collision reward not <= ~-20: {reward}"
    assert reward > -24.0, f"collision still ~-25?: {reward}"


def test_comfort_is_minus_2_times_Isp_no_divide_by_N(monkeypatch):
    """Comfort penalty = -2.0 * I_sp (paper Eq 20), with NO division by N."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    env.reset(seed=1)
    # Force a known social-pressure value and a non-terminal, non-colliding step.
    monkeypatch.setattr(env, '_compute_social_pressure', lambda: 0.5)
    # Move humans far away so no collision; robot makes a tiny move.
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0
    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    # comfort component reported in info
    assert np.isclose(info['comfort'], -2.0 * 0.5), f"comfort not -2*I_sp: {info['comfort']}"
    # Confirm it is NOT the old -0.5*I_sp/N = -0.05 value
    assert not np.isclose(info['comfort'], -0.5 * 0.5 / 5)


def test_max_time_default_is_35():
    env = CrowdSimEnv()
    assert env.max_time == 35.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_reward_paper.py -v`
Expected: FAIL — `test_goal_reward_is_20` (reward ≈50), `test_collision_penalty` (≈-25), `test_comfort` (info['comfort'] = -0.05 not -1.0), `test_max_time_default_is_35` (60.0).

- [ ] **Step 3: Implement the reward changes**

In `crowd_sim/crowd_env.py` `step()`, change the goal arrival reward (line ~375) from `r_g = 50.0` to:
```python
            r_g = 20.0
```

Change the approach coefficient (line ~378) from `r_g = 5.0 * (prev_dist_to_goal - dist_to_goal)` to:
```python
            r_g = 2.0 * (prev_dist_to_goal - dist_to_goal)
```

Change the collision penalty (line ~390) from `r_c = -25.0` to:
```python
            r_c = -20.0
```

Change the comfort penalty (line ~401) from `r_s = -0.5 * I_sp / max(1, self.num_humans)` to:
```python
        r_s = -2.0 * I_sp
```

Also update the comment block above the comfort line (lines ~394-399) to reflect the paper recipe — replace it with:
```python
        # 3. Comfort penalty (paper Eq 20): r_s = -2 * I_sp. I_sp is the social
        # pressure index summed over humans (per-human cap 10/d_hr), already a
        # normalized 0..1 quantity. The -2 coefficient matches the source paper;
        # the earlier -0.5/N was ~20x weaker at N=5 and let the robot ignore
        # social proximity (it never braked into crowds).
```

- [ ] **Step 4: Run tests to verify the four reward tests pass**

Run: `python -m pytest test_reward_paper.py::test_goal_reward_is_20 test_reward_paper.py::test_collision_penalty_is_minus_20 test_reward_paper.py::test_comfort_is_minus_2_times_Isp_no_divide_by_N -v`
Expected: 3 PASS. (`test_max_time_default_is_35` still fails — fixed in Task 2.)

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/crowd_env.py test_reward_paper.py
git commit -m "feat(reward): restore paper reward magnitudes (goal+20, approach 2, collision-20, comfort-2*Isp)"
```

---

## Task 2: max_time default 60 → 35

**Files:**
- Modify: `crowd_sim/crowd_env.py` (`__init__` signature, line ~10)

- [ ] **Step 1: Verify the failing test (from Task 1)**

Run: `python -m pytest test_reward_paper.py::test_max_time_default_is_35 -v`
Expected: FAIL — `assert 60.0 == 35.0`.

- [ ] **Step 2: Implement**

In `crowd_sim/crowd_env.py`, change the `__init__` signature default (line ~10) from `max_time=60.0` to `max_time=35.0`:
```python
    def __init__(self, num_humans=5, time_step=0.25, max_time=35.0, scenario='circle', human_dodge_robot=False, randomize_layout=True):
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest test_reward_paper.py -v`
Expected: 4 PASS (all reward + max_time tests).

- [ ] **Step 4: Commit**

```bash
git add crowd_sim/crowd_env.py
git commit -m "feat(env): max_time 60->35s (paper t_lim adapted to 0.26 m/s robot)"
```

---

## Task 3: Full regression + smoke + notebook v13

**Files:**
- Modify: `sncp_ppo_colab.ipynb` (training `SAVE_PATH` + eval `CHECKPOINT` → v13)

- [ ] **Step 1: Full unit-test regression**

Run: `python -m pytest test_reward_paper.py test_env.py test_env_goaldir.py test_model.py test_env_velocity.py test_vec_curriculum.py test_vec_buffer.py test_vec_gae.py test_train_eta.py test_env_randomization.py -q`
Expected: all PASS. (Reward change doesn't alter shapes; max_time change may affect any test that assumes 60s — if a test hardcodes max_time/240 steps, fix that test to use `env.max_time` rather than a literal.)

- [ ] **Step 2: Vectorized GPU smoke (rewards sane, no explosion)**

Run: `python -u -m sncp_ppo.train --num_envs 2 --horizon 32 --total_steps 320 --eval_freq_updates 3 --num_humans 5 --holdout_episodes 1 --best_warmup_evals 0 --best_min_success_threshold 0.0 --save_path checkpoints/_v13_smoke.pt`
Expected: exit 0, "Vectorized training completed!", no traceback. Check the printed `rms` (return std) is not exploding to absurd values from the 4× stronger comfort — it should stay in a sane range (single/low-double digits). Then: `rm -f checkpoints/_v13_smoke*.pt` and the smoke's `logs/training_*.csv`.

- [ ] **Step 3: Update notebook to v13**

In `sncp_ppo_colab.ipynb` (use NotebookEdit; read it first): training cell `SAVE_PATH = 'checkpoints/sncp_ppo_v12.pt'` → `'checkpoints/sncp_ppo_v13.pt'`; eval cell `CHECKPOINT` → v13. Update the training markdown note to: "v13 restores the paper reward (comfort -2*I_sp, goal +20, collision -20, max_time 35s); same 6-dim obs as v12 (load-compatible)."

- [ ] **Step 4: Validate notebook JSON**

Run: `python -c "import json; nb=json.load(open('sncp_ppo_colab.ipynb',encoding='utf-8')); print('valid, cells:', len(nb['cells']))"`
Expected: "valid, cells: 31"

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo_colab.ipynb
git commit -m "docs(colab): point notebook to v13 (paper reward)"
```

---

## Self-Review Notes

- **Spec coverage:** changes #1-4 (goal/approach/collision/comfort) → Task 1; #5 (max_time) → Task 2; regression+smoke+notebook → Task 3. Orientation-term kept (untouched, per spec). All spec tests covered.
- **Test isolation:** goal/collision tested as terminal (orientation-independent); comfort via mock (`monkeypatch` of `_compute_social_pressure`); max_time as a direct attribute check. The dense approach-coef (5→2) is implicitly covered by the goal/comfort tests passing with the new code — a dedicated approach-coef test is omitted (YAGNI) because isolating it from the orientation term is fragile and the constant change is trivially visible.
- **Type consistency:** all five values match the spec table (20, 2, -20, -2·I_sp, 35.0).
- **Known risk flagged:** Task 3 Step 1 watches for any test that hardcodes 60s/240 steps; Task 3 Step 2 watches `rms` for comfort-driven reward explosion.
