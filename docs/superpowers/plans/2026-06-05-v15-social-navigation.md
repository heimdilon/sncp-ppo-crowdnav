# v15 Social Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the robot ACTIVELY avoid pedestrians (collision-avoidance + social distance) by training in a non-reactive crowd at speed parity, with density ramped to N=10.

**Architecture:** No model change (NCP from v14 stays). Three changes: (1) env pedestrians non-reactive by default; (2) reward rebalanced — halve approach shaping, triple the social-distance penalty; (3) curriculum uses parity speeds (≤0.26 m/s) and ramps density 1→3→5→8→10. Speeds live in THREE places that must all be capped: `step_to_phase` (training), `SCENARIO_HOLDOUT_CONFIG` (holdout), and `crowd_env.reset()` scenario block (final eval).

**Tech Stack:** Python, PyTorch, Gymnasium, ncps, pytest.

**Spec:** `docs/superpowers/specs/2026-06-05-v15-social-navigation-design.md`

---

### Task 1: Env — non-reactive default + reward rebalance

**Files:**
- Modify: `crowd_sim/crowd_env.py` (constructor line ~10; reward lines ~378, ~400)
- Test: `test_reward_paper.py`, `test_pedestrian_reactive.py`

- [ ] **Step 1: Update reward tests to the v15 values (RED)**

In `test_reward_paper.py`, replace the body of `test_comfort_is_minus_2_times_Isp_no_divide_by_N` assertion and rename:

```python
def test_comfort_is_minus_6_times_Isp(monkeypatch):
    """Comfort penalty = -6.0 * I_sp (v15: strengthened from -2 to teach social
    distance in the non-reactive crowd)."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    env.reset(seed=1)
    monkeypatch.setattr(env, '_compute_social_pressure', lambda: 0.5)
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0
    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.isclose(info['comfort'], -6.0 * 0.5), f"comfort not -6*I_sp: {info['comfort']}"
    assert not np.isclose(info['comfort'], -2.0 * 0.5)
```

Add a new test for the halved approach coefficient:

```python
def test_approach_coefficient_is_1():
    """Dense approach shaping = 1.0 * delta-distance (v15: halved from 2.0 so
    detours around people are not over-penalized vs the straight line)."""
    env = CrowdSimEnv(num_humans=1, scenario='easy')
    env.reset(seed=2)
    # Push the only human far away so comfort ~0 and isolate the approach term.
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0
    # Place robot 1 m from goal along +x, heading toward goal, drive at vpref.
    env.robot_gx, env.robot_gy = env.robot_px + 1.0, env.robot_py
    env.robot_theta = 0.0
    prev = np.hypot(env.robot_px - env.robot_gx, env.robot_py - env.robot_gy)
    _, reward, *_ = env.step(np.array([env.robot_vpref, 0.0], dtype=np.float32))
    moved = prev - np.hypot(env.robot_px - env.robot_gx, env.robot_py - env.robot_gy)
    # reward ~= 1.0*moved (minus tiny orientation/comfort). Confirm it's ~1x not ~2x.
    assert reward < 1.6 * moved, f"approach looks like 2x ({reward} vs moved {moved})"
    assert reward > 0.5 * moved
```

In `test_pedestrian_reactive.py`, change `test_pedestrians_react_to_robot_by_default` to assert the NEW default (non-reactive):

```python
def test_pedestrians_ignore_robot_by_default():
    """v15: the default is NON-reactive ('invisible robot', the paper's CrowdNav
    regime) so the robot must actively avoid. Reactivity stays available via the
    flag for the cooperative-crowd experiments (v14)."""
    assert CrowdSimEnv(num_humans=1, scenario='hard').human_dodge_robot is False
    assert CrowdSimEnv(num_humans=1, scenario='hard', human_dodge_robot=True).human_dodge_robot is True
```

Delete the old `test_pedestrians_react_to_robot_by_default`. Keep `test_reactive_pedestrians_keep_more_clearance` (it still validates the mechanism when the flag is on) but change its `reactive`/`nonreactive` construction to be explicit:

```python
    reactive = CrowdSimEnv(num_humans=1, scenario='hard', human_dodge_robot=True)
    nonreactive = CrowdSimEnv(num_humans=1, scenario='hard', human_dodge_robot=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_reward_paper.py test_pedestrian_reactive.py -q`
Expected: FAIL — comfort is -2 not -6; default is True not False.

- [ ] **Step 3: Apply the env changes (GREEN)**

`crowd_sim/crowd_env.py` constructor line ~10:
```python
    def __init__(self, num_humans=5, time_step=0.25, max_time=50.0, scenario='circle', human_dodge_robot=False, randomize_layout=True):
```

Reward line ~378 (approach coefficient):
```python
            r_g = 1.0 * (prev_dist_to_goal - dist_to_goal)
```

Reward line ~400 (comfort coefficient):
```python
        r_s = -6.0 * I_sp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_reward_paper.py test_pedestrian_reactive.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/crowd_env.py test_reward_paper.py test_pedestrian_reactive.py
git commit -m "v15: non-reactive default + reward rebalance (approach 1x, comfort -6)"
```

---

### Task 2: Env — pedestrian speed parity in scenario block (final-eval path)

**Files:**
- Modify: `crowd_sim/crowd_env.py` (scenario block lines ~101-118)
- Test: `test_speed_parity.py` (create)

- [ ] **Step 1: Write the failing test (RED)**

Create `test_speed_parity.py`:
```python
from crowd_sim.crowd_env import CrowdSimEnv


def test_pedestrian_speed_never_exceeds_robot():
    """v15 parity: at hardware robot speed (0.26 m/s), no scenario sets
    pedestrians faster than the robot, so a slow robot can feasibly avoid a
    non-reactive crowd."""
    for scenario in ['easy', 'easy_plus', 'medium', 'hard', 'extreme', 'circle']:
        env = CrowdSimEnv(num_humans=3, scenario=scenario)
        env.reset(seed=0)
        assert env.human_vpref <= env.robot_vpref + 1e-9, (
            f"{scenario}: human_vpref {env.human_vpref} > robot {env.robot_vpref}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest test_speed_parity.py -q`
Expected: FAIL — hard/extreme/circle use 0.50 > 0.26.

- [ ] **Step 3: Cap scenario speeds (GREEN)**

`crowd_sim/crowd_env.py` scenario block (lines ~101-118): replace the `human_vpref` values:
```python
        if self.scenario == 'easy':
            self.human_vpref = 0.13
            scenario_type = 'circle'
        elif self.scenario == 'easy_plus':
            self.human_vpref = 0.18
            scenario_type = 'circle'
        elif self.scenario == 'medium':
            self.human_vpref = 0.22
            scenario_type = 'circle'
        elif self.scenario == 'hard':
            self.human_vpref = 0.26
            scenario_type = 'circle'
        elif self.scenario == 'extreme':
            self.human_vpref = 0.26
            scenario_type = 'random'
        else:
            self.human_vpref = 0.26
            scenario_type = self.scenario
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest test_speed_parity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/crowd_env.py test_speed_parity.py
git commit -m "v15: cap scenario pedestrian speeds to robot parity (<=0.26)"
```

---

### Task 3: Curriculum — density ramp to N=10 + parity speeds + holdout

**Files:**
- Modify: `sncp_ppo/train.py` (`step_to_phase` lines ~525-540; `SCENARIO_HOLDOUT_CONFIG` lines ~69-77)
- Test: `test_vec_curriculum.py` (add cases)

- [ ] **Step 1: Write the failing tests (RED)**

Add to `test_vec_curriculum.py`:
```python
from sncp_ppo.train import step_to_phase, SCENARIO_HOLDOUT_CONFIG


def test_curriculum_ramps_to_final_humans_with_parity_speed():
    """v15: final phase reaches final_num_humans at parity speed (<=0.26);
    every phase speed is <= 0.26."""
    final = 10
    phases = [step_to_phase(int(f * 1000), 1000, final)
              for f in (0.0, 0.2, 0.4, 0.6, 0.9)]
    names = [p[0] for p in phases]
    humans = [p[1] for p in phases]
    speeds = [p[2] for p in phases]
    assert humans == sorted(humans), f"density not monotonic: {humans}"
    assert humans[-1] == final, f"final phase N != {final}: {humans[-1]}"
    assert max(speeds) <= 0.26 + 1e-9, f"a phase exceeds parity: {speeds}"


def test_holdout_config_is_parity_and_has_highdensity():
    """v15: holdout speeds <= 0.26; a high-density (N=10) holdout exists."""
    for name, (n, v) in SCENARIO_HOLDOUT_CONFIG.items():
        assert v <= 0.26 + 1e-9, f"{name} holdout speed {v} > parity"
    assert SCENARIO_HOLDOUT_CONFIG['circle'][0] >= 10, "no high-density holdout"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest test_vec_curriculum.py -q`
Expected: FAIL — speeds 0.50/0.40 exceed parity; circle holdout N=5.

- [ ] **Step 3: Update `step_to_phase` and `SCENARIO_HOLDOUT_CONFIG` (GREEN)**

`sncp_ppo/train.py` `step_to_phase` body (lines ~532-540):
```python
    frac = steps_seen / max(1, total_steps)
    if frac <= 0.10:
        return ('easy', 1, 0.13)
    if frac <= 0.25:
        return ('easy_plus', 3, 0.18)
    if frac <= 0.50:
        return ('medium', 5, 0.22)
    if frac <= 0.75:
        return ('hard', 8, 0.24)
    return ('circle', final_num_humans, 0.26)
```

`sncp_ppo/train.py` `SCENARIO_HOLDOUT_CONFIG` (lines ~69-77):
```python
SCENARIO_HOLDOUT_CONFIG = {
    'easy':      (1, 0.13),
    'easy_plus': (3, 0.18),
    'medium':    (5, 0.22),
    'hard':      (5, 0.26),
    'extreme':   (10, 0.26),
    'circle':    (10, 0.26),
    'random':    (10, 0.26),
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest test_vec_curriculum.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/train.py test_vec_curriculum.py
git commit -m "v15: curriculum density ramp to N=10 + parity speeds + N=10 holdout"
```

---

### Task 4: Full regression + GPU smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected: ALL PASS. If any pre-existing test asserts the old reactive default / old speeds / old reward, update it to the v15 value (the only legitimate failures are those three categories) and re-run.

- [ ] **Step 2: Short end-to-end smoke (non-reactive + new reward + curriculum)**

Run:
```bash
python -m sncp_ppo.train --num_envs 4 --horizon 32 --total_steps 2560 \
  --eval_freq_updates 10 --num_humans 10 --seed 42 --lr 5e-5 \
  --holdout_scenarios easy hard circle --holdout_episodes 3 \
  --save_path checkpoints/_smoke_v15.pt
```
Expected: exit 0, runs through all phases (easy→circle, N up to 10), no NaN/shape error, avg_reward finite. Then delete the smoke artifacts:
```bash
rm -f checkpoints/_smoke_v15*.pt logs/training_*$(date +%Y%m%d)*.csv 2>/dev/null
```

- [ ] **Step 3: Commit (only if any test files were touched in Step 1)**

```bash
git add -A && git commit -m "v15: regression fixes for non-reactive/parity/reward changes"
```

---

### Task 5: Notebook → v15

**Files:**
- Modify: `sncp_ppo_colab.ipynb` (cells 13 markdown, 14 training, 17 eval)

- [ ] **Step 1: Update training cell (14)**

Set in the training cell: `SAVE_PATH = 'checkpoints/sncp_ppo_v15.pt'`, `TOTAL_STEPS = 2_500_000`, add `--num_humans 10` (final density), and `--holdout_scenarios easy hard circle`. Replace the leading comment block to describe v15 (non-reactive + reward rebalance + parity + density→10). (Use a JSON-edit helper script as in prior version bumps, or NotebookEdit; preserve the rest of the cell verbatim.)

- [ ] **Step 2: Update eval cell (17)**

Set `CHECKPOINT = 'checkpoints/sncp_ppo_v15.pt'`. Change the eval scenario→num_humans map so the sweep covers the new densities, e.g. `{'easy': 1, 'easy_plus': 3, 'medium': 5, 'hard': 10, 'extreme': 10}`, and update the NOTE comment to reference v14 (beeline) as the baseline and the nav-time-vs-density check.

- [ ] **Step 3: Update markdown cell (13)**

Rewrite the heading/description to v15: the goal (real avoidance, non-reactive), the decisions (parity, comfort -6, density→10), and the success criteria (nav-time rises with density; min-distance kept). Note checkpoints are NOT comparable to v14 (different task: cooperative→non-reactive).

- [ ] **Step 4: Validate notebook JSON**

Run: `python -c "import json,io; json.load(io.open('sncp_ppo_colab.ipynb',encoding='utf-8')); print('valid')"`
Expected: `valid`.

- [ ] **Step 5: Commit + push**

```bash
git add sncp_ppo_colab.ipynb
git commit -m "v15: notebook (non-reactive social-nav, density->10, parity)"
git push origin main
```

---

### Task 6: Update memory

**Files:** Modify `~/.claude/projects/.../memory/sncp-paper-vs-impl.md`

- [ ] **Step 1: Append a v15 entry** noting: the beeline diagnosis from the GIFs; the v15 design decisions (non-reactive + comfort -6 + parity + density→10, v16→20); that this is the genuine social-nav attempt; and the success criteria (nav-time-vs-density must rise). No commit needed (memory is outside the repo).

---

## Post-implementation (user runs on Colab)
1. cell-4 (git pull) → cell-14 (A100, ~4h, v15).
2. Watch holdout `circle` (N=10) and whether the robot freezes (success crashing toward 0 = comfort too strong → lower to -4; tune).
3. After: density sweep + trajectory GIFs; verify nav-time RISES with density (the decisive contrast with v14's flat 121.5) and min-distance stays above ~0.5 m.
4. If v15 shows real avoidance at N≤10 without freezing → v16 extends curriculum to N=15–20.
