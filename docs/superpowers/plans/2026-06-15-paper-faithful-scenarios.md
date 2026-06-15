# Paper-Faithful Scenario Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the paper's scattered/large-arena scenarios (`paper_standard`, `paper_challenging`) plus a configurable collision threshold to `CrowdSimEnv`, so we can train/eval in the paper's actual regime and reproduce its 93–95%.

**Architecture:** Additive, default-preserving. New env constructor knobs (`collision_threshold`, `arena_size`, `sense_range`) default to current behaviour. New `paper_*` scenario branches in `reset()` use square-crossing (scattered start, no center funnel) with a fixed bottom→top robot. `train.py`/`evaluate_policy_report.py` get pass-throughs. Existing scenarios (`hard`, `circle`, …) stay byte-identical.

**Tech Stack:** Python, NumPy, gymnasium, pytest (Windows: `--basetemp=.pytest_tmp`), ruff.

---

## File Structure
- `crowd_sim/crowd_env.py` — constructor knobs; collision check uses threshold; `paper_*` placement branches in `reset()`.
- `sncp_ppo/train.py` — `SCENARIO_HOLDOUT_CONFIG` paper entries; `--collision_threshold` flag; `make_env` pass-through.
- `evaluate_policy_report.py` — `--collision_threshold` pass-through (for the full-faithful eval).
- `test_paper_scenarios.py` — new tests (placement, threshold, default preservation).

---

### Task 1: Configurable collision threshold + env knobs

**Files:**
- Modify: `crowd_sim/crowd_env.py` (constructor signature + body; collision check in `step()`)
- Test: `test_paper_scenarios.py`

- [ ] **Step 1: Write the failing test**

```python
# test_paper_scenarios.py
import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv


def test_collision_threshold_defaults_to_radii_sum():
    env = CrowdSimEnv(num_humans=3, scenario='hard', human_motion_model='orca')
    assert env.collision_threshold == env.robot_radius + env.human_radius  # 0.6


def test_collision_threshold_is_configurable():
    env = CrowdSimEnv(num_humans=3, scenario='hard', human_motion_model='orca',
                      collision_threshold=0.3)
    assert env.collision_threshold == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_paper_scenarios.py -q --basetemp=.pytest_tmp`
Expected: FAIL — `AttributeError: 'CrowdSimEnv' object has no attribute 'collision_threshold'`.

- [ ] **Step 3: Add constructor args**

In `crowd_sim/crowd_env.py`, change the `__init__` signature (currently ends with `human_goal_noise=0.0,`):

```python
        human_vpref_override=None,
        human_goal_noise=0.0,
        collision_threshold=None,
        arena_size=None,
        sense_range=None,
    ):
```

After the human physical parameters block (the lines `self.human_radius = 0.3` / `self.human_vpref = 0.5`), add:

```python
        # Collision threshold: distance below which robot-human contact counts as
        # a collision. Default = robot_radius + human_radius (0.6), preserving
        # current behaviour. The paper uses d_col = 0.3 (Table 1).
        self.collision_threshold = (
            collision_threshold if collision_threshold is not None
            else self.robot_radius + self.human_radius
        )
        # Paper scenarios scale the arena and sense range with density; None keeps
        # the legacy circle-crossing layout. sense_range is recorded for paper
        # presets (obs masking by range is intentionally out of scope here).
        self.arena_size = arena_size
        self.sense_range = sense_range
```

- [ ] **Step 4: Use the threshold in the collision check**

In `step()`, replace:

```python
        collision = d_min < (self.robot_radius + self.human_radius)
```

with:

```python
        collision = d_min < self.collision_threshold
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test_paper_scenarios.py -q --basetemp=.pytest_tmp`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add crowd_sim/crowd_env.py test_paper_scenarios.py
git commit -m "Add configurable collision_threshold + arena/sense knobs to CrowdSimEnv"
```

---

### Task 2: Paper scenario placement (standard + challenging)

**Files:**
- Modify: `crowd_sim/crowd_env.py` (`reset()` — scenario mapping + placement branch)
- Test: `test_paper_scenarios.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to test_paper_scenarios.py

def test_paper_standard_layout():
    env = CrowdSimEnv(num_humans=5, scenario='paper_standard', human_motion_model='orca')
    env.reset(seed=0)
    # robot fixed bottom -> top
    assert (env.robot_px, env.robot_py) == (0.0, -4.0)
    assert (env.robot_gx, env.robot_gy) == (0.0, 4.0)
    half = 5.0  # 10x10 arena
    assert np.all(np.abs(env.humans_px) <= half) and np.all(np.abs(env.humans_py) <= half)
    # scattered, NOT all on the radius-4 circle (the antipodal regime)
    radii = np.hypot(env.humans_px, env.humans_py)
    assert np.std(radii) > 0.3, f"humans look circle-placed: radii={radii}"
    # no human starts inside the collision threshold of the robot start
    d_start = np.hypot(env.humans_px - env.robot_px, env.humans_py - env.robot_py)
    assert np.all(d_start >= env.collision_threshold)
    assert env.human_vpref == 1.0  # parity


def test_paper_challenging_scales_arena():
    for n in (10, 15, 20):
        env = CrowdSimEnv(num_humans=n, scenario='paper_challenging', human_motion_model='orca')
        env.reset(seed=1)
        assert (env.robot_px, env.robot_py) == (0.0, -6.0)
        assert (env.robot_gx, env.robot_gy) == (0.0, 6.0)
        half = 7.5  # 15x15 arena
        assert np.all(np.abs(env.humans_px) <= half) and np.all(np.abs(env.humans_py) <= half)
        assert env.humans_px.shape == (n,)


def test_existing_hard_scenario_unchanged():
    # Default preservation: 'hard' still uses radius-4 circle-crossing placement.
    env = CrowdSimEnv(num_humans=8, scenario='hard', human_motion_model='orca',
                      randomize_layout=True)
    env.reset(seed=0)
    radii = np.hypot(env.humans_px, env.humans_py)
    assert np.allclose(radii, 4.0, atol=1e-9), f"hard placement changed: radii={radii}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_paper_scenarios.py -k paper_standard -q --basetemp=.pytest_tmp`
Expected: FAIL — paper_standard falls through to the legacy `random` branch (robot not (0,-4), humans uniform(-5,5)), so the robot-position and parity assertions fail.

- [ ] **Step 3: Add paper scenario mapping**

In `reset()`, in the scenario `if/elif` chain, add before the final `else:` (the `else: self.human_vpref = 0.26; scenario_type = self.scenario` block):

```python
        elif self.scenario == 'paper_standard':
            self.human_vpref = 1.0  # parity with the 1.0 m/s robot
            scenario_type = 'paper'
            self._paper_arena = self.arena_size if self.arena_size is not None else 10.0
            self._paper_robot_y = 4.0
        elif self.scenario == 'paper_challenging':
            self.human_vpref = 1.0
            scenario_type = 'paper'
            self._paper_arena = self.arena_size if self.arena_size is not None else 15.0
            self._paper_robot_y = 6.0
```

- [ ] **Step 4: Add the paper placement branch**

In `reset()`, the placement is currently `if scenario_type == 'circle': ... else: # random ...`. Insert a new branch between them, i.e. change `else: # random` to `elif scenario_type == 'paper':` block first, then keep the existing `else: # random`:

```python
        elif scenario_type == 'paper':
            # Paper-faithful square-crossing: robot fixed bottom -> top, humans
            # scattered uniformly across the (large) arena. NOT antipodal on a
            # circle, so paths do NOT all funnel through the center.
            half = self._paper_arena / 2.0
            self.robot_px = 0.0
            self.robot_py = -self._paper_robot_y
            self.robot_gx = 0.0
            self.robot_gy = self._paper_robot_y
            self.robot_theta = np.pi / 2.0  # facing the goal (north)
            min_sep = self.robot_radius + self.human_radius + 0.5
            gnoise = 1.0
            for i in range(self.num_humans):
                px, py = 0.0, 0.0
                for _ in range(200):
                    px = self.np_random.uniform(-half, half)
                    py = self.np_random.uniform(-half, half)
                    d_start = np.hypot(px - self.robot_px, py - self.robot_py)
                    d_goal = np.hypot(px - self.robot_gx, py - self.robot_gy)
                    if d_start < min_sep or d_goal < min_sep:
                        continue
                    if i == 0 or np.min(np.hypot(
                            px - self.humans_px[:i], py - self.humans_py[:i])) > min_sep:
                        break
                self.humans_px[i] = px
                self.humans_py[i] = py
                nx = self.np_random.uniform(-gnoise, gnoise)
                ny = self.np_random.uniform(-gnoise, gnoise)
                self.humans_gx[i] = np.clip(-px + nx, -half, half)
                self.humans_gy[i] = np.clip(-py + ny, -half, half)
                dx = self.humans_gx[i] - self.humans_px[i]
                dy = self.humans_gy[i] - self.humans_py[i]
                self.humans_theta[i] = np.arctan2(dy, dx)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test_paper_scenarios.py -q --basetemp=.pytest_tmp`
Expected: PASS (all tests, including `test_existing_hard_scenario_unchanged`).

- [ ] **Step 6: Commit**

```bash
git add crowd_sim/crowd_env.py test_paper_scenarios.py
git commit -m "Add paper_standard/paper_challenging square-crossing scenarios"
```

---

### Task 3: train.py integration (holdout config + collision_threshold flag)

**Files:**
- Modify: `sncp_ppo/train.py` (`SCENARIO_HOLDOUT_CONFIG`; `make_env`; `build_parser`; the env builds that call `make_env`/`CrowdSimEnv`)
- Test: `test_paper_scenarios.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to test_paper_scenarios.py
from sncp_ppo.train import SCENARIO_HOLDOUT_CONFIG, make_env, build_parser


def test_paper_scenarios_in_holdout_config():
    assert SCENARIO_HOLDOUT_CONFIG['paper_standard'][1] == 1.0      # parity vpref
    assert SCENARIO_HOLDOUT_CONFIG['paper_challenging'][1] == 1.0


def test_train_parser_has_collision_threshold():
    args = build_parser().parse_args(['--collision_threshold', '0.3'])
    assert args.collision_threshold == 0.3
    assert build_parser().parse_args([]).collision_threshold is None


def test_make_env_builds_paper_scenario_with_threshold():
    env = make_env(num_humans=10, scenario='paper_challenging', seed=0,
                   collision_threshold=0.3)()
    assert env.scenario == 'paper_challenging'
    assert env.collision_threshold == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_paper_scenarios.py -k "holdout_config or collision_threshold or make_env_builds" -q --basetemp=.pytest_tmp`
Expected: FAIL — `KeyError: 'paper_standard'` / unrecognized `--collision_threshold` / `make_env` has no `collision_threshold` kwarg.

- [ ] **Step 3: Add paper entries to SCENARIO_HOLDOUT_CONFIG**

In `sncp_ppo/train.py`, find the `SCENARIO_HOLDOUT_CONFIG = {` dict (near line 63) and add entries (keep existing entries):

```python
    'paper_standard': (5, 1.0),
    'paper_challenging': (10, 1.0),
```

- [ ] **Step 4: Add `collision_threshold` to `make_env` and pass it through**

Change the `make_env` signature (currently `def make_env(num_humans, scenario, seed, comfort_coeff=6.0, max_time=50.0, ...):`) to add `collision_threshold=None`, and pass it to the `CrowdSimEnv(...)` constructed inside (add `collision_threshold=collision_threshold,` to the constructor call):

The current `make_env` (train.py:214) has explicit params (no `**kwargs`). Add `collision_threshold=None` to the signature and thread it into the `CrowdSimEnv(...)` call:

```python
def make_env(num_humans, scenario, seed, comfort_coeff=6.0, max_time=50.0,
             robot_vpref=0.26, human_vpref_override=None, human_goal_noise=0.0,
             human_motion_model='orca', collision_threshold=None):
    """Factory for a single CrowdSimEnv, used by SyncVectorEnv."""
    def _thunk():
        env = CrowdSimEnv(
            num_humans=num_humans,
            scenario=scenario,
            comfort_coeff=comfort_coeff,
            max_time=max_time,
            robot_vpref=robot_vpref,
            human_vpref_override=human_vpref_override,
            human_goal_noise=human_goal_noise,
            human_motion_model=human_motion_model,
            collision_threshold=collision_threshold,
        )
        env.reset(seed=seed)
        return env
    return _thunk
```

- [ ] **Step 5: Add the CLI flag**

In `build_parser()`, near the other env flags (e.g. after `--max_time`), add:

```python
    parser.add_argument('--collision_threshold', type=float, default=None,
                        help='Robot-human collision distance. Default = robot_radius '
                             '+ human_radius (0.6). The paper uses 0.3 (Table 1).')
```

- [ ] **Step 6: Thread the flag into the training env build**

In `train()`, find the main `CrowdSimEnv(` build (near line 271) and add `collision_threshold=args.collision_threshold,` to it. If training uses `make_env` for the vectorized envs, pass `collision_threshold=args.collision_threshold` there too.

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest test_paper_scenarios.py -q --basetemp=.pytest_tmp`
Expected: PASS (all).

- [ ] **Step 8: Commit**

```bash
git add sncp_ppo/train.py test_paper_scenarios.py
git commit -m "Wire paper scenarios + --collision_threshold into training"
```

---

### Task 4: Full regression + lint

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q --basetemp=.pytest_tmp`
Expected: PASS (existing 184 + the new paper-scenario tests), 1 pre-existing warning.

- [ ] **Step 2: Lint**

Run: `python -m ruff check crowd_sim/crowd_env.py sncp_ppo/train.py test_paper_scenarios.py`
Expected: All checks passed.

- [ ] **Step 3: Local smoke (paper scenarios run end-to-end)**

```bash
python -c "import numpy as np; from crowd_sim.crowd_env import CrowdSimEnv;\
 env=CrowdSimEnv(num_humans=15, scenario='paper_challenging', human_motion_model='orca');\
 env.reset(seed=0);\
 [env.step(np.array([0.5,0.1],dtype=np.float32)) for _ in range(50)];\
 print('paper_challenging 50 steps OK')"
```
Expected: prints OK, no exception.

- [ ] **Step 4: Final commit (if anything uncommitted)**

```bash
git add -A
git commit -m "Paper-faithful scenarios: full suite green + smoke"
```

---

## Post-implementation (run separately, not code)
- **Colab geometry probe (~500k–1M):** `--fixed_scenario paper_challenging --num_humans 10 --human_vpref_override 1.0` (keep current d_col 0.6 / comfort 6 / max_time), eval `--scenario paper_challenging`. Measure success at N=5/10/15/20 vs the paper's 0.94.
- **Full-faithful run (if geometry probe is promising):** add `--collision_threshold 0.3 --comfort_coeff 2.0 --max_time 12.5`.
- **Eval `--collision_threshold` follow-up:** the geometry-probe eval needs only `evaluate_policy_report.py --scenario paper_challenging` (default d_col 0.6). The full-faithful eval at d_col 0.3 needs a small `run_report` (`sncp_ppo/eval_report.py`) pass-through for `collision_threshold` → its env build; plan that when the full-faithful run is set up.
