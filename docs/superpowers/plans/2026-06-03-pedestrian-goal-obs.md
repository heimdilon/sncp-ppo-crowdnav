# Pedestrian Goal-Direction Observation (v12) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add each pedestrian's goal-direction unit vector (robot-local frame) to `spatial_edges` (4→6 dims) so the policy can anticipate pedestrian trajectories, addressing the diagnosed prediction-gap collisions.

**Architecture:** Three production edits — env observation (`crowd_env.py`), policy spatial encoder input size (`models.py`), and a hidden hardcoded spatial dim in the vectorized PPO update (`ppo.py:614`). Plus test updates. The first 4 columns of `spatial_edges` are unchanged (backward-compatible); columns 4-5 are the new goal-direction unit vector. This is a new architecture generation: v8/v11 checkpoints (4-dim) cannot load.

**Tech Stack:** PyTorch, NumPy, Gymnasium, ncps (LTC), pytest.

---

## File Structure

- `crowd_sim/crowd_env.py` — `_get_obs` builds 6-dim spatial_edges; `observation_space` shape 4→6.
- `sncp_ppo/models.py` — `spatial_ltc` input_size 4→6; `forward` reshape 4→6.
- `sncp_ppo/ppo.py` — line ~614 BPTT window tensor spatial dim 4→6 (hidden hardcode).
- `test_env_goaldir.py` (new) — goal-direction correctness + unit norm + backward-compat of first 4 cols.
- `test_env.py`, `test_model.py`, `test_env_velocity.py` — shape asserts / dummy tensors 4→6.

**Order rationale:** Env first (Task 1) — it's the source of the new data and fully unit-testable without the model. Then model (Task 2) consumes it. Then the ppo.py hardcode + full smoke (Task 3) — the integration that's easy to miss. Test-file updates fold into the tasks that change the shape they assert.

---

## Task 1: 6-dim observation in the environment

**Files:**
- Modify: `crowd_sim/crowd_env.py` (`_get_obs` spatial_edges loop ~line 229-239; `observation_space` ~line 52)
- Create: `test_env_goaldir.py`
- Modify: `test_env.py` (line 17), `test_env_velocity.py` (lines 19, 58)

- [ ] **Step 1: Write the failing test**

Create `test_env_goaldir.py` with EXACTLY this content:

```python
import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv


def test_spatial_edges_has_6_dims():
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    obs, _ = env.reset(seed=1)
    assert obs['spatial_edges'].shape == (5, 6)


def test_goal_dir_is_unit_vector():
    """Columns 4-5 (goal direction) must be a unit vector per pedestrian."""
    env = CrowdSimEnv(num_humans=5, scenario='hard')
    obs, _ = env.reset(seed=1)
    gdir = obs['spatial_edges'][:, 4:6]
    norms = np.hypot(gdir[:, 0], gdir[:, 1])
    assert np.allclose(norms, 1.0, atol=1e-4), f"goal-dir norms not unit: {norms}"


def test_goal_dir_rotation_correct():
    """Robot facing east (theta=0), a pedestrian whose goal is due north must
    have local goal-dir (0, 1); robot facing north must give (1, 0)."""
    env = CrowdSimEnv(num_humans=1, scenario='hard')
    env.reset(seed=1)
    # Force a known geometry: pedestrian at origin, goal due north.
    env.humans_px[0], env.humans_py[0] = 0.0, 0.0
    env.humans_gx[0], env.humans_gy[0] = 0.0, 5.0
    env.robot_px, env.robot_py = 2.0, 0.0
    env.robot_theta = 0.0  # facing east
    obs = env._get_obs()
    gx, gy = obs['spatial_edges'][0, 4], obs['spatial_edges'][0, 5]
    assert np.allclose([gx, gy], [0.0, 1.0], atol=1e-4), f"east-facing: got ({gx},{gy})"
    env.robot_theta = np.pi / 2  # facing north
    obs = env._get_obs()
    gx, gy = obs['spatial_edges'][0, 4], obs['spatial_edges'][0, 5]
    assert np.allclose([gx, gy], [1.0, 0.0], atol=1e-4), f"north-facing: got ({gx},{gy})"


def test_first_four_cols_unchanged():
    """The position + relative-velocity columns (0-3) must keep the v8 layout."""
    env = CrowdSimEnv(num_humans=3, scenario='medium')
    env.reset(seed=2)
    obs = env._get_obs()
    se = obs['spatial_edges']
    assert se.shape == (3, 6)
    # Recompute expected pos cols directly from state, robot-local frame.
    cos_t, sin_t = np.cos(env.robot_theta), np.sin(env.robot_theta)
    for i in range(3):
        dx = env.humans_px[i] - env.robot_px
        dy = env.humans_py[i] - env.robot_py
        exp_x = dx * cos_t + dy * sin_t
        exp_y = -dx * sin_t + dy * cos_t
        assert np.allclose(se[i, 0], exp_x, atol=1e-4)
        assert np.allclose(se[i, 1], exp_y, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_env_goaldir.py -v`
Expected: FAIL — `test_spatial_edges_has_6_dims` asserts (5,6) but gets (5,4); others fail on column index 4/5 out of bounds.

- [ ] **Step 3: Implement the 6-dim observation**

In `crowd_sim/crowd_env.py`, change the `observation_space` spatial_edges shape (around line 52) from `shape=(self.num_humans, 4)` to `shape=(self.num_humans, 6)`:

```python
            'spatial_edges': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_humans, 6), dtype=np.float32),
```

Then in `_get_obs` replace the spatial_edges block (the `spatial_edges = np.zeros((self.num_humans, 4)...)` loop) with:

```python
        # Spatial edges: per-pedestrian local-frame position, relative velocity
        # (pedestrian - robot), AND goal-direction unit vector. All rotated into
        # the robot's local frame. Layout per row:
        #   [dx_local, dy_local, rel_vx_local, rel_vy_local, goal_dir_x, goal_dir_y]
        # Goal direction gives the policy each pedestrian's INTENT (where it is
        # heading) so it can anticipate trajectories instead of reacting late.
        spatial_edges = np.zeros((self.num_humans, 6), dtype=np.float32)
        for i in range(self.num_humans):
            dx_global = self.humans_px[i] - self.robot_px
            dy_global = self.humans_py[i] - self.robot_py
            dvx_global = self.humans_vx[i] - self.robot_vx
            dvy_global = self.humans_vy[i] - self.robot_vy
            # Goal-direction unit vector (global), scale-free intent signal.
            gvx = self.humans_gx[i] - self.humans_px[i]
            gvy = self.humans_gy[i] - self.humans_py[i]
            gnorm = np.hypot(gvx, gvy) + 1e-9
            gdx, gdy = gvx / gnorm, gvy / gnorm
            # Rotate position, relative velocity, and goal direction to local frame
            spatial_edges[i, 0] = dx_global * cos_t + dy_global * sin_t
            spatial_edges[i, 1] = -dx_global * sin_t + dy_global * cos_t
            spatial_edges[i, 2] = dvx_global * cos_t + dvy_global * sin_t
            spatial_edges[i, 3] = -dvx_global * sin_t + dvy_global * cos_t
            spatial_edges[i, 4] = gdx * cos_t + gdy * sin_t
            spatial_edges[i, 5] = -gdx * sin_t + gdy * cos_t
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_env_goaldir.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Update the env smoke-test shape asserts**

In `test_env.py` line 17, change `(5, 2)` history... actually change the current assert to:
```python
    assert obs['spatial_edges'].shape == (5, 6)
```

In `test_env_velocity.py`: line 19 change assert to `(5, 6)`; the test function name `test_spatial_edges_has_4_dims` may stay (it now documents history) but update its assert; line 58 dummy tensor `torch.randn(2, 5, 4)` → `torch.randn(2, 5, 6)`. Keep the velocity-correctness assertions (columns 2-3) intact.

- [ ] **Step 6: Run env tests**

Run: `python -m pytest test_env.py test_env_goaldir.py test_env_velocity.py -v`
Expected: PASS. (`test_env_velocity` still checks velocity columns 2-3 correctly because the layout there is unchanged.)

- [ ] **Step 7: Commit**

```bash
git add crowd_sim/crowd_env.py test_env_goaldir.py test_env.py test_env_velocity.py
git commit -m "feat(obs): add pedestrian goal-direction unit vector (spatial_edges 4->6)"
```

---

## Task 2: Policy spatial encoder accepts 6-dim input

**Files:**
- Modify: `sncp_ppo/models.py` (spatial_ltc `input_size` ~line 42; `forward` reshape ~line 145)
- Modify: `test_model.py` (line 15)

- [ ] **Step 1: Update the model smoke test (failing)**

In `test_model.py` line 15, change the dummy spatial tensor:
```python
        'spatial_edges': torch.randn(batch_size, num_humans, 6),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_model.py -v`
Expected: FAIL — a shape/matmul error inside `spatial_ltc` because it still expects input_size=4 but receives 6.

- [ ] **Step 3: Implement the 6-dim spatial encoder**

In `sncp_ppo/models.py`, change the spatial LTC construction (around line 41-42):
```python
        # input_size=6: [dx, dy, rel_vx, rel_vy, goal_dir_x, goal_dir_y] per human
        self.spatial_ltc = LTC(input_size=6, units=self.spatial_wiring)
```

In `forward`, change the spatial reshape (around line 145) from `reshape(batch_size * num_humans, 1, 4)` to:
```python
        spatial_input = spatial_edges.reshape(batch_size * num_humans, 1, 6)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_model.py -v`
Expected: PASS — forward returns mu (4,2), std (4,2), value (4,1).

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/models.py test_model.py
git commit -m "feat(model): spatial_ltc input 4->6 for goal-direction obs"
```

---

## Task 3: Fix the hidden spatial-dim hardcode in the vectorized PPO update + full smoke

**Files:**
- Modify: `sncp_ppo/ppo.py` (line ~614)

- [ ] **Step 1: Update the hardcoded spatial dim**

In `sncp_ppo/ppo.py`, find this line inside `update_vectorized` (around line 614):
```python
            se = torch.zeros(num_win, S, num_humans, 4, device=device)
```
Change the `4` to `6` and add a guarding comment:
```python
            # spatial dim = 6 (pos + rel_vel + goal_dir); must match crowd_env
            # _get_obs spatial_edges width and models.SNCPPolicy spatial_ltc input.
            se = torch.zeros(num_win, S, num_humans, 6, device=device)
```

- [ ] **Step 2: Run the full unit-test suite (regression)**

Run: `python -m pytest test_env_goaldir.py test_env.py test_model.py test_env_velocity.py test_vec_curriculum.py test_vec_buffer.py test_vec_gae.py test_train_eta.py test_env_randomization.py -q`
Expected: all PASS. (The vec_buffer/vec_gae tests build their own spatial tensors; confirm they either use the env or a matching width — if any hardcodes 4, see Step 3.)

- [ ] **Step 3: Check vec tests for a 4-dim spatial hardcode**

Run: `grep -n "1, 6)\|num_humans, 4\|H, 4\|, 4)" test_vec_buffer.py test_vec_gae.py`
Expected: if any test constructs a spatial tensor with width 4, it must move to 6 to match. If a test fails in Step 2 due to a width-4 spatial tensor, update that tensor to width 6 and re-run. If Step 2 was all-green, this is a no-op (those tests are dim-agnostic).

- [ ] **Step 4: Vectorized GPU smoke (exercises ppo.py:614 end-to-end)**

Run: `python -m sncp_ppo.train --num_envs 2 --horizon 48 --total_steps 1200 --eval_freq_updates 2 --num_humans 5 --holdout_episodes 2 --best_warmup_evals 0 --best_min_success_threshold 0.0 --save_path checkpoints/_v12_smoke.pt`
Expected: exit 0, prints "Vectorized training completed!", ≥1 curriculum shift, a checkpoint saved, NO traceback/shape error. Then clean up:
`rm -f checkpoints/_v12_smoke*.pt` and the smoke's `logs/training_*.csv`.

- [ ] **Step 5: Single-env smoke (legacy path still works with 6-dim obs)**

Run: `python -m sncp_ppo.train --num_envs 1 --episodes 6 --eval_freq 100 --holdout_episodes 2 --log_freq 1 --save_path checkpoints/_v12_legacy.pt`
Expected: exit 0, single-env "Ep N/6" format, no shape error. Then: `rm -f checkpoints/_v12_legacy*.pt` and its `logs/training_*.csv`.

- [ ] **Step 6: Confirm protected files untouched**

Run: `git diff main..HEAD --stat -- sncp_ppo/vec_buffer.py sncp_ppo/train.py`
Expected: EMPTY (this feature touches crowd_env.py, models.py, ppo.py only; vec_buffer.py and train.py are NOT modified). If non-empty, revert unintended changes.

- [ ] **Step 7: Commit**

```bash
git add sncp_ppo/ppo.py
git commit -m "fix(ppo): vectorized BPTT spatial dim 4->6 for goal-direction obs"
```

---

## Task 4: Update the Colab notebook to v12

**Files:**
- Modify: `sncp_ppo_colab.ipynb` (training cell `SAVE_PATH` → v12; eval cell `CHECKPOINT` → v12)

- [ ] **Step 1: Point training + eval cells at v12**

In `sncp_ppo_colab.ipynb`, in the training cell change `SAVE_PATH = 'checkpoints/sncp_ppo_v11.pt'` to `SAVE_PATH = 'checkpoints/sncp_ppo_v12.pt'`. In the eval cell change `CHECKPOINT = 'checkpoints/sncp_ppo_v11.pt'` to `CHECKPOINT = 'checkpoints/sncp_ppo_v12.pt'`. (Use the NotebookEdit tool; read the notebook first.) Add a one-line note in the training markdown cell: "v12 adds pedestrian goal-direction to the observation (spatial 4→6); v11 checkpoints cannot be loaded by this code."

- [ ] **Step 2: Validate notebook JSON**

Run: `python -c "import json; nb=json.load(open('sncp_ppo_colab.ipynb',encoding='utf-8')); print('valid, cells:', len(nb['cells']))"`
Expected: "valid, cells: 31"

- [ ] **Step 3: Commit**

```bash
git add sncp_ppo_colab.ipynb
git commit -m "docs(colab): point notebook to v12 (goal-direction obs)"
```

---

## Self-Review Notes

- **Spec coverage:** Component A (obs 4→6) → Task 1; Component B (model input 4→6) → Task 2; Component C (ppo.py:614 hardcode) → Task 3; new-generation note → notebook Task 4. All 5 spec tests covered: shape (T1.1), unit norm (T1.2), rotation (T1.3), backward-compat cols 0-3 (T1.4), model forward (T2), PPO smoke both paths (T3.4-5).
- **Type consistency:** spatial width is `6` everywhere (crowd_env shape, models input_size + reshape, ppo.py:614). Test tensors all width 6.
- **Out of scope confirmed:** no reward change, no vec_buffer/train.py change (verified in T3.6), no LTC→GRU.
- **Known check:** Task 3 Step 3 guards against a width-4 hardcode in the vec tests; if Step 2 is green this is a no-op, but it's called out so it isn't a silent gap.
