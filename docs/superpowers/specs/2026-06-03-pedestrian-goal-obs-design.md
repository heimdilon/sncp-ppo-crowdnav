# Pedestrian Goal-Direction Observation (v12) — Design Spec

**Date:** 2026-06-03
**Status:** Approved (brainstorming) → ready for implementation plan
**Builds on:** v11 (LR-fixed vectorized training, hard randomized eval = 28%)

## Problem (diagnosed, not guessed)

Collision forensics on the latest v11 checkpoint (40 hard episodes, 23 collisions):
- **100%** of collisions: robot at full speed (0.246/0.26, std=0) — the policy never brakes.
- **87%** of collisions: the *pedestrian* is moving toward the robot (closing speed +0.27) — the robot is caught, not charging in.
- Only **22%** crowded (≥2 humans within 1m); **<22%** head-down charging.
- Success degrades monotonically with crowd size (easy 82% → hard 28% → extreme 21%).

Conclusion: the dominant failure is a **prediction gap** — the policy cannot anticipate where pedestrians are going, so it gets caught when a Social-Force pedestrian moves into its path. v8 added relative velocity (instantaneous motion), but SFM pedestrians change heading to dodge/avoid, so instantaneous velocity is not enough; the policy needs each pedestrian's **goal direction** (true intent).

## Goal

Add each pedestrian's goal-direction unit vector (robot-local frame) to the observation, so the policy can anticipate pedestrian trajectories. `spatial_edges` grows 4 → 6 dims per pedestrian.

## Decision (locked)

Per-pedestrian goal representation = **goal-direction unit vector** (2 dims), rotated into the robot-local frame. Chosen over raw goal-relative vector (scale varies 0-8m → normalization issues) and over dir+distance (3 dims, more than needed). Unit vector is scale-free, stable in [-1,1], and directly answers "where is this pedestrian heading."

## Components

### A. Observation — `crowd_sim/crowd_env.py::_get_obs`
`spatial_edges` per pedestrian: **4 → 6**:
`[dx_local, dy_local, rel_vx_local, rel_vy_local, goal_dir_x_local, goal_dir_y_local]`

```python
goal_vec_x = self.humans_gx[i] - self.humans_px[i]
goal_vec_y = self.humans_gy[i] - self.humans_py[i]
gnorm = np.hypot(goal_vec_x, goal_vec_y) + 1e-9
gdx, gdy = goal_vec_x / gnorm, goal_vec_y / gnorm          # unit vector
spatial_edges[i, 4] =  gdx * cos_t + gdy * sin_t           # same rotation as pos/vel
spatial_edges[i, 5] = -gdx * sin_t + gdy * cos_t
```
The first 4 columns are unchanged (backward-compatible layout). `observation_space.spatial_edges` shape → `(num_humans, 6)`. `humans_gx/gy` already exist in the env.

### B. Model — `sncp_ppo/models.py`
`spatial_ltc` `input_size` **4 → 6** (line ~42). `forward` reshape `(B*H, 1, 4)` → `(B*H, 1, 6)`. Everything downstream (LTC 32→256, attention, fusion) is unchanged.

### C. Hidden hardcode — `sncp_ppo/ppo.py:614` (CRITICAL — easy to miss)
`update_vectorized` allocates the BPTT window tensor with a hardcoded spatial dim:
```python
se = torch.zeros(num_win, S, num_humans, 4, device=device)   # MUST become 6
```
This is NOT auto-derived. It was silently updated 2→4 in v8; v12 must update 4→6 or the vectorized PPO update will slice spatial data at the wrong width (crash or silent corruption). Add a comment so future generations don't miss it. (`vec_buffer.py:84` and `train.py:233` matched the grep but are only a comment and the curriculum tuple respectively — no code change needed there.)

### D. New architecture generation
v8/v11 checkpoints (4-dim spatial) cannot be loaded by this code (dim mismatch), exactly like the v7→v8 transition. v11 results are recorded (hard 28% baseline) for comparison. This is acceptable and expected.

## Testing (TDD)
1. `reset` → `obs['spatial_edges'].shape == (N, 6)`.
2. New `test_env_goaldir.py`: a pedestrian whose goal is due-north (in global frame, robot facing east) yields the correct rotated goal-dir; the goal-dir sub-vector has unit norm (≈1.0); the first 4 columns equal the v8 pos+rel_vel values for the same state (backward-compatible).
3. `SNCPPolicy` forward accepts `(B, H, 6)` spatial, returns mu (B,2), std (B,2), value (B,1) — shapes preserved.
4. Update smoke tests: `test_env.py` (shape 4→6), `test_model.py` (dummy spatial 4→6), `test_env_velocity.py` (shape asserts 4→6; keep the velocity-correctness assertions).
5. PPO update path: a short vectorized smoke (`--num_envs 2 --total_steps ~1200`) AND a single-env smoke (`--num_envs 1 --episodes ~6`) both run without crash and produce healthy kl/entropy — this exercises the `ppo.py:614` 6-dim change end-to-end on GPU.

## Files
- `crowd_sim/crowd_env.py` (obs + observation_space)
- `sncp_ppo/models.py` (spatial_ltc input 4→6, forward reshape)
- `sncp_ppo/ppo.py` (line 614 hardcode 4→6 + comment)
- `test_env_goaldir.py` (new)
- `test_env.py`, `test_model.py`, `test_env_velocity.py` (shape 4→6)

## Out of scope (future, if v12 still falls short)
- Near-miss / proximity reward penalty (the "no braking" finding) — a natural follow-up if goal-obs alone doesn't lift hard; deferred so we can isolate the effect of the observation change.
- Deriving spatial_dim instead of hardcoding it in ppo.py (would prevent this bug class permanently; larger refactor, separate effort).
- LTC→GRU architecture ablation.

## Training & success criterion
After this lands: Colab v12 run with the same vectorized v11 command (only `--save_path` → v12). **Success = randomized-hard eval > v11's 28%** (independent 100-ep, 4-scenario, seed 100). If the diagnosis is right (prediction gap), collisions should drop and hard should rise. If it doesn't move, the next lever is the reward (near-miss) — but we test one variable at a time.
