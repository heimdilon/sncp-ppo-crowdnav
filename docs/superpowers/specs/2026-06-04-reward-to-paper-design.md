# Restore Reward Function to the Paper (v13) — Design Spec

**Date:** 2026-06-04
**Status:** Approved (brainstorming) → ready for implementation plan
**Builds on:** v12 (goal-direction obs; training curve was identical to v11 → observation was NOT the bottleneck)

## Problem

v12 added pedestrian goal-direction to the observation, but its training curve is byte-for-byte the shape of v11's (easy best ~66% by update 70, hard holdout stuck ~48% by update 230, no best updates through medium/hard/circle). The prediction-gap hypothesis is disproven: enriching the observation did not move hard.

Two independent lines of evidence point to the REWARD FUNCTION instead:
1. **Collision forensics (v11):** robot at 100% full speed at impact (never brakes), 87% of collisions are a pedestrian moving into the robot. The policy ignores social proximity.
2. **Paper comparison (Ao et al. 2026, the source paper):** the architecture matches ours, but our reward drifted badly. The paper reaches 94% on the hard (10-20 human) scenario; we reach ~28%.

The paper's comfort penalty is `-2·I_sp`; ours is `-0.5·I_sp/N` — at N=5 that is **20× weaker**. The robot barely feels social pressure, so it charges through crowds.

## Goal

Restore the reward function to the paper's recipe (Eq 17-20, Table 1), adapting only the episode time limit to our robot's physical speed.

## Changes — all in `crowd_sim/crowd_env.py`

| # | Parameter | Current | Paper | Notes |
|---|-----------|:-------:|:-----:|-------|
| 1 | Goal arrival reward | `r_g = 50.0` | **`20.0`** | Eq 18 |
| 2 | Goal dense (approach) coef | `5.0·Δd` | **`2.0·Δd`** | Eq 18: `2(‖pᵗ⁻¹−pg‖−‖pᵗ−pg‖)` |
| 3 | Collision penalty | `r_c = -25.0` | **`-20.0`** | Eq 19 |
| 4 | Comfort penalty | `-0.5·I_sp/N` | **`-2.0·I_sp`** | Eq 20 (drop the ÷N; 20× stronger at N=5) |
| 5 | Max nav time | `max_time=60.0` | **`35.0`** | Adapted, see below |

### Why max_time = 35.0 (not the paper's 12.5s)
The paper's `t_lim=12.5s` is for a robot with **max speed 1.0 m/s** (paper p.9), giving ~12.5m reach. Our robot is a TurtleBot3 Waffle at **0.26 m/s** (real hardware limit). Copying 12.5s literally would give only ~3.25m reach — our randomized circle start-to-goal distance is ~8m, so every episode would time out and learning would be impossible. We instead preserve the paper's *reach-to-time ratio*: 8m / 0.26 m/s ≈ 31s minimum, so `max_time=35.0` (≈9.1m reach) covers the path with modest margin while removing the 60s loitering slack. This is adapting the intent (tight-but-reachable horizon), not blindly copying the constant.

### Orientation penalty (current line ~376)
The current dense reward also subtracts a small orientation term (`weight·|angle_diff|`, weight gated by `d_min`). The paper's Eq 18 does not include it. **Keep it as-is** — it was added to fix a "rotate but don't move" equilibrium and is unrelated to the social-pressure fix; removing it is out of scope (one variable family at a time: the reward magnitudes). Flagged so it's a conscious keep, not an oversight.

## What stays unchanged
- Observation (6-dim spatial, v12), model, ppo.py, train.py — untouched. This is NOT a new architecture generation; v12 checkpoints are load-compatible code-wise (though we retrain fresh as v13 since the reward changed).
- `_compute_social_pressure` (I_sp computation) — unchanged; only its coefficient in `step()` changes.

## Testing (TDD)
1. New `test_reward_paper.py`:
   - On a step that reaches the goal: `r_g == 20.0` contribution.
   - On a collision step: reward includes `-20.0`.
   - Comfort: with a forced known `I_sp`, `r_s == -2.0 * I_sp` (no ÷N). Verify it differs from the old `-0.5·I_sp/N` for N=5 (20× check).
   - Dense approach: moving Δd toward goal yields `2.0·Δd` (minus the unchanged orientation term).
   - `CrowdSimEnv().max_time == 35.0` (new default).
2. Existing smoke tests (`test_env.py`, `test_model.py`, `test_env_velocity.py`, vec tests) still pass — reward change doesn't alter shapes.
3. Env smoke + a short vectorized training smoke run without crash; confirm rewards in the printed diagnostics are in a sane range (not exploding from the 4× stronger comfort).

## Files
- `crowd_sim/crowd_env.py` (5 constants in `__init__` / `step`)
- `test_reward_paper.py` (new)
- (Notebook v13 checkpoint name — folded into the plan)

## Out of scope (future)
- Orientation-term removal, robot max-speed change (0.26→higher), ORCA pedestrians (paper uses ORCA; we use Social-Force) — each a separate variable.
- LTC neuron count increase (capacity) — revisit if the reward fix still leaves hard short.

## Training & success criterion
Colab v13 run, same vectorized command, `--save_path checkpoints/sncp_ppo_v13.pt`. **Success = randomized-hard eval meaningfully > v11/v12's 28%** (independent 100-ep, 4-scenario, seed 100). If the paper's reward is the missing piece, collisions should drop sharply and hard should climb toward the paper's regime. If hard still stalls, the next levers are capacity (LTC size) or the pedestrian model (Social-Force vs ORCA).
