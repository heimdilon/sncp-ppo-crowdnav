# v15 — Genuine Social Navigation (non-reactive crowd, full density)

**Date:** 2026-06-05
**Status:** Design approved, pending implementation plan

## Problem

v14 (NCP architecture + reactive "cooperative" crowd) reached ~100% success at
N≤5 and 92/84/58% at N=10/15/20. BUT trajectory + density analysis proved the
robot **beelines**: it drives a nearly straight line to the goal (constant
~121.5 steps regardless of N), and collisions are avoided only by luck or by
pedestrians yielding. It does **not** actually avoid people.

**Root cause:** in a reactive crowd, avoidance is unnecessary (pedestrians
yield), and the comfort penalty (−2·I_sp, with I_sp≈0.02 → ~−0.04/step) is
negligible next to goal +20 and approach +0.13/step. The robot learned the
rational policy for that environment: beeline.

## Goal

Train a policy that **actively avoids** pedestrians — both collision-avoidance
and social-distance keeping — in a **non-reactive** crowd (pedestrians ignore
the robot, the "invisible robot" setting that matches the source paper's
CrowdNav/ORCA regime). Validate with **genuine avoidance behavior** (detouring
paths, nav-time growing with density, maintained personal distance), not just
success rate.

## Decisions (from brainstorming)

- **Mechanism: BOTH** — avoidance must be NECESSARY (non-reactive crowd) AND
  REWARDED (strong social-distance penalty).
- **Speed: PARITY at hardware speed** — keep the real TurtleBot3 robot at
  0.26 m/s; cap pedestrian speed at ≤0.26 m/s so a slow robot can feasibly avoid
  non-reactive pedestrians. (The paper used speed parity at 1.0/1.0; we use
  parity at our hardware scale.)
- **Scope: FULL social-nav, reached in two de-risked steps** — v15 ramps density
  to N=10; v16 extends to N=15–20 once v15 validates that real avoidance emerges
  (no freezing, nav-time rises). Going straight to N=20 risks a confounded stall.
- **Unchanged:** NCP architecture (v14), spatio-temporal observation,
  PPO/vectorized infrastructure, goal +20, collision −20.

## Design

### 1. Environment (`crowd_sim/crowd_env.py`)
- `human_dodge_robot` default: `True` → **`False`** (non-reactive). Pedestrians
  run their SFM among themselves and ignore the robot. (Reverts v14; the
  cooperative option remains available via the flag.)
- Pedestrian speed parity: per-phase `human_vpref` capped at ≤0.26 m/s.
- `max_time`: keep 50s initially; raise to 60–75s only if the first run shows
  timeout-dominant failures from detours at high density.

### 2. Reward (`crowd_sim/crowd_env.py` `step`)

| Term | v14 | v15 | Rationale |
|---|---|---|---|
| Goal arrival | +20 | +20 | keep (paper) |
| Collision | −20 | −20 | keep; now an ACTIVE signal (non-reactive → real collisions) |
| Approach shaping | 2·Δd | **1·Δd** | halve so detours around people are not over-penalized vs the straight line |
| Social distance | −2·I_sp | **−6·I_sp** (conservative start; tune UP to −8/−10 if it still grazes, DOWN if it freezes) | make personal-space intrusion bite → keep distance. Started conservative: non-reactivity already forces collision-avoidance (−20 now active), and a frozen robot is the costlier failure to recover from than a robot that merely grazes too close (which the I_sp diagnostic reveals → raise the coefficient). |
| Orientation | small (kept) | small (kept) | unchanged |
| Anti-freeze | none | small per-step time cost (≈−0.01/step) **only if** run 1 shows freezing | prevent the robot waiting indefinitely |

These are starting values; the social/approach balance will be tuned
empirically (expect 1–2 iterations).

### 3. Curriculum (`sncp_ppo/train.py`, vectorized path)
- **v15 (this run): density phases 1 → 3 → 5 → 10** (final `num_humans = 10`).
  Speeds (parity, m/s): **0.13 → 0.18 → 0.22 → 0.26**.
- **v16 (follow-up, after v15 validates real avoidance): extend to 15 → 20.**
- Step budget: **~2.5M** for v15; phase boundaries spread across the budget.
- Holdout: add a high-density scenario (N=10) alongside easy/hard, to monitor the
  real target during training.
- Keep anti-forgetting replay if present.

### 4. Success criteria
**Quantitative:**
- v15 (target up to N=10): N≤5 success >90% / collision <10%; N=10 success >70%.
- v16 (after extension): N=15–20 success >60–70% (stretch: paper's ~94% at parity).

**Behavioral (the real proof, vs v14 beeline):**
- Nav-time **increases** with density (not the constant ~121.5) → detours.
- Min robot–pedestrian distance > social threshold (~0.5–0.6 m) for the large
  majority of steps.
- Per-step I_sp stays low **despite** the non-reactive crowd (robot actively
  keeps distance, rather than the crowd parting).
- Trajectory plots show curved/detouring paths around pedestrians.

### 5. Validation plan
- Density sweep (N=5/10/15/20): success + collision + I_sp.
- Trajectory plots + GIFs at several N.
- **Nav-time-vs-density curve** — must rise (the decisive contrast with v14's
  flat 121.5).
- Side-by-side vs v14 (beeline baseline) to demonstrate the behavior changed.

### 6. Risks & mitigations
- **Freezing robot** (strong comfort + non-reactive) → curriculum ramp, balanced
  comfort coefficient, anti-freeze term, keep the goal attractive. Tune.
- **Non-convergence at N=20** (slow robot, dense non-reactive crowd, à la v13
  stall) → parity speed makes it feasible; curriculum, generous max_time, longer
  training.
- **Reward-tuning cost** → expect 1–2 A100 iterations; start conservative and
  adjust from diagnostics.

### 7. Tests (TDD)
- `crowd_env`: default `human_dodge_robot is False`; pedestrian speed cap ≤0.26;
  reward terms (approach 1·Δd, comfort −6·I_sp, collision −20, goal +20).
- Repurpose `test_pedestrian_reactive.py`: its current tests assert the default
  is reactive (True) — now inverted. Change to assert the **non-reactive
  default** and that reactivity remains available as an explicit option.
- `train.py` curriculum: phases reach N=20 with parity speeds; holdout includes
  high density.
- Full regression green.

### 8. Versioning
- **v15** (fresh retrain): non-reactive + reward redesign + parity speed,
  curriculum to N=10. **v16**: extend curriculum to N=15–20 (after v15 validates
  real avoidance). Architecture unchanged from v14 (checkpoints load-compatible);
  retrain fresh because env + reward changed.

## Out of scope
- Robot speed changes (staying at 0.26 m/s hardware).
- Switching the pedestrian model SFM → ORCA (non-reactive SFM approximates the
  paper's regime; revisit only if needed).
- Model architecture changes (NCP stays).
