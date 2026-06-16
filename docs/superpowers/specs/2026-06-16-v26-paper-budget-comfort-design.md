# v26 — Paper budget + crossing + comfort-gradient fixes

**Date:** 2026-06-16
**Status:** approved (Gemini DeepThink external review + my ground-truth audit; user decisions)

## Problem

v25 (challenging at 12.5s, 10m crossing) capped at ~62% (N=10), collision-dominated.
An external review (Gemini DeepThink) plus my verification against the paper PDF and our
code found **three mis-specifications** — two of them my own errors from v24→v25:

1. **Challenging time budget was wrong.** Table 1's `t_lim = 12.5s` applies to the
   STANDARD scenario only. Table 3 (challenging) reports SNCP-PPO Nav-Time = **15.92s**
   and Pred-AttnGraph = **28.55s** (with only 5% timeout) — averages that are
   mathematically impossible under a 12.5s hard cap. The challenging budget is ~30s+.
   My v25 "tighten to 12.5s" starved the robot of time, forcing it to rush the crowd
   (accept −20 collisions for the only chance at +20), which matches the
   collision-dominated symptom.
2. **Crossing distance was wrong.** Paper §5.3.2 states verbatim: *"starting position of
   (0, −4) to a target destination of (0, 4)"* = an **8 m** crossing (robot ±4). My v25
   used ±5 (10 m). (I had inferred 10 m from the 10.37s nav-time, which is actually the
   STANDARD scenario, Table 2.)
3. **Comfort `I_sp` is not the paper's normalized form.** Eq 6-7 + text: `I_sp` is a
   *weighted average* of per-human pressure `I_2 = ω·I_1` with weights `1/d_hr`,
   *normalized* to [0,1]: `I_sp = Σ(I_2_i / d_hr_i) / Σ(1 / d_hr_i)`. Our
   `_compute_social_pressure` computes only the **numerator** (`I_sp += inv_d_hr * I_2`,
   with `inv_d_hr` capped at 10) and then `np.clip(I_sp, 0, 1)`. In dense crowds the
   un-normalized sum saturates the clip → the comfort penalty pegs at a flat `−2/step` →
   **zero gradient** to back away (0.8 m vs 0.3 m give the same penalty). The robot never
   learns soft avoidance. (Our `ω` (Eq 6) and `I_2 = ω·I_1` (Eq 5) already match the
   paper; only the Eq 7 normalization is missing.)

## Decisions (user)
- `d_col = 0.3` stays (paper Table 1; parity. Gemini suggested 0.6 for realism, but 0.3 is
  the more lenient collision criterion the paper used and it *helps* match 94% — realism
  is a separate goal the user did not pick).
- Challenging budget = **50 s** (user: "whichever increases success"). Success rate is
  monotonic non-decreasing in budget; the paper's slowest method averages 28.55s nav, so
  50s removes time-starvation at every density N=10-20. We will **report Nav-Time** so the
  loose budget is transparent (not exploited silently). 35s is the more paper-faithful
  alternative; 50s is the success-maximal one and matches v24's empirically-working value.

## Scope (v26 = the three verified fixes only)
Single-purpose: correct the scenario/reward mis-specification. Speculative improvements are
deferred to post-v26 ablations (below) to keep attribution clean.

### 1. Env config (`crowd_sim/crowd_env.py`, `PAPER_SCENARIO_CONFIG`)
- `paper_standard`: `robot_y` 5.0 → **4.0** (8 m). `max_time` stays **12.5**.
- `paper_challenging`: `robot_y` 5.0 → **4.0** (8 m). `max_time` 12.5 → **50.0**.
- `d_col` (collision_threshold) stays 0.3; `comfort_coeff` stays 2.0; sense 4/6 unchanged.

### 2. Comfort gradient (`crowd_sim/crowd_env.py`, `_compute_social_pressure`)
Replace the un-normalized sum + clip with the paper's Eq 7 weighted average:
```python
num = 0.0; den = 0.0
for i in range(N):
    ...                      # unchanged: I_1, omega (Eq 6), I_2 = omega * I_1 (Eq 5)
    w_i = 1.0 / (d_hr + 1e-5)
    num += w_i * I_2
    den += w_i
return float(num / (den + 1e-9))   # Eq 7 weighted average, naturally in [0,1]
```
Remove the `min(1/d_hr, 10.0)` cap and the `np.clip(I_sp, 0, 1)`. This affects ALL
scenarios (incl. the legacy 0.26 regime) — byte-parity with old comfort is intentionally
broken; the new form is paper-faithful and gradient-preserving.

### 3. Eval + notebook + readiness (v25 → v26)
- Eval beeline: 8 m at 1.0 m/s = 32 steps. Revert `--baseline_nav_steps` 40 → **32**
  (v25 had set it to 40 for the wrong 10 m).
- Notebook: title/markers v25 → v26; training cell `SAVE_PATH=…v26.pt` (still passes NO
  CLI budget flags — env derives 50s/2.0/0.3 from `--fixed_scenario paper_challenging`);
  eval cell `--version 26`, `--baseline_nav_steps 32`, no `--max_time`; persist + cell-24
  diagnostics → eval_v26.
- `sncp_ppo/run_readiness.py` + tests: markers v25 → v26, beeline 32.

### 4. Tests (TDD)
- Paper env: `paper_standard.max_time == 12.5`, `paper_challenging.max_time == 50.0`;
  both crossings = 8 m (`robot_py == -4`, `robot_gy == 4`); `d_col == 0.3`,
  `comfort_coeff == 2.0` unchanged.
- `_compute_social_pressure`: (a) result in [0,1] with NO clip; (b) **gradient preserved**
  — a robot closer to a fixed human yields a strictly higher `I_sp` than one farther
  (the property the clip destroyed); (c) a known small-N hand case matches the Eq 7
  weighted average within tolerance.
- Notebook v26 guard + readiness v26 + non-paper regime unchanged (hard/circle still
  max_time 50 / d_col 0.6 / comfort 6 — note: comfort *value* unchanged at 6.0, but the
  `I_sp` it multiplies now uses the normalized form, so update any legacy comfort-value
  assertions to the new I_sp).

## Success criteria
- All tests green; the `I_sp` gradient test passes (the core comfort fix).
- After Colab v26: challenging N=10 at the corrected budget should jump well above v25's
  62%; timeouts → ~0 (time-starvation removed); failures become collision-limited.
  Compare the density sweep (N=5/10/15/20, eval at 50s challenging) to the paper's ~94%.

## Deferred ablations (post-v26, single-variable)
- `pre_mlp=True` (Eq 11: MLP→NCP) — architecture fidelity.
- Train `N ~ U(10,20)` instead of N=10 curriculum + extrapolation (paper Fig 6b; our
  N=15/20 are currently OOD).
- Random pedestrian goals instead of antipodal `−start ± 1` (reduce the central funnel).
- Action range (TurtleBot 2 Kobuki `w_max ≈ 3.14`, reverse allowed) vs our `w_max=1.8`,
  `v ≥ 0`.

## Not changing (with reasons)
- `d_col = 0.3` (parity, helps; Gemini's 0.6 is a realism goal not chosen).
- `attn_count_scaling = False` (Gemini: Eq 13's "n" is a misread transpose `QK^T`; our
  probe already showed no benefit).
- Pedestrians stop at goal / no respawn (Gemini: paper Fig 7 shows no respawn → our static
  implementation is faithful).
