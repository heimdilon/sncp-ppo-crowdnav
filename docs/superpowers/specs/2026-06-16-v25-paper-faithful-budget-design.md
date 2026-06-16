# v25 — Paper-faithful time budget & geometry

**Date:** 2026-06-16
**Status:** approved (brainstorm), pending spec review → plan

## Problem

v24 reproduced the paper *geometry* (scattered humans, 15×15 arena, square-crossing,
robot 1.0 m/s) but **not the paper's training regime**. The Colab density sweep reported
0% at every density, which alarmed us, but diagnosis proved that is a measurement
artifact plus a genuine faithfulness gap:

- **Eval artifact:** training ran at the env-default `max_time=50s` (the training cell
  never passed `--max_time`); the post-eval cell forced `--max_time 15.0`. The model,
  trained with a 201-step budget, learned a slow/cautious policy that cannot finish in
  61 steps → spurious 0%. Re-evaluating the same checkpoint at its training budget gives
  the honest curve: **N=5/10/15/20 → 80/56/32/16%** (50s); N=10@50s = 56.00% matches the
  best checkpoint's holdout `chal=0.56` exactly.
- **Faithfulness gap (the real issue):** under the paper's actual budget (12.5s) the same
  checkpoint scores **0% at every density**. The paper scores 94% in 12.5s.

Four paper deviations, all verified against `s12369-026-01389-9.pdf` (Table 1 + §5.1),
combined to train a robot that never felt time pressure:

| Parameter | Paper | v24 | Source |
|---|---|---|---|
| Max nav time (t_lim) | **12.5s** | 50s | Table 1 |
| Comfort coeff (Eq 20, −k·I_sp) | **2.0** | 6.0 | Eq 20 |
| Safe distance (d_col) | **0.30** | 0.60 | Table 1 |
| Robot crossing distance | ~10m (nav-time 10.4–11.8s) | 12m (challenging) | §5.1 + nav-times |

Lax time (50s) + harsh comfort (6.0) is the perfect recipe for a dawdling policy:
"you have plenty of time and getting close is expensive, so creep with huge margins."
Observed exactly: avg min d_min 1.2m, timeout-dominant when the budget is cut.

## Goal

Make the `paper_standard` / `paper_challenging` scenarios **self-configure** to the
paper's faithful physics so train and eval cannot disagree, then retrain v25 under the
paper budget. This is the real test of whether the architecture can reach ~94% at 12.5s.

## Non-goals (single-variable discipline)

- **Sense-range masking is OUT OF SCOPE.** `sense_range` is currently recorded only
  (`crowd_env.py:79`) and never used in observation/detection; the robot observes all
  humans. Wiring distance masking is a separate behavioral axis that would confound the
  budget/geometry experiment. v25 records the paper values (4m/6m) for documentation but
  does **not** change observation behavior.
- **Density curriculum unchanged.** Keep the N=1→10 ramp; do not extend to N=20. Eval
  still sweeps 5/10/15/20 (extrapolation). Mixing a curriculum change would confound the
  comparison with v22/v24.
- No architecture, PPO, or pedestrian-model changes.

## Design

### Approach: bake paper physics into the env (single source of truth)

Chosen over "fix the notebook/CLI" (fragile — exactly how v24 failed by forgetting
`--max_time`) and "new scenario names" (the existing `paper_*` were never used
successfully; redefine them correctly).

### 1. Env (`crowd_sim/crowd_env.py`)

Add a module-level `PAPER_SCENARIO_CONFIG`:

```python
# robot_y = start (0,-robot_y) → goal (0,+robot_y); 2*robot_y = crossing distance (10m).
# human vpref (1.0) is already set by the existing paper mapping; robot_vpref stays a CLI
# arg (1.0). This config carries only the paper physics that v25 changes/consolidates.
PAPER_SCENARIO_CONFIG = {
    "paper_standard":    {"arena": 10.0, "sense_range": 4.0, "collision_threshold": 0.3,
                           "robot_y": 5.0, "max_time": 12.5, "comfort_coeff": 2.0},
    "paper_challenging": {"arena": 15.0, "sense_range": 6.0, "collision_threshold": 0.3,
                           "robot_y": 5.0, "max_time": 12.5, "comfort_coeff": 2.0},
}
```

- Constructor defaults change to **`max_time=None`, `comfort_coeff=None`** (and keep
  `collision_threshold=None`, `arena_size=None`, `sense_range=None`). Resolution order
  per field: **explicit non-None arg wins**; else if scenario is a paper scenario use the
  paper-config value; else use the legacy default (`max_time=50.0`, `comfort_coeff=6.0`,
  `collision_threshold=robot_radius+human_radius=0.6`). This preserves the 0.26 m/s
  TurtleBot regime exactly while making paper scenarios self-faithful and still allowing
  explicit diagnostic overrides (e.g. the corrected-sweep at 50s).
- `reset()`: paper crossing becomes **robot y = −5 → goal y = +5 (10m)** for both
  scenarios (currently ±4 / ±6 at `crowd_env.py:178/183` area).

### 2. Reward (comfort) wiring

`comfort_coeff` already lives on the env (`crowd_env.py:56`, applied at `:522`
`r_s = -self.comfort_coeff * I_sp`). Because the env now resolves `comfort_coeff=None`
to the paper value, the only change needed upstream is to stop forcing 6.0:

- `train.py` `make_env` default `comfort_coeff` and `max_time` → **None** (pass-through),
  so the env resolves them per scenario.
- `train.py` CLI `--comfort_coeff` / `--max_time` defaults → **None**. Any explicit value
  still overrides. Code that needs a concrete value reads it from the **built env**
  (`env.max_time`, as holdout eval already does at `train.py:108`), not from `args`.

### 3. Eval (`run_post_eval.py` / `post_run_pipeline` / notebook)

- `run_post_eval` `--max_time` default → **None** so paper scenarios eval at 12.5s
  automatically; `evaluate_density` passes it through (explicit value still honored).
- v25 eval cell passes **no** `--max_time`; sets `--baseline_nav_steps 40` (10m beeline
  at 1.0 m/s = 40 steps) and a proportional `--nav_margin_steps` (~10).

### 4. Notebook (`sncp_ppo_colab.ipynb` → v25)

- Title/markers → v25. Training cell identical to v24 **minus** any
  `--max_time / --comfort_coeff / --collision_threshold` (env handles them);
  `--fixed_scenario paper_challenging --num_humans 10 --robot_vpref 1.0
  --holdout_scenarios paper_standard paper_challenging`, 2.5M, `SAVE_PATH=…v25.pt`.
- Eval cell → `--version 25`, scenario `paper_challenging`, densities 5/10/15/20,
  baseline `eval_v22/density_sweep.json`, no max_time override.

### 5. Tests (TDD)

- `test_paper_scenarios.py`: paper envs report `max_time=12.5`, `collision_threshold=0.3`,
  `comfort_coeff=2.0`, `sense_range` 4/6, crossing 10m (robot_py=−5, robot_gy=+5).
- Non-paper env unchanged: `max_time=50.0`, `collision_threshold=0.6`, `comfort_coeff=6.0`.
- Explicit overrides win (e.g. `CrowdSimEnv(scenario="paper_challenging", max_time=50.0)`
  → `env.max_time == 50.0`).
- Parser/`make_env`: paper run with no `--max_time/--comfort_coeff` yields a 12.5s / 2.0
  env; holdout-config parity test still passes.

## Success criteria

- All tests green; non-paper regime bit-identical in behavior.
- After Colab: v25 evaluated **at 12.5s** shows non-zero success that meaningfully beats
  v24's 0% at the paper budget. Stretch: approach the paper's challenging ~94%.
- Honest framing retained: v22 stays the best antipodal-1.0 result; v18 the 0.26 robot
  baseline; v25 is the first genuinely paper-budget-faithful attempt.

## Risks

- **Tight budget may stall learning** (the v13 max_time=35 collapse). Mitigated: 10m/12.5s
  leaves 2.5s (10-step) margin and the density curriculum starts at N=1 where the margin is
  ample, so the goal signal is reachable early. If training still collapses, the fallback
  is a time-budget anneal (deferred per brainstorm).
- **None-default ripple in train.py**: any code reading `args.max_time` as a float must
  tolerate None by reading `env.max_time` post-construction instead. Covered by the plan.
