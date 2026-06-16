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

- Constructor defaults change to **`max_time=None`, `comfort_coeff=None`** (keep
  `collision_threshold=None`, `arena_size=None`, `sense_range=None`) and a new
  **`paper_regime=False`** flag.
- **Two-tier resolution** (this is the key correction over the naive
  "resolve-by-construction-scenario" idea — see Mechanism note below):
  - **Regime params** (`max_time`, `comfort_coeff`, `collision_threshold`): resolved
    against `is_paper = paper_regime or (scenario in PAPER_SCENARIO_CONFIG)`. Explicit
    non-None arg wins; else paper value (12.5 / 2.0 / 0.3) if `is_paper`; else legacy
    (50.0 / 6.0 / 0.6). The `paper_regime` flag is what carries the budget into the
    **easy-bootstrap phase** and the **holdout `eval_env`**, neither of which is
    constructed with a paper *scenario*.
  - **Geometry** (`arena`, `sense_range`, robot crossing): keyed only on
    `scenario in PAPER_SCENARIO_CONFIG`, because geometry is only read when the active
    scenario actually is paper (the easy bootstrap uses the circle geometry untouched).
- `reset()`: paper crossing becomes **robot y = −5 → goal y = +5 (10m)** for both paper
  scenarios (currently ±4 / ±6 at `crowd_env.py:179/184`); `_paper_robot_y` reads
  `PAPER_SCENARIO_CONFIG[scenario]["robot_y"]`; `sense_range` set from the config.
- Non-paper, non-`paper_regime` construction is byte-for-byte the legacy 0.26 TurtleBot
  regime. Explicit args always win (e.g. the corrected-sweep at `max_time=50`).

> **Mechanism note (why `paper_regime`, not scenario-only).** The training env is
> constructed with `scenario='easy'` (`train.py:274`) and phases are *rebuilt* via
> `build_envs(...)` (`train.py:747`) — so during the easy bootstrap the scenario is not
> paper. The holdout path is worse: `evaluate_holdout` **mutates** `env.scenario`
> post-construction (`train.py:97`) without rebuilding, so `env.max_time` keeps its
> `__init__` value. Resolving the regime purely from the construction scenario would
> therefore leave bootstrap and holdout at 50s/6.0. `paper_regime` (derived once from
> `--fixed_scenario`) forces the budget across *all* env constructions.

### 2. Train wiring (`sncp_ppo/train.py`)

- `make_env` and `build_envs` gain a `paper_regime` parameter, threaded to the
  `CrowdSimEnv(...)` constructor at every build site: the main env (`:274`), each phase
  env (`build_envs`, `:748`), and the holdout `eval_env` (`:780`).
- `make_env`/CLI `--comfort_coeff` and `--max_time` defaults → **None** (pass-through),
  so a paper run leaves them unset and the env resolves them. Any explicit value still
  overrides. Code needing a concrete value reads it from the **built env**
  (`env.max_time`, as holdout already does at `train.py:108`), not from `args`.
- `train()` and `_train_vectorized()` derive
  `paper_regime = getattr(args, 'fixed_scenario', None) in PAPER_SCENARIO_CONFIG`
  and pass it to every build site above. `comfort_coeff` already lives on the env
  (`crowd_env.py:56`, applied at `:522` `r_s = -self.comfort_coeff * I_sp`), so no
  reward-code change is needed beyond the resolved value.

### 3. Eval (`run_post_eval.py` / `eval_report` / notebook)

- The post-training eval builds envs **with the paper scenario** (`evaluate_density`,
  `eval_report.py:451`), so scenario-keyed resolution suffices there — no `paper_regime`
  needed. Change `evaluate_density` `max_time` default and `run_post_eval` /
  `run_post_eval.py` `--max_time` default → **None** so a paper-scenario eval lands at
  12.5s automatically; explicit values still honored.
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

- `test_paper_scenarios.py`:
  - Paper-scenario env auto-resolves: `max_time=12.5`, `collision_threshold=0.3`,
    `comfort_coeff=2.0`, `sense_range` 4/6, crossing 10m (`robot_py=−5`, `robot_gy=+5`).
  - **`paper_regime=True` with a non-paper scenario** (`scenario='easy'`) →
    `max_time=12.5`, `comfort_coeff=2.0`, `collision_threshold=0.3` (regime forced) but
    **circle geometry preserved** (radii ≈ 4.0, no robot ±5 override).
  - Non-paper, non-regime env unchanged: `max_time=50.0`, `collision_threshold=0.6`,
    `comfort_coeff=6.0`.
  - Explicit override wins: `CrowdSimEnv(scenario="paper_challenging", max_time=50.0)`
    → `env.max_time == 50.0`.
  - Update the two existing geometry tests to the 10m crossing (`test_paper_standard_layout`
    expects (0,−5)/(0,5); `test_paper_challenging_scales_arena` expects (0,−5)/(0,5)).
  - `make_env(..., paper_regime=True)` yields a 12.5s / 2.0 / 0.3 env; holdout-config
    parity test still passes.
- `test_post_run_pipeline.py`: notebook guard — assert the v25 training cell sets
  `--fixed_scenario paper_challenging` and `SAVE_PATH=checkpoints/sncp_ppo_v25.pt`, and the
  v25 eval cell uses `--version 25` + `--baseline_nav_steps 40` and passes **no**
  `--max_time` (regression against the v24 forgot-the-budget failure).
- `test_eval_report.py` (or equivalent): `evaluate_density(scenario='paper_challenging')`
  with `max_time=None` builds a 12.5s env; non-paper default stays 50s.

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
