# v32 — Curriculum reach N→25 + budget 4M (high-N generalization, experiment #3)

Date: 2026-06-20
Branch: `feat/v32-curriculum-budget`

## Goal

Improve high-N (N=15/20) generalization by (a) extending the density curriculum from
`N∈[10,20]` to `N∈[10,25]` so N=20 sits mid-range instead of at the edge, and (b) lengthening
training from 2.5M to 4M steps. Built on the v30 champion (mean+max pooling). This is
experiment #3 of the pure-performance roadmap (#1 mean+max ✅ directional → #2 node capacity ❌
→ **#3 budget/curriculum**). **Config-only: no model code** — both knobs are already CLI-wired,
so there is no AutoNCP topology-reroll confound (the suspected cause of #2's regression).

**Two-variable caveat (user-chosen):** v32 changes BOTH the curriculum reach and the step budget,
so a positive result will not attribute cleanly to one knob. The user accepted this to maximize
effect; it is recorded honestly and carried into the verdict.

## Diagnosis (why these knobs)

v30 honest 5-seed (the champion): success 97.2/89.6/85.6/79.2, collision 2.8/10.4/14.4/20.8
(N=5/10/15/20); timeout 0 everywhere. The remaining gap is high-N collision, and the failure
grows toward N=20 — the top of the trained range `N∈[10,20]` (v28's density curriculum). A model
trained only up to N=20 is extrapolating exactly at the test point that fails most. Extending the
curriculum to N=25 turns N=20 into a mid-range (interpolation) density; the longer 4M budget gives
the harder curriculum more gradient steps to converge. v31's node-256 capacity is dropped (it
regressed in #2) — v32 returns to v30's node 128/48.

## Verified ground truth (current code + notebook)

- `--num_humans_range MIN MAX` and `--total_steps N` are already CLI args (v28 added the range;
  `select_vectorized_phase` samples `N∈[min,max]` per window via `rng.randint`). **No train.py /
  models.py change is needed for v32** — only config values.
- Notebook (v31) training cell: `TOTAL_STEPS = 2_500_000` (line 257), `'--num_humans_range', '10',
  '20'` (line 271), `'--node_units', '256'` / `'--node_output', '96'` (lines 282-283).
- `run_readiness.py` TRAINING_TOKENS (v31) include `"TOTAL_STEPS = 2_500_000"`,
  `"'--num_humans_range'"`, `"'--meanmax_pool'"`, `"'--node_units', '256'"`, `"'--node_output', '96'"`.
- The holdout (fixed N=10) and the easy bootstrap (N=10, 200k steps) are independent of the range
  max, so they are unaffected; only the curriculum windows sample up to N=25.
- The env is parametric in `num_humans`, so N=25 scenes build fine (denser, but feasible at robot 1.0).

## Design (config-only)

### v32 training config = v30 + two value changes, minus v31's node flags

- `--num_humans_range 10 20` → `--num_humans_range 10 25`
- `--total_steps 2_500_000` → `--total_steps 4_000_000`
- DROP `--node_units 256 --node_output 96` (return to default node 128/48 = v30)
- Keep: `--pre_mlp --meanmax_pool --fixed_scenario paper_challenging --num_humans 10
  --bootstrap_easy_steps 200000 --robot_vpref 1.0 --lr 1e-4 --holdout_scenarios paper_standard
  paper_challenging --holdout_episodes 50`. Checkpoint `checkpoints/sncp_ppo_v32.pt`.

No new constructor args, no auto-detect change: v32's checkpoint has the v30 architecture
(node 128/48 + pre_mlp + meanmax), which `build_policy_for_checkpoint` already loads.

## Components / files (no model code)

- `sncp_ppo_colab.ipynb` — training cell: TOTAL_STEPS 4M, range `10 25`, remove node flags; v31→v32 paths.
- `sncp_ppo/run_readiness.py` — v31→v32 markers; TRAINING_TOKENS: TOTAL_STEPS 4M, drop node tokens
  (keep meanmax/pre_mlp/num_humans_range).
- `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` — version-marker bump v31→v32.

No changes to `models.py`, `train.py`, `crowd_env.py`, `eval_report.py`.

## Testing (version markers only — no model code to TDD)

Update `test_notebook_is_v31_node_capacity` → `test_notebook_is_v32_curriculum_budget`:
- assert `checkpoints/sncp_ppo_v32.pt`, `"'--num_humans_range', '10', '25'"`,
  `"TOTAL_STEPS = 4_000_000"`, `"'--meanmax_pool'"`, `"'--pre_mlp'"`, `"'--node_units'" not in train`,
  `"'--version', '32'"`, beeline 32, no `--max_time`.
Update the 3 readiness tests v31→v32 (`test_v32_run_readiness_passes_current_repo`,
`..._flags_stale_notebook` with notes "v32 training"/"v32 evaluation", `..._eval_v32_artifact_bundle`).

Full suite stays green (`--basetemp=./.pytmp`, `C:/ProgramData/miniconda3/python.exe`). Readiness
preflight returns `pass` with "v32 ... ready". A tiny CLI training smoke
(`--pre_mlp --meanmax_pool --num_humans_range 10 25 --total_steps 4096 ...`) exits 0 (verifies the
N=25 curriculum window builds and trains).

## Evaluation (run-time, post-Colab)

Same honest protocol (base-conda, 5 seeds 100–500 × 50 ep at N=5/10/15/20, paper_challenging,
robot 1.0, human 1.0, max_time None, goal_noise 0) on `sncp_ppo_v32.pt`. Compare to the **v30
baseline** (success 97.2/89.6/85.6/79.2; collision 2.8/10.4/14.4/20.8) with Wilson CIs +
two-proportion z (Bonferroni α=0.0125), both success and collision (reuse `_analyze_v30/31` →
`_analyze_v32`). Optional informational N=25 probe (no v30 baseline at 25).

**Decision rule:** curriculum+budget helps iff high-N success rises and/or collision drops (esp
N=15/20) with no regression at N=5/10 and timeout 0, vs v30. A flat/negative result is reported
honestly. Write the verdict to memory (`sncp-paper-vs-impl.md`).

## Out of scope / deferred

- Isolating curriculum vs budget (the two-variable split) — would need two separate runs; the user
  chose to combine. If v32 helps and attribution matters later, run the single-knob variants then.
- Adding N=25 to the canonical eval/decision (kept at 5/10/15/20 for direct v30 comparison).
- Any model-architecture change (that was #1/#2).
- Multi-seed training (user declined).

## Irreversibility note

Config-only; v32's checkpoint uses the existing v30 architecture (loads with no code change). Work
proceeds on `feat/v32-curriculum-budget`; merge to `main` + push only at the finishing step after the
user confirms (Colab pulls `main` to train v32).

## Honest caveats (carried into the verdict)

- Two variables (curriculum reach + budget) → a win is not cleanly attributable.
- v28/v30 were ~peak=final (not obviously undertrained), so the budget half may add little; any gain
  likely comes from the curriculum reach.
- Single training seed (as for v27–v31); ~±5-7pp swings can be partly seed noise.
- Training to N=25 pushes the densest scenes toward the feasibility limit at robot 1.0 (env handles
  it, but very dense).
