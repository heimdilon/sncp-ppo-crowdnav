# v28: N~U(10,20) density curriculum (high-N gap)

Date: 2026-06-17
Branch: `v28-density-curriculum`

## Goal

Close gap (b) -- the high-N falloff -- by training across the paper's pedestrian-count
range instead of a fixed count. The paper trains on 10--20 humans (§5.3.3); we train at
fixed N=10 and test at 15/20 (out-of-distribution). **Single-variable ablation: v28 = v27
(pre-MLP ON) + N~U(10,20) density curriculum, nothing else changes.**

## Ground truth

- **Paper §5.3.3:** "this study evaluates the navigation ability of each algorithm with 10
  to 20 humans ... 500 test cases in each group ... 11 groups." SNCP-PPO holds 93--95 %
  across the whole range with little falloff.
- **v27 honest baseline (local multi-seed, 5 seeds × 50 ep):** challenging N=5/10/15/20 =
  93.6 / 80.0 / 70.8 / 59.6 %. Gap (a) at the trained density N=10 shrank to ~14 pp after
  pre-MLP; gap (b) is the remaining falloff (80 → 60 from N=10 → 20).
- **Vectorization constraint (`train.py:739`):** "All parallel envs share num_humans so
  observations batch cleanly ... envs are recreated between PPO updates." Per-env different
  N would break observation stacking; the supported path is a homogeneous N per update
  window that varies over training, reusing the existing env-recreation machinery
  (`train.py:839-847`).

## Approach (chosen)

**Random U(10,20), one N per update window, homogeneous across all 16 envs.** Reuses the
tested phase-shift rebuild path (close → rebuild → reinit hidden → discard in-flight). Most
faithful to "trained on 10--20" with minimal new code.

Rejected: linear ramp 10→20 (under-exposes early high-N / late low-N); per-env mixed N with
obs padding/masking (violates the shared-num_humans constraint, large complexity).

## Scope -- what changes

1. **`sncp_ppo/train.py` -- `build_parser`:** add `--num_humans_range MIN MAX`
   (`nargs=2, type=int, default=None`).
2. **`sncp_ppo/train.py` -- `select_vectorized_phase`:** add a `num_humans_range` parameter.
   In the `fixed_scenario` post-bootstrap branch, when the range is set, return
   `rng.randint(min, max)` (inclusive) as N instead of `final_num_humans`. The easy
   bootstrap branch (N=1) is unchanged. When the range is `None`, behaviour is byte-identical
   to v27.
3. **`sncp_ppo/train.py` -- `_train_vectorized`:** read `args.num_humans_range` and pass it
   to both `select_vectorized_phase` calls (the initial phase and the per-window phase). The
   existing `if next_H != H` block rebuilds envs when the sampled N changes.
4. **`sncp_ppo_colab.ipynb`:** training cell adds `--num_humans_range 10 20`; `SAVE_PATH →
   checkpoints/sncp_ppo_v28.pt`; keeps `--pre_mlp` (v28 = v27 + range). Eval/persist/
   diagnostics cells → v28 paths.
5. **`sncp_ppo/run_readiness.py`:** bump v27→v28 markers; add a `--num_humans_range` token
   to TRAINING_TOKENS (and keep the `--pre_mlp` token).
6. **Tests (TDD):** unit tests for the sampling logic (`select_vectorized_phase` returns N in
   [10,20] when the range is set + past bootstrap; returns N=1 during bootstrap; returns the
   fixed N when the range is `None`); CLI parse test; notebook/readiness version-marker tests
   bumped to v28 + assert `--num_humans_range` and `--pre_mlp`.

No model/env/reward changes. `models.py`, `crowd_env.py`, `eval_report.py` untouched.

## Holdout / selection (unchanged -- deliberate)

The best-checkpoint holdout stays `paper_standard` (N=5) + `paper_challenging` (N=10), so the
best-checkpoint criterion is directly comparable to v27. The independent local multi-seed
sweep judges the full N=5/10/15/20 range. A possible refinement -- adding an N=20 holdout so
selection optimises for high density -- is a *second* change and is deferred to keep this a
clean single-variable ablation.

## Evaluation protocol (the real deliverable)

- Train v28 on Colab (main pull). Download `sncp_ppo_v28.pt` + training CSV.
- Local multi-seed sweep, IDENTICAL protocol to v26/v27: 5 seeds (100--500) × 50 ep at
  N=5/10/15/20, `paper_challenging`, robot 1.0, human 1.0, max_time None→50 s, goal_noise 0.
- Compare v28 vs v27 (93.6 / 80.0 / 70.8 / 59.6) pooled success ± 95 % CI per density.
  **Decision rule:** density curriculum helps iff v28's CI clears v27's at the high densities
  (N=15, N=20) AND there is no regression at N=5/10. Confirm timeout stays 0.

## Caveats

1. **Rebuild overhead / in-flight discard:** sampling a new N per window triggers an env
   rebuild + discards in-flight episodes + reinitialises hidden state -- the same behaviour
   as the existing phase-shift path, just more frequent. Bounded; if it proves slow or
   sample-wasteful, a resample-every-K-windows cadence knob can be added (out of scope now).
2. **Single training seed (42),** as with v26/v27.
3. **Capacity unchanged:** v28 uses the v27 (pre-MLP) network as-is; only training-data
   density distribution changes.

## Out of scope / deferred

- `attn_count_scaling` (Eq 13 `n`; next lever for the residual gap a).
- N=20 holdout / selection-criterion change.
- Per-env mixed-density (obs padding/masking).
- Resample cadence knob (only if per-window proves too costly).
