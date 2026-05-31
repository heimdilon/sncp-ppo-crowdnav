# Curriculum + Holdout in the Vectorized Path (v11) — Design Spec

**Date:** 2026-05-31
**Status:** Approved (brainstorming) → ready for implementation plan
**Builds on:** v10 vectorized rollout (PR #15, merged `b39ea75`)
**Closes the conscious gap:** the vectorized path (`--num_envs>1`) currently trains on a fixed `circle` scenario with no curriculum and no holdout/best-checkpoint. The single-env path has both.

## Problem

`_train_vectorized` in `sncp_ppo/train.py` runs a fixed-horizon SyncVectorEnv rollout but:
- trains only on a hardcoded `'circle'` scenario (no 1→5 pedestrian curriculum),
- never evaluates holdout scenarios, so it saves only a `_final` checkpoint (no generalist best-checkpoint selection).

Before a serious Pro+ run we want the vectorized path to match the single-env path's curriculum + holdout semantics.

## Decisions (locked during brainstorming)

1. **Recreate envs at phase boundaries.** The policy batches all N envs through one forward pass, so they must share `num_humans` (spatial tensor is `(N, H, 4)`; mixed H cannot batch). At each curriculum phase change, close and rebuild the `SyncVectorEnv` with the new `num_humans` and re-init the LTC hidden. Boundaries are rare (4 over the whole run), so cost is negligible. Recreation happens **between** PPO updates, never mid-rollout (a half-collected buffer at a boundary is discarded by construction).
2. **Reuse `evaluate_holdout`** (the existing single-env, deterministic function) periodically on a throwaway single-env `CrowdSimEnv`. No new eval code path.
3. **Total env-step budget** drives both curriculum phases and run length via `--total_steps` (default 2_000_000). Phase boundaries are step fractions (same 10/25/50/75% schedule as single-env). This is N/T-independent and comparable across configs.

## Curriculum schedule (identical to single-env)

Same 5 phases as the single-env `curriculum` list in `train.py`:

| Phase | scenario | num_humans | vpref | step-fraction boundary |
|---|---|---|---|---|
| 1 | easy | 1 | 0.15 | ≤ 10% |
| 2 | easy_plus | 2 | 0.20 | ≤ 25% |
| 3 | medium | 3 | 0.30 | ≤ 50% |
| 4 | hard | 4 | 0.40 | ≤ 75% |
| 5 | circle | 5 | 0.50 | ≤ 100% |

(`num_humans` for phase 5 = `args.num_humans`, default 5.)

## Components — all in `sncp_ppo/train.py` (+ one test file)

### A. `step_to_phase(total_steps_seen, total_steps, num_humans)` — pure module-level helper
Returns `(scenario, n_humans, vpref)` for the current step count. Pure arithmetic (no env, no GPU) so it is fully unit-testable across all 5 phases and 4 boundaries. Mirrors the single-env phase thresholds (10/25/50/75% of `total_steps`).

### B. Extended `_train_vectorized` loop
```
phase = step_to_phase(0, total_steps, num_humans)
envs = SyncVectorEnv([make_env(phase.n, phase.scenario, seed+i) for i in range(N)])
obs, _ = envs.reset(seed=...); apply phase.vpref; h = init_hidden(N, phase.n)
while total_steps_seen < total_steps:
    new_phase = step_to_phase(total_steps_seen, total_steps, num_humans)
    if new_phase.n != phase.n or new_phase.scenario != phase.scenario:
        envs.close(); rebuild envs with new_phase; reset; re-apply vpref;
        h = init_hidden(N, new_phase.n); phase = new_phase
    # collect one fixed-horizon rollout (T steps), update_vectorized (unchanged)
    ... rollout + buf.finish + agent.update_vectorized ...
    total_steps_seen += N * T
    if update_idx % eval_freq_updates == 0:
        run holdout + best-checkpoint (component C)
```
- vpref re-applied after every env reset (mirrors single-env line 293) so the phase's intended speed ramp is used.
- Hidden re-init on recreation prevents stale-memory leak across the H change.

### C. Periodic holdout + best-checkpoint (reuses single-env logic)
Every `--eval_freq_updates` updates:
- Build a throwaway single-env `CrowdSimEnv(num_humans=args.num_humans)`.
- For each scenario in `args.holdout_scenarios` (default easy, hard): call the existing `evaluate_holdout(env, policy, agent, device, args.holdout_episodes, scenario, base_seed)`.
- Compute generalist metric `min(success across holdout scenarios)`.
- Apply the SAME best-checkpoint logic as single-env: warmup (`best_warmup_evals`), threshold (`best_min_success_threshold`), tie-break (min_success, then avg_reward, then lower collision). Save to `args.save_path` when improved.
- Write the same CSV columns as the single-env path (episode→use total_steps_seen; scenario/num_humans/vpref from current phase; per-scenario holdout tail).

## What stays untouched
- Single-env path (`--num_envs 1`): zero change — byte-identical.
- `compute_gae_vectorized`, `VectorizedRolloutBuffer`, `update_vectorized`, `evaluate_holdout`: reused, not modified.
- `sncp_ppo/models.py`, `crowd_sim/crowd_env.py`, `sncp_ppo/ppo.py`: untouched.

## New CLI args
- `--total_steps` (int, default 2_000_000): env-step budget; drives curriculum phases + run length in vectorized mode.
- `--eval_freq_updates` (int, default 20): holdout cadence in PPO updates.
- In vectorized mode `--episodes` is ignored (print a one-line note); `--total_steps` governs.

## Testing (TDD)
1. **Phase mapping (pure):** `step_to_phase` returns correct (scenario, N, vpref) at 0%, just-below/at each of 10/25/50/75%, and 100%. No env/GPU.
2. **Env recreation:** simulate a phase change; assert rebuilt `envs.num_envs == N`, per-env `num_humans` updated, hidden re-init shapes correct.
3. **Holdout integration (smoke-level):** vectorized loop calls `evaluate_holdout`, produces a CSV row, and saves a checkpoint when `min(success)` improves.
4. **End-to-end smoke:** `--num_envs 4 --total_steps 8000 --horizon 64 --eval_freq_updates 5` runs on local GPU, prints ≥1 phase shift + holdout line, exits 0, saves a checkpoint.
5. **Regression:** the 29 existing unit tests still pass; single-env smoke (`--num_envs 1`) output unchanged.

## Files
- `sncp_ppo/train.py` — add module-level `step_to_phase`; extend `_train_vectorized` (curriculum + recreation + holdout/best-ckpt); add the two CLI args.
- `test_vec_curriculum.py` (new) — phase-mapping unit tests + recreation + holdout-integration tests.

## Out of scope (future)
- Curriculum replay in vectorized mode (single-env `--curriculum_replay_ratio` stays single-env only).
- Vectorized (parallel) holdout eval — we deliberately reuse the slower single-env `evaluate_holdout` for correctness/comparability.
- AsyncVectorEnv, reward shaping, pedestrian-goal obs, LTC→GRU — separate efforts.

## Default Pro+ run (after this lands)
`--num_envs 16 --horizon 128 --total_steps 2000000 --eval_freq_updates 20 --holdout_episodes 50 --holdout_scenarios easy hard --lr 5e-5 --target_kl 0.01 --save_path checkpoints/sncp_ppo_v11.pt`
