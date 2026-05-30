# Vectorized Environment Rollout — Design Spec (v10)

**Date:** 2026-05-30
**Status:** Approved (brainstorming) → ready for implementation plan
**Root cause addressed:** #1 data starvation (see `docs/critique-synthesis-2026-05-30.md`)

## Problem

The training loop collects ~500 transitions per PPO update (`update_freq=5` ×
single serial env, ~100 steps/episode). Standard PPO uses 2048+. This data
starvation — flagged independently by two critique agents — is the primary
reason hard-scenario success is stuck at 30-38% regardless of hyperparameters.
No lr/KL/episode change can fix it; the gradient signal per update is too noisy.

## Goal

Add a vectorized rollout path that collects N parallel environments × T steps
per PPO update (default 16×128 = 2048 transitions), while preserving the
existing single-env path for reproducibility and A/B comparison.

## Decisions (locked during brainstorming)

1. **Fixed-horizon rollout** (CleanRL standard): N envs × T steps; episodes may
   be cut mid-rollout; on `done`, that env auto-resets and its hidden state is
   zeroed; bootstrap V(s_T) at horizon end.
2. **Fixed-window BPTT** (seq_len=16, stored hidden): the vectorized form of the
   current `_extract_subsequences` — split each env's T-step rollout into
   seq_len windows that respect env + episode boundaries.
3. **New parallel path, keep the old**: `--num_envs` flag; `num_envs=1` calls
   the existing single-env `update()` (zero behavior change). `num_envs>1` uses
   the new vectorized path.
4. **Defaults:** `num_envs=16`, `horizon=128` (→2048 transitions). Smoke tests
   use `num_envs=4` for speed.

## Components

### A. `sncp_ppo/vec_buffer.py` (new) — `VectorizedRolloutBuffer`
Fixed `(num_envs, horizon)` tensor storage (replaces list-based, episode-bounded
`PPOMemory` for the vectorized path):
- Per step stores, for all N envs: obs (robot_node/spatial_edges/temporal_edges),
  action, log_prob, reward, value, **done** (terminated OR truncated), and the
  **hidden state** fed into that step (temporal/spatial/node).
- `done[env, t]` marks episode boundaries → drives both GAE cutoff and BPTT
  hidden-reset.
- Stores per-env bootstrap values for truncated/horizon-end steps.

### B. Vectorized rollout loop (`train.py`, `--num_envs > 1` path)
```
envs = gymnasium.vector.SyncVectorEnv([make_env(seed+i) for i in range(N)])
h = policy.init_hidden(batch_size=N, num_humans, device)
obs_N, _ = envs.reset(seed=...)
for t in range(horizon):
    action_N, logp_N, v_N, h_next = policy(obs_N, h)         # one batched forward
    env_action_N = clip_action_for_env(action_N, ...)
    obs_N, r_N, term_N, trunc_N, info = envs.step(env_action_N)
    done_N = term_N | trunc_N
    buffer.store(obs_N, h, action_N, logp_N, r_N, v_N, done_N, mask=~term_N)
    h = reset_hidden_where_done(h_next, done_N)   # zero hidden rows where done
# horizon end: bootstrap V(next_obs) for every env, respecting truncation
```

**Critical detail — `reset_hidden_where_done`:** `SyncVectorEnv` auto-resets a
done env, so the next obs is already the new episode's first observation. The
LTC hidden must be zeroed for those env rows or the new episode starts with
stale memory (silent leak). This is the #1 silent-bug source in recurrent
vectorized PPO and is covered by a dedicated test.

**terminated vs truncated:** GAE mask uses `terminated` only (collision/goal →
mask 0, no bootstrap). `truncated` (timeout) and horizon-cut keep bootstrap
V(s_next). This mirrors the existing single-env logic.

### C. Vectorized PPO update (`ppo.py`, new `update_vectorized`)
- Pull `(N, T, ...)` tensors from the buffer.
- **Done-masked GAE**: process each env independently along the time axis,
  resetting the advantage accumulator at `done` and using the stored bootstrap.
- Split `(N, T)` into seq_len=16 windows that never cross an env boundary or an
  episode boundary; use the stored hidden at each window start (vectorized
  `_extract_subsequences`).
- Reuse unchanged: clipped surrogate, clipped value loss, return RMS
  normalization, advantage normalization, KL early-stop, grad clip.

## Backward compatibility
- `--num_envs 1` (default for the legacy cell) → existing single-env `update()`,
  byte-identical behavior; v9 reproducibility preserved.
- `--num_envs N>1` → new path.
- Bug 5 (value-loss scale mismatch) is **out of scope** here — separate fix, not
  bundled, to keep this refactor's effect measurable.

## Testing (TDD)
1. **GAE equivalence (safety anchor):** feed the *same* transition sequence to
   the legacy episode-aware GAE and the new done-masked vectorized GAE → results
   must match to floating-point tolerance. Proves the refactor preserves the
   advantage math.
2. **Hidden reset:** after a `done` at env i, hidden row i is zero; other rows
   unchanged.
3. **Buffer shape/bootstrap:** N×T accumulation correct; bootstrap placed at
   truncation and horizon end, not at termination.
4. **Subsequence boundary:** no seq_len window mixes two envs or crosses an
   episode boundary within an env.
5. **Smoke:** `--num_envs 4 --episodes 50` runs on local GPU without crash; kl /
   entropy / std diagnostics healthy.

## Performance expectation
N=16 → 2048 transitions/update (vs ~500). Env steps remain serial on CPU
(SyncVectorEnv) but the policy forward is batched and update quality rises ~4×.
AsyncVectorEnv (CPU parallelism) is a follow-up, not in this scope.

## Files
- `sncp_ppo/vec_buffer.py` (new)
- `sncp_ppo/ppo.py` (`update_vectorized` added; `update` untouched)
- `sncp_ppo/train.py` (`--num_envs` + vectorized rollout path; `make_env` helper)
- `test_vec_buffer.py`, `test_vec_gae.py` (new)

## Out of scope (future)
- AsyncVectorEnv (CPU-parallel stepping)
- Bug 5 value-loss fix, reward shaping (#2), obs additions (#5), LTC→GRU (#6) —
  each measured separately *after* this lands and the data pipeline is healthy.
