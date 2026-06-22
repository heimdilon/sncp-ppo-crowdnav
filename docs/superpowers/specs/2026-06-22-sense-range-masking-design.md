# v35 — Sense-range masking (paper-fidelity perception, high-N washout)

Date: 2026-06-22
Branch: `feat/v35-sense-range`

## Goal

Limit the robot's crowd perception to humans within a **6 m sensing radius** (the paper's
challenging-scenario sense range), matching Ao et al. 2026, instead of our current behavior of
observing **all** humans. Built on the v30 champion, **single-variable, model-only**. Two motivations:
(1) **fidelity** — for a fair reproduction we must match the paper's limited sensing (right now we
give the robot more perception than the paper had); (2) **mechanism** — feeding all N humans into the
attention/pool at high N dilutes the nearest threat (washout); masking distant humans could reduce
the high-N collision deficit, which is exactly where we lag the paper.

## Diagnosis (why sensing range, found by the user)

Verified against the paper PDF and our code:
- **Paper** (§5.1.2 + Fig 6b + Table 1): the robot senses only humans within a radius — **4 m
  (standard) / 6 m (challenging)**; Fig 6 draws this as a dashed circle, humans outside are not
  sensed. (Table 1 also lists "observation range 3 m" — a minor internal inconsistency; the
  scenario-specific 4/6 m is what Fig 6 depicts.)
- **Us** (`crowd_sim/crowd_env.py:107-108, 383`): the robot observes **all** `num_humans`; the env
  stores `sense_range` (4/6 from `PAPER_SCENARIO_CONFIG`) but explicitly does **not** apply it
  ("obs masking by range is intentionally out of scope here").

This is the only concrete regime mismatch remaining (d_col 0.3, robot/human 1.0, arena 10/15,
invisible-robot ORCA, reward/γ/clip/dt/lr all already match). v30 lags the paper most in **high-N
collision** (N=20: 20.8% collision vs paper ~4%). Masking to the paper's 6 m gives the policy only
the nearby, relevant neighbours — directly attacking the washout that v30's mean+max only partly fixed.

## Verified ground truth (current code, 2026-06-22)

- `forward` (models.py:283-316): `spatial_edges = obs['spatial_edges']` is `[B, N, 6]` in the
  **robot-local frame**; `spatial_edges[:, :, 0]` = local dx (m), `[:, :, 1]` = local dy (m). The
  spatial LTC produces `M_rh` `[B, N, 256]`; then `u_att = self._attention_pool(M_rh, m_rr, num_humans)`.
- `_attention_pool` (models.py:230-251): single-head + meanmax is the v30/v35 path — `attn_scores`
  `[B, N, 1]` → softmax over N → `a_mean`; `a_max = M_rh.max(dim=1)`; `pool_merge(cat[a_mean, a_max])`.
  (Multi-head branch is v33's, off here.)
- `build_policy_for_checkpoint` (models.py:~367) auto-detects pre_mlp / attn_count_scaling /
  meanmax_pool / node dims / attn_heads / action_dist from the state_dict.
- The robot↔human distance is directly available as `hypot(spatial_edges[:,:,0], spatial_edges[:,:,1])`
  — no env change needed; masking can live entirely in the model (the prior flag pattern).

## Design (single-variable, model-only, auto-detected)

### New constructor arg `sense_range=0.0` (default = off, byte-identical)

`SNCPPolicy.__init__(..., sense_range=0.0)`. `sense_range <= 0` → no masking (every v14–v34 checkpoint
unchanged). `sense_range > 0` → register buffer `_sense_range = tensor(float(sense_range))` for
auto-detect.

### `forward`: compute the visibility mask from the raw obs

After reading `spatial_edges`, before `_attention_pool`:
```
mask = None
if self.sense_range > 0:
    dist = torch.hypot(spatial_edges[:, :, 0], spatial_edges[:, :, 1])   # [B, N] metres
    mask = dist <= self.sense_range                                       # [B, N] bool, True = visible
u_att = self._attention_pool(M_rh, m_rr, num_humans, mask)
```

### `_attention_pool(self, M_rh, m_rr, num_humans, mask=None)`: apply the mask

`mask=None` → unchanged (byte-identical). When given (`[B, N]` bool):
- **Attention:** set masked humans' scores to `-inf` before softmax (weight 0). Single-head:
  `attn_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))`. Multi-head (for generality):
  mask on the key dim.
- **Max-pool branch:** exclude masked humans — `M_rh.masked_fill(~mask.unsqueeze(-1), float('-inf')).max(dim=1)`.
- **All-masked guard:** for rows where no human is within range (`~mask.any(dim=1)`), the softmax over
  all `-inf` is NaN and the max is `-inf`; replace those rows' `a_mean` and `a_max` with **0** (a
  well-defined "no nearby threat" crowd vector). This keeps the output finite.

### Auto-detect + wiring

- `build_policy_for_checkpoint`: `sense_range = float(state_dict['_sense_range']) if '_sense_range' in
  state_dict else 0.0`; pass to `SNCPPolicy`. The model masks internally → **no eval/env change**; a
  v35 checkpoint masks at eval automatically, v30 (no buffer) does not.
- `train.py`: `--sense_range` (float, default 0.0) + `build_or_load_policy` passthrough.

### v35 training config = v30 recipe + `--sense_range 6.0`
`--pre_mlp --meanmax_pool --num_humans_range 10 20 --total_steps 2_500_000 --sense_range 6.0`
(challenging 6 m). No node/attn/beta flags.

## Components / files

- `sncp_ppo/models.py` — `__init__` (`sense_range` arg + `_sense_range` buffer), `forward` (mask
  compute), `_attention_pool` (mask arg + apply + all-masked guard), `build_policy_for_checkpoint`.
- `sncp_ppo/train.py` — `--sense_range` arg + passthrough.
- `tests/test_sense_range.py` — NEW.
- `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`, `tests/test_post_run_pipeline.py`,
  `tests/test_v16_run_readiness.py` — v34→v35 markers; drop `--action_dist beta`, add `--sense_range 6.0`.

No change to `crowd_sim/crowd_env.py` (distance is already in the obs) or `eval_report.py`.

## Testing (TDD, red-first)

`tests/test_sense_range.py`:
1. `sense_range=6.0` build → `_sense_range` buffer == 6.0; `sense_range=0` (default) → no buffer.
2. Mask mechanism: craft an obs where 1 human is at 2 m (near) and others at 12 m (far); with
   `sense_range=6` the far humans get ~0 attention weight (recompute alpha) and the pooled vector
   equals the near-only pool; without masking it differs.
3. All-masked: every human at 20 m (> 6 m) → forward returns finite `mu/std/value` (no NaN), pooled
   crowd vector is zero.
4. Auto-detect roundtrip: save a `sense_range=6` state_dict → `build_policy_for_checkpoint` rebuilds
   with `_sense_range==6`, loads with 0 missing/unexpected.
5. v30-compat: a `meanmax_pool` (sense_range 0) state_dict → no `_sense_range`, byte-identical forward.
6. `build_or_load_policy` respects `--sense_range`.

Plus version-marker bumps v34→v35. Full suite green (`--basetemp=./.pytmp`,
`C:/ProgramData/miniconda3/python.exe`); readiness `pass` "v35 ... ready"; CLI smoke
(`-m sncp_ppo.train --pre_mlp --meanmax_pool --sense_range 6.0 --num_humans_range 10 20
--total_steps 4096 ...`) exits 0.

## Evaluation (run-time, post-Colab)

Honest 5-seed sweep (base-conda, seeds 100–500 × 50 ep, N=5/10/15/20, paper_challenging, robot 1.0,
human 1.0, max_time None, goal_noise 0) on `sncp_ppo_v35.pt` — masking is auto-detected, so the eval
applies it. Compare to the **v30 baseline** (97.2/89.6/85.6/79.2; collision 2.8/10.4/14.4/20.8) with
Wilson CIs + two-proportion z (Bonferroni α=0.0125), success and collision (reuse `_sweep`/`_analyze`
→ `_v35`).

**Decision rule:** sense-masking helps iff high-N success rises and/or collision drops (esp N=15/20)
with no regression at N=5/10 and timeout 0, vs v30. A flat/negative result is reported honestly —
but note that even a flat result still makes the comparison **fairer** (matches the paper's sensing),
which is worth keeping for the writeup. Write the verdict to memory.

## Out of scope / deferred

- Env-level masking (literal obs removal) — behaviourally equivalent for the action; more plumbing.
- Standard-scenario 4 m (we train/eval challenging = 6 m).
- Any other lever (capacity/attention/distribution — those failed).
- Multi-seed training.

## Irreversibility note

Model code; v35's checkpoint masks internally (auto-detected; v14–v34 stay loadable and unmasked).
Work on `feat/v35-sense-range`; merge to `main` + push only at the finishing step after the user
confirms (Colab pulls `main`).

## Honest caveats (carried into the verdict)

- **Removes information:** masking distant humans could also hurt (no early warning). At parity speed
  (1.0 m/s) a human > 6 m away is > 6 s out, so this is likely safe — but it is a hypothesis.
- **May not help if collisions are from NEAR humans:** if our high-N collisions already involve
  humans within 6 m, masking the far ones won't reduce them. The result is genuinely uncertain.
- **Model-internal mask still encodes distant humans** in the spatial LTC (then discards them from the
  pool). Behaviourally faithful for the action; not a literal obs removal.
- **Fidelity value independent of outcome:** matching the paper's 6 m sensing makes the comparison
  fairer regardless of whether the metric improves — worth keeping either way.
- Single training seed; 6 m fixed (challenging).
