# v29: attention count-scaling (Eq 13 n-factor)

Date: 2026-06-18
Branch: `v29-attn-count-scaling`

## Goal

Test whether the paper's Eq 13 attention count-scaling (the `n/√d_k` factor that feeds the
pedestrian count into the attention softmax) closes the remaining gap to the paper at the
highest densities. **Single-variable ablation: v29 = v28 (pre-MLP + density curriculum) +
`--attn_count_scaling`, nothing else changes.**

## Ground truth (verified against PDF + code)

- **Eq 13:** `α = softmax((n/√d_k) Q Kᵀ)`, with `n` = number of humans, `d_k = 64`. The `n`
  factor sharpens the attention as the crowd grows (count enters the softmax temperature).
- **Code is faithful + fully wired (zero model code to write):**
  - `models.py:174-176`: `attn_scores = QKᵀ / 8.0` (8 = √64); `if attn_count_scaling:
    attn_scores *= num_humans` → `softmax((n/√d_k)·QKᵀ)` exactly.
  - `models.py:33-35`: a `_attn_count_scaling` buffer is registered ONLY when on, so default
    checkpoints stay byte-identical and the variant is auto-detectable.
  - `models.py:274`: `build_policy_for_checkpoint` auto-detects `'_attn_count_scaling' in
    state_dict`.
  - `train.py:1092` CLI `--attn_count_scaling` (store_true); `train.py:265` threads it to the
    policy.
- **v28 honest baseline (local multi-seed, 5 seeds × 50 ep):** N=5/10/15/20 =
  94.4 / 87.6 / 79.2 / 73.2 %. Paper holds ~93-95 % across 10-20; our remaining gap is
  concentrated at the highest density (N=20 ≈ 20 pp).

## Approach (chosen) + alternatives

**Flip `--attn_count_scaling` as-is.** This is the only paper-faithful path: the `n` factor
is written in Eq 13 and already implemented exactly. No alternative scaling is worth pursuing
(any other factor would be non-faithful invention — YAGNI).

Prior context: this flag was probed once in the *wrong* (v23-era antipodal) regime and showed
no gain. v29 retests it in the current v28 regime — the same situation as pre-MLP, which was
also null in the wrong regime and then a breakthrough in the right one. A null result here is
still valuable (a clean negative: "count-scaling does not help in our setup").

## Scope -- what changes (harness only, zero model code)

1. **Notebook training cell:** add `--attn_count_scaling`; `SAVE_PATH →
   checkpoints/sncp_ppo_v29.pt`; keep `--pre_mlp` and `--num_humans_range 10 20`
   (v29 = v28 + the new flag). Eval/persist/diagnostics cells → v29 paths.
2. **`run_readiness.py`:** bump v28→v29 markers; add an `--attn_count_scaling` token to
   TRAINING_TOKENS (keep the `--pre_mlp` and `--num_humans_range` tokens).
3. **Tests (TDD):** version-marker tests → v29; assert the training cell carries
   `--attn_count_scaling`, `--pre_mlp`, `--num_humans_range`.

No changes to `models.py`, `train.py` (logic), `crowd_env.py`, `eval_report.py`.

## Holdout / selection (unchanged)

Holdout stays `paper_standard` (N=5) + `paper_challenging` (N=10) for comparability with
v28; the independent local multi-seed sweep judges the full N=5/10/15/20 range.

## Evaluation protocol (the real deliverable)

- Train v29 on Colab (main pull). Download `sncp_ppo_v29.pt` + training CSV. (The Colab eval
  cell has been non-functional for v27/v28; not needed — we eval locally. Reminder: copy the
  checkpoint to the repo root.)
- Local multi-seed sweep, IDENTICAL protocol to v26/v27/v28: 5 seeds (100–500) × 50 ep at
  N=5/10/15/20, `paper_challenging`, robot 1.0, human 1.0, max_time None, goal_noise 0.
- Compare v29 vs v28 (94.4 / 87.6 / 79.2 / 73.2) pooled success ± 95 % CI per density.
  **Decision rule:** count-scaling helps iff v29's CI clears v28's at the high densities
  (especially N=20, the remaining gap), with no regression at N=5/10. Confirm timeout stays 0.
  A flat/negative result is reported honestly as a clean negative.

## Caveats

1. **Single training seed (42)** — as with v26/v27/v28.
2. **No capacity confound (cleaner than pre-MLP):** `attn_count_scaling` adds only a scalar
   buffer; it is a pure re-weighting of attention, not added capacity. A gain (or null) is
   attributable to the `n`-scaling mechanism itself.

## Out of scope / deferred

- Further levers (we are near the paper; this is the last concrete paper-fidelity item).
- Multi-seed robustness re-runs of the v27/v28/v29 line (separate hardening effort).
- Report update with the v27→v29 results (separate task).
