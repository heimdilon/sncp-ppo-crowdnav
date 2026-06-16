# v27: pre-MLP (Eq 11) architecture-fidelity ablation

Date: 2026-06-17
Branch: `v27-pre-mlp-ablation`

## Goal

Test whether adding the paper's Eq 11 pre-MLP edge embedding (raw edge → MLP →
256-dim embedding → NCP) narrows the v26 collision gap to the paper's ~94 %
challenging success. **Single-variable ablation: v27 = v26 + `--pre_mlp`, nothing
else changes.**

## Ground truth (verified against the PDF)

- **Eq 11 (p7):** `embedding = φ(inputs, W_mlp)` then `feature, h' = ψ(embedding,
  h_state, W_ncp)`. Raw edge inputs (spatial `E_hr`, temporal `E_rr`) pass through
  an MLP **before** the NCP. v26 (`pre_mlp=False`) feeds the raw 2/6-dim signal
  straight into the NCP, skipping the MLP — a genuine deviation from Eq 11.
- **Dims (p8):** "encoding vectors for the time edge and robot node are adjusted to
  256 and 128 ... attention dimension 64." Our `pre_mlp` matches exactly: temporal
  embedding 256, robot node 128, attention 64. Spatial-edge dim is unspecified in
  the paper; we use 256 (symmetric to temporal).
- **v26 honest baseline (2026-06-17 local multi-seed, 5 seeds × 50 ep):**
  challenging N=5/10/15/20 = 74.8 / 61.6 / 53.2 / 43.6 %, timeout 0 % at every N
  (failure is pure collision). Two gaps vs paper 94 %: (a) 32 pp at the *trained*
  density N=10, (b) falloff toward high N.
- `pre_mlp` is **already implemented** (`models.py`) and **CLI-wired**
  (`train.py --pre_mlp`); `build_policy_for_checkpoint` auto-detects it from the
  checkpoint keys. **Zero model/env/reward code to write.**

## Acknowledged caveats

1. **Capacity confound.** `pre_mlp` adds two MLPs (2→128→256, 6→128→256) and grows
   the LTC sensory input (2→256, 6→256) — a large parameter increase. A gain cannot
   be cleanly attributed to "Eq 11 ordering" vs "more capacity." Accepted because
   the paper's own architecture carries this capacity.
2. **Prior null result.** `pre_mlp` was probed once in the *wrong* (v22 antipodal)
   regime and showed no benefit. v27 retests it in the correct v26 paper regime;
   there is no guarantee of a different outcome. Cost ≈ 7 h Colab.

## Scope — what changes (harness only, no model/env/reward)

1. **Notebook training cell:** add `--pre_mlp`; `SAVE_PATH → checkpoints/sncp_ppo_v27.pt`.
   Everything else byte-identical to v26 (seed 42, 2.5M steps, `paper_challenging`,
   `--num_humans 10`, holdouts `paper_standard paper_challenging`, `--holdout_episodes 50`,
   LR 1e-4 + 0.1 decay, target_kl 0.01, bootstrap_easy 200k, 16 envs × 128 horizon,
   robot_vpref 1.0). Budget/geometry/comfort/d_col are env-derived (unchanged).
2. **Notebook eval / persist / trajectory cells:** v27 paths (`eval_v27`, `--version 27`,
   `sncp_ppo_v27.pt`); `--baseline_json eval_v26/density_sweep.json` for continuity
   (its pass/fail verdict is non-authoritative — the real comparison is the local
   multi-seed sweep below).
3. **`run_readiness.py`:** bump v26→v27 markers, and add a NEW check that the
   training cell contains `--pre_mlp` (the defining feature of v27).
4. **Tests (TDD):** version-marker tests → v27; persist bundle → `eval_v27`; a NEW
   single-variable guard test asserting the v27 training cell equals the v26 config
   plus exactly `--pre_mlp` and the v27 `SAVE_PATH` (no other drift).

## Evaluation protocol (the real deliverable)

- Train v27 on Colab (main pull). Download `sncp_ppo_v27.pt` + training CSV + eval
  bundle.
- **Local multi-seed sweep**, IDENTICAL protocol to the v26 honest sweep: 5 seeds
  (100–500) × 50 ep at N=5/10/15/20, `paper_challenging`, robot 1.0, human 1.0,
  max_time None→50 s, goal_noise 0.
- Compare v27 vs v26 pooled success ± 95 % CI per density. **Decision rule:**
  pre_mlp helps iff v27's CI clears v26's CI at one or more densities — especially
  N=10 (the fundamental gap a). Also report timeout (expect 0) and collision.
- Caution carried from #5 finding: do NOT compare best-holdout numbers (selection-
  biased); only multi-seed sweeps.

## Single-variable discipline

v27 differs from v26 ONLY by `pre_mlp=True`. Same seed, steps, regime, budget,
geometry, comfort, d_col, LR. The notebook guard test enforces this.

## Out of scope / deferred

- `attn_count_scaling` (Eq 13 `n/√d_k`; the PDF DOES show the `n` factor — Gemini
  was wrong, I had deferred to it). Paper-faithful, but a separate single-variable
  test later.
- N~U(10,20) density curriculum (targets gap b) — next experiment if pre_mlp helps
  gap a.
- Best-checkpoint selection-bias fix (more holdout episodes / periodic checkpoints)
  — methodology improvement, tracked separately.
