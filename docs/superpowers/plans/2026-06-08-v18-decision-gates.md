# v18 Decision Gates After v17

## Goal

Pick the next single-variable A100 experiment only after the completed v17 artifacts explain why
generalist holdout success stayed low. Do not launch v18 from stdout alone.

v17 changed only one training variable: `comfort_coeff=6.0 -> 5.0`, keeping replay 0.20,
`max_time=50`, approach 1, non-reactive pedestrians, speed parity, and AutoNCP unchanged.
User-provided stdout through update ~1120 showed stable PPO but weak learning:

- Best generalist min stayed at 16% from update 310.
- Hard phase did not improve the generalist score.
- After circle/N=10 began, holdout min remained mostly 0-2%.
- `std` stayed controlled around `[0.156, 0.239]`; KL was mostly low.

This is evidence against numerical PPO collapse. It is not enough to distinguish timeout/freezing,
collision-dominant high-density failure, or scenario forgetting.

## Required Inputs

From Colab, preserve:

```text
checkpoints/sncp_ppo_v17.pt
checkpoints/sncp_ppo_v17_final.pt   # if present
logs/training_20260608_070945.csv
eval_v17_artifacts.zip              # if the post-run cell completed
```

Then run or re-run locally:

```bash
python stage_colab_run_artifacts.py --version 17
python run_post_eval.py --version 17 --training_csv logs/training_20260608_070945.csv
python select_v18_candidate.py --version 17
python verify_v18_ready.py
```

The version-aware post-run wrapper derives `checkpoints/sncp_ppo_v17.pt` and `eval_v17/`, then
regenerates the density sweep, v15 comparison, training diagnostics, and artifact verification.
The v18 selector then writes `eval_v17/v18_decision.md/json`; use it as a structured summary, not a
replacement for inspecting the trajectory plots.
The v18 readiness gate writes `eval_v17/v18_ready.md` and fails until all required v17 artifacts,
trajectory PNGs, and a non-waiting v18 decision are present.

The active v17 run started before the new holdout `avg_steps/avg_I_sp/min_d_min` CSV columns were
added. Its CSV should still have per-scenario success/collision/timeout/reward, and the density
sweep report still provides nav-time, I_sp, and trajectory artifacts.

## Verdict Before v18

Reject v17 if any of these hold:

- Artifact verifier fails the v15 comparison gates.
- Density sweep regresses success at two or more of N=1/3/5/8/10 versus v15.
- Trajectories stop routing around clusters.
- Nav-time collapses toward the v14 beeline reference (~121.5 steps).
- I_sp rises materially while collision remains high.

Accept v17 only if:

- Real avoidance is preserved: nav-time remains above beeline, I_sp stays low, and trajectories detour.
- Low-density timeout improves versus v16 without increasing collisions.
- N=10 success improves versus v15/v16, or at least collision falls without timeout/freezing rising.

## v18 Candidate Selection

Choose exactly one branch.

### Branch A: Timeout / Slow-Detour Dominant

Evidence:

- N=1 or N=3 failures are mostly timeout, with low collision.
- Density sweep successful episodes still have long nav-time and detour trajectories.
- I_sp remains low.
- v17 did not become a beeline policy.

Next one-variable run:

```text
v18_max_time60: max_time 50 -> 60, keep comfort_coeff 5.0, replay 0.20, AutoNCP, speed parity.
```

Risk:

- This can mask freezing. The accept gate must check trajectories and successful-episode steps, not
  just success rate.

### Branch B: High-Density Collision Dominant

Evidence:

- N=1/3 are acceptable, but N=8/10 fail mainly by collision.
- Timeout is not the main failure mode at high density.
- Trajectories still attempt detours but cannot find enough clearance.
- I_sp/min distance show crowd-contact pressure.

Next one-variable run:

```text
v18_high_density_training: keep AutoNCP; change only high-density training exposure.
```

Do not increase NCP capacity yet. First isolate training distribution. Acceptable single-variable
options include extending total steps while keeping all hyperparameters fixed, or adding a controlled
phase schedule knob if implemented with TDD and smoke-tested. Pick one, not both.

### Branch C: Easy/Hard Forgetting Dominant

Evidence:

- Final v17 easy/hard holdout drops while circle is being trained.
- Replay ratio observed from CSV is below target or insufficient.
- Policy std is stable, so this is behavioral forgetting, not numerical collapse.

Next one-variable run:

```text
v18_replay30: curriculum_replay_ratio 0.20 -> 0.30, keep comfort_coeff 5.0 and all env/reward settings.
```

Risk:

- More replay may slow high-density learning. Check N=10 collision and success, not only low-density
  recovery.

### Branch D: Comfort Relaxation Damaged Avoidance

Evidence:

- v17 has higher collision or higher I_sp than v16/v15.
- Nav-time shortens toward beeline while success does not improve.
- Trajectories cut through clusters.

Next step:

```text
Do not run comfort_coeff 4.0. Revert comfort candidate and select Branch B or C from v16/v17 evidence.
```

## Review Checklist

Before editing code or Colab for v18:

- Attach `eval_v17/comparison_vs_v15.md`.
- Attach `eval_v17/training_diagnostics.md`.
- Inspect `eval_v17/density_sweep.csv` for success, collision, timeout, avg success steps, I_sp.
- Inspect `eval_v17/traj_hard_n5.png` and `eval_v17/traj_hard_n10.png`.
- Generate and attach `eval_v17/v18_decision.md`.
- Generate and attach `eval_v17/v18_ready.md`.
- State which branch above is supported by evidence.
- State the single variable to change.
- Smoke-test locally before A100.

No v18 run should be started until this checklist is satisfied.
