# v18 Decision Gates After v17

## REVISED DECISION (supersedes Branch A below) — v18 = paper-faithful reward restoration

After a full re-read of the paper (Ao et al. 2026, *Int. J. Social Robotics* 18:52) against the
implementation, the v18 variable is **the reward function `r_g`, not `max_time`.** The earlier
Branch-A pick (`max_time 50 → 60`) is rejected: it treats the *symptom* (timeouts) and the plan
itself warns it "can mask freezing." The N=1 timeout=60% with `I_sp≈0.009` proves the low-density
failure is **pure goal-reaching**, not avoidance — so the fix belongs in the goal reward.

Root cause (best-evidenced): the dense goal reward had drifted from the paper's Eq 18 in two ways
that *both* starve the progress signal and reinforce timeouts:

1. approach coefficient halved `2.0 → 1.0` (a misread of potential-based shaping — it telescopes to
   `k·(d₀−d_f)` regardless of path, so halving cannot "spare detours", it only halves the gradient);
2. an ad-hoc heading penalty `-weight·|angle_diff|` (NOT in the paper), up to `0.157`/step — larger
   than the entire max per-step progress reward (`0.065` at 0.26 m/s) — making progress-while-turning
   net-negative and teaching the policy to optimise heading over arrival.

**v18 change (single concept):** `crowd_env.py` `r_g` → paper Eq 18 exactly — approach `1→2·Δd`,
heading penalty removed. Everything else is held at v16: `comfort_coeff=6.0`, `max_time=50`,
replay `0.20`, non-reactive pedestrians, speed parity, AutoNCP. The notebook + readiness contract
(`sncp_ppo/run_readiness.py`) and `test_reward_paper.py` were updated to match (`MAX_TIME=50.0`,
`test_approach_coefficient_is_2`, `test_no_orientation_penalty`).

Deferred (documented, not changed in v18, to keep the run interpretable):
comfort `-6*I_sp` vs paper `-2`; `I_sp` unbounded vs paper `[0,1]`; non-reactive crowd vs paper ORCA;
`v≥0` action space. See the notebook roadmap (cell 31) for the v19/v20 ordering.

## Goal

Prepare the next single-variable A100 experiment after v17 produced no usable checkpoint/artifacts.
Do not wait for `eval_v17/`: the user confirmed v17 was bad and there is no checkpoint to evaluate.
Use the completed v16 artifact bundle plus v17 stdout as the evidence base.

v17 changed only one variable (`comfort_coeff=6.0 -> 5.0`) while keeping replay 0.20, `max_time=50`,
approach 1, non-reactive pedestrians, speed parity, and AutoNCP unchanged. User-provided stdout through
update ~1120 showed stable PPO but weak learning:

- Best generalist min stayed at 16% from update 310.
- Hard phase did not improve the generalist score.
- After circle/N=10 began, holdout min remained mostly 0-2%.
- `std` stayed controlled around `[0.156, 0.239]`; KL was mostly low.
- User later confirmed there is no v17 checkpoint/artifact to preserve.

Conclusion: discard the comfort-5 branch. Do not lower comfort further and do not use comfort 5.0 as the
base for v18.

## Required Inputs

Authoritative evidence now comes from:

```text
eval_v16/density_sweep.json
eval_v16/comparison_vs_v15.md
eval_v16/training_diagnostics.md
eval_v16/traj_hard_n5.png
eval_v16/traj_hard_n10.png
user-provided v17 stdout through update ~1120
custom-map GIF/action-trace observations
```

The v16 artifact facts that support the next branch:

- Real avoidance is preserved: trajectories route around clusters, nav-time 166-182 is far above the
  v14 121.5-step beeline, and `I_sp` is low.
- Low-density failure is timeout-dominant: N=1 timeout 60% with only 4% collision.
- N=5/N=8 also have timeout pressure (26%) and success regressions.
- N=10 collision remains high, but capacity is not proven while timeout/braking are unresolved.

The custom-map observation adds a separate diagnostic warning: the robot appears not to stop or reverse
and only turns mildly before collision. This matches the current action space: linear velocity is bounded
non-negative (`sigmoid * vpref`, then clipped to `[0, 0.26]`). The custom evaluator now records action
traces and braking metrics; inspect them before any action-space change.

## v18 Candidate Selection

Choose exactly one branch. The selected branch is **Branch A from v16 evidence**, with comfort reverted to
v16's 6.0.

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
v18_max_time60: max_time 50 -> 60, keep comfort_coeff 6.0, replay 0.20, AutoNCP, speed parity.
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
v18_replay30: curriculum_replay_ratio 0.20 -> 0.30, keep comfort_coeff 6.0 and all env/reward settings.
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

Before launching the v18 A100 run:

- Confirm Colab pulled `main` with `SAVE_PATH='checkpoints/sncp_ppo_v18.pt'`.
- Confirm the training cell uses `COMFORT_COEFF = 6.0` and `MAX_TIME = 50.0` (unchanged from v16).
- Confirm `crowd_env.py` `r_g` uses approach `2.0` and has NO `angle_diff` heading term
  (`pytest test_reward_paper.py` covers this: `test_approach_coefficient_is_2`, `test_no_orientation_penalty`).
- Run `python verify_v16_run_ready.py`; `eval_v18/run_readiness.md` must pass.
- Optionally run a custom-map probe and inspect `raw_actions`, `env_actions`, `linear_speeds`, and
  `angular_speeds` in the JSON summary to verify whether braking/turning is the immediate failure mode.
- After training, evaluate with `python run_post_eval.py --version 18 --training_csv logs/<v18_csv>.csv`.
- Judge with success/collision/timeout (expect timeout to DROP at low/mid density), low `I_sp`,
  action traces, and trajectories. NOTE: read nav-time *per density* — a near-straight route at N=1
  is correct (efficiency), not a "beeline regression"; the blanket no-beeline gate is density-blind.

No v18 run should be started if the notebook shows comfort 5.0 or save path v17.
