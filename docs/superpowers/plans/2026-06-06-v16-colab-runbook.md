# v16 Colab Runbook

Purpose: run the single-variable v16 replay experiment and produce the artifact set needed to decide
whether replay fixes v15 catastrophic forgetting without regressing real avoidance.

## Preconditions

- Colab has pulled current `main` and has `run_v16_post_eval.py`.
- Runtime is A100 if available.
- Do not change reward, environment defaults, NCP capacity, pedestrian reactivity, or robot speed.
- The only v16 training variable is `--curriculum_replay_ratio 0.20`.

## Notebook sequence

1. Run setup cells through dependency install and `git pull`.
2. Run the v16 readiness preflight:
   - `python verify_v16_run_ready.py`
   - `eval_v16/run_readiness.md` should report `Overall status: pass`
3. Run the v16 full training cell:
   - `SAVE_PATH = 'checkpoints/sncp_ppo_v16.pt'`
   - `TOTAL_STEPS = 2_500_000`
   - `--num_humans 10`
   - `--curriculum_replay_ratio 0.20`
   - holdout scenarios: `easy hard circle`
   - the cell raises `SystemExit` if training exits nonzero; do not continue to evaluation after a
     failed training subprocess
4. Run the evaluation cell. It calls `run_v16_post_eval.py` and should write:
   - `eval_v16/density_sweep.csv`
   - `eval_v16/density_sweep.json`
   - `eval_v16/density_sweep.png`
   - `eval_v16/report.md`
   - `eval_v16/comparison_vs_v15.md`
   - `eval_v16/training_diagnostics.json`
   - `eval_v16/training_diagnostics.md`
   - `eval_v16/artifact_verification.md`
   - `eval_v16/traj_hard_n5.png`
   - `eval_v16/traj_hard_n10.png`
5. Run the training-curves cell if you want the Colab plot. It should write:
   - `training_curves_colab.png`

## Equivalent CLI sequence

Preferred one-command post-run pipeline:

```bash
python verify_v16_run_ready.py
python run_v16_post_eval.py --checkpoint checkpoints/sncp_ppo_v16.pt --output_dir eval_v16
```

Run the readiness preflight before training. After training, the post-run pipeline uses the newest
`logs/training_*.csv`, writes the density report, v15 comparison, training diagnostics, and artifact
verification, then exits nonzero only if the final artifact verification is `fail`.

Equivalent manual sequence:

```bash
python evaluate_policy_report.py --checkpoint checkpoints/sncp_ppo_v16.pt --output_dir eval_v16 --densities 1 3 5 8 10 --scenario hard --n_episodes 50 --seed 100 --trajectory_densities 5 10
python compare_policy_reports.py --baseline eval_v15/density_sweep.json --candidate eval_v16/density_sweep.json --output eval_v16/comparison_vs_v15.md
python analyze_training_log.py --csv logs/<training_csv>.csv --output_dir eval_v16
python verify_v16_artifacts.py --checkpoint checkpoints/sncp_ppo_v16.pt --eval_dir eval_v16 --output eval_v16/artifact_verification.md
```

Use the newest `logs/training_*.csv` from the v16 run for `<training_csv>`.

## Decision gates

Read these files in order:

1. `eval_v16/artifact_verification.md`
   - `fail`: artifact set is incomplete or a hard gate failed. Do not treat v16 as evaluated.
   - `warn`: artifacts are complete, but at least one gate needs engineering interpretation.
   - `pass`: artifact-level gates passed; still inspect trajectories.
2. `eval_v16/comparison_vs_v15.md`
   - A `fail` row means v16 regressed against v15 on real-avoidance gates.
   - A high-density `warn` means N=10 did not improve; this does not satisfy the v16 objective.
3. `eval_v16/training_diagnostics.md`
   - Replay ratio should be logged and near 20%.
   - `Collapse detected: yes` means replay did not fix the v15 failure mode.
   - Check policy std deltas for drift during the final phases.
4. `eval_v16/traj_hard_n5.png` and `eval_v16/traj_hard_n10.png`
   - The robot should route around the crowd.
   - A straight path near v14's 121.5-step beeline baseline is failure even if success is high.

## Update after run

If the artifact set is complete, update:

- `AGENTS.md`: v16 result summary, density table, collapse verdict, trajectory verdict, next experiment.
- `~/.claude/projects/C--Users-kor-a-Desktop-deneme/memory/sncp-paper-vs-impl.md`: detailed result log.

Only claim improvement if success/collision improve while nav-time remains well above the v14 beeline
baseline and `I_sp` stays low.
