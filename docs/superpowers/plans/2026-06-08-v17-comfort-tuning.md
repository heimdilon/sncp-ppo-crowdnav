# v17 Comfort Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one controlled v17 experiment that relaxes the comfort penalty from 6.0 to 5.0 during training, preserving v16 replay and AutoNCP while testing whether timeout/freezing drops without losing genuine avoidance.

**Architecture:** Add a configurable `comfort_coeff` to the environment and training CLI while keeping the default at 6.0 so v15/v16 baselines remain reproducible. The v17 Colab command sets only `--comfort_coeff 5.0` and `--save_path checkpoints/sncp_ppo_v17.pt`; evaluation uses the existing post-run pipeline with `--output_dir eval_v17`.

**Tech Stack:** Python, Gymnasium env, PyTorch PPO training, pytest, existing Colab notebook/runbook.

---

## Evidence Behind This Plan

v16 replay reduced the v15 collapse but failed the v15 comparison gate:

- Training: best holdout min 56%, final min 46%, `Collapse detected: no`, final std `[0.153, 0.243]`.
- Behavior: trajectories still route around the crowd; nav-time stays far above the v14 121.5-step beeline.
- Failure: N=1 success 36% with timeout 60%; N=5 success 56% vs v15 66%; N=8 success 40% vs v15 50%; N=10 success 44% vs v15 46%.
- `I_sp` is lower than v15 at every density and low-density collision is very low, so there is margin to relax comfort before touching capacity.

Do not combine this with `max_time=60` or capacity changes in the same run.

## Files

- Modify: `crowd_sim/crowd_env.py`
  - Add `comfort_coeff=6.0` constructor argument.
  - Store `self.comfort_coeff`.
  - Compute comfort as `-self.comfort_coeff * I_sp`.
- Modify: `test_reward_paper.py`
  - Preserve the default `-6.0 * I_sp` guard.
  - Add a red test for custom `comfort_coeff=5.0`.
- Modify: `sncp_ppo/train.py`
  - Extract parser construction into `build_parser()`.
  - Add `--comfort_coeff` with default `6.0`.
  - Pass it to every `CrowdSimEnv(...)` used by training and holdout evaluation.
- Modify: `test_vec_curriculum.py` or create `test_train_config.py`
  - Verify the parser exposes `--comfort_coeff`.
  - Verify environment creation receives the configured coefficient in the vectorized path with a short smoke.
- Modify: `sncp_ppo_colab.ipynb`
  - Bump save path to `checkpoints/sncp_ppo_v17.pt`.
  - Add `COMFORT_COEFF = 5.0`.
  - Add `--comfort_coeff`, `str(COMFORT_COEFF)` to the training command.
  - Keep `REPLAY_RATIO = 0.20`, `TOTAL_STEPS = 2_500_000`, `NUM_ENVS = 16`, `HORIZON = 128`, N=10, and holdouts `easy hard circle`.
- Modify or create readiness tests only if notebook config is changed in-repo.
- Create after Colab run: `eval_v17/` artifacts using the existing post-run pipeline.

---

### Task 1: Environment Comfort Coefficient

**Files:**
- Modify: `test_reward_paper.py`
- Modify: `crowd_sim/crowd_env.py`

- [ ] **Step 1: Add the failing custom-coefficient test**

Add this test below `test_comfort_is_minus_6_times_Isp`:

```python
def test_custom_comfort_coeff_controls_Isp(monkeypatch):
    env = CrowdSimEnv(num_humans=5, scenario='hard', comfort_coeff=5.0)
    env.reset(seed=1)
    monkeypatch.setattr(env, '_compute_social_pressure', lambda: 0.5)
    env.humans_px[:] = 100.0
    env.humans_py[:] = 100.0

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert np.isclose(info['comfort'], -5.0 * 0.5)
```

- [ ] **Step 2: Run the red test**

Run:

```bash
python -m pytest test_reward_paper.py::test_custom_comfort_coeff_controls_Isp -q
```

Expected: fail with `TypeError: CrowdSimEnv.__init__() got an unexpected keyword argument 'comfort_coeff'`.

- [ ] **Step 3: Implement the minimal environment change**

In `crowd_sim/crowd_env.py`, change the constructor signature:

```python
def __init__(
    self,
    num_humans=5,
    time_step=0.25,
    max_time=50.0,
    scenario='circle',
    human_dodge_robot=False,
    randomize_layout=True,
    comfort_coeff=6.0,
):
```

Store the coefficient after `self.max_time = max_time`:

```python
self.comfort_coeff = comfort_coeff
```

Change the comfort reward:

```python
r_s = -self.comfort_coeff * I_sp
```

- [ ] **Step 4: Verify env tests**

Run:

```bash
python -m pytest test_reward_paper.py -q
```

Expected: all tests in `test_reward_paper.py` pass, including the default `-6.0` guard and the custom `-5.0` guard.

---

### Task 2: Training CLI Wiring

**Files:**
- Modify: `sncp_ppo/train.py`
- Modify: `test_vec_curriculum.py` or create `test_train_config.py`

- [ ] **Step 1: Add parser and env-wiring tests**

If creating `test_train_config.py`, add:

```python
from sncp_ppo.train import build_parser


def test_train_parser_exposes_comfort_coeff():
    args = build_parser().parse_args(['--comfort_coeff', '5.0'])
    assert args.comfort_coeff == 5.0


def test_train_parser_default_preserves_v16_comfort_coeff():
    args = build_parser().parse_args([])
    assert args.comfort_coeff == 6.0
```

- [ ] **Step 2: Run the parser tests red**

Run:

```bash
python -m pytest test_train_config.py -q
```

Expected: import failure because `build_parser` does not exist yet.

- [ ] **Step 3: Extract `build_parser()` and add `--comfort_coeff`**

Move the parser construction from the `if __name__ == '__main__':` block into:

```python
def build_parser():
    parser = argparse.ArgumentParser(description='Train SNCP-PPO with curriculum + holdout eval.')
    ...
    parser.add_argument(
        '--comfort_coeff',
        type=float,
        default=6.0,
        help='Social-pressure comfort penalty coefficient. v15/v16 default is 6.0; v17 candidate uses 5.0.',
    )
    ...
    return parser
```

Then the bottom block becomes:

```python
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    ...
    train(args)
```

- [ ] **Step 4: Pass the coefficient into env creation**

Update every training-side `CrowdSimEnv(...)` call:

```python
env = CrowdSimEnv(num_humans=num_humans, scenario=scenario, comfort_coeff=args.comfort_coeff)
```

```python
env = CrowdSimEnv(num_humans=args.num_humans, scenario='easy', comfort_coeff=args.comfort_coeff)
```

```python
eval_env = CrowdSimEnv(num_humans=args.num_humans, scenario='circle', comfort_coeff=args.comfort_coeff)
```

If a nested env factory lacks `args`, add a `comfort_coeff` parameter to that helper and pass `args.comfort_coeff` from the caller.

- [ ] **Step 5: Verify CLI tests**

Run:

```bash
python -m pytest test_train_config.py test_reward_paper.py -q
```

Expected: parser tests and reward tests pass.

---

### Task 3: Colab v17 Run Configuration

**Files:**
- Modify: `sncp_ppo_colab.ipynb`
- Modify or create tests for notebook readiness if the existing readiness tests assert v16-specific values.
- Modify: `docs/superpowers/plans/2026-06-08-v17-comfort-tuning.md` only if implementation details change during execution.

- [ ] **Step 1: Add the notebook regression test**

Add a test that parses `sncp_ppo_colab.ipynb` and asserts the training cell contains:

```python
SAVE_PATH = 'checkpoints/sncp_ppo_v17.pt'
COMFORT_COEFF = 5.0
'--comfort_coeff'
str(COMFORT_COEFF)
REPLAY_RATIO = 0.20
TOTAL_STEPS = 2_500_000
```

Expected red result: the current notebook is still configured for v16.

- [ ] **Step 2: Patch the notebook**

In the v16/v17 training cell, set:

```python
SAVE_PATH = 'checkpoints/sncp_ppo_v17.pt'
COMFORT_COEFF = 5.0
```

Add to the command list:

```python
'--comfort_coeff', str(COMFORT_COEFF),
```

Keep all other training arguments unchanged from v16.

- [ ] **Step 3: Verify notebook readiness tests**

Run:

```bash
python -m pytest test_v16_run_readiness.py test_post_run_pipeline.py -q
```

Expected: pass after updating any intentionally version-specific assertions to allow the v17 comfort run.

---

### Task 4: Local Smoke Before A100

**Files:**
- No production file changes.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python -m pytest test_reward_paper.py test_train_config.py test_vec_curriculum.py -q
```

Expected: pass.

- [ ] **Step 2: Run a tiny vectorized smoke**

Run:

```bash
python -m sncp_ppo.train --num_envs 2 --horizon 8 --total_steps 64 --eval_freq_updates 1 --num_humans 3 --seed 7 --lr 5e-05 --target_kl 0.01 --curriculum_replay_ratio 0.2 --comfort_coeff 5.0 --holdout_scenarios easy hard circle --holdout_episodes 2 --save_path checkpoints/sncp_ppo_v17_smoke.pt
```

Expected: exit 0, stdout includes replay diagnostics and no shape errors.

- [ ] **Step 3: Run full regression**

Run:

```bash
python -m pytest -q
```

Expected: full suite green.

- [ ] **Step 4: Commit and push**

Commit message:

```text
v17: configure comfort tuning experiment

Add a configurable comfort coefficient and wire the v17 Colab run to use 5.0 while preserving v15/v16 defaults at 6.0. This isolates over-conservative timeout tuning from replay, capacity, and environment changes.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

Push to `main` for Colab.

---

### Task 5: Colab Run and Evaluation

**Files:**
- Create after run: `eval_v17/`
- Create after run: `checkpoints/sncp_ppo_v17.pt`
- Preserve locally: `logs/training_<v17>.csv`

- [ ] **Step 1: Run Colab training**

Command shape:

```bash
python -m sncp_ppo.train --num_envs 16 --horizon 128 --total_steps 2500000 --eval_freq_updates 20 --num_humans 10 --seed 42 --lr 5e-05 --lr_end_factor 0.1 --target_kl 0.01 --curriculum_replay_ratio 0.2 --comfort_coeff 5.0 --holdout_scenarios easy hard circle --holdout_episodes 50 --save_path checkpoints/sncp_ppo_v17.pt
```

- [ ] **Step 2: Run post-run evaluation**

After training:

```bash
python run_v16_post_eval.py --checkpoint checkpoints/sncp_ppo_v17.pt --training_csv logs/<v17_training_csv>.csv --output_dir eval_v17
```

- [ ] **Step 3: Judge v17**

Read:

```text
eval_v17/artifact_verification.md
eval_v17/comparison_vs_v15.md
eval_v17/training_diagnostics.md
eval_v17/density_sweep.csv
eval_v17/traj_hard_n5.png
eval_v17/traj_hard_n10.png
```

Required to accept v17:

- N=1 timeout materially below v16's 60% without collision exceeding v15's 30%.
- N=5 success at least recovers toward v15 66%.
- N=10 success exceeds v15 46% or at minimum collision falls without timeout/freezing increase.
- Nav-time remains well above 121.5 beeline and trajectories still route around clusters.
- `I_sp` remains no worse than v15 beyond the existing comparison tolerance.

---

## Self-Review

- Spec coverage: the plan changes only comfort coefficient for training; replay, NCP, non-reactive pedestrians, speed parity, and max_time stay fixed.
- Placeholder scan: no `TBD`, no unspecified test commands, no combined comfort/max_time run.
- Type consistency: `comfort_coeff` is the same name in env constructor, parser arg, notebook variable, and tests.
