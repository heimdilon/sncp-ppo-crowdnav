# v16 Vectorized Curriculum Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add anti-forgetting replay to the vectorized PPO curriculum so v15's genuine detour behavior is sustained through the N=8/N=10 phases without changing reward, environment, or NCP capacity.

**Architecture:** Keep the v15 task unchanged: non-reactive pedestrians, speed parity, comfort `-6*I_sp`, approach `1*delta_distance`, and the existing sparse AutoNCP policy. The only training-behavior change is that vectorized PPO can spend a configured fraction of update windows on earlier curriculum phases; each update still uses a single `num_humans` value so rollout tensors remain rectangular.

**Tech Stack:** Python, PyTorch, Gymnasium, ncps, pytest.

---

## File Structure

- Modify `sncp_ppo/train.py`: factor curriculum phases into reusable helpers; add a vectorized replay phase selector; use it in `_train_vectorized`; log whether each update was replay; update CLI help for the now-live vectorized replay path.
- Modify `test_vec_curriculum.py`: add unit tests for replay selection and update the vectorized smoke test to cover `curriculum_replay_ratio`; use `DictReader` for CSV assertions so an added logging column does not make tests brittle.
- Modify `sncp_ppo_colab.ipynb`: bump the training/eval cells to v16 and pass `--curriculum_replay_ratio 0.20`; keep all v15 environment/reward/capacity settings unchanged.
- After the Colab run, update `AGENTS.md` and `~/.claude/projects/C--Users-kor-a-Desktop-deneme/memory/sncp-paper-vs-impl.md` with v16 results and whether replay fixed collapse.

## Evidence Summary

- v15 CSV `logs/training_20260605_140811.csv` shows the best generalist checkpoint at update 460, still in `medium/5h`: easy 66%, hard 60%, circle 36%, min 36%.
- The same run later drops through `hard/8h` and `circle/10h`: by update 1040, easy/hard/circle success is 0/0/0 with timeouts 90/92/64%.
- `sncp_ppo/train.py` has replay logic only in the legacy single-env path; `_train_vectorized` always uses `step_to_phase(total_steps, ...)` and therefore trains monotonically.
- v15 result plots show detour behavior is real: nav-time is about 187 steps rather than v14's about 121.5 straight line, while I_sp remains low. Do not change the task/reward before testing replay.

---

### Task 1: Unit-Test Vectorized Replay Phase Selection

**Files:**
- Modify: `test_vec_curriculum.py`
- Modify later: `sncp_ppo/train.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `test_vec_curriculum.py`:

```python
class _ReplayRng:
    def __init__(self, random_value=0.0, randint_value=0):
        self.random_value = random_value
        self.randint_value = randint_value
        self.randint_bounds = None

    def random(self):
        return self.random_value

    def randint(self, low, high):
        self.randint_bounds = (low, high)
        return self.randint_value


def test_select_vectorized_phase_uses_current_phase_when_replay_disabled():
    from sncp_ppo.train import select_vectorized_phase

    phase, is_replay = select_vectorized_phase(
        steps_seen=800,
        total_steps=1000,
        final_num_humans=10,
        replay_ratio=0.0,
        rng=_ReplayRng(random_value=0.0, randint_value=0),
    )

    assert phase == ('circle', 10, 0.26)
    assert is_replay is False


def test_select_vectorized_phase_samples_only_earlier_phases_for_replay():
    from sncp_ppo.train import select_vectorized_phase

    rng = _ReplayRng(random_value=0.0, randint_value=1)
    phase, is_replay = select_vectorized_phase(
        steps_seen=800,
        total_steps=1000,
        final_num_humans=10,
        replay_ratio=1.0,
        rng=rng,
    )

    assert phase == ('easy_plus', 3, 0.18)
    assert is_replay is True
    assert rng.randint_bounds == (0, 3)


def test_select_vectorized_phase_never_replays_before_second_phase():
    from sncp_ppo.train import select_vectorized_phase

    phase, is_replay = select_vectorized_phase(
        steps_seen=0,
        total_steps=1000,
        final_num_humans=10,
        replay_ratio=1.0,
        rng=_ReplayRng(random_value=0.0, randint_value=0),
    )

    assert phase == ('easy', 1, 0.13)
    assert is_replay is False
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:PYTHONPATH='C:\tmp\codex-pydeps'
& 'C:\Users\kor_a\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest test_vec_curriculum.py::test_select_vectorized_phase_uses_current_phase_when_replay_disabled test_vec_curriculum.py::test_select_vectorized_phase_samples_only_earlier_phases_for_replay test_vec_curriculum.py::test_select_vectorized_phase_never_replays_before_second_phase -q
```

Expected: fails with `ImportError` for `select_vectorized_phase`.

- [ ] **Step 3: Implement minimal helpers**

In `sncp_ppo/train.py`, place these helpers near `step_to_phase`:

```python
def curriculum_phases(final_num_humans):
    return [
        ('easy', 1, 0.13),
        ('easy_plus', 3, 0.18),
        ('medium', 5, 0.22),
        ('hard', 8, 0.24),
        ('circle', final_num_humans, 0.26),
    ]


def phase_index_for_steps(steps_seen, total_steps):
    frac = steps_seen / max(1, total_steps)
    if frac <= 0.10:
        return 0
    if frac <= 0.25:
        return 1
    if frac <= 0.50:
        return 2
    if frac <= 0.75:
        return 3
    return 4


def select_vectorized_phase(steps_seen, total_steps, final_num_humans,
                            replay_ratio=0.0, rng=random):
    phases = curriculum_phases(final_num_humans)
    current_idx = phase_index_for_steps(steps_seen, total_steps)
    if replay_ratio > 0.0 and current_idx > 0 and rng.random() < replay_ratio:
        replay_idx = rng.randint(0, current_idx - 1)
        return phases[replay_idx], True
    return phases[current_idx], False
```

Update `step_to_phase` to delegate:

```python
def step_to_phase(steps_seen, total_steps, final_num_humans):
    """Map an env-step count to a curriculum phase."""
    return curriculum_phases(final_num_humans)[phase_index_for_steps(steps_seen, total_steps)]
```

- [ ] **Step 4: Verify GREEN**

Run the same targeted pytest command. Expected: pass.

---

### Task 2: Wire Replay Into `_train_vectorized`

**Files:**
- Modify: `sncp_ppo/train.py`
- Modify: `test_vec_curriculum.py`

- [ ] **Step 1: Write a failing integration test**

In `test_vec_curriculum.py`, add:

```python
def test_vectorized_replay_updates_are_logged(tmp_path):
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.models import SNCPPolicy
    from sncp_ppo.ppo import PPOAgent

    save_path = tmp_path / 'vc_replay_smoke.pt'
    log_path = tmp_path / 'vc_replay_log.csv'
    args = argparse.Namespace(
        num_envs=2,
        horizon=8,
        total_steps=96,
        eval_freq_updates=0,
        episodes=1,
        num_humans=10,
        seed=42,
        holdout_scenarios=['easy', 'hard'],
        holdout_episodes=1,
        best_warmup_evals=0,
        best_min_success_threshold=0.0,
        save_path=str(save_path),
        lr=1e-4,
        target_kl=0.01,
        curriculum_replay_ratio=1.0,
    )
    device = torch.device('cpu')
    env = CrowdSimEnv(num_humans=args.num_humans, scenario='circle')
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)
    agent = PPOAgent(policy=policy, lr=args.lr, target_kl=args.target_kl,
                     epochs=1, batch_size=2, seq_len=4)

    with log_path.open('w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        _train_vectorized(args, env, policy, agent, device, str(log_path), csv_writer, csv_file)

    with log_path.open(newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert any(row['is_replay_update'] == '1' for row in rows)
    assert any(row['is_replay_update'] == '0' for row in rows)
```

Update the existing `test_vectorized_runs_with_curriculum_holdout_and_saves` args to include:

```python
        curriculum_replay_ratio=0.0,
```

Change its CSV assertions to use `DictReader`:

```python
    with log_path.open(newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))

    num_humans_seen = {int(row['num_humans']) for row in rows}
    assert 1 in num_humans_seen
    assert len(num_humans_seen) > 1
    assert rows[-1]['is_best_checkpoint'] == '1'
    assert rows[-1]['holdout_easy_success'] != 'nan'
    assert rows[-1]['holdout_hard_success'] != 'nan'
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:PYTHONPATH='C:\tmp\codex-pydeps'
& 'C:\Users\kor_a\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest test_vec_curriculum.py::test_vectorized_replay_updates_are_logged -q
```

Expected: fails because `_train_vectorized` does not log `is_replay_update` and does not use replay.

- [ ] **Step 3: Implement vectorized replay**

In `_train_vectorized`, print replay status at startup:

```python
    replay_ratio = getattr(args, 'curriculum_replay_ratio', 0.0)
    print(f"Replay ratio: {replay_ratio:.0%} of vectorized update windows sample earlier phases")
```

Replace direct `step_to_phase(...)` selection in the update loop:

```python
        (next_scenario, next_H, next_vpref), is_replay_update = select_vectorized_phase(
            total_steps,
            args.total_steps,
            args.num_humans,
            replay_ratio=replay_ratio,
            rng=random,
        )
```

Include replay status in the curriculum-shift print:

```python
            replay_mark = " replay" if is_replay_update else ""
            print(f"\n  [Curriculum shift @ step {total_steps}{replay_mark}] "
                  f"{scenario}/{H}h -> {next_scenario}/{next_H}h")
```

Add `is_replay_update` to vectorized CSV rows:

```python
                total_steps, scenario, H, vpref, int(is_replay_update), T,
```

This requires the CSV header in `train()` to add `is_replay_update` immediately before `steps`.

Update the progress log:

```python
            replay_mark = "R" if is_replay_update else " "
            print(f"Update {update_idx} | step {total_steps}/{args.total_steps} "
                  f"[{replay_mark} {scenario} {H}h] | "
                  f"ent={agent.last_entropy:+.3f} kl={agent.last_approx_kl:.5f} "
                  f"std=[{stdv[0]:.3f},{stdv[1]:.3f}] rms={agent.return_rms.std:.2f}")
```

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
$env:PYTHONPATH='C:\tmp\codex-pydeps'
& 'C:\Users\kor_a\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest test_vec_curriculum.py -q
```

Expected: pass.

---

### Task 3: Local Regression and Smoke

**Files:** none unless tests expose a necessary fix.

- [ ] **Step 1: Run full pytest**

```powershell
$env:PYTHONPATH='C:\tmp\codex-pydeps'
& 'C:\Users\kor_a\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run vectorized replay smoke**

```powershell
$env:PYTHONPATH='C:\tmp\codex-pydeps'
& 'C:\Users\kor_a\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m sncp_ppo.train --num_envs 4 --horizon 32 --total_steps 4096 --eval_freq_updates 2 --num_humans 10 --seed 42 --lr 5e-5 --target_kl 0.01 --curriculum_replay_ratio 0.20 --holdout_scenarios easy hard circle --holdout_episodes 2 --save_path checkpoints/_smoke_v16.pt
```

Expected: exits 0, prints a nonzero replay ratio, visits current and replay phases, saves a smoke checkpoint or final checkpoint, and has finite KL/RMS/std values.

- [ ] **Step 3: Remove smoke artifacts**

Use native PowerShell `Remove-Item -LiteralPath` only after verifying paths are inside the workspace:

```powershell
Get-ChildItem -Path checkpoints -Filter '_smoke_v16*.pt' | Select-Object FullName
Get-ChildItem -Path logs -Filter 'training_*.csv' | Sort-Object LastWriteTime -Descending | Select-Object -First 3 FullName,LastWriteTime
```

Then delete only the smoke checkpoint files and, if identifiable, the smoke CSV.

---

### Task 4: Notebook v16 Run Configuration

**Files:**
- Modify: `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Update the training cell**

Targeted replacements:

```python
TOTAL_STEPS = 2_500_000
SAVE_PATH = 'checkpoints/sncp_ppo_v16.pt'
REPLAY_RATIO = 0.20
```

Add the CLI args:

```python
    '--curriculum_replay_ratio', str(REPLAY_RATIO),
```

Update comments to say v16 changes only one training variable relative to v15: vectorized anti-forgetting replay.

- [ ] **Step 2: Update the eval cell**

Set:

```python
CHECKPOINT = 'checkpoints/sncp_ppo_v16.pt'
```

Keep the density sweep N=1/3/5/8/10 and the nav-time/I_sp criteria. Do not switch to N=15/20 until replay sustains N=10.

- [ ] **Step 3: Validate notebook JSON**

```powershell
$env:PYTHONPATH='C:\tmp\codex-pydeps'
& 'C:\Users\kor_a\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "import json, pathlib; json.loads(pathlib.Path('sncp_ppo_colab.ipynb').read_text(encoding='utf-8')); print('valid')"
```

Expected: prints `valid`.

---

### Task 5: Commit and Push Code for Colab

**Files:**
- Stage modified tests, `sncp_ppo/train.py`, `sncp_ppo_colab.ipynb`, and this plan.

- [ ] **Step 1: Inspect status**

```powershell
git -c safe.directory=C:/Users/kor_a/Desktop/deneme status --short
```

- [ ] **Step 2: Commit**

Commit message:

```text
v16: add vectorized curriculum replay

Add replay-window selection to the vectorized PPO curriculum so a configured
fraction of updates samples earlier density phases instead of training
monotonically into N=10. This is the only v16 training variable; reward,
environment defaults, and NCP capacity stay at v15 values.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

- [ ] **Step 3: Push**

```powershell
git -c safe.directory=C:/Users/kor_a/Desktop/deneme push origin main
```

Expected: `main` pushed so Colab can pull.

---

### Task 6: Post-Colab Evaluation and Documentation

**Files:**
- Modify after results: `AGENTS.md`
- Modify after results: `~/.claude/projects/C--Users-kor-a-Desktop-deneme/memory/sncp-paper-vs-impl.md`

- [ ] **Step 1: Evaluate v16 checkpoint**

Run the v16 notebook eval cell or local equivalents:

```powershell
$env:PYTHONPATH='C:\tmp\codex-pydeps'
& 'C:\Users\kor_a\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' evaluate_policy_report.py --checkpoint checkpoints/sncp_ppo_v16.pt --output_dir eval_v16 --densities 1 3 5 8 10 --scenario hard --n_episodes 50 --seed 100 --trajectory_densities 5 10
```

This writes `eval_v16/density_sweep.csv`, `eval_v16/density_sweep.json`,
`eval_v16/density_sweep.png`, `eval_v16/report.md`, and N=5/N=10 trajectory PNGs.

- [ ] **Step 2: Inspect trajectory plots**

Open the generated N=5/N=10 trajectory PNGs and verify the path routes around clusters rather than
through them. Generate GIFs only if the static plots are ambiguous.

- [ ] **Step 3: Compare against v15 gates**

Pass criteria:

- No beeline regression: successful nav-time remains well above v14's 121.5-step straight-line baseline.
- Replay fixes collapse: final/best checkpoint holdout min-success should not crash to 0 in the N=10 phase.
- Density sweep improves or at least preserves v15 genuine-avoidance behavior: N=5 remains near or above 66%, N=10 improves above 46%, I_sp stays low, collisions do not rise for the same density.
- If sparse N timeout remains high but collision/I_sp improve, next single-variable run is `max_time 50 -> 60` or comfort `-6 -> -5`, not capacity.

- [ ] **Step 4: Update AGENTS and memory**

Append v16 results, including:

- replay ratio used;
- best-checkpoint timing and whether final checkpoint collapsed;
- density sweep table;
- nav-time/I_sp vs v15;
- trajectory verdict;
- next single-variable candidate.

---

## Review Gate

Do not implement Tasks 1-5 until this plan is reviewed. The proposed v16 run changes exactly one training variable relative to v15: vectorized curriculum replay with `--curriculum_replay_ratio 0.20`.
