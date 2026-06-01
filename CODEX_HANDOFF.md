# CODEX HANDOFF — v11: Curriculum + Holdout in the Vectorized Training Path

**Date:** 2026-05-31
**Repo:** `C:\Users\kor_a\Desktop\deneme` (GitHub: heimdilon/sncp-ppo-crowdnav)
**You are continuing work started by another agent.** This document is self-contained — you do NOT need prior conversation history. Read it fully before touching code.

---

## 0. TL;DR — what you are building

The repo has an RL training script with TWO paths in `sncp_ppo/train.py`:
- **single-env path** (`--num_envs 1`, default): full curriculum (1→5 pedestrians) + holdout evaluation + best-checkpoint selection. WORKS. Do not change its behavior.
- **vectorized path** (`--num_envs > 1`, function `_train_vectorized`): fast parallel rollout, but currently trains on a FIXED `'circle'` scenario with NO curriculum and NO holdout/best-checkpoint. It only saves a `_final` checkpoint.

**Your job:** add curriculum + holdout + best-checkpoint to the vectorized path, matching the single-env semantics. The full design is approved and written in:
- **Spec:** `docs/superpowers/specs/2026-05-31-vec-curriculum-holdout-design.md` (READ THIS FIRST — it is the source of truth)
- This handoff = the spec + exact code anchors + step-by-step TDD plan + verification.

---

## 1. Environment & ground rules

- **OS:** Windows. Python is `python` (Python 3.11). GPU: RTX 3060 Ti, CUDA available.
- **Shell quoting:** This is Windows. If using PowerShell, `&&` does NOT chain; use `;` + `if ($?)`. Prefer running ONE command at a time and writing results to a file you then read, rather than long pipelines (the terminal in this project has shown intermittent output-garbling — verify via files + `git`/`ast`/`pytest`, never trust a single noisy echo).
- **Do NOT run full training.** Only smoke tests (tiny `--total_steps`).
- **Branch:** create `feat/vec-curriculum` off `main`. Current `main` HEAD should be the v11 spec commit (`467ab0d` "docs: spec for curriculum + holdout in the vectorized path (v11)"). Verify with `git log --oneline -3`.
- **Commit discipline:** TDD — failing test first, then minimal code, then commit. Conventional commit messages. End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Do NOT push or open a PR.** Leave commits on the local branch; the human will review with the original agent first.

---

## 2. DO NOT TOUCH (regression-critical)

These are reused, not modified. Changing them breaks proven, tested code:
- `sncp_ppo/ppo.py` — `update`, `update_vectorized`, `compute_gae_*` all stay as-is.
- `sncp_ppo/vec_buffer.py` — `compute_gae_vectorized`, `VectorizedRolloutBuffer`, `reset_hidden_where_done` stay as-is.
- `sncp_ppo/models.py`, `crowd_sim/crowd_env.py` — untouched.
- The **single-env path** in `train.py` (`def train` body up to the `if args.num_envs > 1:` branch, and everything the for-loop `for episode in range(1, args.episodes + 1):` runs) must stay byte-identical in behavior.
- `evaluate_holdout` (train.py:80) — REUSE it, do not modify it.

After your work, this MUST hold:
```
git diff main..HEAD -- sncp_ppo/ppo.py sncp_ppo/vec_buffer.py sncp_ppo/models.py crowd_sim/crowd_env.py
```
→ **empty output** (zero changes to those files).

---

## 3. Locked design decisions (do not re-litigate)

1. **Recreate envs at phase boundaries.** All N parallel envs must share `num_humans` (policy batches them; spatial tensor is `(N, H, 4)` — mixed H cannot batch). At each curriculum phase change, `envs.close()` and rebuild `SyncVectorEnv` with the new `num_humans`, reset, and re-init the LTC hidden. Recreation happens BETWEEN PPO updates, never mid-rollout.
2. **Reuse `evaluate_holdout`** on a throwaway single-env, periodically. No new eval code path.
3. **Total env-step budget** (`--total_steps`, default 2_000_000) drives BOTH curriculum phase boundaries AND run length. Phases use the same 10/25/50/75% fractions as single-env.

Curriculum schedule (IDENTICAL to single-env `curriculum` list at train.py:225):
| phase | scenario | num_humans | vpref | boundary (fraction of total_steps) |
|---|---|---|---|---|
| 1 | easy | 1 | 0.15 | ≤ 10% |
| 2 | easy_plus | 2 | 0.20 | ≤ 25% |
| 3 | medium | 3 | 0.30 | ≤ 50% |
| 4 | hard | 4 | 0.40 | ≤ 75% |
| 5 | circle | args.num_humans (5) | 0.50 | ≤ 100% |

CSV note (approved): in vectorized mode the existing `episode` CSV column holds `total_steps_seen` (a step count, not an episode index). This is a deliberate reinterpretation — keep the same column names as single-env so `plot_training.py` still works.

---

## 4. Relevant existing code (anchors you will use)

### 4a. `evaluate_holdout` signature (train.py:80) — REUSE
```python
def evaluate_holdout(env, policy, agent, device, n_episodes, scenario, base_seed):
    # returns dict: {'success_rate', 'collision_rate', 'timeout_rate', 'avg_reward'}
```
It internally sets the env's scenario/num_humans/vpref to the canonical values for `scenario`, runs `n_episodes` deterministic rollouts, restores env config, and returns the dict. It works on a single env instance.

### 4b. Best-checkpoint logic (single-env, train.py:368-399) — REPLICATE in vec loop
```python
            min_success = min(r['success_rate'] for r in last_holdout_per_scenario.values())
            avg_reward = float(np.mean([r['avg_reward'] for r in last_holdout_per_scenario.values()]))
            avg_collision = float(np.mean([r['collision_rate'] for r in last_holdout_per_scenario.values()]))
            current_score = (min_success, avg_reward, -avg_collision)

            if holdout_eval_count <= args.best_warmup_evals:
                best_reason = (f"best skipped due to warmup (eval {holdout_eval_count}/{args.best_warmup_evals})")
            elif min_success < args.best_min_success_threshold:
                best_reason = (f"best skipped due to threshold (min_success={min_success:.1%} < {args.best_min_success_threshold:.1%})")
            elif current_score > best_holdout_score:
                best_holdout_min_success = min_success
                best_holdout_score = current_score
                torch.save(policy.state_dict(), args.save_path)
                is_best_checkpoint = 1
            else:
                best_reason = 'best not updated: score did not improve'
```
Initial values (single-env uses these — copy them):
```python
    best_holdout_min_success = -1.0
    best_holdout_score = (-1.0, -float('inf'), -float('inf'))  # (min_success, avg_reward, -collision_rate)
    holdout_eval_count = 0
    last_holdout_per_scenario = {sc: {'success_rate': float('nan'), 'collision_rate': float('nan'),
                                      'timeout_rate': float('nan'), 'avg_reward': float('nan')}
                                 for sc in args.holdout_scenarios}
```

### 4c. Current `_train_vectorized` (train.py:485-558) — the function you EXTEND
It already does: build SyncVectorEnv on fixed 'circle', rollout T steps into a `VectorizedRolloutBuffer`, `buf.finish`, `agent.update_vectorized`, periodic diagnostics print, final save. You will wrap its loop with curriculum-phase logic + periodic holdout. Key existing helpers in scope: `make_env(num_humans, scenario, seed)` (train.py:141), `reset_hidden_where_done`, `VectorizedRolloutBuffer`.

### 4d. argparse anchors (train.py)
- `args = parser.parse_args()` at line 646 — add new args BEFORE it.
- Existing relevant args already present: `--num_envs`, `--horizon`, `--num_humans`, `--seed`, `--holdout_scenarios` (nargs+, default `['easy','hard']`), `--holdout_episodes`, `--best_warmup_evals`, `--best_min_success_threshold`, `--save_path`, `--lr`, `--target_kl`.

---

## 5. Implementation plan (TDD, bite-sized)

Create test file `test_vec_curriculum.py`. Run tests with `python -m pytest <file> -v`.

### Task A — pure helper `step_to_phase` (unit-testable, no env/GPU)

**A1. Failing test** — append to `test_vec_curriculum.py`:
```python
from sncp_ppo.train import step_to_phase

def test_step_to_phase_boundaries():
    total = 1000
    # fractions: easy<=10%, easy_plus<=25%, medium<=50%, hard<=75%, circle<=100%
    assert step_to_phase(0, total, 5) == ('easy', 1, 0.15)
    assert step_to_phase(99, total, 5) == ('easy', 1, 0.15)
    assert step_to_phase(100, total, 5) == ('easy', 1, 0.15)        # 10% inclusive
    assert step_to_phase(101, total, 5) == ('easy_plus', 2, 0.20)
    assert step_to_phase(250, total, 5) == ('easy_plus', 2, 0.20)   # 25% inclusive
    assert step_to_phase(251, total, 5) == ('medium', 3, 0.30)
    assert step_to_phase(500, total, 5) == ('medium', 3, 0.30)
    assert step_to_phase(501, total, 5) == ('hard', 4, 0.40)
    assert step_to_phase(750, total, 5) == ('hard', 4, 0.40)
    assert step_to_phase(751, total, 5) == ('circle', 5, 0.50)
    assert step_to_phase(1000, total, 5) == ('circle', 5, 0.50)
    assert step_to_phase(99999, total, 5) == ('circle', 5, 0.50)    # clamp past end
```

**A2. Run, confirm it fails** (`ImportError: cannot import name 'step_to_phase'`).

**A3. Implement** — add this MODULE-LEVEL function in `train.py` (place it right BEFORE `def _train_vectorized`):
```python
def step_to_phase(steps_seen, total_steps, final_num_humans):
    """Map an env-step count to a curriculum phase (scenario, num_humans, vpref).

    Boundaries are inclusive fractions of total_steps: 10/25/50/75%, matching
    the single-env curriculum. The final 'circle' phase uses final_num_humans.
    """
    frac = steps_seen / max(1, total_steps)
    if frac <= 0.10:
        return ('easy', 1, 0.15)
    if frac <= 0.25:
        return ('easy_plus', 2, 0.20)
    if frac <= 0.50:
        return ('medium', 3, 0.30)
    if frac <= 0.75:
        return ('hard', 4, 0.40)
    return ('circle', final_num_humans, 0.50)
```

**A4. Run, confirm PASS. Commit:**
`feat(vec): step_to_phase curriculum helper (step-budgeted, matches single-env)`

### Task B — new CLI args

**B1.** Add BEFORE `args = parser.parse_args()` (train.py:646):
```python
    parser.add_argument('--total_steps', type=int, default=2_000_000,
                        help='Env-step budget (vectorized mode): drives curriculum '
                             'phase boundaries (10/25/50/75%%) and total run length.')
    parser.add_argument('--eval_freq_updates', type=int, default=20,
                        help='Holdout evaluation cadence in PPO updates (vectorized mode).')
```
**B2.** Verify `python -c "import sncp_ppo.train"` imports OK and `python -m sncp_ppo.train --help` lists both args. Commit:
`feat(vec): --total_steps and --eval_freq_updates CLI args`

### Task C — rewrite `_train_vectorized` loop with curriculum + holdout

This is the integration task. Replace the body of `_train_vectorized` (train.py:485-558) with the version below. It keeps the rollout/update core identical and adds: (1) phase tracking via `step_to_phase`, (2) env recreation on phase change, (3) periodic holdout + best-checkpoint, (4) CSV rows.

**C1. Failing test (smoke-integration)** — append to `test_vec_curriculum.py`:
```python
import os, glob, torch
from sncp_ppo.train import _train_vectorized  # ensure importable

def test_vectorized_runs_with_curriculum_and_saves(tmp_path):
    # End-to-end tiny smoke: a short step budget that crosses >=1 phase boundary,
    # one holdout eval, and a saved checkpoint or final file.
    import argparse
    from sncp_ppo.models import SNCPPolicy
    from sncp_ppo.ppo import PPOAgent
    from crowd_sim.crowd_env import CrowdSimEnv
    import csv as _csv
    save = str(tmp_path / 'vc_smoke.pt')
    log = str(tmp_path / 'vc_log.csv')
    args = argparse.Namespace(
        num_envs=4, horizon=32, total_steps=2000, eval_freq_updates=2,
        num_humans=5, seed=42, holdout_scenarios=['easy', 'hard'],
        holdout_episodes=2, best_warmup_evals=0, best_min_success_threshold=0.0,
        save_path=save, lr=1e-4, target_kl=0.01,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = CrowdSimEnv(num_humans=args.num_humans, scenario='circle')
    policy = SNCPPolicy(robot_vpref=env.robot_vpref, robot_wmax=env.robot_wmax).to(device)
    agent = PPOAgent(policy=policy, lr=args.lr, target_kl=args.target_kl)
    f = open(log, 'w', newline='')
    w = _csv.writer(f)
    _train_vectorized(args, env, policy, agent, device, log, w, f)
    # a checkpoint (best or final) must exist
    assert os.path.exists(save) or os.path.exists(save.replace('.pt', '_final.pt'))
```
NOTE: the exact `args` fields you require depend on your final `_train_vectorized` signature/usage. If you reference an arg not in this Namespace, ADD it to the test's Namespace. The intent: prove the vec loop runs end-to-end, crosses a phase boundary, evaluates holdout, and saves.

**C2. Run, confirm it fails** (current `_train_vectorized` ignores curriculum/holdout — the test may still pass on "final exists"; if so, ALSO assert a phase-shift happened by capturing stdout or checking the CSV has >1 distinct `num_humans`. Make the test meaningfully fail before implementing.)

**C3. Implement** — here is the full replacement body for `_train_vectorized`:
```python
def _train_vectorized(args, env, policy, agent, device, log_path, csv_writer, csv_file):
    """Vectorized fixed-horizon rollout (N envs x T steps per PPO update) WITH
    step-budgeted curriculum + periodic holdout + best-checkpoint selection.

    All N envs share num_humans (the policy batches them), so at each curriculum
    phase boundary we close and rebuild the SyncVectorEnv with the new
    num_humans and re-init the LTC hidden. Recreation happens between updates,
    never mid-rollout. Holdout reuses the single-env evaluate_holdout on a
    throwaway env, with the same warmup/threshold/tie-break best-checkpoint rule.
    """
    import gymnasium as gym
    from sncp_ppo.vec_buffer import VectorizedRolloutBuffer, reset_hidden_where_done

    N, T = args.num_envs, args.horizon

    def build_envs(n_humans, scenario):
        e = gym.vector.SyncVectorEnv(
            [make_env(n_humans, scenario, args.seed + i) for i in range(N)]
        )
        o, _ = e.reset(seed=args.seed)
        return e, o

    def to_tensor(o):
        return {
            'robot_node': torch.tensor(o['robot_node'], dtype=torch.float32, device=device),
            'spatial_edges': torch.tensor(o['spatial_edges'], dtype=torch.float32, device=device),
            'temporal_edges': torch.tensor(o['temporal_edges'], dtype=torch.float32, device=device),
        }

    # --- holdout / best-checkpoint state (mirrors single-env) ---
    nan = float('nan')
    last_holdout = {sc: {'success_rate': nan, 'collision_rate': nan,
                         'timeout_rate': nan, 'avg_reward': nan}
                    for sc in args.holdout_scenarios}
    best_min_success = -1.0
    best_score = (-1.0, -float('inf'), -float('inf'))
    holdout_eval_count = 0
    eval_env = CrowdSimEnv(num_humans=args.num_humans, scenario='circle')

    # --- curriculum init ---
    scenario, H, vpref = step_to_phase(0, args.total_steps, args.num_humans)
    envs, obs_np = build_envs(H, scenario)
    for e in envs.envs:
        e.unwrapped.human_vpref = vpref
    h = policy.init_hidden(batch_size=N, num_humans=H, device=device)

    total_steps = 0
    update_idx = 0
    print(f"\nVectorized training: {N} envs x {T} steps = {N*T} transitions/update")
    print(f"Curriculum by step budget: total={args.total_steps}, phases 10/25/50/75%")
    print(f"Holdout every {args.eval_freq_updates} updates on {args.holdout_scenarios}")
    print("-" * 90)

    while total_steps < args.total_steps:
        # phase transition check (between updates, never mid-rollout)
        new_scenario, new_H, new_vpref = step_to_phase(total_steps, args.total_steps, args.num_humans)
        if new_H != H or new_scenario != scenario:
            print(f"\n  [Curriculum shift @ step {total_steps}] "
                  f"{scenario}/{H}h -> {new_scenario}/{new_H}h")
            envs.close()
            scenario, H, vpref = new_scenario, new_H, new_vpref
            envs, obs_np = build_envs(H, scenario)
            for e in envs.envs:
                e.unwrapped.human_vpref = vpref
            h = policy.init_hidden(batch_size=N, num_humans=H, device=device)

        # rollout T steps
        buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
        for t in range(T):
            obs_t = to_tensor(obs_np)
            with torch.no_grad():
                mu, std, value, h_next = policy(obs_t, h)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            act_np = action.cpu().numpy()
            act_np[:, 0] = np.clip(act_np[:, 0], 0.0, env.robot_vpref)
            act_np[:, 1] = np.clip(act_np[:, 1], -env.robot_wmax, env.robot_wmax)
            next_obs, reward, term, trunc, info = envs.step(act_np)
            done = np.logical_or(term, trunc)
            done_t = torch.tensor(done, dtype=torch.float32, device=device)
            reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
            mask_t = torch.tensor(1.0 - term.astype('float32'), device=device)
            buf.store(obs=obs_t, hidden=h, actions=action, log_probs=log_prob,
                      rewards=reward_t, values=value.squeeze(-1),
                      dones=done_t, masks=mask_t)
            obs_np = next_obs
            h = reset_hidden_where_done(h_next, done_t, H)
            total_steps += N

        with torch.no_grad():
            last_v = policy(to_tensor(obs_np), h)[2].squeeze(-1)
        buf.finish(last_values=last_v, last_dones=torch.zeros(N, device=device))
        agent.update_vectorized(buf, device)
        update_idx += 1

        # periodic holdout + best-checkpoint
        is_best = 0
        if update_idx % args.eval_freq_updates == 0:
            for sc in args.holdout_scenarios:
                last_holdout[sc] = evaluate_holdout(
                    eval_env, policy, agent, device,
                    n_episodes=args.holdout_episodes, scenario=sc,
                    base_seed=args.seed + 10_000 + total_steps,
                )
            holdout_eval_count += 1
            min_success = min(r['success_rate'] for r in last_holdout.values())
            avg_reward = float(np.mean([r['avg_reward'] for r in last_holdout.values()]))
            avg_collision = float(np.mean([r['collision_rate'] for r in last_holdout.values()]))
            score = (min_success, avg_reward, -avg_collision)
            if holdout_eval_count <= args.best_warmup_evals:
                print(f"  --> warmup eval {holdout_eval_count}/{args.best_warmup_evals}: min={min_success:.1%}")
            elif min_success < args.best_min_success_threshold:
                print(f"  --> best skipped (min_success={min_success:.1%} < {args.best_min_success_threshold:.1%})")
            elif score > best_score:
                best_min_success = min_success
                best_score = score
                torch.save(policy.state_dict(), args.save_path)
                is_best = 1
                per_sc = {sc: f"{r['success_rate']:.0%}" for sc, r in last_holdout.items()}
                print(f"  --> New best generalist min={min_success:.1%}, "
                      f"avg_reward={avg_reward:.3f}, collision={avg_collision:.1%} {per_sc}, saved {args.save_path}")
            else:
                print(f"  --> best not updated (min_success={min_success:.1%})")

        # CSV row (same columns as single-env; 'episode' holds total_steps here)
        ho_row = []
        for sc in args.holdout_scenarios:
            r = last_holdout[sc]
            ho_row += [f"{r['success_rate']:.4f}", f"{r['collision_rate']:.4f}",
                       f"{r['timeout_rate']:.4f}", f"{r['avg_reward']:.4f}"]
        try:
            csv_writer.writerow([
                total_steps, scenario, H, vpref, T, '',
                '', '', '', '', is_best, '',
            ] + ho_row)
            csv_file.flush()
        except OSError:
            pass

        if update_idx % 10 == 0:
            with torch.no_grad():
                stdv = policy.actor_logstd.exp().squeeze().cpu().numpy()
            print(f"Update {update_idx} | step {total_steps}/{args.total_steps} [{scenario} {H}h] | "
                  f"ent={agent.last_entropy:+.3f} kl={agent.last_approx_kl:.5f} "
                  f"std=[{stdv[0]:.3f},{stdv[1]:.3f}] rms={agent.return_rms.std:.2f}")

    envs.close()
    torch.save(policy.state_dict(), args.save_path.replace('.pt', '_final.pt'))
    csv_file.close()
    print(f"\nVectorized training completed! Best generalist min(success)={best_min_success:.1%}")
```

IMPORTANT CAVEATS to verify while implementing:
- `envs.envs` and `e.unwrapped.human_vpref`: confirm `gymnasium.vector.SyncVectorEnv` exposes sub-envs as `.envs` and that `make_env` returns a `CrowdSimEnv` whose `human_vpref` attribute is settable. If `make_env` wraps the env, adjust the vpref-set loop accordingly (read `make_env` at train.py:141 and `crowd_env.py`). If setting per-sub-env vpref is awkward, an acceptable alternative is to pass vpref into `make_env`/scenario so each sub-env starts with it. Pick whichever actually works and note it.
- The CSV header is written by the CALLER (`train`) before `_train_vectorized` is invoked in the real CLI flow. In the standalone test you open your own csv writer; that's fine. In the real `train()` path, ensure the vec branch still gets a valid `csv_writer`/`csv_file` (it currently does — see the `if args.num_envs > 1:` branch in `train`).

**C4. Run the smoke test, confirm PASS.**

**C5. Commit:**
`feat(vec): curriculum + holdout + best-checkpoint in vectorized training`

### Task D — full verification (see section 6). Commit any test additions.

---

## 6. Verification checklist (MUST all pass before reporting done)

Run each, one at a time, write output to a file if the terminal garbles:

1. **New + existing unit tests:**
   `python -m pytest test_vec_curriculum.py test_vec_buffer.py test_vec_gae.py -v`
   → all pass.

2. **Full regression (the 29 pre-existing tests):**
   `python -m pytest test_train_eta.py test_env_randomization.py test_env_velocity.py -q`
   → all pass.

3. **Vectorized smoke with curriculum (crosses a boundary, evaluates holdout):**
   `python -m sncp_ppo.train --num_envs 4 --horizon 64 --total_steps 12000 --eval_freq_updates 3 --num_humans 5 --holdout_episodes 2 --best_warmup_evals 0 --save_path checkpoints/_vc_smoke.pt`
   → prints ≥1 "Curriculum shift", ≥1 holdout line, "Vectorized training completed!", exits 0, creates a checkpoint. Then delete `checkpoints/_vc_smoke*.pt` and any stray `logs/training_*.csv` from the smoke.

4. **Single-env path unchanged:**
   `python -m sncp_ppo.train --num_envs 1 --episodes 6 --eval_freq 100 --holdout_episodes 2 --save_path checkpoints/_legacy_smoke.pt`
   → single-env "Ep N/6 ..." format, NO "Vectorized" text, exits 0. Delete `checkpoints/_legacy_smoke*` after.

5. **Protected files untouched:**
   `git diff main..HEAD -- sncp_ppo/ppo.py sncp_ppo/vec_buffer.py sncp_ppo/models.py crowd_sim/crowd_env.py`
   → EMPTY.

6. **Clean tree:** `git status --porcelain` → only intended files; no stray temp files (`_*.txt`, `_*.py` scratch).

---

## 7. What to leave for the human + original agent

- Do NOT push, do NOT open/modify a PR. Commits stay local on `feat/vec-curriculum`.
- Write a short `CODEX_RESULT.md` at repo root summarizing: commits made (SHAs + subjects), every verification command's result (pass/fail + key output lines), any caveat you hit (especially the `envs.envs`/vpref-setting detail in Task C), and anything you were unsure about. The human will review this with the original agent.
- If you get genuinely stuck (e.g. SyncVectorEnv sub-env vpref cannot be set cleanly), STOP and document the blocker in `CODEX_RESULT.md` rather than guessing — note exactly what you tried.

---

## 8. Context: why this matters (optional reading)

Three independent critique agents found the project's hard-scenario success (~30-40%) is capped by DATA STARVATION: the single-env path collects only ~500 transitions per PPO update vs the standard 2048+. The vectorized path (v10, already merged) fixes throughput (N×T = 2048+). But it lacked curriculum/holdout, so it couldn't yet replace the single-env path for a real run. THIS task closes that gap. After it lands, the human will do a long run on a Colab Pro+ A100 with roughly:
`--num_envs 16 --horizon 128 --total_steps 2000000 --eval_freq_updates 20 --holdout_episodes 50 --holdout_scenarios easy hard --lr 5e-5 --target_kl 0.01 --save_path checkpoints/sncp_ppo_v11.pt`

End of handoff.
