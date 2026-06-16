# v27 pre-MLP (Eq 11) Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure a single-variable v27 run that is byte-identical to v26 except it adds the paper's Eq 11 pre-MLP edge embedding (`--pre_mlp`), so Colab can train it and we can compare it to the v26 honest baseline.

**Architecture:** No model/env/reward code changes — `pre_mlp` is already implemented in `models.py` and wired through `train.py --pre_mlp`, and `build_policy_for_checkpoint` auto-detects it from checkpoint keys. The work is a v26→v27 harness version bump (notebook + run-readiness checker + version-marker tests) plus the one `--pre_mlp` flag in the training cell.

**Tech Stack:** Python, pytest, Jupyter notebook (`sncp_ppo_colab.ipynb`), PyTorch (ncps LTC).

---

## File structure

- `test_post_run_pipeline.py` — notebook-faithfulness test (rename v26→v27, assert `--pre_mlp`).
- `test_v16_run_readiness.py` — readiness tests (rename v26→v27 markers, persist `eval_v27`).
- `sncp_ppo/run_readiness.py` — preflight token lists + checker (v26→v27, add `--pre_mlp` token, fix stale 10 m comment).
- `sncp_ppo_colab.ipynb` — training cell (`--pre_mlp` + v27 save path), eval/diagnostics/persist cells (v27 paths).

No new files. No changes to `models.py`, `train.py`, `crowd_env.py`, `eval_report.py`.

---

### Task 1: Update version-marker tests to v27 (TDD red)

**Files:**
- Modify: `test_post_run_pipeline.py:266-282`
- Modify: `test_v16_run_readiness.py:12-18,21-49,68-82`

- [ ] **Step 1: Replace the notebook-faithfulness test**

In `test_post_run_pipeline.py`, replace the whole `test_notebook_is_v26_paper_faithful` function (lines 266-282) with:

```python
def test_notebook_is_v27_pre_mlp_ablation():
    # v27 = v26 + pre-MLP (Eq 11) ONLY. The per-scenario paper budget + 8m crossing +
    # normalized comfort stay env-DERIVED from --fixed_scenario (no CLI budget), so the
    # single change vs v26 is `--pre_mlp` plus the v27 save path.
    code = _colab_code_sources()
    train_cells = [s for s in code if "sncp_ppo.train" in s and "--fixed_scenario" in s]
    eval_cells = [s for s in code if "run_post_eval.py" in s]
    assert len(train_cells) == 1 and len(eval_cells) == 1
    train, ev = train_cells[0], eval_cells[0]
    # Training: paper scenario + v27 save path + the defining --pre_mlp flag.
    assert "paper_challenging" in train
    assert "checkpoints/sncp_ppo_v27.pt" in train
    assert "'--pre_mlp'" in train
    # Single-variable guard: every v26 invariant still present (regime unchanged).
    for tok in ("TOTAL_STEPS = 2_500_000", "SEED = 42", "'--robot_vpref', '1.0'",
                "'--num_humans', '10'", "'--holdout_episodes', '50'"):
        assert tok in train, tok
    # Eval: v27, paper baseline beeline 32 (8 m crossing at 1.0 m/s), no forced budget.
    assert "'--version', '27'" in ev
    assert "'--baseline_nav_steps', '32'" in ev
    assert "'--max_time'" not in ev
```

- [ ] **Step 2: Rename the readiness tests v26→v27**

In `test_v16_run_readiness.py`, make these edits:
- `def test_v26_run_readiness_passes_current_repo():` → `def test_v27_run_readiness_passes_current_repo():` (body unchanged).
- `def test_v26_run_readiness_flags_stale_notebook(tmp_path):` → `def test_v27_run_readiness_flags_stale_notebook(tmp_path):`, and inside it change the three asserts to v27:

```python
    assert summary.status == "fail"
    assert any("v27 training" in note for note in summary.notes)
    assert any("v27 evaluation" in note for note in summary.notes)
    assert any("baseline densities" in note for note in summary.notes)
```

- `def test_colab_persist_cell_downloads_eval_v26_artifact_bundle():` → `def test_colab_persist_cell_downloads_eval_v27_artifact_bundle():`, and change the two asserts:

```python
    assert "'eval_v27_artifacts'" in persist_cell
    assert "'eval_v27'" in persist_cell
```

- [ ] **Step 3: Run the tests to verify they fail (red)**

Run: `python -m pytest test_post_run_pipeline.py::test_notebook_is_v27_pre_mlp_ablation test_v16_run_readiness.py -v`
Expected: FAIL — the notebook still says v26 / `sncp_ppo_v26.pt` and has no `--pre_mlp`; the readiness checker still emits "v26 training"/"v26 evaluation" notes and the persist cell still says `eval_v26`.

- [ ] **Step 4: Commit the red tests**

```bash
git add test_post_run_pipeline.py test_v16_run_readiness.py
git commit -m "v27 tests: bump version markers to v27 + assert --pre_mlp (red)"
```

---

### Task 2: Update the run-readiness checker to v27 + `--pre_mlp`

**Files:**
- Modify: `sncp_ppo/run_readiness.py:11-56,134-149`

- [ ] **Step 1: Update the module comment (lines 11-14)**

Replace:

```python
# v26 = paper-faithful per-scenario budget (challenging 50s, standard 12.5s), 8m crossing,
# and normalized comfort (Eq 7). All DERIVED by the env from --fixed_scenario
# paper_challenging, so the training cell must NOT pass any budget on the CLI (the v24/v26
# failure was a wrong/forgotten CLI budget). Trains from scratch (no IL warm-start).
```

with:

```python
# v27 = v26 + paper Eq 11 pre-MLP edge embedding (--pre_mlp), single-variable ablation.
# Budget (challenging 50s, standard 12.5s), 8m crossing and normalized comfort (Eq 7)
# stay env-DERIVED from --fixed_scenario paper_challenging, so the training cell must NOT
# pass any budget on the CLI. The only change vs v26 is --pre_mlp + the v27 save path.
```

- [ ] **Step 2: Add the `--pre_mlp` token and bump the save path in TRAINING_TOKENS**

In the `TRAINING_TOKENS` tuple, change `"SAVE_PATH = 'checkpoints/sncp_ppo_v26.pt'",` to `"SAVE_PATH = 'checkpoints/sncp_ppo_v27.pt'",`, and add a new token line immediately before `"'--save_path', SAVE_PATH",`:

```python
    "'--pre_mlp'",
    "'--save_path', SAVE_PATH",
```

- [ ] **Step 3: Bump EVALUATION_TOKENS to v27 and fix the stale 10 m comment**

In the `EVALUATION_TOKENS` tuple:
- `"CHECKPOINT = 'checkpoints/sncp_ppo_v26.pt'",` → `"CHECKPOINT = 'checkpoints/sncp_ppo_v27.pt'",`
- `"EVAL_OUT = 'eval_v26'",` → `"EVAL_OUT = 'eval_v27'",`
- `"'--version', '26'",` → `"'--version', '27'",`

Then replace the stale comment block (currently mentioning a "10 m crossing ... (40 steps)"):

```python
    # No --max_time: the env resolves the paper scenario to 12.5s. The comparison vs the
    # antipodal v22 sweep is regime-invalid (the eval cell is resilient to its verdict);
    # the beeline gate is scaled to the 10 m crossing at 1.0 m/s (40 steps).
```

with:

```python
    # No --max_time: the env resolves the paper budget itself. The comparison vs the
    # antipodal v22 sweep is regime-invalid (the eval cell is resilient to its verdict);
    # the beeline gate is scaled to the 8 m crossing at 1.0 m/s (32 steps).
```

- [ ] **Step 4: Bump the checker markers and PASS message (lines ~134-149)**

In `verify_v16_run_ready`:
- `"SAVE_PATH = 'checkpoints/sncp_ppo_v26.pt'",` → `"SAVE_PATH = 'checkpoints/sncp_ppo_v27.pt'",` and its name arg `"v26 training",` → `"v27 training",`
- `"CHECKPOINT = 'checkpoints/sncp_ppo_v26.pt'",` → `"CHECKPOINT = 'checkpoints/sncp_ppo_v27.pt'",` and its name arg `"v26 evaluation",` → `"v27 evaluation",`
- `_check_tokens(training_cell, TRAINING_TOKENS, notes, "v26 training")` → `"v27 training"`
- `_check_tokens(evaluation_cell, EVALUATION_TOKENS, notes, "v26 evaluation")` → `"v27 evaluation"`
- `notes.append("PASS: v26 Colab training and evaluation configuration is ready")` → `"PASS: v27 Colab training and evaluation configuration is ready"`

- [ ] **Step 5: Run readiness tests — still red until the notebook is updated**

Run: `python -m pytest test_v16_run_readiness.py -v`
Expected: `test_v27_run_readiness_flags_stale_notebook` now PASSES (checker emits "v27 training"/"v27 evaluation"); `test_v27_run_readiness_passes_current_repo` still FAILS (notebook is still v26). This is expected — Task 3 fixes the notebook.

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo/run_readiness.py
git commit -m "v27 readiness: v27 markers + --pre_mlp token + fix stale 8m comment"
```

---

### Task 3: Update the notebook to v27 + `--pre_mlp`

**Files:**
- Modify: `sncp_ppo_colab.ipynb` (training, eval, diagnostics, persist, and the Drive markdown cell)

All edits are literal substring replacements on the raw `.ipynb` JSON (single quotes are not JSON-escaped, so the substrings match exactly).

- [ ] **Step 1: Bump all checkpoint paths v26→v27 (replace_all)**

Replace every occurrence of `sncp_ppo_v26.pt` with `sncp_ppo_v27.pt` (2 occurrences: training `SAVE_PATH`, eval `CHECKPOINT`).

- [ ] **Step 2: Bump all eval output dirs v26→v27 (replace_all)**

Replace every occurrence of `eval_v26` with `eval_v27` (eval `EVAL_OUT`, diagnostics-cell paths, persist-cell dir + archive name, and the Drive markdown cell text).

- [ ] **Step 3: Bump the eval version flag**

Replace `'--version', '26'` with `'--version', '27'` (1 occurrence, eval cell).

- [ ] **Step 4: Insert the `--pre_mlp` flag in the training cell**

Find the unique training-cell line (raw JSON, line ~279):

```
    "    '--save_path', SAVE_PATH,\n",
```

Replace it with (insert `--pre_mlp` immediately before it):

```
    "    '--pre_mlp',\n",
    "    '--save_path', SAVE_PATH,\n",
```

- [ ] **Step 5: Update the training-cell header comment**

The training cell's first lines describe v26. Replace the comment line that begins `# v26 = paper-faithful geometry probe.` (or equivalent v26 description) so it reads:

```
# v27 = v26 + paper Eq 11 pre-MLP (--pre_mlp), single-variable ablation. Trains in
# the paper CHALLENGING scenario (scattered humans, 15x15 arena), robot 1.0 m/s,
# from SCRATCH. Only change vs v26: the --pre_mlp edge embedding.
```

- [ ] **Step 6: Verify the notebook is valid JSON and v27-clean**

Run:
```bash
python -c "import json; nb=json.load(open('sncp_ppo_colab.ipynb',encoding='utf-8')); src=''.join(''.join(c['source']) for c in nb['cells']); print('v26 refs:', src.count('v26'), '| v27 refs:', src.count('v27'), '| --pre_mlp:', \"'--pre_mlp'\" in src, '| version 27:', \"'--version', '27'\" in src)"
```
Expected: `v26 refs: 0 | v27 refs: >=9 | --pre_mlp: True | version 27: True`

- [ ] **Step 7: Run the full marker suite (green)**

Run: `python -m pytest test_post_run_pipeline.py::test_notebook_is_v27_pre_mlp_ablation test_v16_run_readiness.py -v`
Expected: ALL PASS.

- [ ] **Step 8: Commit**

```bash
git add sncp_ppo_colab.ipynb
git commit -m "v27 notebook: add --pre_mlp + bump v26->v27 paths"
```

---

### Task 4: Full-suite + end-to-end `--pre_mlp` smoke, then commit

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass (same count as v26 baseline; the renamed tests replace the v26 ones — no net change in count).

- [ ] **Step 2: End-to-end pre_mlp pipeline smoke (construct → save → auto-detect → eval)**

Run:
```bash
python -c "
import torch
from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
from sncp_ppo.eval_report import evaluate_density
p = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, pre_mlp=True)
torch.save(p.state_dict(), 'checkpoints/sncp_ppo_v27_smoke.pt')
sd = torch.load('checkpoints/sncp_ppo_v27_smoke.pt', map_location='cpu')
p2 = build_policy_for_checkpoint(sd, robot_vpref=1.0, robot_wmax=1.8)
assert p2.pre_mlp is True, 'auto-detect failed'
p2.load_state_dict(sd)
r = evaluate_density(checkpoint_path='checkpoints/sncp_ppo_v27_smoke.pt', num_humans=5, scenario='paper_challenging', n_episodes=2, seed=100, robot_vpref=1.0, human_vpref_override=1.0)
print('SMOKE OK pre_mlp=', p2.pre_mlp, 'episodes=', len(r))
"
```
Expected: `SMOKE OK pre_mlp= True episodes= 2` with no shape/key errors (confirms the exact Colab eval path: build_policy_for_checkpoint auto-detects pre_mlp and evaluate_density forwards through it).

- [ ] **Step 3: Tiny CLI training smoke with `--pre_mlp` (train forward path)**

Run:
```bash
python -m sncp_ppo.train --pre_mlp --fixed_scenario paper_challenging --num_humans 10 --robot_vpref 1.0 --holdout_scenarios paper_standard paper_challenging --holdout_episodes 1 --total_steps 2048 --num_envs 2 --horizon 128 --bootstrap_easy_steps 0 --eval_freq_updates 2 --seed 42 --save_path checkpoints/sncp_ppo_v27_clismoke.pt
echo "exit=$?"
```
Expected: `exit=0` (training constructs the pre_mlp policy and runs the vectorized rollout/update loop without error). A `checkpoints/sncp_ppo_v27_clismoke_final.pt` is written.

- [ ] **Step 4: Clean up smoke artifacts**

```bash
rm -f checkpoints/sncp_ppo_v27_smoke.pt checkpoints/sncp_ppo_v27_clismoke.pt checkpoints/sncp_ppo_v27_clismoke_final.pt
```

- [ ] **Step 5: Run readiness checker against the repo (operator preflight)**

Run:
```bash
python -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready; s=verify_v16_run_ready(Path('.')); print(s.status); [print(' ', n) for n in s.notes]"
```
Expected: `pass` with the note `PASS: v27 Colab training and evaluation configuration is ready`.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "v27: full suite + end-to-end --pre_mlp smoke verified"
```

---

### Task 5 (run-time, post-Colab — NOT code): evaluation protocol

This task documents the comparison; it runs after the Colab training finishes. No code change.

- [ ] **Step 1: Stage the downloaded artifacts**

```bash
python stage_colab_run_artifacts.py --version 27
```
(Expects `colabout/sncp_ppo_v27.pt`, `colabout/training_*.csv`, `colabout/eval_v27_artifacts.zip`.)

- [ ] **Step 2: Local multi-seed sweep, IDENTICAL protocol to the v26 honest baseline**

Run a 5-seed (100,200,300,400,500) × 50-episode sweep at N=5/10/15/20, `paper_challenging`, `robot_vpref=1.0`, `human_vpref_override=1.0`, `max_time=None`, `human_goal_noise=0.0` against `sncp_ppo_v27.pt` (reuse the v26 sweep script). Record pooled success ± 95 % CI per density.

- [ ] **Step 3: Decision**

pre_mlp helps iff v27's CI clears v26's CI (74.8 / 61.6 / 53.2 / 43.6) at one or more densities — especially N=10 (gap a). Confirm timeout stays 0 % and report collision. Write the verdict to memory (`sncp-paper-vs-impl.md`). If pre_mlp helps gap (a), the next experiment is N~U(10,20) for gap (b); if not, pre_mlp is ruled out in the correct regime and we move to the next deferred candidate.

---

## Self-review

- **Spec coverage:** training `--pre_mlp` + v27 path (Task 3 + readiness Task 2); eval/persist v27 (Task 3); readiness `--pre_mlp` check (Task 2); version-marker tests + single-variable guard (Task 1); end-to-end smoke (Task 4); local multi-seed eval protocol (Task 5). All spec sections covered.
- **Deliberate deviation from spec:** `--baseline_json` stays `eval_v22/density_sweep.json` (NOT switched to `eval_v26`). Reason: the readiness `_baseline_densities` check requires densities (1,3,5,8,10); `eval_v26` has (5,10,15,20), which would break the check, and the in-notebook verdict is non-authoritative anyway (real comparison is the local multi-seed sweep). Documented here.
- **Placeholder scan:** none — every code/command step is concrete.
- **Naming consistency:** test names `test_notebook_is_v27_pre_mlp_ablation`, `test_v27_run_readiness_*`, `test_colab_persist_cell_downloads_eval_v27_artifact_bundle`; checker markers "v27 training"/"v27 evaluation"; paths `sncp_ppo_v27.pt` / `eval_v27` — consistent across tasks.
