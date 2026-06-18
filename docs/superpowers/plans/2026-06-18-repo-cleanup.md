# Repo Cleanup & Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** De-clutter the repo — move loose root `test_*.py` → `tests/` and scripts → `scripts/(+archive/)`, untrack experiment-artifact binaries from GitHub, and expand `.gitignore` — with the full test suite green at every phase.

**Architecture:** A root `conftest.py` puts repo-root + `scripts/` + `scripts/archive/` on `sys.path` so the 10 tests that `import` root scripts at function scope keep resolving after the move. Training is unaffected (`python -m sncp_ppo.train`). Pipeline-referenced artifacts (`eval_v15/16/22`) stay tracked; only zero-reference ones are untracked.

**Tech Stack:** git, Python, pytest, Jupyter notebook.

All `pytest` runs use `--basetemp=./.pytmp` (the default pytest temp dir has a Windows ACL that breaks `tmp_path`). `.pytmp` is gitignored in Task 4 and removed at the end.

---

### Task 1: conftest.py + move tests → tests/

- [ ] **Step 1: Create `conftest.py` at repo root**

```python
"""Pytest import bridge for the tests/ + scripts/ reorg.

Several tests import root-level scripts at function scope (e.g. `import run_post_eval`,
`import select_v18_candidate`). After moving tests into tests/ and scripts into scripts/,
these sys.path entries keep those imports resolving. Training/eval are unaffected.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "scripts", _ROOT / "scripts" / "archive"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
```

- [ ] **Step 2: Move all test files into `tests/`**

```bash
mkdir -p tests
for f in test_*.py; do git mv "$f" tests/; done
git ls-files 'tests/test_*.py' | wc -l   # expect 38
```

- [ ] **Step 3: Verify the full suite still passes (the key gate)**

Run: `python -m pytest -q --basetemp=./.pytmp`
Expected: `206 passed` (tests now in tests/; conftest bridges the function-level script imports; scripts are still at root and reachable via the repo-root sys.path entry).

- [ ] **Step 4: Commit**

```bash
git add conftest.py && git add -A tests/
git commit -m "chore: add conftest sys.path bridge + move test_*.py into tests/"
```

---

### Task 2: Move scripts → scripts/ (active) and scripts/archive/ (one-offs)

- [ ] **Step 1: Move the active tool scripts**

```bash
mkdir -p scripts/archive
for f in run_post_eval.py stage_colab_run_artifacts.py visualize_trajectory.py \
         visualize_trajectory_gif.py visualize_all_scenarios_gif.py plot_training.py \
         evaluate_policy_report.py evaluate_custom_scenario.py compare_policy_reports.py \
         analyze_training_log.py benchmark.py benchmark_orca.py benchmark_ppo.py; do
  git mv "$f" scripts/; done
```

- [ ] **Step 2: Move the version-specific / one-off scripts to archive**

```bash
for f in run_v16_post_eval.py run_v17_review.py select_v18_candidate.py \
         verify_v16_artifacts.py verify_v16_run_ready.py verify_v18_ready.py \
         run_probes.py _bench_sfm.py visualize_architecture.py; do
  git mv "$f" scripts/archive/; done
```

- [ ] **Step 3: Verify the full suite still passes**

Run: `python -m pytest -q --basetemp=./.pytmp`
Expected: `206 passed` (tests' function-level `import run_post_eval` / `import select_v18_candidate` etc. now resolve via the conftest `scripts/` + `scripts/archive/` entries).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: move utility scripts into scripts/ (+ archive/ for version one-offs)"
```

---

### Task 3: Notebook script paths + test_attention_scaling skip-guard

**Files:** `sncp_ppo_colab.ipynb`, `tests/test_attention_scaling.py`

- [ ] **Step 1: Update the 4 notebook script-call paths**

```bash
python - <<'PY'
import json
p='sncp_ppo_colab.ipynb'; s=open(p,encoding='utf-8').read()
reps=[("python visualize_trajectory_gif.py","python scripts/visualize_trajectory_gif.py"),
      ("python visualize_trajectory.py","python scripts/visualize_trajectory.py"),
      ("python plot_training.py","python scripts/plot_training.py"),
      ("'run_post_eval.py'","'scripts/run_post_eval.py'")]
for a,b in reps:
    assert a in s, f"NOT FOUND: {a}"
    s=s.replace(a,b)
json.loads(s); open(p,'w',encoding='utf-8',newline='\n').write(s)
print("notebook script paths updated; run_post_eval token still substring:", "run_post_eval.py" in s)
PY
```
Expected: prints `... True` (the readiness/marker `run_post_eval.py` substring tokens still match).

- [ ] **Step 2: Add the skip-guard to `tests/test_attention_scaling.py`**

Find the line that loads the milestone checkpoint:

```python
    state = torch.load('checkpoints/sncp_ppo_v18.pt', map_location='cpu')
```

Insert, immediately before it (same indentation):

```python
    import os, pytest
    if not os.path.exists('checkpoints/sncp_ppo_v18.pt'):
        pytest.skip('milestone checkpoint checkpoints/sncp_ppo_v18.pt is git-ignored; present only locally')
    state = torch.load('checkpoints/sncp_ppo_v18.pt', map_location='cpu')
```

- [ ] **Step 3: Verify marker tests + readiness + the attn test (still runs locally)**

```bash
python -m pytest tests/test_post_run_pipeline.py::test_notebook_is_v29_attn_count_scaling \
  tests/test_v16_run_readiness.py tests/test_attention_scaling.py -q --basetemp=./.pytmp
python -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready as v; print(v(Path('.')).status)"
```
Expected: tests pass (attn test runs, not skipped — the .pt is present locally); readiness prints `pass`.

- [ ] **Step 4: Commit**

```bash
git add sncp_ppo_colab.ipynb tests/test_attention_scaling.py
git commit -m "chore: point notebook at scripts/ paths + guard attn test on git-ignored checkpoint"
```

---

### Task 4: .gitignore expansion + untrack artifacts + scratch/

- [ ] **Step 1: Append artifact patterns to `.gitignore`**

Append this block to `.gitignore`:

```
# --- Project: experiment artifacts & local deliverables (local only) ---
/eval_v*/
/*.zip
/training_*.csv
/*_multiseed_result.json
/sncp_ppo_v*.pt
checkpoints/*.pt
/*.png
/*.gif
rapor/
ltc_sunum/
ltc_blog_app/
scratch/
.pytmp/
```

- [ ] **Step 2: Untrack zero-reference artifacts (local copies + history kept)**

```bash
git rm -r --cached eval_v18 eval_v19 eval_v21
git rm --cached checkpoints/sncp_ppo_v14.pt checkpoints/sncp_ppo_v15.pt \
  checkpoints/sncp_ppo_v16.pt checkpoints/sncp_ppo_v17.pt checkpoints/sncp_ppo_v18.pt \
  checkpoints/sncp_ppo_v19.pt checkpoints/sncp_ppo_v21.pt checkpoints/sncp_ppo_v22.pt 2>/dev/null; \
  git rm --cached checkpoints/*.pt 2>/dev/null; true
git rm --cached density_sweep_v14.png gif_v14_hard_n10.gif gif_v14_hard_n20.gif \
  gif_v14_hard_n20_FAIL.gif gif_v14_hard_n5.gif traj_v14_hard_n10.png traj_v14_hard_n5.png \
  traj_v15_hard_n10.png traj_v15_hard_n5.png v15_results.png
```
(`checkpoints/*.pt` removes whatever milestone `.pt` files are actually tracked; the explicit list is belt-and-suspenders. Files remain on disk.)

- [ ] **Step 3: Move the untracked one-off probes into a gitignored `scratch/`**

```bash
mkdir -p scratch
mv _bench_endtoend.py _eval_v24_corrected.py _make_crowd_gifs.py _oracle_feasibility.py \
   _probe_attn.py _probe_il.py _sweep_expert.py scratch/ 2>/dev/null; true
ls scratch/
```

- [ ] **Step 4: Verify suite + readiness + git status are clean**

```bash
python -m pytest -q --basetemp=./.pytmp                       # 206 passed (files still on disk)
python -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready as v; print(v(Path('.')).status)"  # pass (eval_v22 + checkpoints still on disk)
git status --short | grep -E '^\?\?' || echo "no untracked clutter remaining"
git ls-files | wc -l                                          # tracked count dropped
```
Expected: `206 passed`; `pass`; untracked list now empty (or only intended). Confirm `eval_v15`, `eval_v16`, `eval_v22` are STILL tracked: `git ls-files eval_v15 eval_v16 eval_v22 | wc -l` > 0.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: expand .gitignore + untrack artifact binaries (checkpoints, old eval dirs, root images)"
```

---

### Task 5: Final verification

- [ ] **Step 1: Full suite + readiness + moved-script smoke**

```bash
python -m pytest -q --basetemp=./.pytmp                       # 206 passed
python scripts/run_post_eval.py --help >/dev/null && echo "run_post_eval import OK from scripts/"
python -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready as v; print(v(Path('.')).status)"
rm -rf ./.pytmp
```
Expected: `206 passed`; `run_post_eval import OK from scripts/` (package imports resolve from the new location); `pass`.

- [ ] **Step 2: Show the tidied tree**

```bash
git ls-files | awk -F/ '{print (NF==1)?"[root]":$1}' | sort | uniq -c | sort -rn
ls -1
```
Confirm root is now mostly: README/AGENTS/requirements/ruff/.gitignore/conftest.py + sncp_ppo_colab.ipynb + package/dir folders (no loose test_*.py or scripts).

---

## Self-review

- **Spec coverage:** conftest sys.path bridge (T1.1); tests→tests/ (T1.2); scripts→scripts/(+archive) (T2); notebook 4 paths (T3.1); skip-guard for the v18-checkpoint test (T3.2); untrack eval_v18/19/21 + checkpoints + root images, KEEP eval_v15/16/22 (T4.2 + T4.4 confirm); .gitignore expansion incl. scratch/rapor/ltc_* (T4.1); scratch/ for one-off probes (T4.3); verification net every task. All spec items covered.
- **Placeholder scan:** none — exact commands/code throughout.
- **Naming consistency:** `conftest.py`, `tests/`, `scripts/`, `scripts/archive/`, `scratch/` used consistently; the 13 active + 9 archive script names match the spec tree; eval keep/untrack split matches the verified references.
- **Risk note:** the only fresh-clone behaviour change is `test_attention_scaling` skipping when `checkpoints/sncp_ppo_v18.pt` is absent — intended and guarded. Push to main happens only at the finishing step after user confirmation.
