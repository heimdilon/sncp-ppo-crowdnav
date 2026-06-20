# v32 Curriculum Reach N→25 + Budget 4M Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train v32 = the v30 champion with the density curriculum extended to `N∈[10,25]` and the budget raised to 4M steps (v31's node-256 dropped), to improve high-N generalization.

**Architecture:** CONFIG-ONLY. `--num_humans_range` and `--total_steps` are already CLI-wired (v28), so there is NO `models.py`/`train.py` change and NO AutoNCP topology-reroll confound. The work is a harness version bump (notebook + run-readiness + version-marker tests) v31→v32 with the two new values and the node flags removed.

**Tech Stack:** Python, pytest, Jupyter notebook.

Spec: `docs/superpowers/specs/2026-06-20-curriculum-reach-budget-design.md`. Branch: `feat/v32-curriculum-budget`.

---

## File structure

- `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` — version-marker bump v31→v32.
- `sncp_ppo/run_readiness.py` — v31→v32 markers; TRAINING_TOKENS: TOTAL_STEPS 4M, drop node tokens.
- `sncp_ppo_colab.ipynb` — training cell: TOTAL_STEPS 4M, range `10 25`, remove `--node_units`/`--node_output`; v31→v32 paths.

No changes to `models.py`, `train.py`, `crowd_env.py`, `eval_report.py`.

All `pytest` runs use `--basetemp=./.pytmp` and `C:/ProgramData/miniconda3/python.exe`.

---

### Task 1: Version-marker tests v31→v32 (TDD red)

**Files:**
- Modify: `tests/test_post_run_pipeline.py` (replace `test_notebook_is_v31_node_capacity`)
- Modify: `tests/test_v16_run_readiness.py` (rename the 3 v31 tests + asserts)

- [ ] **Step 1: Replace the notebook-faithfulness test**

In `tests/test_post_run_pipeline.py`, replace `test_notebook_is_v31_node_capacity` with:

```python
def test_notebook_is_v32_curriculum_budget():
    # v32 = v30 (pre-MLP + mean+max) + extended curriculum N->25 + budget 4M. CONFIG-only.
    # Defining changes: --num_humans_range 10 25 and TOTAL_STEPS 4M; v31's node flags are DROPPED.
    code = _colab_code_sources()
    train_cells = [s for s in code if "sncp_ppo.train" in s and "--fixed_scenario" in s]
    eval_cells = [s for s in code if "run_post_eval.py" in s]
    assert len(train_cells) == 1 and len(eval_cells) == 1
    train, ev = train_cells[0], eval_cells[0]
    assert "paper_challenging" in train
    assert "checkpoints/sncp_ppo_v32.pt" in train
    assert "'--num_humans_range', '10', '25'" in train    # curriculum reach (v32)
    assert "TOTAL_STEPS = 4_000_000" in train             # budget (v32)
    assert "'--meanmax_pool'" in train                    # v30 carried forward
    assert "'--pre_mlp'" in train                          # v27 carried forward
    assert "'--node_units'" not in train                   # v31 node capacity dropped
    for tok in ("SEED = 42", "'--robot_vpref', '1.0'", "'--holdout_episodes', '50'"):
        assert tok in train, tok
    assert "'--version', '32'" in ev
    assert "'--baseline_nav_steps', '32'" in ev
    assert "'--max_time'" not in ev
```

- [ ] **Step 2: Update the readiness tests**

In `tests/test_v16_run_readiness.py`:
- `def test_v31_run_readiness_passes_current_repo():` → `def test_v32_run_readiness_passes_current_repo():` (body unchanged).
- `def test_v31_run_readiness_flags_stale_notebook(tmp_path):` → `def test_v32_run_readiness_flags_stale_notebook(tmp_path):`; change the comment to `# A pre-v32 notebook (v23..v31 markers) must be flagged: the v32 cells are absent.` and the two note asserts:

```python
    assert any("v32 training" in note for note in summary.notes)
    assert any("v32 evaluation" in note for note in summary.notes)
```

- `def test_colab_persist_cell_downloads_eval_v31_artifact_bundle():` → `def test_colab_persist_cell_downloads_eval_v32_artifact_bundle():`; the two asserts:

```python
    assert "'eval_v32_artifacts'" in persist_cell
    assert "'eval_v32'" in persist_cell
```

- [ ] **Step 3: Run to verify they fail**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py::test_notebook_is_v32_curriculum_budget tests/test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: FAIL (notebook/readiness still v31).

- [ ] **Step 4: Commit**

```bash
git add tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py
git commit -m "v32 tests: bump version markers to v32 + assert curriculum 10 25 / 4M (red)"
```

---

### Task 2: Update run-readiness checker to v32

**Files:**
- Modify: `sncp_ppo/run_readiness.py`

- [ ] **Step 1: Update the module comment**

Replace the `# v31 = ...` comment block (the 4 lines starting `# v31 =`) with:

```python
# v32 = v30 (pre-MLP + mean+max pooling) + extended density curriculum N in [10,25] and
# budget 4M steps (--num_humans_range 10 25, --total_steps 4_000_000), CONFIG-only (no model
# code; v31's node-256 dropped). Budget/8m crossing/comfort stay env-DERIVED from
# --fixed_scenario paper_challenging. Two-variable run (curriculum + budget) by user choice.
```

- [ ] **Step 2: Bump the budget token + drop the node tokens**

In `TRAINING_TOKENS`: replace `"TOTAL_STEPS = 2_500_000",` with `"TOTAL_STEPS = 4_000_000",`; delete the two lines `"'--node_units', '256'",` and `"'--node_output', '96'",` (keep `"'--pre_mlp'",`, `"'--num_humans_range'",`, `"'--meanmax_pool'",`).

- [ ] **Step 3: Bump every v31→v32 path/version string**

Replace every `sncp_ppo_v31.pt` → `sncp_ppo_v32.pt`; every `eval_v31` → `eval_v32`; the `"'--version', '31'"` token → `"'--version', '32'"`; every `"v31 training"` → `"v32 training"`, `"v31 evaluation"` → `"v32 evaluation"`; the PASS message → `"PASS: v32 Colab training and evaluation configuration is ready"`.

- [ ] **Step 4: Commit**

```bash
git add sncp_ppo/run_readiness.py
git commit -m "v32 readiness: v32 markers + 4M budget token, drop node tokens"
```

---

### Task 3: Update the notebook to v32 + green

**Files:**
- Modify: `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Apply the v32 edits**

Run (removes the two node-flag lines, bumps the budget + range values, then v31→v32 substitutions):

```bash
C:/ProgramData/miniconda3/python.exe - <<'PY'
path='sncp_ppo_colab.ipynb'; raw=open(path,encoding='utf-8').read(); lines=raw.split('\n')
out=[]; removed=0
for ln in lines:
    if "'--node_units'," in ln or "'--node_output'," in ln:    # drop v31's node flags
        removed+=1; continue
    out.append(ln)
assert removed==2, removed
s='\n'.join(out)
s=s.replace("TOTAL_STEPS = 2_500_000","TOTAL_STEPS = 4_000_000")
s=s.replace("'--num_humans_range', '10', '20'","'--num_humans_range', '10', '25'")
s=s.replace('sncp_ppo_v31.pt','sncp_ppo_v32.pt')
s=s.replace('eval_v31','eval_v32')
s=s.replace("'--version', '31'","'--version', '32'")
s=s.replace("v31 - v30 + node-fusion capacity 256/96 (--node_units/--node_output)","v32 - v30 + curriculum N->25 + budget 4M")
s=s.replace("## 3. Training (v31 - v30 + node capacity 256/96)","## 3. Training (v32 - v30 + curriculum N->25 + budget 4M)")
s=s.replace("## 8. Notes & roadmap (current: v31 - v30 + node capacity 256/96)","## 8. Notes & roadmap (current: v32 - v30 + curriculum N->25 + budget 4M)")
import json; json.loads(s); open(path,'w',encoding='utf-8',newline='\n').write(s)
print('removed node lines:', removed, '| range 10 25:', "'--num_humans_range', '10', '25'" in s,
      '| 4M:', "TOTAL_STEPS = 4_000_000" in s, '| no node:', "'--node_units'" not in s,
      '| meanmax:', "'--meanmax_pool'" in s, '| v32:', "'--version', '32'" in s,
      '| no v31 ckpt:', 'sncp_ppo_v31.pt' not in s, '| no 2.5M:', "TOTAL_STEPS = 2_500_000" not in s)
PY
```
Expected: `removed node lines: 2 | range 10 25: True | 4M: True | no node: True | meanmax: True | v32: True | no v31 ckpt: True | no 2.5M: True`.

- [ ] **Step 2: Run the marker suite (green) + readiness preflight**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py::test_notebook_is_v32_curriculum_budget tests/test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: PASS.

Run: `C:/ProgramData/miniconda3/python.exe -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready as v; s=v(Path('.')); print(s.status); [print(' ',n) for n in s.notes]"`
Expected: `pass` with `PASS: v32 Colab training and evaluation configuration is ready`.

- [ ] **Step 3: Full suite**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest -q --basetemp=./.pytmp`
Expected: all pass.

- [ ] **Step 4: Tiny CLI training smoke (verify the N=25 curriculum window builds + trains)**

Run:
```bash
C:/ProgramData/miniconda3/python.exe -m sncp_ppo.train --pre_mlp --meanmax_pool --num_humans_range 10 25 \
  --fixed_scenario paper_challenging --num_humans 10 --robot_vpref 1.0 \
  --holdout_scenarios paper_standard paper_challenging --holdout_episodes 1 \
  --total_steps 4096 --num_envs 2 --horizon 128 --bootstrap_easy_steps 0 \
  --seed 42 --save_path checkpoints/sncp_ppo_v32_clismoke.pt > /tmp/v32_smoke.log 2>&1
echo "exit=$?"; tail -3 /tmp/v32_smoke.log
```
Expected: `exit=0` and "Vectorized training completed!" (the curriculum samples N up to 25 and the env builds those scenes).

- [ ] **Step 5: Clean up smoke artifacts**

```bash
rm -f checkpoints/sncp_ppo_v32_clismoke.pt checkpoints/sncp_ppo_v32_clismoke_final.pt /tmp/v32_smoke.log
rm -rf ./.pytmp
```
(Also delete the newest `logs/training_*.csv` the CLI smoke wrote this session.)

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo_colab.ipynb
git commit -m "v32 notebook: curriculum 10 25 + 4M budget, drop node flags, bump v31->v32 paths"
```

---

### Task 4 (run-time, post-Colab — NOT code): evaluation protocol

- [ ] **Step 1: Merge + push** — finishing-a-development-branch merges `feat/v32-curriculum-budget` into `main` and pushes (Colab pulls `main`).
- [ ] **Step 2: Train on Colab** — pull `main`, run the v32 training cell (~5-6 h A100 at 4M steps + N up to 25), copy `sncp_ppo_v32.pt` to the repo root.
- [ ] **Step 3: Honest local multi-seed sweep** — copy `scratch/_sweep_v31.py` → `scratch/_sweep_v32.py`, set `CKPT='sncp_ppo_v32.pt'` and `OUT='v32_multiseed_result.json'`; run with `C:/ProgramData/miniconda3/python.exe scratch/_sweep_v32.py` (5 seeds 100–500 × 50 ep at N=5/10/15/20, paper_challenging, robot 1.0, human 1.0, max_time None, goal_noise 0). Optional: add 25 to DENSITIES for an informational N=25 probe.
- [ ] **Step 4: Decide** — copy `scratch/_analyze_v31.py` → `_analyze_v32.py`, load v32 vs the v30 baseline (success 97.2/89.6/85.6/79.2; collision 2.8/10.4/14.4/20.8); curriculum+budget helps iff high-N success rises and/or collision drops (esp N=15/20) with no regression at N=5/10 and timeout 0 (Wilson CIs + two-proportion z, Bonferroni α=0.0125). Two-variable, so a win is not cleanly attributable. Report honestly; write the verdict to memory (`sncp-paper-vs-impl.md`).

---

## Self-review

- **Spec coverage:** curriculum range 10→25 + budget 4M (Tasks 2-3); drop node flags (Tasks 2-3, asserted in Task 1); config-only / no model code (no models/train task); version-marker + readiness + notebook bump (Tasks 1-3); CLI smoke verifies N=25 window (Task 3 Step 4); honest v30-baseline eval + two-variable caveat (Task 4). All spec items covered.
- **Placeholder scan:** none — every code/command step is concrete.
- **Type/name consistency:** values `--num_humans_range 10 25`, `TOTAL_STEPS = 4_000_000`; node flags removed (asserted `"'--node_units'" not in train`); test `test_notebook_is_v32_curriculum_budget`, readiness `test_v32_*`; markers `sncp_ppo_v32.pt` / `eval_v32` / `'--version', '32'`; v32 keeps `--meanmax_pool`/`--pre_mlp` — consistent across tasks.
