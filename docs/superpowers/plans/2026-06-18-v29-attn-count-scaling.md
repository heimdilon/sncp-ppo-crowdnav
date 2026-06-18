# v29 Attention Count-Scaling (Eq 13 n) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure a single-variable v29 run that is byte-identical to v28 except it enables the paper's Eq 13 attention count-scaling (`--attn_count_scaling`), so Colab can train it and we can compare it to the v28 honest baseline.

**Architecture:** `attn_count_scaling` is already implemented (`models.py:174-176`, `softmax((n/√d_k)·QKᵀ)`) and CLI-wired (`train.py:1092/265`); `build_policy_for_checkpoint` auto-detects it from the `_attn_count_scaling` buffer key. Zero model code. The work is a v28→v29 harness version bump (notebook + run-readiness + version-marker tests) plus the one `--attn_count_scaling` flag in the training cell.

**Tech Stack:** Python, pytest, PyTorch, Jupyter notebook.

---

## File structure

- `test_post_run_pipeline.py` — notebook-faithfulness test (v28→v29, assert `--attn_count_scaling`).
- `test_v16_run_readiness.py` — readiness tests (v28→v29 markers, persist `eval_v29`).
- `sncp_ppo/run_readiness.py` — v28→v29 markers + `--attn_count_scaling` token.
- `sncp_ppo_colab.ipynb` — training cell (`--attn_count_scaling` + v29 paths, keep `--pre_mlp` + `--num_humans_range`), eval/persist/diagnostics → v29.

No changes to `models.py`, `train.py`, `crowd_env.py`, `eval_report.py`.

All `pytest` runs use `--basetemp=./.pytmp` (the default `pytest-of-kor_a` temp dir has a Windows ACL that breaks `tmp_path`).

---

### Task 1: Update version-marker tests to v29 (TDD red)

**Files:**
- Modify: `test_post_run_pipeline.py` (the v28 notebook test)
- Modify: `test_v16_run_readiness.py` (v28 tests)

- [ ] **Step 1: Replace the notebook-faithfulness test**

In `test_post_run_pipeline.py`, replace `test_notebook_is_v28_density_curriculum` with:

```python
def test_notebook_is_v29_attn_count_scaling():
    # v29 = v28 (pre-MLP + density curriculum) + Eq 13 attention count-scaling ONLY.
    # The defining change is --attn_count_scaling; --pre_mlp and --num_humans_range stay.
    code = _colab_code_sources()
    train_cells = [s for s in code if "sncp_ppo.train" in s and "--fixed_scenario" in s]
    eval_cells = [s for s in code if "run_post_eval.py" in s]
    assert len(train_cells) == 1 and len(eval_cells) == 1
    train, ev = train_cells[0], eval_cells[0]
    assert "paper_challenging" in train
    assert "checkpoints/sncp_ppo_v29.pt" in train
    assert "'--attn_count_scaling'" in train        # the v29 change
    assert "'--pre_mlp'" in train                    # v27 carried forward
    assert "'--num_humans_range'" in train           # v28 carried forward
    for tok in ("TOTAL_STEPS = 2_500_000", "SEED = 42", "'--robot_vpref', '1.0'",
                "'--holdout_episodes', '50'"):
        assert tok in train, tok
    assert "'--version', '29'" in ev
    assert "'--baseline_nav_steps', '32'" in ev
    assert "'--max_time'" not in ev
```

- [ ] **Step 2: Update the readiness tests**

In `test_v16_run_readiness.py`:
- `def test_v28_run_readiness_passes_current_repo():` → `def test_v29_run_readiness_passes_current_repo():` (body unchanged).
- `def test_v28_run_readiness_flags_stale_notebook(tmp_path):` → `def test_v29_run_readiness_flags_stale_notebook(tmp_path):`; comment → `# A pre-v29 notebook (v23..v28 markers) ...`; the two note asserts:

```python
    assert any("v29 training" in note for note in summary.notes)
    assert any("v29 evaluation" in note for note in summary.notes)
```

- `def test_colab_persist_cell_downloads_eval_v28_artifact_bundle():` → `def test_colab_persist_cell_downloads_eval_v29_artifact_bundle():`; the two asserts:

```python
    assert "'eval_v29_artifacts'" in persist_cell
    assert "'eval_v29'" in persist_cell
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest test_post_run_pipeline.py::test_notebook_is_v29_attn_count_scaling test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: FAIL (notebook/readiness still v28).

- [ ] **Step 4: Commit**

```bash
git add test_post_run_pipeline.py test_v16_run_readiness.py
git commit -m "v29 tests: bump version markers to v29 + assert --attn_count_scaling (red)"
```

---

### Task 2: Update run-readiness checker to v29

**Files:**
- Modify: `sncp_ppo/run_readiness.py`

- [ ] **Step 1: Update the module comment**

Replace the v28 comment block (the 4 lines starting `# v28 = ...`) with:

```python
# v29 = v28 (pre-MLP + density curriculum) + paper Eq 13 attention count-scaling
# (--attn_count_scaling), single-variable. Budget (challenging 50s, standard 12.5s), 8m
# crossing and normalized comfort (Eq 7) stay env-DERIVED from --fixed_scenario
# paper_challenging. The only change vs v28 is --attn_count_scaling + the v29 save path.
```

- [ ] **Step 2: Bump all v28→v29 path strings (replace every occurrence)**

Replace every `sncp_ppo_v28.pt` → `sncp_ppo_v29.pt`; every `eval_v28` → `eval_v29`; the
`"'--version', '28'"` token → `"'--version', '29'"`.

- [ ] **Step 3: Add the `--attn_count_scaling` token**

In `TRAINING_TOKENS`, after the `"'--num_humans_range'"` line, add:

```python
    "'--num_humans_range'",
    "'--attn_count_scaling'",
    "'--save_path', SAVE_PATH",
```

- [ ] **Step 4: Bump the checker names + PASS message**

Replace every `"v28 training"` → `"v29 training"`, every `"v28 evaluation"` → `"v29 evaluation"`, and `"PASS: v28 ..."` → `"PASS: v29 Colab training and evaluation configuration is ready"`.

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/run_readiness.py
git commit -m "v29 readiness: v29 markers + --attn_count_scaling token"
```

---

### Task 3: Update the notebook to v29

**Files:**
- Modify: `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Apply the v29 edits**

Run this script (inserts `--attn_count_scaling` after `--pre_mlp`, then v28→v29 substitutions):

```bash
python - <<'PY'
path='sncp_ppo_colab.ipynb'; raw=open(path,encoding='utf-8').read(); lines=raw.split('\n')
ins=0; out=[]
for ln in lines:
    out.append(ln)
    if "'--pre_mlp'," in ln:                                  # insert attn flag right after pre_mlp
        out.append(ln.replace("'--pre_mlp',", "'--attn_count_scaling',"))
        ins+=1
assert ins==1, ins
s='\n'.join(out)
s=s.replace('sncp_ppo_v28.pt','sncp_ppo_v29.pt')
s=s.replace('eval_v28','eval_v29')
s=s.replace("'--version', '28'","'--version', '29'")
s=s.replace("# v28 = v27 (pre-MLP) + N~U(10,20) density curriculum (--num_humans_range), single-variable. Trains in the paper CHALLENGING scenario",
            "# v29 = v28 + paper Eq 13 attention count-scaling (--attn_count_scaling), single-variable. Trains in the paper CHALLENGING scenario")
s=s.replace("## Current run: v28 - v27 + N~U(10,20) density curriculum (--num_humans_range), single-variable",
            "## Current run: v29 - v28 + Eq 13 attention count-scaling (--attn_count_scaling), single-variable")
s=s.replace("## 3. Training (v28 - v27 + density curriculum)","## 3. Training (v29 - v28 + attn count-scaling)")
s=s.replace("## 8. Notes & roadmap (current: v28 - v27 + density curriculum)","## 8. Notes & roadmap (current: v29 - v28 + attn count-scaling)")
import json; json.loads(s); open(path,'w',encoding='utf-8',newline='\n').write(s)
print('v28 left:', 'v28' in s.replace('eval_v28','').replace('sncp_ppo_v28',''), '| attn:', "'--attn_count_scaling'" in s, '| pre_mlp:', "'--pre_mlp'" in s, '| range:', "'--num_humans_range'" in s, '| ver29:', "'--version', '29'" in s)
PY
```
Expected: `attn: True | pre_mlp: True | range: True | ver29: True` (and no `sncp_ppo_v28.pt` / `eval_v28` remain).

- [ ] **Step 2: Run the marker suite (green) + readiness preflight**

Run: `python -m pytest test_post_run_pipeline.py::test_notebook_is_v29_attn_count_scaling test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: PASS.

Run: `python -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready as v; s=v(Path('.')); print(s.status); [print(' ',n) for n in s.notes]"`
Expected: `pass` with `PASS: v29 Colab training and evaluation configuration is ready`.

- [ ] **Step 3: Commit**

```bash
git add sncp_ppo_colab.ipynb
git commit -m "v29 notebook: add --attn_count_scaling + bump v28->v29 paths"
```

---

### Task 4: Full suite + end-to-end smoke

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q --basetemp=./.pytmp`
Expected: all pass (same count as v28; the v28 marker tests are renamed, not added).

- [ ] **Step 2: End-to-end smoke — construct (pre_mlp + attn) → save → auto-detect → eval**

Run:
```bash
python -c "
import torch
from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
from sncp_ppo.eval_report import evaluate_density
p = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, pre_mlp=True, attn_count_scaling=True)
torch.save(p.state_dict(), 'checkpoints/sncp_ppo_v29_smoke.pt')
sd = torch.load('checkpoints/sncp_ppo_v29_smoke.pt', map_location='cpu')
p2 = build_policy_for_checkpoint(sd, robot_vpref=1.0, robot_wmax=1.8)
assert p2.pre_mlp is True and p2.attn_count_scaling is True, 'auto-detect failed'
p2.load_state_dict(sd)
r = evaluate_density(checkpoint_path='checkpoints/sncp_ppo_v29_smoke.pt', num_humans=5, scenario='paper_challenging', n_episodes=2, seed=100, robot_vpref=1.0, human_vpref_override=1.0)
print('SMOKE OK pre_mlp=', p2.pre_mlp, 'attn=', p2.attn_count_scaling, 'episodes=', len(r))
"
```
Expected: `SMOKE OK pre_mlp= True attn= True episodes= 2` (confirms both flags auto-detect together and the eval path forwards through count-scaled attention).

- [ ] **Step 3: Tiny CLI training smoke with all three flags**

Run:
```bash
python -m sncp_ppo.train --pre_mlp --attn_count_scaling --num_humans_range 10 12 \
  --fixed_scenario paper_challenging --num_humans 10 --robot_vpref 1.0 \
  --holdout_scenarios paper_standard paper_challenging --holdout_episodes 1 \
  --total_steps 4096 --num_envs 2 --horizon 128 --bootstrap_easy_steps 0 \
  --eval_freq_updates 0 --seed 42 --save_path checkpoints/sncp_ppo_v29_clismoke.pt > /tmp/v29_smoke.log 2>&1
echo "exit=$?"; tail -3 /tmp/v29_smoke.log
```
Expected: `exit=0` and "Vectorized training completed!" (the train forward path runs with all three flags). A `checkpoints/sncp_ppo_v29_clismoke_final.pt` is written.

- [ ] **Step 4: Clean up smoke artifacts**

```bash
rm -f checkpoints/sncp_ppo_v29_smoke.pt checkpoints/sncp_ppo_v29_clismoke.pt checkpoints/sncp_ppo_v29_clismoke_final.pt /tmp/v29_smoke.log
rm -f logs/training_*$(date +%Y%m%d)*.pt 2>/dev/null
rm -rf ./.pytmp
```
(Also delete the junk smoke training CSV the CLI run wrote under `logs/` — find the newest `logs/training_*.csv` created in this session and remove it.)

- [ ] **Step 5: Commit (if any tracked changes remain; otherwise skip)**

```bash
git add -A && git commit -m "v29: full suite + end-to-end attn+pre_mlp smoke verified" || echo "nothing to commit"
```

---

### Task 5 (run-time, post-Colab — NOT code): evaluation protocol

- [ ] **Step 1: Stage** — copy `sncp_ppo_v29.pt` + training CSV to the repo root (Colab eval cell is non-functional; checkpoint is enough), or `python stage_colab_run_artifacts.py --version 29`.
- [ ] **Step 2: Local multi-seed sweep**, IDENTICAL protocol to v26/v27/v28: 5 seeds (100–500) × 50 ep at N=5/10/15/20, `paper_challenging`, robot 1.0, human 1.0, max_time None, goal_noise 0, on `sncp_ppo_v29.pt`. Record pooled success ± 95 % CI.
- [ ] **Step 3: Decision** — count-scaling helps iff v29's CI clears v28's (94.4 / 87.6 / 79.2 / 73.2) at the high densities (esp. N=20), with no regression at N=5/10. Confirm timeout stays 0. Report a flat/negative result honestly as a clean negative. Write the verdict to memory (`sncp-paper-vs-impl.md`).

---

## Self-review

- **Spec coverage:** training `--attn_count_scaling` + keeps `--pre_mlp`/`--num_humans_range` (Task 3 + readiness Task 2); v29 paths (Task 3); readiness `--attn_count_scaling` token (Task 2); version-marker tests (Task 1); end-to-end auto-detect + eval smoke (Task 4 Step 2) and train-path smoke (Task 4 Step 3); holdout untouched (no task changes it); local multi-seed eval (Task 5). All spec items covered.
- **Placeholder scan:** none — every code/command step is concrete.
- **Naming consistency:** flag `--attn_count_scaling`; test names `test_notebook_is_v29_attn_count_scaling`, `test_v29_run_readiness_*`, `test_colab_persist_cell_downloads_eval_v29_artifact_bundle`; markers `sncp_ppo_v29.pt` / `eval_v29` / `'--version', '29'` — consistent across tasks.
