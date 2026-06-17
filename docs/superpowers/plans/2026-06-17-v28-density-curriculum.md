# v28 Density Curriculum (N~U(10,20)) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train v28 across the paper's pedestrian-count range (N~U(10,20)) instead of fixed N=10, to close the high-N falloff (gap b), changing only the training-density distribution vs v27.

**Architecture:** Add a `--num_humans_range MIN MAX` flag. In the vectorized trainer's phase selector (`select_vectorized_phase`), the `fixed_scenario` post-bootstrap branch samples `N = rng.randint(MIN, MAX)` per update window when the range is set; the existing per-window env-rebuild path (`train.py:839-847`) handles the changing N. Flag absent → byte-identical to v27.

**Tech Stack:** Python, pytest, PyTorch, gymnasium SyncVectorEnv, Jupyter notebook.

---

## File structure

- `sncp_ppo/train.py` — `select_vectorized_phase` (sample N), `build_parser` (`--num_humans_range`), `_train_vectorized` (thread the range to both call sites).
- `test_density_curriculum.py` — NEW: unit tests for the sampler + CLI parse.
- `sncp_ppo_colab.ipynb` — training cell (`--num_humans_range 10 20`, keep `--pre_mlp`, v28 paths), eval/persist/diagnostics → v28.
- `sncp_ppo/run_readiness.py` — v27→v28 markers + `--num_humans_range` token.
- `test_post_run_pipeline.py`, `test_v16_run_readiness.py` — version-marker tests → v28.

No changes to `models.py`, `crowd_env.py`, `eval_report.py`.

---

### Task 1: Density sampling in the phase selector + CLI flag

**Files:**
- Create: `test_density_curriculum.py`
- Modify: `sncp_ppo/train.py` (`select_vectorized_phase` def ~687-709; `_train_vectorized` ~754 + call sites 811, 830; `build_parser` ~1017)

- [ ] **Step 1: Write the failing unit tests**

Create `test_density_curriculum.py`:

```python
import random as _random
from sncp_ppo.train import select_vectorized_phase, build_parser


def test_density_curriculum_samples_in_range():
    rng = _random.Random(0)
    for _ in range(50):
        (sc, n, _vpref), is_replay = select_vectorized_phase(
            300000, 2_500_000, 10, rng=rng,
            fixed_scenario='paper_challenging', bootstrap_easy_steps=200000,
            num_humans_range=(10, 20))
        assert sc == 'paper_challenging'
        assert 10 <= n <= 20
        assert is_replay is False


def test_density_curriculum_actually_varies():
    rng = _random.Random(1)
    seen = {select_vectorized_phase(
        300000, 2_500_000, 10, rng=rng, fixed_scenario='paper_challenging',
        bootstrap_easy_steps=200000, num_humans_range=(10, 20))[0][1]
        for _ in range(60)}
    assert len(seen) >= 5  # spread across the range, not a constant


def test_density_curriculum_bootstrap_still_easy():
    (sc, n, _v), _ = select_vectorized_phase(
        1000, 2_500_000, 10, fixed_scenario='paper_challenging',
        bootstrap_easy_steps=200000, num_humans_range=(10, 20))
    assert sc == 'easy' and n == 1


def test_no_range_uses_fixed_num_humans():
    (sc, n, _v), _ = select_vectorized_phase(
        300000, 2_500_000, 10, fixed_scenario='paper_challenging',
        bootstrap_easy_steps=200000, num_humans_range=None)
    assert sc == 'paper_challenging' and n == 10


def test_parser_num_humans_range():
    assert build_parser().parse_args(['--num_humans_range', '10', '20']).num_humans_range == [10, 20]
    assert build_parser().parse_args([]).num_humans_range is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_density_curriculum.py -q --basetemp=./.pytmp`
Expected: FAIL — `select_vectorized_phase` has no `num_humans_range` parameter (TypeError) and `--num_humans_range` is an unrecognized argument.

- [ ] **Step 3: Add the `num_humans_range` parameter + sampling**

In `sncp_ppo/train.py`, change the `select_vectorized_phase` signature (line ~687-689) to add the parameter:

```python
def select_vectorized_phase(steps_seen, total_steps, final_num_humans,
                            replay_ratio=0.0, rng=random, fixed_scenario=None,
                            bootstrap_easy_steps=0, num_humans_range=None):
```

Then change the `fixed_scenario` post-bootstrap return (currently lines 708-709):

```python
        _, vpref = SCENARIO_HOLDOUT_CONFIG.get(fixed_scenario, (5, 0.26))
        return (fixed_scenario, final_num_humans, vpref), False
```

to:

```python
        _, vpref = SCENARIO_HOLDOUT_CONFIG.get(fixed_scenario, (5, 0.26))
        n_humans = final_num_humans
        if num_humans_range is not None:
            n_humans = rng.randint(int(num_humans_range[0]), int(num_humans_range[1]))
        return (fixed_scenario, n_humans, vpref), False
```

(`random.Random.randint(a, b)` is inclusive, so MIN and MAX are both reachable.)

- [ ] **Step 4: Add the CLI flag**

In `build_parser`, immediately after the `--num_humans` argument (line ~1018), add:

```python
    parser.add_argument('--num_humans_range', type=int, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Density curriculum: sample N~U[MIN,MAX] per update window '
                             '(paper trains 10-20). Default None = fixed --num_humans.')
```

- [ ] **Step 5: Thread the range into `_train_vectorized`**

In `_train_vectorized`, near the other `getattr(args, ...)` reads (after line ~755 `bootstrap_easy_steps = ...`), add:

```python
    num_humans_range = getattr(args, 'num_humans_range', None)
```

Then add `num_humans_range=num_humans_range` to BOTH `select_vectorized_phase` calls:

- The initial-phase call (line ~811-814):

```python
    (scenario, H, vpref), _ = select_vectorized_phase(
        0, args.total_steps, args.num_humans, fixed_scenario=fixed_scenario,
        bootstrap_easy_steps=bootstrap_easy_steps, num_humans_range=num_humans_range,
    )
```

- The per-window call (line ~830-838):

```python
        (next_scenario, next_H, next_vpref), is_replay_update = select_vectorized_phase(
            total_steps,
            args.total_steps,
            args.num_humans,
            replay_ratio=replay_ratio,
            rng=random,
            fixed_scenario=fixed_scenario,
            bootstrap_easy_steps=bootstrap_easy_steps,
            num_humans_range=num_humans_range,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest test_density_curriculum.py -q --basetemp=./.pytmp`
Expected: PASS (5 passed).

- [ ] **Step 7: Commit**

```bash
git add test_density_curriculum.py sncp_ppo/train.py
git commit -m "v28: --num_humans_range density curriculum in vectorized phase selector"
```

---

### Task 2: Notebook + run-readiness + version-marker tests → v28

**Files:**
- Modify: `test_post_run_pipeline.py` (the v27 notebook test), `test_v16_run_readiness.py` (v27 tests)
- Modify: `sncp_ppo/run_readiness.py`
- Modify: `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Update the notebook-faithfulness test (red)**

In `test_post_run_pipeline.py`, replace the `test_notebook_is_v27_pre_mlp_ablation` function with:

```python
def test_notebook_is_v28_density_curriculum():
    # v28 = v27 (pre-MLP) + N~U(10,20) density curriculum ONLY. The defining change
    # is --num_humans_range; --pre_mlp and the env-derived paper budget are retained.
    code = _colab_code_sources()
    train_cells = [s for s in code if "sncp_ppo.train" in s and "--fixed_scenario" in s]
    eval_cells = [s for s in code if "run_post_eval.py" in s]
    assert len(train_cells) == 1 and len(eval_cells) == 1
    train, ev = train_cells[0], eval_cells[0]
    assert "paper_challenging" in train
    assert "checkpoints/sncp_ppo_v28.pt" in train
    assert "'--pre_mlp'" in train                       # v27 carried forward
    assert "'--num_humans_range'" in train              # the v28 change
    for tok in ("TOTAL_STEPS = 2_500_000", "SEED = 42", "'--robot_vpref', '1.0'",
                "'--holdout_episodes', '50'"):
        assert tok in train, tok
    assert "'--version', '28'" in ev
    assert "'--baseline_nav_steps', '32'" in ev
    assert "'--max_time'" not in ev
```

- [ ] **Step 2: Update the readiness tests (red)**

In `test_v16_run_readiness.py`:
- `def test_v27_run_readiness_passes_current_repo():` → `def test_v28_run_readiness_passes_current_repo():` (body unchanged).
- `def test_v27_run_readiness_flags_stale_notebook(tmp_path):` → `def test_v28_run_readiness_flags_stale_notebook(tmp_path):`; change the comment to `# A pre-v28 notebook ...` and the two note asserts:

```python
    assert any("v28 training" in note for note in summary.notes)
    assert any("v28 evaluation" in note for note in summary.notes)
```

- `def test_colab_persist_cell_downloads_eval_v27_artifact_bundle():` → `def test_colab_persist_cell_downloads_eval_v28_artifact_bundle():`; change the two asserts:

```python
    assert "'eval_v28_artifacts'" in persist_cell
    assert "'eval_v28'" in persist_cell
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest test_post_run_pipeline.py::test_notebook_is_v28_density_curriculum test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: FAIL (notebook/readiness still v27).

- [ ] **Step 4: Update `run_readiness.py`**

In `sncp_ppo/run_readiness.py`:
- Update the module comment block to describe v28 (v27 + density curriculum).
- In `TRAINING_TOKENS`: change `"SAVE_PATH = 'checkpoints/sncp_ppo_v27.pt'"` → `"SAVE_PATH = 'checkpoints/sncp_ppo_v28.pt'"`, and add a new token line after `"'--pre_mlp'"`:

```python
    "'--pre_mlp'",
    "'--num_humans_range'",
    "'--save_path', SAVE_PATH",
```

- In `EVALUATION_TOKENS`: `"CHECKPOINT = 'checkpoints/sncp_ppo_v27.pt'"` → `v28`; `"EVAL_OUT = 'eval_v27'"` → `'eval_v28'`; `"'--version', '27'"` → `"'--version', '28'"`.
- In `verify_v16_run_ready`: the two `_find_unique_cell` markers `sncp_ppo_v27.pt` → `v28` and names `"v27 training"`/`"v27 evaluation"` → `"v28 training"`/`"v28 evaluation"`; the two `_check_tokens` names → `"v28 training"`/`"v28 evaluation"`; the PASS message `"PASS: v27 ..."` → `"PASS: v28 ..."`.

- [ ] **Step 5: Update the notebook to v28**

Run this script (literal substring substitutions on the raw `.ipynb`, plus inserting the `--num_humans_range` arg after `--num_humans`):

```bash
python - <<'PY'
path='sncp_ppo_colab.ipynb'; raw=open(path,encoding='utf-8').read(); lines=raw.split('\n')
ins=0; out=[]
for ln in lines:
    out.append(ln)
    if "'--num_humans', '10'," in ln:                       # insert range right after num_humans
        out.append(ln.replace("'--num_humans', '10',", "'--num_humans_range', '10', '20',"))
        ins+=1
assert ins==1, ins
s='\n'.join(out)
s=s.replace('sncp_ppo_v27.pt','sncp_ppo_v28.pt')
s=s.replace('eval_v27','eval_v28')
s=s.replace("'--version', '27'","'--version', '28'")
s=s.replace("# v27 = v26 + paper Eq 11 pre-MLP (--pre_mlp), single-variable ablation. Trains in the paper CHALLENGING scenario",
            "# v28 = v27 (pre-MLP) + N~U(10,20) density curriculum (--num_humans_range), single-variable. Trains in the paper CHALLENGING scenario")
s=s.replace("## Current run: v27 - v26 + Eq 11 pre-MLP edge embedding (--pre_mlp), single-variable ablation",
            "## Current run: v28 - v27 + N~U(10,20) density curriculum (--num_humans_range), single-variable")
s=s.replace("## 3. Training (v27 - v26 + pre-MLP)","## 3. Training (v28 - v27 + density curriculum)")
s=s.replace("## 8. Notes & roadmap (current: v27 - v26 + pre-MLP)","## 8. Notes & roadmap (current: v28 - v27 + density curriculum)")
import json; json.loads(s); open(path,'w',encoding='utf-8',newline='\n').write(s)
print('ckpt v27 left:', 'sncp_ppo_v27.pt' in s, '| eval_v27 left:', 'eval_v27' in s, '| range:', "'--num_humans_range'" in s)
PY
```
Expected: `ckpt v27 left: False | eval_v27 left: False | range: True`

- [ ] **Step 6: Run the marker suite (green) + readiness preflight**

Run: `python -m pytest test_post_run_pipeline.py::test_notebook_is_v28_density_curriculum test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: PASS.

Run: `python -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready as v; s=v(Path('.')); print(s.status); [print(' ',n) for n in s.notes]"`
Expected: `pass` with `PASS: v28 Colab training and evaluation configuration is ready`.

- [ ] **Step 7: Commit**

```bash
git add test_post_run_pipeline.py test_v16_run_readiness.py sncp_ppo/run_readiness.py sncp_ppo_colab.ipynb
git commit -m "v28 notebook/readiness/tests: --num_humans_range 10 20 + v27->v28 bump"
```

---

### Task 3: Full suite + end-to-end density-curriculum smoke

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q --basetemp=./.pytmp`
Expected: all pass (v27 marker tests replaced by v28; +1 new file `test_density_curriculum.py`).

- [ ] **Step 2: End-to-end density-curriculum CLI smoke**

Run (tiny run with a narrow range; confirms the sampler + per-window env rebuild integrate without error and N actually varies):
```bash
python -m sncp_ppo.train --pre_mlp --num_humans_range 10 12 \
  --fixed_scenario paper_challenging --num_humans 10 --robot_vpref 1.0 \
  --holdout_scenarios paper_standard paper_challenging --holdout_episodes 1 \
  --total_steps 6144 --num_envs 2 --horizon 128 --bootstrap_easy_steps 0 \
  --eval_freq_updates 0 --seed 42 --save_path checkpoints/sncp_ppo_v28_smoke.pt 2>&1 | tee /tmp/v28_smoke.log | tail -20
echo "exit=${PIPESTATUS[0]}"
grep -c "Curriculum shift @ step .* paper_challenging/1[0-2]h" /tmp/v28_smoke.log || true
```
Expected: `exit=0`; at least one "Curriculum shift" line shows `paper_challenging/1{0,1,2}h` (N sampled in [10,12] and envs rebuilt). A `checkpoints/sncp_ppo_v28_smoke_final.pt` is written.

- [ ] **Step 3: Clean up smoke artifacts**

```bash
rm -f checkpoints/sncp_ppo_v28_smoke.pt checkpoints/sncp_ppo_v28_smoke_final.pt /tmp/v28_smoke.log
rm -rf ./.pytmp
```

- [ ] **Step 4: Commit (if any tracked changes remain; otherwise skip)**

```bash
git add -A && git commit -m "v28: full suite + density-curriculum smoke verified" || echo "nothing to commit"
```

---

### Task 4 (run-time, post-Colab — NOT code): evaluation protocol

- [ ] **Step 1: Stage artifacts** — `python stage_colab_run_artifacts.py --version 28`.
- [ ] **Step 2: Local multi-seed sweep**, IDENTICAL protocol to v26/v27: 5 seeds (100–500) × 50 ep at N=5/10/15/20, `paper_challenging`, robot 1.0, human 1.0, max_time None, goal_noise 0, on `sncp_ppo_v28.pt`. Record pooled success ± 95 % CI.
- [ ] **Step 3: Decision** — density curriculum helps gap (b) iff v28's CI clears v27's (93.6 / 80.0 / 70.8 / 59.6) at N=15 and N=20, with no regression at N=5/10. Confirm timeout stays 0. Write the verdict to memory (`sncp-paper-vs-impl.md`). If it helps, the remaining lever is `attn_count_scaling` (gap a residual); if not, density mismatch is ruled out and re-examine.

---

## Self-review

- **Spec coverage:** `--num_humans_range` flag (Task 1 Step 4); per-window sampling in `select_vectorized_phase` post-bootstrap branch (Task 1 Step 3); threaded into `_train_vectorized` both call sites (Task 1 Step 5); bootstrap N=1 unchanged + range=None byte-compatible (Task 1 tests); notebook keeps `--pre_mlp` + adds range, v28 paths (Task 2 Step 5); readiness v28 + range token (Task 2 Step 4); holdout untouched (no task changes `SCENARIO_HOLDOUT_CONFIG` or the eval_env — deliberate); version-marker tests (Task 2); end-to-end smoke (Task 3); local multi-seed eval (Task 4). All spec items covered.
- **Placeholder scan:** none — every code/command step is concrete.
- **Naming consistency:** flag `--num_humans_range` / attr `num_humans_range` / param `num_humans_range`; test names `test_notebook_is_v28_density_curriculum`, `test_v28_run_readiness_*`, `test_colab_persist_cell_downloads_eval_v28_artifact_bundle`; markers `sncp_ppo_v28.pt` / `eval_v28` / `'--version', '28'` — consistent across tasks.
- **Note:** all `pytest` runs use `--basetemp=./.pytmp` (the machine's default pytest temp dir `pytest-of-kor_a` has a Windows ACL that breaks `tmp_path`; the alternate basetemp avoids it). `.pytmp` is cleaned in Task 3 Step 3.
