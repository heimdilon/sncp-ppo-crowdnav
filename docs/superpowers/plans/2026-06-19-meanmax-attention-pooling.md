# v30 Mean+Max Attention Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the attention pool's pure convex-combination output with a cardinality-robust mean+max representation (`--meanmax_pool`), so the most-threatening agent's signal survives at high N — targeting the high-N collision gap vs the paper.

**Architecture:** `_attention_pool` keeps the attention-weighted mean and adds an element-wise max over humans, merged by `pool_merge = Linear(512→256)`. The layer is built only when `meanmax_pool=True`; `build_policy_for_checkpoint` auto-detects it from the `pool_merge` state-dict key, exactly like `pre_mlp` / `attn_count_scaling`. Single-variable on the v28 champion config.

**Tech Stack:** Python, PyTorch, ncps (LTC/AutoNCP), pytest, Jupyter notebook.

Spec: `docs/superpowers/specs/2026-06-19-meanmax-attention-pooling-design.md`. Branch: `feat/v30-meanmax-pooling`.

---

## File structure

- `sncp_ppo/models.py` — `meanmax_pool` ctor arg, `pool_merge` layer, `_attention_pool` branch, orthogonal init, `build_policy_for_checkpoint` auto-detect. (the model change)
- `sncp_ppo/train.py` — `--meanmax_pool` parser arg + thread through `build_or_load_policy`.
- `tests/test_meanmax_pool.py` (NEW) — model + CLI behaviour.
- `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` — version-marker bump v29→v30.
- `sncp_ppo/run_readiness.py` — v29→v30 markers + `--meanmax_pool` token (drop `--attn_count_scaling`).
- `sncp_ppo_colab.ipynb` — training cell `--meanmax_pool` (drop `--attn_count_scaling`) + v29→v30 paths.

No changes to `crowd_env.py`, `eval_report.py`, `ppo.py`.

All `pytest` runs use `--basetemp=./.pytmp` (the default `pytest-of-kor_a` temp dir has a Windows ACL that breaks `tmp_path`). Use the interpreter that has the full stack: `C:/ProgramData/miniconda3/python.exe` (bare `python` lacks `ncps`).

---

### Task 1: Mean+max pooling in models.py (TDD)

**Files:**
- Create: `tests/test_meanmax_pool.py`
- Modify: `sncp_ppo/models.py` (ctor `__init__:19-20`, init `_init_linear_weights:~136`, `_attention_pool:167-178`, `build_policy_for_checkpoint:266-276`)

- [ ] **Step 1: Write the failing tests (model-only)**

Create `tests/test_meanmax_pool.py`:

```python
"""High-N collision fix: the attention pool is a convex combination of per-human
features (value = M_rh, no W_v), so at high N the pooled vector regresses toward
the mean and the most-threatening agent's signal is diluted. meanmax_pool concats
the attention-weighted mean with an element-wise MAX over humans (cardinality-robust,
PointNet/DeepSet) through pool_merge = Linear(512->256). Default False keeps every
v14..v29 checkpoint byte-identical; build_policy_for_checkpoint auto-detects from the
pool_merge key (same pattern as pre_mlp / attn_count_scaling). The unit tests prove
the mechanism is WIRED and live; the high-N efficacy itself is validated empirically
by the Colab eval, not asserted here.
"""
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch, humans):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_default_off_no_pool_merge_and_checkpoint_compatible():
    default = SNCPPolicy()
    assert default.meanmax_pool is False
    assert not any(k.startswith('pool_merge') for k in default.state_dict())
    import os, pytest
    if not os.path.exists('checkpoints/sncp_ppo_v18.pt'):
        pytest.skip('milestone checkpoint checkpoints/sncp_ppo_v18.pt is git-ignored; present only locally')
    state = torch.load('checkpoints/sncp_ppo_v18.pt', map_location='cpu')
    policy = build_policy_for_checkpoint(state)
    policy.load_state_dict(state)
    assert policy.meanmax_pool is False


def test_meanmax_builds_pool_merge_and_is_autodetected(tmp_path):
    policy = SNCPPolicy(meanmax_pool=True)
    assert policy.meanmax_pool is True
    assert any(k.startswith('pool_merge') for k in policy.state_dict())

    path = tmp_path / 'meanmax.pt'
    torch.save(policy.state_dict(), path)
    state = torch.load(path, map_location='cpu')
    rebuilt = build_policy_for_checkpoint(state)
    assert rebuilt.meanmax_pool is True
    rebuilt.load_state_dict(state)  # must not raise


def test_forward_runs_and_action_bounded_with_meanmax():
    policy = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, meanmax_pool=True)
    h = policy.init_hidden(2, 10, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 10), h)
    assert mu.shape == (2, 2)
    assert torch.isfinite(mu).all() and torch.isfinite(std).all() and torch.isfinite(value).all()
    assert float(mu[:, 0].min()) >= 0.0 and float(mu[:, 0].max()) <= 1.0


def test_meanmax_changes_pooled_representation_vs_mean_only():
    """With shared encoder/attention weights, the ONLY difference is the max branch
    + merge. The pooled vector must differ from the mean-only pool — proving the new
    operation is live (not a no-op), tested directly on _attention_pool."""
    mean_only = SNCPPolicy(robot_vpref=1.0, meanmax_pool=False)
    meanmax = SNCPPolicy(robot_vpref=1.0, meanmax_pool=True)
    meanmax.load_state_dict(mean_only.state_dict(), strict=False)  # share W_q/W_k; pool_merge stays

    torch.manual_seed(0)
    M_rh = torch.randn(1, 12, 256)
    m_rr = torch.randn(1, 256)
    with torch.no_grad():
        u_mean = mean_only._attention_pool(M_rh, m_rr, num_humans=12)
        u_mm = meanmax._attention_pool(M_rh, m_rr, num_humans=12)
    assert u_mean.shape == u_mm.shape == (1, 256)
    assert not torch.allclose(u_mean, u_mm, atol=1e-4), "max branch had no effect on the pool"


def test_pre_mlp_and_meanmax_coexist():
    policy = SNCPPolicy(robot_vpref=1.0, pre_mlp=True, meanmax_pool=True)
    assert policy.pre_mlp is True and policy.meanmax_pool is True
    h = policy.init_hidden(2, 8, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 8), h)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_meanmax_pool.py -q --basetemp=./.pytmp`
Expected: FAIL — `SNCPPolicy.__init__() got an unexpected keyword argument 'meanmax_pool'`.

- [ ] **Step 3: Add the constructor arg + pool_merge layer**

In `sncp_ppo/models.py`, change the signature (line 19-20):

```python
    def __init__(self, robot_vpref=0.26, robot_wmax=1.8, pre_mlp=False,
                 attn_count_scaling=False, meanmax_pool=False):
```

After the `attn_count_scaling` block (right after line 35, before `# 1. Robot Node Encoder`), add:

```python
        # Mean+max attention pooling (v30): the default pool is a convex combination
        # Sum_h alpha_h * M_rh,h, which regresses to the mean at high N and dilutes the
        # most-threatening agent. meanmax_pool concats that mean with an element-wise
        # max over humans (cardinality-robust) and merges them with pool_merge. The
        # layer exists ONLY when on, so default checkpoints stay byte-identical and
        # build_policy_for_checkpoint can auto-detect the variant (pre_mlp pattern).
        self.meanmax_pool = meanmax_pool
```

In section `# 4. Attention Pooling weights` (after `self.W_k = nn.Linear(256, 64)`, line 91), add:

```python
        if meanmax_pool:
            self.pool_merge = nn.Linear(512, 256)
```

- [ ] **Step 4: Add the merge to `_attention_pool`**

Replace the return line of `_attention_pool` (line 178) so the method ends:

```python
        alpha = F.softmax(attn_scores, dim=1)   # [B, H, 1]
        a_mean = torch.bmm(M_rh.transpose(1, 2), alpha).squeeze(2)  # [B, 256]
        if not self.meanmax_pool:
            return a_mean
        a_max = M_rh.max(dim=1).values          # [B, 256] cardinality-robust
        return self.pool_merge(torch.cat([a_mean, a_max], dim=1))  # [B, 256]
```

- [ ] **Step 5: Orthogonal-init pool_merge**

In `_init_linear_weights`, after `_orthogonal_linear(self.node_proj, gain=sqrt2)` (line 138), add:

```python
        if self.meanmax_pool:
            _orthogonal_linear(self.pool_merge, gain=sqrt2)
```

- [ ] **Step 6: Auto-detect in build_policy_for_checkpoint**

In `build_policy_for_checkpoint` (lines 273-276), add the detection and pass it:

```python
    pre_mlp = any(key.startswith('temporal_pre_mlp') for key in state_dict)
    attn_count_scaling = '_attn_count_scaling' in state_dict
    meanmax_pool = any(key.startswith('pool_merge') for key in state_dict)
    return SNCPPolicy(robot_vpref=robot_vpref, robot_wmax=robot_wmax,
                      pre_mlp=pre_mlp, attn_count_scaling=attn_count_scaling,
                      meanmax_pool=meanmax_pool)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_meanmax_pool.py -q --basetemp=./.pytmp`
Expected: PASS (5 passed; the v18-compat test passes locally, would skip on a fresh clone).

- [ ] **Step 8: Commit**

```bash
git add sncp_ppo/models.py tests/test_meanmax_pool.py
git commit -m "v30: mean+max attention pooling (models.py) + tests"
```

---

### Task 2: --meanmax_pool CLI flag (TDD)

**Files:**
- Modify: `tests/test_meanmax_pool.py` (add CLI test)
- Modify: `sncp_ppo/train.py` (`build_or_load_policy:261-266`, parser `~1101`)

- [ ] **Step 1: Add the failing CLI test**

Append to `tests/test_meanmax_pool.py`:

```python
def test_train_cli_and_build_thread_the_flag():
    import argparse
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.train import build_or_load_policy, build_parser

    assert build_parser().parse_args(['--meanmax_pool']).meanmax_pool is True
    assert build_parser().parse_args([]).meanmax_pool is False

    env = CrowdSimEnv(num_humans=3, scenario='hard', robot_vpref=1.0)
    args = argparse.Namespace(init_checkpoint=None, pre_mlp=False,
                              attn_count_scaling=False, meanmax_pool=True)
    policy = build_or_load_policy(args, env, torch.device('cpu'))
    assert policy.meanmax_pool is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_meanmax_pool.py::test_train_cli_and_build_thread_the_flag -q --basetemp=./.pytmp`
Expected: FAIL — `unrecognized arguments: --meanmax_pool` (or AttributeError on `.meanmax_pool`).

- [ ] **Step 3: Thread the flag through build_or_load_policy**

In `sncp_ppo/train.py`, in the fresh-policy `return SNCPPolicy(...)` (lines 261-266), add the kwarg:

```python
    return SNCPPolicy(
        robot_vpref=env.robot_vpref,
        robot_wmax=env.robot_wmax,
        pre_mlp=getattr(args, 'pre_mlp', False),
        attn_count_scaling=getattr(args, 'attn_count_scaling', False),
        meanmax_pool=getattr(args, 'meanmax_pool', False),
    ).to(device)
```

- [ ] **Step 4: Add the parser argument**

In `build_parser`, after the `--attn_count_scaling` argument (ends line 1105), add:

```python
    parser.add_argument('--meanmax_pool', action='store_true',
                        help='Mean+max attention pooling (v30): concat the attention-weighted '
                             'mean with an element-wise max over humans, merged by Linear(512->256). '
                             'Cardinality-robust fix for the high-N convex-combination washout. '
                             'Default off preserves v14..v29 architecture and checkpoint compatibility.')
```

- [ ] **Step 5: Run it to verify it passes**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_meanmax_pool.py -q --basetemp=./.pytmp`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo/train.py tests/test_meanmax_pool.py
git commit -m "v30: --meanmax_pool CLI flag threaded through build_or_load_policy"
```

---

### Task 3: Full suite + end-to-end smoke

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest -q --basetemp=./.pytmp`
Expected: all pass (206 + the new `test_meanmax_pool.py` cases; no regressions).

- [ ] **Step 2: End-to-end smoke — construct (pre_mlp + meanmax) → save → auto-detect → eval**

Run:
```bash
C:/ProgramData/miniconda3/python.exe -c "
import torch
from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
from sncp_ppo.eval_report import evaluate_density
p = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, pre_mlp=True, meanmax_pool=True)
torch.save(p.state_dict(), 'checkpoints/sncp_ppo_v30_smoke.pt')
sd = torch.load('checkpoints/sncp_ppo_v30_smoke.pt', map_location='cpu')
p2 = build_policy_for_checkpoint(sd, robot_vpref=1.0, robot_wmax=1.8)
assert p2.pre_mlp is True and p2.meanmax_pool is True, 'auto-detect failed'
p2.load_state_dict(sd)
r = evaluate_density(checkpoint_path='checkpoints/sncp_ppo_v30_smoke.pt', num_humans=5, scenario='paper_challenging', n_episodes=2, seed=100, robot_vpref=1.0, human_vpref_override=1.0)
print('SMOKE OK pre_mlp=', p2.pre_mlp, 'meanmax=', p2.meanmax_pool, 'episodes=', len(r))
"
```
Expected: `SMOKE OK pre_mlp= True meanmax= True episodes= 2` (both flags auto-detect together; eval forwards through the merged pool).

- [ ] **Step 3: Tiny CLI training smoke with the v30 flag set**

Run:
```bash
C:/ProgramData/miniconda3/python.exe -m sncp_ppo.train --pre_mlp --meanmax_pool --num_humans_range 10 12 \
  --fixed_scenario paper_challenging --num_humans 10 --robot_vpref 1.0 \
  --holdout_scenarios paper_standard paper_challenging --holdout_episodes 1 \
  --total_steps 4096 --num_envs 2 --horizon 128 --bootstrap_easy_steps 0 \
  --seed 42 --save_path checkpoints/sncp_ppo_v30_clismoke.pt > /tmp/v30_smoke.log 2>&1
echo "exit=$?"; tail -3 /tmp/v30_smoke.log
```
Expected: `exit=0` and "Vectorized training completed!" (the train forward path runs with pre_mlp + meanmax + density curriculum).

- [ ] **Step 4: Clean up smoke artifacts**

```bash
rm -f checkpoints/sncp_ppo_v30_smoke.pt checkpoints/sncp_ppo_v30_clismoke.pt checkpoints/sncp_ppo_v30_clismoke_final.pt /tmp/v30_smoke.log
rm -rf ./.pytmp
```
(Also delete the newest `logs/training_*.csv` the CLI smoke wrote this session.)

- [ ] **Step 5: Commit (only if tracked changes remain; else skip)**

```bash
git add -A && git commit -m "v30: full suite + end-to-end pre_mlp+meanmax smoke verified" || echo "nothing to commit"
```

---

### Task 4: Version-marker tests v29→v30 (TDD red)

**Files:**
- Modify: `tests/test_post_run_pipeline.py` (replace `test_notebook_is_v29_attn_count_scaling`, lines 266-285)
- Modify: `tests/test_v16_run_readiness.py` (rename the 3 v29 tests + asserts)

- [ ] **Step 1: Replace the notebook-faithfulness test**

In `tests/test_post_run_pipeline.py`, replace `test_notebook_is_v29_attn_count_scaling` with:

```python
def test_notebook_is_v30_meanmax_pool():
    # v30 = v28 (pre-MLP + density curriculum) + mean+max attention pooling ONLY.
    # The defining change is --meanmax_pool; --pre_mlp and --num_humans_range stay;
    # v29's --attn_count_scaling is DROPPED (v30 builds on the v28 champion, not v29).
    code = _colab_code_sources()
    train_cells = [s for s in code if "sncp_ppo.train" in s and "--fixed_scenario" in s]
    eval_cells = [s for s in code if "run_post_eval.py" in s]
    assert len(train_cells) == 1 and len(eval_cells) == 1
    train, ev = train_cells[0], eval_cells[0]
    assert "paper_challenging" in train
    assert "checkpoints/sncp_ppo_v30.pt" in train
    assert "'--meanmax_pool'" in train               # the v30 change
    assert "'--pre_mlp'" in train                     # v27 carried forward
    assert "'--num_humans_range'" in train            # v28 carried forward
    assert "'--attn_count_scaling'" not in train      # v29 dropped
    for tok in ("TOTAL_STEPS = 2_500_000", "SEED = 42", "'--robot_vpref', '1.0'",
                "'--holdout_episodes', '50'"):
        assert tok in train, tok
    assert "'--version', '30'" in ev
    assert "'--baseline_nav_steps', '32'" in ev
    assert "'--max_time'" not in ev
```

- [ ] **Step 2: Update the readiness tests**

In `tests/test_v16_run_readiness.py`:
- `def test_v29_run_readiness_passes_current_repo():` → `def test_v30_run_readiness_passes_current_repo():` (body unchanged).
- `def test_v29_run_readiness_flags_stale_notebook(tmp_path):` → `def test_v30_run_readiness_flags_stale_notebook(tmp_path):`; change the comment to `# A pre-v30 notebook (v23..v29 markers) must be flagged: the v30 cells are absent.` and the two note asserts:

```python
    assert any("v30 training" in note for note in summary.notes)
    assert any("v30 evaluation" in note for note in summary.notes)
```

- `def test_colab_persist_cell_downloads_eval_v29_artifact_bundle():` → `def test_colab_persist_cell_downloads_eval_v30_artifact_bundle():`; the two asserts:

```python
    assert "'eval_v30_artifacts'" in persist_cell
    assert "'eval_v30'" in persist_cell
```

- [ ] **Step 3: Run to verify they fail**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py::test_notebook_is_v30_meanmax_pool tests/test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: FAIL (notebook/readiness still v29).

- [ ] **Step 4: Commit**

```bash
git add tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py
git commit -m "v30 tests: bump version markers to v30 + assert --meanmax_pool (red)"
```

---

### Task 5: Update run-readiness checker to v30

**Files:**
- Modify: `sncp_ppo/run_readiness.py`

- [ ] **Step 1: Update the module comment (lines 11-14)**

Replace the 4-line `# v29 = ...` block with:

```python
# v30 = v28 (pre-MLP + density curriculum) + mean+max attention pooling
# (--meanmax_pool), single-variable on the v28 champion. v29's --attn_count_scaling is
# dropped. Budget (challenging 50s, standard 12.5s), 8m crossing and normalized comfort
# stay env-DERIVED from --fixed_scenario paper_challenging. Change vs v28 is --meanmax_pool.
```

- [ ] **Step 2: Swap the training token (lines 32-34)**

In `TRAINING_TOKENS`, replace `"'--attn_count_scaling'",` with `"'--meanmax_pool'",` (keep `"'--pre_mlp'",` and `"'--num_humans_range'",`).

- [ ] **Step 3: Bump every v29→v30 path/version string**

Replace every `sncp_ppo_v29.pt` → `sncp_ppo_v30.pt`; every `eval_v29` → `eval_v30`; the `"'--version', '29'"` token → `"'--version', '30'"`; every `"v29 training"` → `"v30 training"`, `"v29 evaluation"` → `"v30 evaluation"`; the PASS message → `"PASS: v30 Colab training and evaluation configuration is ready"`.

- [ ] **Step 4: Commit**

```bash
git add sncp_ppo/run_readiness.py
git commit -m "v30 readiness: v30 markers + --meanmax_pool token (drop attn_count_scaling)"
```

---

### Task 6: Update the notebook to v30 + green

**Files:**
- Modify: `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Apply the v30 edits**

Run (drops `--attn_count_scaling`, inserts `--meanmax_pool` after `--pre_mlp`, then v29→v30 substitutions):

```bash
C:/ProgramData/miniconda3/python.exe - <<'PY'
path='sncp_ppo_colab.ipynb'; raw=open(path,encoding='utf-8').read(); lines=raw.split('\n')
out=[]; removed=0; inserted=0
for ln in lines:
    if "'--attn_count_scaling'," in ln:                       # drop v29's flag
        removed+=1
        continue
    out.append(ln)
    if "'--pre_mlp'," in ln:                                  # insert meanmax after pre_mlp
        out.append(ln.replace("'--pre_mlp',", "'--meanmax_pool',"))
        inserted+=1
assert removed==1 and inserted==1, (removed, inserted)
s='\n'.join(out)
s=s.replace('sncp_ppo_v29.pt','sncp_ppo_v30.pt')
s=s.replace('eval_v29','eval_v30')
s=s.replace("'--version', '29'","'--version', '30'")
s=s.replace("v29 - v28 + Eq 13 attention count-scaling (--attn_count_scaling)","v30 - v28 + mean+max attention pooling (--meanmax_pool)")
s=s.replace("## 3. Training (v29 - v28 + attn count-scaling)","## 3. Training (v30 - v28 + mean+max pooling)")
s=s.replace("## 8. Notes & roadmap (current: v29 - v28 + attn count-scaling)","## 8. Notes & roadmap (current: v30 - v28 + mean+max pooling)")
import json; json.loads(s); open(path,'w',encoding='utf-8',newline='\n').write(s)
print('removed attn:', removed, '| meanmax:', "'--meanmax_pool'" in s, '| pre_mlp:', "'--pre_mlp'" in s,
      '| range:', "'--num_humans_range'" in s, '| no attn:', "'--attn_count_scaling'" not in s,
      '| v30:', "'--version', '30'" in s, '| no v29 ckpt:', 'sncp_ppo_v29.pt' not in s)
PY
```
Expected: `removed attn: 1 | meanmax: True | pre_mlp: True | range: True | no attn: True | v30: True | no v29 ckpt: True`.

- [ ] **Step 2: Run the marker suite (green) + readiness preflight**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py::test_notebook_is_v30_meanmax_pool tests/test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: PASS.

Run: `C:/ProgramData/miniconda3/python.exe -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready as v; s=v(Path('.')); print(s.status); [print(' ',n) for n in s.notes]"`
Expected: `pass` with `PASS: v30 Colab training and evaluation configuration is ready`.

- [ ] **Step 3: Full suite once more**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest -q --basetemp=./.pytmp`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add sncp_ppo_colab.ipynb
git commit -m "v30 notebook: add --meanmax_pool + bump v29->v30 paths (drop attn_count_scaling)"
```

---

### Task 7 (run-time, post-Colab — NOT code): evaluation protocol

- [ ] **Step 1: Merge + push** — finishing-a-development-branch merges `feat/v30-meanmax-pooling` into `main` and pushes (Colab pulls `main` to train v30).
- [ ] **Step 2: Train on Colab** — pull `main`, run the v30 training cell (~3-4 h A100), download/copy `sncp_ppo_v30.pt` to the repo root (Colab eval cell may be skipped; the checkpoint is enough).
- [ ] **Step 3: Honest local multi-seed sweep** — copy `scratch/_sweep_v29.py` → `scratch/_sweep_v30.py`, point `CKPT='sncp_ppo_v30.pt'` and `OUT='v30_multiseed_result.json'`; run with `C:/ProgramData/miniconda3/python.exe scratch/_sweep_v30.py` (5 seeds 100–500 × 50 ep at N=5/10/15/20, `paper_challenging`, robot 1.0, human 1.0, max_time None, goal_noise 0).
- [ ] **Step 4: Decide** — extend `scratch/_analyze_v29.py` to load v30; mean+max helps iff high-N **collision drops** and success rises (esp. N=15/20) with no regression at N=5/10 and timeout 0, vs the v28 baseline (94.4/87.6/79.2/73.2) with Wilson CIs + two-proportion z (Bonferroni α=0.0125). Report a flat/negative result honestly. Write the verdict to memory (`sncp-paper-vs-impl.md`); then experiment #2 (capacity) follows.

---

## Self-review

- **Spec coverage:** architecture mean+max + pool_merge (Task 1 Steps 3-4); orthogonal init (Task 1 Step 5); auto-detect (Task 1 Step 6); `--meanmax_pool` CLI + build wiring (Task 2); washout/mechanism + compat + roundtrip + forward + coexistence tests (Task 1 Step 1, Task 2 Step 1); single-variable v30 config drops `--attn_count_scaling` (Tasks 4-6); run-readiness + notebook + version-marker bump (Tasks 4-6); end-to-end auto-detect + eval + train smoke (Task 3); honest multi-seed eval + decision rule (Task 7). All spec items covered.
- **Placeholder scan:** none — every code/command step is concrete.
- **Type/name consistency:** flag `--meanmax_pool`; attr `meanmax_pool`; layer `pool_merge` (detected via `startswith('pool_merge')`); method `_attention_pool`; tests `test_meanmax_pool.py`; markers `sncp_ppo_v30.pt` / `eval_v30` / `'--version', '30'`; readiness/notebook drop `--attn_count_scaling` consistently — aligned across tasks.
