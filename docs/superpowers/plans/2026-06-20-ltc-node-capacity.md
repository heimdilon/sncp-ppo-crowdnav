# v31 Node-Fusion LTC Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen the node-fusion NCP from `AutoNCP(128, 48)` to `AutoNCP(256, 96)` (parametrized + auto-detected), built on the v30 champion, to test whether decision-integration capacity is the remaining high-N collision bottleneck.

**Architecture:** Add `node_units`/`node_output` constructor args (defaults 128/48) so the node `AutoNCP` is sized from them; `build_policy_for_checkpoint` infers the size from `node_ltc.rnn_cell.gleak` (→units) and `node_ltc.rnn_cell.output_w` (→output_size), so every v14–v30 checkpoint still loads. Single-variable on v30 (pre_mlp + meanmax + density curriculum).

**Tech Stack:** Python, PyTorch, ncps (LTC/AutoNCP), pytest, Jupyter notebook.

Spec: `docs/superpowers/specs/2026-06-20-ltc-node-capacity-design.md`. Branch: `feat/v31-node-capacity`.

---

## File structure

- `sncp_ppo/models.py` — `node_units`/`node_output` ctor args; `node_wiring` built from them; `build_policy_for_checkpoint` node-size inference. (the model change)
- `sncp_ppo/train.py` — `--node_units`/`--node_output` parser args + thread through `build_or_load_policy`.
- `tests/test_node_capacity.py` (NEW) — model + CLI behaviour.
- `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` — version-marker bump v30→v31.
- `sncp_ppo/run_readiness.py` — v30→v31 markers + `--node_units`/`--node_output` tokens (keep `--meanmax_pool`).
- `sncp_ppo_colab.ipynb` — training cell `--node_units 256 --node_output 96` (keep `--meanmax_pool`) + v30→v31 paths.

No changes to `crowd_env.py`, `eval_report.py`, `ppo.py`.

All `pytest` runs use `--basetemp=./.pytmp` and the interpreter `C:/ProgramData/miniconda3/python.exe` (bare `python` lacks `ncps`).

---

### Task 1: Parametrized node capacity in models.py (TDD)

**Files:**
- Create: `tests/test_node_capacity.py`
- Modify: `sncp_ppo/models.py` (ctor signature line 19-20; node section line ~98; `build_policy_for_checkpoint` line ~273-276)

- [ ] **Step 1: Write the failing tests (model-only)**

Create `tests/test_node_capacity.py`:

```python
"""High-N collision, experiment #2: widen the node-fusion NCP (the 640->128 decision
bottleneck) from AutoNCP(128,48) to AutoNCP(256,96), built on v30 (mean+max). node_units/
node_output are constructor args (defaults 128/48); build_policy_for_checkpoint infers them
from node_ltc.rnn_cell.gleak (units) and output_w (output_size), so v14..v30 checkpoints still
load. The unit tests prove the size is wired + auto-detected; the high-N efficacy is validated
empirically by the Colab eval, not asserted here.
"""
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch, humans):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_default_node_size_unchanged_and_compatible():
    default = SNCPPolicy()
    assert default.node_units == 128 and default.node_output == 48
    assert default.node_wiring.units == 128
    import os, pytest
    if not os.path.exists('checkpoints/sncp_ppo_v18.pt'):
        pytest.skip('milestone checkpoint checkpoints/sncp_ppo_v18.pt is git-ignored; present only locally')
    state = torch.load('checkpoints/sncp_ppo_v18.pt', map_location='cpu')
    policy = build_policy_for_checkpoint(state)
    policy.load_state_dict(state)
    assert policy.node_units == 128


def test_widened_node_builds():
    policy = SNCPPolicy(node_units=256, node_output=96)
    assert policy.node_units == 256 and policy.node_output == 96
    assert policy.node_wiring.units == 256
    assert tuple(policy.node_proj.weight.shape) == (256, 96)


def test_forward_runs_and_action_bounded_widened():
    policy = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, node_units=256, node_output=96)
    h = policy.init_hidden(2, 10, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 10), h)
    assert mu.shape == (2, 2)
    assert torch.isfinite(mu).all() and torch.isfinite(std).all() and torch.isfinite(value).all()
    assert float(mu[:, 0].min()) >= 0.0 and float(mu[:, 0].max()) <= 1.0


def test_widened_node_is_autodetected(tmp_path):
    policy = SNCPPolicy(node_units=256, node_output=96)
    path = tmp_path / 'node256.pt'
    torch.save(policy.state_dict(), path)
    state = torch.load(path, map_location='cpu')
    rebuilt = build_policy_for_checkpoint(state)
    assert rebuilt.node_units == 256 and rebuilt.node_output == 96
    rebuilt.load_state_dict(state)  # must not raise


def test_default_node_state_dict_infers_128(tmp_path):
    policy = SNCPPolicy()  # node 128/48
    path = tmp_path / 'node128.pt'
    torch.save(policy.state_dict(), path)
    state = torch.load(path, map_location='cpu')
    rebuilt = build_policy_for_checkpoint(state)
    assert rebuilt.node_units == 128 and rebuilt.node_output == 48
    rebuilt.load_state_dict(state)  # must not raise


def test_node_capacity_coexists_with_premlp_and_meanmax():
    policy = SNCPPolicy(robot_vpref=1.0, pre_mlp=True, meanmax_pool=True,
                        node_units=256, node_output=96)
    assert policy.pre_mlp and policy.meanmax_pool and policy.node_units == 256
    h = policy.init_hidden(2, 8, torch.device('cpu'))
    mu, std, value, _ = policy(_obs(2, 8), h)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_node_capacity.py -q --basetemp=./.pytmp`
Expected: FAIL — `SNCPPolicy.__init__() got an unexpected keyword argument 'node_units'`.

- [ ] **Step 3: Add the constructor args**

In `sncp_ppo/models.py`, change the signature (lines 19-20):

```python
    def __init__(self, robot_vpref=0.26, robot_wmax=1.8, pre_mlp=False,
                 attn_count_scaling=False, meanmax_pool=False, node_units=128, node_output=48):
```

- [ ] **Step 4: Size the node wiring from the args**

In `sncp_ppo/models.py`, in section `# 5. Node NCP Encoder`, replace the fixed line:

```python
        self.node_wiring = AutoNCP(units=128, output_size=48, seed=48203)
```

with:

```python
        self.node_units, self.node_output = node_units, node_output
        self.node_wiring = AutoNCP(units=node_units, output_size=node_output, seed=48203)
```

(`node_ltc = LTC(input_size=640, units=self.node_wiring)` and `node_proj = nn.Linear(self.node_wiring.output_dim, 256)` already follow `node_wiring`, so they resize automatically.)

- [ ] **Step 5: Auto-detect node size in build_policy_for_checkpoint**

In `build_policy_for_checkpoint` (lines ~273-276, currently inferring pre_mlp/attn_count_scaling/meanmax_pool), replace the body with:

```python
    pre_mlp = any(key.startswith('temporal_pre_mlp') for key in state_dict)
    attn_count_scaling = '_attn_count_scaling' in state_dict
    meanmax_pool = any(key.startswith('pool_merge') for key in state_dict)
    gleak = state_dict.get('node_ltc.rnn_cell.gleak')
    node_units = int(gleak.shape[0]) if gleak is not None else 128
    out_w = state_dict.get('node_ltc.rnn_cell.output_w')
    node_output = int(out_w.shape[0]) if out_w is not None else 48
    return SNCPPolicy(robot_vpref=robot_vpref, robot_wmax=robot_wmax,
                      pre_mlp=pre_mlp, attn_count_scaling=attn_count_scaling,
                      meanmax_pool=meanmax_pool, node_units=node_units, node_output=node_output)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_node_capacity.py -q --basetemp=./.pytmp`
Expected: PASS (6 passed; the v18-compat test passes locally, would skip on a fresh clone).

- [ ] **Step 7: Commit**

```bash
git add sncp_ppo/models.py tests/test_node_capacity.py
git commit -m "v31: parametrized node-fusion capacity + auto-detect (models.py) + tests"
```

---

### Task 2: --node_units / --node_output CLI flags (TDD)

**Files:**
- Modify: `tests/test_node_capacity.py` (add CLI test)
- Modify: `sncp_ppo/train.py` (`build_or_load_policy` fresh-policy return; parser after `--meanmax_pool`)

- [ ] **Step 1: Add the failing CLI test**

Append to `tests/test_node_capacity.py`:

```python
def test_train_cli_and_build_thread_node_size():
    import argparse
    from crowd_sim.crowd_env import CrowdSimEnv
    from sncp_ppo.train import build_or_load_policy, build_parser

    a = build_parser().parse_args(['--node_units', '256', '--node_output', '96'])
    assert a.node_units == 256 and a.node_output == 96
    d = build_parser().parse_args([])
    assert d.node_units == 128 and d.node_output == 48

    env = CrowdSimEnv(num_humans=3, scenario='hard', robot_vpref=1.0)
    args = argparse.Namespace(init_checkpoint=None, pre_mlp=False, attn_count_scaling=False,
                              meanmax_pool=False, node_units=256, node_output=96)
    policy = build_or_load_policy(args, env, torch.device('cpu'))
    assert policy.node_units == 256 and policy.node_output == 96
```

- [ ] **Step 2: Run it to verify it fails**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_node_capacity.py::test_train_cli_and_build_thread_node_size -q --basetemp=./.pytmp`
Expected: FAIL — `unrecognized arguments: --node_units 256 --node_output 96`.

- [ ] **Step 3: Thread the flags through build_or_load_policy**

In `sncp_ppo/train.py`, in the fresh-policy `return SNCPPolicy(...)` (the block that already passes `meanmax_pool=getattr(args, 'meanmax_pool', False),`), add two kwargs so it reads:

```python
    return SNCPPolicy(
        robot_vpref=env.robot_vpref,
        robot_wmax=env.robot_wmax,
        pre_mlp=getattr(args, 'pre_mlp', False),
        attn_count_scaling=getattr(args, 'attn_count_scaling', False),
        meanmax_pool=getattr(args, 'meanmax_pool', False),
        node_units=getattr(args, 'node_units', 128),
        node_output=getattr(args, 'node_output', 48),
    ).to(device)
```

- [ ] **Step 4: Add the parser arguments**

In `build_parser`, immediately after the `--meanmax_pool` argument block, add:

```python
    parser.add_argument('--node_units', type=int, default=128,
                        help='Node-fusion NCP total neuron count (v31 capacity experiment; default '
                             '128 preserves v14..v30). Auto-detected from the checkpoint on load.')
    parser.add_argument('--node_output', type=int, default=48,
                        help='Node-fusion NCP motor (output) neuron count; must be < --node_units '
                             '(default 48; v31 uses 96 for units=256). Auto-detected on load.')
```

- [ ] **Step 5: Run it to verify it passes**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_node_capacity.py -q --basetemp=./.pytmp`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo/train.py tests/test_node_capacity.py
git commit -m "v31: --node_units/--node_output CLI flags threaded through build_or_load_policy"
```

---

### Task 3: Full suite + end-to-end smoke

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest -q --basetemp=./.pytmp`
Expected: all pass (prior count + the new `test_node_capacity.py` cases; no regressions).

- [ ] **Step 2: End-to-end smoke — construct (pre_mlp + meanmax + node256) → save → auto-detect → eval**

Run:
```bash
C:/ProgramData/miniconda3/python.exe -c "
import torch
from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint
from sncp_ppo.eval_report import evaluate_density
p = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8, pre_mlp=True, meanmax_pool=True, node_units=256, node_output=96)
torch.save(p.state_dict(), 'checkpoints/sncp_ppo_v31_smoke.pt')
sd = torch.load('checkpoints/sncp_ppo_v31_smoke.pt', map_location='cpu')
p2 = build_policy_for_checkpoint(sd, robot_vpref=1.0, robot_wmax=1.8)
assert p2.pre_mlp and p2.meanmax_pool and p2.node_units==256 and p2.node_output==96, 'auto-detect failed'
p2.load_state_dict(sd)
r = evaluate_density(checkpoint_path='checkpoints/sncp_ppo_v31_smoke.pt', num_humans=5, scenario='paper_challenging', n_episodes=2, seed=100, robot_vpref=1.0, human_vpref_override=1.0)
print('SMOKE OK node_units=', p2.node_units, 'node_output=', p2.node_output, 'episodes=', len(r))
"
```
Expected: `SMOKE OK node_units= 256 node_output= 96 episodes= 2` (confirms `AutoNCP(256,96)` is valid, all three variants auto-detect together, and eval forwards through the wider node).

- [ ] **Step 3: Tiny CLI training smoke with the v31 flags**

Run:
```bash
C:/ProgramData/miniconda3/python.exe -m sncp_ppo.train --pre_mlp --meanmax_pool --node_units 256 --node_output 96 --num_humans_range 10 12 \
  --fixed_scenario paper_challenging --num_humans 10 --robot_vpref 1.0 \
  --holdout_scenarios paper_standard paper_challenging --holdout_episodes 1 \
  --total_steps 4096 --num_envs 2 --horizon 128 --bootstrap_easy_steps 0 \
  --seed 42 --save_path checkpoints/sncp_ppo_v31_clismoke.pt > /tmp/v31_smoke.log 2>&1
echo "exit=$?"; tail -3 /tmp/v31_smoke.log
```
Expected: `exit=0` and "Vectorized training completed!" (the 640→256 node LTC trains with all flags).

- [ ] **Step 4: Clean up smoke artifacts**

```bash
rm -f checkpoints/sncp_ppo_v31_smoke.pt checkpoints/sncp_ppo_v31_clismoke.pt checkpoints/sncp_ppo_v31_clismoke_final.pt /tmp/v31_smoke.log
rm -rf ./.pytmp
```
(Also delete the newest `logs/training_*.csv` the CLI smoke wrote this session.)

- [ ] **Step 5: Commit (only if tracked changes remain; else skip)**

```bash
git add -A && git commit -m "v31: full suite + end-to-end node256 smoke verified" || echo "nothing to commit"
```

---

### Task 4: Version-marker tests v30→v31 (TDD red)

**Files:**
- Modify: `tests/test_post_run_pipeline.py` (replace `test_notebook_is_v30_meanmax_pool`)
- Modify: `tests/test_v16_run_readiness.py` (rename the 3 v30 tests + asserts)

- [ ] **Step 1: Replace the notebook-faithfulness test**

In `tests/test_post_run_pipeline.py`, replace `test_notebook_is_v30_meanmax_pool` with:

```python
def test_notebook_is_v31_node_capacity():
    # v31 = v30 (pre-MLP + density curriculum + mean+max) + node-fusion capacity 256/96 ONLY.
    # The defining change is --node_units/--node_output; --meanmax_pool/--pre_mlp/--num_humans_range stay.
    code = _colab_code_sources()
    train_cells = [s for s in code if "sncp_ppo.train" in s and "--fixed_scenario" in s]
    eval_cells = [s for s in code if "run_post_eval.py" in s]
    assert len(train_cells) == 1 and len(eval_cells) == 1
    train, ev = train_cells[0], eval_cells[0]
    assert "paper_challenging" in train
    assert "checkpoints/sncp_ppo_v31.pt" in train
    assert "'--node_units', '256'" in train              # the v31 change
    assert "'--node_output', '96'" in train              # the v31 change
    assert "'--meanmax_pool'" in train                   # v30 carried forward
    assert "'--pre_mlp'" in train                         # v27 carried forward
    assert "'--num_humans_range'" in train                # v28 carried forward
    for tok in ("TOTAL_STEPS = 2_500_000", "SEED = 42", "'--robot_vpref', '1.0'",
                "'--holdout_episodes', '50'"):
        assert tok in train, tok
    assert "'--version', '31'" in ev
    assert "'--baseline_nav_steps', '32'" in ev
    assert "'--max_time'" not in ev
```

- [ ] **Step 2: Update the readiness tests**

In `tests/test_v16_run_readiness.py`:
- `def test_v30_run_readiness_passes_current_repo():` → `def test_v31_run_readiness_passes_current_repo():` (body unchanged).
- `def test_v30_run_readiness_flags_stale_notebook(tmp_path):` → `def test_v31_run_readiness_flags_stale_notebook(tmp_path):`; change the comment to `# A pre-v31 notebook (v23..v30 markers) must be flagged: the v31 cells are absent.` and the two note asserts:

```python
    assert any("v31 training" in note for note in summary.notes)
    assert any("v31 evaluation" in note for note in summary.notes)
```

- `def test_colab_persist_cell_downloads_eval_v30_artifact_bundle():` → `def test_colab_persist_cell_downloads_eval_v31_artifact_bundle():`; the two asserts:

```python
    assert "'eval_v31_artifacts'" in persist_cell
    assert "'eval_v31'" in persist_cell
```

- [ ] **Step 3: Run to verify they fail**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py::test_notebook_is_v31_node_capacity tests/test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: FAIL (notebook/readiness still v30).

- [ ] **Step 4: Commit**

```bash
git add tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py
git commit -m "v31 tests: bump version markers to v31 + assert --node_units/--node_output (red)"
```

---

### Task 5: Update run-readiness checker to v31

**Files:**
- Modify: `sncp_ppo/run_readiness.py`

- [ ] **Step 1: Update the module comment**

Replace the `# v30 = ...` comment block (the 4 lines starting `# v30 =`) with:

```python
# v31 = v30 (pre-MLP + density curriculum + mean+max pooling) + node-fusion NCP capacity
# 256/96 (--node_units 256 --node_output 96), single-variable. Budget (challenging 50s,
# standard 12.5s), 8m crossing and normalized comfort stay env-DERIVED from --fixed_scenario
# paper_challenging. Change vs v30 is the node capacity + the v31 save path.
```

- [ ] **Step 2: Add the node tokens**

In `TRAINING_TOKENS`, after the `"'--meanmax_pool'",` line, add:

```python
    "'--meanmax_pool'",
    "'--node_units', '256'",
    "'--node_output', '96'",
```

- [ ] **Step 3: Bump every v30→v31 path/version string**

Replace every `sncp_ppo_v30.pt` → `sncp_ppo_v31.pt`; every `eval_v30` → `eval_v31`; the `"'--version', '30'"` token → `"'--version', '31'"`; every `"v30 training"` → `"v31 training"`, `"v30 evaluation"` → `"v31 evaluation"`; the PASS message → `"PASS: v31 Colab training and evaluation configuration is ready"`.

- [ ] **Step 4: Commit**

```bash
git add sncp_ppo/run_readiness.py
git commit -m "v31 readiness: v31 markers + --node_units/--node_output tokens"
```

---

### Task 6: Update the notebook to v31 + green

**Files:**
- Modify: `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Apply the v31 edits**

Run (clones the `--meanmax_pool` arg line twice to insert `--node_units 256` / `--node_output 96` with the same JSON wrapping, then v30→v31 substitutions):

```bash
C:/ProgramData/miniconda3/python.exe - <<'PY'
path='sncp_ppo_colab.ipynb'; raw=open(path,encoding='utf-8').read(); lines=raw.split('\n')
out=[]; inserted=0
for ln in lines:
    out.append(ln)
    if "'--meanmax_pool'," in ln:                              # insert node flags after meanmax
        out.append(ln.replace("'--meanmax_pool',", "'--node_units', '256',"))
        out.append(ln.replace("'--meanmax_pool',", "'--node_output', '96',"))
        inserted+=1
assert inserted==1, inserted
s='\n'.join(out)
s=s.replace('sncp_ppo_v30.pt','sncp_ppo_v31.pt')
s=s.replace('eval_v30','eval_v31')
s=s.replace("'--version', '30'","'--version', '31'")
s=s.replace("v30 - v28 + mean+max attention pooling (--meanmax_pool)","v31 - v30 + node-fusion capacity 256/96 (--node_units/--node_output)")
s=s.replace("## 3. Training (v30 - v28 + mean+max pooling)","## 3. Training (v31 - v30 + node capacity 256/96)")
s=s.replace("## 8. Notes & roadmap (current: v30 - v28 + mean+max pooling)","## 8. Notes & roadmap (current: v31 - v30 + node capacity 256/96)")
import json; json.loads(s); open(path,'w',encoding='utf-8',newline='\n').write(s)
print('inserted:', inserted, '| node_units:', "'--node_units', '256'" in s, '| node_output:', "'--node_output', '96'" in s,
      '| meanmax:', "'--meanmax_pool'" in s, '| pre_mlp:', "'--pre_mlp'" in s, '| range:', "'--num_humans_range'" in s,
      '| v31:', "'--version', '31'" in s, '| no v30 ckpt:', 'sncp_ppo_v30.pt' not in s)
PY
```
Expected: `inserted: 1 | node_units: True | node_output: True | meanmax: True | pre_mlp: True | range: True | v31: True | no v30 ckpt: True`.

- [ ] **Step 2: Run the marker suite (green) + readiness preflight**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py::test_notebook_is_v31_node_capacity tests/test_v16_run_readiness.py -q --basetemp=./.pytmp`
Expected: PASS.

Run: `C:/ProgramData/miniconda3/python.exe -c "from pathlib import Path; from sncp_ppo.run_readiness import verify_v16_run_ready as v; s=v(Path('.')); print(s.status); [print(' ',n) for n in s.notes]"`
Expected: `pass` with `PASS: v31 Colab training and evaluation configuration is ready`.

- [ ] **Step 3: Full suite once more**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest -q --basetemp=./.pytmp`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add sncp_ppo_colab.ipynb
git commit -m "v31 notebook: add --node_units 256 --node_output 96 + bump v30->v31 paths"
```

---

### Task 7 (run-time, post-Colab — NOT code): evaluation protocol

- [ ] **Step 1: Merge + push** — finishing-a-development-branch merges `feat/v31-node-capacity` into `main` and pushes (Colab pulls `main`).
- [ ] **Step 2: Train on Colab** — pull `main`, run the v31 training cell (~3-4 h A100; the wider node is a bit slower), copy `sncp_ppo_v31.pt` to the repo root.
- [ ] **Step 3: Honest local multi-seed sweep** — copy `scratch/_sweep_v30.py` → `scratch/_sweep_v31.py`, set `CKPT='sncp_ppo_v31.pt'` and `OUT='v31_multiseed_result.json'`; run with `C:/ProgramData/miniconda3/python.exe scratch/_sweep_v31.py` (5 seeds 100–500 × 50 ep at N=5/10/15/20, paper_challenging, robot 1.0, human 1.0, max_time None, goal_noise 0).
- [ ] **Step 4: Decide** — copy `scratch/_analyze_v30.py` → `_analyze_v31.py`, load v31 vs the v30 baseline (success 97.2/89.6/85.6/79.2; collision 2.8/10.4/14.4/20.8); node capacity helps iff high-N collision drops further and/or success rises (esp N=15/20) with no regression at N=5/10 and timeout 0 (Wilson CIs + two-proportion z, Bonferroni α=0.0125, both success and collision). Report a flat/negative result honestly. Write the verdict to memory (`sncp-paper-vs-impl.md`); then experiment #3 (training budget / curriculum reach) follows.

---

## Self-review

- **Spec coverage:** node_units/node_output args + AutoNCP from them (Task 1 Steps 3-4); auto-detect from gleak/output_w (Task 1 Step 5); CLI + build wiring (Task 2); default-unchanged/widened/forward/roundtrip/compat/coexist tests (Task 1 Step 1, Task 2 Step 1); single-variable v31 keeps meanmax (Tasks 4-6); run-readiness + notebook + marker bump (Tasks 4-6); end-to-end auto-detect + eval + AutoNCP(256,96) validity + train smoke (Task 3); honest v30-baseline eval (Task 7). All spec items covered.
- **Placeholder scan:** none — every code/command step is concrete.
- **Type/name consistency:** args `node_units`/`node_output` (ints, defaults 128/48); attrs `policy.node_units`/`policy.node_output`/`policy.node_wiring.units`; inference keys `node_ltc.rnn_cell.gleak` (units) and `node_ltc.rnn_cell.output_w` (output_size); tests `test_node_capacity.py`; markers `sncp_ppo_v31.pt` / `eval_v31` / `'--version', '31'`; v31 keeps `--meanmax_pool` — consistent across tasks.
