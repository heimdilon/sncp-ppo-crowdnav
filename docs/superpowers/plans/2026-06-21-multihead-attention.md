# v33 Multi-head cross-attention — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-head crowd-attention pool with 4-head canonical cross-attention to improve the high-N (N=15/20) collision tail, built single-variable on the v30 champion.

**Architecture:** New `attn_heads` constructor arg (default 1 = byte-identical single-head). When >1, the robot temporal feature `m_rr` is the query token and the per-human spatial features `M_rh` are key/value tokens; each head has `d_head = 256 // heads`. The multi-head output replaces the `a_mean` branch and is merged with v30's element-wise `a_max` via the existing `pool_merge`. Head count is persisted in an `_attn_heads` buffer for auto-detection. Training reverts v32's flat N→25/4M back to the v30 recipe and adds `--attn_heads 4`.

**Tech Stack:** PyTorch, ncps (LTC/AutoNCP), pytest. Local interpreter `C:/ProgramData/miniconda3/python.exe`; pytest needs `--basetemp=./.pytmp`.

---

## File structure

- `sncp_ppo/models.py` — MHA layers + `_multihead_attention` + `_attention_pool` branch + auto-detect (the substantive change).
- `sncp_ppo/train.py` — `--attn_heads` arg + `build_or_load_policy` passthrough.
- `tests/test_multihead_attn.py` — NEW; all MHA behavior + train wiring.
- `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`, `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` — v32→v33 markers + revert to v30 recipe + `--attn_heads 4`.

---

## Task 1: Multi-head attention core (models.py)

**Files:**
- Create: `tests/test_multihead_attn.py`
- Modify: `sncp_ppo/models.py` (signature line 19-20; attention block 97-101; `_init_linear_weights` 147-149; `_attention_pool` 180-195; `build_policy_for_checkpoint` 290-299)

- [ ] **Step 1: Write the failing test**

Create `tests/test_multihead_attn.py`:

```python
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch=2, humans=5):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_multihead_build_has_mha_layers_and_buffer():
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)
    assert hasattr(p, 'W_v') and hasattr(p, 'W_o')
    assert p.W_q.out_features == 256 and p.W_k.out_features == 256
    assert p.W_v.out_features == 256 and p.W_o.out_features == 256
    assert int(p._attn_heads.item()) == 4


def test_single_head_default_is_byte_compatible_surface():
    p = SNCPPolicy(meanmax_pool=True)  # attn_heads defaults to 1
    assert not hasattr(p, 'W_v')
    assert not hasattr(p, 'W_o')
    assert '_attn_heads' not in dict(p.named_buffers())
    assert p.W_q.out_features == 64  # legacy single-head projection unchanged


def test_multihead_forward_shapes():
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)
    h = p.init_hidden(2, 5, torch.device('cpu'))
    mu, std, value, new_h = p(_obs(2, 5), h)
    assert mu.shape == (2, 2)
    assert std.shape == (2, 2)
    assert value.shape == (2, 1)
    assert set(new_h) == {'temporal_edge', 'spatial_edge', 'node'}


def test_multihead_autodetect_roundtrip():
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)
    sd = p.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert int(rebuilt._attn_heads.item()) == 4
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_v30_meanmax_checkpoint_loads_as_single_head():
    v30 = SNCPPolicy(meanmax_pool=True)  # single-head v30 architecture
    sd = v30.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert '_attn_heads' not in dict(rebuilt.named_buffers())
    assert not hasattr(rebuilt, 'W_v')
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_heads_differentiate():
    """With several humans, the per-head attention distributions are not all identical."""
    torch.manual_seed(0)
    p = SNCPPolicy(meanmax_pool=True, attn_heads=4)
    M_rh = torch.randn(1, 6, 256)
    m_rr = torch.randn(1, 256)
    _, alpha = p._multihead_attention(M_rh, m_rr)  # alpha: [1, 4, 1, 6]
    a = alpha[0, :, 0, :]  # [4 heads, 6 humans]
    pair_diffs = (a.unsqueeze(0) - a.unsqueeze(1)).abs().sum(-1)  # [4, 4]
    assert pair_diffs.max().item() > 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_multihead_attn.py -v --basetemp=./.pytmp`
Expected: FAIL — `SNCPPolicy.__init__() got an unexpected keyword argument 'attn_heads'`.

- [ ] **Step 3a: Add the `attn_heads` constructor arg**

In `sncp_ppo/models.py`, change the signature (lines 19-20):

```python
    def __init__(self, robot_vpref=0.26, robot_wmax=1.8, pre_mlp=False,
                 attn_count_scaling=False, meanmax_pool=False, node_units=128,
                 node_output=48, attn_heads=1):
```

- [ ] **Step 3b: Replace the attention-weights block (lines 97-101)**

Old:

```python
        # 4. Attention Pooling weights
        self.W_q = nn.Linear(256, 64)
        self.W_k = nn.Linear(256, 64)
        if meanmax_pool:
            self.pool_merge = nn.Linear(512, 256)
```

New:

```python
        # 4. Attention Pooling weights
        # Single-head (default, attn_heads=1): legacy projection — robot key m_rr
        # scores each human, Value = raw M_rh. Byte-identical to v14-v32.
        # Multi-head (attn_heads>1, v33): canonical cross-attention — robot is the
        # query token, humans are key/value tokens, d_model=256 split across heads
        # so each head specializes on a different simultaneous threat (the high-N
        # failure mode). A buffer persists the head count (not recoverable from any
        # weight shape) so build_policy_for_checkpoint can auto-detect the variant.
        self.attn_heads = attn_heads
        if attn_heads > 1:
            assert 256 % attn_heads == 0, "attn_heads must divide d_model=256"
            self.W_q = nn.Linear(256, 256)
            self.W_k = nn.Linear(256, 256)
            self.W_v = nn.Linear(256, 256)
            self.W_o = nn.Linear(256, 256)
            self.register_buffer('_attn_heads', torch.tensor(float(attn_heads)))
        else:
            self.W_q = nn.Linear(256, 64)
            self.W_k = nn.Linear(256, 64)
        if meanmax_pool:
            self.pool_merge = nn.Linear(512, 256)
```

- [ ] **Step 3c: Init the new MHA layers in `_init_linear_weights` (lines 147-149)**

Old:

```python
        _orthogonal_linear(self.W_q, gain=sqrt2)
        _orthogonal_linear(self.W_k, gain=sqrt2)
        _orthogonal_linear(self.node_proj, gain=sqrt2)
```

New:

```python
        _orthogonal_linear(self.W_q, gain=sqrt2)
        _orthogonal_linear(self.W_k, gain=sqrt2)
        if self.attn_heads > 1:
            _orthogonal_linear(self.W_v, gain=sqrt2)
            _orthogonal_linear(self.W_o, gain=sqrt2)
        _orthogonal_linear(self.node_proj, gain=sqrt2)
```

- [ ] **Step 3d: Add `_multihead_attention` + branch `_attention_pool` (replace lines 180-195)**

Old:

```python
    def _attention_pool(self, M_rh, m_rr, num_humans):
        """Attention-weighted pooling of the per-human spatial features M_rh
        against the robot/temporal key m_rr. With attn_count_scaling, scores are
        scaled by n (paper Eq 13, n/sqrt(d_k)) so the pedestrian count enters the
        softmax temperature instead of being averaged away."""
        Q = self.W_q(M_rh)                      # [B, H, 64]
        K = self.W_k(m_rr).unsqueeze(1)         # [B, 1, 64]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / 8.0   # /sqrt(d_k)
        if self.attn_count_scaling:
            attn_scores = attn_scores * num_humans
        alpha = F.softmax(attn_scores, dim=1)   # [B, H, 1]
        a_mean = torch.bmm(M_rh.transpose(1, 2), alpha).squeeze(2)  # [B, 256]
        if not self.meanmax_pool:
            return a_mean
        a_max = M_rh.max(dim=1).values          # [B, 256] cardinality-robust
        return self.pool_merge(torch.cat([a_mean, a_max], dim=1))  # [B, 256]
```

New:

```python
    def _multihead_attention(self, M_rh, m_rr):
        """Canonical multi-head cross-attention: robot m_rr is the single query
        token, humans M_rh are the key/value tokens. Returns (a_attn [B,256],
        alpha [B, heads, 1, H]); each head has d_head = 256 // heads dims."""
        B, H, _ = M_rh.shape
        nh = self.attn_heads
        dh = 256 // nh
        Q = self.W_q(m_rr).view(B, nh, 1, dh)                        # [B, nh, 1, dh]
        K = self.W_k(M_rh).view(B, H, nh, dh).permute(0, 2, 1, 3)    # [B, nh, H, dh]
        V = self.W_v(M_rh).view(B, H, nh, dh).permute(0, 2, 1, 3)    # [B, nh, H, dh]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dh)  # [B, nh, 1, H]
        alpha = F.softmax(scores, dim=-1)
        ctx = torch.matmul(alpha, V).reshape(B, 256)                 # [B, 256]
        return self.W_o(ctx), alpha

    def _attention_pool(self, M_rh, m_rr, num_humans):
        """Attention-weighted pooling of per-human spatial features M_rh against
        the robot/temporal key m_rr. attn_heads>1 uses multi-head cross-attention;
        otherwise the legacy single-head weighted average. attn_count_scaling
        (single-head only) scales scores by n (paper Eq 13)."""
        if self.attn_heads > 1:
            a_attn, _ = self._multihead_attention(M_rh, m_rr)
            if not self.meanmax_pool:
                return a_attn
            a_max = M_rh.max(dim=1).values
            return self.pool_merge(torch.cat([a_attn, a_max], dim=1))
        Q = self.W_q(M_rh)                      # [B, H, 64]
        K = self.W_k(m_rr).unsqueeze(1)         # [B, 1, 64]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / 8.0   # /sqrt(d_k)
        if self.attn_count_scaling:
            attn_scores = attn_scores * num_humans
        alpha = F.softmax(attn_scores, dim=1)   # [B, H, 1]
        a_mean = torch.bmm(M_rh.transpose(1, 2), alpha).squeeze(2)  # [B, 256]
        if not self.meanmax_pool:
            return a_mean
        a_max = M_rh.max(dim=1).values          # [B, 256] cardinality-robust
        return self.pool_merge(torch.cat([a_mean, a_max], dim=1))  # [B, 256]
```

- [ ] **Step 3e: Auto-detect head count in `build_policy_for_checkpoint` (lines 290-299)**

Old:

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

New:

```python
    pre_mlp = any(key.startswith('temporal_pre_mlp') for key in state_dict)
    attn_count_scaling = '_attn_count_scaling' in state_dict
    meanmax_pool = any(key.startswith('pool_merge') for key in state_dict)
    gleak = state_dict.get('node_ltc.rnn_cell.gleak')
    node_units = int(gleak.shape[0]) if gleak is not None else 128
    out_w = state_dict.get('node_ltc.rnn_cell.output_w')
    node_output = int(out_w.shape[0]) if out_w is not None else 48
    ah = state_dict.get('_attn_heads')
    attn_heads = int(ah.item()) if ah is not None else 1
    return SNCPPolicy(robot_vpref=robot_vpref, robot_wmax=robot_wmax,
                      pre_mlp=pre_mlp, attn_count_scaling=attn_count_scaling,
                      meanmax_pool=meanmax_pool, node_units=node_units,
                      node_output=node_output, attn_heads=attn_heads)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_multihead_attn.py -v --basetemp=./.pytmp`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_multihead_attn.py sncp_ppo/models.py
git commit -m "v33: multi-head cross-attention in SNCPPolicy (auto-detected)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: train.py CLI wiring

**Files:**
- Modify: `sncp_ppo/train.py` (`build_or_load_policy` 261-269; parser after line 1117)
- Modify (test): `tests/test_multihead_attn.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_multihead_attn.py`:

```python
def test_build_or_load_policy_respects_attn_heads():
    from types import SimpleNamespace
    from sncp_ppo.train import build_or_load_policy

    class FakeEnv:
        robot_vpref = 0.26
        robot_wmax = 1.8

    args = SimpleNamespace(init_checkpoint=None, pre_mlp=True, attn_count_scaling=False,
                           meanmax_pool=True, node_units=128, node_output=48, attn_heads=4)
    policy = build_or_load_policy(args, FakeEnv(), torch.device('cpu'))
    assert int(policy._attn_heads.item()) == 4
    assert hasattr(policy, 'W_v')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_multihead_attn.py::test_build_or_load_policy_respects_attn_heads -v --basetemp=./.pytmp`
Expected: FAIL — `SNCPPolicy.__init__()` receives no `attn_heads` (policy has no `_attn_heads`), AttributeError on `policy._attn_heads`.

- [ ] **Step 3a: Pass `attn_heads` through `build_or_load_policy` (lines 261-269)**

Add the line inside the `return SNCPPolicy(...)` call (after `node_output=...`):

```python
        node_output=getattr(args, 'node_output', 48),
        attn_heads=getattr(args, 'attn_heads', 1),
    ).to(device)
```

- [ ] **Step 3b: Add the parser arg (after line 1117, near `--node_output`)**

```python
    parser.add_argument('--attn_heads', type=int, default=1,
                        help='Attention heads for crowd pooling. 1 (default) = legacy '
                             'single-head; >1 = canonical multi-head cross-attention (v33).')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_multihead_attn.py -v --basetemp=./.pytmp`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/train.py tests/test_multihead_attn.py
git commit -m "v33: wire --attn_heads through train.build_or_load_policy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Version markers v32→v33 + revert to v30 recipe + add --attn_heads 4

**Files:**
- Modify: `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` (red first)
- Modify: `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Update the marker tests (red)**

In `tests/test_post_run_pipeline.py`, rename `test_notebook_is_v32_curriculum_budget` to
`test_notebook_is_v33_multihead_attention` and set its assertions to the v33 training cell:
- training cell contains `"'--attn_heads', '4'"`, `"'--num_humans_range', '10', '20'"`,
  `"TOTAL_STEPS = 2_500_000"`, `"'--meanmax_pool'"`, `"'--pre_mlp'"`, `"'--version', '33'"` (eval),
  `"checkpoints/sncp_ppo_v33.pt"`;
- training cell does NOT contain `"'--node_units'"` or `"'--attn_count_scaling'"` or `"'10', '25'"`.

In `tests/test_v16_run_readiness.py`, rename the three v32 tests to v33
(`test_v33_run_readiness_passes_current_repo`, `test_v33_run_readiness_flags_stale_notebook`,
`test_colab_persist_cell_downloads_eval_v33_artifact_bundle`); in the stale-notebook test assert
notes contain `"v33 training"` and `"v33 evaluation"`; in the persist test assert
`"'eval_v33_artifacts'"` and `"'eval_v33'"`.

- [ ] **Step 2: Run the marker tests to verify they fail**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py -v --basetemp=./.pytmp`
Expected: FAIL (current repo still has v32 markers / 4M / N→25).

- [ ] **Step 3a: Update `sncp_ppo/run_readiness.py`**

- Header comment block (lines 11-14): replace the v32 description with: v33 = v30 (pre-MLP +
  mean+max) + 4-head multi-head cross-attention (`--attn_heads 4`); reverts v32's flat N→25/4M to
  the v30 recipe (`--num_humans_range 10 20`, `--total_steps 2_500_000`); model-only single-variable.
- `TRAINING_TOKENS`: `"TOTAL_STEPS = 4_000_000"` → `"TOTAL_STEPS = 2_500_000"`;
  `"SAVE_PATH = 'checkpoints/sncp_ppo_v32.pt'"` → `..._v33.pt`; add `"'--attn_heads'"`.
  Keep `"'--num_humans_range'"`, `"'--pre_mlp'"`, `"'--meanmax_pool'"`.
- `EVALUATION_TOKENS`: `"CHECKPOINT = 'checkpoints/sncp_ppo_v32.pt'"` → `..._v33.pt`;
  `"EVAL_OUT = 'eval_v32'"` → `'eval_v33'`; `"'--version', '32'"` → `"'--version', '33'"`.
- `_find_unique_cell` markers: `sncp_ppo_v32.pt` → `_v33.pt` (both training & evaluation);
  names `"v32 training"`/`"v32 evaluation"`.
- `_check_tokens(...)` name args (currently `"v31 training"`/`"v31 evaluation"`) → `"v33 training"`/`"v33 evaluation"`.
- PASS note `"v32 ..."` → `"v33 Colab training and evaluation configuration is ready"`.

- [ ] **Step 3b: Update `sncp_ppo_colab.ipynb`** (training cell):
- `TOTAL_STEPS = 4_000_000` → `TOTAL_STEPS = 2_500_000`
- `'--num_humans_range', '10', '25'` → `'--num_humans_range', '10', '20'`
- add `'--attn_heads', '4',` immediately after the `'--meanmax_pool',` line
- `SAVE_PATH = 'checkpoints/sncp_ppo_v32.pt'` → `..._v33.pt`

(evaluation cell):
- `CHECKPOINT = 'checkpoints/sncp_ppo_v32.pt'` → `..._v33.pt`
- `EVAL_OUT = 'eval_v32'` → `'eval_v33'`
- `'--version', '32'` → `'--version', '33'`

(persist/download cell):
- `'eval_v32_artifacts'` → `'eval_v33_artifacts'`; `'eval_v32'` → `'eval_v33'`

- [ ] **Step 4: Run the marker tests + full suite + readiness**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py -v --basetemp=./.pytmp`
Expected: PASS.

Run full suite: `C:/ProgramData/miniconda3/python.exe -m pytest --basetemp=./.pytmp -q`
Expected: all green (prior baseline ~212 + 7 new MHA tests).

Run readiness: `C:/ProgramData/miniconda3/python.exe -c "from sncp_ppo.run_readiness import verify_v16_run_ready; s=verify_v16_run_ready('.'); print(s.status); [print(n) for n in s.notes]"`
Expected: `pass` with "v33 ... ready".

- [ ] **Step 5: CLI training smoke (real `--attn_heads 4` parse + build + 1 update)**

Run (mirror the notebook training flags, tiny budget; adjust `--num_envs/--horizon` to the notebook):
`C:/ProgramData/miniconda3/python.exe sncp_ppo/train.py --pre_mlp --meanmax_pool --attn_heads 4 --fixed_scenario paper_challenging --num_humans 10 --num_humans_range 10 20 --bootstrap_easy_steps 0 --robot_vpref 1.0 --lr 1e-4 --num_envs 4 --horizon 64 --total_steps 4096 --holdout_scenarios paper_standard paper_challenging --holdout_episodes 2 --save_path ./.pytmp/smoke_v33.pt`
Expected: exit 0; `./.pytmp/smoke_v33.pt` written; no NaN/shape error. Then remove the smoke checkpoint.

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo/run_readiness.py sncp_ppo_colab.ipynb tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py
git commit -m "v33 markers: notebook+readiness to multi-head 4 heads, revert N->25/4M to v30 recipe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: (run-time, post-Colab) honest 5-seed eval vs v30

Deferred until the user trains v33 on Colab and downloads `sncp_ppo_v33.pt` to the repo root.

- [ ] Verify checkpoint architecture: `_attn_heads`==4, `pool_merge` present, `temporal_pre_mlp` present, node 128/48, no `_attn_count_scaling`.
- [ ] Copy `scratch/_sweep_v32.py` → `_sweep_v33.py` (`CKPT='sncp_ppo_v33.pt'`, `OUT='v33_multiseed_result.json'`); run with base-conda. 5 seeds × 50 ep at N=5/10/15/20, paper_challenging, robot 1.0, human 1.0, max_time None, goal_noise 0.
- [ ] Copy `scratch/_analyze_v32.py` → `_analyze_v33.py` (load v33, baseline = v30: success 97.2/89.6/85.6/79.2, collision 2.8/10.4/14.4/20.8; Wilson CI + two-prop z + Bonferroni; report success AND collision).
- [ ] **Decision rule:** multi-head helps iff high-N (N=15/20) success rises and/or collision drops with no regression at N=5/10 and timeout 0, vs v30. Report honestly (negatives included).
- [ ] Write the verdict to `MEMORY.md` (sncp-paper-vs-impl log) + chart.

---

## Self-review

- **Spec coverage:** MHA layers + buffer (T1 3b), forward (T1 3d), init (T1 3c), auto-detect (T1 3e), single-head byte-compat (T1 tests), train wiring (T2), markers + v30-recipe revert + `--attn_heads 4` (T3), honest eval vs v30 (T4). All spec sections covered.
- **Placeholder scan:** none — every code step shows full code; commands have expected output.
- **Type consistency:** `attn_heads` (int) used identically in signature, `__init__`, `build_or_load_policy`, `build_policy_for_checkpoint`; `_attn_heads` buffer (float tensor) read via `.item()`; `_multihead_attention` returns `(a_attn, alpha)` and is consumed as `a_attn, _` in `_attention_pool` and `_, alpha` in the test. `pool_merge` stays `Linear(512,256)`.
