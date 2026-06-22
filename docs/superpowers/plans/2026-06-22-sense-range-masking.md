# v35 Sense-range masking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mask humans beyond a 6 m sensing radius from the crowd attention pool (the paper's challenging-scenario perception), single-variable on the v30 champion, computed in-model from the observation.

**Architecture:** New `sense_range` constructor arg (default 0.0 = off = byte-identical). When >0, `forward` computes each human's distance from the local `spatial_edges` positions and passes a `[B,N]` visibility mask to `_attention_pool`, which sets hidden humans' attention scores to `-inf` and excludes them from the max-pool (all-masked rows fall back to a zero crowd vector). Auto-detected via a `_sense_range` buffer; no env/eval change.

**Tech Stack:** PyTorch, ncps, pytest. Local interpreter `C:/ProgramData/miniconda3/python.exe`; pytest needs `--basetemp=./.pytmp`.

---

## File structure

- `sncp_ppo/models.py` — `sense_range` arg + `_sense_range` buffer + `forward` mask compute + `_attention_pool` mask + `_masked_max` helper + `_multihead_attention` mask + auto-detect.
- `sncp_ppo/train.py` — `--sense_range` arg + passthrough.
- `tests/test_sense_range.py` — NEW.
- `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`, `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` — v34→v35 markers; drop `--action_dist beta`, add `--sense_range 6.0`.

---

## Task 1: In-model sense-range masking (models.py)

**Files:**
- Create: `tests/test_sense_range.py`
- Modify: `sncp_ppo/models.py` (signature 19-21; flags ~44; `_multihead_attention` 215-228; `_attention_pool` 230-251; forward call 315-316; `build_policy_for_checkpoint` ~367)

- [ ] **Step 1: Write the failing test** — create `tests/test_sense_range.py`:

```python
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch=2, humans=4, offset=0.0):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6) + offset,
        'temporal_edges': torch.randn(batch, 2),
    }


def test_sense_range_build_registers_buffer():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    assert float(p._sense_range.item()) == 6.0
    assert p.sense_range == 6.0


def test_default_has_no_sense_buffer():
    p = SNCPPolicy(meanmax_pool=True)  # sense_range defaults to 0.0
    assert '_sense_range' not in dict(p.named_buffers())
    assert p.sense_range == 0.0


def test_attention_pool_mask_excludes_hidden_humans():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    B, N = 1, 4
    M_rh = torch.randn(B, N, 256)
    m_rr = torch.randn(B, 256)
    mask_first = torch.tensor([[True, False, False, False]])
    first_only = p._attention_pool(M_rh, m_rr, N, mask_first)
    # Pooling with only human 0 visible must equal pooling a 1-human input.
    only = p._attention_pool(M_rh[:, :1, :], m_rr, 1, torch.ones(B, 1, dtype=torch.bool))
    assert torch.allclose(first_only, only, atol=1e-5)
    # And it must differ from pooling all four humans.
    full = p._attention_pool(M_rh, m_rr, N, torch.ones(B, N, dtype=torch.bool))
    assert not torch.allclose(first_only, full)


def test_all_masked_is_finite_zero_pool():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    B, N = 2, 3
    M_rh = torch.randn(B, N, 256)
    m_rr = torch.randn(B, 256)
    out = p._attention_pool(M_rh, m_rr, N, torch.zeros(B, N, dtype=torch.bool))
    assert torch.isfinite(out).all()
    expected = p.pool_merge(torch.zeros(B, 512))
    assert torch.allclose(out, expected, atol=1e-5)


def test_forward_all_far_humans_is_finite():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    h = p.init_hidden(2, 4, torch.device('cpu'))
    obs = _obs(2, 4, offset=100.0)  # all humans ~100 m away -> all masked
    mu, std, value, _ = p(obs, h)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()


def test_sense_range_autodetect_roundtrip():
    p = SNCPPolicy(meanmax_pool=True, sense_range=6.0)
    sd = p.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert float(rebuilt._sense_range.item()) == 6.0
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_v30_checkpoint_autodetects_no_masking():
    p = SNCPPolicy(meanmax_pool=True)  # sense_range 0
    rebuilt = build_policy_for_checkpoint(p.state_dict())
    assert rebuilt.sense_range == 0.0
    assert '_sense_range' not in dict(rebuilt.named_buffers())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_sense_range.py -v --basetemp=./.pytmp`
Expected: FAIL — `SNCPPolicy.__init__() got an unexpected keyword argument 'sense_range'`.

- [ ] **Step 3a: Add the `sense_range` arg + buffer**

Signature (models.py:19-21):

```python
    def __init__(self, robot_vpref=0.26, robot_wmax=1.8, pre_mlp=False,
                 attn_count_scaling=False, meanmax_pool=False, node_units=128,
                 node_output=48, attn_heads=1, action_dist='gaussian', sense_range=0.0):
```

After `self.action_dist = action_dist` (models.py:44):

```python
        self.action_dist = action_dist
        # Sense-range masking (v35): when >0, humans beyond this radius (metres) are
        # excluded from the crowd attention pool — the paper's limited perception
        # (challenging = 6 m). A buffer persists it for auto-detect; default 0 = sense
        # all humans (v14-v34 byte-identical).
        self.sense_range = sense_range
        if sense_range > 0:
            self.register_buffer('_sense_range', torch.tensor(float(sense_range)))
```

- [ ] **Step 3b: Add the `_masked_max` helper + thread the mask into `_multihead_attention`**

Add this static method just above `_multihead_attention` (models.py:215):

```python
    @staticmethod
    def _masked_max(M_rh, mask, none_visible):
        """Element-wise max over visible humans (mask True = visible); rows with no
        visible human return a zero vector instead of -inf."""
        masked = M_rh.masked_fill((~mask).unsqueeze(-1), float('-inf'))
        a_max = masked.max(dim=1).values
        return torch.where(none_visible.unsqueeze(-1), torch.zeros_like(a_max), a_max)

```

In `_multihead_attention`, change the signature to accept a mask and apply it (models.py:215-228). Replace:

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
```

with:

```python
    def _multihead_attention(self, M_rh, m_rr, mask=None):
        """Canonical multi-head cross-attention: robot m_rr is the single query
        token, humans M_rh are the key/value tokens. Returns (a_attn [B,256],
        alpha [B, heads, 1, H]); each head has d_head = 256 // heads dims. mask
        ([B,H] bool, True = visible) zeroes hidden humans' attention weights."""
        B, H, _ = M_rh.shape
        nh = self.attn_heads
        dh = 256 // nh
        Q = self.W_q(m_rr).view(B, nh, 1, dh)                        # [B, nh, 1, dh]
        K = self.W_k(M_rh).view(B, H, nh, dh).permute(0, 2, 1, 3)    # [B, nh, H, dh]
        V = self.W_v(M_rh).view(B, H, nh, dh).permute(0, 2, 1, 3)    # [B, nh, H, dh]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dh)  # [B, nh, 1, H]
        if mask is not None:
            scores = scores.masked_fill((~mask).view(B, 1, 1, H), float('-inf'))
        alpha = F.softmax(scores, dim=-1)
        if mask is not None:
            alpha = torch.nan_to_num(alpha, nan=0.0)                 # all-hidden rows -> 0
        ctx = torch.matmul(alpha, V).reshape(B, 256)                 # [B, 256]
        return self.W_o(ctx), alpha
```

- [ ] **Step 3c: Add the `mask` param to `_attention_pool` and apply it (models.py:230-251)**

Replace the whole `_attention_pool` method with:

```python
    def _attention_pool(self, M_rh, m_rr, num_humans, mask=None):
        """Attention-weighted pooling of per-human spatial features M_rh against
        the robot/temporal key m_rr. attn_heads>1 uses multi-head cross-attention;
        otherwise the legacy single-head weighted average. attn_count_scaling
        (single-head only) scales scores by n (paper Eq 13). mask ([B,N] bool, True
        = visible) restricts pooling to humans within the sensing radius: hidden
        humans get zero attention weight and are excluded from the max; rows with no
        visible human return a zero crowd vector."""
        none_visible = (~mask).all(dim=1) if mask is not None else None
        if self.attn_heads > 1:
            a_attn, _ = self._multihead_attention(M_rh, m_rr, mask)
            if not self.meanmax_pool:
                return a_attn
            a_max = (self._masked_max(M_rh, mask, none_visible) if mask is not None
                     else M_rh.max(dim=1).values)
            return self.pool_merge(torch.cat([a_attn, a_max], dim=1))
        Q = self.W_q(M_rh)                      # [B, H, 64]
        K = self.W_k(m_rr).unsqueeze(1)         # [B, 1, 64]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / 8.0   # /sqrt(d_k)
        if self.attn_count_scaling:
            attn_scores = attn_scores * num_humans
        if mask is not None:
            attn_scores = attn_scores.masked_fill((~mask).unsqueeze(-1), float('-inf'))
        alpha = F.softmax(attn_scores, dim=1)   # [B, H, 1]
        if mask is not None:
            alpha = torch.nan_to_num(alpha, nan=0.0)   # all-hidden rows -> 0 weights
        a_mean = torch.bmm(M_rh.transpose(1, 2), alpha).squeeze(2)  # [B, 256]
        if not self.meanmax_pool:
            return a_mean
        a_max = (self._masked_max(M_rh, mask, none_visible) if mask is not None
                 else M_rh.max(dim=1).values)          # [B, 256] cardinality-robust
        return self.pool_merge(torch.cat([a_mean, a_max], dim=1))  # [B, 256]
```

- [ ] **Step 3d: Compute the mask in `forward` and pass it (models.py:315-316)**

Replace:

```python
        # 4. Attention Pooling
        u_att = self._attention_pool(M_rh, m_rr, num_humans)
```

with:

```python
        # 4. Attention Pooling — optionally mask humans beyond the sensing radius
        # (paper's limited perception: only nearby humans influence the action).
        sense_mask = None
        if self.sense_range > 0:
            dist = torch.hypot(spatial_edges[:, :, 0], spatial_edges[:, :, 1])  # [B, N] m
            sense_mask = dist <= self.sense_range                                # [B, N] bool
        u_att = self._attention_pool(M_rh, m_rr, num_humans, sense_mask)
```

- [ ] **Step 3e: Auto-detect in `build_policy_for_checkpoint`**

Replace the `action_dist`/`return` tail with:

```python
    action_dist = 'gaussian' if 'actor_logstd' in state_dict else 'beta'
    sr = state_dict.get('_sense_range')
    sense_range = float(sr.item()) if sr is not None else 0.0
    return SNCPPolicy(robot_vpref=robot_vpref, robot_wmax=robot_wmax,
                      pre_mlp=pre_mlp, attn_count_scaling=attn_count_scaling,
                      meanmax_pool=meanmax_pool, node_units=node_units,
                      node_output=node_output, attn_heads=attn_heads,
                      action_dist=action_dist, sense_range=sense_range)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_sense_range.py -v --basetemp=./.pytmp`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_sense_range.py sncp_ppo/models.py
git commit -m "v35: in-model sense-range masking of the crowd attention pool (auto-detected)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: train.py wiring

**Files:**
- Modify: `sncp_ppo/train.py` (`build_or_load_policy` passthrough; parser after `--action_dist`)
- Modify (test): `tests/test_sense_range.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_sense_range.py`:

```python
def test_build_or_load_policy_respects_sense_range():
    from types import SimpleNamespace
    from sncp_ppo.train import build_or_load_policy

    class FakeEnv:
        robot_vpref = 1.0
        robot_wmax = 1.8

    args = SimpleNamespace(init_checkpoint=None, pre_mlp=True, attn_count_scaling=False,
                           meanmax_pool=True, node_units=128, node_output=48,
                           attn_heads=1, action_dist='gaussian', sense_range=6.0)
    policy = build_or_load_policy(args, FakeEnv(), torch.device('cpu'))
    assert policy.sense_range == 6.0
    assert float(policy._sense_range.item()) == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_sense_range.py::test_build_or_load_policy_respects_sense_range -v --basetemp=./.pytmp`
Expected: FAIL — policy built without `sense_range` (no `_sense_range`, AttributeError).

- [ ] **Step 3a: Passthrough in `build_or_load_policy`** — add the line after `action_dist=...`:

```python
        action_dist=getattr(args, 'action_dist', 'gaussian'),
        sense_range=getattr(args, 'sense_range', 0.0),
    ).to(device)
```

- [ ] **Step 3b: Parser arg** — add after the `--action_dist` argument:

```python
    parser.add_argument('--sense_range', type=float, default=0.0,
                        help='Robot crowd sensing radius (m). 0 (default) = sense all humans; '
                             '>0 = mask humans beyond this range in the attention pool (v35; '
                             'paper challenging = 6.0). Auto-detected on load.')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_sense_range.py -v --basetemp=./.pytmp`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/train.py tests/test_sense_range.py
git commit -m "v35: wire --sense_range through train.build_or_load_policy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Version markers v34→v35 (drop --action_dist beta, add --sense_range 6.0)

**Files:**
- Modify: `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` (red first)
- Modify: `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Update the marker tests (red)**

In `tests/test_post_run_pipeline.py`, rename `test_notebook_is_v34_beta_action` to
`test_notebook_is_v35_sense_range`; assertions: `"'--sense_range', '6.0'"` in train,
`"'--action_dist'"` NOT in train, keep `"'--num_humans_range', '10', '20'"`, `"TOTAL_STEPS = 2_500_000"`,
`"'--meanmax_pool'"`, `"'--pre_mlp'"`, `"checkpoints/sncp_ppo_v35.pt"`, `"'--version', '35'"`,
`"'--baseline_nav_steps', '32'"`, `"'--max_time'" not in ev`.

In `tests/test_v16_run_readiness.py`, rename the three v34 tests to v35
(`test_v35_run_readiness_passes_current_repo`, `test_v35_run_readiness_flags_stale_notebook`,
`test_colab_persist_cell_downloads_eval_v35_artifact_bundle`); stale-test notes contain
`"v35 training"`/`"v35 evaluation"`; persist test asserts `"'eval_v35_artifacts'"` and `"'eval_v35'"`.

- [ ] **Step 2: Run marker tests to verify they fail**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py -v --basetemp=./.pytmp`
Expected: FAIL (repo still has v34 markers / `--action_dist beta`).

- [ ] **Step 3a: Update `sncp_ppo/run_readiness.py`**

- Header comment (lines 11-15): replace with v35 description: v35 = v30 + sense-range masking
  (`--sense_range 6.0`); reverts v34's Beta; v30 recipe (`--num_humans_range 10 20`,
  `--total_steps 2_500_000`); model-only, auto-detected from the `_sense_range` buffer.
- `TRAINING_TOKENS`: replace `"'--action_dist'"` with `"'--sense_range'"`;
  `"SAVE_PATH = 'checkpoints/sncp_ppo_v34.pt'"` → `..._v35.pt`.
- `EVALUATION_TOKENS`: `..._v34.pt` → `..._v35.pt`; `"EVAL_OUT = 'eval_v34'"` → `'eval_v35'`;
  `"'--version', '34'"` → `"'--version', '35'"`.
- `_find_unique_cell` markers `sncp_ppo_v34.pt` → `_v35.pt`; names `"v34 training"`/`"v34 evaluation"` → v35.
- `_check_tokens` name args `"v34 training"`/`"v34 evaluation"` → v35.
- PASS note `"v34 ..."` → `"v35 Colab training and evaluation configuration is ready"`.

- [ ] **Step 3b: Update `sncp_ppo_colab.ipynb`** (raw-text edit; the `--action_dist beta` element becomes `--sense_range 6.0`):

```bash
C:/ProgramData/miniconda3/python.exe - <<'PY'
import io, json
p = "sncp_ppo_colab.ipynb"
t = io.open(p, encoding="utf-8").read()
for old, new, exp in [
    ("'--action_dist', 'beta'", "'--sense_range', '6.0'", 1),
    ("sncp_ppo_v34", "sncp_ppo_v35", None),
    ("eval_v34", "eval_v35", None),
    ("'--version', '34'", "'--version', '35'", 1),
]:
    c = t.count(old)
    assert c and (exp is None or c == exp), (old, c, exp)
    t = t.replace(old, new)
json.loads(t)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("OK")
PY
```

- [ ] **Step 4: Run marker tests + full suite + readiness**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py -v --basetemp=./.pytmp` → PASS.
Run full suite: `C:/ProgramData/miniconda3/python.exe -m pytest --basetemp=./.pytmp -q` → all green (~237 + 8 sense-range).
Run readiness: `C:/ProgramData/miniconda3/python.exe -c "from sncp_ppo.run_readiness import verify_v16_run_ready; s=verify_v16_run_ready('.'); print(s.status); [print(n) for n in s.notes]"` → `pass` "v35 ... ready".

- [ ] **Step 5: CLI training smoke (real `--sense_range 6.0` parse + masked forward + update)**

Run:
`C:/ProgramData/miniconda3/python.exe -m sncp_ppo.train --pre_mlp --meanmax_pool --sense_range 6.0 --fixed_scenario paper_challenging --num_humans 10 --num_humans_range 10 20 --bootstrap_easy_steps 0 --robot_vpref 1.0 --lr 1e-4 --num_envs 4 --horizon 64 --total_steps 4096 --holdout_scenarios paper_standard paper_challenging --holdout_episodes 2 --save_path ./.pytmp/smoke_v35.pt`
Expected: exit 0; no NaN/shape error (masked attention + all-masked guard run end-to-end). Then remove `./.pytmp/smoke_v35.pt` and the smoke `logs/training_*.csv`.

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo/run_readiness.py sncp_ppo_colab.ipynb tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py
git commit -m "v35 markers: notebook+readiness to --sense_range 6.0, drop Beta, v30 recipe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: (run-time, post-Colab) honest 5-seed eval vs v30

Deferred until v35 is trained on Colab and `sncp_ppo_v35.pt` is at the repo root.

- [ ] Verify checkpoint: `_sense_range`==6.0, pre_mlp, meanmax, node 128/48, no attn_heads/beta (has actor_logstd).
- [ ] Copy `scratch/_sweep_v34.py` → `_sweep_v35.py` (`CKPT='sncp_ppo_v35.pt'`, `OUT='v35_multiseed_result.json'`); run base-conda. 5 seeds × 50 ep at N=5/10/15/20, paper_challenging, robot 1.0, human 1.0, max_time None, goal_noise 0. (Masking auto-detected from the checkpoint — the eval applies it.)
- [ ] Copy `scratch/_analyze_v34.py` → `_analyze_v35.py` (load v35, baseline v30: success 97.2/89.6/85.6/79.2, collision 2.8/10.4/14.4/20.8; Wilson CI + two-prop z + Bonferroni; success AND collision).
- [ ] **Decision rule:** sense-masking helps iff high-N (N=15/20) success rises and/or collision drops, no regression at N=5/10, timeout 0, vs v30. Report honestly; even a flat result keeps the comparison paper-faithful.
- [ ] Write verdict to `MEMORY.md` + chart.

---

## Self-review

- **Spec coverage:** `sense_range` arg + buffer (T1 3a), forward mask (T1 3d), `_attention_pool` mask + all-masked guard (T1 3c), `_masked_max` + multihead mask (T1 3b), auto-detect (T1 3e), train wiring (T2), markers + recipe revert (T3), honest eval (T4). All spec sections covered.
- **Placeholder scan:** none — every code step shows full code; commands have expected output.
- **Type consistency:** `sense_range` (float) identical in signature / `build_or_load_policy` / `build_policy_for_checkpoint`; `_sense_range` buffer read via `.item()`; `_attention_pool(..., mask=None)` and `_multihead_attention(..., mask=None)` and `_masked_max(M_rh, mask, none_visible)` signatures match call sites; `sense_mask` is `[B,N]` bool consumed identically. Default `sense_range=0` → `mask=None` → byte-identical (gaussian + v30 paths untouched).
