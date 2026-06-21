# v34 Beta action distribution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the policy's Gaussian action head with a Beta distribution on the bounded action box, single-variable on the v30 champion, to remove boundary-clip bias and give state-dependent bounded control.

**Architecture:** New `action_dist` arg (default `'gaussian'` = byte-identical). In `'beta'` mode the actor head emits 4 values (α,β per dim, `softplus+1`), `forward` returns `(alpha, beta, value, hidden)`, and `ppo.py` builds `Beta` (sampling in [0,1], scaled to physical bounds; the affine Jacobian cancels in the PPO ratio). The Gaussian path is kept byte-identical and locked by a regression test. Auto-detected from the absence of `actor_logstd`.

**Tech Stack:** PyTorch (`torch.distributions.Beta/Normal`), ncps, pytest. Local interpreter `C:/ProgramData/miniconda3/python.exe`; pytest needs `--basetemp=./.pytmp`.

---

## File structure

- `sncp_ppo/models.py` — Beta head + `_scale_action`/`_unscale_action`/`deterministic_action` + forward branch + auto-detect.
- `sncp_ppo/ppo.py` — `select_action` + `update` Beta branches (Gaussian untouched, nested under `else`).
- `sncp_ppo/train.py` — `--action_dist` arg + passthrough.
- `tests/test_beta_action.py` — NEW.
- `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`, `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` — v33→v34 markers; drop `--attn_heads 4`, add `--action_dist beta`.

---

## Task 1: Beta action head + helpers (models.py)

**Files:**
- Create: `tests/test_beta_action.py`
- Modify: `sncp_ppo/models.py` (signature line 19-21; flags ~43; actor block 113-126; `_init_linear_weights` 153-163; forward 253-280; add helper methods; `build_policy_for_checkpoint` ~290-299)

- [ ] **Step 1: Write the failing test** — create `tests/test_beta_action.py`:

```python
import torch

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint


def _obs(batch=2, humans=5):
    return {
        'robot_node': torch.randn(batch, 7),
        'spatial_edges': torch.randn(batch, humans, 6),
        'temporal_edges': torch.randn(batch, 2),
    }


def test_beta_build_head_and_no_logstd():
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta')
    finals = [m for m in p.actor_mu if isinstance(m, torch.nn.Linear)]
    assert finals[-1].out_features == 4          # alpha,beta for 2 action dims
    assert not hasattr(p, 'actor_logstd')
    assert p.action_dist == 'beta'


def test_gaussian_default_unchanged_surface():
    p = SNCPPolicy(meanmax_pool=True)            # action_dist defaults to gaussian
    finals = [m for m in p.actor_mu if isinstance(m, torch.nn.Linear)]
    assert finals[-1].out_features == 2
    assert hasattr(p, 'actor_logstd')
    assert p.action_dist == 'gaussian'


def test_beta_forward_returns_valid_alpha_beta():
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta')
    h = p.init_hidden(2, 5, torch.device('cpu'))
    alpha, beta, value, new_h = p(_obs(2, 5), h)
    assert alpha.shape == (2, 2) and beta.shape == (2, 2)
    assert torch.all(alpha >= 1.0) and torch.all(beta >= 1.0)   # softplus+1
    assert value.shape == (2, 1)
    assert set(new_h) == {'temporal_edge', 'spatial_edge', 'node'}


def test_gaussian_forward_still_returns_mu_std():
    p = SNCPPolicy(meanmax_pool=True)
    h = p.init_hidden(2, 5, torch.device('cpu'))
    mu, std, value, _ = p(_obs(2, 5), h)
    assert mu.shape == (2, 2) and std.shape == (2, 2)
    # std is the broadcast exp(logstd) — gaussian path byte-identical
    assert torch.allclose(std[0], torch.exp(p.actor_logstd).squeeze(0))
    assert torch.all(mu[:, 0:1] >= 0.0) and torch.all(mu[:, 0:1] <= p.robot_vpref)


def test_scale_unscale_roundtrip_and_deterministic_mean():
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta', robot_vpref=1.0, robot_wmax=1.8)
    a = torch.tensor([[0.4, -0.9], [1.0, 1.8]])
    x = p._unscale_action(a)
    assert torch.allclose(p._scale_action(x), a, atol=1e-4)
    alpha = torch.tensor([[3.0, 2.0]]); beta = torch.tensor([[1.0, 2.0]])
    det = p.deterministic_action(alpha, beta)        # scaled alpha/(alpha+beta)
    expected = p._scale_action(alpha / (alpha + beta))
    assert torch.allclose(det, expected)
    assert det[0, 0] >= 0.0 and det[0, 0] <= 1.0     # v within [0,vpref]
    assert det[0, 1] >= -1.8 and det[0, 1] <= 1.8    # w within [-wmax,wmax]


def test_beta_autodetect_roundtrip():
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta')
    sd = p.state_dict()
    rebuilt = build_policy_for_checkpoint(sd)
    assert rebuilt.action_dist == 'beta'
    missing, unexpected = rebuilt.load_state_dict(sd, strict=False)
    assert not missing and not unexpected


def test_gaussian_checkpoint_autodetects_gaussian():
    p = SNCPPolicy(meanmax_pool=True)                # gaussian (has actor_logstd)
    rebuilt = build_policy_for_checkpoint(p.state_dict())
    assert rebuilt.action_dist == 'gaussian'
    assert hasattr(rebuilt, 'actor_logstd')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_beta_action.py -v --basetemp=./.pytmp`
Expected: FAIL — `SNCPPolicy.__init__() got an unexpected keyword argument 'action_dist'`.

- [ ] **Step 3a: Add the `action_dist` arg + flag**

Signature (models.py:19-21):

```python
    def __init__(self, robot_vpref=0.26, robot_wmax=1.8, pre_mlp=False,
                 attn_count_scaling=False, meanmax_pool=False, node_units=128,
                 node_output=48, attn_heads=1, action_dist='gaussian'):
```

After `self.meanmax_pool = meanmax_pool` (models.py:43) add:

```python
        self.action_dist = action_dist
```

- [ ] **Step 3b: Conditional actor head (replace models.py 113-126)**

Old (the actor_mu + std comment + actor_logstd block, through the `actor_logstd = nn.Parameter(...)` line):

```python
        # 6. Actor & Critic Heads
        self.actor_mu = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
```

(keep replacing through the `self.actor_logstd = nn.Parameter(torch.tensor([[-2.0, -1.5]]), requires_grad=True)` line and its comment block) — replace ALL of lines 113-126 with:

```python
        # 6. Actor & Critic Heads
        # action_dist='gaussian' (default): mean head (2) + global logstd, mean
        # scaled by sigmoid/tanh, sampled as Normal then clipped by the env.
        # action_dist='beta' (v34): head emits 4 raw values -> alpha,beta (softplus+1
        # => unimodal) for a Beta on [0,1]^2, scaled to the physical action box by the
        # PPO layer. No logstd. Naturally bounded (no clip bias), state-dependent.
        if action_dist == 'beta':
            self.actor_mu = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 4)
            )
            self.register_buffer('action_low', torch.tensor([0.0, -robot_wmax]))
            self.register_buffer('action_high', torch.tensor([robot_vpref, robot_wmax]))
        else:
            self.actor_mu = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            # Initial std per action dim (exp(-2.0)=0.135 for v, exp(-1.5)=0.22 for w)
            # keeps the heading random-walk bounded; load-bearing at the old vpref=0.26.
            self.actor_logstd = nn.Parameter(torch.tensor([[-2.0, -1.5]]), requires_grad=True)
```

- [ ] **Step 3c: Branch the actor-head init in `_init_linear_weights` (models.py 153-163)**

Old:

```python
        linears = [m for m in self.actor_mu if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _orthogonal_linear(m, gain=sqrt2)
        _orthogonal_linear(linears[-1], gain=0.01)
        # Bias the linear-velocity pre-activation so sigmoid(2.0)·vpref ≈ 0.88·vpref
        # at init (~0.23 m/s vs the old 0.13 m/s = sigmoid(0)·vpref). The robot
        # needs ≥0.133 m/s to cover the 8 m goal distance within max_time = 60 s;
        # the old default was right on the timeout cliff so the agent never
        # received a goal-reward signal to bootstrap learning.
        with torch.no_grad():
            linears[-1].bias.data.copy_(torch.tensor([2.0, 0.0]))
```

New:

```python
        linears = [m for m in self.actor_mu if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _orthogonal_linear(m, gain=sqrt2)
        _orthogonal_linear(linears[-1], gain=0.01)
        # Gaussian only: bias the linear-velocity pre-activation so sigmoid(2.0)·vpref
        # ≈ 0.88·vpref at init. The robot needs ≥0.133 m/s to cover the 8 m goal within
        # the budget; the old default sat on the timeout cliff. Beta keeps bias 0 ->
        # alpha=beta=softplus(0)+1≈1.69 (symmetric, mean≈0.5·vpref, ample at vpref=1.0).
        if self.action_dist == 'gaussian':
            with torch.no_grad():
                linears[-1].bias.data.copy_(torch.tensor([2.0, 0.0]))
```

- [ ] **Step 3d: Branch the forward actor block (replace models.py 253-264)**

Old:

```python
        # 6. Actor & Critic Outputs
        mu_raw = self.actor_mu(sf)  # [batch_size, 2]
        
        # Scale actor outputs to physical robot limits
        # Linear velocity: [0, robot_vpref]
        v_mu = torch.sigmoid(mu_raw[:, 0:1]) * self.robot_vpref
        # Angular velocity: [-robot_wmax, robot_wmax]
        w_mu = torch.tanh(mu_raw[:, 1:2]) * self.robot_wmax
        mu = torch.cat([v_mu, w_mu], dim=-1)
        
        # Standard deviation for PPO exploration
        std = torch.exp(self.actor_logstd).expand_as(mu)
```

New:

```python
        # 6. Actor & Critic Outputs
        actor_raw = self.actor_mu(sf)
        if self.action_dist == 'beta':
            # alpha,beta for a Beta on [0,1]^2; +1 => unimodal. Scaling to the
            # physical action box happens in the PPO layer (_scale_action).
            out1 = F.softplus(actor_raw[:, :2]) + 1.0   # alpha [B,2]
            out2 = F.softplus(actor_raw[:, 2:]) + 1.0   # beta  [B,2]
        else:
            # Gaussian: scale mean to physical limits, std from the global logstd.
            v_mu = torch.sigmoid(actor_raw[:, 0:1]) * self.robot_vpref
            w_mu = torch.tanh(actor_raw[:, 1:2]) * self.robot_wmax
            out1 = torch.cat([v_mu, w_mu], dim=-1)               # mu  [B,2]
            out2 = torch.exp(self.actor_logstd).expand_as(out1)  # std [B,2]
```

And the return (models.py:280):

Old: `        return mu, std, value, new_hidden_states`
New: `        return out1, out2, value, new_hidden_states`

- [ ] **Step 3e: Add the action helper methods** (place immediately above `def forward` in models.py)

```python
    def _scale_action(self, x):
        """Map x in [0,1]^2 to the physical action box [0,vpref]x[-wmax,wmax]."""
        return self.action_low + (self.action_high - self.action_low) * x

    def _unscale_action(self, a):
        """Inverse of _scale_action; clamp to (eps,1-eps) for Beta.log_prob safety."""
        x = (a - self.action_low) / (self.action_high - self.action_low)
        return x.clamp(1e-6, 1.0 - 1e-6)

    def deterministic_action(self, out1, out2):
        """Greedy action: Gaussian mean (already physical) or scaled Beta mean."""
        if self.action_dist == 'beta':
            return self._scale_action(out1 / (out1 + out2))   # alpha/(alpha+beta)
        return out1

```

- [ ] **Step 3f: Auto-detect in `build_policy_for_checkpoint`** — add before the `return SNCPPolicy(...)` and pass it:

```python
    action_dist = 'gaussian' if 'actor_logstd' in state_dict else 'beta'
    return SNCPPolicy(robot_vpref=robot_vpref, robot_wmax=robot_wmax,
                      pre_mlp=pre_mlp, attn_count_scaling=attn_count_scaling,
                      meanmax_pool=meanmax_pool, node_units=node_units,
                      node_output=node_output, attn_heads=attn_heads,
                      action_dist=action_dist)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_beta_action.py -v --basetemp=./.pytmp`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_beta_action.py sncp_ppo/models.py
git commit -m "v34: Beta action head + scale/unscale/deterministic helpers (auto-detected)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: PPO Beta branch + train.py wiring

**Files:**
- Modify: `sncp_ppo/ppo.py` (`select_action` 209-222; `update` BPTT 474-496)
- Modify: `sncp_ppo/train.py` (`build_or_load_policy` ~261-270; parser ~1117)
- Modify (test): `tests/test_beta_action.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_beta_action.py`:

```python
import numpy as np


def _np_obs(humans=10):
    return {
        'robot_node': np.random.randn(7).astype(np.float32),
        'spatial_edges': np.random.randn(humans, 6).astype(np.float32),
        'temporal_edges': np.random.randn(2).astype(np.float32),
    }


def test_select_action_beta_is_bounded():
    from sncp_ppo.ppo import PPOTrainer
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta', robot_vpref=1.0, robot_wmax=1.8)
    agent = PPOTrainer(p)
    dev = torch.device('cpu')
    h = p.init_hidden(1, 10, dev)
    for det in (False, True):
        a, lp, v, _ = agent.select_action(_np_obs(10), h, dev, deterministic=det)
        assert 0.0 <= a[0] <= 1.0          # v in [0, vpref]
        assert -1.8 <= a[1] <= 1.8         # w in [-wmax, wmax]


def test_beta_update_math_is_finite_and_differentiable():
    # The core of the PPO beta branch: forward -> Beta -> unscale stored action ->
    # log_prob/entropy -> backward. Mirrors ppo.update without the full rollout.
    p = SNCPPolicy(meanmax_pool=True, action_dist='beta', robot_vpref=1.0, robot_wmax=1.8)
    h = p.init_hidden(3, 6, torch.device('cpu'))
    alpha, beta, _, _ = p(_obs(3, 6), h)
    actions = torch.stack([p._scale_action(torch.rand(2)) for _ in range(3)])  # physical
    x = p._unscale_action(actions)
    dist = torch.distributions.Beta(alpha, beta)
    logp = dist.log_prob(x).sum(-1)
    ent = dist.entropy().sum(-1)
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all()
    logp.sum().backward()
    assert p.actor_mu[-1].weight.grad is not None


def test_build_or_load_policy_respects_action_dist():
    from types import SimpleNamespace
    from sncp_ppo.train import build_or_load_policy

    class FakeEnv:
        robot_vpref = 1.0
        robot_wmax = 1.8

    args = SimpleNamespace(init_checkpoint=None, pre_mlp=True, attn_count_scaling=False,
                           meanmax_pool=True, node_units=128, node_output=48,
                           attn_heads=1, action_dist='beta')
    policy = build_or_load_policy(args, FakeEnv(), torch.device('cpu'))
    assert policy.action_dist == 'beta'
    assert not hasattr(policy, 'actor_logstd')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_beta_action.py -k "select_action_beta or build_or_load_policy_respects_action_dist" -v --basetemp=./.pytmp`
Expected: FAIL — `select_action` builds `Normal(alpha,beta)` (treats α,β as μ,σ → out-of-bounds / wrong), and `build_or_load_policy` rejects/ignores `action_dist`.

- [ ] **Step 3a: Branch `select_action` (replace ppo.py 209-222)**

Old:

```python
        with torch.no_grad():
            mu, std, value, h_states_new = self.policy(obs_tensor, h_states)

        if deterministic:
            action = mu
            log_prob_value = 0.0
        else:
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob_value = dist.log_prob(action).sum(-1).item()

        action_np = action.cpu().numpy()[0]

        return action_np, log_prob_value, value.item(), h_states_new
```

New:

```python
        with torch.no_grad():
            out1, out2, value, h_states_new = self.policy(obs_tensor, h_states)

        if self.policy.action_dist == 'beta':
            if deterministic:
                action = self.policy.deterministic_action(out1, out2)
                log_prob_value = 0.0
            else:
                dist = torch.distributions.Beta(out1, out2)
                x = dist.sample()
                action = self.policy._scale_action(x)        # store physical, in-bounds
                log_prob_value = dist.log_prob(x).sum(-1).item()
        else:
            mu, std = out1, out2
            if deterministic:
                action = mu
                log_prob_value = 0.0
            else:
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                log_prob_value = dist.log_prob(action).sum(-1).item()

        action_np = action.cpu().numpy()[0]

        return action_np, log_prob_value, value.item(), h_states_new
```

- [ ] **Step 3b: Branch the `update` BPTT collection + distribution (ppo.py 448-496)**

Rename the collectors (ppo.py 448-450):

Old:
```python
                all_mu = []
                all_std = []
                all_values = []
```
New:
```python
                all_p1 = []
                all_p2 = []
                all_values = []
```

Inside the `for t in range(S):` loop (ppo.py 474-477):

Old:
```python
                    mu, std, value, new_h = self.policy(step_obs, step_h)
                    all_mu.append(mu)
                    all_std.append(std)
                    all_values.append(value)
```
New:
```python
                    out1, out2, value, new_h = self.policy(step_obs, step_h)
                    all_p1.append(out1)
                    all_p2.append(out2)
                    all_values.append(value)
```

The stack + distribution block (ppo.py 488-496):

Old:
```python
                # Stack: [B, S, ...]
                all_mu = torch.stack(all_mu, dim=1)      # [B, S, 2]
                all_std = torch.stack(all_std, dim=1)     # [B, S, 2]
                all_values = torch.stack(all_values, dim=1).squeeze(-1)  # [B, S]
                
                # Compute log probs and entropy
                dist = torch.distributions.Normal(all_mu, all_std)
                new_log_probs = dist.log_prob(b_actions).sum(-1)  # [B, S]
                entropy = dist.entropy().sum(-1)                    # [B, S]
```
New:
```python
                # Stack: [B, S, ...]
                all_p1 = torch.stack(all_p1, dim=1)      # [B, S, 2]  mu or alpha
                all_p2 = torch.stack(all_p2, dim=1)      # [B, S, 2]  std or beta
                all_values = torch.stack(all_values, dim=1).squeeze(-1)  # [B, S]

                # Compute log probs and entropy under the policy's distribution.
                # Beta: log_prob on the [0,1] pre-image of the stored physical action;
                # the affine _scale Jacobian is constant so it cancels in the ratio.
                if self.policy.action_dist == 'beta':
                    x = self.policy._unscale_action(b_actions)
                    dist = torch.distributions.Beta(all_p1, all_p2)
                    new_log_probs = dist.log_prob(x).sum(-1)        # [B, S]
                    entropy = dist.entropy().sum(-1)                 # [B, S]
                else:
                    dist = torch.distributions.Normal(all_p1, all_p2)
                    new_log_probs = dist.log_prob(b_actions).sum(-1)  # [B, S]
                    entropy = dist.entropy().sum(-1)                  # [B, S]
```

- [ ] **Step 3c: train.py wiring**

`build_or_load_policy` (ppo.py is models; this is train.py) — add to the `return SNCPPolicy(...)` call after `attn_heads=...`:

```python
        attn_heads=getattr(args, 'attn_heads', 1),
        action_dist=getattr(args, 'action_dist', 'gaussian'),
    ).to(device)
```

Parser (train.py, after the `--attn_heads` arg):

```python
    parser.add_argument('--action_dist', type=str, default='gaussian',
                        choices=['gaussian', 'beta'],
                        help='Policy action distribution. gaussian (default) = Normal+clip; '
                             'beta = bounded state-dependent Beta head (v34). Auto-detected on load.')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_beta_action.py -v --basetemp=./.pytmp`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/ppo.py sncp_ppo/train.py tests/test_beta_action.py
git commit -m "v34: PPO Beta branch (select_action + update) + --action_dist wiring

Gaussian path preserved byte-identical under else-branch.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Version markers v33→v34 + recipe (drop --attn_heads, add --action_dist beta)

**Files:**
- Modify: `tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py` (red first)
- Modify: `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`

- [ ] **Step 1: Update marker tests (red)**

In `tests/test_post_run_pipeline.py`, rename `test_notebook_is_v33_multihead_attention` to
`test_notebook_is_v34_beta_action`; change assertions: assert `"'--action_dist', 'beta'"` in train,
`"'--attn_heads'"` NOT in train, keep `"'--num_humans_range', '10', '20'"`, `"TOTAL_STEPS = 2_500_000"`,
`"'--meanmax_pool'"`, `"'--pre_mlp'"`, `"checkpoints/sncp_ppo_v34.pt"`, `"'--version', '34'"`,
`"'--baseline_nav_steps', '32'"`, `"'--max_time'" not in ev`.

In `tests/test_v16_run_readiness.py`, rename the three v33 tests to v34
(`test_v34_run_readiness_passes_current_repo`, `test_v34_run_readiness_flags_stale_notebook`,
`test_colab_persist_cell_downloads_eval_v34_artifact_bundle`); stale-test asserts notes contain
`"v34 training"` and `"v34 evaluation"`; persist test asserts `"'eval_v34_artifacts'"` and `"'eval_v34'"`.

- [ ] **Step 2: Run marker tests to verify they fail**

Run: `C:/ProgramData/miniconda3/python.exe -m pytest tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py -v --basetemp=./.pytmp`
Expected: FAIL (repo still has v33 markers / `--attn_heads`).

- [ ] **Step 3a: Update `sncp_ppo/run_readiness.py`**

- Header comment (lines 11-15): replace with v34 description: v34 = v30 + Beta action distribution
  (`--action_dist beta`); reverts v33's multi-head; v30 recipe (`--num_humans_range 10 20`,
  `--total_steps 2_500_000`); model+ppo, auto-detected from the absence of `actor_logstd`.
- `TRAINING_TOKENS`: replace `"'--attn_heads'"` with `"'--action_dist'"`;
  `"SAVE_PATH = 'checkpoints/sncp_ppo_v33.pt'"` → `..._v34.pt`. (TOTAL_STEPS 2.5M, num_humans_range,
  pre_mlp, meanmax stay.)
- `EVALUATION_TOKENS`: `..._v33.pt` → `..._v34.pt`; `"EVAL_OUT = 'eval_v33'"` → `'eval_v34'`;
  `"'--version', '33'"` → `"'--version', '34'"`.
- `_find_unique_cell` markers `sncp_ppo_v33.pt` → `_v34.pt`; names `"v33 training"`/`"v33 evaluation"` → v34.
- `_check_tokens` name args `"v33 training"`/`"v33 evaluation"` → v34.
- PASS note `"v33 ..."` → `"v34 Colab training and evaluation configuration is ready"`.

- [ ] **Step 3b: Update `sncp_ppo_colab.ipynb`** (use the raw-text edit pattern via miniconda python; the insert is a same-element swap):

Training cell: replace `'--attn_heads', '4',` with `'--action_dist', 'beta',`;
`sncp_ppo_v33` → `sncp_ppo_v34`. Eval cell: `eval_v33` → `eval_v34`, `'--version', '33'` → `'--version', '34'`.
Persist cell: `eval_v33` → `eval_v34` (covers `eval_v33_artifacts`).

Concretely, run from repo root:

```bash
C:/ProgramData/miniconda3/python.exe - <<'PY'
import io, json
p = "sncp_ppo_colab.ipynb"
t = io.open(p, encoding="utf-8").read()
for old, new, exp in [
    ("'--attn_heads', '4'", "'--action_dist', 'beta'", 1),
    ("sncp_ppo_v33", "sncp_ppo_v34", None),
    ("eval_v33", "eval_v34", None),
    ("'--version', '33'", "'--version', '34'", 1),
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
Run full suite: `C:/ProgramData/miniconda3/python.exe -m pytest --basetemp=./.pytmp -q` → all green (~226 + 10 beta).
Run readiness: `C:/ProgramData/miniconda3/python.exe -c "from sncp_ppo.run_readiness import verify_v16_run_ready; s=verify_v16_run_ready('.'); print(s.status); [print(n) for n in s.notes]"` → `pass` "v34 ... ready".

- [ ] **Step 5: CLI training smoke (real `--action_dist beta` parse + build + update)**

Run:
`C:/ProgramData/miniconda3/python.exe -m sncp_ppo.train --pre_mlp --meanmax_pool --action_dist beta --fixed_scenario paper_challenging --num_humans 10 --num_humans_range 10 20 --bootstrap_easy_steps 0 --robot_vpref 1.0 --lr 1e-4 --num_envs 4 --horizon 64 --total_steps 4096 --holdout_scenarios paper_standard paper_challenging --holdout_episodes 2 --save_path ./.pytmp/smoke_v34.pt`
Expected: exit 0; no NaN/shape error (Beta sampling + update run end-to-end). Then remove `./.pytmp/smoke_v34.pt` and the smoke `logs/training_*.csv` it created.

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo/run_readiness.py sncp_ppo_colab.ipynb tests/test_post_run_pipeline.py tests/test_v16_run_readiness.py
git commit -m "v34 markers: notebook+readiness to Beta action dist, drop multi-head, v30 recipe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: (run-time, post-Colab) honest 5-seed eval vs v30

Deferred until v34 is trained on Colab and `sncp_ppo_v34.pt` is at the repo root.

- [ ] Verify checkpoint: `action_dist`=='beta' (no `actor_logstd`; `actor_mu` final out 4), pre_mlp, meanmax, node 128/48, no attn_heads buffer.
- [ ] Copy `scratch/_sweep_v33.py` → `_sweep_v34.py` (`CKPT='sncp_ppo_v34.pt'`, `OUT='v34_multiseed_result.json'`); run base-conda. 5 seeds × 50 ep at N=5/10/15/20, paper_challenging, robot 1.0, human 1.0, max_time None, goal_noise 0.
- [ ] Copy `scratch/_analyze_v33.py` → `_analyze_v34.py` (load v34, baseline v30: success 97.2/89.6/85.6/79.2, collision 2.8/10.4/14.4/20.8; Wilson CI + two-prop z + Bonferroni; success AND collision).
- [ ] **Decision rule:** Beta helps iff high-N (N=15/20) success rises and/or collision drops, no regression at N=5/10, timeout 0, vs v30. Report honestly (negatives included).
- [ ] Write verdict to `MEMORY.md` + chart.

---

## Self-review

- **Spec coverage:** Beta head (T1 3b), forward branch (T1 3d), helpers (T1 3e), init (T1 3c), auto-detect (T1 3f), ppo select_action + update branches (T2 3a/3b), train wiring (T2 3c), Gaussian byte-identical + regression test (T1 `test_gaussian_forward_still_returns_mu_std`, T2 reuses existing gaussian suites in the full run), markers + recipe (T3), honest eval (T4). All spec sections covered.
- **Placeholder scan:** none — every code step shows full code; commands have expected output.
- **Type consistency:** `action_dist` (str) identical in signature / `build_or_load_policy` / `build_policy_for_checkpoint`. `forward` returns `(out1, out2, value, hidden)` consumed as `out1,out2` in select_action and `all_p1/all_p2` in update; `_scale_action`/`_unscale_action`/`deterministic_action` signatures match call sites; `action_low`/`action_high` buffers used only in beta helpers. Gaussian path keeps `(mu,std)` semantics (out1=mu, out2=std).
