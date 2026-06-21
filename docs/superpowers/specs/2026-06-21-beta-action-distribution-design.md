# v34 — Beta action distribution (control-head lever, post-roadmap experiment)

Date: 2026-06-21
Branch: `feat/v34-beta-action`

## Goal

Replace the policy's Gaussian action distribution (unbounded `Normal(μ,σ)` + sigmoid/tanh-scaled
mean + env clipping) with a **Beta distribution** on the bounded action space, to remove the
boundary-clipping bias and give state-dependent, naturally-bounded control. Built on the v30
champion, **single-variable**: only the policy's output head + distribution change; the crowd
encoder/attention/fusion stay exactly v30. This is a deliberate move to a **different subsystem**
(the action head, not the encoder) after two capacity-add experiments on the encoder failed
(v31 node capacity ❌, v33 multi-head attention ❌ — both uniform regressions).

## Diagnosis (why Beta, why a different subsystem)

The pure-performance line: v27 (pre-MLP) ✅, v28 (curriculum) ✅, v30 (mean+max) ✅ were the wins;
v29 (attn-scaling) ❌, v31 (node 256) ❌, v32 (curriculum N→25 + 4M) flat, v33 (multi-head) ❌.
**Pattern: adding capacity to v30's crowd encoder regresses high-N** (single seed / 2.5M budget).
So the next lever must NOT add encoder capacity — it changes a different subsystem.

The action head is currently `Normal(μ,σ)` sampled in **physical** action units, then the env
**clips** to `[0,vpref]×[−wmax,wmax]`. This piles probability mass at the bounds (boundary bias)
and uses a single global `σ` (state-independent). At high N the robot must thread gaps with sharp,
decisive control; a Beta head is **naturally bounded** (no clip bias), **state-dependent** (α,β are
functions of the observation), and can represent skewed/peaked control near the bounds. The
high-N failure is pure collision, and better bounded control is a plausible (untested) lever for it.
(Chou, Maturana, Scherer 2017 — Beta policies beat Gaussian on bounded continuous control.)

## Verified ground truth (current code, 2026-06-21)

- `models.py` actor head: `self.actor_mu = Sequential(Linear(256,64), ReLU, Linear(64,2))`
  (final key `actor_mu.2.weight`, shape [2,64]); `self.actor_logstd = Parameter([[−2.0,−1.5]])`
  (models.py:114-126). `_init_linear_weights` orthogonal-inits the head and copies bias `[2.0,0.0]`
  to the final layer (a forward-velocity bias that was load-bearing only at the OLD vpref=0.26).
- `forward` (models.py:254-261): `mu_raw = actor_mu(sf)`; `v_mu = sigmoid(mu_raw[:,0:1])*robot_vpref`;
  `w_mu = tanh(mu_raw[:,1:2])*robot_wmax`; returns `(mu, std, value, hidden)` with
  `std = exp(actor_logstd).expand_as(mu)`.
- `ppo.py select_action` (191-222): `mu,std,value,h = policy(...)`; deterministic → `action=mu`
  (already in-bounds); else `Normal(mu,std).sample()` (stored UN-clipped to preserve the PPO ratio
  identity), `log_prob.sum(-1)`. `clip_action_for_env` (225) clips only for the env step.
- `ppo.py update` (462-496): BPTT loop calls `policy(step_obs, step_h)`, stacks `all_mu/all_std`,
  builds `Normal(all_mu, all_std)`, `new_log_probs = dist.log_prob(b_actions).sum(-1)`,
  `entropy = dist.entropy().sum(-1)`. `b_actions` are the stored physical (un-clipped) samples.
- `eval_report.py` (476) calls `agent.select_action(..., deterministic=True)` — insulated from the
  distribution type as long as `select_action` handles both.
- `build_policy_for_checkpoint` (models.py:283-299) auto-detects pre_mlp / attn_count_scaling /
  meanmax_pool / node dims / attn_heads from the state_dict.

## Design (single-variable: only the action head + distribution)

### New constructor arg `action_dist='gaussian'` (default = current, byte-identical)

`SNCPPolicy.__init__(..., action_dist='gaussian')`. When `'gaussian'`: **unchanged**
(`actor_mu`→2, `actor_logstd`). Every v14–v33 checkpoint loads byte-identically.

When `action_dist='beta'`:
- `self.actor_mu = Sequential(Linear(256,64), ReLU, Linear(64,4))` (raw α,β for the 2 action dims).
- **No** `actor_logstd`.
- `forward` returns `(alpha, beta, value, hidden)` where, from `raw = actor_mu(sf)` [B,4]:
  `alpha = softplus(raw[:, :2]) + 1`, `beta = softplus(raw[:, 2:]) + 1` (the `+1` forces α,β ≥ 1 →
  unimodal Beta, the standard RL parameterization; avoids U-shaped/bimodal control).
- Final-layer init: orthogonal gain 0.01, **bias 0** (no forward-velocity bias — at the paper-regime
  vpref=1.0 the symmetric init mean ≈ 0.5·vpref easily covers 8 m in the 50 s budget, so the old
  0.26-era sigmoid bias is unnecessary).

### Policy-owned action helpers (keep ppo.py thin, DRY between rollout and update)

- `self.action_dist = action_dist`; `self.action_low/high` = tensors `[0, −wmax]` / `[vpref, wmax]`.
- `_scale(x)`: maps `x ∈ [0,1]^2` → physical action `low + (high−low)·x`.
- `_unscale(a)`: physical action → `x = (a − low)/(high − low)` (clamped to (eps, 1−eps) for log_prob
  numerical safety).
- `deterministic_action(p1, p2)`: gaussian → `p1` (μ); beta → `_scale(α/(α+β))` (scaled Beta mean).

### `ppo.py` — branch on `self.policy.action_dist`; **Gaussian path stays byte-identical**

- `select_action`: if beta → build `Beta(alpha, beta)`, `x = dist.sample()`,
  `action = policy._scale(x)` (store physical, in-bounds), `log_prob = dist.log_prob(x).sum(-1)`;
  deterministic → `policy.deterministic_action(...)`. Else → existing Normal code unchanged.
- `update` BPTT loop: collect `(p1, p2)` per step (generic), stack; if beta →
  `x = policy._unscale(b_actions)`, `dist = Beta(all_p1, all_p2)`,
  `new_log_probs = dist.log_prob(x).sum(-1)`, `entropy = dist.entropy().sum(-1)`. Else → existing
  `Normal(all_mu, all_std)` path unchanged. The affine `_scale/_unscale` Jacobian is constant, so it
  cancels in the PPO ratio `exp(new_lp − old_lp)` as long as rollout and update both compute
  log_prob on `x ∈ [0,1]` (they do).

### Auto-detect

`build_policy_for_checkpoint`: `action_dist = 'beta' if 'actor_logstd' not in state_dict else
'gaussian'` (every Gaussian checkpoint has `actor_logstd`; Beta drops it). Pass to `SNCPPolicy`.

## Components / files

- `sncp_ppo/models.py` — `__init__` (arg + Beta head + helpers + skip actor_logstd),
  `forward` (beta branch returns α,β), `_init_linear_weights` (beta head init), `deterministic_action`,
  `_scale`/`_unscale`, `build_policy_for_checkpoint` (detect).
- `sncp_ppo/ppo.py` — `select_action` beta branch + `update` BPTT beta branch (gaussian untouched).
- `sncp_ppo/train.py` — `--action_dist {gaussian,beta}` arg + `build_or_load_policy` passthrough.
- `tests/test_beta_action.py` — NEW (red-first).
- `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`, `tests/test_post_run_pipeline.py`,
  `tests/test_v16_run_readiness.py` — v33→v34 markers; training cell = v30 recipe (the v33
  multi-head flag is dropped) **plus `--action_dist beta`**; `sncp_ppo_v34.pt`, `eval_v34`,
  `--version 34`.

No change to `crowd_env.py` or `eval_report.py` (the latter is insulated via `select_action`).

## Testing (TDD, red-first)

`tests/test_beta_action.py`:
1. `action_dist='beta'` build → `actor_mu` final out_features == 4, no `actor_logstd`.
2. `forward` (beta) returns `alpha, beta` with shape [B,2], all ≥ 1; `value` [B,1].
3. `select_action` (beta, stochastic + deterministic) returns actions within
   `[0,vpref]×[−wmax,wmax]` (naturally bounded, no clip needed).
4. `_scale(_unscale(a)) ≈ a` round-trip; deterministic action == scaled Beta mean.
5. Auto-detect roundtrip: save a beta `state_dict`, `build_policy_for_checkpoint` rebuilds a beta
   policy, `load_state_dict` with 0 missing/unexpected.
6. Gaussian-compat: a v30 (gaussian) `state_dict` builds a gaussian policy and loads.
7. **Regression guard:** a gaussian policy's `forward` still returns `(mu, std, value, hidden)` and a
   short `PPOTrainer.update` step runs unchanged (gaussian path byte-identical).
8. A tiny beta `PPOTrainer.update` runs end-to-end (log_prob/entropy/backward finite, no NaN).

Plus version-marker bumps v33→v34 (notebook + readiness + the 4 marker tests). Full suite green
(`--basetemp=./.pytmp`, `C:/ProgramData/miniconda3/python.exe`); readiness `pass` "v34 ... ready";
CLI smoke (`-m sncp_ppo.train --pre_mlp --meanmax_pool --action_dist beta --num_humans_range 10 20
--total_steps 4096 ...`) exits 0.

## Evaluation (run-time, post-Colab)

Same honest protocol (base-conda, 5 seeds 100–500 × 50 ep at N=5/10/15/20, paper_challenging,
robot 1.0, human 1.0, max_time None, goal_noise 0) on `sncp_ppo_v34.pt`. Compare to the **v30
baseline** (success 97.2/89.6/85.6/79.2; collision 2.8/10.4/14.4/20.8) with Wilson CIs +
two-proportion z (Bonferroni α=0.0125), both success and collision (reuse `_sweep_v33`→`_sweep_v34`,
`_analyze_v33`→`_analyze_v34`, CKPT `sncp_ppo_v34.pt`).

**Decision rule:** Beta helps iff high-N success rises and/or collision drops (esp N=15/20) with no
regression at N=5/10 and timeout 0, vs v30. A flat/negative result is reported honestly. Write the
verdict to memory.

## Out of scope / deferred

- tanh-squashed Gaussian (the lighter alternative; user chose Beta).
- State-dependent std for the Gaussian (subsumed — Beta is inherently state-dependent).
- Encoder/attention/capacity changes (v29/v31/v33 territory).
- Multi-seed training (separate robustness track).

## Irreversibility note

Model + training-loop code; v34's checkpoint uses the Beta head (auto-detected; v14–v33 stay
loadable). Work on `feat/v34-beta-action`; merge to `main` + push only at the finishing step after
the user confirms (Colab pulls `main`).

## Honest caveats (carried into the verdict)

- **More invasive than prior levers:** this touches the PPO training loop (`ppo.py`), not just
  `models.py`. The Gaussian path is kept byte-identical and locked by a regression-guard test, but
  the surface area is larger.
- **Entropy scale changes:** Beta entropy is on a different scale than Gaussian entropy, so the fixed
  entropy coefficient `c2` has an implicitly different regularization strength. Kept as-is for
  single-variable cleanliness; flagged because a flat result could partly reflect entropy-bonus
  mismatch rather than the distribution itself.
- **Single training seed** (as for v27–v33); ±~5pp swings can be partly seed noise.
- **Diminishing returns:** four consecutive non-wins precede this; Beta fixes a real bias but whether
  that moves the high-N collision tail is genuinely open. A negative is a real possible outcome.
- 2.5M budget matched to v30 (Beta has ~equal params, so undertraining risk is low, unlike v33).
