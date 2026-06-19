# v30 — Mean+Max attention pooling (high-N collision fix)

Date: 2026-06-19
Branch: `feat/v30-meanmax-pooling`

## Goal

Reduce the high-N collision rate (the dominant remaining gap vs the paper) by replacing the
attention pooling's pure convex-combination output with a **mean + max** pooled
representation. This is the first of a user-approved sequence of pure-performance experiments
(no longer constrained by paper fidelity): **#1 mean+max pooling → #2 capacity → #3 training
budget/curriculum**, each single-variable, each evaluated before the next. This spec covers #1
only.

## Diagnosis (why this lever)

Honest 5-seed sweep of the champion v28 (250 ep/density, `paper_challenging`, robot 1.0):

| N | success | collision | timeout |
|---:|---:|---:|---:|
| 5 | 94.4% | 5.6% | 0% |
| 10 | 87.6% | 12.4% | 0% |
| 15 | 79.2% | 20.8% | 0% |
| 20 | 73.2% | 27.2% | 0% |

The gap is concentrated at high N and the failure mode is **collision, not timeout** — the robot
reaches toward the goal but cannot thread dense crowds. The paper reaches ~94% at N=10–20 in the
same regime, so the gap is model/method, not geometry.

**Root mechanism (verified in code, `models.py::_attention_pool`):** the pooled vector is
`u = Σ_h α_h · M_rh,h` with `α = softmax_h(Q·Kᵀ/√d_k)` — a convex combination of the per-human
256-dim features (the value is `M_rh` itself, no separate `W_v`). As N grows the softmax mass
spreads and `u` regresses toward the mean; duplicating similar pedestrians barely changes `u`, so
the single most-threatening agent's signal is diluted exactly when collisions happen. v29's Eq-13
count-scaling only rescaled this average's temperature (it failed: −6.8pp at N=5, no significant
high-N gain). Mean+max changes the pooling **operation**, which is the canonical cardinality-robust
fix (PointNet/DeepSet): a max branch preserves the most-salient agent's signal regardless of crowd
size.

## Verified ground truth (current code)

- `sncp_ppo/models.py::_attention_pool(M_rh, m_rr, num_humans)`:
  `Q = W_q(M_rh)` → [B,H,64]; `K = W_k(m_rr)` → [B,1,64]; `scores = Q·Kᵀ/8` (×`num_humans` iff
  `attn_count_scaling`); `α = softmax(dim=1)`; returns `bmm(M_rh.transpose, α)` → [B,256].
- `SNCPPolicy` already carries optional, default-off, auto-detected flags `pre_mlp` and
  `attn_count_scaling`; `build_policy_for_checkpoint` infers them from state-dict keys
  (`temporal_pre_mlp.*`, `_attn_count_scaling`). v30 follows this exact pattern.
- Training threads optional flags through `build_parser` → `build_or_load_policy`
  (`sncp_ppo/train.py`); the notebook trains via `python -m sncp_ppo.train`.
- Eval: `sncp_ppo/eval_report.py::evaluate_density` builds the policy via
  `build_policy_for_checkpoint`, so a v30 checkpoint auto-loads with the new pooling — the honest
  local sweep (`scratch/_sweep_*.py`, base-conda interpreter) needs no change beyond the checkpoint
  path.

## Design

### 1. Architecture (`models.py`)

- New constructor arg `meanmax_pool: bool = False`. When true, build
  `self.pool_merge = nn.Linear(512, 256)` (orthogonal init, gain √2, like the other projections).
- `_attention_pool` (when `meanmax_pool`):
  - `a_mean = bmm(M_rh.transpose(1,2), α).squeeze(2)`  → [B,256]  (unchanged convex combination)
  - `a_max  = M_rh.max(dim=1).values`                  → [B,256]  (element-wise max over humans)
  - `u_att  = self.pool_merge(torch.cat([a_mean, a_max], dim=1))`  → [B,256]
  - Default path (`meanmax_pool=False`) is byte-identical to today.
- `attn_count_scaling` and `meanmax_pool` are independent and may coexist (the count-scaling
  multiply still applies to `scores` before softmax); v30's run uses `meanmax_pool` only.

### 2. Flag wiring + auto-detect

- `SNCPPolicy.__init__(... meanmax_pool=False)`, stored as `self.meanmax_pool`.
- `build_policy_for_checkpoint`: `meanmax_pool = any(k.startswith('pool_merge') for k in state_dict)`,
  passed to the constructor (alongside the existing `pre_mlp`/`attn_count_scaling` detection).
- `train.py`: `build_parser` adds `--meanmax_pool` (store_true); `build_or_load_policy` reads
  `getattr(args, 'meanmax_pool', False)` and passes it when constructing a fresh policy.

### 3. Single-variable training config (v30)

Identical to the v28 champion except the one flag:
`--pre_mlp --num_humans_range 10 20 --fixed_scenario paper_challenging --num_humans 10
--bootstrap_easy_steps 200000 --robot_vpref 1.0 --lr 1e-4 --total_steps 2_500_000
--holdout_scenarios paper_standard paper_challenging --holdout_episodes 50 --meanmax_pool`
→ `--save_path checkpoints/sncp_ppo_v30.pt`. **No `--attn_count_scaling`** (v29 eliminated).

## Components / files

- `sncp_ppo/models.py` — `meanmax_pool` arg, `pool_merge` layer, `_attention_pool` branch,
  `_init_linear_weights` (orthogonal init for `pool_merge`), `build_policy_for_checkpoint` detect.
- `sncp_ppo/train.py` — `--meanmax_pool` parser arg + thread through `build_or_load_policy`.
- `tests/test_meanmax_pool.py` (NEW) — see Testing.
- `sncp_ppo/run_readiness.py` — v29→v30 markers + `--meanmax_pool` token.
- `sncp_ppo_colab.ipynb` — v29→v30 paths + `--meanmax_pool` in the training cell.
- Version-marker tests (`tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py`) —
  v29→v30, assert `--meanmax_pool`.

## Testing (TDD, red first)

`tests/test_meanmax_pool.py`:
1. **Washout test (key):** with shared weights, duplicating one human N→2N leaves mean-only `u`
   ~unchanged but changes mean+max `u` materially (`not allclose`). Locks the mechanism.
2. **Default off + checkpoint-compat:** `SNCPPolicy().meanmax_pool is False`; no `pool_merge` key in
   a default state-dict; a v18 checkpoint (skip-guarded if absent) loads into a default policy.
3. **Auto-detect roundtrip:** save a `meanmax_pool=True` policy → `build_policy_for_checkpoint`
   rebuilds with `meanmax_pool=True` and `load_state_dict` does not raise.
4. **Forward shapes + action bounds** with `meanmax_pool=True` (mu∈[0,vpref]×[−wmax,wmax], finite).
5. **CLI/build wiring:** `build_parser().parse_args(['--meanmax_pool']).meanmax_pool is True`;
   `build_or_load_policy` returns a policy with `meanmax_pool True`.
6. **Coexistence:** `SNCPPolicy(pre_mlp=True, meanmax_pool=True)` forward runs (both flags live).

Full suite must stay green (`python -m pytest -q --basetemp=./.pytmp`). Tiny CLI training smoke
with `--pre_mlp --meanmax_pool --num_humans_range 10 12 ... --total_steps 4096` exits 0.

## Evaluation (run-time, post-Colab)

Same honest protocol as v26–v29: base-conda interpreter, 5 seeds (100–500) × 50 ep at
N=5/10/15/20, `paper_challenging`, robot 1.0, human 1.0, `max_time=None`, goal_noise 0, on
`sncp_ppo_v30.pt`. Compare to the v28 baseline (94.4/87.6/79.2/73.2) with Wilson CIs + two-proportion
z-tests (Bonferroni α=0.0125).

**Decision rule:** mean+max helps iff high-N **collision drops** and success rises (especially
N=15/20) with **no regression at N=5/10** and timeout stays 0. A flat/negative result is reported
honestly as a clean negative; then experiment #2 (capacity) follows. Optional ~20-min local
prototype probe before Colab: confirm mean+max actually breaks the duplication washout on the v28
weights (live form of test #1).

## Out of scope / deferred

- Multi-head attention and the mean+max+multi-head hybrid — deferred follow-ups if #1 helps but is
  insufficient (cleaner attribution knowing max already helped).
- Experiments #2 (LTC capacity sweep) and #3 (training budget / curriculum reach to N=25) — separate
  spec→plan→run cycles after #1's verdict.
- A separate learned value projection (`W_v`) or a per-human salience MLP before max — kept out to
  keep #1 a single clean variable.
- Multi-seed training to de-confound single-seed runs (user declined).

## Irreversibility note

The architecture change is checkpoint-compatible by default (new layer only built when
`meanmax_pool=True`; existing v14–v29 checkpoints load unchanged). Work proceeds on
`feat/v30-meanmax-pooling`; merge to `main` + push happens only at the finishing step after the user
confirms (Colab pulls `main` to train v30).
