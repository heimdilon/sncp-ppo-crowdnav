# v33 — Multi-head cross-attention (high-N generalization, pure-perf experiment #4)

Date: 2026-06-21
Branch: `feat/v33-multihead-attention`

## Goal

Improve high-N (N=15/20) success and/or reduce collision by replacing the single-head
attention pool with **multi-head canonical cross-attention** (4 heads). Built on the v30
champion (mean+max pooling). **Model-only, single-variable** (only the attention block
changes; everything else = v30's recipe). This continues the pure-performance line after the
planned 3-lever roadmap (#1 mean+max ✅ directional → #2 node capacity ❌ → #3 budget/curriculum
v32 ❌/flat); the user chose to brainstorm a fresh model lever, and selected multi-head attention.

## Diagnosis (why multi-head)

v30 honest 5-seed (the champion): success 97.2/89.6/85.6/79.2, collision 2.8/10.4/14.4/20.8
(N=5/10/15/20); timeout 0. N=10 is now ≈ paper (~94%); the remaining gap is the **high-N
collision tail** (N=20 ~15pp vs paper's ~94%). v32 (curriculum N→25 + 4M) was flat there.

The current pool produces a **single softmax over humans** → one compromise weighting of the
crowd. As N grows the robot faces **multiple simultaneous threats from different bearings**
(a left-crossing pedestrian and a right-approaching one). A single attention distribution must
either lock onto the single most-threatening agent (ignoring the others) or spread out
(diluting — the washout v30's max branch partially fixed). **Multi-head** attention gives each
head its own Q/K (and V/O) projection, so heads can specialize (nearest-frontal threat / lateral
crossing / goal-direction clearance) and the fusion node receives a multi-faceted threat picture
instead of one scalar weighting. This capability scales with N — directly targeting N=15/20.

## Verified ground truth (current code, 2026-06-21)

- `_attention_pool` (models.py:180-195): `Q=W_q(M_rh)` [B,H,64], `K=W_k(m_rr)` [B,1,64],
  `scores = bmm(Q,Kᵀ)/8` [B,H,1], `alpha = softmax(scores, dim=1)`, `a_mean = Σ_h alpha·M_rh`
  [B,256] (**Value = raw M_rh, no W_v**). v30 (`meanmax_pool`): `a_max = M_rh.max(dim=1)` [B,256],
  `return pool_merge(cat([a_mean, a_max]))` (`pool_merge`: Linear(512,256), models.py:100-101).
- `W_q`, `W_k`: `Linear(256, 64)` (models.py:98-99). Robot temporal feature `m_rr` is the key,
  per-human spatial features `M_rh` are the queries/values.
- `attn_count_scaling` (v29, ❌): multiplies scores by n; registered buffer `_attn_count_scaling`
  used for auto-detect. Default off (v33 does not use it).
- `build_policy_for_checkpoint` (models.py:283-299) auto-detects `pre_mlp` (temporal_pre_mlp keys),
  `attn_count_scaling` (`_attn_count_scaling`), `meanmax_pool` (pool_merge keys),
  `node_units`/`node_output` (gleak/output_w shapes) from the state_dict.
- `train.py`: `build_or_load_policy` (line 261-269) passes flags via `getattr`; parser args at
  1096-1117 (`--pre_mlp`, `--attn_count_scaling`, `--meanmax_pool`, `--node_units`, `--node_output`).
- v30 training recipe = `--pre_mlp --meanmax_pool --num_humans_range 10 20 --total_steps 2_500_000`
  + node 128/48 (the v28 curriculum). **v33 reverts v32's N→25 / 4M** (those were flat) and adds
  only the new attention flag — single-variable vs the v30 champion.

## Design (model-only, single-variable)

### New constructor arg `attn_heads=1` (default 1 = current single-head, byte-identical)

`SNCPPolicy.__init__(..., attn_heads=1)`. When `attn_heads == 1`: **unchanged** — `W_q`/`W_k`
stay `Linear(256,64)`, no `W_v`/`W_o`, no buffer. Every v14–v32 checkpoint loads byte-identically.

When `attn_heads > 1` (canonical multi-head cross-attention, d_model=256 preserved):
- Layers: `W_q = W_k = W_v = Linear(256, 256)`, `W_o = Linear(256, 256)`.
- Register buffer `_attn_heads = tensor(float(attn_heads))` so the head count is persisted and
  auto-detectable (heads are not recoverable from any weight shape; `d_head = 256 // attn_heads`).
- In `_attention_pool`, robot is the single **query** token, humans are **key/value** tokens:
  - `Q = W_q(m_rr)` → [B,1,256] → view [B, heads, 1, d_head]
  - `K = W_k(M_rh)` → [B,H,256] → view [B, heads, H, d_head]
  - `V = W_v(M_rh)` → [B, heads, H, d_head]
  - `scores = Q·Kᵀ / sqrt(d_head)` → [B, heads, 1, H]; `alpha = softmax(scores, dim=-1)`
  - `ctx = alpha·V` → [B, heads, 1, d_head] → concat heads → [B, 256]
  - `a_attn = W_o(ctx)` → [B, 256]
  - With `meanmax_pool` (v33 always on): `a_max = M_rh.max(dim=1)` [B,256];
    `return pool_merge(cat([a_attn, a_max]))` — **pool_merge stays Linear(512,256), shape unchanged.**
  - (`attn_count_scaling` path stays compatible — scores × n if ever on — but v33 keeps it off.)
- With 4 heads, `d_head = 64` and `sqrt(d_head) = 8` — same scale as the current single-head /8.

### `_init_linear_weights`: orthogonal-init `W_q/W_k/W_v/W_o` with gain √2 when `attn_heads>1`
(matching the existing W_q/W_k/pool_merge init pattern).

### `build_policy_for_checkpoint`: `attn_heads = int(state_dict['_attn_heads'])` if present else 1
(then passed to `SNCPPolicy`). Presence of `_attn_heads`/`W_v`/`W_o` keys distinguishes the variant.

### `train.py`
- `build_or_load_policy`: add `attn_heads=getattr(args, 'attn_heads', 1),` to the fresh-policy call.
- parser: `--attn_heads` (int, default 1).

## Components / files

- `sncp_ppo/models.py` — `__init__` (arg + conditional MHA layers + `_attn_heads` buffer),
  `_attention_pool` (multi-head branch), `_init_linear_weights` (init MHA layers),
  `build_policy_for_checkpoint` (detect `_attn_heads`).
- `sncp_ppo/train.py` — `build_or_load_policy` passthrough + `--attn_heads` arg.
- `tests/test_multihead_attn.py` — NEW (red-first).
- `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`, `tests/test_post_run_pipeline.py`,
  `tests/test_v16_run_readiness.py` — v32→v33 markers; training cell reverts to v30 recipe
  (`--num_humans_range 10 20`, `TOTAL_STEPS = 2_500_000`, drop node flags) **plus `--attn_heads 4`**;
  `sncp_ppo_v33.pt`, `eval_v33`, `--version 33`.

No changes to `crowd_env.py`, `eval_report.py`, `ppo.py` (the std/action sampling is unchanged;
only the pooled feature vector feeding the node LTC changes).

## Testing (TDD, red-first)

`tests/test_multihead_attn.py`:
1. `attn_heads=4` build → `W_v`/`W_o` exist, `_attn_heads` buffer == 4; `attn_heads=1` build → no
   `W_v`/`W_o`/buffer (byte-compat surface).
2. Forward pass with 4 heads returns `mu` [B,2], `std` [B,2], `value` [B,1], valid hidden dict.
3. Auto-detect roundtrip: save a 4-head `state_dict`, `build_policy_for_checkpoint` reconstructs a
   4-head policy that `load_state_dict`s with no missing/unexpected keys.
4. v30-compat: a `meanmax_pool` state_dict WITHOUT `_attn_heads` builds a single-head policy and loads.
5. Mechanism (multi-head is live): for a batch with several humans, the per-head attention
   distributions are not all identical (heads differentiate).

Plus: bump `test_notebook_is_v32_*` → `test_notebook_is_v33_multihead_attention` and the 3 readiness
tests v32→v33. Full suite green (`--basetemp=./.pytmp`, `C:/ProgramData/miniconda3/python.exe`).
Readiness preflight returns `pass` "v33 ... ready". CLI training smoke
(`--pre_mlp --meanmax_pool --attn_heads 4 --num_humans_range 10 20 --total_steps 4096 ...`) exits 0.

## Evaluation (run-time, post-Colab)

Same honest protocol (base-conda, 5 seeds 100–500 × 50 ep at N=5/10/15/20, paper_challenging,
robot 1.0, human 1.0, max_time None, goal_noise 0) on `sncp_ppo_v33.pt`. Compare to the **v30
baseline** (success 97.2/89.6/85.6/79.2; collision 2.8/10.4/14.4/20.8) with Wilson CIs +
two-proportion z (Bonferroni α=0.0125), both success and collision (reuse `_analyze_v32` →
`_analyze_v33`; `_sweep_v32` → `_sweep_v33`, CKPT `sncp_ppo_v33.pt`).

**Decision rule:** multi-head attention helps iff high-N success rises and/or collision drops (esp
N=15/20) with no regression at N=5/10 and timeout 0, vs v30. A flat/negative result is reported
honestly. Write the verdict to memory (`sncp-paper-vs-impl.md` / `MEMORY.md`).

## Out of scope / deferred

- `attn_count_scaling` (v29 ❌), node capacity (v31 ❌), curriculum N→25 / 4M (v32 flat — reverted).
- Beta / tanh-squashed action distribution, state-dependent std (other brainstorm options, not chosen).
- Multi-seed training (user declined earlier).

## Irreversibility note

Model code; v33's checkpoint uses the new MHA architecture, auto-detected on load (v14–v32 stay
loadable). Work proceeds on `feat/v33-multihead-attention`; merge to `main` + push only at the
finishing step after the user confirms (Colab pulls `main` to train v33).

## Honest caveats (carried into the verdict)

- **Capacity is bundled with structure:** canonical MHA adds `W_v`/`W_o` and widens `W_q`/`W_k`
  (256→64 ⇒ 256→256). A win is "multi-head + its params," not pure structure. BUT unlike v31 there
  is **no AutoNCP topology-reroll randomness** (deterministic orthogonal init), so v31's actual
  confound is absent. The minimal Value=raw variant (which would isolate structure) was offered and
  not chosen.
- **Single training seed** (as for v27–v32); ±~5pp swings can be partly seed noise.
- **Budget matched to v30 (2.5M):** v30 was ~peak=final at 2.5M, but MHA has more params and could
  want more steps; a null result could in principle be undertraining.
- v30 itself was not Bonferroni-significant over v28; the bar here is a high-N improvement vs v30.
