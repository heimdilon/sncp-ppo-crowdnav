# v31 — Node-fusion LTC capacity 128→256 (high-N collision, experiment #2)

Date: 2026-06-20
Branch: `feat/v31-node-capacity`

## Goal

Widen the node-fusion NCP (the decision-integration circuit) from `AutoNCP(128, 48)` to
`AutoNCP(256, 96)`, built on the v30 champion (mean+max pooling), to test whether decision
capacity is the remaining bottleneck for high-N collisions. This is experiment #2 of the
pure-performance roadmap (#1 mean+max ✅ directional → **#2 capacity** → #3 budget/curriculum),
single-variable: the node circuit is the only change. This spec covers #2 only.

## Diagnosis (why node capacity, why now)

v30 (mean+max) is the honest champion by point estimate — it weakly dominates v28 at every
density (success ≥, collision ≤, no regression, timeout 0) and directionally confirmed the
high-N collision-reduction mechanism (N=15 collision 20.8→14.4, N=20 27.2→20.8), but not at
Bonferroni significance (N=15 collision drop p=0.060, N=20 p=0.094 — underpowered at n=250 /
single training seed). v30 honest 5-seed: success 97.2/89.6/85.6/79.2, collision
2.8/10.4/14.4/20.8 (N=5/10/15/20).

The remaining high-N failure is still collision. v30 fixed the crowd-*summary* washout (pooling);
the next candidate bottleneck is the *decision* that maps the pooled summary → action. The node
NCP fuses 640 dims (robot 128 + temporal 256 + attention 256) through only 128 neurons
(`models.py:98` comment already flags the 640→128 squeeze as a historical capacity concern).
Widening it gives the policy more room to map dense-crowd summaries to safe actions.

## Verified ground truth (current code + checkpoint)

- `sncp_ppo/models.py:98-100`: `node_wiring = AutoNCP(units=128, output_size=48, seed=48203)`;
  `node_ltc = LTC(input_size=640, units=node_wiring)`; `node_proj = nn.Linear(node_wiring.output_dim, 256)`.
- The node LTC tensor shapes in a checkpoint (verified on `sncp_ppo_v30.pt`) make capacity
  recoverable: `node_ltc.rnn_cell.gleak` is `(units,)` → `(128,)`; `node_ltc.rnn_cell.output_w` is
  `(output_size,)` → `(48,)`; `node_proj.weight` is `(256, output_size)` → `(256, 48)`.
- `build_policy_for_checkpoint` (`models.py:266`) already auto-detects `pre_mlp` (temporal_pre_mlp
  keys), `attn_count_scaling` (`_attn_count_scaling`), `meanmax_pool` (`pool_merge`). v31 adds node
  size inference alongside these.
- `node_proj` reads `node_wiring.output_dim`, so it resizes automatically when `node_output` changes.
- Eval (`eval_report.evaluate_density`) and IL warm-start build the policy via
  `build_policy_for_checkpoint`, so node-size inference makes a v31 checkpoint load with no caller change.

## Design

### 1. Architecture (`models.py`)

- `SNCPPolicy.__init__` gains `node_units: int = 128, node_output: int = 48` (defaults = current).
  - `self.node_units, self.node_output = node_units, node_output`
  - `self.node_wiring = AutoNCP(units=node_units, output_size=node_output, seed=48203)`
  - `node_ltc` and `node_proj` follow unchanged (input 640 fixed; `node_proj` reads `output_dim`).
- temporal (32/16), spatial (48/24), attention (W_q/W_k 64), robot MLP, meanmax `pool_merge`,
  pre_mlp — all unchanged. Default path (`node_units=128, node_output=48`) is byte-identical.
- v31 run: `node_units=256, node_output=96` (keeps the ~0.375 motor ratio, 48/128 = 96/256).

### 2. Auto-detect + CLI

- `build_policy_for_checkpoint`:
  - `node_units = state_dict['node_ltc.rnn_cell.gleak'].shape[0] if 'node_ltc.rnn_cell.gleak' in state_dict else 128`
  - `node_output = state_dict['node_ltc.rnn_cell.output_w'].shape[0] if 'node_ltc.rnn_cell.output_w' in state_dict else 48`
  - passed to `SNCPPolicy(...)` alongside pre_mlp/attn_count_scaling/meanmax_pool.
  - v14–v30 checkpoints (node 128/48) infer the defaults → load unchanged.
- `train.py`: `--node_units` (int, default 128) and `--node_output` (int, default 48);
  `build_or_load_policy` reads `getattr(args, 'node_units', 128)` / `getattr(args, 'node_output', 48)`
  and passes them when building a fresh policy.

### 3. Single-variable v31 config

Identical to v30 except the node size:
`--pre_mlp --meanmax_pool --num_humans_range 10 20 --fixed_scenario paper_challenging
--num_humans 10 --bootstrap_easy_steps 200000 --robot_vpref 1.0 --lr 1e-4 --total_steps 2_500_000
--holdout_scenarios paper_standard paper_challenging --holdout_episodes 50
--node_units 256 --node_output 96` → `--save_path checkpoints/sncp_ppo_v31.pt`.

## Components / files

- `sncp_ppo/models.py` — `node_units`/`node_output` args, `node_wiring` from them,
  `build_policy_for_checkpoint` node-size inference.
- `sncp_ppo/train.py` — `--node_units` / `--node_output` parser args + thread through `build_or_load_policy`.
- `tests/test_node_capacity.py` (NEW) — see Testing.
- `sncp_ppo/run_readiness.py` — v30→v31 markers + `--node_units` / `--node_output` tokens.
- `sncp_ppo_colab.ipynb` — training cell `--node_units 256 --node_output 96` + v30→v31 paths.
- Version-marker tests (`tests/test_post_run_pipeline.py`, `tests/test_v16_run_readiness.py`) — v30→v31.

## Testing (TDD, red first)

`tests/test_node_capacity.py`:
1. **Default unchanged + compat:** `SNCPPolicy().node_units == 128`; a v18 checkpoint (skip-guarded
   if absent) loads into the default policy.
2. **Widened build:** `SNCPPolicy(node_units=256, node_output=96)` builds; `node_wiring.units == 256`;
   `node_proj.weight.shape == (256, 96)`.
3. **Forward shapes + action bounds** with `node_units=256, node_output=96` (mu∈[0,vpref]×[−wmax,wmax], finite).
4. **Auto-detect roundtrip:** save a node-256/96 policy → `build_policy_for_checkpoint` rebuilds with
   `node_units=256, node_output=96` and `load_state_dict` does not raise.
5. **v30-compat:** a default (node-128) state_dict → `build_policy_for_checkpoint` infers 128/48 → loads.
6. **CLI + build wiring:** `build_parser().parse_args(['--node_units','256','--node_output','96'])`
   yields those ints; `build_or_load_policy` returns a policy with `node_units == 256`.
7. **Coexistence:** `SNCPPolicy(pre_mlp=True, meanmax_pool=True, node_units=256, node_output=96)` forward runs.

Full suite stays green (`--basetemp=./.pytmp`, interpreter `C:/ProgramData/miniconda3/python.exe`).
Tiny CLI training smoke with `--pre_mlp --meanmax_pool --node_units 256 --node_output 96
--num_humans_range 10 12 ... --total_steps 4096` exits 0 (verifies AutoNCP(256,96) is valid and the
640→256 node LTC trains).

## Evaluation (run-time, post-Colab)

Same honest protocol (base-conda, 5 seeds 100–500 × 50 ep at N=5/10/15/20, paper_challenging,
robot 1.0, human 1.0, max_time None, goal_noise 0) on `sncp_ppo_v31.pt`. Compare to the **v30
baseline** (success 97.2/89.6/85.6/79.2; collision 2.8/10.4/14.4/20.8) with Wilson CIs +
two-proportion z (Bonferroni α=0.0125), reporting BOTH success and collision per density (reuse
`scratch/_analyze_v30.py` pattern → `_analyze_v31.py`).

**Decision rule:** node capacity helps iff high-N collision drops further and/or success rises
(esp N=15/20) with no regression at N=5/10 and timeout 0 — ideally pushing v30's near-significant
high-N collision reduction over the Bonferroni bar. A flat/negative result is reported honestly as a
clean negative; then experiment #3 (training budget / curriculum reach) follows.

## Out of scope / deferred

- Widening the spatial (48/24) or temporal (32/16) encoders — separate experiments if #2 helps but is
  insufficient (cleaner attribution knowing node capacity already helped). Auto-detect added for the
  node circuit only (YAGNI); spatial/temporal inference can be added when an experiment needs it.
- Experiment #3 (training budget / curriculum reach to N=25) — separate spec→plan→run after #2's verdict.
- Multi-seed training to de-confound single-seed runs (user declined).

## Irreversibility note

Checkpoint-compatible by default (node-128 checkpoints infer the defaults and load unchanged; the
bigger circuit is built only when `node_units/node_output` are passed). Work proceeds on
`feat/v31-node-capacity`; merge to `main` + push happens only at the finishing step after the user
confirms (Colab pulls `main` to train v31).

## Honest caveats (carried into the verdict)

- AutoNCP re-rolls its sparse topology when `units` changes, so v31 is a *larger, differently-wired*
  circuit, not pure added capacity — an inherent confound of the architecture.
- Single training seed (as for v27–v30); swings up to ~±7pp can be partly seed noise.
- v30 itself was not a Bonferroni-significant win over v28; v31 is measured against v30.
