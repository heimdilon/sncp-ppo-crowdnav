# CfC-NCP side branch — optional Closed-form Continuous-time cells

> **Status:** code + tests. No multi-seed training, no "CfC beats LTC" claim.
> **Stack:** SNCP + v39 (risk head / Lagrangian PPO). No runtime action shield.
> **Default:** `cell_type='ltc'` — existing checkpoints and behavior are unchanged.

## 1. Why

The LTC cell is an ODE. `ncps.torch.LTC` integrates it with a numerical solver
on every step, which is the expensive part of `SNCPPolicy` on a Raspberry Pi.
Closed-form Continuous-time (CfC) cells (Hasani et al., *Nat Mach Intell* 2022;
arXiv:2106.13898; [raminmh/CfC](https://github.com/raminmh/CfC)) approximate that
solution in closed form. They live in the same NCP family and, in `ncps`, take
the same `AutoNCP` wiring as LTC:

```python
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP

wiring = AutoNCP(units=32, output_size=16, seed=48201)
ltc = LTC(input_size=2, units=wiring)
cfc = CfC(input_size=2, units=wiring)   # same hidden [B, units], readout [B, T, output_dim]
```

This branch is a **drop-in temporal encoder swap**, not a new observation or
reward experiment.

## 2. What changed

| Surface | Default (`ltc`) | CfC (`cfc`) |
|---|---|---|
| `SNCPPolicy(..., cell_type='ltc'\|'cfc')` | LTC modules `temporal_ltc` / `spatial_ltc` / `node_ltc` | CfC modules `temporal_cfc` / `spatial_cfc` / `node_cfc` |
| CLI | `--temporal_cell ltc` (also `--cell_type`) | `--temporal_cell cfc` |
| Checkpoint keys | historical `*_ltc.rnn_cell.gleak` etc. | `*_cfc.*` + `_cell_type_cfc` buffer |
| Forward contract | 4-tuple `(out1, out2, value, hidden)` | identical |
| Obs schema | `robot_node` 7, `spatial_edges` (H,6), `temporal_edges` [v, w] | unchanged |
| v39 `risk_head` / `cost_critic` | fusion-level, after the node cell | still works; trained the same way |
| Runtime shield | off | off |

All **three** NCP encoders swap together. Node fusion is also an LTC ODE in the
default stack; leaving it as LTC would keep the solver on the critical path.

Module names are intentionally different so `load_state_dict` cannot silently
transplant LTC weights into CfC cells (or the reverse). Helpers:

- `detect_cell_type(state_dict) -> 'ltc'|'cfc'`
- `assert_cell_type_compatible(policy, state_dict)`
- `load_policy_state_dict(policy, state_dict)` — compatibility check, then load
- `build_policy_for_checkpoint` auto-detects `cell_type` (eval / viz / waffle)

`--init_checkpoint` with an explicit `--temporal_cell` that does not match the
file is a hard error.

## 3. How to train

Same v39 recipe, one extra flag. Example (paper-regime-ish, CfC from scratch):

```bash
python -m sncp_ppo.train \
  --temporal_cell cfc \
  --risk_head --lagrange_ppo \
  --action_dist beta --pre_mlp --meanmax_pool \
  --num_humans 10 --total_steps 2500000 \
  --save_path checkpoints/sncp_ppo_cfc.pt
```

Warm-start from a **CfC** checkpoint:

```bash
python -m sncp_ppo.train \
  --temporal_cell cfc \
  --init_checkpoint checkpoints/sncp_ppo_cfc.pt \
  --risk_head --lagrange_ppo
```

Do **not** pass `--temporal_cell cfc --init_checkpoint checkpoints/sncp_ppo_v34.pt`.
LTC weights are not a CfC initialization; the loader refuses.

Eval / viz need no flag: `build_policy_for_checkpoint` reads the keys.

Smoke (CPU-ok):

```bash
python -m sncp_ppo.train --temporal_cell cfc --risk_head \
  --num_envs 2 --horizon 8 --total_steps 64 --eval_freq_updates 0 \
  --save_path /tmp/sncp_ppo_cfc_smoke.pt
python -m pytest tests/test_cfc_ncp.py tests/test_model.py tests/test_ncp_wiring.py \
  tests/test_v39_risk_head.py -q --basetemp=./.pytmp
```

## 4. Planned ablation (not this PR)

Same width, same wiring seeds, same v39 heads, same eval bank:

| Arm | Cell | Notes |
|---|---|---|
| A0 | LTC AutoNCP (current default) | champion / latency baseline |
| A1 | CfC AutoNCP | this branch |
| A2 | GRU of matching hidden width | "is the liquid inductive bias doing anything?" — **not implemented here** |

Report success / collision / timeout at N=5/10/15/20 **and** a Pi latency metric
(below). Do not declare a winner from a single seed.

## 5. Pi latency metric

The reason for CfC is wall-clock on the robot, not holdout success.

Suggested measurement (deterministic `eval()` forward, no shield, no viz):

1. Build the policy the waffle node actually runs (`build_policy_for_checkpoint`).
2. Warm up 20 steps, then time **N ≥ 200** forwards at the live batch size
   (usually 1) and the live human count (pad to the policy's trained H).
3. Record **median and p95 milliseconds per step** on the Pi, plus
   `1e3 / median_ms` as a rough Hz.
4. Compare A0 vs A1 on the **same** device, same torch build, same `num_threads`.

A run is only interesting for deployment if CfC p95 stays under the control
budget (dt = 0.25 s ⇒ 4 Hz hard floor; aim for comfortable headroom, e.g. p95
≪ 50 ms once perception is included). Success-rate gains that miss the budget
do not ship.

This PR does not collect those numbers.

## 6. Optional Colab cell

Not a notebook rewrite. Paste into a scratch cell after `git pull`:

```python
# CfC-NCP side experiment — does not change the default LTC training cell.
# CELL_TYPE = "ltc"   # champion / existing checkpoints
CELL_TYPE = "cfc"

# Train:
# !python -m sncp_ppo.train --temporal_cell {CELL_TYPE} --risk_head --lagrange_ppo \
#     --save_path checkpoints/sncp_ppo_{CELL_TYPE}.pt ...existing v39 flags...

from sncp_ppo.models import SNCPPolicy, build_policy_for_checkpoint, detect_cell_type
p = SNCPPolicy(cell_type=CELL_TYPE, risk_head=True)
print(p.cell_type, detect_cell_type(p.state_dict()), type(p._ncp("temporal")))
```

## 7. Non-goals

- Full multi-seed training or claiming CfC beats LTC
- Changing obs, reward, or bringing back the v38 runtime shield
- DS-RNN, distillation, or a GRU cell in this PR
