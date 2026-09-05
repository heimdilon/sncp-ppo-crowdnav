# M2′ SparseHIG — top-k human–human graph on CfC + RH

> **Status:** code + tests. No multi-seed training, no “SparseHIG beats dense HH” claim.
> **Stack:** SNCP + optional CfC (`--temporal_cell cfc`) + v39 risk head / Lagrangian PPO.
> **Default:** SparseHIG **off**. LTC + no `--sparse_hig` keeps existing checkpoints loadable.
> **Inference:** one forward. No v38 runtime action shield.

## 1. What it is

v37 `hh_intent_graph` already adds a gated human–human residual **before** the
existing robot←human (RH) `_attention_pool`:

```text
Z = LayerNorm(M_rh + CV_embed)
HH = MultiheadAttention(Z, Z, Z)          # full H×H
M_rh ← M_rh + tanh(hh_gate) * HH
u_att = RH_attention_pool(M_rh, m_rr)     # unchanged
```

That full MHA is the real path (not a HEIGHT transformer). On a Pi / Edge box
H×H is the wrong cost model once H grows; the constraint is **k ≤ 3–4 neighbors**.

**SparseHIG** keeps the CV features, residual, and `hh_gate`, but each human
attends only to its **k nearest others** (Euclidean on `spatial_edges[..., :2]`).
Self is never a neighbor. When `H < k` the extra slots are padded and masked.
`sense_range`, if set, hides humans outside the radius from both queries and keys.

Module names stay distinct (`hh_sparse_attn` vs v37 `hh_attn`) so dense HH and
SparseHIG weights cannot silently mix.

## 2. Flags

| Flag | Default | Meaning |
|---|---|---|
| `--sparse_hig` | off | Enable top-k HH. Implies `--hh_intent_graph`. |
| `--hh_topk` | 3 | Neighbor count `k ∈ {0,1,2,3,4}`. `k=0` is an identity residual. |
| `--hh_intent_graph` | off | Dense v37 H×H (unchanged) when SparseHIG is off. |
| `--temporal_cell {ltc,cfc}` | `ltc` | Orthogonal CfC encoder swap. |
| `--risk_head` / `--lagrange_ppo` | off | v39 fusion-level risk; still one forward. |

Ctor: `SNCPPolicy(sparse_hig=True, hh_topk=3, cell_type='cfc', risk_head=True)`.

Helpers (same spirit as CfC cell-type checks):

- `detect_sparse_hig(state_dict) -> bool`
- `detect_hh_topk(state_dict) -> int`
- `assert_sparse_hig_compatible(policy, state_dict)`
- `load_policy_state_dict` also refuses SparseHIG ↔ dense HH mixes and k mismatches
- `build_policy_for_checkpoint` auto-detects SparseHIG (eval / viz / waffle)

`--init_checkpoint` of a SparseHIG file needs no extra flag (auto-detect).
`--init_checkpoint` of a dense / no-HH file **plus** `--sparse_hig` is a hard
error — use `--upgrade_checkpoint --sparse_hig` to attach a fresh zero-gated branch.

## 3. Train smoke

```bash
python -m sncp_ppo.train --temporal_cell cfc --risk_head --sparse_hig --hh_topk 3 \
  --num_envs 2 --horizon 8 --total_steps 64 --eval_freq_updates 0 \
  --save_path /tmp/sncp_ppo_sparsehig_smoke.pt

python -m pytest tests/test_m2_sparsehig.py tests/test_cfc_ncp.py \
  tests/test_v37_intention_graph.py tests/test_v39_risk_head.py -q --basetemp=./.pytmp
```

Eval / viz: `build_policy_for_checkpoint` reads `_hh_sparse_k` / `hh_sparse_attn.*`.

## 4. Non-goals

- No training-win claim, no multi-seed ablation
- No HEIGHT / `nn.Transformer*` full-graph module
- No second inference forward, no v38 action shield on deploy paths
- No observation or reward change (`robot_node` 7, spatial `(H,6)`, temporal `[v,w]`)
- Dense v37 `--hh_intent_graph` without `--sparse_hig` stays the full H×H path
