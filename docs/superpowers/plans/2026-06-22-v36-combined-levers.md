# v36 Birleşik Lever Koşusu — Uygulama Planı

> **For agentic workers:** TDD, tek tek task. Spec: `docs/superpowers/specs/2026-06-22-v36-combined-levers-design.md`.

**Goal:** v30 üzerine altı leveri birleştir (v36) + `--ent_coef` flag'i; auto-detect ile değerlendirilebilir checkpoint üret.

**Tech Stack:** PyTorch, ncps LTC, PPO; pytest (`--basetemp=./.pytmp`); miniconda base python.

---

### Task 1: `models.py` — sayı-ölçeklemesini çok-baş dikkate taşı

**Files:** Modify `sncp_ppo/models.py` (`_multihead_attention` ~satır 241); Test `tests/test_combined_v36.py`

- [ ] **Step 1: Failing test** — `attn_heads=4 + attn_count_scaling=True`: çok-baş skorları n ile ölçeklenmeli (farklı çıktı vermeli vs count_scaling kapalı); `attn_count_scaling=False` çok-baş byte-aynı kalmalı.
- [ ] **Step 2:** `_multihead_attention` içinde `scores = ... / sqrt(dh)` satırından sonra:
```python
if self.attn_count_scaling:
    scores = scores * H   # paper Eq 13 n-faktörü; tek-baştaki num_humans ile tutarlı
```
- [ ] **Step 3:** docstring "single-head only" ifadesini düzelt.
- [ ] **Step 4:** test geç.
- [ ] **Step 5:** commit.

### Task 2: `--ent_coef` flag'i (`train.py` + `ppo.py`)

**Files:** Modify `sncp_ppo/train.py` (parser + `PPOAgent(...)`); Test `tests/test_combined_v36.py`

- [ ] **Step 1: Failing test** — `build`/parser: `--ent_coef 0.001` → `agent.c2 == 0.001`; default 0.01.
- [ ] **Step 2:** parser: `parser.add_argument('--ent_coef', type=float, default=0.01)`; `PPOAgent(...)` çağrısına `c2=args.ent_coef` ekle.
- [ ] **Step 3:** test geç.
- [ ] **Step 4:** commit.

### Task 3: markerlar + notebook + readiness → v36

**Files:** Modify `sncp_ppo/run_readiness.py`, `sncp_ppo_colab.ipynb`, `tests/test_v16_run_readiness.py`, `tests/test_post_run_pipeline.py`

- [ ] **Step 1:** marker testlerini v35→v36 + yeni bayraklar (`--node_units 256`, `--attn_heads 4`, `--attn_count_scaling`, `--action_dist beta`, `--ent_coef 0.001`, `--num_humans_range 10 25`, `--sense_range 6.0`, `TOTAL_STEPS 4_000_000`, sncp_ppo_v36.pt, version 36, eval_v36). (red)
- [ ] **Step 2:** `run_readiness.py` TRAINING/EVALUATION tokenlarını v36'ya güncelle.
- [ ] **Step 3:** notebook v35→v36 (raw-text python, miniconda) — training cell tüm bayraklar; eval `--version 36`.
- [ ] **Step 4:** readiness pass + marker testler geç. commit.

### Task 4: birleşik auto-detect + suite + smoke

**Files:** `tests/test_combined_v36.py`

- [ ] **Step 1:** test — SNCPPolicy(node 256/96, attn_heads=4, attn_count_scaling, action_dist beta, sense_range 6, pre_mlp, meanmax) kur → state_dict → `build_policy_for_checkpoint` HEPSİNİ algılar (node 256/96, heads 4, count_scaling, beta, sense 6) → load missing/unexpected boş.
- [ ] **Step 2:** tam suite (`pytest --basetemp=./.pytmp`) yeşil.
- [ ] **Step 3:** CLI smoke: kısa `--total_steps 4096` tüm bayraklarla (beta + 4-baş + count + node256 + sense6 + ent_coef 0.001) exit 0, NaN yok.
- [ ] **Step 4:** readiness pass. commit.

### Finish
finishing-a-development-branch: testler geç → main'e merge + push (Colab pull eder).
