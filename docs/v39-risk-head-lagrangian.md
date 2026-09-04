# v39 — Pi-friendly risk head + Lagrangian / constrained PPO

> **Durum:** kod + testler. Tam çok-seed sweep bu PR'nin parçası değil.
> **Kuzey yıldızı:** kısa ufuklu çarpışma kaçınmasını *öğrenmeye* gömmek.
> Runtime action shield (v38) eval-only oracle tavanı olarak kalır; **deploy
> edilen politika tek bir `SNCPPolicy` forward'udur.**

## 1. Gerekçe

v38, kilitli v34-fixed-beta aksiyonunu 1.5 saniyelik constant-velocity (CV)
lattice ile süzerek high-N çarpışmayı düşürdü. Kullanıcı bunu runtime "hile"
olarak reddediyor: Raspberry Pi'de aday tarama döngüsü yok, tek forward pass var.

v39 aynı kısa-ufuk geometriyi **ayrıcalıklı eğitim etiketi** olarak kullanır ve
küçük bir risk kafası + Lagrangian kısıt ile politikaya pişirir.

## 2. Mimari

```
obs (sim şeması değişmez)
  robot_node [B,7] local
  spatial_edges [B,H,6]
  temporal_edges [B,2] = [v, w]
        │
        ▼
   SNCP fusion sf [B,256]
        │
        ├─ actor / critic  (mevcut)
        └─ risk_mlp 256→32→2     # yalnızca risk_head=True
              p_coll = sigmoid
              min_clearance = softplus
```

- Risk kafası ~8k parametre. Eski checkpoint'ler `_risk_head` / `risk_mlp.*`
  anahtarı olmadığı için `build_policy_for_checkpoint` ile byte-uyumlu yüklenir.
- Forward hâlâ 4-tuple `(out1, out2, value, hidden)` döner (waffle_ros / eval
  kırılmaz). Risk çıktıları `policy.last_p_coll` / `last_min_clearance` üzerindedir.
- **Non-goal:** inference'da `shield_action`, aday lattice, çok-adımlı CV döngüsü
  yok. `models.py` `action_shield` import etmez.

## 3. Offline labeler

`sncp_ppo/risk_labeler.py` v38'in CV rollout'unu (`horizon=6`, `dt=0.25s`)
**yalnızca eğitimde** çağırır:

| Etiket | Anlam |
|---|---|
| `collision` | 1 eğer min predicted clearance `< 0` |
| `min_clearance` | `clip(max(raw, 0), 10)` — softplus kafasıyla uyumlu |

v38 `action_shield.shield_action` eval oracle olarak durur; eğitim/deploy
yolu onu çağırmaz.

## 4. Kayıplar

```
L = L_PPO(A_r − λ A_c) + c1 L_V + c2 H
  + risk_bce_coef · BCE(p_coll, collision)
  + risk_clearance_coef · Huber(min_clearance, clearance)

λ ← clip(λ + α (E[collision] − d), 0, λ_max)
```

- `A_c`: ayrıcalıklı collision maliyeti üzerinde GAE; `p_coll` rollout değeri
  cost-critic bootstrap olarak kullanılır.
- `d = --lagrange_cost_limit` (varsayılan 0.05).
- `λ_init = 0`: ilk güncellemeler vanilla PPO + `L_risk`; çarpışma limiti
  aşılırsa dual yükselir.

## 5. CLI (varsayılanlar kapalı — v34 eğitimi değişmez)

```bash
python -m sncp_ppo.train \
  --risk_head --lagrange_ppo \
  --risk_horizon 6 \
  --risk_bce_coef 1.0 --risk_clearance_coef 0.1 \
  --lagrange_cost_limit 0.05 --lagrange_lr 0.01 \
  --init_checkpoint checkpoints/sncp_ppo_v34.pt \
  --action_dist beta --save_path checkpoints/sncp_ppo_v39.pt
```

`--lagrange_ppo` risk kafasını otomatik açar. `--init_checkpoint` eski bir
v34 ağırlığıysa taze risk kafası eklenir (`strict=False`, yalnızca `risk_mlp`
anahtarları eksik olabilir).

Smoke (kısa, GPU şart değil):

```bash
python -m sncp_ppo.train --risk_head --lagrange_ppo \
  --num_envs 2 --horizon 8 --total_steps 64 --eval_freq_updates 0 \
  --save_path /tmp/sncp_ppo_v39_smoke.pt --action_dist beta
python -m pytest tests/test_v39_risk_head.py tests/test_v38_action_shield.py \
  -q --basetemp=./.pytmp
```

Eval **shield OFF** olmalıdır (`evaluate_policy_report.py` varsayılanı zaten
`--action_shield` geçmeden).

## 6. Planlanan eval matrisi (bu PR koşmaz)

paper_challenging, N ∈ {5,10,15,20}, aynı seed bankası:

| Kol | Ne | Rol |
|---|---|---|
| C0 | v34 raw, shield OFF | taban |
| C1 | v34 + v38 shield ON | oracle tavan (öğrenilmeyen) |
| C2 | v39, shield OFF | asıl aday |

GO taslağı (preregister, henüz koşulmadı): C2 high-N success ≥ C0, collision
C0'dan düşük, timeout C0'ı bozmasın; C1'i geçmek iddia değil, tavan referansı.

Bu PR sonuç iddia etmez.

## 7. Non-goals

- Runtime action shield'ı deploy politikasına gömmek
- Obs şemasını değiştirmek
- Full 4M / multi-seed sweep
- Paper 93–95% iddiası
