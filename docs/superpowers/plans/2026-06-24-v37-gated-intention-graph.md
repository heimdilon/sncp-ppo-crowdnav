# v37 Gated Intention Graph — Uygulama ve Deney Planı

> **Durum (2026-06-24): KOD PROBE'A HAZIR; v36 DEĞERLENDİRMESİ TAMAMLANDI, FULL RUN BAŞLAMADI.**
> v36 preregistered kapıyı geçmedi; v37 tabanı `v34-fixed-beta` olarak kilitlendi. Notebook/readiness
> marker'ları artık V37 paired-probe launcher'ını doğrular; probe `GO` olmadan tam v37 eğitimi
> başlatılmayacak.

**Amaç:** Yüksek yoğunlukta (N=15/20) kapanan geçitleri ve yaya-yaya etkileşimlerini açıkça modelleyerek başarıyı artırmak ve çarpışmayı azaltmak; mevcut en iyi politikanın düşük yoğunluk performansını korumak.

**Ana fikir:** Seçilecek şampiyon checkpoint'i sıfır davranış farkıyla yükle; mevcut per-human Spatial-LTC özelliklerine dört ufuklu constant-velocity geometri ekle; maskeli human-human self-attention çıktısını sıfırdan başlayan öğrenilebilir bir residual gate üzerinden mevcut robot-human attention'a ver.

**Tech stack:** Python, PyTorch, `nn.MultiheadAttention`, ncps/LTC, PPO, Gymnasium, pytest (`--basetemp=./.pytmp`).

---

## 1. Neden bu tasarım?

### Repo kanıtı

- Düzeltilmiş v34 Beta koşusu bugüne kadarki en güçlü yüksek-N nokta tahminlerini verdi: başarı N=5/10/15/20 için `96.8/92.8/91.2/86.0`, çarpışma `2.8/7.2/8.8/13.2`, timeout `0`.
- v31 node büyütme, v33 robot-human multi-head, v29 count-scaling ve v35 sense-range tek başlarına net kazanç vermedi. v37 bunları varsayılan olarak taşımayacak.
- Mevcut mimari her yayayı shared Spatial-LTC ile bağımsız kodluyor; robot-human attention'dan önce yaya-yaya mesajlaşması yok. Bu, yüksek-N'de boşluk/geçit geometrisini doğrudan temsil etmeyen gerçek bir mimari eksik.
- v36 birleşik 4M koşusu ve dürüst çok-tohumlu değerlendirmesi tamamlandı: verdict `NEGATIVE/FLAT`.
  Yüksek-N'de v34'ten açıkça gerilediği için checkpoint tabanı düzeltilmiş v34'tür.

### Birincil literatür kanıtı

- **Intention-Aware Crowd Navigation** aynı probleme yakın bir PPO/yoğun-kalabalık kurulumunda HH attention eklenince başarıyı `77% -> 89%`; constant-velocity gelecek konumları eklenince `67% -> 87%` raporluyor. Çalışma 20M step ve farklı benchmark kullandığı için büyüklük doğrudan taşınamaz, fakat yön güçlüdür: <https://arxiv.org/abs/2203.01821>.
- **Relational Graph Learning** higher-order human-human ilişkileri ve multi-step lookahead ile tek-adım modeline göre başarı/çarpışmayı iyileştiriyor: <https://arxiv.org/abs/1909.13165>.
- **DS-RNN** yoğun kalabalıkta ayrıştırılmış spatio-temporal etkileşim modellemesinin düz RNN+attention'a göre daha iyi genelleştiğini gösteriyor: <https://arxiv.org/abs/2011.04820>.
- **From Crowd Motion Prediction to Robot Navigation** daha karmaşık S-GAN tahmininin navigasyonda basit constant-velocity tahminini geçmediğini gösteriyor. Bu nedenle v37 ayrı bir ağır trajectory predictor kullanmayacak: <https://arxiv.org/abs/2303.01424>.
- **Social-NCE** collision-aware negative sampling ile BC çarpışmasını `11.11% -> 3.40%` düşürüyor ve off-policy RL sample efficiency'sini artırıyor: <https://arxiv.org/abs/2012.11717>. Ancak PPO/on-policy entegrasyonu bu projede doğrulanmadığı için v37 core kapsamına alınmayacak; aşağıdaki contingent v38 adayı olarak tutulacak.

### Hipotez

Yüksek-N darboğazı salt kapasite değil, **gelecek birkaç adımda hangi yaya-yaya etkileşiminin hangi geçidi kapatacağını temsil edememek**. Constant-velocity geometri ORCA kalabalığı için yeterli düşük-varyanslı niyet sinyali; HH self-attention bu sinyali tüm yaya çiftleri arasında paylaşır.

---

## 2. Değişmezler ve kapsam dışı işler

### Korunacaklar

- Doğru vektörize Beta yolu: `SNCPPolicy.make_action_dist`.
- `action_dist=beta`, `ent_coef=0.001`.
- `pre_mlp=True`, `meanmax_pool=True`.
- Default node kapasitesi `128/48` (seçilen base checkpoint v36 değilse).
- Mevcut Spatial-LTC, robot-human attention, Node-LTC ve actor/critic başları.
- Observation şeması değişmeyecek; CV özellikleri mevcut `[dx,dy,rel_vx,rel_vy]` alanlarından model içinde türetilecek.
- Pedestrians robotu görmeyecek; ORCA/environment rejimi değişmeyecek.

### v37 core kapsamı dışında

- Learned GST/S-GAN trajectory predictor.
- Social-NCE auxiliary loss (core gate geçtikten sonra ayrı v38 probe adayı).
- Yeni reward katsayısı veya comfort ayarı.
- v31 node büyütme, v29 count-scaling, v35 sense-range'i v34 tabanına yeniden eklemek.
- Benchmark veya karar kuralını sonuç görüldükten sonra değiştirmek.

---

## 3. Mimari sözleşme

### Yeni bayraklar

```text
--hh_intent_graph
--hh_attn_heads 4
--cv_horizons 1 2 3 4
--cv_dt 0.25
--upgrade_checkpoint checkpoints/sncp_ppo_<base>.pt
```

`--init_checkpoint` exact-architecture yükleme olarak kalacak. `--upgrade_checkpoint`, eski checkpoint'e yeni v37 branch'i ekleyen ayrı ve açık bir yol olacak.

### Tensor akışı

Mevcut spatial observation:

```text
spatial_edges[B,H,6] = [dx,dy,rel_vx,rel_vy,goal_dir_x,goal_dir_y]
```

Constant-velocity gelecek geometrisi:

```python
rel_pos = spatial_edges[..., 0:2]                 # [B,H,2]
rel_vel = spatial_edges[..., 2:4]                 # [B,H,2]
future_k = rel_pos + rel_vel * (k * cv_dt)         # k in [1,2,3,4]
cv_features = cat(future_1, ..., future_4, dim=-1) # [B,H,8]
```

Yeni branch:

```text
cv_features 8 -> 128 -> 256 (ReLU)
Z = LayerNorm(M_rh + CV_embed)
HH = MultiheadAttention(d_model=256, heads=4, batch_first=True)(Z,Z,Z)
M_rh_v37 = M_rh + tanh(hh_gate) * HH
```

- `hh_gate = nn.Parameter(torch.tensor(0.0))`.
- Gate sıfırken upgrade edilmiş politika base politika ile aynı actor parametrelerini ve value'yu üretmeli (`atol <= 1e-6`).
- `hh_gate` negatif veya pozitif öğrenebilir; `tanh` residual büyüklüğünü sınırlar.
- MHA human ekseni üzerinde çalışır; bu v33'teki robot-query/human-key multi-head ile aynı lever değildir.
- Mevcut sense mask varsa MHA key-padding mask olarak yeniden kullanılır. Tüm insanların maskeli olduğu satırlar NaN üretmeden sıfır HH residual dönmelidir.
- H=1 durumunda finite ve base-eşdeğer davranış korunmalıdır.

### Checkpoint auto-detect

Yalnız v37 açıkken şu buffer'lar persist edilir:

```text
_hh_intent_graph = 1
_hh_attn_heads = 4
_cv_horizons = [1,2,3,4]
_cv_dt = 0.25
```

`build_policy_for_checkpoint()` bunları okuyup v37'yi otomatik kuracak. v14-v36 state dict davranışı değişmeyecek.

### Upgrade yükleme güvenliği

`--upgrade_checkpoint`:

1. Base checkpoint mimarisini mevcut auto-detect ile belirler.
2. Aynı base kwargs + `hh_intent_graph=True` ile yeni politika kurar.
3. `strict=False` yükler.
4. `unexpected == []` olmasını zorunlu kılar.
5. `missing` listesinin yalnızca `cv_encoder.*`, `hh_norm.*`, `hh_attn.*`, `hh_gate` ve v37 buffer'larından oluşmasına izin verir; başka eksik anahtar varsa fail-fast.
6. Gate=0 equivalence testini geçmeden eğitim başlamaz.

---

## 4. v36 sonrası base seçimi — KİLİTLİ

v36 sweep tamamlandı ve preregistered verdict `NEGATIVE/FLAT` oldu. Seçilen base=`v34-fixed-beta`.

1. v36, v30'a karşı anlamlı high-N başarı artışı veya çarpışma düşüşü üretmedi; N=15/20 timeout sıfır değildi.
2. v34'e karşı N=15/20 başarı farkı `−9.2/−12.8 pp`, çarpışma farkı `+6.8/+10.0 pp` oldu.
3. Bu nedenle v36 warm-start olarak kullanılmayacak.
4. v34 checkpoint'inin 300k continuation koşusu probe kontrol kolu olacak; yeni branch'in kazancı bu continuation'a karşı ölçülecek.

---

## 5. Probe-first deney protokolü (tam v37'den önce zorunlu)

### Kollar

- **C0 — control continuation:** seçilen base checkpoint, yeni branch kapalı.
- **C1 — v37 core:** aynı checkpoint `--upgrade_checkpoint` ile, gated HH+CV branch açık.

Social-NCE bu aşamada üçüncü kol olarak eklenmeyecek. Önce core mekanizmanın öğrenip öğrenmediği izole edilecek.

### Eşit eğitim ayarları

```text
training seeds: 40, 41, 42
additional steps: 300_000 / seed
num_envs: 16
horizon: 128
fixed_scenario: paper_challenging
num_humans_range: 10 25
bootstrap_easy_steps: 0
robot_vpref: 1.0
lr: 5e-5              # warm-start fine-tune; scratch LR değildir
lr_end_factor: 0.5
target_kl: 0.01
holdout scenarios: paper_standard paper_challenging
holdout episodes: 50
action_dist: checkpoint-derived Beta
ent_coef: 0.001
```

### Sabit eval bankası

- Densities: `5,10,15,20`.
- Her training seed için aynı episode seed blokları kullanılacak.
- Probe: density başına en az 100 ortak vaka.
- Episode-level success/collision/timeout saklanacak; yalnız aggregate JSON yeterli değil.
- C0/C1 aynı vakalarda değerlendirileceği için paired bootstrap veya McNemar; ayrıca proje uyumluluğu için Wilson CI raporlanacak.

### GO kapısı

C1 tam v37 koşusuna yalnız şu koşulların tamamında gider:

1. N=15/20'de ortalama başarı en az `+3 pp` **veya** çarpışma en az `-3 pp`.
2. İyileşme yönü üç fine-tune seed'inin en az ikisinde aynı.
3. N=5/10 başarı düşüşü `>2 pp` veya çarpışma artışı `>2 pp` yok.
4. Timeout tüm density'lerde `0`.
5. `entropy`, `approx_kl`, `return_rms` finite; Beta global std alanlarının NaN olması beklenen davranış.
6. `abs(hh_gate) >= 0.01`; branch pratikte no-op kalmamış.
7. Best-final holdout-min farkı `<=5 pp`; belirgin fine-tune collapse yok.

Kapı fail ise tam v37 A100 koşusu yapılmaz. Base champion korunur ve sonuç negatif olarak kaydedilir.

---

## 6. Tam v37 koşusu

Probe GO verirse:

```text
base checkpoint: Section 4'te seçilen
upgrade: --hh_intent_graph --hh_attn_heads 4 --cv_horizons 1 2 3 4 --cv_dt 0.25
additional total_steps: 1_500_000
training seed: probe median seed'i (önceden seçilen tie-break: 42)
num_humans_range: 10 25
bootstrap_easy_steps: 0
lr: probe kazanan ayarı (default 5e-5)
lr_end_factor: 0.2
save_path: checkpoints/sncp_ppo_v37.pt
```

Base v34 ise toplam öğrenme bütçesi yaklaşık `2.5M + 1.5M = 4M` olur; fakat v37 sıfırdan başlamaz. Full run sırasında best-checkpoint seçimi devam eder.

Tam sweep: 5 seeds (`100,200,300,400,500`) x 50 episode x N=`5,10,15,20`; `paper_challenging`, robot/human vpref `1.0`, max_time env-derived, goal_noise `0`.

### Nihai karar

Seçilen base'e karşı:

- High-N N=15/20 başarı artışı ve/veya çarpışma düşüşü Bonferroni `alpha=0.0125` ile anlamlı.
- N=5/10'da anlamlı başarı/çarpışma gerilemesi yok.
- Timeout `0`.
- Trajectory görselleri N=10/20'de collision avoidance davranışını doğruluyor.
- Başarılı olsa bile sonuç farklı benchmark'lara doğrudan “paper SOTA” olarak sunulmayacak.

---

## 7. Uygulama görevleri (TDD)

### Task 0 — v36'yı bitir ve base'i kilitle

**Files:** Bu planın Run Record bölümü; v36 artifact'leri.

- [x] v36'yı preregistered 4M bütçeyle bitir.
- [x] `scratch/_sweep_v36.py` ve `_analyze_v36.py` çalıştır.
- [x] v34/v36 success-collision-timeout tablosunu doldur.
- [x] Base seçimini bu plana yaz: `v34-fixed-beta`.
- [ ] Plan ve post-run kayıtlarını commit et.

### Task 1 — Model contract testleri (RED)

**Create:** `tests/test_v37_intention_graph.py`

- [x] Default policy state dict byte-compatible yüzeyi korur; v37 buffer/modülleri yoktur.
- [x] `hh_intent_graph=True` buffer ve modülleri kurar.
- [x] CV horizon hesabı deterministic küçük tensörle doğrulanır.
- [x] H=1, H=10, H=25 forward finite.
- [x] Maskeli ve all-masked satırlar finite.
- [x] Gate=0 base equivalence (`alpha`, `beta`, `value`, hidden states).
- [x] Gate manuel `0.2` iken output base'den farklıdır; branch no-op değildir.
- [x] Gate backward sonrası finite gradient alır.
- [x] v37 state dict auto-detect round-trip missing/unexpected boş.

Run:

```powershell
C:\ProgramData\miniconda3\python.exe -m pytest tests/test_v37_intention_graph.py -q --basetemp=./.pytmp
```

### Task 2 — Gated HH+CV branch implementasyonu (GREEN)

**Modify:** `sncp_ppo/models.py`

- [x] Constructor args ve conditional modules.
- [x] `_constant_velocity_features()` helper.
- [x] `_human_intent_graph()` helper ve all-masked guard.
- [x] Spatial-LTC sonrasında, mevcut `_attention_pool` öncesinde residual branch.
- [x] Orthogonal/uygun initialization.
- [x] Auto-detect buffer parsing.
- [x] Task 1 testlerini geçir.

### Task 3 — Güvenli checkpoint upgrade yolu

**Modify:** `sncp_ppo/train.py`; **Test:** `tests/test_v37_intention_graph.py`

- [x] `--hh_intent_graph`, `--hh_attn_heads`, `--cv_horizons`, `--cv_dt`, `--upgrade_checkpoint` parser args.
- [x] `--init_checkpoint` exact davranışı değişmeden kalır.
- [x] Upgrade missing-key allowlist ve fail-fast.
- [x] v34 Beta checkpoint -> v37 upgrade -> gate0 exact equivalence.
- [x] v36 combined checkpoint -> v37 upgrade round-trip de test edilir.

### Task 4 — Diagnostics

**Modify:** `sncp_ppo/train.py`, gerekirse `sncp_ppo/training_diagnostics.py`; tests.

- [x] CSV'ye `hh_gate` kolonu ekle (`0`/boş default politika için backward-compatible).
- [x] Console'a eval cadence'de gate yaz.
- [x] Diagnostics raporu gate trajectory min/final/max değerlerini özetler.
- [x] Existing CSV/report testlerini güncelle.

### Task 5 — Probe runner ve ortak vaka bankası

**Create:** `scripts/run_v37_probes.py`, `scratch/_analyze_v37_probe.py`; tests.

- [x] C0/C1 ve seeds 40/41/42 otomatik çalışır veya Colab komutlarını üretir.
- [x] Episode-level ortak case kimliği kaydedilir.
- [x] Paired success/collision tablosu + Wilson CI.
- [x] Section 5 GO kuralı makine-okunur verdict üretir.
- [x] Probe artifact'leri `eval_v37_probe/` altında toplanır.

### Task 6 — Probe gate kararı

- [ ] Üç seed tamamlanır.
- [ ] Verdict ve tablolar Run Record'a yazılır.
- [ ] FAIL: notebook v37'ye çevrilmez; plan negatif sonuçla kapanır.
- [ ] PASS: Task 7'ye geçilir.

### Task 7 — Notebook/readiness v37 probe

**Modify:** `sncp_ppo_colab.ipynb`, `sncp_ppo/run_readiness.py`, `tests/test_v16_run_readiness.py`, `tests/test_post_run_pipeline.py`.

- [x] Notebook full v36 koşusundan V37 C0/C1 paired-probe launcher'ına taşındı.
- [x] Readiness/test marker'ları `sncp_ppo_v34.pt` tabanlı `eval_v37_probe/` akışını doğrular.
- [x] Persist/download `eval_v37_probe_artifacts.zip`.
- [ ] Probe `GO` verirse ayrı bir full-v37 notebook/readiness geçişi yapılır.

### Task 8 — Zorunlu doğrulama

- [x] Auto-detect/upgrade round-trip missing/unexpected boş.
- [x] Gate0 equivalence.
- [x] HH branch no-op değil.
- [x] Beta vektörize yol gerçek Beta.
- [x] Kısa CLI smoke exit 0; entropy/KL/RMS finite, `hh_gate` finite ve hareket ediyor.
- [x] Tam suite:

```powershell
C:\ProgramData\miniconda3\python.exe -m pytest --basetemp=./.pytmp -q
```

- [x] Testler yeşil: 272 passed, 1 pre-existing warning.

### Task 9 — Colab ve nihai değerlendirme

- [ ] `main` pull.
- [ ] Base checkpoint hazırla/pull et.
- [ ] v37 1.5M fine-tune.
- [ ] Checkpoint + CSV + artifacts indir.
- [ ] 5x50 honest sweep.
- [ ] Bonferroni kararını ve trajectory audit'i raporla.
- [ ] AGENTS.md ve makale sonuçlarını dürüst verdict ile güncelle.

---

## 8. Social-NCE için sonraki karar (v38 adayı)

Social-NCE kanıtı güçlüdür fakat resmî sonuçlar BC ve off-policy Rainbow üzerindedir; mevcut PPO buffer'ında robot/human global future events doğrudan saklanmıyor. v37 core başarısız olmadan bunu aynı koşuya eklemek mekanizma atfını ve uygulama güvenliğini düşürür.

v37 core GO verdikten sonra ayrı bir 300k v38 probe düşünülebilir:

- Mevcut rollout sequence'den executed unicycle robot path'ini 1–4 adım entegre et.
- Negatif event'leri CV-predicted human merkezlerinin robot+human çapı çevresinden örnekle.
- `L = L_PPO + lambda * L_SocialNCE`, başlangıç `lambda=0.1`, temperature `0.2`.
- Gradient yalnız shared representation + projection/event heads'e gider; PPO ratio matematiği değişmez.
- Ayrı control ve üç seed olmadan full run yapılmaz.

---

## 9. Run Record

```text
v36 final budget: 4,001,792 steps (requested budget 4M)
v36 checkpoint: checkpoints/sncp_ppo_v36.pt
v36 N=5/10/15/20 success: 97.6 / 88.4 / 82.0 / 73.2%
v36 N=5/10/15/20 collision: 2.4 / 11.6 / 15.6 / 23.2%
v36 N=5/10/15/20 timeout: 0.0 / 0.0 / 2.4 / 3.6%
v36 vs v30 preregistered verdict: NEGATIVE/FLAT

v34 N=5/10/15/20 success: 96.8 / 92.8 / 91.2 / 86.0%
v34 N=5/10/15/20 collision: 2.8 / 7.2 / 8.8 / 13.2%
v34 N=5/10/15/20 timeout: 0.4 / 0.0 / 0.0 / 0.8%

selected v37 base: v34-fixed-beta
selection reason: v36 failed the preregistered gate and regressed vs v34 at high N
                  (N=15/20 success -9.2/-12.8 pp; collision +6.8/+10.0 pp).

probe C0 runs: TBD
probe C1 runs: TBD
probe verdict: TBD
learned hh_gate: TBD

full v37 checkpoint: checkpoints/sncp_ppo_v37.pt
full v37 verdict: TBD
```

---

## 10. Handoff özeti

Yeni bir agent bu dosyayı gördüğünde:

1. v36 sonucu `NEGATIVE/FLAT`; tekrar taban seçimi yapılmamalı.
2. Default ve kilitli model tabanı `v34-fixed-beta`; v36 warm-start olarak kullanılmamalı.
3. v37 core tek mekanizmadır: **gated HH self-attention + model-içi 1–4 adım constant-velocity intent geometry**.
4. Social-NCE v37'ye otomatik dahil değildir; v38 contingent probe'dur.
5. Notebook marker'ları V37 paired-probe launcher'ını doğrular; probe PASS/GO olmadan full A100 v37 run yapılmaz.
6. Tüm testler yeşil olmadan commit/push yapılmaz.
