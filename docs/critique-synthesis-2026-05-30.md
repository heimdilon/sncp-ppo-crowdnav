# SNCP-PPO — Çok-Agent Eleştiri Sentezi ve Yol Haritası

**Tarih:** 2026-05-30
**Yöntem:** 3 bağımsız uzman agent (AI Engineer / Model QA / Code Reviewer) repoyu ayrı ayrı okuyup eleştirdi. Bulgular burada sentezlendi ve **her somut iddia koddan/runtime'dan doğrulandı** (bazıları çürütüldü).

---

## TL;DR

Mevcut hard-senaryo tavanı (%30-38) **hiperparametre sorunu değil, yapısal**. İki kök neden, her iki üst-seviye agent tarafından bağımsızca işaret edildi:

1. **Veri açlığı** — `update_freq=5` × tek seri ortam ≈ **~500 transition/PPO-update** (standart PPO 2048+). Tüm eğitim ~150k env-adımı; recurrent crowd-nav için çok az.
2. **Goal-greedy attractor** — ödül yapısı çarpışmayı "ekonomik olarak kabul edilebilir" kılıyor; near-miss için sıfır gradyan.

**Karar:** Önce **vectorized env** (kök neden #1). Diğer tüm iyileştirmeler ancak veri pipeline'ı düzeldikten sonra ölçülebilir hale gelir.

---

## Doğrulama tablosu (agent iddiası → benim teyidim)

| İddia | Kaynak | Durum | Kanıt |
|---|---|---|---|
| Veri açlığı (~500 transition/update) | AI Eng | ✅ **Doğru** | `update_freq=5`, `batch_size=16`, `seq_len=16` (train.py:498-502); episode ~100-240 adım |
| n=1 training seed → karşılaştırmalar geçersiz | QA | ✅ **Doğru** | `--seed 42` default; v6/v7/v8 ±9pt CI içinde, ayırt edilemez |
| Best-ckpt winner's curse (~10-15pt şişme) | QA | ✅ **Doğru** | Kodun kendi yorumu: "v7 '%50' gerçekte %38" (train.py); 200 gürültülü argmax |
| Goal-greedy / near-miss cezası yok | AI Eng | ✅ **Doğru** | step() reward; `collision = d_min<0.6` binary, 0.61m'de 0 gradyan (crowd_env.py:348) |
| Bug 5: value-loss ölçek uyumsuzluğu (old_values stale ret_std) | Code Rev | ⚠️ **Makul, doğrulanmalı** | ppo.py:388-392 vs 524-528; `ret_std` monoton büyüyor, old_values eski ölçekte |
| Bug 2: test_eval clip'siz aksiyon | QA + Code Rev | ⚠️ **Deterministik modda zararsız** | test_eval.py:31-32 `deterministic=True`→`action=mu`, sigmoid/tanh zaten sınırlı. Stokastik modda bug olurdu ama kullanılmıyor |
| Bug 3: h_spat 2D/3D şekil tutarsızlığı → crash | Code Rev | ❌ **YANLIŞ ALARM** | Runtime testi: forward batch=1'de tutarlı 2D `[5,32]` döndürüyor; `torch.stack` OK `[3,5,32]`. Onlarca koşu crash etmedi |
| LTC cargo-cult (sabit dt'de GRU'ya indirgenir) | AI Eng | ✅ **Teorik olarak doğru** | `time_step=0.25` sabit; LTC'nin adaptif-dt avantajı kullanılmıyor. temporal LTC `[v,w]` zaten robot_node'da var |
| Attention yönü ters (Q=yaya, K=robot) | AI Eng + Code Rev | ⚠️ **Geçerli ama suboptimal** | models.py:155-159; çalışır ama standart "robot-query" indüktif biası eksik |

---

## Kök nedenler (öncelik sırası)

### 1. Veri açlığı (EN YÜKSEK ETKI) — kök neden
- `update_freq=5` × tek seri ortam = ~500 transition/update. Standart PPO 2048+.
- Tek seri rollout (train.py:303 `while not done`) → ~5-7 s/episode, vectorize edilmemiş.
- 1500-10000 episode bile ~150k-1M env-adımı; recurrent 5-yaya görevi için literatürün altında.
- **Çözüm:** `gymnasium.vector.SyncVectorEnv` (8-16 ortam), update başına 2048+ transition topla.

### 2. Goal-greedy attractor — ödül tasarımı
- Dense approach ~0.325/adım × 100 adım ≈ +32; collision sadece -25 (terminal). Detour'dan kaçınmak ekonomik değil.
- Near-miss cezası yok: 0.61m (collision eşiği 0.60m hemen üstü) ile 5m aynı ödülü alıyor.
- Comfort term (`-0.5·I_sp/N`) çok zayıf, asla terminal olmuyor.
- Orientation cezası `d_min<0.6`'da sıfırlanıyor — yani yayaya en yakınken ceza kayboluyor (ters).
- **Çözüm:** sürekli proximity penalty (örn. d<1.5m'de lineer), approach coef 5→2-3, orientation gate düzelt.

### 3. Mimari (orta etki, sonraki kuşak)
- LTC sabit dt'de GRU'ya indirgenir + 3x yavaş forward → vectorize'ı zorlaştırıyor.
- 32-unit LTC bottleneck (640→32→256 node füzyonu).
- temporal LTC `[v,w]` gereksiz (zaten robot_node'da).
- Attention yönü: robot-query'e çevrilmeli.
- Gözlem eksikleri: yaya yarıçapı, yaya hedefi/niyeti (en kritik), yaya-yaya etkileşimi.
- **Çözüm:** GRU + robot-query attention; yaya-hedefi gözleme ekle.

### 4. Deneysel rigor (ölçüm geçerliliği)
- n=1 training seed → hiçbir versiyon karşılaştırması savunulamaz.
- 100-ep eval, tek eval-seed → ±9pt CI; winner's curse ek bias.
- **Çözüm:** ≥3-5 training seed, ≥300 ep eval × 3 eval-seed, mean±SD, two-proportion z-test.

---

## Yol Haritası (önceliklendirilmiş)

| # | Eylem | Etki | Efor | Kök neden |
|---|---|---|---|---|
| **1** | **Vectorized env** (8-16 ortam, 2048+ transition/update) | 🔴 Yüksek | 2-3 gün | #1 veri açlığı |
| 2 | Near-miss proximity penalty + approach coef düşür + orientation gate fix | 🔴 Yüksek | 0.5 gün | #2 ödül |
| 3 | Bug 5 fix (value-loss: old_values'u güncel ret_std ile normalize, ya da clipped value loss'u kaldır) | 🟡 Orta | 0.5 gün | critic underfitting |
| 4 | Çok-seed eval aracı (`eval_multiseed.py`, mean±CI) | 🟡 Orta | 1 gün | #4 rigor |
| 5 | Yaya-hedefi/niyeti gözleme ekle (humans_gx/gy → local birim vektör) | 🟡 Orta | 1 gün | #3 gözlem |
| 6 | LTC→GRU + robot-query attention (sonraki mimari kuşağı) | 🔴 Yüksek | 3-5 gün | #3 mimari |

**Sıra mantığı:** 1 olmadan 2-6'nın etkisi ölçülemez (gürültü maskeleyor). Önce #1, sonra #2+#3 (ucuz, ödül/critic), #4 ile ölç, sonra #5/#6.

---

## v6→v9 iterasyonu: bilimsel sağlamlık değerlendirmesi

- **v6→v7 (env randomize):** ✅ Gerçek ilerleme; ezber bug'ı doğru teşhis + düzeltildi.
- **v7→v8 (yaya hızı):** ⚠️ Makul hipotez ama tek-seed + 100-ep ile etki ölçülemedi (−4pt gürültü içinde).
- **v8→v9 (lr/episode):** ❌ Semptom tedavisi. Dalgalanmanın nedeni lr değil veri açlığı; lr düşürmek genliği azaltır ama yakınsamayı çözmez. 3-seed planı tek bilimsel doğru ekleme.
- **Genel:** Plateau gözle → görünür hiperparametre değiştir → tek-seed → gürültüden ayırt edememe → tekrar. Kök neden (veri throughput + mimari) ele alınmadı. **Bu sentez o döngüyü kırmak için.**

---

## Sonraki adım

Kullanıcı kararı: **#1 Vectorized env** ile başla. v9'u bununla birleştir (lr/KL ayarları kalabilir ama asıl kazanım paralel rollout'tan gelecek). Uygulama `superpowers:brainstorming` → `writing-plans` → TDD akışıyla yapılacak (yeni özellik + mevcut PPO/hidden-state pipeline'ına entegrasyon).
