# v36 — Birleşik Lever Koşusu (Tasarım)

**Tarih:** 2026-06-22
**Durum:** Onaylandı (kullanıcı), uygulamaya hazır.

## Amaç

Daha önce TEK TEK denenip v30 şampiyonunu geçemeyen altı kaldıracı **aynı anda** tek bir
eğitim koşusunda birleştir (v36) ve dürüst 5-seed protokolüyle v30'a karşı değerlendir.

## Motivasyon ve hipotez

Tek-değişken dizisi (v29/v31/v32/v33/v34/v35) tükendi; hepsi v30'un altında veya düz kaldı.
Açık soru: *bu leverler özünde mi kötü, yoksa tek tek **yetersiz-kaynak** (kapasite/bütçe) veya
**yanlış-ayar** (Gaussian'a göre entropy) yüzünden mi battı?* v36, capacity (v31) + bütçe/erişim
(v32) + düzgün-ayarlı entropy ile birlikte verildiğinde kombinasyonun v30'u geçip geçmediğini
sınar. Sonuç ne olursa olsun makale için tamamlayıcı ("hepsini birlikte de denedik").

## Dürüst çerçeve (over-claim önleme)

- **ÇOK-DEĞİŞKENLİ, bilinçli.** Bu, projenin tek-değişken disiplininden maksimum sapmadır;
  sonuç pozitif/negatif olsun, **hangi leverin etkisi olduğu ayrıştırılamaz** (sıfır atfedilebilirlik).
- Bileşenlerin çoğu tek tek NEGATİFTİ (v34 felaket, v31/v33 gerileme, v29/v32 düz). Negatiflerin
  birleşimi genelde negatifi pekiştirir → **beklenti düşük-orta**; en olası sonuç yine v30'un altı.
- `--ent_coef 0.001` SÜPÜRÜLMEMİŞ sezgisel bir seçimdir (tek değer); ideal değer için ayrı arama gerekir.
- Tek eğitim-seed (tüm projede olduğu gibi).

## Birleşim (v30 tabanı + hepsi)

| Lever | Bayrak | Kaynak |
|---|---|---|
| pre-MLP (Eq 11) | `--pre_mlp` | v27 (v30 tabanında) |
| mean+max havuzlama | `--meanmax_pool` | v30 (taban) |
| node kapasite | `--node_units 256 --node_output 96` | v31 |
| yoğunluk erişimi | `--num_humans_range 10 25` | v32 |
| bütçe | `--total_steps 4000000` | v32 |
| çok-baş dikkat | `--attn_heads 4` | v33 |
| sayı-ölçekleme | `--attn_count_scaling` | v29 |
| beta aksiyon | `--action_dist beta` | v34 |
| entropy ayarı | `--ent_coef 0.001` | YENİ ("düzgün beta") |
| algı menzili | `--sense_range 6.0` | v35 |

## İki yeni kod parçası

### 1. `models.py`: sayı-ölçeklemesini çok-baş dikkatin içine taşı
Şu an `attn_count_scaling` yalnız tek-baş dalında (`_attention_pool` satır ~269) uygulanıyor;
`attn_heads>1` çok-baş dalına gidip onu ATLIYOR. v29+v33 birlikte istendiğinden, count-scaling
`_multihead_attention` içinde de uygulanmalı: `scores = scores * H` (softmax öncesi, maskeden önce;
H = yaya sayısı = tek-baştaki `num_humans` ile tutarlı). `attn_count_scaling` kapalıyken çok-baş
davranışı BYTE-AYNI kalır; tek-baş yolu hiç değişmez.

### 2. `train.py` + `ppo.py`: `--ent_coef` CLI flag'i
`PPOAgent.c2` şu an sabit 0.01 (CLI yok). Yeni `--ent_coef` argümanı (default 0.01 = geriye-uyumlu)
`build_or_load_policy`/agent kurulumunda `c2`'ye geçer. v36 beta için 0.001 kullanır. Gaussian
koşuları default 0.01 ile byte-aynı kalır.

## Auto-detect ve eval

Tüm varyantlar checkpoint'ten OTOMATİK okunur (`build_policy_for_checkpoint`): node dims (gleak/output_w),
`attn_heads` buffer, `_attn_count_scaling` buffer, `action_dist` (actor_logstd yok → beta), `_sense_range`
buffer, pre_mlp/meanmax anahtarları. Dürüst 5-seed×50=250 ep sweep (paper_challenging, robot 1.0,
human 1.0, max_time None, goal_noise 0) v30'a karşı; sweep kodu DEĞİŞMEZ (yalnız CKPT/OUT). ent_coef
eğitim-zamanı; eval'i etkilemez. Beta deterministik aksiyon = ölçekli beta ortalaması (mevcut).

## Karar kuralı (önceden-kayıtlı)

v30 baseline: başarı 97.2/89.6/85.6/79.2; çarpışma 2.8/10.4/14.4/20.8; timeout 0 (N=5/10/15/20).
v36 "yardımcı" sayılır ANCAK: yüksek-N (15/20) başarı↑ ve/veya çarpışma↓ (Bonferroni α=0.0125 anlamlı)
+ N=5/10'da gerileme yok + timeout 0. Null = dürüst negatif. Ayrıca v32 (en yüksek high-N nokta-tahminleri)
ile de kıyas raporlanır.

## Bütçe

4M step, N~U(10,25), 16 env × 128 ufuk, ~5-6h A100. Kullanıcı koşu-içi holdout'u izleyebilir
(v32'de ~2.6M'de plato gözlemlenmişti; beta+kapasite daha çok isteyebilir).

## Kapsam dışı (YAGNI)

- ent_coef SÜPÜRMESİ (tek değer denenir).
- Ablation/atıf (çok-değişkenli, bilinçli).
- Mimari yeniden tasarım (mevcut bayraklar + 2 küçük yama).
