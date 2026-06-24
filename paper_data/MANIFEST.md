# paper_data — IEEE makalesinin sayısal kanıt zinciri (provenance)

Bu klasör, `rapor/sncp_ppo_ieee.tex` içindeki tüm sayısal iddiaların kaynağıdır:
dürüst çok-tohumlu değerlendirme sonuçları (JSON) + bunları üreten/analiz eden
scriptler. Checkpoint'ler (`.pt`) büyük olduğundan sürüm kontrolünde **değildir**
(yerel kalır); aşağıdaki SHA-256 ile sabitlenmiştir.

## Değerlendirme protokolü (tüm sürümlerde aynı)
- Tohum kümesi: `[100, 200, 300, 400, 500]` (5 tohum)
- Yoğunluk başına: 5 × 50 = **250 bölüm**; yoğunluklar N = 5/10/15/20
- Senaryo: `paper_challenging` (dağınık yaya, 15×15 m arena, görünmez-robot, ORCA kalabalık)
- Robot hızı 1.0 m/s; yaya hızı 1.0 m/s; `max_time=None` (bütçeyi ortam çözer); `goal_noise=0`
- Belirlenimci (deterministic) politika: eylem = dağılım ortalaması
- İstatistik: Wilson %95 GA, havuzlanmış iki-oran `z`, Cohen `h`, Bonferroni
  α = 0.05/4 = 0.0125 (başarı ve çarpışma ayrı önceden-kayıtlı test aileleri)

Protokol uygulaması: `sweep_v34.py` (checkpoint → JSON). Diğer sürümler birebir
aynı protokolü kullanır; yalnızca `CKPT`/`OUT` adları farklıdır.

## Sonuç dosyaları
| Dosya | Sürüm | Açıklama |
|---|---|---|
| `v27_multiseed_result.json` | v27 | +ön-MLP gömmesi (Eq 11) — sadakat düzeltmesi |
| `v28_multiseed_result.json` | v28 | +yoğunluk müfredatı (N∼U(10,20)) |
| `v29_multiseed_result.json` | v29 | attn count-scaling (Eq 13, **v28 tabanlı**) |
| `v30_multiseed_result.json` | v30 | +mean+max havuzlama — **benimsenen yapılandırma** |
| `v31_multiseed_result.json` | v31 | ablasyon: node kapasite 256/96 |
| `v32_multiseed_result.json` | v32 | ablasyon: curriculum N→25 + 4M bütçe |
| `v33_multiseed_result.json` | v33 | ablasyon: 4-başlı çapraz dikkat |
| `v35_multiseed_result.json` | v35 | ablasyon: 6 m havuz-seviyesi algı maskesi |
| **`v34_multiseed_result.json`** | **v34** | **Beta eylem dağılımı (temiz, fb0bf07 sonrası)** |
| `v30_standard_result.json` | v30 | makalenin `standard` senaryosu (5 yaya, 12.5 s bütçe) |

> v26 (naif taban) ayrı JSON taşımaz; figürlerde hard-coded'dur.

## v34 (temiz Beta) checkpoint provenance
- Dosya (yerel): `sncp_ppo_v34.pt`
- **SHA-256:** `C4CF0FEA2FF2A92DA04B5D8D8DE274FC44190C78D7291FEE27BF288867D094BA`
- Kod commit'i: `fb0bf07` — vektörize PPO yolu artık gerçekten Beta üretir
  (`SNCPPolicy.make_action_dist`); öncesinde Beta, `Normal(α,β)` olarak eğitiliyordu.
- Reçete: v30 tabanı (pre-MLP + mean+max + yoğunluk müfredatı) +
  `--action_dist beta --ent_coef 0.001`, 2.5M adım, Colab A100.
- ⚠️ Bug-öncesi v34 (`Normal(α,β)` olarak eğitilen) **geçersizdir** ve bu paket dışındadır.

## Tekrar üretim
- İstatistik tablosu (checkpoint gerekmez): `cd paper_data && python analyze_v34.py`
- Şampiyon ayrıntılı metrik tablosu: `cd paper_data && python v30_detail_table.py`
- Sıfırdan sweep (yerel checkpoint gerekir): repo kökünden `python paper_data/sweep_v34.py`
  (CKPT yolu repo köküne göredir; SHA-256 yukarıdaki ile eşleşmelidir).
