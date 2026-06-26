# SNCP-PPO: Kalabalıkta İnsan-Merkezli Navigasyon — Reprodüksiyon, Ablasyonlar ve Eğitimsiz Güvenlik Kalkanı

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heimdilon/sncp-ppo-crowdnav/blob/main/sncp_ppo_colab.ipynb)

Ao vd. (2026, *Int. J. Social Robotics*) **SNCP-PPO** mimarisinin (Yapısal Nöral Devre
Politikası + PPO) bağımsız bir reprodüksiyonu. Politika, kalabalığı bir uzay-zamansal grafa
dönüştürür; üç **LTC** (Liquid Time-Constant) nöral-devre kodlayıcıyla işler, dikkat
havuzlamasıyla özetler ve PPO ile uçtan uca eğitir.

> **Makale (bu çalışma):** [`rapor/sncp_ppo_ieee.pdf`](rapor/sncp_ppo_ieee.pdf) — Türkçe,
> IEEEtran. Kaynak: [`rapor/sncp_ppo_ieee.tex`](rapor/sncp_ppo_ieee.tex) (+ `refs.bib`, figür
> üretici scriptler). Overleaf'e `rapor/` klasörünü yükleyip `pdfLaTeX` ile derleyin.

> **Yerel GPU olmadan eğitim/değerlendirme:** [`sncp_ppo_colab.ipynb`](sncp_ppo_colab.ipynb)
> (yukarıdaki Colab rozeti) — uçtan uca kurulum, eğitim, değerlendirme, yörünge görselleştirme.

## Ana sonuç: "açığı ne kapatır, ne kapatmaz"

Makale "challenging" senaryosunda (10–20 yaya) **~%94 başarı / ~%4 çarpışma** bildirir. Naif bir
reprodüksiyon bunun çok altında kalır. Üç **sadakat düzeltmesi** açığın yarıdan çoğunu kapatır;
buna karşın denenen altı model kaldıracı şampiyonu yüksek yoğunlukta **geçemez**. Tüm kıyaslar
dürüst bir çok-tohumlu protokolle yapılır (5 tohum × 50 bölüm = 250 bölüm/yoğunluk, Wilson %95 GA,
iki-oranlı *z*, Bonferroni α=0.0125).

### 🛡️ Final sistem: v38 eğitimsiz eylem kalkanı (en güçlü sonuç)

Kilitli **v34 Beta** politikasının deterministik eylemini, yürütmeden önce kısa-horizon (6 adım, 1.5 s)
sabit-hız öngörüsüyle kontrol eden **eğitimsiz** bir güvenlik filtresi (`sncp_ppo/action_shield.py`):
çarpışma öngörülürse küçük bir aday-eylem kümesinden en güvenlisi seçilir. PPO'ya, checkpoint'e veya
eğitime dokunmaz. Aynı dürüst 5-tohum protokolde (250 bölüm/yoğunluk):

| N (yaya) | v34 (ham politika) | **v38 (+ kalkan)** | v38 çarpışma | Δ başarı vs v34 |
|:--------:|:------------------:|:------------------:|:------------:|:---------------:|
| 5  | 96.8% | **99.6%** | 0.0% | +2.8 pp |
| 10 | 92.8% | **99.6%** | 0.0% | +6.8 pp ✓ |
| 15 | 91.2% | **99.6%** | 0.4% | +8.4 pp ✓ |
| 20 | 86.0% | **98.8%** | 0.4% | +12.8 pp ✓ |

Kalkan, ham v34'e göre N=10/15/20'de hem başarıda hem çarpışmada **Bonferroni-anlamlı** iyileşme verir
(N=20: +12.8 pp başarı / −12.8 pp çarpışma, *p* < 0.0001, Cohen *h* ≈ 0.55) ve çarpışmayı her yoğunlukta
pratikte **sıfıra** indirir — sıfır ek eğitim maliyetiyle. (✓ = Bonferroni-anlamlı.)

**Politika tarafı şampiyonu v30** — `paper_challenging`, robot 1.0 m/s, görünmez-robot ORCA kalabalık:

| N (yaya) | Başarı | Çarpışma | Zaman aşımı |
|:--------:|:------:|:--------:|:-----------:|
| 5  | 97.2% | 2.8%  | 0% |
| 10 | 89.6% | 10.4% | 0% |
| 15 | 85.6% | 14.4% | 0% |
| 20 | 79.2% | 20.8% | 0% |

*(Makale: standard %99.5, challenging %94. N=10'da ~4 pp'lik kalan açık gerçektir; ayrıntı +
negatif ablasyonlar makalede.)*

**Sadakat düzeltmeleri (açığı kapatır):** Denklem 11 ön-MLP gömmesi (v27) + yoğunluk müfredatı
N~U(10,20) (v28) + mean+max dikkat havuzlaması (v30). N=10: %61.6 → %89.6.

**Açığı kapatmayan kaldıraçlar:** node kapasite (v31), N→25 + 4M bütçe (v32), 4-başlı dikkat (v33),
Beta aksiyon dağılımı (v34), sayı-ölçekleme (v29), 6 m algı-menzili maskesi (v35). **Kilit bulgu:**
robotun algısını makalenin kendi 6 m bütçesine indirmek açığı *genişletir* → kalan açık "ne kadar
görüyor" değil "gördüğüyle ne yapıyor" (temsil kullanımı). *(v36 = hepsini birleştiren koşu: **NEGATİF**;
v37 niyet-grafiği probe'u: **NO-GO**. Kalan çarpışma açığı, yeni eğitim yerine yukarıdaki v38 eylem
kalkanıyla kapatıldı.)*

## Mimari

Politika (`sncp_ppo/models.py::SNCPPolicy`), makalenin anlamsal bölümlemesini izler:

- **Girdiler** — `robot_node` (7), `spatial_edges` (N×6, yaya başına), `temporal_edges` (2)
- **Ön-MLP (Eq 11)** — ham kenar girdisi NCP'ye girmeden önce 256 boyuta gömülür *(v27)*
- **Kodlayıcılar** — Robot MLP (→128) + iki LTC (temporal + uzaysal, seyrek AutoNCP kablolama)
- **Dikkat havuzu** — yayalar robot/zaman anahtarına göre ağırlıklanır; v30 buna kardinaliteye-
  dayanıklı bir **mean+max** dalı ekler (yoğunlukta washout'u azaltır)
- **Füzyon + başlar** — Node LTC füzyonu → Aktör (ω, v) + Kritik V(s)

Üç LTC gizli durumu zaman boyunca taşınır (BPTT). Mimari diyagramı:
[`rapor/figures/ieee_arch.png`](rapor/figures/ieee_arch.png).

## Değerlendirme rejimi (makale-sadık)

`paper_challenging`: dağınık yayalar, 15×15 m arena, 8 m geçiş, **robot 1.0 m/s**, ORCA yayalar
(hız paritesi 1.0, **görünmez-robot** = CrowdNav rejimi), çarpışma eşiği 0.3 m, dt 0.25 s,
challenging bütçe 50 s / standard 12.5 s (ortamdan türetilir). Ödül = makale Eq 17–20
(`r_g`/`r_c`/`r_s`, normalize sosyal-baskı indeksi).

## Proje düzeni

```
.
├── crowd_sim/crowd_env.py     Gymnasium ortamı (ORCA kalabalık, paper senaryoları)
├── sncp_ppo/
│   ├── models.py              SNCPPolicy (ön-MLP + LTC + dikkat + mean+max + aktör/kritik)
│   ├── ppo.py                 PPOAgent + GAE (gaussian & beta dalları)
│   ├── train.py               Eğitim döngüsü: yoğunluk müfredatı + holdout-best
│   ├── eval_report.py         Yoğunluk taraması / rapor / yörünge render (shield destekli)
│   ├── action_shield.py       v38 eğitimsiz runtime güvenlik filtresi (kısa-horizon çarpışma kontrolü)
│   └── run_readiness.py       Colab koşu-öncesi marker denetimi
├── tests/                     Pytest (250+ test); `--basetemp=./.pytmp` ŞART (ACL)
├── scripts/                   Eval/karşılaştırma/görselleştirme CLI'ları
├── rapor/                     Makale: sncp_ppo_ieee.tex + refs.bib + figures/ + PDF
├── sncp_ppo_colab.ipynb       Uçtan uca Colab notebook'u
└── requirements.txt
```

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Eğitim (paper rejimi)

```bash
python -m sncp_ppo.train \
    --num_envs 16 --horizon 128 --total_steps 2500000 \
    --fixed_scenario paper_challenging --num_humans 10 --num_humans_range 10 20 \
    --robot_vpref 1.0 --lr 1e-4 \
    --holdout_scenarios paper_standard paper_challenging --holdout_episodes 50 \
    --pre_mlp --meanmax_pool \
    --save_path checkpoints/sncp_ppo_v30.pt
```

Mimari bayrakları (hepsi varsayılan-kapalı, checkpoint'ten **otomatik algılanır**): `--pre_mlp`
(Eq 11), `--meanmax_pool` (v30), `--node_units/--node_output` (v31), `--attn_heads` (v33),
`--attn_count_scaling` (v29), `--action_dist beta` + `--ent_coef` (v34), `--sense_range` (v35).

## Değerlendirme (dürüst çok-tohumlu tarama)

```bash
python scripts/run_post_eval.py --version 30 \
    --densities 5 10 15 20 --scenario paper_challenging \
    --n_episodes 50 --robot_vpref 1.0 --human_vpref_override 1.0
```

**v38 eylem kalkanı (eğitimsiz, eval-zamanı):**

```bash
python scripts/run_v38_shield_probe.py --checkpoint sncp_ppo_v34.pt \
    --densities 15 20 --n_episodes 50 --robot_vpref 1.0 --human_vpref_override 1.0
```

Ham (C0) vs kalkanlı (C1) eşleştirilmiş kıyas. Dürüst 5-tohum havuz için `scratch/_sweep_v38.py`
(`evaluate_density(..., action_shield=True)`), istatistik için `scratch/_analyze_v38.py`.

## Sınırlamalar (dürüst)

Tüm sürümler **tek eğitim-tohumuyla** eğitildi (değerlendirme çok-tohumlu); makale 500 test
kullanır, biz 250. Makalede belirtilmeyen ~12 değişken (NCP kablolama, ORCA parametreleri, eğitim
bütçesi vb.) bizim mühendislik seçimimizdir → reprodüksiyon açığının bir kısmı kaçınılmazdır.
Ayrıntı ve istatistik: [`rapor/sncp_ppo_ieee.pdf`](rapor/sncp_ppo_ieee.pdf).

## Referans

Ao, T., Li, H., Tian, Y. vd. (2026). *Human-Centric Motion Planning in Crowded Spaces: A
Structured Neural Circuit Approach with Social Interaction-Awareness.* International Journal of
Social Robotics, 18(52). DOI: 10.1007/s12369-026-01389-9
