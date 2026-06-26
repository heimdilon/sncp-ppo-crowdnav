# v38 Training-Free Action Shield — Plan ve Runbook

> **Durum (2026-06-25): KOD HAZIR; eğitim yok.** v37 paired probe `NO-GO` verdiği için yeni
> PPO branch/full train başlatılmayacak. v38, kilitli `v34-fixed-beta` checkpoint'inin aksiyonunu
> çalıştırmadan önce kısa vadeli çarpışma riskiyle filtreleyen runtime safety layer'dır.

## 1. Gerekçe

V31/v33/v36/v37 sonuçları aynı yönde: daha fazla model kapasitesi veya attention eklemek high-N
çarpışmayı güvenilir biçimde azaltmadı. V37'de HH gate aktifleşmedi ve C1, C0'a göre N=15'te açıkça
geriledi.

Bu nedenle v38 öğrenilecek yeni bir temsil eklemez. Mevcut en iyi politika olan v34'ü korur ve sadece
şu soruyu test eder:

> Politika iyi bir genel rota seçerken bazı high-N anlarda çarpışmaya giden son aksiyonu seçiyorsa,
> bunu kısa horizon aksiyon filtresiyle azaltabilir miyiz?

## 2. Mekanizma

Modül: `sncp_ppo/action_shield.py`

1. Policy deterministic aksiyon üretir: `[v, w]`.
2. Shield mevcut env state'inden robot ve pedestrian pozisyon/hızlarını okur.
3. `horizon_steps=6` boyunca constant-velocity rollout yapar (`6 × 0.25s = 1.5s`).
4. Eğer orijinal aksiyon `collision_threshold + safety_margin` kadar güvenliyse aksiyonu aynen bırakır.
   Default `safety_margin=0.0`: yalnız predicted collision varsa müdahale eder. İlk lokal smoke'ta
   `0.10m` buffer gereksiz müdahaleyle collision yaratabildiği için default konservatif tutuldu.
5. Riskliyse küçük bir action lattice skorlanır:
   - lineer hız: `0, .25, .50, .75, 1.0 × vpref` + orijinalin yavaşlatılmış halleri
   - açısal hız: `-wmax, -.5wmax, 0, .5wmax, wmax` + orijinal `w`
6. Skor: collision/clearance shortfall + orijinal aksiyondan sapma + hız kaybı - goal progress.
7. En düşük skorlu aksiyon env'e verilir.

Bu PPO matematiğine, checkpoint'e veya training log'a dokunmaz.

## 3. Hızlı probe

Colab veya lokal:

```bash
python scripts/run_v38_shield_probe.py \
  --checkpoint sncp_ppo_v34.pt \
  --output_dir eval_v38_shield_probe \
  --densities 15 20 \
  --n_episodes 50 \
  --seed 100 \
  --robot_vpref 1.0 \
  --human_vpref_override 1.0
```

Çıktılar:

- `eval_v38_shield_probe/summary.json`
- `eval_v38_shield_probe/report.md`

## 4. GO/NO-GO kuralı

Hızlı probe için `GO`:

- N=15/20 ortalama collision delta `<= -3 pp`
- N=15/20 ortalama success delta `>= -2 pp`
- N=15/20 ortalama timeout delta `<= +2 pp`
- Eğer N=5/10 da koşulmuşsa düşük-N regression yok

`NO-GO` ise shield kapatılır; v34 unchanged champion kalır.

## 5. Geniş eval

Sadece hızlı probe `GO` derse:

```bash
python scripts/run_v38_shield_probe.py \
  --checkpoint sncp_ppo_v34.pt \
  --output_dir eval_v38_shield_full \
  --densities 5 10 15 20 \
  --n_episodes 100 \
  --seed 100 \
  --robot_vpref 1.0 \
  --human_vpref_override 1.0
```

Geniş eval da `GO` derse `scripts/evaluate_policy_report.py --action_shield ...` ile rapor/görsel
artifacts üretilebilir.

## 6. Riskler

- Shield çok agresif olursa timeout/freezing artar.
- Constant-velocity pedestrian tahmini kısa horizon dışında güvenilir değildir; bu yüzden horizon kısa tutuldu.
- Shield training-free olduğu için policy'nin long-horizon planını iyileştirmez; sadece son-anda riskli
  aksiyonları düzeltmeye çalışır.

## 7. Run Record

```text
base checkpoint: sncp_ppo_v34.pt
training: none
default probe densities: 15 / 20
default episodes: 50 per arm per density
default horizon: 6 steps = 1.5s
default safety margin: 0.0m (intervene only on predicted collision)
quick probe artifact: eval_v38_shield_probe_artifacts.zip (runtime artifact, git-ignored)
quick C0 N=15/20 success: 94.0 / 90.0%
quick C1 N=15/20 success: 100.0 / 96.0%
quick C0 N=15/20 collision: 6.0 / 8.0%
quick C1 N=15/20 collision: 0.0 / 2.0%
quick C0 N=15/20 timeout: 0.0 / 2.0%
quick C1 N=15/20 timeout: 0.0 / 2.0%
quick high-N success delta: +6.0 pp
quick high-N collision delta: -6.0 pp
quick high-N timeout delta: 0.0 pp
quick verdict: GO
wide eval: pending (required before final V38 decision)
```
