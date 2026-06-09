# SNCP-PPO Baseline Comparison

Overall verdict: warn

Baseline: `eval_v15/density_sweep.json`
Candidate: `eval_v19/density_sweep.json`
Beeline baseline: 121.5 successful steps

| N | v15 Success | Candidate Success | Success Delta | v15 Collision | Candidate Collision | v15 Timeout | Candidate Timeout | Timeout Delta | Nav Margin | I_sp Delta | Status | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 44.0% | 64.0% | +20.0 pp | 30.0% | 14.0% | 26.0% | 22.0% | -4.0 pp | 34.8 | +0.0146 | warn | WARN: I_sp rose by 0.0146 |
| 3 | 50.0% | 80.0% | +30.0 pp | 30.0% | 6.0% | 20.0% | 14.0% | -6.0 pp | 42.5 | -0.0143 | pass | PASS: preserved real-avoidance gates |
| 5 | 66.0% | 74.0% | +8.0 pp | 18.0% | 12.0% | 16.0% | 14.0% | -2.0 pp | 42.1 | -0.0088 | pass | PASS: preserved real-avoidance gates |
| 8 | 50.0% | 52.0% | +2.0 pp | 34.0% | 26.0% | 16.0% | 22.0% | +6.0 pp | 49.1 | -0.0114 | pass | PASS: preserved real-avoidance gates |
| 10 | 46.0% | 48.0% | +2.0 pp | 46.0% | 34.0% | 8.0% | 18.0% | +10.0 pp | 52.8 | -0.0086 | pass | PASS: preserved real-avoidance gates |
