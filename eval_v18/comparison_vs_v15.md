# SNCP-PPO Baseline Comparison

Overall verdict: pass

Baseline: `eval_v15/density_sweep.json`
Candidate: `eval_v18/density_sweep.json`
Beeline baseline: 121.5 successful steps

| N | v15 Success | Candidate Success | Success Delta | v15 Collision | Candidate Collision | v15 Timeout | Candidate Timeout | Timeout Delta | Nav Margin | I_sp Delta | Status | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 44.0% | 66.0% | +22.0 pp | 30.0% | 14.0% | 26.0% | 20.0% | -6.0 pp | 31.1 | +0.0063 | pass | PASS: preserved real-avoidance gates |
| 3 | 50.0% | 86.0% | +36.0 pp | 30.0% | 4.0% | 20.0% | 10.0% | -10.0 pp | 36.2 | -0.0147 | pass | PASS: preserved real-avoidance gates |
| 5 | 66.0% | 86.0% | +20.0 pp | 18.0% | 10.0% | 16.0% | 4.0% | -12.0 pp | 41.5 | -0.0077 | pass | PASS: preserved real-avoidance gates |
| 8 | 50.0% | 70.0% | +20.0 pp | 34.0% | 26.0% | 16.0% | 4.0% | -12.0 pp | 43.9 | -0.0101 | pass | PASS: preserved real-avoidance gates |
| 10 | 46.0% | 64.0% | +18.0 pp | 46.0% | 34.0% | 8.0% | 2.0% | -6.0 pp | 48.1 | -0.0076 | pass | PASS: preserved real-avoidance gates |
