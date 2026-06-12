# SNCP-PPO Baseline Comparison

Overall verdict: fail

Baseline: `eval_v21/density_sweep.json`
Candidate: `eval_v22/density_sweep.json`
Beeline baseline: 32.0 successful steps

| N | v15 Success | Candidate Success | Success Delta | v15 Collision | Candidate Collision | v15 Timeout | Candidate Timeout | Timeout Delta | Nav Margin | I_sp Delta | Status | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 74.0% | 84.0% | +10.0 pp | 6.0% | 6.0% | 20.0% | 10.0% | -10.0 pp | 18.4 | +0.0018 | pass | PASS: preserved real-avoidance gates |
| 3 | 64.0% | 74.0% | +10.0 pp | 18.0% | 10.0% | 18.0% | 16.0% | -2.0 pp | 19.1 | +0.0086 | pass | PASS: preserved real-avoidance gates |
| 5 | 48.0% | 66.0% | +18.0 pp | 32.0% | 20.0% | 20.0% | 14.0% | -6.0 pp | 20.0 | +0.0107 | warn | WARN: I_sp rose by 0.0107 |
| 8 | 32.0% | 38.0% | +6.0 pp | 48.0% | 48.0% | 20.0% | 16.0% | -4.0 pp | 20.3 | +0.0228 | fail | FAIL: I_sp rose by 0.0228 |
| 10 | 22.0% | 36.0% | +14.0 pp | 58.0% | 48.0% | 20.0% | 16.0% | -4.0 pp | 19.9 | +0.0194 | warn | WARN: I_sp rose by 0.0194 |
