# SNCP-PPO Baseline Comparison

Overall verdict: fail

Baseline: `eval_v15\density_sweep.json`
Candidate: `eval_v16\density_sweep.json`
Beeline baseline: 121.5 successful steps

| N | v15 Success | Candidate Success | Success Delta | v15 Collision | Candidate Collision | v15 Timeout | Candidate Timeout | Timeout Delta | Nav Margin | I_sp Delta | Status | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 44.0% | 36.0% | -8.0 pp | 30.0% | 4.0% | 26.0% | 60.0% | +34.0 pp | 45.2 | -0.0001 | fail | FAIL: success dropped by -8.0 pp; FAIL: timeout/freezing rose by 34.0 pp |
| 3 | 50.0% | 62.0% | +12.0 pp | 30.0% | 8.0% | 20.0% | 30.0% | +10.0 pp | 46.1 | -0.0109 | pass | PASS: preserved real-avoidance gates |
| 5 | 66.0% | 56.0% | -10.0 pp | 18.0% | 18.0% | 16.0% | 26.0% | +10.0 pp | 50.2 | -0.0092 | fail | FAIL: success dropped by -10.0 pp |
| 8 | 50.0% | 40.0% | -10.0 pp | 34.0% | 34.0% | 16.0% | 26.0% | +10.0 pp | 52.9 | -0.0075 | fail | FAIL: success dropped by -10.0 pp |
| 10 | 46.0% | 44.0% | -2.0 pp | 46.0% | 44.0% | 8.0% | 12.0% | +4.0 pp | 60.4 | -0.0036 | warn | WARN: high-density success did not improve |
