# SNCP-PPO Baseline Comparison

Overall verdict: fail

Baseline: `eval_v15/density_sweep.json`
Candidate: `eval_v21/density_sweep.json`
Beeline baseline: 121.5 successful steps

| N | v15 Success | Candidate Success | Success Delta | v15 Collision | Candidate Collision | v15 Timeout | Candidate Timeout | Timeout Delta | Nav Margin | I_sp Delta | Status | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 44.0% | 74.0% | +30.0 pp | 30.0% | 6.0% | 26.0% | 20.0% | -6.0 pp | -75.8 | -0.0047 | fail | FAIL: beeline/nav-time regression (45.7 steps) |
| 3 | 50.0% | 64.0% | +14.0 pp | 30.0% | 18.0% | 20.0% | 18.0% | -2.0 pp | -71.4 | -0.0102 | fail | FAIL: beeline/nav-time regression (50.1 steps) |
| 5 | 66.0% | 48.0% | -18.0 pp | 18.0% | 32.0% | 16.0% | 20.0% | +4.0 pp | -70.5 | -0.0035 | fail | FAIL: beeline/nav-time regression (51.0 steps); FAIL: success dropped by -18.0 pp; FAIL: collision rose by 14.0 pp |
| 8 | 50.0% | 32.0% | -18.0 pp | 34.0% | 48.0% | 16.0% | 20.0% | +4.0 pp | -69.8 | -0.0046 | fail | FAIL: beeline/nav-time regression (51.8 steps); FAIL: success dropped by -18.0 pp; FAIL: collision rose by 14.0 pp |
| 10 | 46.0% | 22.0% | -24.0 pp | 46.0% | 58.0% | 8.0% | 20.0% | +12.0 pp | -70.0 | -0.0075 | fail | FAIL: beeline/nav-time regression (51.5 steps); FAIL: success dropped by -24.0 pp; FAIL: collision rose by 12.0 pp; FAIL: timeout/freezing rose by 12.0 pp; WARN: high-density success did not improve |
