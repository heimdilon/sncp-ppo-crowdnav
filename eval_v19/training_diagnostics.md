# SNCP-PPO Training Diagnostics

CSV: `logs/training_20260609_073534.csv`
Rows: 1221
Evaluated holdout rows: 1202
Observed replay ratio: 17.8%

## Best Holdout

Best step: 1679360
Best min success: 68.0%
Best reason: best updated (priority: min_success, tie-break: avg_reward, then lower collision_rate)

## Final Holdout

Final step: 2500608
Final min success: 60.0%
Collapse delta: -8.0%
Collapse detected: no

## PPO Stability

Final std linear: 0.142
Final std angular: 0.239
Max std linear: 0.142
Max std angular: 0.239
Std linear delta: 0.006
Std angular delta: 0.016

## Per-Scenario Success

| Scenario | Best | Final | Delta |
|---|---:|---:|---:|
| easy | 100.0% | 100.0% | +0.0% |
| hard | 80.0% | 88.0% | +8.0% |
| circle | 68.0% | 60.0% | -8.0% |

## Per-Scenario Failure Profile

| Scenario | Final success | Final collision | Final timeout | Final avg steps | Final avg I_sp | Final min d_min |
|---|---:|---:|---:|---:|---:|---:|
| easy | 100.0% | 0.0% | 0.0% | 156.2 | 0.000 | 0.762 |
| hard | 88.0% | 2.0% | 10.0% | 159.4 | 0.002 | 0.542 |
| circle | 60.0% | 34.0% | 6.0% | 124.6 | 0.010 | 0.489 |
