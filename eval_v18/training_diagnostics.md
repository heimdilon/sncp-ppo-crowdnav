# SNCP-PPO Training Diagnostics

CSV: `logs/training_20260608_170436.csv`
Rows: 1221
Evaluated holdout rows: 1202
Observed replay ratio: 17.8%

## Best Holdout

Best step: 2334720
Best min success: 70.0%
Best reason: best updated (priority: min_success, tie-break: avg_reward, then lower collision_rate)

## Final Holdout

Final step: 2500608
Final min success: 58.0%
Collapse delta: -12.0%
Collapse detected: no

## PPO Stability

Final std linear: 0.137
Final std angular: 0.233
Max std linear: 0.139
Max std angular: 0.233
Std linear delta: 0.002
Std angular delta: 0.010

## Per-Scenario Success

| Scenario | Best | Final | Delta |
|---|---:|---:|---:|
| easy | 100.0% | 94.0% | -6.0% |
| hard | 82.0% | 76.0% | -6.0% |
| circle | 70.0% | 58.0% | -12.0% |

## Per-Scenario Failure Profile

| Scenario | Final success | Final collision | Final timeout | Final avg steps | Final avg I_sp | Final min d_min |
|---|---:|---:|---:|---:|---:|---:|
| easy | 94.0% | 0.0% | 6.0% | 162.5 | 0.000 | 0.651 |
| hard | 76.0% | 6.0% | 18.0% | 160.7 | 0.001 | 0.539 |
| circle | 58.0% | 32.0% | 10.0% | 128.1 | 0.011 | 0.513 |
