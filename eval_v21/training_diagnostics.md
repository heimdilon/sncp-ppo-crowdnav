# SNCP-PPO Training Diagnostics

CSV: `logs/training_20260610_075806.csv`
Rows: 1221
Evaluated holdout rows: 1202
Observed replay ratio: 17.8%

## Best Holdout

Best step: 1515520
Best min success: 38.0%
Best reason: best updated (priority: min_success, tie-break: avg_reward, then lower collision_rate)

## Final Holdout

Final step: 2500608
Final min success: 10.0%
Collapse delta: -28.0%
Collapse detected: yes

## PPO Stability

Final std linear: 0.148
Final std angular: 0.230
Max std linear: 0.148
Max std angular: 0.230
Std linear delta: 0.012
Std angular delta: 0.007

## Per-Scenario Success

| Scenario | Best | Final | Delta |
|---|---:|---:|---:|
| easy | 84.0% | 76.0% | -8.0% |
| hard | 52.0% | 32.0% | -20.0% |
| circle | 38.0% | 10.0% | -28.0% |

## Per-Scenario Failure Profile

| Scenario | Final success | Final collision | Final timeout | Final avg steps | Final avg I_sp | Final min d_min |
|---|---:|---:|---:|---:|---:|---:|
| easy | 76.0% | 2.0% | 22.0% | 48.5 | 0.002 | 0.598 |
| hard | 32.0% | 6.0% | 62.0% | 56.0 | 0.005 | 0.547 |
| circle | 10.0% | 20.0% | 70.0% | 53.2 | 0.005 | 0.423 |
